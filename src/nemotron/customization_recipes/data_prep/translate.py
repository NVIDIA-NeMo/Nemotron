# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Translation backends: Google Cloud, AWS Translate, LLM-based.

Each backend inherits from TranslationBackend and implements async
translation with semaphore-based concurrency, exponential-backoff retry,
and streaming order-restoration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

log = logging.getLogger(__name__)

MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TranslationConfig:
    """Configuration for translation pipelines."""

    source_lang: str = "en"
    target_lang: str = "hi"
    output_file: str = "translations.jsonl"
    max_concurrent_requests: int = 32
    async_position_key: str = "_async_position"
    dry_run: bool = False
    skip_filled: bool = False
    backend: str = "google"
    """Backend name: google | aws | llm"""

    # Backend-specific sub-configs (dict form for flexibility)
    google: dict = field(default_factory=dict)
    aws: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "TranslationConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(TranslationConfig)
        merged = OmegaConf.merge(schema, cfg)
        return TranslationConfig(**OmegaConf.to_container(merged, resolve=True))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TranslationBackend(ABC):
    """Abstract base for async translation backends.

    Subclasses implement ``_translate_single_async``, ``_setup_client``,
    ``check_server``, and ``_backend_name``.
    """

    def __init__(self, cfg: TranslationConfig):
        self.cfg = cfg
        self._semaphore = asyncio.Semaphore(cfg.max_concurrent_requests)
        self._output_lock: Optional[asyncio.Lock] = None

    @property
    @abstractmethod
    def _backend_name(self) -> str: ...

    @abstractmethod
    def _setup_client(self) -> None: ...

    @abstractmethod
    def check_server(self) -> bool: ...

    @abstractmethod
    async def _translate_single_async(self, text: str) -> str: ...

    def close(self) -> None:
        """Override to release resources."""

    # -- template method ---------------------------------------------------

    def translate(self, data: List[dict]) -> None:
        """Translate *data* (list of dicts with ``src`` key) and write JSONL."""
        log.info(
            "%s translation: concurrent=%d, total=%d",
            self._backend_name,
            self.cfg.max_concurrent_requests,
            len(data),
        )
        if self.cfg.dry_run:
            log.info("Dry run -- skipping")
            return

        if not self.cfg.skip_filled:
            for suffix in ("", "-async"):
                p = Path(self.cfg.output_file + suffix)
                if p.exists():
                    p.unlink()

        if not self.check_server():
            raise RuntimeError(f"{self._backend_name} health-check failed")

        asyncio.run(self._async_loop(data))
        log.info("%s translation complete", self._backend_name)

    async def _async_loop(self, data: list) -> None:
        if self._output_lock is None:
            self._output_lock = asyncio.Lock()

        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None

        pbar = tqdm(total=len(data), desc=f"{self._backend_name}") if tqdm else None
        async_path = self.cfg.output_file + "-async"

        with open(async_path, "at", encoding="utf-8", buffering=1) as fout:
            tasks = [
                asyncio.create_task(self._translate_and_save(dp, fout, pbar))
                for dp in data
            ]
            if tasks:
                await asyncio.gather(*tasks)

        if pbar:
            pbar.close()
        self._restore_order()

    async def _translate_and_save(self, dp: dict, fout, pbar) -> None:
        text = dp.get("src", "")
        async with self._semaphore:
            t0 = time.time()
            translation = await self._translate_single_async(text)
            t1 = time.time()

        result = {
            **dp,
            "generation": translation,
            "generation_time": t1 - t0,
        }
        async with self._output_lock:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            if pbar:
                pbar.update(1)

    def _restore_order(self) -> None:
        async_file = Path(self.cfg.output_file + "-async")
        if not async_file.exists():
            return

        with open(async_file, "rt", encoding="utf-8") as fin:
            generations = [json.loads(line) for line in fin if line.strip()]

        ordered = [None] * len(generations)
        unpositioned = []
        for g in generations:
            pos = g.pop(self.cfg.async_position_key, None)
            if pos is not None and isinstance(pos, int) and 0 <= pos < len(ordered):
                ordered[pos] = g
            else:
                unpositioned.append(g)

        with open(self.cfg.output_file, "wt", encoding="utf-8") as fout:
            for g in ordered:
                if g is not None:
                    fout.write(json.dumps(g, ensure_ascii=False) + "\n")
            for g in unpositioned:
                fout.write(json.dumps(g, ensure_ascii=False) + "\n")

        async_file.unlink()
        if unpositioned:
            log.warning(
                "%d translations had no valid position key and were appended at the end",
                len(unpositioned),
            )
        log.info("Restored order for %d translations", len(generations))


# ---------------------------------------------------------------------------
# Google Cloud backend
# ---------------------------------------------------------------------------


class GoogleBackend(TranslationBackend):
    """Google Cloud Translation API (v2 or v3)."""

    def __init__(self, cfg: TranslationConfig):
        super().__init__(cfg)
        raw = cfg.google or {}
        self._google_config = dict(raw) if not isinstance(raw, dict) else raw
        self._project_id = self._google_config.get("project_id") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        )
        self._location = self._google_config.get("location", "global")
        self._api_version = self._google_config.get("api_version", "v2")
        self._client = None
        self._parent = None
        self._setup_client()

    @property
    def _backend_name(self) -> str:
        return "Google"

    def _setup_client(self) -> None:
        try:
            if self._api_version == "v3":
                from google.cloud import translate_v3 as translate

                self._client = translate.TranslationServiceClient()
                if not self._project_id:
                    raise ValueError(
                        "project_id required for v3. Set google.project_id or GOOGLE_CLOUD_PROJECT."
                    )
                self._parent = f"projects/{self._project_id}/locations/{self._location}"
            else:
                from google.cloud import translate_v2 as translate

                self._client = translate.Client()
        except ImportError as exc:
            raise ImportError(
                "google-cloud-translate is required. Install with: pip install google-cloud-translate"
            ) from exc

    def check_server(self) -> bool:
        try:
            if self._api_version == "v3":
                self._client.translate_text(
                    parent=self._parent,
                    contents=["Hello"],
                    source_language_code="en",
                    target_language_code="hi",
                    mime_type="text/plain",
                )
            else:
                self._client.translate("Hello", target_language="hi", format_="text")
            return True
        except Exception as exc:
            log.error("Google health-check failed: %s", exc)
            return False

    async def _translate_single_async(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        loop = asyncio.get_running_loop()
        for attempt in range(MAX_RETRIES):
            try:
                return await loop.run_in_executor(None, self._sync, text)
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                    log.warning("Google retry %d: %s", attempt + 1, exc)
                else:
                    raise

    def _sync(self, text: str) -> str:
        if self._api_version == "v3":
            resp = self._client.translate_text(
                parent=self._parent,
                contents=[text],
                source_language_code=self.cfg.source_lang,
                target_language_code=self.cfg.target_lang,
                mime_type="text/plain",
            )
            return resp.translations[0].translated_text
        result = self._client.translate(
            text,
            source_language=self.cfg.source_lang,
            target_language=self.cfg.target_lang,
            format_="text",
        )
        return result["translatedText"]


# ---------------------------------------------------------------------------
# AWS Translate backend
# ---------------------------------------------------------------------------

AWS_MAX_BYTES = 10_000


class AWSBackend(TranslationBackend):
    """Amazon Translate backend."""

    def __init__(self, cfg: TranslationConfig):
        super().__init__(cfg)
        raw = cfg.aws or {}
        self._aws_config = dict(raw) if not isinstance(raw, dict) else raw
        self._region = (
            self._aws_config.get("region")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-2"
        )
        self._client = None
        self._setup_client()

    @property
    def _backend_name(self) -> str:
        return "AWS"

    def _setup_client(self) -> None:
        try:
            import boto3

            self._client = boto3.client("translate", region_name=self._region)
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for AWS backend. Install with: pip install boto3"
            ) from exc

    def check_server(self) -> bool:
        try:
            self._client.translate_text(
                Text="Hello",
                SourceLanguageCode="en",
                TargetLanguageCode="hi",
            )
            return True
        except Exception as exc:
            log.error("AWS health-check failed: %s", exc)
            return False

    async def _translate_single_async(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        loop = asyncio.get_running_loop()
        for attempt in range(MAX_RETRIES):
            try:
                return await loop.run_in_executor(None, self._sync, text)
            except ValueError:
                raise
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                    log.warning("AWS retry %d: %s", attempt + 1, exc)
                else:
                    raise

    def _sync(self, text: str) -> str:
        nbytes = len(text.encode("utf-8"))
        if nbytes > AWS_MAX_BYTES:
            raise ValueError(
                f"AWS input too large: {nbytes}B (limit {AWS_MAX_BYTES}B). Chunk upstream."
            )
        resp = self._client.translate_text(
            Text=text,
            SourceLanguageCode=self.cfg.source_lang,
            TargetLanguageCode=self.cfg.target_lang,
        )
        return resp.get("TranslatedText", "")


# ---------------------------------------------------------------------------
# LLM-based translation backend
# ---------------------------------------------------------------------------


class LLMBackend(TranslationBackend):
    """LLM-based translation using an OpenAI-compatible endpoint."""

    def __init__(self, cfg: TranslationConfig):
        super().__init__(cfg)
        raw = cfg.llm or {}
        self._llm_config = dict(raw) if not isinstance(raw, dict) else raw
        self._base_url = self._llm_config.get(
            "base_url", os.environ.get("LLM_BASE_URL", "")
        )
        self._model = self._llm_config.get("model", "")
        self._api_key = self._llm_config.get(
            "api_key", os.environ.get("LLM_API_KEY", "EMPTY")
        )
        self._client = None
        self._setup_client()

    @property
    def _backend_name(self) -> str:
        return "LLM"

    def _setup_client(self) -> None:
        try:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        except ImportError as exc:
            raise ImportError(
                "openai is required for LLM backend. Install with: pip install openai"
            ) from exc

    def check_server(self) -> bool:
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Translate: Hello"}],
                max_tokens=16,
            )
            return True
        except Exception as exc:
            log.error("LLM health-check failed: %s", exc)
            return False

    async def _translate_single_async(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        loop = asyncio.get_running_loop()
        for attempt in range(MAX_RETRIES):
            try:
                return await loop.run_in_executor(None, self._sync, text)
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                    log.warning("LLM retry %d: %s", attempt + 1, exc)
                else:
                    raise

    def _sync(self, text: str) -> str:
        prompt = (
            f"Translate the following text from {self.cfg.source_lang} "
            f"to {self.cfg.target_lang}. Output ONLY the translation.\n\n{text}"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Faithfulness evaluation (back-translation + BLEU)
# ---------------------------------------------------------------------------


def evaluate_faithfulness(
    source_texts: List[str],
    translated_texts: List[str],
    back_translated_texts: List[str],
) -> dict:
    """Compute BLEU between source and back-translated texts.

    Args:
        source_texts: Original source strings.
        translated_texts: Forward translations (unused in BLEU but kept for logging).
        back_translated_texts: Back-translated strings.

    Returns:
        Dict with ``bleu`` score and per-pair scores.
    """
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError(
            "sacrebleu is required for faithfulness evaluation. "
            "Install with: pip install sacrebleu"
        ) from exc

    bleu = sacrebleu.corpus_bleu(back_translated_texts, [source_texts])
    log.info("Faithfulness BLEU: %.2f", bleu.score)
    return {
        "bleu": bleu.score,
        "num_pairs": len(source_texts),
    }


# ---------------------------------------------------------------------------
# BYOB benchmark translation facade
# ---------------------------------------------------------------------------

_BACKEND_MAP = {
    "google": GoogleBackend,
    "aws": AWSBackend,
    "llm": LLMBackend,
}


def translate_byob_benchmark(cfg: "DictConfig") -> dict:
    """Translate a BYOB benchmark dataset to a target language.

    Reads the benchmark JSONL, translates question and option fields using
    the configured translation backend, and writes a translated JSONL.

    Args:
        cfg: OmegaConf DictConfig with translation parameters (including
            a ``translate`` sub-key with ``dataset_path``,
            ``source_language``, ``target_language``, and backend config).

    Returns:
        Dict with ``output_path`` and ``num_translated``.
    """
    from omegaconf import OmegaConf

    t_cfg = OmegaConf.select(cfg, "translate", default=cfg)
    if isinstance(t_cfg, DictConfig):
        t_cfg_dict = OmegaConf.to_container(t_cfg, resolve=True)
    else:
        t_cfg_dict = dict(t_cfg)

    dataset_path = t_cfg_dict.get("dataset_path")
    if not dataset_path:
        raise ValueError("translate.dataset_path must be set")

    source_lang = t_cfg_dict.get("source_language", "en")
    target_lang = t_cfg_dict.get("target_language", "hi")

    # Load the benchmark records
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        log.warning("No records found in %s", dataset_path)
        return {"output_path": None, "num_translated": 0}

    # Build translation data items from text fields
    data_items = []
    for i, rec in enumerate(records):
        src_parts = []
        if "question" in rec:
            src_parts.append(rec["question"])
        if "options" in rec and isinstance(rec["options"], dict):
            for key in sorted(rec["options"].keys()):
                src_parts.append(rec["options"][key])
        data_items.append({"src": " ||| ".join(src_parts), "_async_position": i, "_record": rec})

    # Build a TranslationConfig and use the appropriate backend
    model_cfg = t_cfg_dict.get("translation_model_config", {})
    backend_name = model_cfg.get("mode", "llm")

    output_file = str(Path(dataset_path).with_suffix(".translated.jsonl"))
    trans_cfg = TranslationConfig(
        source_lang=source_lang.split("-")[0],
        target_lang=target_lang.split("-")[0],
        output_file=output_file,
        backend=backend_name,
        llm=model_cfg.get("params", {}),
    )

    backend_cls = _BACKEND_MAP.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown translation backend '{backend_name}'. "
            f"Supported: {list(_BACKEND_MAP.keys())}"
        )

    backend = backend_cls(trans_cfg)
    try:
        backend.translate([{"src": item["src"], "_async_position": item["_async_position"]} for item in data_items])
    finally:
        backend.close()

    log.info("BYOB benchmark translation complete: %s", output_file)
    return {"output_path": output_file, "num_translated": len(data_items)}
