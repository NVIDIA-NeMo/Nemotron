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
Data acquisition: download, language-ID, domain classification, chat templates.

Wraps NeMo Curator's download/classify stages with OmegaConf-driven config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports with clear error messages
# ---------------------------------------------------------------------------

_CURATOR_MSG = (
    "nemo-curator is required for data acquisition stages. "
    "Install with: pip install nemo-curator"
)


def _require_curator():
    try:
        import nemo_curator  # noqa: F401
    except ImportError as exc:
        raise ImportError(_CURATOR_MSG) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AcquireConfig:
    """Configuration for the data acquisition pipeline."""

    download_dir: str = "data/raw"
    """Directory to store downloaded files."""

    output_dir: str = "data/acquired"
    """Directory for processed output."""

    record_format: str = "jsonl"
    """Format of source records (jsonl | parquet)."""

    url_limit: Optional[int] = None
    """Max URLs to fetch (None = all)."""

    record_limit: Optional[int] = None
    """Max records to iterate per source file."""

    chat_template_model: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    """HuggingFace model whose chat template to apply."""

    domain_classifier_model: str = "nvidia/multilingual-domain-classifier"
    """Model identifier for domain classification."""

    domain_classifier_batch_size: int = 256
    """Inference batch size for the domain classifier."""

    domain_text_field: str = "conversations"
    """Column name containing text for domain classification."""

    lid_model_path: Optional[str] = None
    """Path to FastText language-ID model (lid.176.bin)."""

    lid_text_field: str = "text"
    """Column name containing text for language identification."""

    sources: list[str] = field(default_factory=list)
    """List of URLs or HuggingFace dataset identifiers."""

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "AcquireConfig":
        """Create AcquireConfig from an OmegaConf DictConfig."""
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(AcquireConfig)
        merged = OmegaConf.merge(schema, cfg)
        return AcquireConfig(**OmegaConf.to_container(merged, resolve=True))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_dataset(cfg: AcquireConfig) -> Path:
    """Download data using NeMo Curator's DocumentDownloadExtractStage.

    Args:
        cfg: Acquisition configuration.

    Returns:
        Path to the directory containing downloaded/extracted records.
    """
    _require_curator()
    from nemo_curator.stages.text.download.base.stage import (
        DocumentDownloadExtractStage,
    )

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    log.info("Starting dataset download -> %s", cfg.download_dir)

    # NeMo Curator exposes generic URL/download/iterator/extractor hooks.
    # Recipe scripts can subclass or compose stages as needed; this helper
    # provides a one-call convenience wrapper.
    stage = DocumentDownloadExtractStage(
        download_dir=cfg.download_dir,
        url_limit=cfg.url_limit,
    )
    stage.run()
    log.info("Download complete -> %s", cfg.download_dir)
    return Path(cfg.download_dir)


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------


def classify_domains(
    cfg: AcquireConfig,
    dataset=None,
    *,
    pred_column: str = "domain",
    prob_column: str = "domain_prob",
    max_chars: int = 6000,
):
    """Run multilingual domain classification on a Dask/cuDF dataset.

    Args:
        cfg: Acquisition configuration (uses domain_classifier_model, etc.).
        dataset: NeMo Curator DocumentDataset.  If *None* the caller must
            supply one after construction.
        pred_column: Output column for predicted domain label.
        prob_column: Output column for prediction probability.
        max_chars: Truncate input text to this many characters.

    Returns:
        The dataset augmented with ``pred_column`` and ``prob_column``.
    """
    _require_curator()
    from nemo_curator.stages.text.classifiers.base import DistributedDataClassifier

    classifier = DistributedDataClassifier(
        model_identifier=cfg.domain_classifier_model,
        pred_column=pred_column,
        prob_column=prob_column,
        text_field=cfg.domain_text_field,
        max_chars=max_chars,
        model_inference_batch_size=cfg.domain_classifier_batch_size,
    )
    log.info(
        "Running domain classification (model=%s, field=%s)",
        cfg.domain_classifier_model,
        cfg.domain_text_field,
    )

    if dataset is not None:
        return classifier(dataset)
    return classifier


# ---------------------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------------------


_LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_LID_CACHE_DIR = Path.home() / ".cache" / "nemotron"
_LID_CACHE_PATH = _LID_CACHE_DIR / "lid.176.bin"


def _ensure_lid_model() -> str:
    """Auto-download FastText lid.176.bin if not already cached.

    Returns:
        Path to the lid.176.bin model file.
    """
    if _LID_CACHE_PATH.exists():
        log.info("Using cached FastText LID model: %s", _LID_CACHE_PATH)
        return str(_LID_CACHE_PATH)

    log.info(
        "Downloading FastText lid.176.bin to %s (one-time download, ~130 MB)...",
        _LID_CACHE_PATH,
    )
    _LID_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import urllib.request

    try:
        urllib.request.urlretrieve(_LID_URL, str(_LID_CACHE_PATH))
    except Exception as exc:
        # Clean up partial download
        if _LID_CACHE_PATH.exists():
            _LID_CACHE_PATH.unlink()
        raise RuntimeError(
            f"Failed to download lid.176.bin from {_LID_URL}. "
            "Download it manually and set lid_model_path in AcquireConfig. "
            "URL: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        ) from exc

    log.info("FastText LID model downloaded successfully.")
    return str(_LID_CACHE_PATH)


def identify_languages(
    cfg: AcquireConfig,
    dataset=None,
    *,
    pred_column: str = "language",
    prob_column: str = "language_prob",
):
    """Run FastText language identification on a dataset.

    If ``cfg.lid_model_path`` is *None*, the FastText ``lid.176.bin``
    model is automatically downloaded from Facebook's CDN to
    ``~/.cache/nemotron/lid.176.bin`` on first use.

    Args:
        cfg: Acquisition configuration.
        dataset: NeMo Curator DocumentDataset (optional).
        pred_column: Output column for predicted language code.
        prob_column: Output column for prediction probability.

    Returns:
        The dataset augmented with language columns, or the classifier
        stage if no dataset is provided.
    """
    _require_curator()
    try:
        from nemo_curator.stages.text.classifiers.fasttext import (
            FastTextLangId,
        )
    except ImportError as exc:
        raise ImportError(
            "FastText language-ID requires nemo-curator[fasttext]. "
            "Install with: pip install nemo-curator[fasttext]"
        ) from exc

    model_path = cfg.lid_model_path
    if model_path is None:
        model_path = _ensure_lid_model()

    lid = FastTextLangId(
        model_path=model_path,
        text_field=cfg.lid_text_field,
        pred_column=pred_column,
        prob_column=prob_column,
    )
    log.info("Running language identification (model=%s)", model_path)

    if dataset is not None:
        return lid(dataset)
    return lid


# ---------------------------------------------------------------------------
# Chat template application
# ---------------------------------------------------------------------------


def acquire_and_filter(cfg: "DictConfig") -> dict:
    """Orchestrate the full data acquisition pipeline.

    Runs download, domain classification, and language identification
    sequentially, returning a summary dict with output paths and stats.

    Args:
        cfg: OmegaConf DictConfig with acquisition parameters.

    Returns:
        Dict with ``output_dir``, ``num_records``, and stage results.
    """
    from omegaconf import OmegaConf

    acq_cfg = AcquireConfig.from_omegaconf(cfg)

    result: dict = {"output_dir": acq_cfg.output_dir}

    # Step 1: Download
    download_path = download_dataset(acq_cfg)
    result["download_dir"] = str(download_path)

    # Step 2: Domain classification (optional, if domain_classifier_model is set)
    if acq_cfg.domain_classifier_model:
        log.info("Running domain classification stage")
        classifier = classify_domains(acq_cfg)
        result["domain_classifier"] = "applied"

    # Step 3: Language identification (optional, if lid_model_path is set)
    if acq_cfg.lid_model_path:
        log.info("Running language identification stage")
        lid = identify_languages(acq_cfg)
        result["language_id"] = "applied"

    log.info("Acquire-and-filter pipeline complete: %s", result)
    return result


def apply_chat_template(
    messages: list[dict],
    tokenizer=None,
    model_name: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """Apply a HuggingFace chat template to a list of messages.

    Args:
        messages: OpenAI-format messages list.
        tokenizer: Pre-loaded HuggingFace tokenizer. If *None*, one is
            loaded from *model_name*.
        model_name: Model to load tokenizer from (ignored when *tokenizer*
            is provided).
        add_generation_prompt: Append a generation prompt to the end.

    Returns:
        Rendered chat string.
    """
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Provide either tokenizer or model_name.")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
