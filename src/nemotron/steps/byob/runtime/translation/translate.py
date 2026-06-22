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

"""Curator experimental translation for BYOB benchmark rows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nemotron.steps.byob.runtime.config import ByobTranslationConfig

_DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_PROVIDER_KEY_ENVS = {
    "nvidia": ("NGC_API_KEY", "NVIDIA_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}
_GENERATION_CONFIG_KEYS = {
    "extra_kwargs",
    "max_tokens",
    "n",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_k",
    "top_p",
}


@dataclass(frozen=True)
class _CuratorSymbols:
    async_openai_client: type
    document_batch: type
    generation_config: type
    translation_stage: type


def _load_curator_symbols() -> _CuratorSymbols:
    try:
        from nemo_curator.models.client.llm_client import GenerationConfig
        from nemo_curator.models.client.openai_client import AsyncOpenAIClient
        from nemo_curator.stages.text.experimental.translation import TranslationStage
        from nemo_curator.tasks import DocumentBatch
    except ImportError as exc:  # pragma: no cover - exercised only without Curator installed
        raise ImportError("BYOB translation requires Curator experimental translation support.") from exc

    return _CuratorSymbols(
        async_openai_client=AsyncOpenAIClient,
        document_batch=DocumentBatch,
        generation_config=GenerationConfig,
        translation_stage=TranslationStage,
    )


class CuratorTranslationModule:
    """Translate BYOB seed rows with Curator experimental translation stages."""

    def __init__(
        self,
        model_params: dict[str, Any],
        config: ByobTranslationConfig,
        text_field: str,
        translation_field: str,
        id_field: str,
        src_lang_field: str,
        tgt_lang_field: str,
    ):
        self.model_params = model_params
        self.config = config
        self.translation_config = config.translation_model_config
        self.text_field = text_field
        self.translation_field = translation_field
        self.id_field = id_field
        self.src_lang_field = src_lang_field
        self.tgt_lang_field = tgt_lang_field

    def translate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Translate text rows and return the original rows plus a translation column."""
        if dataframe.empty:
            out = dataframe.copy()
            out[self.translation_field] = []
            return out

        self._validate_required_columns(dataframe)
        source_lang, target_lang = self._get_language_pair(dataframe)

        symbols = _load_curator_symbols()
        stage = self._build_translation_stage(symbols, source_lang=source_lang, target_lang=target_lang)
        batch = symbols.document_batch(
            task_id=f"{self.config.expt_name}-translation",
            dataset_name=self.config.expt_name,
            data=dataframe.reset_index(drop=True).copy(),
        )

        for execution_stage in stage.decompose():
            execution_stage.setup()
            try:
                result = execution_stage.process(batch)
            finally:
                execution_stage.teardown()
            batch = _coerce_single_batch(result)

        out = batch.to_pandas()
        if self.translation_field not in out.columns:
            raise RuntimeError(
                f"Curator experimental translation did not emit the expected {self.translation_field!r} column."
            )
        return out

    def _build_translation_stage(
        self,
        symbols: _CuratorSymbols,
        *,
        source_lang: str,
        target_lang: str,
    ) -> Any:
        backend_type = _get_nested_value(self.translation_config, self.model_params, "backend_type", default="llm")
        stage_config = self.translation_config.get("stage", {})
        if not isinstance(stage_config, dict):
            raise ValueError("translation_model_config.stage must be a mapping when provided")
        if stage_config.get("enable_faith_eval", False):
            raise ValueError("BYOB translation uses backtranslation quality metrics; FAITH evaluation is not enabled")

        client = None
        model_name = str(
            _get_nested_value(self.translation_config, self.model_params, "model", "model_name", default="")
        )
        generation_config = None
        if backend_type == "llm":
            client = self._build_openai_client(symbols)
            generation_config = self._build_generation_config(symbols)

        backend_config = _get_nested_value(self.translation_config, self.model_params, "backend_config", default={})
        if backend_config is None:
            backend_config = {}
        if not isinstance(backend_config, dict):
            raise ValueError("translation_model_config.backend_config must be a mapping when provided")

        output_mode = str(stage_config.get("output_mode", "both"))
        if output_mode == "raw":
            raise ValueError(
                "BYOB translation requires Curator output_mode='both' or 'replaced' so translated text is emitted"
            )

        segment_stage_config = self.translation_config.get("segment_stage", {})
        if not isinstance(segment_stage_config, dict):
            raise ValueError("translation_model_config.segment_stage must be a mapping when provided")

        return symbols.translation_stage(
            source_lang=_normalize_language_code(source_lang),
            target_lang=_normalize_language_code(target_lang),
            text_field=self.text_field,
            output_field=self.translation_field,
            segmentation_mode=str(stage_config.get("segmentation_mode", "coarse")),
            min_segment_chars=int(stage_config.get("min_segment_chars", 0)),
            client=client,
            model_name=model_name,
            generation_config=generation_config,
            translation_prompt_path=stage_config.get("translation_prompt_path"),
            max_concurrent_requests=int(segment_stage_config.get("max_concurrent_requests", 64)),
            health_check=bool(segment_stage_config.get("health_check", True)),
            dry_run=bool(segment_stage_config.get("dry_run", False)),
            dry_run_log_count=int(segment_stage_config.get("dry_run_log_count", 5)),
            backend_type=str(backend_type),
            backend_config=backend_config,
            output_mode=output_mode,
            skip_translated=bool(stage_config.get("skip_translated", False)),
            translation_column=str(stage_config.get("translation_column", self.translation_field)),
        )

    def _build_openai_client(self, symbols: _CuratorSymbols) -> Any:
        provider = str(
            _get_nested_value(self.translation_config, self.model_params, "provider", default="nvidia")
        ).lower()
        client_config = self.translation_config.get("client", {})
        if not isinstance(client_config, dict):
            raise ValueError("translation_model_config.client must be a mapping when provided")

        inference_parameters = self._get_inference_parameters()
        max_concurrent_requests = _get_nested_value(
            client_config,
            self.model_params,
            inference_parameters,
            "max_concurrent_requests",
            "max_parallel_requests",
            default=5,
        )

        api_key = self._resolve_api_key(provider, client_config)
        client_kwargs = {
            "max_concurrent_requests": int(max_concurrent_requests),
            "max_retries": int(client_config.get("max_retries", self.model_params.get("max_retries", 3))),
            "base_delay": float(client_config.get("base_delay", self.model_params.get("base_delay", 1.0))),
            "timeout": client_config.get("timeout", self.model_params.get("timeout", 120)),
        }
        if api_key:
            client_kwargs["api_key"] = api_key

        base_url = _get_nested_value(client_config, self.model_params, "base_url", default=None)
        if base_url is None and provider == "nvidia":
            base_url = _DEFAULT_NVIDIA_BASE_URL
        if base_url:
            client_kwargs["base_url"] = str(base_url)

        return symbols.async_openai_client(**client_kwargs)

    def _build_generation_config(self, symbols: _CuratorSymbols) -> Any:
        inference_parameters = self._get_inference_parameters()
        raw_config = _get_nested_value(
            self.translation_config,
            self.model_params,
            "generation_config",
            "inference_parameters",
            default=inference_parameters,
        )
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, dict):
            raise ValueError("Curator generation configuration must be a mapping")

        generation_kwargs: dict[str, Any] = {}
        extra_kwargs = dict(raw_config.get("extra_kwargs") or {})
        for key, value in raw_config.items():
            if key in {"max_parallel_requests", "max_concurrent_requests"}:
                continue
            if key == "extra_kwargs":
                continue
            if key == "temperature":
                value = _normalize_temperature(value)
            if key in _GENERATION_CONFIG_KEYS:
                generation_kwargs[key] = value
            else:
                extra_kwargs[key] = value

        if extra_kwargs:
            generation_kwargs["extra_kwargs"] = extra_kwargs
        return symbols.generation_config(**generation_kwargs)

    def _get_inference_parameters(self) -> dict[str, Any]:
        inference_parameters = self.model_params.get("inference_parameters", {})
        if inference_parameters is None:
            return {}
        if not isinstance(inference_parameters, dict):
            raise ValueError("translation_model_config.params.inference_parameters must be a mapping")
        return inference_parameters

    def _resolve_api_key(self, provider: str, client_config: dict[str, Any]) -> str | None:
        explicit_key = client_config.get("api_key") or self.model_params.get("api_key")
        if explicit_key:
            return str(explicit_key)

        api_key_env = client_config.get("api_key_env") or self.model_params.get("api_key_env")
        env_names = [str(api_key_env)] if api_key_env else list(_DEFAULT_PROVIDER_KEY_ENVS.get(provider, ()))
        for env_name in env_names:
            if env_name and os.environ.get(env_name):
                return os.environ[env_name]

        if provider == "nvidia":
            raise RuntimeError(
                "Set NGC_API_KEY or NVIDIA_API_KEY before running Curator experimental BYOB translation."
            )
        return None

    def _validate_required_columns(self, dataframe: pd.DataFrame) -> None:
        required_columns = [self.id_field, self.text_field, self.src_lang_field, self.tgt_lang_field]
        missing = [column for column in required_columns if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Curator experimental translation input is missing required columns: {missing}")

    def _get_language_pair(self, dataframe: pd.DataFrame) -> tuple[str, str]:
        source_langs = _unique_nonempty_values(dataframe[self.src_lang_field])
        target_langs = _unique_nonempty_values(dataframe[self.tgt_lang_field])
        if len(source_langs) != 1 or len(target_langs) != 1:
            raise ValueError(
                "Curator experimental BYOB translation expects one source/target language pair per translation pass. "
                f"Got source={source_langs}, target={target_langs}."
            )
        return source_langs[0], target_langs[0]


def _coerce_single_batch(result: Any) -> Any:
    if result is None:
        raise RuntimeError("Curator experimental translation stage returned no batch")
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"Curator experimental translation stage returned {len(result)} batches; expected 1")
        return result[0]
    return result


def _get_nested_value(*values: Any, default: Any = None) -> Any:
    config_values = [value for value in values if isinstance(value, dict)]
    keys = [value for value in values if isinstance(value, str)]
    for key in keys:
        for config in config_values:
            if key in config:
                return config[key]
    return default


def _normalize_temperature(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    distribution_type = value.get("distribution_type")
    params = value.get("params") or {}
    if distribution_type == "uniform" and {"low", "high"}.issubset(params):
        return (float(params["low"]) + float(params["high"])) / 2.0
    return value


def _normalize_language_code(language_code: str) -> str:
    if language_code == "hinglish":
        return language_code
    return language_code.split("-")[0].split("_")[0].lower()


def _unique_nonempty_values(series: pd.Series) -> list[str]:
    values = []
    for value in series.dropna().tolist():
        value_str = str(value).strip()
        if value_str and value_str not in values:
            values.append(value_str)
    return values
