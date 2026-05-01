#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/translate/translation"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Run corpus translation with NeMo Curator's TranslationStage."""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
log = logging.getLogger(__name__)


def _required_path(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not value:
        raise ValueError(f"{key} is required")
    return str(value)


def _infer_local_dir_format(input_path: Path) -> str:
    has_jsonl = any(input_path.glob("*.jsonl"))
    has_parquet = any(input_path.glob("*.parquet"))
    if has_jsonl and has_parquet:
        raise ValueError(
            f"Input directory {input_path} mixes JSONL and Parquet files. "
            "Set input_format explicitly or split the dataset."
        )
    if has_jsonl:
        return "jsonl"
    if has_parquet:
        return "parquet"
    raise FileNotFoundError(f"No .jsonl or .parquet files found in {input_path}")


def _infer_input_format(input_path: str, configured_format: str | None) -> str:
    if configured_format and configured_format != "auto":
        return configured_format

    lower_path = input_path.lower()
    if lower_path.endswith(".jsonl") or ".jsonl" in lower_path:
        return "jsonl"
    if lower_path.endswith(".parquet") or ".parquet" in lower_path:
        return "parquet"

    path_obj = Path(input_path)
    if path_obj.exists() and path_obj.is_dir():
        return _infer_local_dir_format(path_obj)

    matches = glob.glob(input_path)
    matched_suffixes = {Path(match).suffix for match in matches}
    if matched_suffixes == {".jsonl"}:
        return "jsonl"
    if matched_suffixes == {".parquet"}:
        return "parquet"

    raise ValueError(
        "Could not infer translation input format. Use a .jsonl/.parquet path, "
        "a homogeneous directory, or set input_format explicitly."
    )


def _build_reader(input_path: str, config: dict[str, Any]) -> Any:
    from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader

    input_format = _infer_input_format(input_path, config.get("input_format"))
    reader_kwargs = {
        "file_paths": input_path,
        "files_per_partition": config.get("files_per_partition"),
        "blocksize": config.get("blocksize"),
    }
    if input_format == "jsonl":
        return JsonlReader(**reader_kwargs)
    if input_format == "parquet":
        return ParquetReader(**reader_kwargs)
    raise ValueError(f"Unsupported input_format: {input_format}")


def _build_writer(output_dir: str, config: dict[str, Any]) -> Any:
    from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter

    output_format = str(config.get("output_format", "jsonl"))
    if output_format == "jsonl":
        return JsonlWriter(path=output_dir, mode="overwrite")
    if output_format == "parquet":
        return ParquetWriter(path=output_dir, mode="overwrite")
    raise ValueError(f"Unsupported output_format: {output_format}")


def _build_curator_client(config: dict[str, Any], *, enable_faith: bool) -> Any | None:
    backend = str(config.get("backend", "llm"))
    if backend != "llm" and not enable_faith:
        return None

    from nemo_curator.models.client.openai_client import AsyncOpenAIClient

    server = config.get("server", {}) or {}
    api_key_env = str(server.get("api_key_env", "NVIDIA_API_KEY"))
    api_key = server.get("api_key") or os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            "An API key is required when backend='llm' or faith_eval.enabled=true. "
            f"Set server.api_key or ${api_key_env}."
        )

    return AsyncOpenAIClient(
        max_concurrent_requests=int(config.get("max_concurrent_requests", 64)),
        base_url=server.get("url", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
    )


def _backend_config(config: dict[str, Any]) -> dict[str, Any]:
    backend = str(config.get("backend", "llm"))
    if backend in {"google", "aws", "nmt"}:
        return dict(config.get(backend, {}) or {})
    return {}


def _text_field(value: Any) -> str | list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


def _build_translation_stage(config: dict[str, Any]) -> Any:
    from nemo_curator.stages.text.translation import TranslationStage

    faith_cfg = config.get("faith_eval", {}) or {}
    enable_faith = bool(faith_cfg.get("enabled", False))
    server = config.get("server", {}) or {}

    return TranslationStage(
        source_lang=str(config["source_language"]).lower(),
        target_lang=str(config["target_language"]).lower(),
        text_field=_text_field(config.get("text_field", "messages.*.content")),
        output_field=str(config.get("output_field", "translated_text")),
        segmentation_mode=str(config.get("segmentation_mode", "coarse")),
        min_segment_chars=int(config.get("min_segment_chars", 0)),
        client=_build_curator_client(config, enable_faith=enable_faith),
        model_name=str(server.get("model") or ""),
        backend_type=str(config.get("backend", "llm")),
        backend_config=_backend_config(config),
        enable_faith_eval=enable_faith,
        faith_threshold=float(faith_cfg.get("threshold", 2.5)),
        faith_model_name=str(faith_cfg.get("model_name") or server.get("model") or ""),
        segment_level=bool(faith_cfg.get("segment_level", False)),
        filter_enabled=bool(faith_cfg.get("filter_enabled", True)),
        output_mode=str(config.get("output_mode", "both")),
        merge_scores=bool(config.get("merge_scores", True)),
        reconstruct_messages=bool(config.get("reconstruct_messages", True)),
        messages_field=str(config.get("messages_field", "messages")),
        messages_content_field=str(config.get("messages_content_field", "content")),
        skip_translated=bool(config.get("skip_translated", False)),
        translation_column=str(config.get("translation_column", "translated_text")),
    )


def run(config: dict[str, Any]) -> Path:
    from nemo_curator.pipeline import Pipeline

    input_path = _required_path(config, "input_path")
    output_dir = Path(_required_path(config, "output_dir"))

    pipeline = Pipeline(name="nemotron_translate")
    pipeline.add_stage(_build_reader(input_path, config))
    pipeline.add_stage(_build_translation_stage(config))
    pipeline.add_stage(_build_writer(str(output_dir), config))
    pipeline.run()

    log.info("Translation complete. Wrote output shards under %s", output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(args.config.read_text()) or {}
    run(config)


if __name__ == "__main__":
    main()
