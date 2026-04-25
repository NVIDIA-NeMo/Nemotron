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

"""Translation utilities for customization recipes."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from omegaconf import DictConfig

log = logging.getLogger(__name__)

_DEFAULT_JSONL_CHUNK_SIZE = 5000
_FAITH_COLUMN_TO_KEY = {
    "faith_fluency": "Fluency",
    "faith_accuracy": "Accuracy",
    "faith_idiomaticity": "Idiomaticity",
    "faith_terminology": "Terminology",
    "faith_handling_of_format": "Handling_of_Format",
    "faith_avg": "average",
}


def _require_pandas():
    """Import pandas lazily so this module remains import-safe without it."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised in import-only envs
        raise ImportError(
            "pandas is required for translation data loading. "
            "Install with: pip install pandas"
        ) from exc
    return pd


def _write_jsonl(path: Path, records: list[dict], append: bool = False) -> None:
    """Write records as JSONL."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of records."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _iter_parquet_frames(
    path: Path,
    pd: Any,
    chunk_size: int,
) -> Iterator[tuple[Any, str]]:
    """Yield one parquet file in record batches when pyarrow is available."""
    if chunk_size <= 0:
        yield pd.read_parquet(path), path.stem
        return

    try:
        import pyarrow.parquet as pq
    except ImportError:
        yield pd.read_parquet(path), path.stem
        return

    parquet_file = pq.ParquetFile(path)
    for chunk_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        yield batch.to_pandas(), f"{path.stem}-chunk{chunk_idx}"


def _iter_record_batches(
    path: Path,
    pd: Any,
    chunk_size: int,
) -> Iterator[tuple[list[dict[str, Any]], str]]:
    """Yield BYOB records in manageable batches."""
    if path.suffix in (".jsonl", ".parquet"):
        for df, dataset_name in _iter_input_frames(path, pd, chunk_size):
            yield [dict(record) for record in df.to_dict(orient="records")], dataset_name
        return

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("Expected .json benchmark file to contain a top-level list")
        batch_size = chunk_size if chunk_size > 0 else len(payload)
        for chunk_idx in range(0, len(payload), batch_size):
            yield payload[chunk_idx : chunk_idx + batch_size], f"{path.stem}-chunk{chunk_idx // batch_size}"
        return

    raise ValueError(
        f"Unsupported BYOB benchmark format: {path.suffix} "
        "(expected .parquet or .jsonl)"
    )


def _iter_input_frames(
    input_path: Path,
    pd: Any,
    jsonl_chunk_size: int,
) -> Iterator[tuple[Any, str]]:
    """Yield input DataFrames one file/chunk at a time."""

    def _yield_jsonl(path: Path) -> Iterator[tuple[Any, str]]:
        if jsonl_chunk_size > 0:
            for chunk_idx, chunk in enumerate(
                pd.read_json(path, lines=True, chunksize=jsonl_chunk_size)
            ):
                yield chunk, f"{path.stem}-chunk{chunk_idx}"
        else:
            yield pd.read_json(path, lines=True), path.stem

    if input_path.is_dir():
        saw_supported_file = False
        for path in sorted(input_path.iterdir()):
            if path.suffix == ".jsonl":
                saw_supported_file = True
                yield from _yield_jsonl(path)
            elif path.suffix == ".parquet":
                saw_supported_file = True
                yield from _iter_parquet_frames(path, pd, jsonl_chunk_size)
        if not saw_supported_file:
            raise FileNotFoundError(f"No .jsonl or .parquet files found in {input_path}")
        return

    if input_path.suffix == ".jsonl":
        yield from _yield_jsonl(input_path)
        return

    if input_path.suffix == ".parquet":
        yield from _iter_parquet_frames(input_path, pd, jsonl_chunk_size)
        return

    raise ValueError(f"Unsupported input format: {input_path}")


def _build_curator_client(translation_cfg: dict[str, Any], *, enable_faith: bool) -> Any | None:
    """Create the Curator LLM client when the translation config needs one."""
    backend = translation_cfg.get("backend", "llm")
    if backend != "llm" and not enable_faith:
        return None

    from nemo_curator.models.client.openai_client import AsyncOpenAIClient

    server = translation_cfg.get("server", {}) or {}
    api_key = server.get("api_key") or os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise ValueError(
            "server.api_key is required when backend='llm' or "
            "faith_eval.enabled=True (set NVIDIA_API_KEY env var or "
            "config server.api_key)"
        )

    return AsyncOpenAIClient(
        max_concurrent_requests=translation_cfg.get("max_concurrent_requests", 64),
        base_url=server.get("url", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
    )


def _build_curator_backend_config(translation_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the backend-specific Curator config from a translation config."""
    backend = translation_cfg.get("backend", "llm")
    if backend == "google":
        return dict(translation_cfg.get("google", {}) or {})
    if backend == "aws":
        return dict(translation_cfg.get("aws", {}) or {})
    if backend == "nmt":
        return dict(translation_cfg.get("nmt", {}) or {})
    return {}


def _run_curator_stage(df: Any, stage: Any, dataset_name: str) -> Any:
    """Run one Curator stage or composite stage on a DataFrame and return its DataFrame output."""
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import DocumentBatch

    batch = DocumentBatch(
        task_id=f"{stage.name}-{dataset_name}",
        dataset_name=dataset_name,
        data=df,
    )
    results = Pipeline(name=f"{stage.name}-{dataset_name}", stages=[stage]).run(
        initial_tasks=[batch]
    )
    if not results:
        raise RuntimeError(f"{stage.name} returned no results")

    for result in results:
        if hasattr(result, "to_pandas"):
            result_df = result.to_pandas()
            if not result_df.empty:
                return result_df

    raise RuntimeError(f"{stage.name} returned no DataFrame results")


def _build_translation_stage(translation_cfg: dict[str, Any]) -> Any:
    """Build one Curator translation stage from a recipe config."""
    try:
        from nemo_curator.stages.text.translation import TranslationPipeline
    except ImportError as exc:
        raise ImportError(
            "nemo-curator is required for the Curator translation pipeline. "
            "Install with: pip install nemo-curator"
        ) from exc

    faith_cfg = translation_cfg.get("faith_eval", {}) or {}
    enable_faith = bool(faith_cfg.get("enabled", False))

    return TranslationPipeline(
        source_lang=str(translation_cfg.get("source_lang", "en")),
        target_lang=str(translation_cfg.get("target_lang", "hi")),
        text_field=str(translation_cfg.get("text_field", "text")),
        output_field=str(translation_cfg.get("output_field", "translated_text")),
        segmentation_mode=str(translation_cfg.get("segmentation_mode", "coarse")),
        client=_build_curator_client(translation_cfg, enable_faith=enable_faith),
        model_name=str((translation_cfg.get("server", {}) or {}).get("model", "")),
        backend_type=str(translation_cfg.get("backend", "llm")),
        backend_config=_build_curator_backend_config(translation_cfg),
        enable_faith_eval=enable_faith,
        faith_threshold=float(faith_cfg.get("threshold", 2.5)),
        segment_level=bool(faith_cfg.get("segment_level", False)),
        filter_enabled=bool(faith_cfg.get("filter_enabled", True)),
        preserve_segment_pairs=bool(translation_cfg.get("preserve_segment_pairs", True)),
        output_mode=str(translation_cfg.get("output_mode", "both")),
        merge_scores=bool(translation_cfg.get("merge_scores", True)),
        skip_translated=bool(translation_cfg.get("skip_translated", False)),
    )


def _translate_frame(df: Any, translation_cfg: dict[str, Any], dataset_name: str) -> Any:
    """Run Curator translation on one DataFrame."""
    return _run_curator_stage(
        df,
        _build_translation_stage(translation_cfg),
        dataset_name=dataset_name,
    )


def translate_data(cfg: "DictConfig") -> Path:
    """Translate a dataset with Curator's ``TranslationPipeline``."""
    from omegaconf import OmegaConf

    pd = _require_pandas()

    t_cfg = cfg.get("translation") if isinstance(cfg, DictConfig) else None
    if t_cfg is None:
        raise ValueError("Missing required 'translation' config key")
    t = OmegaConf.to_container(t_cfg, resolve=True)

    if not t.get("input_path"):
        raise ValueError("translation.input_path is required")
    if not t.get("output_dir"):
        raise ValueError("translation.output_dir is required")

    input_path = Path(t["input_path"])
    output_dir = Path(t["output_dir"])
    backend = t.get("backend", "llm")
    jsonl_chunk_size = int(
        t.get("jsonl_chunk_size", t.get("chunk_size", _DEFAULT_JSONL_CHUNK_SIZE))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "translated.jsonl"
    _write_jsonl(output_file, [])

    total_rows_in = 0
    total_rows_out = 0
    total_batches = 0

    for df, dataset_name in _iter_input_frames(input_path, pd, jsonl_chunk_size):
        if df.empty:
            log.info("Skipping empty translation batch %s", dataset_name)
            continue

        total_rows_in += len(df)
        total_batches += 1
        log.info(
            "Loaded %d rows from %s [%s] using backend=%s",
            len(df),
            input_path,
            dataset_name,
            backend,
        )

        result_df = _translate_frame(df, t, dataset_name)
        _write_jsonl(output_file, result_df.to_dict(orient="records"), append=True)
        total_rows_out += len(result_df)

    log.info(
        "Translation complete: %d input rows -> %d output rows written to %s across %d batch(es)",
        total_rows_in,
        total_rows_out,
        output_file,
        total_batches,
    )
    return output_dir


# ---------------------------------------------------------------------------
# BYOB MCQ translation (Stage 4)
# ---------------------------------------------------------------------------


def _collect_mcq_translatable_strings(
    records: list[dict],
) -> tuple[list[dict], list[tuple[int, str, object]]]:
    """Extract translatable MCQ strings while preserving their positions."""
    staged: list[dict] = []
    index: list[tuple[int, str, object]] = []
    for rec_idx, rec in enumerate(records):
        q = rec.get("question")
        if isinstance(q, str) and q.strip():
            staged.append({"text": q})
            index.append((rec_idx, "question", None))

        opts = rec.get("options")
        if isinstance(opts, dict):
            for key in opts:
                val = opts[key]
                if isinstance(val, str) and val.strip():
                    staged.append({"text": val})
                    index.append((rec_idx, "options_dict", key))
        elif isinstance(opts, list):
            for i, val in enumerate(opts):
                if isinstance(val, str) and val.strip():
                    staged.append({"text": val})
                    index.append((rec_idx, "options_list", i))
    return staged, index


def _reassemble_mcq_records(
    original_records: list[dict],
    index: list[tuple[int, str, object]],
    translated_rows: list[dict[str, Any]],
    target_lang: str,
    translated_field: str = "translated_text",
) -> list[dict]:
    """Merge translated rows back into a deep copy of the original records."""
    out = [copy.deepcopy(r) for r in original_records]
    if len(index) != len(translated_rows):
        raise RuntimeError(
            f"Translation output length mismatch: expected {len(index)} "
            f"translated strings, got {len(translated_rows)}.  This usually "
            "means rows were dropped by FAITH filtering; ensure the BYOB "
            "translation path runs with faith_eval.filter_enabled=False."
        )
    record_metadata = [
        _init_mcq_translation_metadata(record, target_lang) for record in original_records
    ]
    record_score_values = [
        {column: [] for column in _FAITH_COLUMN_TO_KEY} for _ in original_records
    ]
    record_time_totals = [0.0 for _ in original_records]
    record_error_lists = [[] for _ in original_records]

    for (rec_idx, kind, key), translated_row in zip(index, translated_rows):
        translated = str(translated_row.get(translated_field, ""))
        segment_pairs = _extract_segment_pairs(
            translated_row=translated_row,
            source_text=_lookup_source_text(original_records[rec_idx], kind, key),
            translated_text=translated,
        )

        if kind == "question":
            out[rec_idx]["question"] = translated
            record_metadata[rec_idx]["translation"]["question"] = translated
            record_metadata[rec_idx]["segmented_translation"]["question"] = segment_pairs
        elif kind == "options_dict":
            out[rec_idx]["options"][key] = translated
            options_translation = record_metadata[rec_idx]["translation"].setdefault(
                "options", copy.deepcopy(original_records[rec_idx].get("options", {}))
            )
            options_segments = record_metadata[rec_idx]["segmented_translation"].setdefault(
                "options",
                {k: [] for k in original_records[rec_idx].get("options", {})},
            )
            options_translation[key] = translated
            options_segments[key] = segment_pairs
        elif kind == "options_list":
            out[rec_idx]["options"][key] = translated
            options_translation = record_metadata[rec_idx]["translation"].setdefault(
                "options", copy.deepcopy(original_records[rec_idx].get("options", []))
            )
            options_segments = record_metadata[rec_idx]["segmented_translation"].setdefault(
                "options",
                [[] for _ in original_records[rec_idx].get("options", [])],
            )
            options_translation[key] = translated
            options_segments[key] = segment_pairs

        for column in _FAITH_COLUMN_TO_KEY:
            value = translated_row.get(column)
            if value is None or value != value:
                continue
            existing = record_score_values[rec_idx].setdefault(column, [])
            existing.append(float(value))

        time_value = translated_row.get("translation_time")
        if time_value is not None and time_value == time_value:
            record_time_totals[rec_idx] += float(time_value)

        error_value = str(translated_row.get("translation_errors", "")).strip()
        if error_value:
            record_error_lists[rec_idx].append(error_value)

    for rec_idx, metadata in enumerate(record_metadata):
        score_values = record_score_values[rec_idx]
        faith_scores = {
            score_key: sum(values) / len(values)
            for column, score_key in _FAITH_COLUMN_TO_KEY.items()
            for values in [score_values.get(column, [])]
            if values
        }
        if faith_scores:
            metadata["faith_scores"] = faith_scores
        out[rec_idx]["translation_metadata"] = metadata

        for column, score_key in _FAITH_COLUMN_TO_KEY.items():
            values = score_values.get(column, [])
            if values:
                out[rec_idx][column] = sum(values) / len(values)

        if record_time_totals[rec_idx]:
            out[rec_idx]["translation_time"] = record_time_totals[rec_idx]
        combined_errors = "; ".join(record_error_lists[rec_idx])
        if combined_errors:
            out[rec_idx]["translation_errors"] = combined_errors

    return out


def _init_mcq_translation_metadata(record: dict[str, Any], target_lang: str) -> dict[str, Any]:
    """Build the record-level raw translation metadata."""
    metadata: dict[str, Any] = {
        "target_lang": target_lang,
        "translation": {},
        "segmented_translation": {},
    }
    if "question" in record:
        metadata["translation"]["question"] = record.get("question")
        metadata["segmented_translation"]["question"] = []

    options = record.get("options")
    if isinstance(options, dict):
        metadata["translation"]["options"] = copy.deepcopy(options)
        metadata["segmented_translation"]["options"] = {key: [] for key in options}
    elif isinstance(options, list):
        metadata["translation"]["options"] = copy.deepcopy(options)
        metadata["segmented_translation"]["options"] = [[] for _ in options]
    return metadata


def _lookup_source_text(record: dict[str, Any], kind: str, key: object) -> str:
    """Return the source string for one staged MCQ field."""
    if kind == "question":
        value = record.get("question", "")
    elif kind == "options_dict":
        value = record.get("options", {}).get(key, "")
    else:
        value = record.get("options", [""])[key]
    return value if isinstance(value, str) else str(value)


def _extract_segment_pairs(
    translated_row: dict[str, Any],
    source_text: str,
    translated_text: str,
) -> list[dict[str, str]]:
    """Extract per-string segment pairs from a translated row."""
    metadata_json = translated_row.get("translation_metadata")
    metadata: dict[str, Any] = {}
    if isinstance(metadata_json, dict):
        metadata = metadata_json
    elif isinstance(metadata_json, str) and metadata_json.strip():
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}
    if metadata:
        segmented = metadata.get("segmented_translation")
        if isinstance(segmented, list):
            return segmented
        if isinstance(segmented, dict):
            content_pairs = segmented.get("content")
            if isinstance(content_pairs, list):
                return content_pairs
            for value in segmented.values():
                if isinstance(value, list):
                    return value
    return [{"src": source_text, "tgt": translated_text}]


def _options_to_list(options: Any) -> list[str]:
    """Normalize MCQ options to an ordered list of strings."""
    if isinstance(options, dict):
        return [str(value) for value in options.values()]
    if isinstance(options, list):
        return [str(value) for value in options]
    return []


def _format_mcq(question: str, options: Any) -> str:
    """Format an MCQ the same way Speaker did for backtranslation metrics."""
    choices = _options_to_list(options)
    choices_flat = "\n".join(
        f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)
    )
    return f"Question: {question}\nOptions:\n{choices_flat}"


def _apply_backtranslation_quality(
    *,
    cfg_dict: dict[str, Any],
    translation_cfg: dict[str, Any],
    source_records: list[dict],
    translated_records: list[dict],
    dataset_name: str,
) -> list[dict]:
    """Run round-trip quality checks with Curator translation and metric stages."""
    from nemo_curator.stages.text.translation import TextQualityMetricStage

    pd = _require_pandas()

    metric_specs = list(cfg_dict.get("backtranslation_quality_metrics") or [])
    if not metric_specs:
        return translated_records

    staged_rows, index = _collect_mcq_translatable_strings(translated_records)
    if not staged_rows:
        for record in translated_records:
            record["is_quality_metric_passed"] = True
        return translated_records

    backtranslation_cfg = dict(translation_cfg)
    backtranslation_cfg["source_lang"] = translation_cfg["target_lang"]
    backtranslation_cfg["target_lang"] = translation_cfg["source_lang"]
    backtranslation_cfg["text_field"] = "text"
    backtranslation_cfg["output_field"] = "backtranslated_text"
    backtranslation_cfg["faith_eval"] = {"enabled": False, "filter_enabled": False}
    backtranslation_cfg["output_mode"] = "replaced"
    backtranslation_cfg["merge_scores"] = False
    backtranslation_cfg["skip_translated"] = False

    backtranslation_df = pd.DataFrame(staged_rows)
    backtranslated_df = _translate_frame(
        backtranslation_df,
        backtranslation_cfg,
        dataset_name=f"{dataset_name}-backtranslation",
    )
    backtranslated_rows = backtranslated_df.to_dict(orient="records")
    backtranslated_records = _reassemble_mcq_records(
        original_records=translated_records,
        index=index,
        translated_rows=backtranslated_rows,
        target_lang=str(translation_cfg["source_lang"]),
        translated_field="backtranslated_text",
    )

    quality_rows = []
    for source_record, backtranslated_record in zip(source_records, backtranslated_records):
        quality_rows.append(
            {
                "reference_text": _format_mcq(
                    str(source_record.get("question", "")),
                    source_record.get("options"),
                ),
                "hypothesis_text": _format_mcq(
                    str(backtranslated_record.get("question", "")),
                    backtranslated_record.get("options"),
                ),
            }
        )

    quality_df = _run_curator_stage(
        pd.DataFrame(quality_rows),
        TextQualityMetricStage(
            reference_text_field="reference_text",
            hypothesis_text_field="hypothesis_text",
            metrics=metric_specs,
            filter_enabled=False,
        ),
        dataset_name=f"{dataset_name}-roundtrip-metrics",
    )
    quality_rows = quality_df.to_dict(orient="records")

    for translated_record, quality_row in zip(translated_records, quality_rows):
        for metric_spec in metric_specs:
            metric_type = str(metric_spec["type"])
            translated_record[f"score_{metric_type}"] = quality_row[f"score_{metric_type}"]
            translated_record[f"score_{metric_type}_passed"] = quality_row[
                f"score_{metric_type}_passed"
            ]
        translated_record["is_quality_metric_passed"] = bool(
            quality_row["is_quality_metric_passed"]
        )

    if cfg_dict.get("remove_low_quality", False):
        translated_records = [
            record for record in translated_records if record.get("is_quality_metric_passed", False)
        ]

    return translated_records


def translate_byob_benchmark(cfg: "DictConfig") -> Path:
    """Translate a BYOB benchmark dataset and preserve MCQ structure."""
    from omegaconf import OmegaConf

    pd = _require_pandas()

    t_cfg = OmegaConf.select(cfg, "translate", default=cfg)
    if isinstance(t_cfg, DictConfig):
        t_cfg_dict = OmegaConf.to_container(t_cfg, resolve=True)
    else:
        t_cfg_dict = dict(t_cfg)

    dataset_path_raw = t_cfg_dict.get("dataset_path")
    if not dataset_path_raw:
        raise ValueError("translate.dataset_path must be set")
    dataset_path_str = str(dataset_path_raw)

    if dataset_path_str.startswith(("http://", "https://", "s3://", "gs://")):
        raise NotImplementedError(
            "Remote URLs are not supported yet for BYOB translation. "
            "Download the dataset first and pass a local path via "
            "translate.dataset_path."
        )

    dataset_path = Path(dataset_path_str)
    chunk_size = int(
        t_cfg_dict.get("jsonl_chunk_size", t_cfg_dict.get("chunk_size", _DEFAULT_JSONL_CHUNK_SIZE))
    )

    out_dir = Path(t_cfg_dict.get("output_dir") or (dataset_path.parent / "translated"))
    out_dir.mkdir(parents=True, exist_ok=True)

    final_file = out_dir / "translated_mcq.jsonl"
    _write_jsonl(final_file, [])

    source_lang = str(t_cfg_dict.get("source_lang", "en"))
    target_lang = str(t_cfg_dict.get("target_lang", "hi"))

    model_cfg = t_cfg_dict.get("translation_model_config", {}) or {}
    backend_name = model_cfg.get("mode", "llm")
    params = model_cfg.get("params", {}) or {}
    infer_params = params.get("inference_parameters", {}) or {}

    user_faith_eval = t_cfg_dict.get("faith_eval") or {}
    faith_eval_cfg: dict = dict(user_faith_eval)
    if faith_eval_cfg.get("filter_enabled", False):
        log.warning(
            "BYOB translation: ignoring user-set faith_eval.filter_enabled=True. "
            "Row dropping would break MCQ reassembly; forcing filter_enabled=False. "
            "Filter post-hoc on translated_mcq.jsonl using the faith_* columns."
        )
    faith_eval_cfg["filter_enabled"] = False

    translation_cfg: dict = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "backend": backend_name,
        "text_field": "text",
        "output_field": "translated_text",
        "segmentation_mode": str(t_cfg_dict.get("segmentation_mode", "coarse")),
        "output_mode": "both",
        "preserve_segment_pairs": True,
        "server": {
            "url": params.get(
                "base_url",
                os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            ),
            "model": params.get("model", ""),
            "api_key": params.get(
                "api_key", os.environ.get("NVIDIA_API_KEY", "")
            ),
        },
        "max_concurrent_requests": infer_params.get("max_parallel_requests", 64),
        "faith_eval": faith_eval_cfg,
        "merge_scores": t_cfg_dict.get("merge_scores", True),
    }

    if backend_name in ("google", "aws", "nmt"):
        translation_cfg[backend_name] = dict(params)

    total_input_records = 0
    total_output_records = 0
    saw_any_records = False

    for records, dataset_name in _iter_record_batches(dataset_path, pd, chunk_size):
        if not records:
            continue

        saw_any_records = True
        total_input_records += len(records)

        staged_rows, index = _collect_mcq_translatable_strings(records)
        if not staged_rows:
            log.info(
                "No translatable strings found in %s [%s]; writing records unchanged",
                dataset_path,
                dataset_name,
            )
            _write_jsonl(final_file, records, append=True)
            total_output_records += len(records)
            continue

        translated_df = _translate_frame(
            pd.DataFrame(staged_rows),
            translation_cfg,
            dataset_name=dataset_name,
        )
        translated_rows = translated_df.to_dict(orient="records")

        merged_records = _reassemble_mcq_records(
            original_records=records,
            index=index,
            translated_rows=translated_rows,
            target_lang=target_lang,
        )
        merged_records = _apply_backtranslation_quality(
            cfg_dict=t_cfg_dict,
            translation_cfg=translation_cfg,
            source_records=records,
            translated_records=merged_records,
            dataset_name=dataset_name,
        )

        _write_jsonl(final_file, merged_records, append=True)
        total_output_records += len(merged_records)

    if not saw_any_records:
        log.warning("No records found in %s", dataset_path)
        return out_dir

    log.info(
        "BYOB benchmark translation complete: %d input records -> %d output records written to %s",
        total_input_records,
        total_output_records,
        final_file,
    )
    return out_dir
