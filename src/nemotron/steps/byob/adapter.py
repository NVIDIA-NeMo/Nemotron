"""Schema adapters for BYOB MCQ benchmark rows."""

from __future__ import annotations

import copy
import json
from typing import Any


def flatten_mcq_records(
    records: list[dict[str, Any]],
    *,
    text_field: str = "text",
) -> tuple[list[dict[str, str]], list[tuple[int, str, object]]]:
    """Extract translatable MCQ strings while preserving their original positions."""
    staged_rows: list[dict[str, str]] = []
    index: list[tuple[int, str, object]] = []

    for rec_idx, record in enumerate(records):
        question = record.get("question")
        if isinstance(question, str) and question.strip():
            staged_rows.append({text_field: question})
            index.append((rec_idx, "question", None))

        options = record.get("options")
        if isinstance(options, dict):
            for key, value in options.items():
                if isinstance(value, str) and value.strip():
                    staged_rows.append({text_field: value})
                    index.append((rec_idx, "options_dict", key))
        elif isinstance(options, list):
            for option_idx, value in enumerate(options):
                if isinstance(value, str) and value.strip():
                    staged_rows.append({text_field: value})
                    index.append((rec_idx, "options_list", option_idx))

    return staged_rows, index


def restore_mcq_records(
    original_records: list[dict[str, Any]],
    index: list[tuple[int, str, object]],
    translated_rows: list[dict[str, Any]],
    *,
    target_lang: str,
    translated_field: str = "translated_text",
) -> list[dict[str, Any]]:
    """Merge translated strings back into the original MCQ schema."""
    if len(index) != len(translated_rows):
        raise RuntimeError(
            "Translation output length mismatch. Reassembly requires one translated row for every staged string."
        )

    out = [copy.deepcopy(record) for record in original_records]
    record_metadata = [_init_translation_metadata(record, target_lang) for record in original_records]
    record_time_totals = [0.0 for _ in original_records]
    record_error_lists = [[] for _ in original_records]

    for (rec_idx, kind, key), translated_row in zip(index, translated_rows, strict=True):
        translated_text = str(translated_row.get(translated_field, ""))
        source_text = _lookup_source_text(original_records[rec_idx], kind, key)
        segment_pairs = _extract_segment_pairs(translated_row, source_text, translated_text)

        if kind == "question":
            out[rec_idx]["question"] = translated_text
            record_metadata[rec_idx]["translation"]["question"] = translated_text
            record_metadata[rec_idx]["segmented_translation"]["question"] = segment_pairs
        elif kind == "options_dict":
            out[rec_idx]["options"][key] = translated_text
            _restore_dict_option_metadata(record_metadata[rec_idx], original_records[rec_idx], key, translated_text)
            record_metadata[rec_idx]["segmented_translation"].setdefault(
                "options",
                {option_key: [] for option_key in original_records[rec_idx].get("options", {})},
            )[key] = segment_pairs
        else:
            out[rec_idx]["options"][key] = translated_text
            _restore_list_option_metadata(record_metadata[rec_idx], original_records[rec_idx], key, translated_text)
            record_metadata[rec_idx]["segmented_translation"].setdefault(
                "options",
                [[] for _ in original_records[rec_idx].get("options", [])],
            )[key] = segment_pairs

        _collect_row_run_info(rec_idx, translated_row, record_time_totals, record_error_lists)

    for rec_idx, metadata in enumerate(record_metadata):
        out[rec_idx]["translation_metadata"] = metadata
        if record_time_totals[rec_idx]:
            out[rec_idx]["translation_time"] = record_time_totals[rec_idx]
        if record_error_lists[rec_idx]:
            out[rec_idx]["translation_errors"] = "; ".join(record_error_lists[rec_idx])

    return out


def format_mcq_for_metrics(question: str, options: Any) -> str:
    """Format an MCQ into a stable string for round-trip quality metrics."""
    choices = _options_to_list(options)
    choices_flat = "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
    return f"Question: {question}\nOptions:\n{choices_flat}"


def _init_translation_metadata(record: dict[str, Any], target_lang: str) -> dict[str, Any]:
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
    metadata_json = translated_row.get("translation_metadata")
    metadata: dict[str, Any] = {}
    if isinstance(metadata_json, dict):
        metadata = metadata_json
    elif isinstance(metadata_json, str) and metadata_json.strip():
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}

    segmented = metadata.get("segmented_translation") if metadata else None
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


def _restore_dict_option_metadata(
    metadata: dict[str, Any],
    original_record: dict[str, Any],
    key: object,
    translated_text: str,
) -> None:
    metadata["translation"].setdefault("options", copy.deepcopy(original_record.get("options", {})))[key] = (
        translated_text
    )


def _restore_list_option_metadata(
    metadata: dict[str, Any],
    original_record: dict[str, Any],
    key: object,
    translated_text: str,
) -> None:
    metadata["translation"].setdefault("options", copy.deepcopy(original_record.get("options", [])))[key] = (
        translated_text
    )


def _collect_row_run_info(
    rec_idx: int,
    translated_row: dict[str, Any],
    record_time_totals: list[float],
    record_error_lists: list[list[str]],
) -> None:
    time_value = translated_row.get("translation_time")
    if time_value is not None and time_value == time_value:
        record_time_totals[rec_idx] += float(time_value)

    error_value = str(translated_row.get("translation_errors", "")).strip()
    if error_value:
        record_error_lists[rec_idx].append(error_value)


def _options_to_list(options: Any) -> list[str]:
    if isinstance(options, dict):
        return [str(value) for value in options.values()]
    if isinstance(options, list):
        return [str(value) for value in options]
    return []
