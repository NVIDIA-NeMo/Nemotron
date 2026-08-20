"""Stable handoff between retrieval SDG generation and data preparation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

GENERATION_MANIFEST_FILENAME = "generation_result.json"
GENERATION_MANIFEST_SCHEMA_VERSION = 1


def write_generation_manifest(
    *,
    output_dir: Path,
    output_path: Path,
    dataset_name: str,
) -> Path:
    """Atomically publish the exact output of a successful generation run."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = output_path.resolve()
    if not resolved_output.is_file():
        raise FileNotFoundError(f"Generation output does not exist: {resolved_output}")
    if not dataset_name:
        raise ValueError("dataset_name must not be empty")

    try:
        stored_output = str(resolved_output.relative_to(output_dir))
    except ValueError:
        stored_output = str(resolved_output)

    payload = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "output_path": stored_output,
    }
    manifest_path = output_dir / GENERATION_MANIFEST_FILENAME

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{GENERATION_MANIFEST_FILENAME}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        temporary_path.replace(manifest_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return manifest_path


def resolve_generation_input(input_path: Path) -> Path:
    """Resolve a generation-result manifest, preserving explicit data paths."""
    input_path = input_path.resolve()
    if input_path.name != GENERATION_MANIFEST_FILENAME:
        return input_path

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid generation manifest: {input_path}") from exc

    if payload.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported generation manifest schema in {input_path}: {payload.get('schema_version')!r}")

    stored_output = payload.get("output_path")
    dataset_name = payload.get("dataset_name")
    if not isinstance(stored_output, str) or not stored_output:
        raise ValueError(f"Generation manifest has no output_path: {input_path}")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(f"Generation manifest has no dataset_name: {input_path}")

    output_path = Path(stored_output)
    if not output_path.is_absolute():
        output_path = input_path.parent / output_path
    output_path = output_path.resolve()
    if not output_path.is_file():
        raise FileNotFoundError(f"Generation output referenced by {input_path} does not exist: {output_path}")

    return output_path
