"""Stable handoff between retrieval SDG generation and data preparation."""

from __future__ import annotations

import hashlib
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
    portable_bundle: Path | None = None,
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
    if portable_bundle is not None:
        bundle_manifest = portable_bundle.resolve() / "run_manifest.json"
        if not bundle_manifest.is_file():
            raise FileNotFoundError(f"Portable bundle manifest does not exist: {bundle_manifest}")
        payload["portable_bundle_manifest"] = os.path.relpath(bundle_manifest, output_dir)
        payload["portable_bundle_sha256"] = _file_sha256(bundle_manifest)
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


def resolve_generation_input(input_path: Path, *, allow_portable: bool = False) -> Path:
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
    if payload.get("portable_bundle_manifest") and not allow_portable:
        raise ValueError(
            "Portable SDG output requires explicit retrieval_view selection; legacy conversion is disabled"
        )

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


def resolve_portable_training_input(input_path: Path, view: str, split_protocol: str | None = None) -> Path:
    """Resolve and verify an explicitly selected portable training view.

    Args:
        input_path: Stage 0 generation-result manifest.
        view: One of text, image, or image_and_text. No fallback is performed.

    Returns:
        The bundle's nonempty positive-only training JSON for downstream mining.

    Raises:
        ValueError: If the handoff, view, integrity, or training data is invalid.
        FileNotFoundError: If a referenced artifact is missing.
    """
    if view not in {"text", "image", "image_and_text"}:
        raise ValueError(f"Unsupported portable retrieval view: {view}")
    input_path = input_path.resolve()
    resolve_generation_input(input_path, allow_portable=True)
    if input_path.name != GENERATION_MANIFEST_FILENAME:
        raise ValueError("Portable view selection requires a Stage 0 generation-result manifest")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    stored_bundle = payload.get("portable_bundle_manifest")
    if not isinstance(stored_bundle, str) or not stored_bundle:
        raise ValueError("Stage 0 handoff has no portable bundle; enable portable_export during SDG")
    manifest_path = (input_path.parent / stored_bundle).resolve()
    if _file_sha256(manifest_path) != payload.get("portable_bundle_sha256"):
        raise ValueError("Portable bundle manifest changed after generation handoff")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported portable bundle schema")
    _check_split_protocol(manifest, split_protocol)
    paths = _grouped_bundle_paths(manifest_path, view)
    if not json.loads(paths.train.read_text())["data"]:
        raise ValueError(f"Portable view {view} has no accepted training examples; no fallback is allowed")
    return paths.train


def resolve_portable_evaluation_input(
    input_path: Path, view: str, split_protocol: str | None = None
) -> tuple[Path, Path]:
    """Resolve and verify one portable synthetic-evaluation view.

    Args:
        input_path: Stage 0 generation-result manifest.
        view: One of text, image, or image_and_text.

    Returns:
        Tuple of the BEIR evaluation directory and portable bundle root.
    """
    if view not in {"text", "image", "image_and_text"}:
        raise ValueError(f"Unsupported portable retrieval view: {view}")
    input_path = input_path.resolve()
    resolve_generation_input(input_path, allow_portable=True)
    if input_path.name != GENERATION_MANIFEST_FILENAME:
        raise ValueError("Portable view selection requires a Stage 0 generation-result manifest")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    stored_bundle = payload.get("portable_bundle_manifest")
    if not isinstance(stored_bundle, str) or not stored_bundle:
        raise ValueError("Stage 0 handoff has no portable bundle; enable portable_export during SDG")
    manifest_path = (input_path.parent / stored_bundle).resolve()
    if _file_sha256(manifest_path) != payload.get("portable_bundle_sha256"):
        raise ValueError("Portable bundle manifest changed after generation handoff")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported portable bundle schema")
    _check_split_protocol(manifest, split_protocol)
    paths = _grouped_bundle_paths(manifest_path, view)
    return paths.synthetic_eval, paths.root


def _check_split_protocol(manifest: dict, expected: str | None) -> None:
    """Reject obsolete document-partition exports instead of reinterpreting them."""
    actual = manifest.get("split_protocol")
    if actual != "grouped_query_disjoint":
        raise ValueError(f"Portable input requires grouped_query_disjoint; re-export bundle with protocol {actual!r}")
    if expected is not None and expected != actual:
        raise ValueError(f"Requested split protocol {expected} differs from bundle {actual}")
    scope = "full_collection"
    if manifest.get("corpus_scope") != scope:
        raise ValueError(f"Split protocol {actual} requires corpus_scope={scope}")


def _grouped_bundle_paths(manifest_path: Path, view: str):
    """Use complete semantic and artifact validation on the new protocol path."""
    from nemotron.recipes.retrieval_vl import inspect_vl_bundle

    return inspect_vl_bundle(manifest_path, view="text_image" if view == "image_and_text" else view)


def _file_sha256(path: Path) -> str:
    """Hash large portable corpora without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
