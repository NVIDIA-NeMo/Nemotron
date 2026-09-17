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


def resolve_portable_training_input(input_path: Path, view: str) -> Path:
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
    root = manifest_path.parent
    train_file = root / "views" / view / "train.json"
    required = {
        f"views/{view}/train.json",
        f"views/{view}/corpus/train/merlin_metadata.json",
        f"views/{view}/corpus/train/part-00000.parquet",
    }
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    for relative_path in sorted(required):
        if relative_path not in artifacts:
            raise ValueError(f"Portable bundle has no integrity record for {relative_path}")
        actual = _file_sha256(root / relative_path)
        if actual != artifacts[relative_path]["sha256"]:
            raise ValueError(f"Portable bundle artifact changed: {relative_path}")
    training = json.loads(train_file.read_text(encoding="utf-8"))
    if not isinstance(training.get("data"), list) or not training["data"]:
        raise ValueError(f"Portable view {view} has no accepted training examples; no fallback is allowed")
    if training.get("corpus", {}).get("path") != "corpus/train":
        raise ValueError("Portable training corpus reference does not match the verified corpus")
    return train_file


def resolve_portable_evaluation_input(input_path: Path, view: str) -> tuple[Path, Path]:
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
    root = manifest_path.parent
    evaluation_dir = root / "synthetic_eval" / view
    required = {
        f"synthetic_eval/{view}/queries.jsonl",
        f"synthetic_eval/{view}/corpus.jsonl",
        f"synthetic_eval/{view}/qrels/test.tsv",
    }
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    for relative_path in sorted(required):
        if relative_path not in artifacts:
            raise ValueError(f"Portable bundle has no integrity record for {relative_path}")
        if _file_sha256(root / relative_path) != artifacts[relative_path]["sha256"]:
            raise ValueError(f"Portable bundle artifact changed: {relative_path}")
    corpus_path = evaluation_dir / "corpus.jsonl"
    for line_number, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        document = json.loads(line)
        image_path = document.get("image_path")
        if image_path is None:
            continue
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"Invalid corpus image_path at line {line_number}")
        relative = Path(image_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Corpus image_path must be portable and relative: {image_path!r}")
        candidate = root / relative
        if not candidate.exists():
            raise FileNotFoundError(f"Could not resolve corpus image {image_path!r} within {root}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root.resolve(strict=True))
        except ValueError as error:
            raise ValueError(f"Corpus image_path escapes its configured image root: {image_path!r}") from error
        if not resolved.is_file():
            raise ValueError(f"Corpus image_path is not a regular file: {image_path!r}")
        artifact_path = relative.as_posix()
        if artifact_path not in artifacts:
            raise ValueError(f"Portable bundle has no integrity record for {artifact_path}")
        if _file_sha256(resolved) != artifacts[artifact_path]["sha256"]:
            raise ValueError(f"Portable bundle artifact changed: {artifact_path}")
    return evaluation_dir, root


def _file_sha256(path: Path) -> str:
    """Hash large portable corpora without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
