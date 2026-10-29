"""Stable handoff between retrieval SDG generation and data preparation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
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
    """Resolve an explicitly selected portable training view.

    Args:
        input_path: Stage 0 generation-result manifest.
        view: One of text, image, image_and_text, or mixed. No fallback is performed.

    Returns:
        The bundle's nonempty positive-only training JSON for downstream mining.

    Raises:
        ValueError: If the handoff identity, view, or training data is invalid.
        FileNotFoundError: If a referenced artifact is missing.
    """
    paths = _portable_bundle_paths(input_path, view)
    if not json.loads(paths.train.read_text())["data"]:
        raise ValueError(f"Portable view {view} has no accepted training examples; no fallback is allowed")
    return paths.train


def resolve_portable_evaluation_input(input_path: Path, view: str) -> tuple[Path, Path]:
    """Resolve one portable synthetic-evaluation view.

    Args:
        input_path: Stage 0 generation-result manifest.
        view: One of text, image, image_and_text, or mixed.

    Returns:
        Tuple of the BEIR evaluation directory and portable bundle root.

    Raises:
        ValueError: If the selected view has no accepted evaluation queries.
    """
    paths = _portable_bundle_paths(input_path, view)
    with (paths.synthetic_eval / "queries.jsonl").open(encoding="utf-8") as stream:
        if not any(line.strip() for line in stream):
            raise ValueError(f"Portable view {view} has no accepted evaluation queries; no fallback is allowed")
    return paths.synthetic_eval, paths.root


def _portable_bundle_paths(input_path: Path, view: str):
    """Verify handoff identity and resolve the selected plugin export."""
    if view not in {"text", "image", "image_and_text", "mixed"}:
        raise ValueError(f"Unsupported portable retrieval view: {view}")
    input_path = input_path.resolve()
    resolve_generation_input(input_path, allow_portable=True)
    if input_path.name != GENERATION_MANIFEST_FILENAME:
        raise ValueError("Portable view selection requires a Stage 0 generation-result manifest")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    stored_bundle = payload.get("portable_bundle_manifest")
    if not isinstance(stored_bundle, str) or not stored_bundle:
        raise ValueError("Stage 0 handoff has no portable bundle; use sdg_workflow=multimodal")
    manifest_path = (input_path.parent / stored_bundle).resolve()
    if _file_sha256(manifest_path) != payload.get("portable_bundle_sha256"):
        raise ValueError("Portable bundle manifest changed after generation handoff")
    return resolve_bundle_view(manifest_path, view=view)


def _check_split_protocol(manifest: dict) -> None:
    """Reject obsolete document-partition exports instead of reinterpreting them."""
    actual = manifest.get("split_protocol")
    if actual != "grouped_query_disjoint":
        raise ValueError(f"Portable input requires grouped_query_disjoint; re-export bundle with protocol {actual!r}")
    scope = "full_collection"
    if manifest.get("corpus_scope") != scope:
        raise ValueError(f"Split protocol {actual} requires corpus_scope={scope}")


def _file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BundlePaths:
    """Files for one selected Data Designer export view."""

    root: Path
    manifest: Path
    view: str
    train: Path
    validation: Path
    train_corpus: Path
    synthetic_eval: Path


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _contained_path(root: Path, relative_path: str) -> Path:
    """Keep selected artifacts inside the bundle, including symbolic links."""
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
    ):
        raise ValueError(f"Invalid bundle artifact path: {relative_path!r}")
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle artifact escapes its root: {relative_path}")
    return candidate


def resolve_bundle_view(manifest_path: Path, *, view: str) -> BundlePaths:
    """Check the selected view's layout and loader inputs without auditing the export.

    Data Designer owns source consistency and query grouping. Use the explicit
    stage0_sdg.validate command to audit hashes, source contents, and splits.
    """
    import pyarrow.parquet as pq

    classes = {
        "text": "TextQADataset",
        "image": "ColPaliDataset",
        "image_and_text": "WikiSSNQDataset",
        "mixed": "WikiSSNQDataset",
    }
    if view not in classes:
        raise ValueError(f"Unsupported retrieval view: {view!r}")
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Unsupported bundle schema_version: {manifest.get('schema_version')!r}")
    _check_split_protocol(manifest)
    expected_claims = {
        "status": "completed",
        "portable": True,
        "negative_mining_performed": False,
        "relevance_judgements_complete": False,
        "unlisted_document_disposition": "unjudged",
        "relevance_annotation_scope": "known_positives_only",
    }
    for key, expected in expected_claims.items():
        if key in manifest and manifest[key] != expected:
            raise ValueError(f"Unsupported bundle {key}: {manifest[key]!r}")
    if manifest.get("gold_evaluation_status") == "available":
        raise ValueError("Bundle has no independent gold evaluation contract")
    counts = manifest.get("counts")
    if counts is not None and not isinstance(counts, dict):
        raise ValueError("Bundle counts must be an object")
    views = counts.get("views") if isinstance(counts, dict) else None
    if views is not None and not isinstance(views, dict):
        raise ValueError("Bundle views must be an object")
    if isinstance(views, dict) and view not in views:
        available = ", ".join(name for name in classes if name in views) or "none"
        raise ValueError(
            f"Requested retrieval view '{view}' is unavailable in this bundle. "
            f"Available views: {available}. Select an available retrieval_view or re-export the requested view."
        )
    root = manifest_path.parent
    paths = BundlePaths(
        root,
        manifest_path,
        view,
        _contained_path(root, f"views/{view}/train.json"),
        _contained_path(root, f"views/{view}/validation.json"),
        _contained_path(root, f"views/{view}/corpus/shared"),
        _contained_path(root, f"synthetic_eval/{view}"),
    )
    required = [paths.train, paths.validation, paths.train_corpus / "merlin_metadata.json"]
    required += [paths.synthetic_eval / name for name in ("queries.jsonl", "corpus.jsonl", "qrels/test.tsv")]
    for path in required:
        path = _contained_path(root, path.relative_to(root).as_posix())
        if not path.is_file():
            raise ValueError(f"Bundle missing required artifact: {path.relative_to(root)}")
    metadata = _load_json(paths.train_corpus / "merlin_metadata.json")
    corpus_id = metadata.get("corpus_id")
    if (
        metadata.get("class") != classes[view]
        or metadata.get("format") != "parquet"
        or not isinstance(corpus_id, str)
        or not corpus_id.strip()
    ):
        raise ValueError(f"Unsupported corpus metadata for {view}: {paths.train_corpus}")
    shards = sorted(paths.train_corpus.glob("*.parquet"))
    if not shards:
        raise ValueError(f"Bundle missing required corpus shard: {paths.train_corpus}")
    columns = {
        "text": {"id", "text"},
        "image": {"image_filename", "image"},
        "image_and_text": {"docid", "text", "image"},
        "mixed": {"docid", "text", "image"},
    }
    for path in shards:
        path = _contained_path(root, path.relative_to(root).as_posix())
        try:
            schema = pq.read_schema(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"Unreadable Parquet corpus shard: {path}") from error
        if not columns[view].issubset(schema.names):
            raise ValueError(f"Corpus shard is missing columns for {view}: {path}")
    for path in (paths.train, paths.validation):
        payload = _load_json(path)
        if payload.get("corpus") != {"path": "corpus/shared"} or not isinstance(payload.get("data"), list):
            raise ValueError(f"Invalid training corpus reference or data list: {path}")
    return paths
