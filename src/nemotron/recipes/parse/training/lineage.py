"""Explicit, content-verified model lineage for Parse stage artifacts."""

import json
from pathlib import Path

from nemotron.recipes.parse.common import model_from_artifact, sha256


def model_reference(path, processor_source=None, *, source=None, revision=None):
    """Record the exact model and processor bytes without modifying either."""
    path = Path(path).resolve()
    processor = Path(processor_source or path).resolve()
    if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"Incomplete model checkpoint: {path}")
    if not processor.is_dir():
        raise FileNotFoundError(f"Processor directory does not exist: {processor}")

    def files(directory, weights):
        suffixes = {".json", ".py", ".txt", ".model", ".tiktoken"}
        if weights:
            suffixes.add(".safetensors")
        return {p.name: sha256(p) for p in sorted(directory.iterdir())
                if p.is_file() and p.suffix in suffixes}

    return dict(path=str(path), processor_source=str(processor), source=source, revision=revision,
                model_files=files(path, True), processor_files=files(processor, False))


def verify_model(reference):
    """Reject changed weights, configuration or processor files."""
    required = {"path", "processor_source", "source", "revision", "model_files", "processor_files"}
    if not isinstance(reference, dict) or set(reference) != required:
        raise ValueError("Incomplete model reference; run the training stages to record lineage")
    actual = model_reference(reference["path"], reference["processor_source"],
                             source=reference["source"], revision=reference["revision"])
    if actual != reference:
        raise ValueError(f"Model or processor changed since lineage was recorded: {reference['path']}")


def read_lineage(artifact_path, stage, dataset_identity):
    """Require newly recorded lineage; never infer ancestors from old runs."""
    artifact_path = Path(artifact_path)
    checkpoint = model_from_artifact(artifact_path)
    record = json.loads(artifact_path.read_text())
    lineage = record.get("lineage", {})
    stages = ("base", "sft") if stage == "sft" else ("base", "sft", "rlvr")
    if (record.get("stage") != stage or lineage.get("schema_version") != 1
            or set(lineage.get("models", {})) != set(stages)):
        raise ValueError(f"Expected {stage} artifact with explicit model lineage; run the training stages again")
    if lineage.get("dataset_identity") != dataset_identity:
        raise ValueError("Evaluation dataset differs from the dataset recorded in model lineage")
    reference = lineage["models"][stage]
    if (Path(reference["path"]).resolve() != checkpoint.resolve()
            or record.get("processor_source") != reference["processor_source"]):
        raise ValueError("Model artifact and lineage disagree about checkpoint or processor")
    for name in stages:
        verify_model(lineage["models"][name])
    return lineage
