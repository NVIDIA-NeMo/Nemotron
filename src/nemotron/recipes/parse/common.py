"""Local artifacts and terminal progress for the three-stage Parse recipe."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from nemotron.recipes.parse import SOURCE_REVISION


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def stage_run(stage, output, config):
    """Claim a new output directory, retaining failures for inspection."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    # Never silently append training to a previous run or replace an artifact.
    if any(output.iterdir()):
        raise FileExistsError(f"Choose a new, empty output directory: {output}")
    with (output / ".started").open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    started = time.time()
    receipt = dict(stage=stage, status="running", started=started, source_revision=SOURCE_REVISION)
    write_json(output / "config.json", config.model_dump(mode="json"))
    write_json(output / "stage.json", receipt)
    try:
        yield output
    except BaseException as exc:
        receipt.update(status="failed", error=type(exc).__name__, finished=time.time())
        write_json(output / "stage.json", receipt)
        raise
    else:
        receipt.update(status="completed", finished=time.time())
        write_json(output / "stage.json", receipt)


def artifact(output: Path, kind: str, path: Path, **metadata) -> Path:
    from nemotron.kit.artifacts.base import Artifact

    # Keep bookkeeping in the stage bundle. Adding metadata to an evaluated
    # checkpoint would change its evaluation identity after the answers are saved.
    # No remote registry or tracking backend is initialized by this recipe.
    Artifact(path=output.absolute(), type=kind,
             metadata=dict(source_revision=SOURCE_REVISION, payload_path=str(path.absolute()), **metadata)).save()
    target = output / "artifact.json"
    write_json(target, dict(schema_version=1, kind=kind, path=str(path.absolute()),
                            source_revision=SOURCE_REVISION, **metadata))
    return target


def model_from_artifact(path: Path) -> Path:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("kind") != "parse/model":
        raise ValueError(f"Expected a Parse model artifact: {path}")
    checkpoint = Path(value["path"])
    if not (checkpoint / "config.json").is_file() or not list(checkpoint.glob("*.safetensors")):
        raise FileNotFoundError(f"Incomplete model checkpoint: {checkpoint}")
    return checkpoint


def dataset_identity(directory: Path) -> dict:
    """Bind the three fixed splits to their actual manifests and image bytes."""
    from nemotron.recipes.parse.dental.data import load_pages

    manifests, images, contexts, counts = {}, {}, {}, {}
    for name in ("train", "monitor", "test"):
        manifest = directory / f"{name}.jsonl"
        manifests[name] = sha256(manifest)
        rows = load_pages(manifest)
        counts[name] = len(rows)
        contexts[name] = {row["context_id"] for row in rows}
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        for row in rows:
            image = Path(row["image"])
            images[str(image.relative_to(directory.absolute()))] = sha256(image)
    for first, second in (("train", "monitor"), ("train", "test"), ("monitor", "test")):
        if not contexts[first].isdisjoint(contexts[second]):
            raise ValueError(f"Patient context overlap: {first}, {second}")
    return dict(manifests=manifests, images=images, pages=counts,
                contexts={key: len(value) for key, value in contexts.items()})


def validate_dataset(directory: Path) -> dict:
    saved = json.loads((directory / "artifact.json").read_text())
    if saved.get("kind") != "parse/dataset" or saved.get("schema_version") != 1:
        raise ValueError("Run the Parse data stage to create a dataset artifact")
    actual = dataset_identity(directory)
    if actual != saved["identity"]:
        raise ValueError("Dataset contents changed since generation; use the original artifact")
    return actual


class Progress:
    """Preserve the notebook's JSONL metric records without its display dependency."""

    def __init__(self, output: Path, stage: str):
        self.output, self.stage = output, stage
        self.started = time.monotonic()
        self.total = 1

    def configure(self, total, per_epoch):
        self.total = int(total)
        write_json(self.output / "schedule.json", dict(total=int(total), per_epoch=int(per_epoch)))

    def update(self, step, phase, metrics, seconds=None):
        event = dict(step=int(step), phase=phase, metrics=metrics,
                     seconds=time.monotonic() - self.started if seconds is None else seconds)
        with (self.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if phase != "train" or step % 32 == 0 or step == self.total:
            print(f"{self.stage}: {phase} {step}/{self.total} {json.dumps(metrics)}", flush=True)


def summarize(answers: list[dict]) -> dict:
    from nemotron.recipes.parse.dental.evaluate import pass_at_k

    counts = [len(group["answers"]) for group in answers]
    if not counts or min(counts) == 0 or len(set(counts)) != 1:
        raise ValueError("Evaluation requires equal, nonzero sample counts per page")
    flattened = [answer for group in answers for answer in group["answers"]]
    keys = ("text_cer", "chart_exact", "patient_fields", "history_rows", "ended")
    return dict(pages=len(answers), answers=len(flattened), samples=counts[0],
                correct=sum(answer["correct"] for answer in flattened),
                pass_at_k={str(k): pass_at_k(answers, k) for k in (1, 2, 4, 8) if k <= counts[0]},
                metrics={key: sum(a[key] for a in flattened) / len(flattened) for key in keys})
