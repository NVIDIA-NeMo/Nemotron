"""Task hooks consumed by the reusable SFT and RLVR stages.

Configure an importable zero-argument factory. SFT uses only ``monitor``;
RLVR uses candidate preparation/selection, evaluation and a reward environment.
"""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

Monitor = Callable[[Any, Any, list[dict]], tuple[list[dict], dict[str, float]]]


class SFTTask(Protocol):
    def monitor(self, model: Any, processor: Any, rows: list[dict]) -> tuple[list[dict], dict[str, float]]:
        """Return per-page predictions and aggregate task metrics."""
        ...


class RLVRTask(Protocol):
    prompt_file: Path
    environment_class: str
    reward_metric: str | None

    def create_candidates(self, output: Path, *, seed: int, contexts: int) -> list[dict]:
        """Write candidate images below output; return rows with absolute image paths."""
        ...

    def select_candidates(
        self, candidates: list[dict], answers: list[dict], *, pages: int, seed: int, excluded: list[dict]
    ) -> list[dict]:
        """Select candidates using task scores, excluding all fixed dataset contexts."""
        ...

    def evaluate(
        self, model: Path, rows: list[dict], output: Path, *, samples: int, processor_source: Path
    ) -> list[dict]:
        """Return ordered groups with page IDs and scored answers; persist predictions."""
        ...

    def summarize(self, answers: list[dict]) -> dict:
        """Aggregate this task's correctness and component measurements."""
        ...


def load_task(factory: str, *, stage: str) -> SFTTask | RLVRTask:
    """Instantiate task code from a fully qualified factory and check its hooks."""
    if stage not in ("sft", "rlvr"):
        raise ValueError(f"Unsupported task stage: {stage}")
    module, separator, name = factory.rpartition(".")
    if not separator or not module or not name:
        raise ValueError("task must be a fully qualified Python factory, e.g. my_domain.task.DocumentTask")
    task = getattr(import_module(module), name)()
    methods = (
        ("monitor",)
        if stage == "sft"
        else (
            "create_candidates",
            "select_candidates",
            "evaluate",
            "summarize",
        )
    )
    for method in methods:
        if not callable(getattr(task, method, None)):
            raise TypeError(f"{factory} must implement {method}() for {stage}")
    if stage == "rlvr":
        if not isinstance(getattr(task, "environment_class", None), str) or not task.environment_class:
            raise TypeError(f"{factory} must provide an importable reward environment_class")
        if not isinstance(getattr(task, "prompt_file", None), (str, Path)) or not Path(task.prompt_file).is_file():
            raise ValueError(f"{factory} must provide an existing prompt_file")
        if (
            not hasattr(task, "reward_metric")
            or task.reward_metric is not None
            and not isinstance(task.reward_metric, str)
        ):
            raise TypeError(f"{factory}.reward_metric must be a metric name or None")
    return task
