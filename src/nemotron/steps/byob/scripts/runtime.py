"""Thin BYOB runtime dispatcher.

Benchmark-specific behavior belongs in `nemotron.steps.byob.runtime.benchmark_families`.
This module only selects the family and requested stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from nemotron.steps.byob.runtime.benchmark_families.registry import get_family, list_families

StageName = Literal["prepare", "generate", "translate"]


def list_family_names() -> tuple[str, ...]:
    """Return the registered benchmark families."""
    return tuple(list_families())


def run_byob(
    *,
    config: str | Path,
    stage: StageName,
    family: str = "mcq",
    skip_until: str | None = None,
) -> Path | None:
    """Run one BYOB stage for a benchmark family."""
    spec = get_family(family)
    config_path = Path(config)

    if stage == "prepare":
        return spec.prepare_data(config_path)
    if stage == "generate":
        return spec.generate(config_path, skip_until=skip_until)
    if stage == "translate":
        if spec.translate is None:
            raise ValueError(f"Benchmark family {family!r} does not define translation")
        return spec.translate(config_path, skip_until=skip_until)

    raise ValueError(f"Unknown BYOB stage {stage!r}")
