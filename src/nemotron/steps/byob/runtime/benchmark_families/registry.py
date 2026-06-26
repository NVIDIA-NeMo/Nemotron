"""Registry for BYOB benchmark families."""

from __future__ import annotations

from .base import BenchmarkFamilySpec
from .mcq.family import SPEC as MCQ_SPEC

_REGISTRY: dict[str, BenchmarkFamilySpec] = {
    MCQ_SPEC.name: MCQ_SPEC,
}


def get_family(name: str) -> BenchmarkFamilySpec:
    """Return a registered benchmark family by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown BYOB benchmark family {name!r}. Available: {sorted(_REGISTRY)}") from exc


def list_families() -> list[str]:
    """Return registered benchmark family names."""
    return sorted(_REGISTRY)
