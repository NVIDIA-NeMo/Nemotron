"""Interfaces for BYOB benchmark-family implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkFamilySpec:
    """Named hooks that let agents add new benchmark families without rewriting orchestration."""

    name: str
    description: str
    prepare_data: Callable
    generate: Callable
    translate: Callable | None = None
