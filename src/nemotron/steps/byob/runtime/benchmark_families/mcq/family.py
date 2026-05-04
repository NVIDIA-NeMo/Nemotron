"""MCQ benchmark-family registration."""

from __future__ import annotations

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkFamilySpec
from nemotron.steps.byob.runtime.benchmark_families.mcq.pipeline import (
    generate_mcq,
    prepare_mcq_data,
    translate_mcq,
)

SPEC = BenchmarkFamilySpec(
    name="mcq",
    description="MMLU-Pro-style multiple-choice benchmark generation and translation.",
    prepare_data=prepare_mcq_data,
    generate=generate_mcq,
    translate=translate_mcq,
)
