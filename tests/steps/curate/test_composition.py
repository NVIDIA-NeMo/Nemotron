# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""How the filter stages combine, pinned against the README that describes them.

The README used to say `mode` "decides what reaches the writer" and that
`annotate` keeps "all". Both are false for any config that also sets
`language_codes`, `quality_filters` or `domains`: those gates are built
independently of `mode` and drop rows under every one of them. A user reading
that table and choosing `annotate` for a non-destructive scoring pass got a
smaller corpus and no warning.

These tests pin the four properties the README now states, so the prose and the
pipeline cannot drift apart again.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

README = Path(__file__).parents[3] / "src/nemotron/steps/curate/nemo_curator/README.md"


class _Stage:
    """Records what a stage was constructed as, so the pipeline can be inspected."""

    def __init__(self, kind: str, **kwargs):
        self.kind = kind
        self.kwargs = kwargs
        self.name = kind


@pytest.fixture
def step(monkeypatch):
    """Import the step against a Curator stub that records stage construction."""
    for name in (
        "nemo_curator",
        "nemo_curator.core",
        "nemo_curator.core.client",
        "nemo_curator.pipeline",
        "nemo_curator.stages",
        "nemo_curator.stages.text",
        "nemo_curator.stages.text.io",
        "nemo_curator.stages.text.io.reader",
        "nemo_curator.stages.text.io.writer",
        "nemo_curator.stages.text.filters",
        "nemo_curator.stages.text.filters.fasttext",
        "nemo_curator.stages.text.filters.heuristic",
        "nemo_curator.stages.text.classifiers",
        "nemo_curator.stages.text.modules",
        "nemo_curator.stages.text.modules.score_filter",
        "huggingface_hub",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class Pipeline:
        def __init__(self, name=None):
            self.name = name
            self.stages = []

        def add_stage(self, stage):
            self.stages.append(stage)

    sys.modules["nemo_curator.core.client"].RayClient = object
    sys.modules["nemo_curator.pipeline"].Pipeline = Pipeline
    sys.modules["nemo_curator.stages.text.io.reader"].JsonlReader = lambda **kw: _Stage("reader", **kw)
    sys.modules["nemo_curator.stages.text.io.writer"].JsonlWriter = lambda **kw: _Stage("writer", **kw)
    sys.modules["huggingface_hub"].snapshot_download = lambda **kw: None

    # text_filter_stages() imports from `...text.modules`, falling back to
    # `...text.filters`; score_stage() imports Score from the same place. Stub
    # the primary location so the fallback path is not what is exercised here.
    for target in ("nemo_curator.stages.text.modules", "nemo_curator.stages.text.modules.score_filter"):
        mods = sys.modules[target]
        mods.Filter = lambda **kw: _Stage("Filter", **kw)
        mods.ScoreFilter = lambda *a, **kw: _Stage("ScoreFilter", args=a, **kw)
        mods.Score = lambda *a, **kw: _Stage("Score", args=a, **kw)
    sys.modules["nemo_curator.stages.text.filters"].FastTextLangId = object
    sys.modules["nemo_curator.stages.text.filters.fasttext"].FastTextLangId = lambda **kw: _Stage("langid", **kw)
    sys.modules["nemo_curator.stages.text.filters.heuristic"].WordCountFilter = lambda **kw: _Stage("wordcount", **kw)
    sys.modules["nemo_curator.stages.text.classifiers"].MultilingualDomainClassifier = lambda **kw: _Stage(
        "domain", **kw
    )

    import importlib

    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    return module


def _kinds(pipeline) -> list[str]:
    return [s.kind for s in pipeline.stages]


@pytest.mark.parametrize("mode", ["filter", "annotate", "both"])
def test_annotate_does_not_stop_the_language_gate_dropping_rows(step, mode) -> None:
    """THE ONE THAT MATTERS: `mode` is not a non-destructive switch.

    A user setting `mode: annotate` to score without removing anything still
    loses every document the language gate rejects.
    """
    pipeline, _ = step.build_pipeline(
        {
            "input_glob": "x/*.jsonl",
            "output_dir": "out",
            "text_field": "text",
            "mode": mode,
            "language_codes": ["VI"],
            "models": {"fasttext_langid": "lid.176.bin"},
        },
        input_files=["x/a.jsonl"],
    )

    assert "Filter" in _kinds(pipeline), f"the language gate vanished under mode={mode!r}"


@pytest.mark.parametrize("mode", ["filter", "annotate", "both"])
def test_annotate_does_not_stop_the_word_count_gate_dropping_rows(step, mode) -> None:
    pipeline, _ = step.build_pipeline(
        {
            "input_glob": "x/*.jsonl",
            "output_dir": "out",
            "text_field": "text",
            "mode": mode,
            "quality_filters": {"min_words": 50, "max_words": 5000},
        },
        input_files=["x/a.jsonl"],
    )

    assert "ScoreFilter" in _kinds(pipeline), f"the word-count gate vanished under mode={mode!r}"


def test_annotate_domains_labels_without_dropping(step) -> None:
    """The one knob that genuinely annotates: filter_by stays None."""
    pipeline, _ = step.build_pipeline(
        {
            "input_glob": "x/*.jsonl",
            "output_dir": "out",
            "text_field": "text",
            "annotate_domains": True,
        },
        input_files=["x/a.jsonl"],
    )

    domain = next(s for s in pipeline.stages if s.kind == "domain")
    assert domain.kwargs["filter_by"] is None, "annotate_domains must not gate on a domain list"


def test_naming_domains_turns_the_classifier_into_a_gate(step) -> None:
    pipeline, _ = step.build_pipeline(
        {
            "input_glob": "x/*.jsonl",
            "output_dir": "out",
            "text_field": "text",
            "domains": ["news"],
        },
        input_files=["x/a.jsonl"],
    )

    domain = next(s for s in pipeline.stages if s.kind == "domain")
    assert domain.kwargs["filter_by"] == ["news"]


def test_a_list_of_language_codes_is_or_not_and(step) -> None:
    """Set membership, so any listed code keeps the document."""
    assert step.keep_language("[0.9, 'vi']", {"VI", "EN"})
    assert step.keep_language("[0.9, 'en']", {"VI", "EN"})
    assert not step.keep_language("[0.9, 'ja']", {"VI", "EN"})
    # A script suffix matches on the part before the underscore.
    assert step.keep_language("[0.9, 'zh_Hans']", {"ZH"})


def test_nothing_counts_how_many_gates_a_document_passed(step) -> None:
    """No voting, no N-of-M: there is no such counter to find."""
    source = Path(step.__file__).read_text(encoding="utf-8")

    for word in ("majority", "n_of_m", "vote", "quorum"):
        assert word not in source.lower(), f"{word!r} suggests a composition rule the docs do not describe"


def test_the_readme_does_not_claim_annotate_keeps_everything() -> None:
    """The prose this module exists to defend.

    The old table said `annotate | all kept`, unqualified, on a page that never
    mentioned the independent gates. Anything equivalent reintroduces the defect.
    """
    text = README.read_text(encoding="utf-8")

    assert "## How The Stages Combine" in text, "the composition section is the fix; it must stay"
    assert "| `annotate` | all kept |" not in text, "the unqualified claim is back"
    assert "combine with AND" in text
    assert "no voting" in text.lower()
