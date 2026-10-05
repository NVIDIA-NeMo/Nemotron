# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Filters that are wrong for a language are refused, not merely documented.

The signal tables explained which measurements break on which scripts, and that
was all: `registry.resolve` still auto-selected every whitespace- and ASCII-based
signal for a Japanese pack with no warning, a policy could still name one, and
`quality_filters.min_words` shipped unconditionally.

Three capabilities close it. Each is an assertion about the writing system, has
no backing file, and must be declared by a human who knows the language:

    word_segmentation   words are delimited by whitespace
    ascii_digits        numbers are written with [0-9]
    ascii_punctuation   sentences are punctuated with . ! ?

There are two doors into these measurements and both are shut. The signal
registry covers profile and policy; `quality_filters` bypasses the registry
entirely and reaches Curator's WordCountFilter directly, so it is checked
against the pack separately -- in the step, and in the flow's preflight.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from nemotron.steps.curate.nemo_curator.runtime import langpack
from nemotron.steps.curate.nemo_curator.runtime import registry as r

PACKS = Path(langpack.__file__).parent.parent / "data" / "langpacks"

#: Every signal whose measurement assumes a writing system, and what it assumes.
GATED = {
    "word_count": "word_segmentation",
    "mean_word_length": "word_segmentation",
    "max_word_length": "word_segmentation",
    "symbol_to_word": "word_segmentation",
    "words_with_alphabets": "word_segmentation",
    "repeating_duplicate_ngrams": "word_segmentation",
    "numbers_ratio": "ascii_digits",
    "punctuation": "ascii_punctuation",
}


def _pack(tmp_path: Path, tag: str, supports: tuple[str, ...]) -> langpack.LanguagePack:
    d = tmp_path / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "charset.txt").write_text("あ\nい\n", encoding="utf-8")
    (d / "pack.toml").write_text(
        f'[pack]\npack_id = "{tag}"\nlanguage_tag = "{tag}"\nversion = "1"\nschema = 1\n\n'
        '[sources]\ncharset = { file = "charset.txt", origin = "test", license = "CC0-1.0" }\n\n'
        "[capabilities]\nsupports = [" + ", ".join(f'"{c}"' for c in supports) + "]\n",
        encoding="utf-8",
    )
    return langpack.load_pack(d)


# -- door one: the signal registry ---------------------------------------------


@pytest.mark.parametrize("signal,capability", sorted(GATED.items()))
def test_every_unsafe_signal_declares_the_assumption_it_makes(signal, capability) -> None:
    assert r.SIGNALS[signal].requires == (capability,)


@pytest.mark.parametrize("signal", sorted(GATED))
def test_auto_selection_skips_an_unsafe_signal_and_says_which(tmp_path, signal) -> None:
    """Skipped with a named reason, not silently: a report missing a signal for
    no stated cause reads as a signal that found nothing."""
    pack = _pack(tmp_path, "x-unsegmented", ("script_ratio",))

    chosen, warnings = r.resolve(None, pack.capabilities)

    assert signal not in {s.name for s in chosen}
    assert any(signal in w for w in warnings), warnings


@pytest.mark.parametrize("signal", sorted(GATED))
def test_naming_an_unsafe_signal_explicitly_is_refused(tmp_path, signal) -> None:
    """Auto-selection may skip; an explicit request must fail. Dropping a signal
    the config asked for would answer a question nobody asked."""
    pack = _pack(tmp_path, "x-unsegmented", ("script_ratio",))

    with pytest.raises(r.SignalRequirementsUnmetError, match=GATED[signal]):
        r.resolve([signal], pack.capabilities)


@pytest.mark.parametrize("signal", sorted(GATED))
def test_a_pack_that_declares_the_capability_keeps_the_signal(tmp_path, signal) -> None:
    """The contrast. Hindi uses spaces, so word-based signals are valid there."""
    pack = _pack(tmp_path, "x-declares", ("script_ratio", *sorted(set(GATED.values()))))

    assert signal in {s.name for s in r.resolve(None, pack.capabilities)[0]}


# -- the shipped packs make the distinction --------------------------------------


def test_hindi_keeps_word_signals_but_loses_the_ascii_ones() -> None:
    """The whole point, in one pack: Hindi IS space-delimited, so word_count is
    valid; its digits are Devanagari and its sentences end with the danda, so
    numbers_ratio and punctuation are not."""
    hi = langpack.load("hi", PACKS)

    assert hi.supports("word_segmentation")
    assert not hi.supports("ascii_digits")
    assert not hi.supports("ascii_punctuation")

    chosen = {s.name for s in r.resolve(None, hi.capabilities)[0]}
    assert "word_count" in chosen
    assert "numbers_ratio" not in chosen
    assert "punctuation" not in chosen


@pytest.mark.parametrize("tag", ["en", "vi"])
def test_latin_packs_keep_all_three(tag) -> None:
    pack = langpack.load(tag, PACKS)

    for capability in sorted(set(GATED.values())):
        assert pack.supports(capability), f"{tag} should declare {capability}"


# -- door two: quality_filters, which bypasses the registry -----------------------


@pytest.fixture
def step(monkeypatch):
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
        "huggingface_hub",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class Pipeline:
        def __init__(self, name=None):
            self.stages = []

        def add_stage(self, stage):
            self.stages.append(stage)

    sys.modules["nemo_curator.core.client"].RayClient = object
    sys.modules["nemo_curator.pipeline"].Pipeline = Pipeline
    sys.modules["nemo_curator.stages.text.io.reader"].JsonlReader = lambda **kw: object()
    sys.modules["nemo_curator.stages.text.io.writer"].JsonlWriter = lambda **kw: object()
    sys.modules["huggingface_hub"].snapshot_download = lambda **kw: None

    import importlib

    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    return module


def test_min_words_is_refused_for_a_language_without_word_segmentation(step, tmp_path) -> None:
    """quality_filters never touches the registry, so the capability gate above
    does not reach WordCountFilter. Same measurement, same hazard, other door."""
    _pack(tmp_path, "x-unsegmented", ("script_ratio",))

    with pytest.raises(ValueError, match="word_segmentation"):
        step._refuse_word_filter_without_segmentation(
            {
                "quality_filters": {"min_words": 50, "max_words": 5000},
                "heuristic_filters": {"language_tag": "x-unsegmented", "langpack_dir": str(tmp_path)},
            }
        )


def test_min_words_is_allowed_for_a_language_that_declares_it(step, tmp_path) -> None:
    """The guard itself, not what happens after it.

    This used to call build_pipeline and assert `pytest.raises(Exception)`, which
    passed only because the Curator stub broke further down. With Curator really
    installed the call succeeds and the assertion fails -- a test whose verdict
    depended on what was installed rather than on the behaviour under test.
    """
    _pack(tmp_path, "x-declares", ("script_ratio", "word_segmentation"))

    step._refuse_word_filter_without_segmentation(
        {
            "quality_filters": {"min_words": 50, "max_words": 5000},
            "heuristic_filters": {"language_tag": "x-declares", "langpack_dir": str(tmp_path)},
        }
    )  # returns; a raise here is the failure


def test_a_config_with_no_pack_keeps_its_historical_behaviour(step, tmp_path) -> None:
    """Additivity. Without a declared language there is nothing to be wrong about,
    so a config written before this work builds what it always built."""
    step._refuse_word_filter_without_segmentation({"quality_filters": {"min_words": 50, "max_words": 5000}})


# -- door three: the shipped default itself ---------------------------------------


def test_the_shipped_default_does_not_enable_a_word_based_filter() -> None:
    """The gap the capability gate did not close.

    The registry gate needs a pack to consult, and `config/default.yaml` declares
    no language -- so a Japanese corpus run with `-c default` still lost 39% of
    its documents to `min_words: 50`, silently, with every gate above in place.
    A default cannot know the language, so it must not choose.
    """
    import yaml

    from .._step_helpers import step_dir

    cfg = yaml.safe_load(
        (step_dir(__file__, "curate", "nemo_curator") / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    quality = cfg.get("quality_filters") or {}

    assert "min_words" not in quality, "the shipped default must not enable WordCountFilter"
    assert "max_words" not in quality
    assert quality.get("min_langid_score") == 0.3, "the language-confidence default is unrelated and stays"


def test_the_step_manifest_default_does_not_enable_it_either() -> None:
    """step.toml supplies the parameter default when a config omits the key, so
    leaving it there would put the filter back for anyone not using default.yaml."""
    import tomllib

    from .._step_helpers import step_dir

    with (step_dir(__file__, "curate", "nemo_curator") / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    default = next(p for p in manifest["parameters"] if p["name"] == "quality_filters")["default"]

    assert "min_words" not in default
    assert "max_words" not in default


# -- door four: applying an approved policy ---------------------------------------


def test_a_policy_naming_an_unsupported_signal_is_refused(step, tmp_path) -> None:
    """registry.resolve guards PROFILE. policy_stages guards the run that filters.

    Loading the pack only for PACK_REQUIREMENTS left the orthographic
    capabilities unchecked here: `ascii_digits` is not a data requirement, so no
    pack was ever loaded to ask, and a numbers_ratio policy built for Hindi.
    """
    with pytest.raises(ValueError, match="does not support"):
        step.policy_stages(
            [{"signal": "numbers_ratio", "max": 0.25}],
            "text",
            "filter",
            {"language_tag": "hi", "langpack_dir": str(PACKS)},
        )


def test_an_orthographic_capability_does_not_demand_a_content_hash(step) -> None:
    """It is not pack DATA, so demanding the hash would refuse a policy for a
    reason that does not apply to it.

    Asserted through load_policy_pack rather than by building stages: constructing
    a Curator filter needs Curator, and a test that passes only where Curator is
    absent measures the environment.
    """
    pack = step.load_policy_pack(
        {"language_tag": "hi", "langpack_dir": str(PACKS)}, ["word_count"], require_hash=False
    )

    assert pack.language_tag == "hi"

    with pytest.raises(ValueError, match="content_hash"):
        step.load_policy_pack({"language_tag": "hi", "langpack_dir": str(PACKS)}, ["stopword_ratio"])


def test_the_capabilities_are_documented_in_the_pack_spec() -> None:
    """A capability an author cannot read about is one they will not declare."""
    spec = PACKS / "SPEC.md"
    text = spec.read_text(encoding="utf-8")

    for capability in ("word_segmentation", "ascii_digits", "ascii_punctuation"):
        assert capability in text, f"SPEC.md does not describe {capability}"
