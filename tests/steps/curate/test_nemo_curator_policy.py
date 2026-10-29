# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""F2: approved-policy filtering and the three run modes.

The mode table is a claim about rows and columns, so it is tested by running
data through stages rather than by asserting which class was constructed. The
stubs below transcribe Curator's own ``score_filter.py`` — in particular that
``ScoreFilter`` writes ``score_field`` *before* applying ``keep_document``,
which is why ``both`` is one stage and not ``Score`` followed by ``Filter``.
``test_stub_signatures_match_curator`` pins the transcription wherever Curator
is actually installed.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from nemotron.steps.curate.nemo_curator.runtime import langpack
from nemotron.steps.curate.nemo_curator.runtime import policy as policy_module
from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

# Three clean documents and two that are almost entirely emoji. Scored by
# ``unicode_alpha_numeric`` at the 0.25 default, the split is 3 kept / 2 dropped.
DOCS = [
    {"id": "a", "text": "Curation needs measurements, not defaults inherited from a sample config."},
    {"id": "b", "text": "Tieng Viet la ngon ngu chinh thuc cua Viet Nam."},
    {"id": "c", "text": "A third ordinary sentence, long enough to score like prose."},
    {"id": "d", "text": "🙂🙂🙂🙂🙂🙂🙂🙂"},
    {"id": "e", "text": "★★★ ☆☆☆ ★★★ ☆☆☆"},
]
CLEAN = 3

SIGNAL = "unicode_alpha_numeric"
SCORE_COLUMN = f"__{SIGNAL}"
BASE_COLUMNS = {"id", "text"}
POLICY_CORPUS_FINGERPRINT = "sha256:" + "d" * 64
PROFILE_DIGEST = "sha256:" + "e" * 64
LANGPACK_FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"


def vietnamese_fixture() -> langpack.LanguagePack:
    return langpack.load("x-test-vi", LANGPACK_FIXTURES)


# -- executable stubs ---------------------------------------------------------


@dataclass
class _Score:
    score_fn: Any
    score_field: str
    text_field: str = "text"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.score_field] = df[self.text_field].apply(self.score_fn.score_document)
        return df


@dataclass
class _Filter:
    filter_fn: Any
    filter_field: str
    invert: bool = False

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df[self.filter_field].apply(self.filter_fn)
        return df[~mask if self.invert else mask]


@dataclass
class _ScoreFilter:
    filter_obj: Any
    text_field: str = "text"
    score_field: str | None = None
    invert: bool = False

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        scores = df[self.text_field].apply(self.filter_obj.score_document)
        if self.score_field is not None:
            df[self.score_field] = scores
        mask = scores.apply(self.filter_obj.keep_document)
        return df[~mask if self.invert else mask]


@pytest.fixture
def step(monkeypatch):
    """``step`` with Curator's filter modules replaced by executable stubs."""
    filters = types.ModuleType("nemo_curator.stages.text.filters")
    filters.Score, filters.Filter, filters.ScoreFilter = _Score, _Filter, _ScoreFilter

    mods = {
        "nemo_curator": {},
        "nemo_curator.core": {},
        "nemo_curator.core.client": {"RayClient": object},
        "nemo_curator.pipeline": {"Pipeline": object},
        "nemo_curator.stages": {},
        "nemo_curator.stages.text": {},
        "nemo_curator.stages.text.io": {},
        "nemo_curator.stages.text.io.reader": {"JsonlReader": object},
        "nemo_curator.stages.text.io.writer": {"JsonlWriter": object},
        # Present but without the filter classes, exercising the release shim in
        # text_filter_stages()/score_stage() the same way Curator 1.3 does.
        "nemo_curator.stages.text.modules": {"AddId": object},
        "huggingface_hub": {"snapshot_download": lambda **kwargs: None},
    }
    for name, attrs in mods.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "nemo_curator.stages.text.filters", filters)

    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    return module


def run(stages) -> pd.DataFrame:
    df = pd.DataFrame(DOCS)
    for stage in stages:
        df = stage.process(df)
    return df


def approved_policy(**overrides) -> dict:
    document = {
        "schema_version": policy_module.SCHEMA_VERSION,
        "approved": True,
        "corpus": {"fingerprint": POLICY_CORPUS_FINGERPRINT},
        "signals_impl_version": signal_registry.IMPL_VERSION,
        "profile_digest": PROFILE_DIGEST,
        "approval": {
            "method": "manual",
            "approver": "reviewer@example.test",
            "date": "2026-08-25",
            "evidence": "retention curve reviewed against a 200-doc sample",
        },
        "thresholds": [{"signal": SIGNAL, "max": 0.25}],
    }
    document.update(overrides)
    return document


def write_policy(tmp_path, document, name="approved_policy.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# -- the mode table -----------------------------------------------------------


def test_filter_mode_drops_rows_and_adds_no_column(step) -> None:
    """The historical shape: gates reject rows and the score is discarded."""
    out = run(step.policy_stages(approved_policy()["thresholds"], "text", "filter"))

    assert len(out) == CLEAN
    assert set(out.columns) == BASE_COLUMNS


def test_annotate_mode_keeps_every_row_and_adds_the_column(step) -> None:
    out = run(step.policy_stages(approved_policy()["thresholds"], "text", "annotate"))

    assert len(out) == len(DOCS), "annotate must not drop anything"
    assert set(out.columns) == BASE_COLUMNS | {SCORE_COLUMN}


def test_both_mode_drops_rows_and_survivors_keep_the_column(step) -> None:
    out = run(step.policy_stages(approved_policy()["thresholds"], "text", "both"))

    assert len(out) == CLEAN
    assert set(out.columns) == BASE_COLUMNS | {SCORE_COLUMN}
    assert out[SCORE_COLUMN].notna().all()


def test_both_is_one_stage_not_score_then_filter(step) -> None:
    """ScoreFilter writes the score before gating, so a second pass is waste."""
    thresholds = approved_policy()["thresholds"]

    assert len(step.policy_stages(thresholds, "text", "both")) == len(thresholds)


def test_annotate_and_filter_agree_on_which_rows_survive(step) -> None:
    """The modes must differ in what they emit, never in what they judge."""
    annotated = run(step.policy_stages(approved_policy()["thresholds"], "text", "annotate"))
    filtered = run(step.policy_stages(approved_policy()["thresholds"], "text", "filter"))

    kept = annotated[annotated[SCORE_COLUMN] <= 0.25]

    assert sorted(kept["id"]) == sorted(filtered["id"])


def test_the_text_field_is_honoured(step) -> None:
    stages = step.policy_stages(approved_policy()["thresholds"], "body", "annotate")

    assert stages[0].text_field == "body"


def test_an_unknown_mode_is_refused(step, tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"mode": "sieve", "input_glob": "x", "text_field": "text", "output_dir": "o"}))
    monkeypatch.setattr(sys, "argv", ["step.py", "--config", str(cfg)])

    with pytest.raises(ValueError, match="mode must be one of"):
        step.main()


# -- approval gating ----------------------------------------------------------


def test_a_candidate_policy_is_refused(step, tmp_path) -> None:
    """curate/profile emits approved: false on purpose; this is the other half."""
    path = write_policy(tmp_path, approved_policy(approved=False), "candidate_policies.yaml")

    with pytest.raises(policy_module.PolicyNotApprovedError) as excinfo:
        step.resolve_policy({"heuristic_filters": {"approved_policy": str(path)}})

    message = str(excinfo.value)
    assert str(path) in message, "the error must name which policy was refused"
    assert "approved must be true" in message, "the error must name the unmet field"


def test_a_policy_with_no_approval_block_is_refused(step, tmp_path) -> None:
    """A policy can be hand-written, so the filter checks it too.

    The block used to be optional here while three shipped documents told users
    it was required. Enforcing it is the half of that contradiction the reviewer
    asked for; allow_unvalidated_policy remains the way through, and records the
    override rather than laundering it as an approval.
    """
    document = approved_policy()
    del document["approval"]
    path = write_policy(tmp_path, document)

    with pytest.raises(ValueError, match="approver"):
        step.resolve_policy({"heuristic_filters": {"approved_policy": str(path)}})


def test_the_override_carries_a_policy_with_no_approval_block(step, tmp_path) -> None:
    """Refusing without an escape would make an unsigned policy unrunnable rather
    than honestly labelled."""
    document = approved_policy()
    del document["approval"]
    path = write_policy(tmp_path, document)

    thresholds, _, _ = step.resolve_policy(
        {"heuristic_filters": {"approved_policy": str(path), "allow_unvalidated_policy": True}}
    )

    assert thresholds, "the override proceeds; the manifest records it as override_unvalidated"


def test_the_override_proceeds_and_warns_naming_the_policy(step, tmp_path) -> None:
    path = write_policy(tmp_path, approved_policy(approved=False))

    thresholds, _pack, warnings = step.resolve_policy(
        {"heuristic_filters": {"approved_policy": str(path), "allow_unvalidated_policy": True}}
    )

    assert thresholds, "the override must actually proceed"
    assert warnings, "an override that logs nothing is indistinguishable from approval"
    assert all(str(path) in w for w in warnings)
    assert any("allow_unvalidated_policy" in w for w in warnings)


def test_an_approved_policy_warns_about_nothing(step, tmp_path) -> None:
    path = write_policy(tmp_path, approved_policy())

    thresholds, _pack, warnings = step.resolve_policy({"heuristic_filters": {"approved_policy": str(path)}})

    assert warnings == []
    assert thresholds == [{"signal": SIGNAL, "max": 0.25}]


def test_policy_is_bound_to_the_configured_input_corpus(step, tmp_path) -> None:
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"id":"1","text":"first corpus"}\n', encoding="utf-8")
    second.write_text('{"id":"1","text":"different corpus"}\n', encoding="utf-8")
    document = approved_policy(corpus={"fingerprint": integrity.corpus_fingerprint(str(first), "text", "id")})
    path = write_policy(tmp_path, document)

    with pytest.raises(ValueError, match="configured input fingerprints"):
        step.resolve_policy(
            {
                "input_glob": str(second),
                "text_field": "text",
                "id_field": "id",
                "heuristic_filters": {"approved_policy": str(path)},
            }
        )


def test_resolved_policy_contents_participate_in_manifest_identity(step, tmp_path) -> None:
    from nemotron.steps.curate.nemo_curator.runtime import integrity, manifest

    corpus = tmp_path / "input.jsonl"
    corpus.write_text('{"id":"1","text":"same corpus"}\n', encoding="utf-8")
    fingerprint = integrity.corpus_fingerprint(str(corpus), "text", "id")
    path = write_policy(tmp_path, approved_policy(corpus={"fingerprint": fingerprint}))
    cfg = {
        "input_glob": str(corpus),
        "text_field": "text",
        "id_field": "id",
        "heuristic_filters": {"approved_policy": str(path)},
    }
    first = step._resolve_policy(cfg)
    first_hash = manifest.config_hash(step.manifest_config(cfg, first))

    changed = approved_policy(
        corpus={"fingerprint": fingerprint},
        thresholds=[{"signal": SIGNAL, "max": 0.20}],
    )
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    second = step._resolve_policy(cfg)
    second_hash = manifest.config_hash(step.manifest_config(cfg, second))

    assert first.identity["policy_digest"] != second.identity["policy_digest"]
    assert first_hash != second_hash


def test_a_policy_from_another_langpack_is_refused(step, tmp_path) -> None:
    """Thresholds are calibrated against a pack's word lists and character set."""
    path = write_policy(tmp_path, approved_policy(langpack={"tag": "vi", "content_hash": "sha256:aaa"}))

    with pytest.raises(ValueError, match="sha256:aaa"):
        step.resolve_policy(
            {
                "heuristic_filters": {
                    "approved_policy": str(path),
                    "langpack_content_hash": "sha256:bbb",
                }
            }
        )


def test_a_matching_langpack_hash_is_accepted(step, tmp_path) -> None:
    path = write_policy(tmp_path, approved_policy(langpack={"tag": "vi", "content_hash": "sha256:aaa"}))

    thresholds, _pack, _warnings = step.resolve_policy(
        {
            "heuristic_filters": {
                "approved_policy": str(path),
                "langpack_content_hash": "sha256:aaa",
            }
        }
    )

    assert thresholds


def test_an_unknown_signal_is_refused_with_the_allowed_names(step) -> None:
    with pytest.raises(ValueError) as excinfo:
        step.policy_stages([{"signal": "os.system", "max": 1}], "text", "filter")

    message = str(excinfo.value)
    assert "os.system" in message
    assert SIGNAL in message, "the error must list what is allowed"


# -- the neutrality guarantee -------------------------------------------------


def test_no_policy_configured_adds_no_stages(step) -> None:
    """A config that predates F2 must build exactly the pipeline it built before."""
    for cfg in ({}, {"heuristic_filters": None}, {"heuristic_filters": {}}):
        assert step.resolve_policy(cfg) == ([], {}, [])

    for mode in step.MODES:
        assert step.policy_stages([], "text", mode) == []


def test_no_policy_resolves_no_filter_classes(step, monkeypatch) -> None:
    """The empty path must not reach an import.

    Curator moved Score/Filter/ScoreFilter between releases, so resolving them
    runs a fallback chain. A run with no policy configured should not depend on
    that chain resolving at all.
    """

    def boom():
        raise AssertionError("the empty path resolved a filter class")

    monkeypatch.setattr(step, "text_filter_stages", boom)
    monkeypatch.setattr(step, "score_stage", boom)

    for mode in step.MODES:
        assert step.policy_stages([], "text", mode) == []


# -- the transcription these tests rest on ------------------------------------


def test_stub_signatures_match_curator() -> None:
    """Wherever Curator is installed, prove the stubs model the real classes."""
    real = pytest.importorskip("nemo_curator.stages.text.filters")

    for stub, name in ((_Score, "Score"), (_Filter, "Filter"), (_ScoreFilter, "ScoreFilter")):
        actual = inspect.signature(getattr(real, name)).parameters
        for param, spec in inspect.signature(stub).parameters.items():
            assert param in actual, f"{name} has no parameter {param}"
            if spec.default is not inspect.Parameter.empty:
                assert actual[param].default == spec.default, f"{name}.{param} default drifted"

    source = inspect.getsource(real.ScoreFilter.compute_filter_mask)
    assert "df[score_field] = scores" in source, (
        "ScoreFilter no longer annotates before filtering; 'both' mode is no longer one stage"
    )


# -- language-pack signals ----------------------------------------------------
#
# Half the registry is parameterised by a language pack, and curate/profile emits
# candidates for exactly those. A policy that could be produced but not executed
# would be a dead end between the two steps, so these cover the handover.

PACK_SIGNAL = "stopword_ratio"


def pack_policy(**overrides):
    pack = vietnamese_fixture()

    document = approved_policy(
        langpack={
            "language_tag": pack.language_tag,
            "content_hash": pack.content_hash,
            "langpack_dir": str(LANGPACK_FIXTURES),
        },
        thresholds=[{"signal": PACK_SIGNAL, "min": 0.05}],
    )
    document.update(overrides)
    return document


def test_a_policy_naming_a_pack_signal_can_actually_be_executed(step) -> None:
    """The handover curate/profile -> curate/nemo_curator, end to end."""
    document = pack_policy()

    stages = step.policy_stages(document["thresholds"], "text", "annotate", document["langpack"])

    assert len(stages) == 1
    assert stages[0].score_fn.score_document("của và những là một trong") > 0


def test_every_pack_backed_signal_in_the_registry_can_be_built(step) -> None:
    """A signal profile can propose but the filter cannot construct is unusable."""
    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    pack = vietnamese_fixture()
    spec = {
        "language_tag": pack.language_tag,
        "content_hash": pack.content_hash,
        "langpack_dir": str(LANGPACK_FIXTURES),
    }
    supported = [n for n in sorted(signal_registry.PACK_SIGNALS) if n in signal_registry.SIGNALS]
    buildable = [n for n in supported if set(signal_registry.SIGNALS[n].requires) <= set(pack.capabilities)]

    assert buildable, "the vi pack must back at least one registry signal"
    for name in buildable:
        # The bound key must match the signal's direction. Feeding every signal
        # "max" is what hid the gate-inversion defect: a max bound on a
        # min-direction signal used to build a filter that gated the wrong way.
        signal = signal_registry.SIGNALS[name]
        entry = {"signal": name}
        for key in step.BOUND_KEYS[signal.direction]:
            entry[key] = 0.5 if key == "max" else 0.1
        assert step.policy_stages([entry], "text", "filter", spec), name


def test_a_pack_signal_without_a_declared_language_is_refused(step) -> None:
    """There is no default language: a wrong one produces plausible wrong numbers."""
    with pytest.raises(ValueError, match="no langpack.language_tag"):
        step.policy_stages([{"signal": PACK_SIGNAL, "min": 0.05}], "text", "filter", {})


def test_a_pack_signal_without_an_explicit_pack_directory_is_refused(step) -> None:
    pack = vietnamese_fixture()
    spec = {"language_tag": pack.language_tag, "content_hash": pack.content_hash}

    with pytest.raises(langpack.LanguagePackNotFoundError, match="explicit langpack_dir"):
        step.policy_stages([{"signal": PACK_SIGNAL, "min": 0.05}], "text", "filter", spec)


def test_a_pack_that_hashes_differently_than_declared_is_refused(step) -> None:
    """Checked against the pack actually loaded, not another string in the config."""
    spec = {
        "language_tag": "x-test-vi",
        "content_hash": "sha256:notthepackonthismachine",
        "langpack_dir": str(LANGPACK_FIXTURES),
    }

    with pytest.raises(ValueError, match="hashes to"):
        step.policy_stages([{"signal": PACK_SIGNAL, "min": 0.05}], "text", "filter", spec)


def test_a_pack_free_policy_never_loads_a_pack(step, monkeypatch) -> None:
    """Otherwise a machine with no packs installed could not run a policy of
    Curator-only signals, which have nothing to do with language packs."""

    def boom(*args, **kwargs):
        raise AssertionError("a pack-free policy loaded a language pack")

    monkeypatch.setattr(step, "load_policy_pack", boom)

    assert step.policy_stages(approved_policy()["thresholds"], "text", "filter")


def test_resolve_policy_hands_the_pack_spec_to_the_caller(step, tmp_path) -> None:
    path = write_policy(tmp_path, pack_policy())

    thresholds, spec, _ = step.resolve_policy({"heuristic_filters": {"approved_policy": str(path)}})

    assert thresholds[0]["signal"] == PACK_SIGNAL
    assert spec["language_tag"] == "x-test-vi"


# -- bound direction ----------------------------------------------------------
#
# A policy's min/max zipped positionally onto a signal's threshold_params builds
# a working pipeline that gates the wrong way round. No test of the pipeline's
# shape can catch that, so it is pinned on behaviour here.


def test_a_max_bound_on_a_min_direction_signal_is_refused(step) -> None:
    """The silent gate inversion: 'drop above 0.9' becoming 'drop below 0.9'."""
    pack = vietnamese_fixture()
    spec = {
        "language_tag": pack.language_tag,
        "content_hash": pack.content_hash,
        "langpack_dir": str(LANGPACK_FIXTURES),
    }

    with pytest.raises(ValueError, match="invert the gate"):
        step.policy_stages([{"signal": "stopword_ratio", "max": 0.9}], "text", "filter", spec)


def test_a_min_bound_on_a_max_direction_signal_is_refused(step) -> None:
    with pytest.raises(ValueError, match="invert the gate"):
        step.policy_stages([{"signal": SIGNAL, "min": 0.9}], "text", "filter")


def test_the_correct_bound_still_builds_the_gate_it_says(step) -> None:
    """The other half: the accepted spelling must mean what it reads as."""
    stages = step.policy_stages([{"signal": SIGNAL, "max": 0.25}], "text", "filter")
    document_filter = stages[0].filter_obj

    assert document_filter.keep_document(0.10) is True
    assert document_filter.keep_document(0.90) is False


def test_an_interval_signal_needs_both_bounds(step) -> None:
    """One bound would fix the other at an unstated value.

    The pack is supplied because word_count now declares word_segmentation: a
    policy naming a word-based signal has to say which language it believes it is
    gating, or the capability cannot be checked at all.
    """
    with pytest.raises(ValueError, match="gates from both sides"):
        step.policy_stages(
            [{"signal": "word_count", "min": 50}],
            "text",
            "filter",
            {"language_tag": "x-test-vi", "langpack_dir": str(LANGPACK_FIXTURES)},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True])
def test_pipeline_refuses_non_finite_or_non_numeric_bounds(step, value) -> None:
    with pytest.raises(ValueError, match="finite number"):
        step.policy_stages(
            [{"signal": SIGNAL, "max": value}],
            "text",
            "filter",
        )


def test_pipeline_refuses_inverted_interval_bounds(step) -> None:
    with pytest.raises(ValueError, match="must not be greater"):
        step.policy_stages(
            [{"signal": "word_count", "min": 5000, "max": 50}],
            "text",
            "filter",
            {"language_tag": "x-test-vi", "langpack_dir": str(LANGPACK_FIXTURES)},
        )


def test_a_signal_with_no_bound_is_refused(step) -> None:
    """This once said "no bound AND no shipped default", because a signal that
    shipped one silently used it. Now the bound is required whether or not
    Curator has an opinion, so the test applies to every signal rather than to
    the nine that happened to declare an empty default."""
    with pytest.raises(ValueError, match="neither min nor max"):
        step.policy_stages([{"signal": "token_count"}], "text", "filter", {"tokenizer": "m"})


def test_every_registry_signal_has_a_known_direction() -> None:
    """BOUND_KEYS must cover the registry, or a signal fails only when used."""
    import importlib

    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    directions = {s.direction for s in signal_registry.SIGNALS.values()}
    assert directions <= set(module.BOUND_KEYS)


# -- requirements -------------------------------------------------------------


def test_a_signal_needing_a_tokenizer_says_so_instead_of_failing_on_a_kwarg(step) -> None:
    """token_count is proposed by profile whenever models.tokenizer is set."""
    with pytest.raises(ValueError, match="needs a tokenizer"):
        step.policy_stages([{"signal": "token_count", "min": 10, "max": 4096}], "text", "filter")


def test_every_requirement_the_registry_declares_is_one_this_step_can_supply() -> None:
    """The check that would have caught token_count before a user did.

    A hand-maintained set of pack-backed names has to be remembered; this derives
    the obligation from the registry itself, so adding a signal with a new
    requirement fails here rather than at someone's pipeline construction.
    """
    import importlib

    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    declared = {r for s in signal_registry.SIGNALS.values() for r in s.requires}

    assert declared <= module.KNOWN_REQUIREMENTS, (
        f"registry declares requirement(s) the filter step cannot satisfy: "
        f"{sorted(declared - module.KNOWN_REQUIREMENTS)}"
    )


def test_the_tokenizer_reaches_the_filter(step, monkeypatch) -> None:
    """Asserted on the kwargs, since constructing it needs Curator installed."""
    import dataclasses

    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    seen: dict = {}
    original = signal_registry.SIGNALS["token_count"]
    monkeypatch.setitem(
        signal_registry.SIGNALS,
        "token_count",
        dataclasses.replace(original, factory=lambda **kwargs: seen.update(kwargs) or object()),
    )

    step.policy_stages(
        [{"signal": "token_count", "min": 10, "max": 4096}],
        "text",
        "filter",
        {"tokenizer": "some/model"},
    )

    assert seen["hf_model_name"] == "some/model"
    assert seen["min_tokens"] == 10 and seen["max_tokens"] == 4096


# -- content hash enforcement -------------------------------------------------


def test_a_policy_omitting_the_pack_hash_is_refused(step) -> None:
    """An unenforced guarantee reads exactly like an enforced one.

    The README promises a pack mismatch stops the run; a policy that simply
    declares no hash used to sail past both checks.
    """
    with pytest.raises(ValueError, match="declares no langpack.content_hash"):
        step.policy_stages(
            [{"signal": "stopword_ratio", "min": 0.1}],
            "text",
            "filter",
            {"language_tag": "x-test-vi", "langpack_dir": str(LANGPACK_FIXTURES)},
        )


def test_a_matching_hash_does_not_license_a_mislabelled_pack(step, tmp_path) -> None:
    """The content_hash gate is not a tag gate, and must not be mistaken for one.

    A hash certifies "the same bytes that were approved", never "the right
    language". A pack mislabelled from the first profile onward is internally
    self-consistent, so it carries its OWN valid hash and sails through the hash
    check -- which is exactly why the tag has to be compared separately. This
    test exists to record that the two guards answer different questions.
    """
    import shutil

    copy = tmp_path / "x-test-vi"
    shutil.copytree(LANGPACK_FIXTURES / "x-test-vi", copy)
    manifest = copy / "pack.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('language_tag = "x-test-vi"', 'language_tag = "x-test-ja"'),
        encoding="utf-8",
    )
    mislabelled = langpack.load_pack(copy)

    with pytest.raises(langpack.LanguagePackInvalidError, match="declares language_tag"):
        step.load_policy_pack(
            {
                "language_tag": "x-test-vi",
                "langpack_dir": str(tmp_path),
                "content_hash": mislabelled.content_hash,
            },
            ["stopword_ratio"],
        )


def test_resolve_policy_carries_the_config_tokenizer_and_pack_dir(step, tmp_path) -> None:
    path = write_policy(tmp_path, pack_policy())

    _, spec, _ = step.resolve_policy(
        {
            "heuristic_filters": {
                "approved_policy": str(path),
                "tokenizer": "some/model",
                "langpack_dir": "/somewhere/packs",
            }
        }
    )

    assert spec["tokenizer"] == "some/model"
    assert spec["langpack_dir"] == "/somewhere/packs"


def test_a_threshold_without_a_bound_is_refused_not_defaulted() -> None:
    """A missing bound must fail, not silently become Curator's English default.

    15 of the 24 signals ship a curator_default. For non_alpha_numeric it is
    0.25 — the ASCII-regex threshold this work exists to replace, which retains
    1.40% of a Vietnamese corpus. Substituting it for a bound the policy author
    forgot to write turns one missing line of YAML into a run that deletes 98.6%
    of the corpus and reports success.
    """
    from nemotron.steps.curate.nemo_curator import step
    from nemotron.steps.curate.nemo_curator.runtime import registry

    signal = registry.SIGNALS["non_alpha_numeric"]
    assert signal.curator_default, "fixture assumes this signal ships a default"

    with pytest.raises(ValueError, match="neither min nor max"):
        step.threshold_bounds(signal, {"signal": "non_alpha_numeric"})


def test_the_shipped_default_is_still_reported_just_never_applied() -> None:
    """Removing the fallback must not remove the value.

    curate/profile reports curator_default so a person can see what Curator would
    have used and decide for themselves. That is the whole difference between
    informing a decision and making it.
    """
    from nemotron.steps.curate.nemo_curator.runtime import registry

    assert registry.SIGNALS["non_alpha_numeric"].curator_default == (0.25,)


def test_the_manifest_says_whether_a_person_approved_the_thresholds(step, tmp_path) -> None:
    """The output directory is named filtered_jsonl either way.

    A first run has nothing to approve — that is the design, not a mistake — but
    its output sits in a directory with the same name as a released corpus and
    carries the same schema. Seven real runs produced exactly that: every
    flow_report.json recorded policy_applied false while the corpus sat in
    filtered_jsonl. The manifest now states the difference.
    """
    import json

    corpus = tmp_path / "in"
    corpus.mkdir()
    (corpus / "part.jsonl").write_text('{"id": "a", "text": "one"}\n{"id": "b", "text": "two"}\n', encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "part.jsonl").write_text('{"id": "a", "text": "one"}\n', encoding="utf-8")

    manifest = tmp_path / "run_manifest.json"
    step.emit_manifest(
        {"input_glob": str(corpus / "*.jsonl"), "output_dir": str(out), "id_field": "id"},
        str(manifest),
        "2026-09-04T00:00:00Z",
        completed=True,
    )

    policy = json.loads(manifest.read_text(encoding="utf-8"))["policy"]
    assert policy["status"] == "unapproved"
    assert policy["thresholds_applied"] == 0
    assert "not as a release" in policy["note"]


def test_an_overridden_policy_is_not_recorded_as_approved(tmp_path) -> None:
    """The manifest is what a reader has once the run log is gone.

    allow_unvalidated_policy runs a policy that failed the approval contract.
    The thresholds do gate the corpus, so "unapproved" would be false too — but
    recording "approved" launders the exact check that was skipped, and the note
    beside it claimed a corpus fingerprint the policy never carried.
    """
    from nemotron.steps.curate.nemo_curator import step as curate_step
    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    incomplete = tmp_path / "hand.yaml"
    incomplete.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "approved": True,
                "signals_impl_version": signal_registry.IMPL_VERSION,
                "thresholds": [{"signal": "numbers_ratio", "max": 0.25}],
            }
        )
    )
    cfg = {"heuristic_filters": {"approved_policy": str(incomplete), "allow_unvalidated_policy": True}}

    resolved = curate_step._resolve_policy(cfg)

    assert resolved.thresholds, "the policy still ran"
    assert resolved.overridden is True

    # The flag alone is not an override: a policy meeting the contract is approved
    # whether or not the escape hatch was left open.
    assert curate_step._resolve_policy({"heuristic_filters": {}}).overridden is False


def test_the_manifest_states_survive_an_empty_override_and_a_refusal(tmp_path) -> None:
    """Three artifact-level defects, all of them a report saying the wrong thing.

    An override with nothing to apply — the documented escape hatch points
    approved_policy at candidate_policies.yaml, which carries `candidates` and no
    `thresholds` — was recorded as plain "unapproved", indistinguishable from a
    run that configured no policy at all.

    A run refused for an unapproved policy wrote no manifest, because
    emit_manifest re-resolved the same file and re-raised inside the failure
    handler: the one failure most worth recording was the only one with no
    record.

    And "approved" never meant anyone approved — `approval` is optional in the
    contract — so the method is now carried separately.
    """
    from nemotron.steps.curate.nemo_curator import step as curate_step
    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    corpus = tmp_path / "in.jsonl"
    corpus.write_text('{"id": "a", "text": "x"}\n')
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def manifest_for(document: dict, **heuristics: object) -> dict:
        policy = tmp_path / f"policy-{len(list(tmp_path.iterdir()))}.yaml"
        policy.write_text(yaml.safe_dump(document))
        cfg = {
            "heuristic_filters": {"approved_policy": str(policy), **heuristics},
            "output_dir": str(output_dir),
            "input_glob": str(corpus),
            "text_field": "text",
            "id_field": "id",
        }
        try:
            resolved = curate_step._resolve_policy(cfg, [str(corpus)])
        except Exception:
            resolved = None
        written = tmp_path / f"manifest-{len(list(tmp_path.iterdir()))}.json"
        curate_step.emit_manifest(
            cfg, str(written), "t", completed=True, input_files=[str(corpus)], policy_resolution=resolved
        )
        return json.loads(written.read_text())["policy"]

    from nemotron.steps.curate.nemo_curator.runtime import integrity

    fingerprint = integrity.corpus_fingerprint([str(corpus)], "text", "id")
    base = {
        "schema_version": 1,
        "signals_impl_version": signal_registry.IMPL_VERSION,
        "corpus": {"fingerprint": fingerprint},
        "profile_digest": "sha256:" + "0" * 64,
    }

    # An override that gates nothing is still an override.
    empty = manifest_for({**base, "approved": False, "candidates": [{"signal": "x"}]}, allow_unvalidated_policy=True)
    assert empty["status"] == "override_unvalidated"
    assert empty["thresholds_applied"] == 0

    # A policy that cannot be resolved at all still produces a manifest.
    refused = manifest_for({"schema_version": 1, "approved": False, "thresholds": []})
    assert refused["status"] == "unresolved"
    assert "could not be resolved" in refused["note"]

    # "approved" is about the schema; the signature is recorded separately.
    unsigned = manifest_for({**base, "approved": True, "thresholds": [{"signal": "numbers_ratio", "max": 0.25}]})
    assert unsigned["approval_declared"] is None
