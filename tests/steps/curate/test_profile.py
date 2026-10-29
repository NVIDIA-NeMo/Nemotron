# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Behaviour of ``steps/curate/profile``, against the plan's acceptance criteria.

The real signals are Curator's, and Curator is a runtime dependency that is not
installed on a plain CI host. The stubs below stand in for it with the same
``score_document`` / ``keep_document`` contract, so the sampling, measurement,
and reporting logic is exercised end to end without the framework.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from nemotron.steps.curate.nemo_curator.runtime import langpack, policy
from nemotron.steps.curate.nemo_curator.runtime import registry as r
from nemotron.steps.curate.nemo_curator.scripts import run_profile

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "nemo_curator", "profile")
LANGPACK_FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"


# -- static -------------------------------------------------------------------


def test_profile_step_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/profile",
        expected_launch="python",
        expected_default_config="default",
    )


def test_language_is_declared_without_a_default() -> None:
    """A wrong default produces plausible numbers for the wrong language."""
    import tomllib

    with (STEP_DIR / "step.toml").open("rb") as fh:
        toml = tomllib.load(fh)

    by_name = {p["name"]: p for p in toml["parameters"]}
    assert "language" in by_name
    assert "default" not in by_name["language"], "language must have no default"
    assert by_name["langpack_dir"]["default"] == "null"


def test_named_english_profile_explicitly_selects_the_reference_pack() -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / "en.yaml").read_text(encoding="utf-8"))
    package_packs = STEP_DIR.parent / "data" / "langpacks"
    pack = langpack.load("en", package_packs)

    assert cfg["language"] == "en"
    assert cfg["langpack_dir"].endswith("/steps/curate/nemo_curator/data/langpacks")
    assert cfg["models"] == {}
    for signal_name in cfg["signals"]:
        assert set(r.SIGNALS[signal_name].requires) <= set(pack.capabilities)


def test_the_fixture_has_unequal_sources() -> None:
    """A fixture with balanced sources cannot exercise macro versus micro."""
    counts: dict[str, int] = {}
    for shard in sorted((STEP_DIR / "data" / "tiny").glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                source = json.loads(line)["source"]
                counts[source] = counts.get(source, 0) + 1

    assert len(counts) >= 2
    assert max(counts.values()) >= 3 * min(counts.values())


# -- stubs --------------------------------------------------------------------


def _make_filter(direction: str, score_fn):
    class _Stub:
        def __init__(self, **kwargs):
            self._thresholds = tuple(v for k, v in kwargs.items() if k not in ("lang", "n"))

        def score_document(self, text):
            return score_fn(text)

        def keep_document(self, score):
            if direction == "max":
                return score <= self._thresholds[0]
            if direction == "min":
                return score >= self._thresholds[0]
            lo, hi = self._thresholds
            return lo <= score <= hi

    return _Stub


def _non_alpha(text: str) -> float:
    import re

    keep = len(re.findall(r"[a-zA-Z0-9\n?!,.]", text))
    return (len(text) - keep) / len(text) if text else 1.0


_STUBS = {
    "NonAlphaNumericFilter": _make_filter("max", _non_alpha),
    "SymbolsToWordsFilter": _make_filter("max", lambda t: t.count("#") / max(len(t.split()), 1)),
    "NumbersFilter": _make_filter("max", lambda t: sum(c.isdigit() for c in t) / max(len(t), 1)),
    "UrlsFilter": _make_filter("max", lambda t: t.count("http") / max(len(t.split()), 1)),
    "BulletsFilter": _make_filter("max", lambda t: 0.0),
    "WhiteSpaceFilter": _make_filter("max", lambda t: sum(c.isspace() for c in t) / max(len(t), 1)),
    "ParenthesesFilter": _make_filter("max", lambda t: t.count("(") / max(len(t), 1)),
    "LongWordFilter": _make_filter("max", lambda t: max((len(w) for w in t.split()), default=0)),
    "PunctuationFilter": _make_filter("max", lambda t: 0.0 if t.rstrip().endswith(".") else 1.0),
    "EllipsisFilter": _make_filter("max", lambda t: 0.0),
    "WordsWithoutAlphabetsFilter": _make_filter("min", lambda t: 1.0),
    "WordCountFilter": _make_filter("interval", lambda t: len(t.split())),
    "MeanWordLengthFilter": _make_filter(
        "interval", lambda t: sum(len(w) for w in t.split()) / max(len(t.split()), 1)
    ),
}

_REPETITION_STUBS = {
    "RepeatingDuplicateNGramsFilter": _make_filter("max", lambda t: 0.1),
}


@pytest.fixture
def curator_stub(monkeypatch):
    """Install a minimal nemo_curator so the registry's lazy imports resolve."""
    modules = {
        "nemo_curator": types.ModuleType("nemo_curator"),
        "nemo_curator.stages": types.ModuleType("nemo_curator.stages"),
        "nemo_curator.stages.text": types.ModuleType("nemo_curator.stages.text"),
        "nemo_curator.stages.text.filters": types.ModuleType("nemo_curator.stages.text.filters"),
        "nemo_curator.stages.text.filters.heuristic": types.ModuleType("nemo_curator.stages.text.filters.heuristic"),
        "nemo_curator.stages.text.filters.heuristic.repetition": types.ModuleType(
            "nemo_curator.stages.text.filters.heuristic.repetition"
        ),
    }
    string_mod = types.ModuleType("nemo_curator.stages.text.filters.heuristic.string")
    for name, cls in _STUBS.items():
        setattr(string_mod, name, cls)
    modules["nemo_curator.stages.text.filters.heuristic.string"] = string_mod

    rep_mod = types.ModuleType("nemo_curator.stages.text.filters.heuristic.repetition.repetition")
    for name, cls in _REPETITION_STUBS.items():
        setattr(rep_mod, name, cls)
    modules["nemo_curator.stages.text.filters.heuristic.repetition.repetition"] = rep_mod

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


def _config(tmp_path, **overrides):
    cfg = {
        "input_glob": str(STEP_DIR / "data" / "tiny" / "*.jsonl"),
        "output_dir": str(tmp_path),
        "text_field": "text",
        "source_field": "source",
        "id_field": "id",
        # unicode_alpha_numeric, not non_alpha_numeric: the pack is Vietnamese
        # and the ASCII signal is gated behind ascii_alphabet, which it does not
        # declare. That refusal is the behaviour under test elsewhere; here the
        # signal is only a vehicle for profiling.
        "language": "x-test-vi",
        "langpack_dir": str(LANGPACK_FIXTURES),
        "signals": ["unicode_alpha_numeric", "word_count"],
        "max_total_docs": 0,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


# -- acceptance criteria ------------------------------------------------------


def test_no_output_is_ever_marked_approved(tmp_path, curator_stub) -> None:
    _report, policies, _manifest = run_profile.build_report(_config(tmp_path))

    assert policies["approved"] is False


def test_an_interval_signal_reports_a_surface(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))

    entry = report["signals"][0]
    assert entry["direction"] == "interval"
    assert entry["retention"]["kind"] == "surface"
    assert entry["retention"]["min_axis"] and entry["retention"]["max_axis"]


def test_a_one_sided_signal_reports_a_curve_and_bands(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=["unicode_alpha_numeric"]))

    entry = report["signals"][0]
    assert entry["retention"]["kind"] == "curve"
    assert "retention_stable_bands" in entry


def test_every_cooccurrence_entry_names_its_operating_point(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path))

    assert report["cooccurrence"]
    for entry in report["cooccurrence"]:
        assert entry["thresholds_a"]
        assert entry["thresholds_b"]


def test_macro_and_micro_are_both_reported_and_labelled(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))

    views = report["signals"][0]["views"]
    assert set(views) == {"macro", "micro"}
    assert views["macro"]["note"] != views["micro"]["note"]


def test_macro_and_micro_differ_on_unequal_sources(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))

    views = report["signals"][0]["views"]
    macro = views["macro"]["quantiles"]["p50"]
    micro = views["micro"]["quantiles"]["p50"]

    # The fixture is 36 short web documents and 4 long wiki ones. Macro gives the
    # two sources equal weight, so it sits between them; micro reconstructs the
    # corpus, so it sits with the short majority. Asserting only that they differ
    # would pass on floating-point noise and would not notice the two views being
    # swapped.
    assert micro < macro, "micro must follow the numerous short documents"
    assert micro < 20, "the web documents are short"
    assert macro > micro * 1.5, "wiki's length must visibly pull the equal-weight view up"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_are_not_counted_as_successfully_scored(value) -> None:
    class NonFiniteFilter:
        @staticmethod
        def score_document(_text):
            return value

    signal = r.Signal(
        name="non_finite",
        factory=lambda **_kwargs: NonFiniteFilter(),
        direction="max",
        units="ratio",
        grid=r.Grid(0.0, 1.0),
        threshold_params=("max_value",),
    )

    _scores, health = run_profile.score_sample([signal], {"source": [("doc-1", "document")]})

    assert health["non_finite"]["documents_attempted"] == 1
    assert health["non_finite"]["documents_scored"] == 0
    assert health["non_finite"]["scoring_failures"] == 1
    assert health["non_finite"]["non_finite_scores"] == 1


def test_a_named_unsupported_signal_fails(tmp_path, curator_stub) -> None:
    with pytest.raises(r.SignalRequirementsUnmetError):
        run_profile.build_report(_config(tmp_path, signals=["token_count"]))


def test_auto_selection_skips_an_unsupported_signal_with_a_note(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=[]))

    assert any("token_count" in note for note in report["notes"])
    assert "token_count" not in {s["signal"] for s in report["signals"]}


# -- provenance ---------------------------------------------------------------


def test_the_report_says_when_and_from_what_config_it_was_made(tmp_path, curator_stub) -> None:
    """Without this a report copied next to another run's output is indistinguishable."""
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))
    producer = report["producer"]

    assert producer["step_id"] == "curate/profile"
    assert producer["started_at"] <= producer["completed_at"]
    assert producer["config_hash"].startswith("sha256:")
    assert producer["tool_revision"]


def test_the_config_hash_tracks_the_config(tmp_path, curator_stub) -> None:
    one, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))
    same, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))
    other, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"], seed=7))

    assert one["producer"]["config_hash"] == same["producer"]["config_hash"]
    assert one["producer"]["config_hash"] != other["producer"]["config_hash"]


def test_the_digest_ignores_provenance(tmp_path, curator_stub) -> None:
    """Re-profiling the same corpus must not make an approved policy look stale."""
    one, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))
    two, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))

    assert one["profile_digest"] == two["profile_digest"]


def test_the_digest_still_follows_the_measurements(tmp_path, curator_stub) -> None:
    one, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))
    other, _, _ = run_profile.build_report(_config(tmp_path, signals=["unicode_alpha_numeric"]))

    assert one["profile_digest"] != other["profile_digest"]


def test_the_digest_is_recomputable_from_the_written_report(tmp_path, curator_stub) -> None:
    """``digest_covers`` has to be precise enough for a reader to check the link."""
    run_profile.run(_config(tmp_path, signals=["word_count"]))

    report = json.loads((tmp_path / "profile_report.json").read_text(encoding="utf-8"))
    policies = yaml.safe_load((tmp_path / "candidate_policies.yaml").read_text(encoding="utf-8"))

    assert report["producer"]["digest_covers"] == "every key except producer and profile_digest"
    measurements = {k: v for k, v in report.items() if k not in ("producer", "profile_digest")}
    assert policy.digest(measurements) == report["profile_digest"]
    assert policies["profile_digest"] == report["profile_digest"]


# -- the tokenizer behind token_count -----------------------------------------


def test_no_tokenizer_configured_reports_none() -> None:
    assert run_profile.resolve_tokenizer({}) is None
    assert run_profile.resolve_tokenizer({"models": {}}) is None
    assert run_profile.resolve_tokenizer({"models": {"tokenizer": None}}) is None


def test_a_pinned_tokenizer_resolves_to_name_and_revision() -> None:
    cfg = {"models": {"tokenizer": {"name": "org/model", "revision": "abc123"}}}

    assert run_profile.resolve_tokenizer(cfg) == ("org/model", "abc123")


def test_a_bare_model_name_is_refused() -> None:
    """It cannot carry a revision, and TokenCountFilter has no revision parameter."""
    with pytest.raises(run_profile.ConfigError, match="mapping with a name and a revision"):
        run_profile.resolve_tokenizer({"models": {"tokenizer": "org/model"}})


def test_an_unpinned_tokenizer_is_refused() -> None:
    """Counts from two revisions are not comparable, so a promoted threshold would be unverifiable."""
    with pytest.raises(run_profile.ConfigError, match="revision is required"):
        run_profile.resolve_tokenizer({"models": {"tokenizer": {"name": "org/model"}}})


def test_the_report_records_the_tokenizer_it_used(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=["word_count"]))

    assert report["tokenizer"] is None, "absence must be stated, not omitted"


def test_the_revision_reaches_the_filter(tmp_path, curator_stub, monkeypatch) -> None:
    """A pin recorded in the report but not passed to the tokenizer would be inert."""
    seen: dict[str, object] = {}
    stub = _make_filter("interval", lambda t: float(len(t.split())))

    def spy(**kwargs):
        seen.update(kwargs)
        return stub(min_tokens=kwargs["min_tokens"], max_tokens=kwargs["max_tokens"])

    monkeypatch.setitem(r.SIGNALS, "token_count", dataclasses.replace(r.SIGNALS["token_count"], factory=spy))

    cfg = _config(tmp_path, signals=["token_count"])
    cfg["models"] = {"tokenizer": {"name": "org/model", "revision": "abc123"}}
    report, _, _ = run_profile.build_report(cfg)

    assert seen["hf_model_name"] == "org/model"
    assert seen["transformers_init_kwargs"] == {"revision": "abc123"}
    assert report["tokenizer"] == {"name": "org/model", "revision": "abc123"}


def test_the_sample_is_reproducible_from_seed_and_cap(tmp_path, curator_stub) -> None:
    cfg = _config(tmp_path, max_total_docs=12, seed=5)

    _, _, first = run_profile.build_report(cfg)
    _, _, second = run_profile.build_report(cfg)
    _, _, other = run_profile.build_report(_config(tmp_path, max_total_docs=12, seed=6))

    assert first["sampled_keys"] == second["sampled_keys"]
    assert first["sampled_keys"] != other["sampled_keys"]


def test_the_sample_manifest_records_population_and_weight(tmp_path, curator_stub) -> None:
    _, _, manifest = run_profile.build_report(_config(tmp_path, max_total_docs=12))

    assert manifest["allocations"]
    for allocation in manifest["allocations"]:
        assert {"source", "population", "sampled", "weight"} <= set(allocation)


# -- reporting rules ----------------------------------------------------------


def test_the_report_states_what_curators_default_would_keep(tmp_path, curator_stub) -> None:
    """The finding the step exists to surface."""
    report, _, _ = run_profile.build_report(_config(tmp_path, signals=["unicode_alpha_numeric"]))

    default = report["signals"][0]["curator_default"]
    assert default["thresholds"] == [0.25]

    # The fixture is plain ASCII English, so the ASCII-only alphanumeric class
    # matches nearly everything and the shipped default keeps the corpus. Pinning
    # the actual value is what makes this test able to fail: a bound like
    # 0 <= x <= 1 is true of every fraction ever computed.
    assert default["retained"] == pytest.approx(1.0)

    curve = {round(pt["threshold"], 4): pt["retained"] for pt in report["signals"][0]["retention"]["points"]}
    assert curve[0.0] == pytest.approx(0.0), "no document is pure alphanumeric"
    assert curve[1.0] == pytest.approx(1.0), "every document passes a threshold of 1.0"


def test_the_report_says_it_is_descriptive(tmp_path, curator_stub) -> None:
    report, _, _ = run_profile.build_report(_config(tmp_path))

    assert "not whether what it removes is low quality" in report["interpretation"]


def test_a_corpus_without_the_source_field_says_so(tmp_path, curator_stub) -> None:
    shard = tmp_path / "in.jsonl"
    shard.write_text('{"id":"1","text":"a document with words"}\n', encoding="utf-8")

    report, _, _ = run_profile.build_report(
        _config(tmp_path, input_glob=str(shard), signals=["unicode_alpha_numeric"])
    )

    assert any("shard path" in note for note in report["notes"])


def test_an_unmatched_glob_is_an_error(tmp_path, curator_stub) -> None:
    """ConfigError, the same class every sibling runner raises for the same mistake."""
    with pytest.raises(run_profile.ConfigError, match="none-"):
        run_profile.build_report(_config(tmp_path, input_glob=str(tmp_path / "none-*.jsonl")))


def test_a_corpus_with_no_parsable_records_is_an_error(tmp_path, curator_stub) -> None:
    shard = tmp_path / "in.jsonl"
    shard.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no records with a string"):
        run_profile.build_report(_config(tmp_path, input_glob=str(shard)))


def test_the_three_artifacts_are_written(tmp_path, curator_stub, monkeypatch) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_config(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_profile", "--config", str(cfg_path)])

    run_profile.main()

    assert (tmp_path / "profile_report.json").exists()
    assert (tmp_path / "candidate_policies.yaml").exists()
    assert (tmp_path / "sample_manifest.json").exists()


def test_unparsable_lines_are_counted_and_noted(tmp_path, curator_stub) -> None:
    """curate/audit calls this damage a finding; profile must not stay silent."""
    shard = tmp_path / "in.jsonl"
    shard.write_text(
        '{"id":"1","source":"a","text":"a real document with several words"}\n'
        "this line will not parse\n"
        '{"id":"2","source":"a","text":"another real document with words"}\n',
        encoding="utf-8",
    )

    report, _, _ = run_profile.build_report(
        _config(tmp_path, input_glob=str(shard), signals=["unicode_alpha_numeric"])
    )

    assert report["corpus"]["unparsable_lines"] == 1
    assert report["corpus"]["damaged_shards"] == 1
    assert any("would not parse" in note for note in report["notes"])


class _StubLangModel:
    """FastText's batch interface, which is the one that works under numpy>=2."""

    def __init__(self, verdicts: dict[str, tuple[str, float]]) -> None:
        self._verdicts = verdicts

    def predict(self, texts, k=1):  # noqa: ANN001, ARG002 - mirrors the library signature
        labels, scores = [], []
        for text in texts:
            code, score = self._verdicts[text]
            labels.append([f"__label__{code}"])
            scores.append([score])
        return labels, scores


def _install_stub_fasttext(monkeypatch, verdicts, tmp_path):
    import sys
    import types

    module = types.ModuleType("fasttext")
    module.load_model = lambda _path: _StubLangModel(verdicts)
    monkeypatch.setitem(sys.modules, "fasttext", module)
    model = tmp_path / "lid.176.bin"
    model.write_bytes(b"")
    return str(model)


def test_the_profile_reports_what_language_the_corpus_is_in(monkeypatch, tmp_path) -> None:
    """No signal can answer this, and it is the question a person asks first.

    script_ratio and latin_ratio come closest and cannot help: languages that
    share a script are indistinguishable to a script ratio. Without this,
    naming the wrong language in
    steps.filter.language_codes removes almost everything and still reports
    success, because the row counts reconcile either way.
    """
    from nemotron.steps.curate.nemo_curator.scripts import run_profile

    verdicts = {f"vi-{i}": ("vi", 0.99) for i in range(8)}
    verdicts.update({f"en-{i}": ("en", 0.60) for i in range(2)})
    model = _install_stub_fasttext(monkeypatch, verdicts, tmp_path)
    sample = {"src": [(str(i), text) for i, text in enumerate(verdicts)]}

    out = run_profile.language_composition(sample, model)

    assert out["scored"] == 10
    by_code = {row["language"]: row for row in out["languages"]}
    assert by_code["vi"]["documents"] == 8
    assert by_code["vi"]["share"] == pytest.approx(0.8)
    assert by_code["en"]["share"] == pytest.approx(0.2)
    # ordered by size, so the corpus's actual language is the first row read
    assert out["languages"][0]["language"] == "vi"


def test_the_confidence_distribution_is_reported_separately(monkeypatch, tmp_path) -> None:
    """Which language to keep and how sure the model must be are two decisions,
    and min_langid_score is the second one. Reporting only the label would leave
    a threshold with no distribution behind it."""
    from nemotron.steps.curate.nemo_curator.scripts import run_profile

    verdicts = {f"doc-{i}": ("vi", i / 100) for i in range(100)}
    model = _install_stub_fasttext(monkeypatch, verdicts, tmp_path)
    sample = {"src": [(str(i), text) for i, text in enumerate(verdicts)]}

    out = run_profile.language_composition(sample, model)

    c = out["confidence"]
    assert c["p1"] < c["p25"] < c["p50"] < c["p75"]
    assert 0.0 <= c["p1"] <= 1.0


def test_no_model_reports_absence_rather_than_guessing() -> None:
    """Nemotron does not choose a language for the user. An absent model is
    recorded as unmeasured, not filled in from the script or a heuristic.

    Only an absent key means that. A path that is set and unreadable is a typo,
    and it is refused — see
    test_a_wrong_fasttext_path_is_refused_not_silently_skipped. This test
    asserted the opposite until then, which is how the two came to be
    indistinguishable in the report.
    """
    from nemotron.steps.curate.nemo_curator.scripts import run_profile

    sample = {"src": [("1", "Tiếng Việt")]}

    assert run_profile.language_composition(sample, None) is None


def test_a_document_with_embedded_newlines_is_still_scored(monkeypatch, tmp_path) -> None:
    """FastText reads one line. A newline mid-document truncates the text it
    sees, so a long article would be classified on its first line alone."""
    from nemotron.steps.curate.nemo_curator.scripts import run_profile

    model = _install_stub_fasttext(monkeypatch, {"one two three": ("vi", 0.9)}, tmp_path)
    sample = {"src": [("1", "one\ntwo\n\nthree")]}

    out = run_profile.language_composition(sample, model)

    assert out["scored"] == 1, "the newlines must be flattened before predict, not truncate it"


def test_a_wrong_fasttext_path_is_refused_not_silently_skipped(tmp_path) -> None:
    """The degraded report was byte-identical to the clean one.

    `models.fasttext_langid` absent means "do not measure language", and that is
    a legitimate choice this step records rather than fills in. A path that is
    set but unreadable is a typo, and returning None for it produced a report
    with language_composition null, no note, and no non-zero exit — identical in
    every byte to a run that asked for no model. The reader would conclude the
    corpus had no language breakdown rather than that theirs had failed.
    """
    from nemotron.steps.curate.nemo_curator.scripts import run_profile

    sample = {"src": [("a", "xin chào thế giới")]}

    assert run_profile.language_composition(sample, None) is None, "absent stays absent"
    assert run_profile.language_composition(sample, "") is None

    with pytest.raises(FileNotFoundError, match="which is not a file"):
        run_profile.language_composition(sample, str(tmp_path / "typo-lid.176.bin"))
