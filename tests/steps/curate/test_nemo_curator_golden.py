# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Record intentional compatibility fixes in the constructed pipeline.

Two deliberate breaks are recorded. MultilingualDomainClassifier receives
``max_chars=None`` because its default mutates delivered text. JsonlReader
receives the exact preflight file list plus ``dtype=False`` and
``convert_dates=False`` because pandas otherwise changes identifiers such as
``"001"`` to ``1`` before the first stage.


The plan's acceptance criterion is a golden-file test proving byte-identical
behaviour when the new keys are absent. Behaviour here means the pipeline that
gets built: the stages, in order, with their arguments. If a config that predates
F1 produces the recorded stage sequence, any later output change is explicit.

``nemo_curator`` is a runtime dependency and is not installed on a plain CI host,
so the framework is stubbed and the constructed pipeline is recorded instead.
That records exactly what changed at the seam F1 touches — the reader's field
projection — which is the only place the diff can alter output.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

GOLDEN = Path(__file__).parent / "golden" / "golden_pipeline_pre_f1.json"
GOLDEN_FILTERS_ON = Path(__file__).parent / "golden" / "golden_pipeline_pre_f1_filters_on.json"


class _Recorder:
    """Stands in for a Curator stage, remembering how it was constructed.

    Positional arguments matter here: ``ScoreFilter`` receives the filter it
    wraps positionally, so a recorder that captured only keywords would not
    notice which filter was wired in.
    """

    def __init__(self, kind: str, log: list, *args, **kwargs):
        self._stub_kind = kind
        # Where a writer would put its shards, so the stubbed Pipeline.run() can
        # produce output the way a real one does — after the step has started,
        # not before it.
        if kind == "JsonlWriter":
            self.__written_to__ = kwargs.get("path") or (args[0] if args else None)
        log.append(
            {
                "stage": kind,
                "args": [_plain(v) for v in args],
                "kwargs": {k: _plain(v) for k, v in sorted(kwargs.items())},
            }
        )


def _plain(value):
    """Reduce a constructor argument to something comparable across runs."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in sorted(value.items())}
    if callable(value):
        return "<callable>"
    return getattr(value, "_stub_kind", type(value).__name__)


@pytest.fixture
def pipeline_log(monkeypatch):
    """Stub nemo_curator and huggingface_hub, recording every stage added."""
    log: list = []

    def stage(kind):
        return lambda *args, **kwargs: _Recorder(kind, log, *args, **kwargs)

    class _Pipeline:
        def __init__(self, **kwargs):
            # Curator's Pipeline keeps its stages in order and exposes them; the
            # ledger's per-gate attribution reads that list, so the stub carries
            # it too rather than letting the step be tested against a shape the
            # real class does not have.
            self.stages = []
            log.append({"stage": "Pipeline", "args": [], "kwargs": {k: _plain(v) for k, v in sorted(kwargs.items())}})

        def add_stage(self, s):
            self.stages.append(s)
            return self

        def run(self):
            # A real pipeline writes its output HERE, not before the step is
            # called. The step now refuses to start when output_dir already holds
            # shards — Curator's writer names them by content hash, so a second
            # run would add to a previous one — and a stub that wrote nothing
            # forced every test to pre-create the file it was checking for.
            for stage in self.stages:
                out = getattr(stage, "__written_to__", None)
                if out:
                    Path(out).mkdir(parents=True, exist_ok=True)
                    (Path(out) / "shard.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
            # list[Task] | None. None here: no executor ran, so no counters.
            return None

    class _RayClient:
        def __init__(self, **kwargs):
            log.append({"stage": "RayClient", "args": [], "kwargs": {k: _plain(v) for k, v in sorted(kwargs.items())}})

        def start(self):
            return None

        def stop(self):
            return None

    mods = {
        "nemo_curator": {},
        "nemo_curator.core": {},
        "nemo_curator.core.client": {"RayClient": _RayClient},
        "nemo_curator.pipeline": {"Pipeline": _Pipeline},
        "nemo_curator.stages": {},
        "nemo_curator.stages.text": {},
        "nemo_curator.stages.text.io": {},
        "nemo_curator.stages.text.io.reader": {"JsonlReader": stage("JsonlReader")},
        "nemo_curator.stages.text.io.writer": {"JsonlWriter": stage("JsonlWriter")},
        "nemo_curator.stages.text.modules": {
            "Filter": stage("Filter"),
            "ScoreFilter": stage("ScoreFilter"),
        },
        "nemo_curator.stages.text.filters": {},
        "nemo_curator.stages.text.filters.fasttext": {"FastTextLangId": stage("FastTextLangId")},
        "nemo_curator.stages.text.filters.heuristic": {"WordCountFilter": stage("WordCountFilter")},
        "nemo_curator.stages.text.classifiers": {
            "MultilingualDomainClassifier": stage("MultilingualDomainClassifier")
        },
        "huggingface_hub": {"snapshot_download": lambda **kwargs: None},
    }
    for name, attrs in mods.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    # Import after the stubs are in place; drop it afterwards so other tests
    # do not inherit a module bound to these fakes.
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    import importlib

    step = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    return step, log


def _legacy_config(tmp_path):
    """A config as it would have been written before F1 existed."""
    return {
        "language_codes": [],
        "domains": [],
        "text_field": "text",
        "input_glob": str(tmp_path / "in" / "*.jsonl"),
        "output_dir": str(tmp_path / "out"),
        "dataset": None,
        "models": {},
        "quality_filters": {},
    }


def _run(step, cfg, tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "in" / "a.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["step", "--config", str(cfg_path)])
    step.main()


def _normalised(log, tmp_path):
    """Replace the run's own directory so the golden describes shape, not location."""
    return json.loads(json.dumps(log).replace(str(tmp_path), "<TMP>"))


# -- the golden comparison ----------------------------------------------------


def test_a_config_predating_f1_builds_the_recorded_pipeline(pipeline_log, tmp_path, monkeypatch) -> None:
    step, log = pipeline_log

    _run(step, _legacy_config(tmp_path), tmp_path, monkeypatch)

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert _normalised(log, tmp_path) == expected, (
        "F1 changed what an existing config builds. The PR promises byte-identical behaviour "
        "for configs that predate it; regenerate the golden file only if the change is intended."
    )


def test_the_reader_projects_text_only_when_metadata_fields_is_absent(pipeline_log, tmp_path, monkeypatch) -> None:
    """The single seam F1 touches. Anything else in the diff cannot reach the output."""
    step, log = pipeline_log

    _run(step, _legacy_config(tmp_path), tmp_path, monkeypatch)

    reader = next(entry for entry in log if entry["stage"] == "JsonlReader")
    assert reader["kwargs"]["fields"] == ["text"]


def test_an_empty_metadata_fields_list_is_identical_to_omitting_it(pipeline_log, tmp_path, monkeypatch) -> None:
    """The shipped default sets it to []; that must not widen the projection."""
    step, log = pipeline_log

    cfg = _legacy_config(tmp_path)
    cfg["metadata_fields"] = []
    cfg["emit_manifest"] = None
    _run(step, cfg, tmp_path, monkeypatch)

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert _normalised(log, tmp_path) == expected


def test_declaring_metadata_fields_widens_the_projection(pipeline_log, tmp_path, monkeypatch) -> None:
    """The opt-in half of the same seam, so the golden test cannot pass vacuously."""
    step, log = pipeline_log

    cfg = _legacy_config(tmp_path)
    cfg["metadata_fields"] = ["id", "source"]
    _run(step, cfg, tmp_path, monkeypatch)

    reader = next(entry for entry in log if entry["stage"] == "JsonlReader")
    assert reader["kwargs"]["fields"] == ["text", "id", "source"]


def test_the_text_field_is_not_duplicated_if_also_listed(pipeline_log, tmp_path, monkeypatch) -> None:
    step, log = pipeline_log

    cfg = _legacy_config(tmp_path)
    cfg["metadata_fields"] = ["text", "id"]
    _run(step, cfg, tmp_path, monkeypatch)

    reader = next(entry for entry in log if entry["stage"] == "JsonlReader")
    assert reader["kwargs"]["fields"] == ["text", "id"]


def test_no_manifest_is_written_when_emit_manifest_is_absent(pipeline_log, tmp_path, monkeypatch) -> None:
    step, _log = pipeline_log

    _run(step, _legacy_config(tmp_path), tmp_path, monkeypatch)

    assert not list(tmp_path.rglob("run_manifest.json"))


def test_a_manifest_is_written_when_asked(pipeline_log, tmp_path, monkeypatch) -> None:
    step, _log = pipeline_log

    cfg = _legacy_config(tmp_path)
    destination = tmp_path / "out" / "run_manifest.json"
    cfg["emit_manifest"] = str(destination)
    _run(step, cfg, tmp_path, monkeypatch)

    from nemotron.steps.curate.nemo_curator.runtime import manifest as m

    assert destination.exists()
    document = m.read_manifest(destination)
    assert m.validate_manifest(document) == []
    assert m.is_complete(document), "a run that finished must record completed_at"


def test_a_config_with_every_filter_enabled_also_matches(pipeline_log, tmp_path, monkeypatch) -> None:
    """The filters-off golden alone cannot see the branches that build filter stages."""
    step, log = pipeline_log

    cfg = _legacy_config(tmp_path)
    cfg["language_codes"] = ["EN"]
    cfg["domains"] = ["News"]
    cfg["models"] = {"fasttext_langid": "/models/lid.176.bin", "hf_cache_dir": "/cache/hf"}
    cfg["quality_filters"] = {"min_langid_score": 0.3, "min_words": 50, "max_words": 5000}
    # Pipeline construction only; label validation is tested separately.
    monkeypatch.setattr(step, "validate_language_codes", lambda _cfg: None)
    _run(step, cfg, tmp_path, monkeypatch)

    expected = json.loads(GOLDEN_FILTERS_ON.read_text(encoding="utf-8"))
    assert _normalised(log, tmp_path) == expected


def test_the_filters_on_golden_actually_records_the_filter_stages() -> None:
    """Guards the guard: a golden that recorded nothing would pass forever."""
    stages = [entry["stage"] for entry in json.loads(GOLDEN_FILTERS_ON.read_text(encoding="utf-8"))]

    assert {"FastTextLangId", "ScoreFilter", "Filter", "WordCountFilter", "MultilingualDomainClassifier"} <= set(
        stages
    )


def test_score_filter_records_the_filter_it_wraps() -> None:
    """A recorder blind to positional arguments could not tell the filters apart."""
    entries = json.loads(GOLDEN_FILTERS_ON.read_text(encoding="utf-8"))
    wrapped = [e["args"][0] for e in entries if e["stage"] == "ScoreFilter" and e["args"]]

    assert wrapped == ["FastTextLangId", "WordCountFilter"]


# -- the language gate --------------------------------------------------------


def test_the_language_gate_matches_the_case_fasttext_emits(pipeline_log) -> None:
    """lid.176 emits '__label__en'; the config's codes are upper-cased on load.

    An exact match rejects every document while looking like a filter that simply
    found nothing, which is the hardest kind of failure to notice.
    """
    step, _log = pipeline_log

    assert step.keep_language("[0.9, 'en']", {"EN"}) is True
    assert step.keep_language("[0.9, 'EN']", {"EN"}) is True
    assert step.keep_language("[0.9, 'vi']", {"EN"}) is False


def test_a_script_suffixed_label_matches_its_base_language(pipeline_log) -> None:
    """'ZH' should select zh_Hans without enumerating every script variant."""
    step, _log = pipeline_log

    assert step.keep_language("[0.9, 'zh_Hans']", {"ZH"}) is True


def test_a_negative_score_is_rejected_whatever_the_label(pipeline_log) -> None:
    step, _log = pipeline_log

    assert step.keep_language("[-1.0, 'en']", {"EN"}) is False


def test_manifest_fields_are_carried_through_the_reader(pipeline_log, tmp_path, monkeypatch) -> None:
    """Naming a field for the manifest but not the projection reads it away."""
    step, log = pipeline_log

    cfg = _legacy_config(tmp_path)
    cfg["id_field"] = "id"
    cfg["source_field"] = "source"
    _run(step, cfg, tmp_path, monkeypatch)

    reader = next(entry for entry in log if entry["stage"] == "JsonlReader")
    assert reader["kwargs"]["fields"] == ["text", "id", "source"]


def test_a_directory_of_json_line_files_is_discovered(pipeline_log, tmp_path) -> None:
    """A directory holding .json rather than .jsonl must not count as zero files.

    The manifest's input counts are only meaningful if this sees what the reader
    saw. Missing an extension makes the manifest claim the run read nothing and
    turns every subsequent audit into a false mismatch.
    """
    step, _log = pipeline_log

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.json").write_text('{"text":"x"}\n', encoding="utf-8")
    (corpus / "b.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")
    (corpus / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    found = step.resolve_inputs(str(corpus))

    assert {Path(f).name for f in found} == {"a.json", "b.jsonl"}


def test_a_literal_file_path_is_used_as_given(pipeline_log, tmp_path) -> None:
    step, _log = pipeline_log
    shard = tmp_path / "one.jsonl"
    shard.write_text('{"text":"x"}\n', encoding="utf-8")

    assert step.resolve_inputs(str(shard)) == [str(shard)]


def test_a_path_that_does_not_exist_yields_nothing(pipeline_log, tmp_path) -> None:
    """Better an empty list than a manifest counting a file that is not there."""
    step, _log = pipeline_log

    assert step.resolve_inputs(str(tmp_path / "absent.jsonl")) == []
