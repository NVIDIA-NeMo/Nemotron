# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""What the filter step actually reads, versus what the shared resolver returns.

Resolution and reading are two questions and this step answers them separately:
``integrity.expand_inputs`` decides what the corpus reference names, and
``READER_EXTENSIONS`` decides what ``JsonlReader`` can parse. Getting that seam
wrong is what N2 was about, and every defect below was live at some point in
fixing it:

* a companion file swept up by ``dir/*`` aborting a run that worked before;
* a stray ``.parquet`` beside the JSONL shards -- the shape a Hugging Face
  snapshot has -- aborting the same way;
* ``.ndjson``, which this category treats as JSONL, silently dropped;
* the refusal firing before the narrowing, so the failure manifest charged the
  corpus with unparsable rows from a file the step never opened;
* ``emit_manifest``/``emit_ledger`` re-resolving wide while ``run`` narrowed.

None of it was caught by a test: mutating ``input_files = readable`` to a no-op
left the suite green.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


class _Reader:
    """Captures the file_paths the pipeline hands the reader."""

    last_file_paths: list[str] | None = None

    def __init__(self, **kwargs):
        _Reader.last_file_paths = list(kwargs.get("file_paths") or [])


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
    sys.modules["nemo_curator.stages.text.io.reader"].JsonlReader = _Reader
    sys.modules["nemo_curator.stages.text.io.writer"].JsonlWriter = lambda **kw: object()
    sys.modules["huggingface_hub"].snapshot_download = lambda **kw: None

    import importlib

    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    _Reader.last_file_paths = None
    return module


def _shard(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": "a", "text": text}) + "\n", encoding="utf-8")
    return path


def _build(step, corpus: Path, **extra):
    cfg = {"input_glob": str(corpus), "output_dir": str(corpus.parent / "out"), "text_field": "text", **extra}
    return step.build_pipeline(cfg, input_files=step.readable_inputs(cfg["input_glob"]))


# -- what reaches the reader ---------------------------------------------------


def test_the_reader_is_handed_only_what_it_can_parse(step, tmp_path) -> None:
    """The narrowing itself. Mutating it to a no-op used to leave the suite green."""
    _shard(tmp_path / "part_0.jsonl")
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")

    _build(step, tmp_path)

    assert [Path(p).name for p in _Reader.last_file_paths] == ["part_0.jsonl"]


def test_an_ndjson_shard_is_read_not_dropped(step, tmp_path) -> None:
    """This category treats .ndjson as JSONL: ingest.detect_format groups it with
    .jsonl/.json, and the shared resolver returns it. Dropping it here was the
    resolve/read divergence N2 exists to close, moved to the reader boundary."""
    _shard(tmp_path / "part_0.jsonl")
    _shard(tmp_path / "part_1.ndjson")

    _build(step, tmp_path)

    assert [Path(p).name for p in _Reader.last_file_paths] == ["part_0.jsonl", "part_1.ndjson"]


def test_a_stray_parquet_beside_the_shards_does_not_abort_the_run(step, tmp_path) -> None:
    """A Hugging Face snapshot keeps both formats side by side, and curate/ingest
    writes JSONL into a directory that still holds the parquet it read."""
    _shard(tmp_path / "part_0.jsonl")
    (tmp_path / "part_0.parquet").write_bytes(b"PAR1")

    _build(step, tmp_path)

    assert [Path(p).name for p in _Reader.last_file_paths] == ["part_0.jsonl"]


# -- when the step must refuse -------------------------------------------------


def test_a_parquet_only_corpus_is_refused_and_names_ingest(step, tmp_path) -> None:
    (tmp_path / "part_0.parquet").write_bytes(b"PAR1")

    with pytest.raises(ValueError, match="cannot read"):
        step.run({"input_glob": str(tmp_path), "output_dir": str(tmp_path / "out"), "text_field": "text"})


def test_a_corpus_of_nothing_readable_is_refused(step, tmp_path) -> None:
    """Distinct from the parquet case: no corpus format at all, just companions."""
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="none this step can read"):
        step.run({"input_glob": str(tmp_path / "*"), "output_dir": str(tmp_path / "out"), "text_field": "text"})


def test_the_failure_manifest_does_not_charge_a_file_the_step_never_read(step, tmp_path) -> None:
    """The refusal must fire AFTER the narrowing.

    Raising while input_files still held the parquet made the failure handler
    write a manifest counting its binary lines as unparsable corpus damage --
    the precise harm the narrowing exists to prevent.
    """
    _shard(tmp_path / "part_0.jsonl")
    (tmp_path / "part_0.parquet").write_bytes(b"PAR1\n\x00\x01binary\nlines\n")
    manifest = tmp_path / "run_manifest.json"

    cfg = {
        "input_glob": str(tmp_path),
        "output_dir": str(tmp_path / "out"),
        "text_field": "text",
        "emit_manifest": str(manifest),
        "mode": "not-a-mode",  # fail late enough that the manifest is written
    }
    with pytest.raises(ValueError):
        step.run(cfg)

    if manifest.is_file():
        written = json.loads(manifest.read_text(encoding="utf-8"))
        assert written["input"].get("unparsable_rows", 0) == 0, written["input"]


# -- the emitters must agree with run() ----------------------------------------


def test_the_emitters_resolve_the_same_corpus_run_filters(step, tmp_path) -> None:
    """emit_manifest and emit_ledger used resolve_inputs while run() narrowed, so
    the same config described two different corpora depending on who asked -- and
    the wider count is one the audit consumes."""
    _shard(tmp_path / "part_0.jsonl")
    (tmp_path / "part_0.parquet").write_bytes(b"PAR1")
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")

    assert step.readable_inputs(str(tmp_path)) == [str(tmp_path / "part_0.jsonl")]
    assert len(step.resolve_inputs(str(tmp_path))) == 2, "the shared resolver still returns both"


# -- preflight and the runtime must agree ---------------------------------------
#
# Sharing `expand_inputs` made them use one RESOLVER. It did not make them apply
# one RULE: `expand_inputs` returns parquet because curate/ingest reads it, so
# preflight accepted a parquet corpus that the filter then refused. "Validate one
# corpus, process another" with an extra step. The reader rule now lives in
# integrity so both paths read it from the same place.

SHAPES = {
    "parquet-only": {"part_0.parquet": b"PAR1"},
    "jsonl-only": {"part_0.jsonl": b'{"id":"a","text":"x"}\n'},
    "parquet-beside-jsonl": {"part_0.jsonl": b'{"id":"a","text":"x"}\n', "part_0.parquet": b"PAR1"},
    "jsonl-and-companions": {"part_0.jsonl": b'{"id":"a","text":"x"}\n', "README.md": b"notes", "_SUCCESS": b""},
    # Resolves to a file and reads none. Preflight used to accept it and the step
    # then refused -- and no shape in this table could see that, which is why the
    # divergence survived a green suite twice.
    "companions-only": {"README.md": b"notes", "_SUCCESS": b""},
}


def _preflight_accepts(corpus: Path) -> bool:
    from nemotron.steps.curate.nemo_curator.scripts import run_flow

    cfg = {
        "corpus": {
            "input": str(corpus),
            "text_field": "text",
            "language": "en",
            "langpack_dir": str(Path(run_flow.__file__).parents[1] / "data" / "langpacks"),
        },
        "output_root": str(corpus.parent / "out"),
        "steps": {"filter": {"enabled": True}},
    }
    try:
        run_flow.plan(cfg)
    except run_flow.FlowConfigError:
        return False
    return True


def _runtime_accepts(step, corpus: Path) -> bool:
    try:
        step.run({"input_glob": str(corpus), "output_dir": str(corpus.parent / "o2"), "text_field": "text"})
    except ValueError as exc:
        if "cannot read" in str(exc) or "none this step can read" in str(exc):
            return False
    except Exception:  # noqa: BLE001 - got past the input checks, which is what is under test
        pass
    return True


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("spelling", ["{root}", "{root}/*"])
def test_preflight_validates_the_same_files_the_runtime_reads(step, tmp_path, shape, spelling) -> None:
    """The literal wording: "validate one corpus but process another".

    This used to compare VERDICTS -- does the run start? -- and reported
    agreement on that basis. Agreeing to start is not the same as agreeing on
    the corpus: preflight validated `[README.md, part_0.jsonl]` while the
    runtime read `[part_0.jsonl]`, on a shape this file already covered, and
    the test could not see it. Both spellings are checked because a bare
    directory and a glob resolve by different rules.
    """
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, content in SHAPES[shape].items():
        (corpus / name).write_bytes(content)
    reference = spelling.format(root=corpus)

    resolved = integrity.expand_inputs(reference)
    readable, _ = integrity.partition_by_reader(resolved)
    runtime = step.readable_inputs(reference)

    # What the runtime reads is exactly what preflight's own rule says it will.
    assert sorted(runtime) == sorted(readable), shape

    # And preflight refuses precisely when that set is empty, so it never
    # validates a run the step will not perform.
    assert _preflight_accepts(corpus) is bool(readable), shape


def test_a_parquet_corpus_is_accepted_when_ingest_will_convert_it(step, tmp_path) -> None:
    """The rule is 'no step can read this', not 'the filter cannot read this'.

    With ingest enabled the filter reads ingest's JSONL output, so refusing the
    raw parquet here would block the very flow that exists to handle it.
    """
    from nemotron.steps.curate.nemo_curator.scripts import run_flow

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "part_0.parquet").write_bytes(b"PAR1")

    cfg = {
        "corpus": {"input": str(corpus), "text_field": "text", "language": "en", "langpack_dir": "x"},
        "output_root": str(tmp_path / "out"),
        "steps": {"ingest": {"enabled": True, "format": "parquet"}, "filter": {"enabled": True}},
    }
    try:
        run_flow.plan(cfg)
    except run_flow.FlowConfigError as exc:
        assert "no step can read" not in str(exc), str(exc)
