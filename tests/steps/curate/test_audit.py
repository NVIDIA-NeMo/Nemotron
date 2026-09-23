# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Behaviour of ``steps/curate/audit``, against the acceptance criteria in the plan."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from nemotron.steps.curate.nemo_curator.runtime import integrity
from nemotron.steps.curate.nemo_curator.runtime import manifest as m
from nemotron.steps.curate.nemo_curator.scripts import run_audit

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "nemo_curator", "audit")


def corpus(tmp_path, *shards: tuple[str, list[str]]):
    for name, records in shards:
        (tmp_path / name).write_text("".join(r + "\n" for r in records), encoding="utf-8")
    return str(tmp_path / "*.jsonl")


def manifest_for(tmp_path, *, completed: bool = True, row_count: int | None = None, file_count: int | None = None):
    files = sorted(tmp_path.glob("*.jsonl"))
    counts = m.count_jsonl(files)
    if row_count is not None:
        counts["row_count"] = row_count
    if file_count is not None:
        counts["file_count"] = file_count
    document = m.build_manifest(
        step_id="curate/nemo_curator",
        config={},
        started_at="2026-08-24T00:00:00+00:00",
        input_glob="in/*.jsonl",
        input_counts=dict(counts),
        output_counts=dict(counts),
        completed_at="2026-08-24T00:00:01+00:00" if completed else None,
    )
    path = tmp_path / "run_manifest.json"
    m.write_manifest(path, document)
    return str(path)


# -- static -------------------------------------------------------------------


def test_audit_step_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/audit",
        expected_launch="python",
        expected_default_config="default",
    )


def test_audit_declares_no_gpu() -> None:
    """It reads files. Requiring a GPU would make it useless as a cheap gate."""
    source = (STEP_DIR / "step.py").read_text(encoding="utf-8")

    assert "gpus_per_node = 0" in source


def test_every_finding_name_is_documented_as_an_error() -> None:
    """A finding a user cannot look up in the manifest is a dead end.

    Names are read out of the runner rather than listed here, so adding a
    finding without documenting it fails instead of passing a fixed list that
    nobody remembers to extend.
    """
    import ast
    import inspect

    with (STEP_DIR / "step.toml").open("rb") as fh:
        toml = tomllib.load(fh)
    documented = {e["name"] for e in toml.get("errors", [])}

    emitted = set()
    tree = ast.parse(inspect.getsource(run_audit))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "name"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                emitted.add(value.value)

    # Observations use the same {"name": ...} shape and are not errors.
    observations = {"reference_delta", "zero_row_shard"}

    assert emitted - observations <= documented, (
        f"undocumented finding(s): {sorted(emitted - observations - documented)}"
    )
    assert {"unreadable_shard", "manifest_mismatch", "containment_violation"} <= emitted


def test_tiny_config_does_not_request_containment_without_a_reference() -> None:
    """containment needs reference_glob; a smoke config that raises is not a smoke test."""
    cfg = yaml.safe_load((STEP_DIR / "config" / "tiny.yaml").read_text(encoding="utf-8"))

    if cfg["mode"] in ("containment", "all"):
        assert cfg["reference_glob"], "tiny requests containment but supplies no reference corpus"


def test_the_packaged_fixture_manifest_is_conformant() -> None:
    """The producer and the auditor must agree on the schema, or tiny proves nothing."""
    document = m.read_manifest(STEP_DIR / "data" / "tiny" / "run_manifest.json")

    assert m.validate_manifest(document) == []
    assert m.is_complete(document)


# -- acceptance criteria ------------------------------------------------------


def test_a_truncated_shard_is_a_finding_naming_file_and_offset(tmp_path) -> None:
    (tmp_path / "a.jsonl").write_text('{"id":"1"}\n{"id":"2", "text":"cut', encoding="utf-8")

    report = run_audit.audit({"target_glob": str(tmp_path / "*.jsonl")})

    assert not report["passed"]
    finding = next(f for f in report["findings"] if f["name"] == "unreadable_shard")
    assert "a.jsonl" in finding["message"]
    assert "byte 11" in finding["message"]


def test_a_valid_json_value_that_is_not_a_record_is_a_finding(tmp_path) -> None:
    (tmp_path / "a.jsonl").write_text('{"id":"1"}\n[1, 2, 3]\n', encoding="utf-8")

    report = run_audit.audit({"target_glob": str(tmp_path / "*.jsonl")})

    finding = next(f for f in report["findings"] if f["name"] == "unreadable_shard")
    assert finding["error"] == "non-object record"
    assert report["target"]["row_count"] == 1
    assert report["passed"] is False


def test_a_manifest_without_completed_at_is_a_finding_not_a_pass(tmp_path) -> None:
    target = corpus(tmp_path, ("a.jsonl", ['{"id":"1"}', '{"id":"2"}']))

    report = run_audit.audit({"target_glob": target, "declared_manifest": manifest_for(tmp_path, completed=False)})

    assert not report["passed"]
    assert any(f["name"] == "manifest_incomplete" for f in report["findings"])


def test_without_a_manifest_no_completeness_is_claimed(tmp_path) -> None:
    target = corpus(tmp_path, ("a.jsonl", ['{"id":"1"}']))

    report = run_audit.audit({"target_glob": target})

    assert report["passed"], "a clean corpus with no manifest is not a failure"
    assert report["completeness"]["claimed"] is False
    assert "informational" in report["completeness"]["reason"]


def test_a_matching_manifest_passes(tmp_path) -> None:
    target = corpus(tmp_path, ("a.jsonl", ['{"id":"1"}', '{"id":"2"}']))

    report = run_audit.audit({"target_glob": target, "declared_manifest": manifest_for(tmp_path)})

    assert report["passed"]
    assert report["completeness"]["claimed"] is True


def test_a_directory_target_excludes_ingest_and_accounting_sidecars(tmp_path) -> None:
    (tmp_path / "part_0.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
    (tmp_path / "ingest_report.json").write_text(
        json.dumps({"counts": {"written": 1}}, indent=2) + "\n", encoding="utf-8"
    )
    (tmp_path / "run_manifest.json").write_text("{}\n", encoding="utf-8")

    report = run_audit.audit({"target_glob": str(tmp_path)})

    assert report["target"]["file_count"] == 1
    assert report["target"]["row_count"] == 1


def test_a_manifest_declaring_more_rows_than_exist_is_a_mismatch(tmp_path) -> None:
    target = corpus(tmp_path, ("a.jsonl", ['{"id":"1"}']))

    report = run_audit.audit({"target_glob": target, "declared_manifest": manifest_for(tmp_path, row_count=999)})

    assert not report["passed"]
    finding = next(f for f in report["findings"] if f["name"] == "manifest_mismatch")
    assert "999" in finding["message"]


def test_containment_without_a_field_choice_fails(tmp_path) -> None:
    target = corpus(tmp_path, ("a.jsonl", ['{"id":"1"}']))
    reference = tmp_path / "ref"
    reference.mkdir()
    (reference / "r.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")

    with pytest.raises(integrity.ContainmentConfigError):
        run_audit.audit(
            {
                "target_glob": target,
                "reference_glob": str(reference / "*.jsonl"),
                "mode": "containment",
                "comparison_fields": [],
            }
        )


def test_the_digest_is_stable_across_enumeration_orders(tmp_path) -> None:
    target = corpus(tmp_path, ("a.jsonl", ['{"id":"1"}']), ("b.jsonl", ['{"id":"2"}']))
    cfg = {"target_glob": target, "mode": "digest", "digest_root": str(tmp_path)}

    first = run_audit.audit(cfg)["target"]["digest"]
    second = run_audit.audit({**cfg, "target_glob": [str(tmp_path / "b.jsonl"), str(tmp_path / "a.jsonl")]})

    assert first == second["target"]["digest"]


# -- reporting rules ----------------------------------------------------------


def test_a_row_delta_alone_is_an_observation_not_a_finding(tmp_path) -> None:
    """Filters remove rows on purpose, so a smaller output is not by itself wrong."""
    target_dir = tmp_path / "out"
    target_dir.mkdir()
    (target_dir / "a.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
    reference_dir = tmp_path / "in"
    reference_dir.mkdir()
    (reference_dir / "a.jsonl").write_text('{"id":"1"}\n{"id":"2"}\n{"id":"3"}\n', encoding="utf-8")

    report = run_audit.audit(
        {"target_glob": str(target_dir / "*.jsonl"), "reference_glob": str(reference_dir / "*.jsonl")}
    )

    assert report["passed"]
    observation = next(o for o in report["observations"] if o["name"] == "reference_delta")
    assert observation["delta"] == 2
    assert report["findings"] == []


def test_an_unmatched_target_glob_is_an_error(tmp_path) -> None:
    """ConfigError, so main() exits 2 with a message like every sibling runner.

    It used to be FileNotFoundError, which no main() in the category catches, so
    the same user mistake gave a raw traceback here and a clean refusal in
    curate/ingest, curate/subset and curate/decontamination.
    """
    with pytest.raises(run_audit.ConfigError, match="nothing-here"):
        run_audit.audit({"target_glob": str(tmp_path / "nothing-here-*.jsonl")})


def test_an_unknown_mode_is_rejected(tmp_path) -> None:
    target = corpus(tmp_path, ("a.jsonl", ['{"id":"1"}']))

    with pytest.raises(run_audit.ConfigError, match="mode must be one of"):
        run_audit.audit({"target_glob": target, "mode": "thorough"})


# -- regressions --------------------------------------------------------------


def test_containment_over_zero_rows_is_not_a_pass(tmp_path) -> None:
    """The false all-clear: a check that compared nothing must not report success."""
    target = tmp_path / "t"
    reference = tmp_path / "r"
    target.mkdir()
    reference.mkdir()
    (target / "a.jsonl").write_text('{"text":"no id here"}\n', encoding="utf-8")
    (reference / "a.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")

    report = run_audit.audit(
        {
            "target_glob": str(target / "*.jsonl"),
            "reference_glob": str(reference / "*.jsonl"),
            "mode": "containment",
            "comparison_fields": ["id"],
        }
    )

    assert report["passed"] is False
    assert any(f["name"] == "containment_unverifiable" for f in report["findings"])
    assert report["containment"]["verifiable"] is False


def test_repeated_ids_are_a_finding_under_reject(tmp_path) -> None:
    target = tmp_path / "t"
    reference = tmp_path / "r"
    target.mkdir()
    reference.mkdir()
    (target / "a.jsonl").write_text('{"id":"1"}\n{"id":"1"}\n', encoding="utf-8")
    (reference / "a.jsonl").write_text('{"id":"1"}\n{"id":"1"}\n', encoding="utf-8")

    report = run_audit.audit(
        {
            "target_glob": str(target / "*.jsonl"),
            "reference_glob": str(reference / "*.jsonl"),
            "mode": "containment",
            "comparison_fields": ["id"],
        }
    )

    assert any(f["name"] == "duplicate_ids" for f in report["findings"])


def test_a_damaged_reference_is_named_before_any_comparison_is_read(tmp_path) -> None:
    """A broken reference makes an intact target look like it gained rows."""
    target = tmp_path / "t"
    reference = tmp_path / "r"
    target.mkdir()
    reference.mkdir()
    (target / "a.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
    (reference / "a.jsonl").write_text('{"id":"1"}\n{"id":"tr', encoding="utf-8")

    report = run_audit.audit({"target_glob": str(target / "*.jsonl"), "reference_glob": str(reference / "*.jsonl")})

    assert any(f["name"] == "unreadable_reference_shard" for f in report["findings"])


def test_a_manifest_that_is_not_an_object_is_reported_not_crashed(tmp_path) -> None:
    (tmp_path / "a.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text('["not", "an", "object"]', encoding="utf-8")

    report = run_audit.audit({"target_glob": str(tmp_path / "*.jsonl"), "declared_manifest": str(bad)})

    assert report["passed"] is False
    assert any(f["name"] == "manifest_mismatch" for f in report["findings"])


def test_a_digest_root_that_is_not_a_parent_is_refused(tmp_path) -> None:
    """Falling back to absolute paths would report 'the corpus changed' after a move."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(integrity.ReleaseLayoutError, match="is not a parent"):
        run_audit.audit({"target_glob": str(corpus / "*.jsonl"), "mode": "digest", "digest_root": str(elsewhere)})


def test_an_empty_but_well_formed_shard_is_an_observation_not_a_finding(tmp_path) -> None:
    """A strict filter can legitimately empty a shard; a killed writer also can."""
    (tmp_path / "full.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")

    report = run_audit.audit({"target_glob": str(tmp_path / "*.jsonl")})

    assert report["passed"], "an empty shard is not by itself a fault"
    observation = next(o for o in report["observations"] if o["name"] == "zero_row_shard")
    assert observation["count"] == 1
    assert observation["paths"][0].endswith("empty.jsonl")


# -- v2: attribution ----------------------------------------------------------
#
# Everything above detects. These cover the only capability that attributes, and
# the reason it needs a producer-emitted artifact to do it.


def write_ledger(tmp_path, name="ledger.json", **kwargs):
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    led = ledger_module.StageLedger(stage=kwargs.pop("stage", "curate/nemo_curator"))
    led.add_input(kwargs.pop("n_input", 0))
    led.add_success(kwargs.pop("n_success", 0))
    for reason, count in (kwargs.pop("filtered", {}) or {}).items():
        led.add_filtered(reason, count)
    for unit, reason, records in kwargs.pop("failed", []) or []:
        led.add_failed(unit, reason, records)
    path = tmp_path / name
    led.write(path, require_balanced=kwargs.pop("require_balanced", True))
    return str(path)


def test_without_a_ledger_the_audit_says_it_cannot_attribute(tmp_path) -> None:
    """v1's honest limit, stated in the report rather than left to be assumed."""
    target = corpus(tmp_path, ("a.jsonl", ['{"id": 1}']))

    report = run_audit.audit({"target_glob": target})

    assert report["attribution"]["available"] is False
    assert "attribute" in report["attribution"]["reason"]


def test_a_ledger_attributes_the_delta_to_the_gates_that_caused_it(tmp_path) -> None:
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}', '{"id": 2}']))
    ledger_path = write_ledger(tmp_path, n_input=5, n_success=2, filtered={"language_id": 2, "word_count": 1})
    declared = manifest_for(tmp_path, row_count=2, file_count=1)
    import json as _json

    document = _json.loads(open(declared).read())
    document["input"]["row_count"] = 5
    open(declared, "w").write(_json.dumps(document))

    report = run_audit.audit({"target_glob": target, "declared_manifest": declared, "ledger_glob": ledger_path})

    attribution = report["attribution"]
    assert attribution["available"] is True
    assert attribution["observed_delta"] == 3
    assert attribution["unexplained"] == 0
    assert attribution["filtered_by_reason"] == {"language_id": 2, "word_count": 1}


def test_manifest_and_ledger_attribution_disagreement_is_a_finding(tmp_path) -> None:
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}', '{"id": 2}']))
    ledger_path = write_ledger(tmp_path, n_input=5, n_success=2, filtered={"language_id": 2, "word_count": 1})
    declared = Path(manifest_for(tmp_path, row_count=2, file_count=1))
    document = json.loads(declared.read_text(encoding="utf-8"))
    document["input"]["row_count"] = 5
    document["declared"] = {"filtered": {"language_id": 1, "word_count": 2}}
    declared.write_text(json.dumps(document), encoding="utf-8")

    report = run_audit.audit(
        {
            "target_glob": target,
            "declared_manifest": str(declared),
            "ledger_glob": ledger_path,
        }
    )

    finding = next(finding for finding in report["findings"] if finding["name"] == "attribution_disagreement")
    assert finding["manifest"] != finding["ledger"]
    assert report["passed"] is False


def test_empty_manifest_attribution_still_disagrees_with_a_nonempty_ledger(tmp_path) -> None:
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}', '{"id": 2}']))
    ledger_path = write_ledger(tmp_path, n_input=3, n_success=2, filtered={"language_id": 1})
    declared = Path(manifest_for(tmp_path, row_count=2, file_count=1))
    document = json.loads(declared.read_text(encoding="utf-8"))
    document["input"]["row_count"] = 3
    document["declared"] = {"filtered": {}}
    declared.write_text(json.dumps(document), encoding="utf-8")

    report = run_audit.audit(
        {
            "target_glob": target,
            "declared_manifest": str(declared),
            "ledger_glob": ledger_path,
        }
    )

    assert any(finding["name"] == "attribution_disagreement" for finding in report["findings"])
    assert report["passed"] is False


def test_loss_no_stage_recorded_is_a_finding(tmp_path) -> None:
    """The capability v1 does not have, on the shape of the original incident."""
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}']))
    ledger_path = write_ledger(tmp_path, n_input=2, n_success=1, filtered={"language_id": 1})
    declared = manifest_for(tmp_path, row_count=1, file_count=1)
    import json as _json

    document = _json.loads(open(declared).read())
    document["input"]["row_count"] = 100
    open(declared, "w").write(_json.dumps(document))

    report = run_audit.audit({"target_glob": target, "declared_manifest": declared, "ledger_glob": ledger_path})

    assert report["attribution"]["unexplained"] == 98
    assert any(f["name"] == "unexplained_loss" for f in report["findings"])
    assert not report["passed"]


def test_a_lost_shard_is_a_finding_even_though_it_reports_zero_rows(tmp_path) -> None:
    """A record-count gate cannot see this; the audit must count units."""
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}']))
    ledger_path = write_ledger(tmp_path, n_input=1, n_success=1, failed=[("shard_42.jsonl", "truncated", 0)])

    report = run_audit.audit({"target_glob": target, "ledger_glob": ledger_path})

    finding = next(f for f in report["findings"] if f["name"] == "lost_units")
    assert "shard_42.jsonl" in finding["message"]
    assert "FLOOR" in finding["message"]


def test_an_imbalanced_ledger_is_a_finding(tmp_path) -> None:
    """A stage that reported success with records unaccounted for."""
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}']))
    ledger_path = write_ledger(tmp_path, n_input=100, n_success=1, require_balanced=False)

    report = run_audit.audit({"target_glob": target, "ledger_glob": ledger_path})

    assert any(f["name"] == "ledger_imbalanced" for f in report["findings"])


def test_an_unreadable_ledger_is_a_finding_not_a_crash(tmp_path) -> None:
    (tmp_path / "bad-ledger.json").write_text("{not json", encoding="utf-8")
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}']))

    report = run_audit.audit({"target_glob": target, "ledger_glob": str(tmp_path / "bad-ledger.json")})

    assert any(f["name"] == "ledger_unreadable" for f in report["findings"])


def test_ledgers_without_an_input_count_report_rather_than_conclude(tmp_path) -> None:
    """No observed delta means nothing to check the declarations against."""
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}']))
    ledger_path = write_ledger(tmp_path, n_input=5, n_success=5)

    report = run_audit.audit({"target_glob": target, "ledger_glob": ledger_path})

    assert report["attribution"]["unexplained"] is None
    assert "declared_manifest" in report["attribution"]["note"]


def test_several_ledgers_merge_across_stages(tmp_path) -> None:
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}']))
    write_ledger(tmp_path, "led-a.json", stage="curate/nemo_curator", n_input=3, n_success=1, filtered={"lang": 2})
    write_ledger(tmp_path, "led-b.json", stage="curate/subset", n_input=1, n_success=1)

    report = run_audit.audit({"target_glob": target, "ledger_glob": str(tmp_path / "led-*.json")})

    assert report["attribution"]["ledgers"] == 2
    assert report["attribution"]["stages"] == ["curate/nemo_curator", "curate/subset"]


def test_rows_appearing_is_reported_as_a_gain_not_a_negative_loss(tmp_path) -> None:
    """A negative loss count states the reverse of what happened.

    Filtering cannot make rows appear, so "-838 records left the pipeline" gives
    an operator no route to the real diagnosis: a retried write, a duplicated
    shard, an output directory that was not empty.
    """
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}', '{"id": 2}', '{"id": 3}']))
    ledger_path = write_ledger(tmp_path, n_input=10, n_success=4, filtered={"lang": 6})
    declared = manifest_for(tmp_path, row_count=3, file_count=1)
    import json as _json

    document = _json.loads(open(declared).read())
    document["input"]["row_count"] = 4
    open(declared, "w").write(_json.dumps(document))

    report = run_audit.audit({"target_glob": target, "declared_manifest": declared, "ledger_glob": ledger_path})

    names = [f["name"] for f in report["findings"]]
    assert "unaccounted_gain" in names
    assert "unexplained_loss" not in names
    gain = next(f for f in report["findings"] if f["name"] == "unaccounted_gain")
    assert "more record(s) than" in gain["message"]


def test_the_attribution_says_it_is_first_gate_wins(tmp_path) -> None:
    """Gates short-circuit, so a per-reason count is not that gate's total."""
    target = corpus(tmp_path, ("out.jsonl", ['{"id": 1}']))
    ledger_path = write_ledger(tmp_path, n_input=3, n_success=1, filtered={"a": 1, "b": 1})

    report = run_audit.audit({"target_glob": target, "ledger_glob": ledger_path})

    assert "FIRST gate that rejected it" in report["attribution"]["attribution_note"]


def test_every_finding_name_is_still_documented() -> None:
    """Re-checked here because this file just added a finding."""
    test_every_finding_name_is_documented_as_an_error()


def test_the_producer_and_the_consumer_meet(tmp_path, monkeypatch) -> None:
    """curate/nemo_curator writes a ledger; curate/audit attributes a loss with it.

    Lives here rather than beside the producer because it needs both steps, and a
    test that spans two branches belongs to the later one — otherwise the earlier
    branch cannot be merged on its own.
    """
    import importlib
    import sys
    import types

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
    sys.modules["nemo_curator.core.client"].RayClient = object
    sys.modules["nemo_curator.pipeline"].Pipeline = object
    sys.modules["nemo_curator.stages.text.io.reader"].JsonlReader = object
    sys.modules["nemo_curator.stages.text.io.writer"].JsonlWriter = object
    sys.modules["huggingface_hub"].snapshot_download = lambda **kwargs: None
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    producer = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)

    src_dir, out_dir = tmp_path / "in", tmp_path / "out"
    src_dir.mkdir()
    out_dir.mkdir()

    def row(i: int) -> str:
        return json.dumps({"id": f"d{i}", "source": "s", "text": "x"})

    (src_dir / "a.jsonl").write_text("".join(row(i) + "\n" for i in range(100)), encoding="utf-8")
    (out_dir / "a.jsonl").write_text("".join(row(i) + "\n" for i in range(82)), encoding="utf-8")

    cfg = {
        "input_glob": str(src_dir / "*.jsonl"),
        "output_dir": str(out_dir),
        "text_field": "text",
        "id_field": "id",
        "source_field": "source",
    }
    ledger_path = tmp_path / "ledger.json"
    producer.emit_ledger(cfg, str(ledger_path), completed=True)

    report = run_audit.audit(
        {
            "target_glob": str(out_dir / "*.jsonl"),
            "reference_glob": cfg["input_glob"],
            "ledger_glob": str(ledger_path),
            "source_field": "source",
        }
    )

    assert report["attribution"]["available"] is True
    assert report["attribution"]["observed_delta"] == 18
    assert report["attribution"]["unexplained"] == 0
