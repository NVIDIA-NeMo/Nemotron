# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""curate/flow: one config, six steps, and the things it must refuse.

The value of a flow is not that it saves typing — it is that two agreements
which fail silently by hand (the manifest path the audit compares against, and
the score column subset stratifies on) become impossible to get wrong. Most of
these tests are about refusals, because a flow that runs and produces a
clean-looking result from a broken configuration is worse than six commands.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

from nemotron.steps.curate.nemo_curator.runtime import policy as policy_module
from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry
from nemotron.steps.curate.nemo_curator.scripts import run_flow

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "nemo_curator")
LANGPACK_FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"
PACKAGE_PACKS = STEP_DIR / "data" / "langpacks"


def corpus_files(tmp_path: Path, n: int = 40) -> str:
    directory = tmp_path / "raw"
    directory.mkdir(exist_ok=True)
    words = "curation tokens budget corpus stratum nesting document policy".split()
    with (directory / "part_0.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            text = " ".join(words[(i + j) % len(words)] for j in range(30)) + "."
            fh.write(json.dumps({"id": f"d{i:03d}", "source": "web", "text": text}) + "\n")
    return str(directory / "*.jsonl")


def config(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "corpus": {
            "input": corpus_files(tmp_path),
            "text_field": "text",
            "id_field": "id",
            "source_field": "source",
            "language": "x-test-vi",
            "langpack_dir": str(LANGPACK_FIXTURES),
        },
        "output_root": str(tmp_path / "out"),
        "steps": {
            "profile": {"enabled": True, "signals": ["unicode_alpha_numeric"], "max_total_docs": 40},
            "filter": {"enabled": False},
            "audit": {"enabled": False},
            "subset": {"enabled": False},
            "decontamination": {"enabled": False},
        },
        "approve": None,
    }
    cfg.update(overrides)
    return cfg


# -- static -------------------------------------------------------------------


def test_flow_step_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/nemo_curator",
        expected_launch="python",
        expected_default_config="default",
    )


def test_the_flow_declares_no_gpu() -> None:
    """Five of the six steps need none; decontamination asks per-run instead."""
    source = (STEP_DIR / "step.py").read_text(encoding="utf-8")

    assert "gpus_per_node = 0" in source


def test_the_default_config_covers_every_step_in_the_plan() -> None:
    """A step missing from the shipped config is one nobody knows they can enable."""
    cfg = yaml.safe_load((STEP_DIR / "config" / "vi_c4_measure.yaml").read_text(encoding="utf-8"))

    assert set(cfg["steps"]) == {plan.key for plan in run_flow.STEP_ORDER}


def test_the_manifest_path_is_derived_for_both_producer_and_consumer(tmp_path) -> None:
    """The agreement that fails silently by hand.

    curate/nemo_curator ships emit_manifest: null and curate/audit ships
    declared_manifest: null. Two nulls in two files: an audit against a producer
    that emitted no manifest claims nothing, which reads as a clean result.
    """
    resolved, paths = run_flow.derive(config(tmp_path))
    by_key = {r.plan.key: r.config for r in resolved}

    assert by_key["filter"]["emit_manifest"] == paths["manifest"]
    assert by_key["audit"]["declared_manifest"] == paths["manifest"]
    assert by_key["filter"]["emit_ledger"] == by_key["audit"]["ledger_glob"]


def test_profile_reads_the_unfiltered_corpus_and_the_rest_read_the_output(tmp_path) -> None:
    """Profiling the filtered output measures gates that have already run."""
    cfg = config(tmp_path)
    resolved, paths = run_flow.derive(cfg)
    by_key = {r.plan.key: r.config for r in resolved}

    assert by_key["profile"]["input_glob"] == cfg["corpus"]["input"]
    assert by_key["profile"]["language"] == cfg["corpus"]["language"]
    assert by_key["profile"]["langpack_dir"] == cfg["corpus"]["langpack_dir"]
    assert by_key["filter"]["input_glob"] == cfg["corpus"]["input"]
    for key in ("subset",):
        assert by_key[key]["input_glob"].startswith(paths["corpus"])
    assert by_key["audit"]["target_glob"].startswith(paths["corpus"])
    assert by_key["decontamination"]["train_glob"].startswith(paths["corpus"])


def test_ingest_hands_canonical_field_names_to_every_consumer(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["corpus"].update(
        {
            "text_field": "body",
            "id_field": "doc_key",
            "id_field_in_source": True,
            "source_field": "dataset",
            "source_field_in_source": "origin",
            "metadata_fields": ["doc_key", "origin"],
        }
    )
    cfg["steps"]["ingest"] = {"enabled": True}

    resolved, _ = run_flow.derive(cfg)
    by_key = {r.plan.key: r.config for r in resolved}

    assert by_key["ingest"]["text_field"] == "body"
    assert by_key["ingest"]["id_from"] == "doc_key"
    assert by_key["ingest"]["source_from"] == "origin"
    for key in ("profile", "filter", "subset"):
        assert by_key[key]["text_field"] == "text"
        assert by_key[key]["id_field"] == "id"
        assert by_key[key]["source_field"] == "source"
    assert by_key["decontamination"]["text_field"] == "text"
    assert by_key["decontamination"]["id_field"] == "id"
    assert {"id", "source"} <= set(by_key["filter"]["metadata_fields"])


def test_an_authored_key_overrides_the_derivation(tmp_path) -> None:
    """A flow config that cannot be escaped needs a second file, which defeats it."""
    cfg = config(tmp_path)
    cfg["steps"]["subset"]["output_dir"] = "/somewhere/else"

    resolved, _ = run_flow.derive(cfg)
    subset = next(r for r in resolved if r.plan.key == "subset")

    assert subset.config["output_dir"] == "/somewhere/else"


def test_a_corpus_without_an_input_is_refused(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["corpus"].pop("input")

    with pytest.raises(run_flow.FlowConfigError, match="corpus.input is required"):
        run_flow.derive(cfg)


# -- preflight ----------------------------------------------------------------


def test_a_step_whose_producer_is_disabled_and_absent_is_refused(tmp_path) -> None:
    """Naming the disabled producer is the point: the alternative is a step
    reporting an empty-looking result nobody can trace."""
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True

    with pytest.raises(run_flow.FlowConfigError) as excinfo:
        resolved, paths = run_flow.derive(cfg)
        run_flow.preflight(cfg, resolved, paths)

    message = str(excinfo.value)
    assert "steps.filter" in message
    assert "disabled" in message


def test_a_previous_runs_artifact_is_reused_when_the_producer_is_disabled(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)

    # Materialise what a previous run would have left behind.
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    Path(paths["manifest"]).write_text("{}", encoding="utf-8")
    Path(paths["ledger"]).write_text("{}", encoding="utf-8")

    run_flow.preflight(cfg, resolved, paths)
    audit = next(r for r in resolved if r.plan.key == "audit")

    assert any("reusing" in note for note in audit.notes)


def test_nothing_enabled_is_refused(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["profile"]["enabled"] = False

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="no steps are enabled"):
        run_flow.preflight(cfg, resolved, paths)


def test_stratifying_on_a_column_the_filter_will_not_write_is_refused(tmp_path) -> None:
    """mode: filter discards scores after use, so the column never exists."""
    cfg = config(tmp_path)
    cfg["steps"]["filter"] = {"enabled": True, "mode": "filter"}
    cfg["steps"]["subset"] = {"enabled": True, "quality_score_field": "__stopword_ratio"}

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="quality_score_field"):
        run_flow.preflight(cfg, resolved, paths)


def test_mode_alone_does_not_satisfy_the_score_column_requirement(tmp_path) -> None:
    """The score columns are written by the policy's signals, not by the mode.

    With no policy there are no signals, so annotate/both write nothing — and
    that is precisely the documented first run. A check that looked only at
    mode would green-light the one configuration it exists to refuse.
    """
    cfg = config(tmp_path)
    cfg["steps"]["filter"] = {"enabled": True, "mode": "both"}
    cfg["steps"]["subset"] = {"enabled": True, "quality_score_field": "__stopword_ratio"}

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="no policy is configured"):
        run_flow.preflight(cfg, resolved, paths)


def test_a_policy_plus_annotate_mode_satisfies_it(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["filter"] = {
        "enabled": True,
        "mode": "both",
        "heuristic_filters": {"approved_policy": "./policy.yaml"},
    }
    cfg["steps"]["subset"] = {"enabled": True, "quality_score_field": "__stopword_ratio"}

    resolved, paths = run_flow.derive(cfg)

    run_flow.preflight(cfg, resolved, paths)  # must not raise


def test_an_audit_without_a_ledger_is_warned_not_silently_uninformative(tmp_path) -> None:
    """attribution.available: false means 'nobody recorded why', not 'nothing left'."""
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    Path(paths["manifest"]).write_text("{}", encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("attribution.available: false" in w for w in warnings)


def test_decontamination_without_a_holdout_is_refused(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["decontamination"] = {"enabled": True}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")

    with pytest.raises(run_flow.FlowConfigError, match="holdout"):
        run_flow.preflight(cfg, resolved, paths)


def test_preflight_refuses_before_any_step_runs(tmp_path) -> None:
    """A flow that fails after the filter rewrote the corpus is worse than five commands."""
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True

    with pytest.raises(run_flow.FlowConfigError):
        run_flow.run(cfg)

    assert not (Path(cfg["output_root"]) / "profile").exists()


def test_profile_preflight_requires_an_external_language_pack(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["corpus"]["langpack_dir"] = None
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="explicit langpack_dir"):
        run_flow.preflight(cfg, resolved, paths)


# -- the approval gate --------------------------------------------------------


def swept_threshold(minimum: float = 0.25) -> float:
    return float(next(v for v in signal_registry.SIGNALS["unicode_alpha_numeric"].grid.values() if v >= minimum))


def approve_block(**overrides) -> dict:
    block = {
        "thresholds": [{"signal": "unicode_alpha_numeric", "max": swept_threshold()}],
        "approver": "someone@example.test",
        "date": "2026-08-26",
        "method": "manual",
        "evidence": "retention curve reviewed against the sampled documents",
    }
    block.update(overrides)
    return block


def profiled(tmp_path: Path) -> dict:
    """A config whose profile has already run, so candidates exist."""
    cfg = config(tmp_path)
    run_flow.run(cfg)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    return cfg


def test_approving_before_profiling_is_refused(tmp_path) -> None:
    """There is nothing to approve until the corpus has been measured."""
    cfg = config(tmp_path)
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="does not exist"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_profile_cannot_replace_candidates_in_the_approval_run(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["steps"]["profile"]["enabled"] = True
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="Profiling and approval are two runs"):
        run_flow.preflight(cfg, resolved, paths)


def test_an_approval_produces_an_executable_policy_wired_into_the_filter(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    run_flow.materialise_policy(cfg, resolved, paths)

    document = yaml.safe_load(Path(paths["approved_policy"]).read_text(encoding="utf-8"))
    assert document["approved"] is True
    assert policy_module.validate_approved_policy(document) == []
    filter_cfg = next(r for r in resolved if r.plan.key == "filter").config
    assert filter_cfg["heuristic_filters"]["approved_policy"] == paths["approved_policy"]
    assert filter_cfg["heuristic_filters"]["langpack_dir"] == str(LANGPACK_FIXTURES)


@pytest.mark.parametrize("pack_root", [None, "bundled"])
def test_pack_backed_approval_refuses_without_the_profiled_pack_root(tmp_path, pack_root) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["profile"]["signals"] = ["stopword_ratio"]
    run_flow.run(cfg)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["corpus"]["langpack_dir"] = pack_root
    threshold = next(value for value in signal_registry.SIGNALS["stopword_ratio"].grid.values() if value > 0)
    cfg["approve"] = approve_block(thresholds=[{"signal": "stopword_ratio", "min": threshold}])
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="no langpack_dir"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_approve_refuses_a_conflicting_per_step_policy_path(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()
    cfg["steps"]["filter"]["heuristic_filters"] = {"approved_policy": str(tmp_path / "other.yaml")}
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="cannot truthfully report which policy"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_an_approval_granted_against_other_data_is_refused(tmp_path) -> None:
    """The failure a shared config file invites: thresholds AND the signature
    that approved them, applied to a corpus nobody approved them for."""
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()

    other = tmp_path / "other"
    other.mkdir()
    rows = [{"id": f"o{i}", "source": "web", "text": "different corpus entirely here"} for i in range(20)]
    (other / "b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg["corpus"]["input"] = str(other / "*.jsonl")
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="granted against corpus"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_the_fingerprint_check_can_be_waived_only_explicitly(tmp_path) -> None:
    """Waivable, but never by accident, and the README says not to."""
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block(verify_corpus=False)
    other = tmp_path / "other2"
    other.mkdir()
    (other / "b.jsonl").write_text(
        json.dumps({"id": "o1", "source": "web", "text": "quite different text here"}) + "\n",
        encoding="utf-8",
    )
    cfg["corpus"]["input"] = str(other / "*.jsonl")
    resolved, paths = run_flow.derive(cfg)

    run_flow.materialise_policy(cfg, resolved, paths)  # must not raise

    assert Path(paths["approved_policy"]).is_file()


def test_an_approval_with_nobody_named_is_refused(tmp_path) -> None:
    """The flow writes the artifact that claims approval, so it is refused here too.

    This test used to assert the opposite. The machine-checkable fields -- corpus
    fingerprint, profile digest, scorer version, bound direction -- still do the
    work of refusing a wrong run and none of them needs a name. What a name is
    for is that `approved: true` asserts a person decided; written with nobody
    named, the manifest and the flow report repeat an assertion no one made.
    """
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block(approver=None, date=None, evidence=None)
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(Exception, match="approver"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_an_approval_that_names_someone_is_written(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block(approver="reviewer@example.test", evidence="read 200 dropped documents")
    resolved, paths = run_flow.derive(cfg)

    run_flow.materialise_policy(cfg, resolved, paths)

    written = yaml.safe_load(Path(paths["approved_policy"]).read_text(encoding="utf-8"))
    assert written["approved"] is True
    assert written["approval"]["approver"] == "reviewer@example.test"


def test_an_approval_with_no_thresholds_is_refused(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block(thresholds=[])
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match="gates nothing"):
        run_flow.materialise_policy(cfg, resolved, paths)


def test_an_approval_nothing_applies_is_warned(tmp_path) -> None:
    cfg = profiled(tmp_path)
    cfg["steps"]["filter"]["enabled"] = False
    cfg["steps"]["subset"] = {"enabled": True}
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True, exist_ok=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("nothing applies it" in w for w in warnings)


def test_language_keep_list_warns_when_corpus_language_is_absent() -> None:
    warnings = run_flow._language_keep_list_warnings(  # noqa: SLF001 - focused preflight contract
        {"steps": {"filter": {"language_codes": ["HI"]}}}
    )

    assert any("corpus.language is unset" in warning for warning in warnings)


def test_invalid_language_code_is_refused_during_flow_preflight(tmp_path, monkeypatch) -> None:
    import sys
    import types

    model_path = tmp_path / "lid.176.bin"
    model_path.write_bytes(b"model")
    model = types.SimpleNamespace(get_labels=lambda: ["__label__en", "__label__hi"])
    fasttext = types.ModuleType("fasttext")
    fasttext.load_model = lambda _path: model
    monkeypatch.setitem(sys.modules, "fasttext", fasttext)

    cfg = config(tmp_path)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"] = {
        "enabled": True,
        "language_codes": ["HIN"],
        "models": {"fasttext_langid": str(model_path)},
    }
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises(run_flow.FlowConfigError, match=r"HIN.*emits none"):
        run_flow.preflight(cfg, resolved, paths)


def test_the_flow_is_not_a_second_approver() -> None:
    """runtime/policy.py::promote must remain the only one.

    Enforced repo-wide by test_policy.py; asserted here too because a flow is
    exactly where a convenience shortcut would be added.
    """
    import inspect

    source = inspect.getsource(run_flow)

    assert "promote(" in source, "the flow must go through promote()"
    assert '"approved": True' not in source
    assert "'approved': True" not in source


# -- running ------------------------------------------------------------------


def test_a_run_writes_a_plan_and_a_report(tmp_path) -> None:
    cfg = config(tmp_path)

    report = run_flow.run(cfg)

    root = Path(cfg["output_root"])
    assert (root / "flow_plan.json").is_file()
    assert (root / "flow_report.json").is_file()
    assert report["step_id"] == "curate/flow"


def test_a_preflight_refusal_preserves_the_previous_plan_and_report(tmp_path) -> None:
    cfg = config(tmp_path)
    run_flow.run(cfg)
    root = Path(cfg["output_root"])
    old_plan = (root / "flow_plan.json").read_bytes()
    old_report = (root / "flow_report.json").read_bytes()
    cfg["steps"]["profile"]["enabled"] = False

    with pytest.raises(run_flow.FlowConfigError, match="no steps are enabled"):
        run_flow.run(cfg)

    assert (root / "flow_plan.json").read_bytes() == old_plan
    assert (root / "flow_report.json").read_bytes() == old_report


def test_an_approval_refusal_preserves_the_previous_plan_and_report(tmp_path) -> None:
    cfg = profiled(tmp_path)
    root = Path(cfg["output_root"])
    old_plan = (root / "flow_plan.json").read_bytes()
    old_report = (root / "flow_report.json").read_bytes()
    cfg["approve"] = approve_block(thresholds=[])

    with pytest.raises(run_flow.FlowConfigError, match="gates nothing"):
        run_flow.run(cfg)

    assert (root / "flow_plan.json").read_bytes() == old_plan
    assert (root / "flow_report.json").read_bytes() == old_report


def test_the_plan_records_every_derived_config_before_anything_runs(tmp_path) -> None:
    """A plan you can only read after the corpus was rewritten cannot reject it."""
    cfg = config(tmp_path)
    run_flow.run(cfg)

    plan = json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))

    assert {s["key"] for s in plan["steps"]} == {p.key for p in run_flow.STEP_ORDER}
    profile = next(s for s in plan["steps"] if s["key"] == "profile")
    assert profile["config"]["input_glob"] == cfg["corpus"]["input"]


def test_disabled_steps_are_reported_as_disabled_not_omitted(tmp_path) -> None:
    """A step missing from the report is indistinguishable from one that failed."""
    report = run_flow.run(config(tmp_path))

    statuses = {s["step_id"]: s["status"] for s in report["steps"]}
    assert statuses["curate/profile"] == "ok"
    assert statuses["curate/nemo_curator"] == "disabled"
    assert len(statuses) == len(run_flow.STEP_ORDER)


def test_the_enabled_step_actually_produced_its_artifacts(tmp_path) -> None:
    cfg = config(tmp_path)

    run_flow.run(cfg)

    profile_dir = Path(cfg["output_root"]) / "profile"
    assert (profile_dir / "profile_report.json").is_file()
    assert (profile_dir / "candidate_policies.yaml").is_file()


def test_custom_raw_fields_run_from_ingest_through_profile(tmp_path) -> None:
    raw_dir = tmp_path / "custom_raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "part.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps({"doc_key": f"k{i}", "origin": "web", "body": f"document {i} has useful text"}) + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )
    cfg = config(tmp_path)
    cfg["corpus"] = {
        "input": str(raw_path),
        "text_field": "body",
        "id_field": "doc_key",
        "id_field_in_source": True,
        "source_field": "dataset",
        "source_field_in_source": "origin",
        "language": "x-test-vi",
        "langpack_dir": str(LANGPACK_FIXTURES),
        "metadata_fields": ["doc_key", "origin"],
    }
    cfg["steps"]["ingest"] = {"enabled": True}

    report = run_flow.run(cfg)

    ingested = [
        json.loads(line)
        for line in (Path(cfg["output_root"]) / "ingested" / "part_0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    profile = json.loads((Path(cfg["output_root"]) / "profile" / "profile_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert profile["corpus"]["document_count"] == 4
    assert profile["corpus"]["source_count"] == 1
    assert {(row["id"], row["source"], row["text"]) for row in ingested} == {
        (f"k{i}", "web", f"document {i} has useful text") for i in range(4)
    }


def test_flow_cpu_chain_preserves_manifest_and_ledger_identities(tmp_path, monkeypatch) -> None:
    """Run the real audit against a CPU producer stub through the flow seam."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity, ledger
    from nemotron.steps.curate.nemo_curator.runtime import manifest as manifest_module

    cfg = config(tmp_path)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["steps"]["audit"]["enabled"] = True
    real_step_runner = run_flow.step_runner
    filter_config: dict = {}

    def cpu_filter(step_cfg):
        filter_config.update(step_cfg)
        inputs = integrity.expand_inputs(step_cfg["input_glob"])
        output = Path(step_cfg["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        shard = output / "part_0.jsonl"
        shard.write_text(
            "".join(Path(path).read_text(encoding="utf-8") for path in inputs),
            encoding="utf-8",
        )
        input_counts = manifest_module.count_jsonl(inputs)
        output_counts = manifest_module.count_jsonl([shard])
        document = manifest_module.build_manifest(
            step_id="curate/nemo_curator",
            config=step_cfg,
            started_at="2026-08-25T00:00:00+00:00",
            completed_at="2026-08-25T00:00:01+00:00",
            input_glob=step_cfg["input_glob"],
            input_counts=input_counts,
            output_counts=output_counts,
            id_field=step_cfg["id_field"],
            source_field=step_cfg["source_field"],
        )
        manifest_module.write_manifest(step_cfg["emit_manifest"], document)
        accounting = ledger.StageLedger(stage="curate/nemo_curator")
        accounting.add_input(input_counts["row_count"])
        accounting.add_success(output_counts["row_count"])
        accounting.write(step_cfg["emit_ledger"], require_balanced=True)
        return {"warnings": []}

    def runner(key):
        return cpu_filter if key == "filter" else real_step_runner(key)

    monkeypatch.setattr(run_flow, "step_runner", runner)

    report = run_flow.run(cfg)

    audit_config = next(
        step["config"]
        for step in json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))["steps"]
        if step["key"] == "audit"
    )
    assert audit_config["declared_manifest"] == filter_config["emit_manifest"]
    assert audit_config["ledger_glob"] == filter_config["emit_ledger"]
    assert Path(audit_config["declared_manifest"]).is_file()
    assert Path(audit_config["ledger_glob"]).is_file()
    assert report["audit_passed"] is True


def test_run_does_not_exit_the_process(tmp_path) -> None:
    """The uniform seam: a caller decides what a failure means."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_flow.run)))
    raises = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Raise)
        and isinstance(getattr(n.exc, "func", n.exc), ast.Name)
        and getattr(n.exc, "func", n.exc).id == "SystemExit"
    ]

    assert not raises


def test_the_step_order_is_a_valid_dependency_order() -> None:
    """Every artifact a step needs is produced by a step earlier in the list."""
    produced: set[str] = set()
    for plan in run_flow.STEP_ORDER:
        missing = set(plan.needs) - produced
        assert not missing, f"{plan.key} needs {sorted(missing)} before anything produces it"
        produced.update(plan.produces)


def test_every_error_the_flow_raises_is_documented() -> None:
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    documented = {e["name"] for e in manifest.get("errors", [])}

    assert {
        "missing_upstream_artifact",
        "score_column_never_written",
        "approval_corpus_mismatch",
        "approve_before_profile",
        "profile_enabled_during_approval",
        "missing_langpack_dir",
        "conflicting_approved_policy",
        "no_steps_enabled",
        "unknown_step",
        "stale_filter_output",
        "ingest_source_field_unmapped",
    } <= documented


# -- the worked examples ------------------------------------------------------
#
# Shipped beside default and tiny, the way sdg/data_designer ships rl_pref and
# customer_support_tools. An example that stops parsing is worse than none: it
# is the first thing a new user copies.

# One shipped worked example, in the two halves the pipeline actually runs in:
# vi_c4_measure profiles the corpus, vi_c4_apply gates it on what that measured.
# en_c4 and hi_sangraha were dropped with the configs they read; vi_c4 is the
# corpus this pipeline was validated on.
EXAMPLE_CONFIGS = ("vi_c4_measure", "vi_c4_apply")


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_derives_six_step_configs(name) -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))

    resolved, _ = run_flow.derive(cfg)

    assert {r.plan.key for r in resolved} == {p.key for p in run_flow.STEP_ORDER}


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_is_reachable_by_name(name) -> None:
    """``-c vi_c4_measure`` resolves against the step's own config dir, so it must be there."""
    assert (STEP_DIR / "config" / f"{name}.yaml").is_file()


def test_the_measure_example_ships_unapproved() -> None:
    """A shipped config that filters on someone else's thresholds is the trap."""
    cfg = yaml.safe_load((STEP_DIR / "config" / "vi_c4_measure.yaml").read_text(encoding="utf-8"))

    assert cfg["approve"] is None
    assert cfg["steps"]["profile"]["enabled"], "run 1 exists to measure"
    assert cfg["steps"]["subset"]["quality_score_field"] is None, (
        "no policy yet, so steps.filter writes no __<signal> column to stratify on"
    )


def test_the_apply_example_shows_a_filled_approve_block() -> None:
    """The half a reader needs and cannot get from a config that ships approve: null."""
    cfg = yaml.safe_load((STEP_DIR / "config" / "vi_c4_apply.yaml").read_text(encoding="utf-8"))

    assert not cfg["steps"]["profile"]["enabled"], "measured already; this run applies"
    thresholds = cfg["approve"]["thresholds"]
    assert thresholds
    assert cfg["approve"]["approver"] == "you@example.com", (
        "a placeholder, never a real name: a shipped approval nobody made is the trap"
    )
    signals = {t["signal"] for t in thresholds}
    assert {"min", "max"} <= {key for t in thresholds for key in t if key != "signal"}, (
        "the example must show both bound directions, not just max"
    )
    column = cfg["steps"]["subset"]["quality_score_field"]
    assert column.removeprefix("__") in signals, "quality_score_field must name a column the policy actually produces"
    assert cfg["steps"]["filter"]["mode"] in ("annotate", "both"), (
        "under mode 'filter' the scores are discarded and the column never lands"
    )


def test_the_two_halves_describe_the_same_corpus() -> None:
    """The approval carries a corpus fingerprint; a drift here is what it refuses."""
    measure = yaml.safe_load((STEP_DIR / "config" / "vi_c4_measure.yaml").read_text(encoding="utf-8"))
    apply_ = yaml.safe_load((STEP_DIR / "config" / "vi_c4_apply.yaml").read_text(encoding="utf-8"))

    assert measure["corpus"] == apply_["corpus"]
    assert measure["output_root"] == apply_["output_root"]


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_names_a_language_and_explicit_pack_root(name) -> None:
    """Examples must not fall back to private test data or an implicit pack."""
    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))

    assert cfg["corpus"]["language"]
    assert cfg["corpus"]["langpack_dir"] not in (None, "", "bundled")


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_points_at_a_pack_root_that_has_its_language(name) -> None:
    """The example must be runnable as shipped.

    This used to assert the literal "./langpacks" -- an external directory no
    user has -- while the config declared `language: vi` and only `en` shipped.
    Running it gave LanguagePackNotFoundError, and the suite stayed green because
    the sibling test substituted the x-test-vi fixture. Pinning the property
    rather than the string is what makes that unrepeatable.
    """
    from nemotron.steps.curate.nemo_curator.runtime import langpack

    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["corpus"]["langpack_dir"])

    assert cfg["corpus"]["langpack_dir"], "langpack_dir has no default and must be explicit"
    assert langpack.load(cfg["corpus"]["language"], root).language_tag == cfg["corpus"]["language"]


@pytest.mark.parametrize("name", EXAMPLE_CONFIGS)
def test_the_worked_example_only_names_signals_its_pack_supports(name) -> None:
    """The check the hi example exists to demonstrate.

    The hi pack declares neither diacritic_ratio nor stopword_ratio_folded —
    Devanagari matras are obligatory vowels, so stripping them yields nonsense.
    Naming one is a hard error at runtime, so an example that named one would
    ship broken.
    """
    from nemotron.steps.curate.nemo_curator.runtime import langpack

    cfg = yaml.safe_load((STEP_DIR / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
    language = cfg["corpus"]["language"]
    # The SHIPPED pack for the language the config names -- not the x-test-*
    # fixture. Substituting the fixture validated the worked example against a
    # pack no user has: vi_c4_measure.yaml declared `language: vi` while only
    # `en` shipped, so the example could not be run and this test still passed.
    pack = langpack.load(language, PACKAGE_PACKS)
    named = cfg["steps"]["profile"].get("signals") or []

    for signal_name in named:
        required = set(signal_registry.SIGNALS[signal_name].requires)
        assert required <= set(pack.capabilities), (
            f"{name}.yaml names {signal_name}, which the {cfg['corpus']['language']} pack "
            f"cannot support (needs {sorted(required)})"
        )


def test_an_absent_optional_artifact_is_unset_not_merely_warned(tmp_path) -> None:
    """curate/audit reads declared_manifest unconditionally.

    Leaving the derived path in place turns the case preflight has just called
    legitimate into a FileNotFoundError deep inside the audit. Standalone
    curate/audit ships these as null and degrades cleanly; the flow has to
    reproduce that behaviour, not just describe it in a warning.
    """
    cfg = config(tmp_path)
    cfg["steps"]["audit"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")

    run_flow.preflight(cfg, resolved, paths)
    audit = next(r for r in resolved if r.plan.key == "audit").config

    assert audit["declared_manifest"] is None
    assert audit["ledger_glob"] is None


def test_a_misspelled_step_name_is_refused(tmp_path) -> None:
    """Otherwise the flow reports success having never attempted the stage."""
    cfg = config(tmp_path)
    cfg["steps"]["decontaminate"] = {"enabled": True}  # the real key is decontamination

    resolved, paths = run_flow.derive(cfg)
    with pytest.raises(run_flow.FlowConfigError, match="unknown step"):
        run_flow.preflight(cfg, resolved, paths)


def test_reusing_a_corpus_from_a_run_that_died_is_warned(tmp_path) -> None:
    """A corpus left by a killed writer parses cleanly and is simply short."""
    cfg = config(tmp_path)
    cfg["steps"]["subset"] = {"enabled": True}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x", "text": "y"}\n', encoding="utf-8")
    # A manifest with no completed_at is how an auditor learns the run died.
    Path(paths["manifest"]).write_text(json.dumps({"producer": {"started_at": "t"}}), encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("did not reach its write barrier" in w for w in warnings)


def test_reusing_a_corpus_with_no_manifest_at_all_is_warned(tmp_path) -> None:
    cfg = config(tmp_path)
    cfg["steps"]["subset"] = {"enabled": True}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text('{"id": "x", "text": "y"}\n', encoding="utf-8")

    warnings = run_flow.preflight(cfg, resolved, paths)

    assert any("nothing to say whether the run that wrote it finished" in w for w in warnings)


def test_a_step_failing_still_writes_a_report_saying_how_far_it_got(tmp_path, monkeypatch) -> None:
    """Otherwise the half-written artifacts sit next to a PREVIOUS run's report."""
    cfg = config(tmp_path)

    def boom(_cfg):
        raise RuntimeError("profile exploded")

    monkeypatch.setattr(run_flow, "step_runner", lambda key: boom)

    with pytest.raises(RuntimeError, match="profile exploded"):
        run_flow.run(cfg)

    report = json.loads((Path(cfg["output_root"]) / "flow_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    failed = next(s for s in report["steps"] if s["status"] == "failed")
    assert failed["step_id"] == "curate/profile"
    assert "profile exploded" in failed["error"]


def test_policy_applied_is_false_when_nothing_applied_it(tmp_path) -> None:
    """Promoting a policy is not applying one, and the report must not conflate them."""
    cfg = config(tmp_path)
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["corpus"]["id_field_in_source"] = True
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)  # produce both candidates and the prepared corpus
    cfg["approve"] = approve_block()
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = False

    report = run_flow.run(cfg)

    assert report["policy_promoted"] is True
    assert report["policy_applied"] is False


def test_each_steps_own_warnings_reach_the_flow_report(tmp_path) -> None:
    """step.toml says the report holds every warning; dropping them makes that false."""
    cfg = config(tmp_path)
    cfg["steps"]["subset"] = {"enabled": True, "token_budgets": [10], "tokenizer": None}
    resolved, paths = run_flow.derive(cfg)
    Path(paths["corpus"]).mkdir(parents=True)
    (Path(paths["corpus"]) / "a.jsonl").write_text(
        "".join(json.dumps({"id": f"d{i}", "source": "s", "text": "one two three four"}) + "\n" for i in range(5)),
        encoding="utf-8",
    )

    report = run_flow.run(cfg)
    subset = next(s for s in report["steps"] if s["step_id"] == "curate/subset")

    assert "warnings" in subset


def test_plan_writes_the_plan_file_it_promises(tmp_path) -> None:
    """--help, README and step.toml all say it does; only a real run did."""
    cfg = config(tmp_path)

    run_flow.plan(cfg, dry_run=True)

    written = json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))
    assert written["dry_run"] is True
    assert {s["key"] for s in written["steps"]} == {p.key for p in run_flow.STEP_ORDER}


def test_plan_runs_the_approval_gate_without_leaving_a_policy(tmp_path) -> None:
    """A preview that skipped the gate would miss the refusals it exists to surface."""
    cfg = profiled(tmp_path)
    cfg["approve"] = approve_block()
    _, paths, _ = run_flow.plan(cfg, dry_run=True)

    assert not Path(paths["approved_policy"]).exists(), "a dry run must not approve anything"

    # ...and the refusal still fires on a preview.
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "c.jsonl").write_text(
        json.dumps({"id": "z1", "source": "web", "text": "an entirely unrelated document"}) + "\n",
        encoding="utf-8",
    )
    cfg["corpus"]["input"] = str(other / "*.jsonl")
    with pytest.raises(run_flow.FlowConfigError, match="granted against corpus"):
        run_flow.plan(cfg, dry_run=True)


def test_plan_and_run_derive_the_same_thing(tmp_path) -> None:
    """A preview describing a different run from the real one is worse than none."""
    cfg = config(tmp_path)
    preview, _, _ = run_flow.plan(cfg, dry_run=True)
    previewed = {r.plan.key: r.config for r in preview}

    run_flow.run(cfg)
    executed = json.loads((Path(cfg["output_root"]) / "flow_plan.json").read_text(encoding="utf-8"))

    assert {s["key"]: s["config"] for s in executed["steps"]} == previewed


def test_the_fingerprint_covers_content_not_just_identity(tmp_path) -> None:
    """Two corpora sharing an id scheme must not fingerprint identically.

    make_key returns the id alone when id_field is set, and every shipped config
    sets one — so a fingerprint built on it would verify an approval against any
    corpus that numbered its documents the same way, while reporting the corpus
    as checked.
    """
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    ids = [f"doc-{i}" for i in range(5)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, body in ((a, "one corpus"), (b, "a completely different corpus")):
        path.write_text(
            "".join(json.dumps({"id": i, "text": f"{body} {i}"}) + "\n" for i in ids),
            encoding="utf-8",
        )

    assert integrity.corpus_fingerprint(str(a), "text", "id") != integrity.corpus_fingerprint(str(b), "text", "id")


def test_the_fingerprint_reads_a_directory_a_glob_and_a_list_alike(tmp_path) -> None:
    """Every other curate step accepts all three spellings of corpus.input."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "a.jsonl").write_text(json.dumps({"id": "1", "text": "hello there"}) + "\n", encoding="utf-8")

    by_dir = integrity.corpus_fingerprint(str(directory), "text", "id")
    by_glob = integrity.corpus_fingerprint(str(directory / "*.jsonl"), "text", "id")
    by_list = integrity.corpus_fingerprint([str(directory / "a.jsonl")], "text", "id")

    assert by_dir == by_glob == by_list


def test_filtering_twice_into_one_output_root_is_refused(tmp_path) -> None:
    """The two-run approval workflow filters into the same place twice. The step
    does not clear the directory, so the second corpus lands beside the first and
    the union is what every later step reads: the ledger counts more out than in,
    the audit calls the surplus unexplained, and subset refuses on duplicate ids.
    None of those say 'stale directory', so the flow says it before any work."""
    cfg = config(tmp_path)
    cfg["steps"]["filter"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    corpus = Path(paths["corpus"])
    corpus.mkdir(parents=True)
    (corpus / "cfe28204c768.jsonl").write_text(
        json.dumps({"id": "d000", "text": "a corpus an earlier run left here"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(run_flow.FlowConfigError, match="already holds"):
        run_flow.preflight(cfg, resolved, paths)


def test_an_empty_output_root_is_not_mistaken_for_a_stale_one(tmp_path) -> None:
    """The reports the previous run wrote are not shards, and a first run must
    not be refused because the directory exists."""
    cfg = config(tmp_path)
    cfg["steps"]["filter"]["enabled"] = True
    resolved, paths = run_flow.derive(cfg)
    corpus = Path(paths["corpus"])
    corpus.mkdir(parents=True)
    (corpus / "run_manifest.json").write_text("{}", encoding="utf-8")

    run_flow.preflight(cfg, resolved, paths)  # must not raise


def test_a_line_that_parses_to_something_other_than_a_document_is_skipped(tmp_path) -> None:
    """Binary input can yield a line that is valid JSON but not an object, and
    calling .get on it raised AttributeError instead of being skipped."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    clean, noisy = tmp_path / "clean.jsonl", tmp_path / "noisy.jsonl"
    record = json.dumps({"id": "1", "text": "hello there"}) + "\n"
    clean.write_text(record, encoding="utf-8")
    noisy.write_text("5\n[1, 2]\n" + record + '"a string"\n', encoding="utf-8")

    assert integrity.corpus_fingerprint(str(noisy), "text", "id") == integrity.corpus_fingerprint(
        str(clean), "text", "id"
    )


def test_the_fingerprint_refuses_a_corpus_it_cannot_read(tmp_path) -> None:
    """expand_inputs resolves parquet because ingest reads it, but this reader is
    JSONL-only. Returning the digest of nothing would be worse than refusing."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    binary = tmp_path / "part_0.parquet"
    binary.write_bytes(b"PAR1\x00\x01\x02\x03rows and columns, not lines\xff\xfePAR1")

    with pytest.raises(integrity.UnreadableCorpusError, match="JSONL-only"):
        integrity.corpus_fingerprint(str(binary), "text", "id")


def test_two_unreadable_corpora_cannot_verify_as_each_other(tmp_path) -> None:
    """The substitution the fingerprint exists to catch: every unreadable corpus
    used to digest to one constant, equal to the digest of no input at all, so an
    approval granted against one verified cleanly against any other."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    hindi, english = tmp_path / "hi.parquet", tmp_path / "en.parquet"
    hindi.write_bytes(b"PAR1\x00\xff\x01 hindi rows PAR1")
    english.write_bytes(b"PAR1\x00\xff\x02 english rows PAR1")

    for path in (hindi, english):
        with pytest.raises(integrity.UnreadableCorpusError):
            integrity.corpus_fingerprint(str(path), "text", "id")


def test_an_absent_corpus_is_refused_rather_than_fingerprinted(tmp_path) -> None:
    """This used to assert the opposite, on a premise that does not hold.

    The reasoning given was that "resolving to no file at all is how preflight
    asks whether an artifact exists". It is not: preflight uses _artifact_exists,
    which goes through expand_inputs. integrity.corpus_fingerprint has exactly
    one caller — the approval check at run_flow.materialise_policy — and there,
    returning a digest for an absent corpus meant every absent corpus shared one
    value. An approval granted against nothing would then verify cleanly against
    anything else that was also nothing.
    """
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    with pytest.raises(integrity.UnreadableCorpusError, match="matched no files"):
        integrity.corpus_fingerprint(str(tmp_path / "nothing" / "*.jsonl"), "text", "id")


def test_the_approval_is_verified_against_the_corpus_the_filter_reads(tmp_path) -> None:
    """With ingest enabled, the corpus the thresholds get applied to is the
    ingested JSONL — which is also what the profile measured. Verifying
    corpus.input instead compared a corpus nobody profiled: ingest mints ids, so
    the two differ even over identical text, and every approval was refused."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    # Its own corpus: the shared one repeats text, and a content-derived id
    # cannot tell two identical documents apart, so ingest refuses it.
    raw = tmp_path / "unique"
    raw.mkdir()
    with (raw / "part_0.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(json.dumps({"id": f"d{i:03d}", "source": "web", "text": f"document {i} " + "body " * 20}) + "\n")

    cfg = config(tmp_path)
    cfg["corpus"]["input"] = str(raw / "*.jsonl")
    # The raw records above carry a `source` column, so ingest is told to map it.
    # Without this the flow refuses: corpus.source_field names what every step
    # AFTER ingest reads, and ingest would have nothing to write into it.
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    warnings = run_flow.materialise_policy(cfg, resolved, paths)

    ingested = next(r for r in resolved if r.plan.key == "filter").config["input_glob"]
    assert ingested != cfg["corpus"]["input"]
    verified = integrity.corpus_fingerprint(ingested, "text", "id")
    assert verified != integrity.corpus_fingerprint(cfg["corpus"]["input"], "text", "id")
    assert any(verified in warning for warning in warnings)


def test_profile_and_the_flow_compute_the_same_fingerprint(tmp_path) -> None:
    """The gate is meaningless unless producer and verifier measure one quantity."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity
    from nemotron.steps.curate.nemo_curator.scripts import run_profile

    path = tmp_path / "c.jsonl"
    path.write_text(
        "".join(json.dumps({"id": f"d{i}", "source": "s", "text": f"document {i} body"}) + "\n" for i in range(6)),
        encoding="utf-8",
    )

    _, stats = run_profile.count_sources([str(path)], "source", "text", "id")

    assert stats["fingerprint"] == integrity.corpus_fingerprint(str(path), "text", "id")


# -- the source column has two halves and only one is obvious ------------------
#
# corpus.source_field names the column every step AFTER ingest reads. What ingest
# WRITES comes from source_field_in_source or source_value. Naming only the first
# is the natural thing to write, produces no source column at all, and every
# downstream step then falls back to "each shard is its own source" — reporting
# per-shard figures that read exactly like per-corpus ones.


def _ingest_cfg(tmp_path, **corpus_over):
    # A real file, because preflight now refuses a corpus.input that matches
    # nothing — the check that tells a user their relative path resolved against
    # the working directory rather than against the config.
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "part.parquet").write_bytes(b"")

    corpus = {
        "input": str(tmp_path / "raw" / "*.parquet"),
        "text_field": "text",
        "id_field": "id",
        "language": "x-test-vi",
        "langpack_dir": str(LANGPACK_FIXTURES),
    }
    corpus.update(corpus_over)
    return {
        "corpus": corpus,
        "output_root": str(tmp_path / "out"),
        "steps": {"ingest": {"enabled": True}, "profile": {"enabled": True}},
    }


def test_a_source_field_ingest_cannot_write_is_refused(tmp_path) -> None:
    cfg = _ingest_cfg(tmp_path, source_field="type")

    with pytest.raises(run_flow.FlowConfigError, match="nothing to write into it"):
        run_flow.plan(cfg, dry_run=True)


def test_a_constant_source_value_satisfies_it(tmp_path) -> None:
    cfg = _ingest_cfg(tmp_path, source_field="source", source_value="c4_vi")

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    by_key = {r.plan.key: r.config for r in resolved}
    assert by_key["ingest"]["source"] == "c4_vi"


def test_a_source_column_in_the_raw_data_satisfies_it(tmp_path) -> None:
    cfg = _ingest_cfg(tmp_path, source_field="source", source_field_in_source="type")

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    by_key = {r.plan.key: r.config for r in resolved}
    assert by_key["ingest"]["source_from"] == "type"


def test_the_check_does_not_fire_when_ingest_is_disabled(tmp_path) -> None:
    """Without ingest the corpus already carries whatever column it carries.

    The corpus is JSONL rather than the parquet _ingest_cfg builds: with ingest
    disabled every step reads the raw corpus as JSONL, so preflight now refuses a
    parquet corpus there. That refusal is a different check from the source_field
    one under test here, and it would mask it.
    """
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "part.jsonl").write_text('{"id":"a","text":"x","type":"s"}\n', encoding="utf-8")

    cfg = _ingest_cfg(tmp_path, source_field="type", input=str(tmp_path / "raw" / "*.jsonl"))
    cfg["steps"]["ingest"]["enabled"] = False

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    assert {r.plan.key for r in resolved} == {p.key for p in run_flow.STEP_ORDER}


# -- ingest and approve must be able to coexist --------------------------------
#
# materialise_policy verifies the approval against the corpus the FILTER reads,
# which is right: ingest mints ids, so the raw input and the ingested corpus are
# different data and the profile measured the latter. But plan() runs it before
# any step, so with steps.ingest enabled that corpus does not exist yet and the
# check compared a real fingerprint against the digest of nothing. Two of the six
# steps could not be enabled together.


def test_a_policy_can_be_approved_with_ingest_enabled(tmp_path) -> None:
    """The corpus ingest will produce is not there at plan time; that is expected."""
    # Every document distinct: a content-derived id cannot tell two identical
    # texts apart, and ingest refuses rather than choosing for you.
    # NOT tmp_path/"raw": corpus_files() writes its own part_0.jsonl there and
    # config() is called below, so a corpus written first would be overwritten.
    raw = tmp_path / "unique_raw"
    raw.mkdir()
    with (raw / "part_0.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(40):
            body = " ".join(f"w{i}x{j}" for j in range(30))
            fh.write(json.dumps({"id": f"d{i:03d}", "source": "web", "text": f"document {i} {body}"}) + "\n")

    cfg = config(tmp_path)
    cfg["corpus"]["input"] = str(raw / "*.jsonl")
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)

    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    # Must not raise: the ingested corpus exists now, and this is the same shape
    # a second run of the flow takes.
    warnings = run_flow.materialise_policy(cfg, resolved, paths)

    assert any("fingerprint verified" in w for w in warnings)


def test_approval_is_reverified_after_ingest_before_filtering(tmp_path, monkeypatch) -> None:
    raw = tmp_path / "changing_raw"
    raw.mkdir()
    path = raw / "part_0.jsonl"

    def write_raw(changed: bool = False) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for i in range(40):
                text = f"document {i} " + " ".join(f"word{i}_{j}" for j in range(20))
                if changed and i == 0:
                    text += " changed after approval profiling"
                fh.write(json.dumps({"source": "web", "text": text}) + "\n")

    write_raw()
    cfg = config(tmp_path)
    cfg["corpus"]["input"] = str(path)
    cfg["corpus"]["source_field_in_source"] = "source"
    cfg["steps"]["ingest"] = {"enabled": True}
    run_flow.run(cfg)

    write_raw(changed=True)
    cfg["steps"]["profile"]["enabled"] = False
    cfg["steps"]["filter"]["enabled"] = True
    cfg["approve"] = approve_block()
    filter_called = False
    real_step_runner = run_flow.step_runner

    def runner(key):
        if key != "filter":
            return real_step_runner(key)

        def fail_if_called(_cfg):
            nonlocal filter_called
            filter_called = True
            return {}

        return fail_if_called

    monkeypatch.setattr(run_flow, "step_runner", runner)

    with pytest.raises(run_flow.FlowConfigError, match="granted against corpus"):
        run_flow.run(cfg)

    assert filter_called is False


def test_a_policy_is_refused_when_the_corpus_it_names_is_absent(tmp_path) -> None:
    """And the refusal must name the emptiness, not compare a digest of nothing."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    cfg = config(tmp_path)
    cfg["steps"]["ingest"] = {"enabled": True}
    cfg["steps"]["profile"]["enabled"] = False
    cfg["approve"] = approve_block()
    resolved, paths = run_flow.derive(cfg)

    with pytest.raises((integrity.UnreadableCorpusError, run_flow.FlowConfigError)) as caught:
        run_flow.materialise_policy(cfg, resolved, paths)

    assert "matched no files" in str(caught.value) or "does not exist" in str(caught.value)


def test_an_unmatched_corpus_names_the_directory_it_looked_in(tmp_path, monkeypatch) -> None:
    """A relative path in a config resolves against the working directory, not
    against the config's own directory, so the same file names two different
    places depending on where the command was run. The refusal has to say which
    one applied, or the reader is left guessing.
    """
    monkeypatch.chdir(tmp_path)
    cfg = {
        "corpus": {"input": "./raw_jsonl/*.jsonl", "text_field": "text", "language": "x-test-vi"},
        "output_root": str(tmp_path / "out"),
        "steps": {"filter": {"enabled": True}},
    }

    with pytest.raises(run_flow.FlowConfigError) as caught:
        run_flow.plan(cfg, dry_run=True)

    message = str(caught.value)
    assert "corpus.input matched no files" in message
    assert str(tmp_path) in message, "the working directory it resolved against must be named"
    assert "not against the directory holding the config" in message


def test_a_corpus_the_run_will_download_is_not_refused_for_being_absent(tmp_path) -> None:
    """curate/nemo_curator calls snapshot_download before it resolves its input
    glob, so a corpus materialised from a Hugging Face snapshot legitimately does
    not exist when preflight runs. Refusing it would block a working config."""
    cfg = {
        "corpus": {"input": str(tmp_path / "never" / "*.jsonl"), "text_field": "text"},
        "output_root": str(tmp_path / "out"),
        "steps": {"filter": {"enabled": True, "dataset": {"repo_id": "x/y", "repo_type": "dataset"}}},
    }

    resolved, _, _ = run_flow.plan(cfg, dry_run=True)

    assert any(r.plan.key == "filter" and r.enabled for r in resolved)


def test_policy_applied_reports_what_the_filter_did_not_what_the_flow_promoted(tmp_path) -> None:
    """A policy can reach the filter without this flow promoting it.

    steps.filter.heuristic_filters.approved_policy is a supported route — the
    flow only refuses it when an approve block would promote a *different*
    policy to the same place. Reporting policy_applied: false there tells a
    reader the corpus was never gated while the filter was removing documents on
    thresholds, which is the same lie the field exists to prevent, inverted.
    """
    manifest = tmp_path / "run_manifest.json"

    manifest.write_text(json.dumps({"policy": {"status": "approved", "thresholds_applied": 2}}))
    assert run_flow._thresholds_applied(str(manifest)) == 2

    manifest.write_text(json.dumps({"policy": {"status": "unapproved", "thresholds_applied": 0}}))
    assert run_flow._thresholds_applied(str(manifest)) == 0, "no thresholds is not application"

    manifest.write_text(json.dumps({"input": {}}))
    assert run_flow._thresholds_applied(str(manifest)) == 0, "a manifest without a policy block"

    assert run_flow._thresholds_applied(str(tmp_path / "absent.json")) == 0, (
        "a run that never wrote a manifest applied nothing"
    )


def test_the_flow_report_distinguishes_an_override_from_an_approval(tmp_path) -> None:
    """policy_applied is true for both, and the READMEs point readers here.

    The step-level manifest was fixed to name three states; leaving flow_report
    with only a boolean reproduces the same conflation one layer up, on the
    surface a reader is actually told to open.
    """
    manifest = tmp_path / "run_manifest.json"

    manifest.write_text(json.dumps({"policy": {"status": "override_unvalidated", "thresholds_applied": 2}}))
    assert run_flow._policy_status(str(manifest)) == "override_unvalidated"

    manifest.write_text(json.dumps({"policy": {"status": "approved", "thresholds_applied": 17}}))
    assert run_flow._policy_status(str(manifest)) == "approved"

    manifest.write_text(json.dumps({"policy": {"status": "unapproved", "thresholds_applied": 0}}))
    assert run_flow._policy_status(str(manifest)) == "unapproved"

    manifest.write_text(json.dumps({"input": {}}))
    assert run_flow._policy_status(str(manifest)) is None, "a manifest without a policy block"
    assert run_flow._policy_status(str(tmp_path / "absent.json")) is None


# -- the documented cascade ---------------------------------------------------
def _table_cells(row: str) -> list[str]:
    """Markdown row to cells, honouring the escaped pipe that separates alternatives."""
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", row.strip().strip("|"))]


def _named_types(cell: str) -> set[str]:
    return set(re.findall(r"`([a-z_]+)`", cell))


def _compatible(candidate: str, declared: set[str], type_defs: dict) -> bool:
    """Whether a type the README names is one a step declaring `declared` accepts."""
    seen, stack = set(), [candidate]
    while stack:
        current = stack.pop()
        if current in declared:
            return True
        parents = type_defs.get(current, {}).get("is_a", [])
        for parent in [parents] if isinstance(parents, str) else parents:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return False


def test_the_readme_steps_table_matches_what_the_steps_declare() -> None:
    """Three descriptions of the same cascade disagreed; this leaves one.

    The category README's table, the six step.toml manifests, and STEP_ORDER
    were written separately and drifted separately — the table named
    `audit_report`, `candidate_policy` and `subset_jsonl`, none of which any
    manifest declares or types.toml registers, and listed the steps in an order
    the flow does not run them in. A table nobody checks is a fourth opinion, so
    it is parsed and compared rather than proof-read.

    A type the README names that the manifest does not is accepted when it is
    reachable by `is_a` from one that is: that is what compatibility means, and
    naming `prepared_jsonl` beside `raw_jsonl` is the whole reason the distinct
    type exists.
    """
    steps_root = STEP_DIR.parent.parent
    type_defs = tomllib.loads((steps_root / "types.toml").read_text(encoding="utf-8"))
    manifests = {
        data["step"]["id"]: data
        for path in sorted(STEP_DIR.parent.rglob("step.toml"))
        for data in [tomllib.loads(path.read_text(encoding="utf-8"))]
        if data.get("step", {}).get("id")
    }

    readme = (STEP_DIR.parent / "README.md").read_text(encoding="utf-8")
    rows = [
        line
        for line in readme.split("## Steps", 1)[1].splitlines()
        if line.startswith("|") and "curate/" in line and "---" not in line
    ]
    assert len(rows) == len(run_flow.STEP_ORDER), f"the table describes {len(rows)} of six steps"

    documented_order: list[str] = []
    for row in rows:
        cells = _table_cells(row)
        step_id = re.search(r"`(curate/[a-z_]+)`", cells[1]).group(1)
        documented_order.append(step_id)
        declared = manifests[step_id]

        for section, cell in (("consumes", cells[2]), ("produces", cells[3])):
            actual = {entry["type"] for entry in declared.get(section, [])}
            documented = _named_types(cell)
            missing = actual - documented
            assert not missing, f"{step_id} declares {section} {sorted(missing)}; the README omits it"
            for extra in documented - actual:
                assert _compatible(extra, actual, type_defs), (
                    f"the README says {step_id} {section} {extra!r}, which is not "
                    f"compatible with anything it declares ({sorted(actual)})"
                )

    assert documented_order == [plan.step_id for plan in run_flow.STEP_ORDER], (
        "the table must list the steps in the order the flow runs them"
    )


# -- decontamination before subset ---------------------------------------------


def decontaminable_corpus(tmp_path: Path) -> tuple[str, str, set[str]]:
    """A corpus, a holdout that shares pages with it, and the ids that must go.

    Overlap is by source page rather than by wording, so the removal happens on
    CPU through the exact group-identity pass and the test needs no GPU.
    """
    words = "curation tokens budget corpus stratum nesting document policy".split()
    body = lambda i: " ".join(words[(i + j) % len(words)] for j in range(30)) + "."  # noqa: E731

    corpus_dir = tmp_path / "out" / "filtered_jsonl"
    corpus_dir.mkdir(parents=True)
    contaminated = {f"d{i:03d}" for i in range(8)}
    with (corpus_dir / "part_0.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(
                json.dumps(
                    {
                        "id": f"d{i:03d}",
                        "source": "web",
                        "url": f"https://example.test/page/{i}",
                        "text": body(i),
                    }
                )
                + "\n"
            )

    holdout = tmp_path / "holdout.jsonl"
    with holdout.open("w", encoding="utf-8") as fh:
        for i in range(8):
            # Same page, different wording: nothing textual links these to the
            # training documents, so only the group-identity pass can find them.
            fh.write(
                json.dumps(
                    {
                        "id": f"h{i:03d}",
                        "url": f"https://example.test/page/{i}",
                        "text": "Entirely different prose about unrelated matters. " * 6,
                    }
                )
                + "\n"
            )

    return str(corpus_dir), str(holdout), contaminated


def tier_documents(subset_root: Path) -> set[str]:
    found: set[str] = set()
    for tier in sorted(subset_root.glob("budget_*")):
        corpus = tier / "subset.jsonl"
        if corpus.is_file():
            found |= {
                json.loads(line)["id"] for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()
            }
    return found


def test_subset_tiers_never_contain_what_decontamination_removed(tmp_path) -> None:
    """The ordering, asserted on documents rather than on STEP_ORDER.

    Both steps declared ``needs=("corpus",)``, so which ran first was whatever
    the tuple happened to say — and it said subset. Tiers were cut from the
    corpus that still held every document decontamination was about to remove,
    and no artifact could show it: the subset report counts documents, not their
    provenance, and the decontamination report describes a corpus the tiers were
    not taken from.

    This asserts the property, not the order, so it stays true if the mechanism
    changes and fails if the ordering regresses.
    """
    corpus_dir, holdout, planted = decontaminable_corpus(tmp_path)
    root = tmp_path / "out"
    cfg = {
        "corpus": {
            "input": f"{corpus_dir}/*.jsonl",
            "text_field": "text",
            "id_field": "id",
            "source_field": "source",
        },
        "output_root": str(root),
        "steps": {
            "ingest": {"enabled": False},
            "profile": {"enabled": False},
            # Disabled: the corpus is already at paths['corpus'], and running
            # Curator here would buy the test nothing it is checking.
            "filter": {"enabled": False},
            "audit": {"enabled": False},
            "decontamination": {"enabled": True, "holdout": holdout, "skip_similarity": True},
            "subset": {
                "enabled": True,
                "token_budgets": [200, 600],
                "length_bands": [16, 50],
                "tokenizer": None,
                "token_cache": None,
                "quality_score_field": None,
                "seed": 0,
            },
        },
        "approve": None,
    }

    run_flow.run(cfg)

    before = {
        json.loads(line)["id"]
        for line in (Path(corpus_dir) / "part_0.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    kept = {
        json.loads(line)["id"]
        for line in (root / "decontaminated" / "train_decontaminated.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    removed = before - kept
    assert removed == planted, (
        f"the fixture proves nothing unless decontamination removed the planted pages: {sorted(removed)}"
    )

    tiers = tier_documents(root / "subset")
    assert tiers, "subset wrote no tier, so the assertion below would hold vacuously"
    assert not (tiers & removed), (
        f"subset tiers carry {sorted(tiers & removed)}, which decontamination removed. "
        "The tiers were cut before decontamination ran."
    )
    assert tiers <= kept, "every tier document must come from the decontaminated corpus"


def test_a_producer_scheduled_after_its_consumer_is_refused(monkeypatch, tmp_path) -> None:
    """The guard that makes the ordering above a checked claim, not a comment.

    Nothing in STEP_ORDER stopped a step from being placed before the step whose
    output it reads: the dependency loop treated "enabled" as "will have run".
    Simulating the regression is the only way to reach this branch, because the
    tuple is now correct.
    """
    corpus_dir, holdout, _ = decontaminable_corpus(tmp_path)
    regressed = tuple(
        sorted(run_flow.STEP_ORDER, key=lambda p: 99 if p.key == "decontamination" else run_flow.STEP_POSITION[p.key])
    )
    monkeypatch.setattr(run_flow, "STEP_ORDER", regressed)
    monkeypatch.setattr(run_flow, "STEP_POSITION", {p.key: i for i, p in enumerate(regressed)})

    cfg = {
        "corpus": {"input": f"{corpus_dir}/*.jsonl", "text_field": "text", "id_field": "id"},
        "output_root": str(tmp_path / "out"),
        "steps": {
            "filter": {"enabled": False},
            "decontamination": {"enabled": True, "holdout": holdout, "skip_similarity": True},
            "subset": {"enabled": True, "token_budgets": [200], "length_bands": [16], "seed": 0},
        },
    }

    with pytest.raises(run_flow.FlowConfigError) as excinfo:
        run_flow.plan(cfg)
    message = str(excinfo.value)
    assert "steps.subset needs 'decontaminated'" in message
    assert "scheduled to run after it" in message


def test_a_decontaminated_corpus_the_run_will_not_read_is_named(tmp_path, caplog) -> None:
    """The same silence from the other side.

    With decontamination disabled, subset legitimately reads the pre-decontamination
    corpus — but if an earlier run left a decontaminated one beside it, the author
    almost certainly meant that. No `needs` edge can state this, because the
    dependency genuinely is not required; it is a warning rather than a refusal
    for the same reason.
    """
    corpus_dir, _, _ = decontaminable_corpus(tmp_path)
    root = tmp_path / "out"
    stale = root / "decontaminated"
    stale.mkdir(parents=True)
    (stale / "train_decontaminated.jsonl").write_text(
        json.dumps({"id": "d000", "text": "kept"}) + "\n", encoding="utf-8"
    )

    cfg = {
        "corpus": {"input": f"{corpus_dir}/*.jsonl", "text_field": "text", "id_field": "id"},
        "output_root": str(root),
        "steps": {
            "filter": {"enabled": False},
            "decontamination": {"enabled": False},
            "subset": {"enabled": True, "token_budgets": [200], "length_bands": [16], "seed": 0},
        },
    }

    _, _, warnings = run_flow.plan(cfg)
    assert any("BEFORE decontamination" in w and str(stale) in w for w in warnings), warnings


def test_subset_reads_the_decontaminated_file_and_not_the_directory(tmp_path) -> None:
    """A glob here would feed subset the holdout it is protecting.

    curate/decontamination's work_dir is derived as output_dir/cache, so with
    similarity enabled the step writes cache/union/union.jsonl beneath its own
    output directory — the train and holdout splits merged, with ids rewritten.
    A `decontaminated/**/*.jsonl` spelling, which is how every other corpus in
    this flow is addressed, would therefore put the evaluation split into the
    training tiers. The artifact is one named file for that reason.
    """
    root = tmp_path / "out"
    paths = run_flow._artifact_paths(root)
    assert paths["decontaminated"] == str(root / "decontaminated" / "train_decontaminated.jsonl")
    assert not any(character in paths["decontaminated"] for character in "*?["), (
        "the decontaminated corpus is addressed as a file, never as a pattern"
    )

    resolved, _ = run_flow.derive(
        {
            "corpus": {"input": str(tmp_path / "raw" / "*.jsonl"), "text_field": "text", "id_field": "id"},
            "output_root": str(root),
            "steps": {"decontamination": {"enabled": True}, "subset": {"enabled": True}},
        }
    )
    subset = next(r for r in resolved if r.plan.key == "subset")
    assert subset.config["input_glob"] == paths["decontaminated"]

    work_dir = next(r for r in resolved if r.plan.key == "decontamination").config["work_dir"]
    assert Path(work_dir).is_relative_to(root / "decontaminated"), (
        "the premise of this test: the step's scratch space lives under its output directory"
    )
