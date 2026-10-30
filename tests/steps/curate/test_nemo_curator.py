# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/curate/nemo_curator``."""

from __future__ import annotations

import os
import tomllib

import pytest
import yaml

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "nemo_curator")

#: Keys F1 adds. Every one must be absent-equivalent at its documented default,
#: because the PR promises byte-identical behaviour for configs that predate it.
F1_KEYS = ("metadata_fields", "id_field", "source_field", "emit_manifest")

#: Keys F2 adds, under the same promise.
F2_KEYS = ("mode", "heuristic_filters")


def test_nemo_curator_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/nemo_curator",
        expected_launch="python",
        expected_default_config="default",
    )


def test_f1_parameters_are_documented_in_the_manifest() -> None:
    """A knob that is not in step.toml cannot be discovered by a user."""
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    documented = {p["name"] for p in manifest.get("parameters", [])}
    missing = [key for key in F1_KEYS + F2_KEYS if key not in documented]
    assert not missing, f"step.toml does not document: {missing}"


def test_f1_defaults_are_neutral() -> None:
    """The shipped default config must not change what the step does.

    ``metadata_fields: []`` collapses the reader projection back to text only,
    and a null ``emit_manifest`` writes nothing, so an existing overlay keeps
    its current output.
    """
    config = yaml.safe_load((STEP_DIR / "config" / "default.yaml").read_text(encoding="utf-8"))

    assert config["metadata_fields"] == [], "default must not widen the reader projection"
    assert config["emit_manifest"] is None, "default must not write a manifest"
    assert config["id_field"] is None
    assert config["source_field"] is None
    assert config["heuristic_filters"] is None, "default must not filter on an unreviewed policy"
    assert config["mode"] == "filter", "default must keep the historical column set"


def test_tiny_config_uses_a_local_and_container_portable_fixture() -> None:
    """The documented local smoke must not depend on the container mount path."""
    config = yaml.safe_load((STEP_DIR / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    repo_root = STEP_DIR.parents[4]

    assert config["input_glob"] == "./src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl"
    assert (repo_root / config["input_glob"]).is_file()
    assert config["emit_manifest"] is None


def test_manifest_defaults_use_native_toml_types() -> None:
    """``default = []`` not ``default = "[]"``.

    The existing manifests use native TOML types throughout; ``"null"`` is the
    accepted quoted workaround only because TOML has no null literal.
    """
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    by_name = {p["name"]: p for p in manifest.get("parameters", [])}

    assert by_name["metadata_fields"]["default"] == []
    for key in ("id_field", "source_field", "emit_manifest"):
        assert by_name[key]["default"] == "null", f"{key} should use the quoted-null convention"


# -- the ledger producer ------------------------------------------------------
#
# curate/audit consumes a curation_ledger to attribute a loss rather than merely
# detect it. Until this existed the artifact had a consumer and no producer, so
# audit always took the attribution-unavailable branch in production.


def _stub_curator(monkeypatch):
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

    import importlib

    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    return module


def test_uv_run_does_not_repackage_ray_workers(monkeypatch) -> None:
    """Workers must reuse the installed env instead of creating an empty one."""
    monkeypatch.delenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", raising=False)

    _stub_curator(monkeypatch)

    assert os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def _corpus(tmp_path, n_in: int, n_out: int):
    import json

    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "a.jsonl").write_text(
        "".join(json.dumps({"id": f"d{i}", "source": "s", "text": "x"}) + "\n" for i in range(n_in)),
        encoding="utf-8",
    )
    (out / "a.jsonl").write_text(
        "".join(json.dumps({"id": f"d{i}", "source": "s", "text": "x"}) + "\n" for i in range(n_out)),
        encoding="utf-8",
    )
    return {
        "input_glob": str(src / "*.jsonl"),
        "output_dir": str(out),
        "text_field": "text",
        "id_field": "id",
        "source_field": "source",
    }


def test_the_ledger_balances_on_a_normal_run(tmp_path, monkeypatch) -> None:
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    step = _stub_curator(monkeypatch)
    cfg = _corpus(tmp_path, 100, 82)
    path = tmp_path / "ledger.json"

    step.emit_ledger(cfg, str(path), completed=True)
    led = ledger_module.load_ledger(path)

    assert led.n_input == 100
    assert led.n_success == 82
    assert led.n_filtered == 18
    assert led.balanced


def test_the_ledger_does_not_name_a_gate_it_cannot_observe(tmp_path, monkeypatch) -> None:
    """No counters reached us, so no breakdown is invented."""
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    step = _stub_curator(monkeypatch)
    path = tmp_path / "ledger.json"

    step.emit_ledger(_corpus(tmp_path, 10, 7), str(path), completed=True)
    led = ledger_module.load_ledger(path)

    assert set(led.filtered) == {step.UNATTRIBUTED}
    assert "unreconciled" not in led.notes, "nothing was collected, so nothing disagreed"


def test_per_gate_counts_are_recorded_when_they_reconcile(tmp_path, monkeypatch) -> None:
    """Curator's own per-stage counters, published only once they add up."""
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    step = _stub_curator(monkeypatch)
    path = tmp_path / "ledger.json"

    step.emit_ledger(
        _corpus(tmp_path, 10, 7),
        str(path),
        completed=True,
        stage_names=["reader", "score_filter_a", "score_filter_b", "writer"],
        stage_counts={"reader": 10, "score_filter_a": 10, "score_filter_b": 8, "writer": 7},
    )
    led = ledger_module.load_ledger(path)

    assert led.filtered == {"score_filter_a": 2, "score_filter_b": 1}
    assert led.balanced


def test_a_breakdown_that_does_not_reconcile_is_discarded(tmp_path, monkeypatch) -> None:
    """The disk count is independent evidence; a breakdown that contradicts it is not published."""
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    step = _stub_curator(monkeypatch)
    path = tmp_path / "ledger.json"

    step.emit_ledger(
        _corpus(tmp_path, 10, 7),  # 3 documents actually lost
        str(path),
        completed=True,
        stage_names=["reader", "score_filter_a", "writer"],
        stage_counts={"reader": 10, "score_filter_a": 9, "writer": 9},  # accounts for 1
    )
    led = ledger_module.load_ledger(path)

    assert set(led.filtered) == {step.UNATTRIBUTED}, "a partial breakdown must not look complete"
    assert led.n_filtered == 3
    assert "unreconciled" in led.notes, "the disagreement is itself a finding"


# -- the attribution arithmetic, without a pipeline ----------------------------


def test_removals_are_the_difference_between_consecutive_stages(monkeypatch) -> None:
    step = _stub_curator(monkeypatch)

    assert step.attribute_removals(["a", "b", "c"], {"a": 100, "b": 90, "c": 85}, 15) == {"a": 10, "b": 5}


def test_a_stage_that_removed_nothing_is_not_named(monkeypatch) -> None:
    """A gate that dropped no documents is not a reason anything was filtered."""
    step = _stub_curator(monkeypatch)

    assert step.attribute_removals(["a", "b", "c"], {"a": 10, "b": 10, "c": 8}, 2) == {"b": 2}


def test_composite_stage_names_are_matched_by_prefix(monkeypatch) -> None:
    """A CompositeStage decomposes into differently-named parts at execution."""
    step = _stub_curator(monkeypatch)

    result = step.attribute_removals(
        ["jsonl_reader", "score_filter"],
        {"jsonl_reader_file_partitioning": 100, "score_filter": 96},
        4,
    )

    assert result == {"jsonl_reader": 4}


def test_overlapping_stage_prefixes_are_matched_to_the_most_specific_name(monkeypatch) -> None:
    step = _stub_curator(monkeypatch)

    result = step.attribute_removals(
        ["reader", "stopword_ratio", "stopword_ratio_folded", "writer"],
        {
            "reader": 100,
            "stopword_ratio_folded_worker": 80,
            "stopword_ratio_worker": 90,
            "writer": 80,
        },
        20,
    )

    assert result == {"reader": 10, "stopword_ratio": 10}


def test_one_stage_cannot_be_differenced(monkeypatch) -> None:
    step = _stub_curator(monkeypatch)

    assert step.attribute_removals(["a"], {"a": 100}, 0) is None


def test_counts_that_do_not_sum_to_the_observed_loss_are_refused(monkeypatch) -> None:
    step = _stub_curator(monkeypatch)

    assert step.attribute_removals(["a", "b"], {"a": 100, "b": 90}, 20) is None


def test_a_run_that_gained_rows_is_reported_not_balanced_away(tmp_path, monkeypatch) -> None:
    """More out than in is a real fault — a retry double-write, say."""
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    step = _stub_curator(monkeypatch)
    path = tmp_path / "ledger.json"

    step.emit_ledger(_corpus(tmp_path, 10, 14), str(path), completed=True)
    led = ledger_module.load_ledger(path)

    assert led.n_filtered == 0, "a negative removal count would hide the gain"
    assert led.quarantined_units, "the anomaly must be recorded as a unit"
    assert "more rows than input" in led.quarantined_units[0]["reason"]


def test_an_incomplete_run_still_writes_its_ledger(tmp_path, monkeypatch) -> None:
    """The unbalanced ledger IS the evidence; refusing to write it destroys it."""
    from nemotron.steps.curate.nemo_curator.runtime import ledger as ledger_module

    step = _stub_curator(monkeypatch)
    path = tmp_path / "ledger.json"

    step.emit_ledger(_corpus(tmp_path, 100, 3), str(path), completed=False)

    assert path.is_file()
    loaded = ledger_module.load_ledger(path)
    assert loaded.notes["completed"] is False
    assert loaded.n_filtered == 0
    assert loaded.n_failed == 97
    assert loaded.failed_units


def test_emit_ledger_is_documented_and_defaults_to_off() -> None:
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)
    documented = {p["name"] for p in manifest.get("parameters", [])}
    config = yaml.safe_load((STEP_DIR / "config" / "default.yaml").read_text(encoding="utf-8"))

    assert "emit_ledger" in documented
    assert config["emit_ledger"] is None, "default must not change what the step writes"


# -- the language gate is two decisions, and both must be reachable ------------
#
# `language_codes` says WHICH languages; `quality_filters.min_langid_score` says
# how sure FastText has to be. The second used to fall back to 0.0 when omitted,
# which disabled the confidence gate entirely — while the shipped default.yaml
# said 0.3 and Curator's own default is 0.3, so the flow and the standalone step
# silently disagreed and nothing in any report said which was in effect.


def test_the_langid_fallback_matches_curator(monkeypatch) -> None:
    step = _stub_curator(monkeypatch)

    assert step.DEFAULT_LANGID_SCORE == 0.3, "a fallback below Curator's own default silently weakens the gate"


def test_language_codes_are_validated_against_the_configured_model(tmp_path, monkeypatch) -> None:
    import sys
    import types

    step = _stub_curator(monkeypatch)
    model_path = tmp_path / "lid.176.bin"
    model_path.write_bytes(b"model")

    model = types.SimpleNamespace(get_labels=lambda: ["__label__en", "__label__hi", "__label__zh_Hans"])
    fasttext = types.ModuleType("fasttext")
    fasttext.load_model = lambda _path: model
    monkeypatch.setitem(sys.modules, "fasttext", fasttext)

    step.validate_language_codes({"language_codes": ["HI", "zh"], "models": {"fasttext_langid": str(model_path)}})
    with pytest.raises(ValueError, match=r"HIN.*emits none"):
        step.validate_language_codes({"language_codes": ["HIN"], "models": {"fasttext_langid": str(model_path)}})


def test_a_total_language_drop_is_a_failure(tmp_path, monkeypatch) -> None:
    step = _stub_curator(monkeypatch)
    cfg = _corpus(tmp_path, 3, 0)
    cfg["language_codes"] = ["HI"]

    with pytest.raises(ValueError, match="kept 0 of 3"):
        step.refuse_empty_language_output(cfg, [str(tmp_path / "in" / "a.jsonl")])


def test_a_zero_input_glob_is_refused_before_ray_starts(tmp_path, monkeypatch) -> None:
    step = _stub_curator(monkeypatch)
    cfg = {
        "input_glob": str(tmp_path / "missing" / "*.jsonl"),
        "output_dir": str(tmp_path / "out"),
        "text_field": "text",
    }

    with pytest.raises(ValueError, match="matched no"):
        step.run(cfg)


def test_a_new_attempt_removes_stale_success_artifacts(tmp_path, monkeypatch) -> None:
    step = _stub_curator(monkeypatch)
    manifest_path = tmp_path / "run_manifest.json"
    ledger_path = tmp_path / "curation_ledger.json"
    manifest_path.write_text('{"producer":{"completed_at":"old"}}', encoding="utf-8")
    ledger_path.write_text('{"notes":{"completed":true}}', encoding="utf-8")

    with pytest.raises(ValueError, match="mode must be"):
        step.run(
            {
                "mode": "invalid",
                "emit_manifest": str(manifest_path),
                "emit_ledger": str(ledger_path),
            }
        )

    assert not manifest_path.exists()
    assert not ledger_path.exists()


def test_manifest_and_ledger_cannot_share_an_artifact_path(tmp_path, monkeypatch) -> None:
    step = _stub_curator(monkeypatch)
    artifact_path = tmp_path / "run_artifact.json"
    artifact_path.write_text('{"old":"artifact"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must use different paths"):
        step.run(
            {
                "emit_manifest": str(artifact_path),
                "emit_ledger": str(artifact_path),
            }
        )

    assert artifact_path.read_text(encoding="utf-8") == '{"old":"artifact"}\n'


def test_the_filter_resolves_with_the_shared_resolver_not_a_private_one(tmp_path, monkeypatch) -> None:
    """Resolution is one contract for the whole category; reading is this step's own limit.

    This used to assert the opposite -- that the filter step silently narrowed a
    glob to the extensions its reader happens to support. That private rule is
    exactly what let preflight validate one corpus while this step processed
    another, so the assertion is inverted on purpose: the resolver returns what
    the shared contract says, and anything unreadable is refused by name below
    rather than dropped here.
    """
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    step = _stub_curator(monkeypatch)
    (tmp_path / "accepted.jsonl").write_text('{"text":"kept"}\n', encoding="utf-8")
    (tmp_path / "ignored.txt").write_text('{"text":"not read"}\n', encoding="utf-8")

    assert step.resolve_inputs(str(tmp_path / "*")) == integrity.expand_inputs(str(tmp_path / "*"))


def test_a_file_the_reader_cannot_read_is_refused_by_name(tmp_path, monkeypatch) -> None:
    """A parquet corpus resolves and preflight accepts it; this step must say so out loud.

    Handing it to JsonlReader would read binary as text and report the damage as
    corpus, which is the failure the shared resolver exists to prevent.
    """
    step = _stub_curator(monkeypatch)
    (tmp_path / "part_0.parquet").write_bytes(b"PAR1")

    with pytest.raises(ValueError, match="cannot read"):
        step.run(
            {
                "input_glob": str(tmp_path),
                "output_dir": str(tmp_path / "out"),
                "text_field": "text",
            }
        )


def test_real_jsonl_reader_preserves_string_ids_and_mixed_metadata(tmp_path) -> None:
    reader_module = pytest.importorskip("nemo_curator.stages.text.io.reader.jsonl")
    path = tmp_path / "input.jsonl"
    path.write_text(
        '{"id":"001","mixed":"01","text":"first"}\n{"id":"1","mixed":2,"text":"second"}\n',
        encoding="utf-8",
    )

    stage = reader_module.JsonlReaderStage(
        fields=["id", "mixed", "text"],
        read_kwargs={"dtype": False, "convert_dates": False},
    )
    records = stage.read_data(
        [str(path)],
        read_kwargs=stage.read_kwargs,
        fields=stage.fields,
    ).to_dict(orient="records")

    assert records[0]["id"] == "001"
    assert records[1]["id"] == "1"
    assert records[0]["mixed"] == "01"


def test_the_shipped_configs_expose_the_langid_score() -> None:
    """A knob nobody can see is a knob nobody sets.

    No flow config mentioned min_langid_score, so a run that named language_codes
    got whatever the code fell back to, and nothing in the config or the report
    said which value was in effect.
    """
    flow_configs = (STEP_DIR / "config").glob("*.yaml")
    checked = 0
    for path in flow_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = (cfg.get("steps") or {}).get("filter") or {}
        if "language_codes" not in block:
            continue
        checked += 1
        quality = block.get("quality_filters") or {}
        assert "min_langid_score" in quality, (
            f"{path.name} offers language_codes but never names the confidence threshold that goes with it"
        )
    assert checked, "no shipped config declares language_codes"


def test_writing_into_a_directory_that_already_has_output_is_refused(tmp_path, monkeypatch) -> None:
    """Curator names shards by content hash, so a second run ADDS to the first.

    Measured on a real run: one corpus ended with 31,689 rows out of 20,000 in,
    because a policy run and an earlier no-policy run both wrote into the same
    directory under different hash names. The ledger and audit both caught it,
    but only after the work had been done twice — and 7.63% of the surviving
    Hindi corpus was Chinese, from the older shard.
    """
    step = _stub_curator(monkeypatch)
    cfg = _corpus(tmp_path, 10, 10)  # _corpus already writes out/a.jsonl

    with pytest.raises(ValueError, match="already holds"):
        step.run(cfg)


def test_the_language_gates_are_named_apart() -> None:
    """The ledger is keyed on stage names, so an unnamed gate is an unreadable one.

    Language identification is TWO gates — confidence, then code — and the
    per-gate attribution has to tell them apart. Curator's Filter hardcodes its
    name to "filter_fn" and __post_init__ discards whatever the constructor is
    given, so the code gate arrived in a real ledger as "filter_fn: 5134": a
    count with nothing saying what removed those documents.

    Asserted against the source because constructing the stage needs Curator,
    which the rest of this file deliberately stubs.
    """
    source = (STEP_DIR / "step.py").read_text(encoding="utf-8")

    assert 'language_stage.name = "language_code"' in source, (
        "Curator resets Filter.name in __post_init__, so it must be set afterwards"
    )
    assert "filter_fn=language_code" in source, "the callable itself should also be named, not a lambda"


def test_the_domain_classifier_must_not_truncate_the_corpus() -> None:
    """The classifier reads text; it must not become the thing that rewrites it.

    Curator's tokenizer truncates IN PLACE on the DataFrame that continues to the
    writer (stages/text/models/tokenizer.py:159-160), and
    MultilingualDomainClassifier defaults max_chars=2000. Constructing it without
    naming max_chars therefore silently rewrote every delivered document longer
    than 2000 characters: measured on one real run, 9,409 of 17,617 Vietnamese
    documents (53.4%) were cut to exactly 2000 chars, destroying 29.7 million
    characters — while the ledger, the manifest and the audit all reported a
    clean run, because every one of them counts ROWS and none reads content.
    """
    source = (STEP_DIR / "step.py").read_text(encoding="utf-8")

    assert '"max_chars"' in source or "max_chars=" in source, (
        "max_chars must be named explicitly; its default rewrites the corpus"
    )
    assert "DOMAIN_MAX_CHARS" in source, "the value deserves a named constant carrying why it is what it is"


def test_domain_score_field_is_documented_as_a_probability_vector() -> None:
    with (STEP_DIR / "step.toml").open("rb") as handle:
        manifest = tomllib.load(handle)

    description = {parameter["name"]: parameter["description"] for parameter in manifest["parameters"]}[
        "domain_score_field"
    ]
    assert "vector" in description
    assert "not a scalar" in description


def test_curate_extra_installs_a_locked_text_runtime() -> None:
    root = STEP_DIR.parents[4]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    curate = project["project"]["optional-dependencies"]["curate"]
    assert any("nemo-curator[text_cpu]==1.3.0" in requirement for requirement in curate)
    runtime = project["tool"]["nemotron"]["runtime"]["curate"]
    assert "nemo-curator" not in runtime.get("omit-packages", [])


def test_project_python_floor_matches_the_curate_runtime() -> None:
    root = STEP_DIR.parents[4]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert project["project"]["requires-python"].startswith(">=3.11")


def test_a_companion_file_beside_the_shards_does_not_abort_the_run(tmp_path, monkeypatch) -> None:
    """`input: ./out/*` on a corpus with a README beside it worked before and must still work.

    Refusing every non-readable file the shared resolver returns would stop a run
    that has always succeeded. A README is not corpus and never was; only a
    recognised corpus format in the wrong step earns a refusal.
    """
    step = _stub_curator(monkeypatch)
    (tmp_path / "part_0.jsonl").write_text('{"text":"kept"}\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")

    # Asserted on the narrowing rather than by running the pipeline: `run` needs
    # Curator to get past the reader, so `pytest.raises(Exception)` here passed
    # only on a machine without it -- a verdict about the environment, not the code.
    assert [p.rsplit("/", 1)[-1] for p in step.readable_inputs(str(tmp_path / "*"))] == ["part_0.jsonl"]
