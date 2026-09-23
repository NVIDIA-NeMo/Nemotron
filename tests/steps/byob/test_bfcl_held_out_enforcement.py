"""Held-out enforcement across Stage 4 binding and Stage 12 publication."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl import pipeline
from nemotron.steps.byob.runtime.benchmark_families.bfcl.bfcl_json_export import (
    BFCL_JSON_QUESTION_FILE,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    HeldOutPolicy,
    fixture_ref,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NEMO_DATASET_FILE,
    NEMO_EVALUATOR_ROOT,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    oracle_runtime_fixtures,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
    expand as expand_stage,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
    oracle_validation,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
    ExpansionError,
    run_expand,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.held_out import (
    HELD_OUT_BINDINGS,
    HELD_OUT_SCAN,
    HeldOutLeakError,
    held_out_policy,
    load_binding_report,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
    _held_out_not_found_context,
    _representative_held_out_policy,
)

ORACLE_VALIDATION_REPORT = "oracle_validation_report.json"

BYOB_ROOT = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"
BFCL_CONFIG_DIR = BYOB_ROOT / "bfcl" / "config"

# The tiny pack reserves one template and one available book. Three templates and
# two remaining book rows still fill tasks_per_category, so the run stays feasible.
HELD_OUT_TEMPLATE = "lib_irrelevant_renew"
HELD_OUT_BOOK = "BK-200"


def _prepare_pack(tmp_path: Path, *, held_out: dict[str, Any] | None) -> Path:
    pack = tmp_path / "pack"
    shutil.copytree(
        BYOB_ROOT / "data" / "tiny_oracle_pack",
        pack,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    if held_out is not None:
        (pack / "held_out.yaml").write_text(yaml.safe_dump(held_out), encoding="utf-8")
        manifest_path = pack / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["held_out"] = "held_out.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return pack


def _write_config(tmp_path: Path, pack: Path, name: str) -> Path:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["oracle_pack"] = {"manifest_path": str(pack / "manifest.yaml")}
    config_data["oracle_runtime"]["allowed_roots"] = [str(tmp_path)]
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return path


def _default_policy() -> dict[str, Any]:
    return {
        "version": "0.1.0",
        "fixtures": {"books": [HELD_OUT_BOOK]},
        "templates": [HELD_OUT_TEMPLATE],
        "policy": {"fixtures_in_backend_state": True, "seed": 11},
    }


@pytest.fixture(scope="module")
def held_out_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[dict[str, Any]], Path]:
    """Generate the tiny pack once with a declared held-out policy."""
    import pyarrow.parquet as pq

    tmp_path = tmp_path_factory.mktemp("held_out_slice")
    pack = _prepare_pack(tmp_path, held_out=_default_policy())
    benchmark_path = generate_bfcl(_write_config(tmp_path, pack, "held-out.yaml"))
    return pq.read_table(benchmark_path).to_pylist(), benchmark_path.parent


def test_generation_no_longer_refuses_a_pack_that_declares_held_out(held_out_run) -> None:
    rows, _ = held_out_run

    assert rows


def test_stage_four_binds_neither_the_reserved_template_nor_the_reserved_row(
    held_out_run,
) -> None:
    import pyarrow.parquet as pq

    rows, output_dir = held_out_run
    instances = pq.read_table(output_dir / "stage_cache" / "task_instances.parquet").to_pylist()

    assert {str(row["template_id"]) for row in rows} == {
        "lib_status_single",
        "lib_checkout_confirm",
        "lib_status_parallel",
    }
    assert all(HELD_OUT_TEMPLATE != str(instance["template_id"]) for instance in instances)
    assert all(
        fixture_ref("books", HELD_OUT_BOOK) not in list(instance["fixture_refs"] or [])
        for instance in instances
    )

    report = json.loads((output_dir / "stage_cache" / HELD_OUT_BINDINGS).read_text(encoding="utf-8"))
    assert report["blocked_templates"] == [HELD_OUT_TEMPLATE]
    assert report["blocked_fixture_refs"] == [fixture_ref("books", HELD_OUT_BOOK)]
    assert report["counts"]["bind_attempts"] > report["counts"]["fixture_refs_blocked"]
    assert report["counts"]["tasks_expanded"] == len(instances)


def test_stage_twelve_scans_every_row_and_stamps_a_checked_result(held_out_run) -> None:
    rows, output_dir = held_out_run
    scan = json.loads((output_dir / "stage_cache" / HELD_OUT_SCAN).read_text(encoding="utf-8"))

    assert all(row["held_out_hit"] is False for row in rows)
    assert scan["hits"] == []
    assert scan["counts"]["rows_hit"] == 0
    assert scan["counts"]["rows_dropped"] == 0
    assert scan["counts"]["rows_scanned"] >= len(rows)
    assert scan["counts"]["rows_published"] == len(rows)
    assert scan["stage_four"]["blocked_templates"] == [HELD_OUT_TEMPLATE]


def test_held_out_enforcement_and_both_exports_share_the_same_safe_task_set(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    pack = _prepare_pack(tmp_path, held_out=_default_policy())
    config_path = _write_config(tmp_path, pack, "held-out-exports.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["exports"] = {"bfcl_json": True, "nemo_evaluator_bundle": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    benchmark_path = generate_bfcl(config_path)
    rows = pq.read_table(benchmark_path).to_pylist()
    expected_ids = [str(row["task_id"]) for row in rows]
    bfcl_ids = [
        json.loads(line)["id"]
        for line in (benchmark_path.parent / BFCL_JSON_QUESTION_FILE).read_text(encoding="utf-8").splitlines()
    ]
    nemo_ids = [
        json.loads(line)["task_id"]
        for line in (benchmark_path.parent / NEMO_EVALUATOR_ROOT / NEMO_DATASET_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert all(row["held_out_hit"] is False for row in rows)
    assert all(row["template_id"] != HELD_OUT_TEMPLATE for row in rows)
    assert expected_ids == bfcl_ids == nemo_ids


def test_the_manifest_reports_enforcement_and_hashes_its_evidence(held_out_run) -> None:
    _, output_dir = held_out_run
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    held_out = manifest["held_out"]

    assert held_out["evaluated"] is True
    assert held_out["contract_version"] == "1.0"
    assert str(held_out["source"]).endswith("held_out.yaml")
    assert held_out["policy"]["fixture_ref_count"] == 1
    assert held_out["policy"]["template_count"] == 1
    assert held_out["rows_dropped"] == 0
    assert held_out["stage_four"]["counts"]["templates_blocked"] == 1
    assert manifest["bias_applicability"]["B7"] == {"status": "applicable"}
    assert manifest["artifacts"]["held_out_bindings"]["content_hash"].startswith("sha256:")
    assert manifest["artifacts"]["held_out_scan"]["content_hash"].startswith("sha256:")
    assert manifest["artifacts"]["held_out_normalized"]["content_hash"].startswith("sha256:")


def test_a_policy_that_reserves_nothing_is_enforced_but_proves_nothing(tmp_path: Path) -> None:
    pack = _prepare_pack(
        tmp_path,
        held_out={
            "version": "0.1.0",
            "fixtures": {},
            "templates": [],
            "policy": {"fixtures_in_backend_state": True, "seed": 0},
        },
    )

    benchmark_path = generate_bfcl(_write_config(tmp_path, pack, "empty-held-out.yaml"))

    manifest = json.loads((benchmark_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["held_out"]["evaluated"] is True
    assert manifest["held_out"]["rows_scanned"] > 0
    assert manifest["bias_applicability"]["B7"] == {
        "status": "na",
        "reason": "held_out policy reserves no fixture row or template",
    }


def test_generation_removes_reserved_rows_from_every_oracle_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _default_policy()
    policy["policy"]["fixtures_in_backend_state"] = False
    pack = _prepare_pack(tmp_path, held_out=policy)
    assertions_path = pack / "assertions.py"
    assertions_path.write_text(
        assertions_path.read_text(encoding="utf-8")
        + f"""

# Import-time guard proving assertion code sees the projected pack mirror.
import json as _json
from pathlib import Path as _Path
if any(
    str(row.get("book_id")) == {HELD_OUT_BOOK!r}
    for row in _json.loads(_Path(__file__).with_name("fixtures.json").read_text())["books"]
):
    raise RuntimeError("assertion module reopened a held-out fixture")
""",
        encoding="utf-8",
    )
    observed_book_ids: list[set[str]] = []
    observed_fixture_sources: list[Path | None] = []
    validated_templates: list[tuple[str, HeldOutPolicy | None]] = []
    run_episode = ProcessWorker.run_episode
    expand_template = oracle_validation.expand_template

    def recording_run_episode(self: ProcessWorker, **kwargs: Any) -> list[Any]:
        fixtures = kwargs.get("fixtures")
        if isinstance(fixtures, dict) and isinstance(fixtures.get("books"), list):
            observed_fixture_sources.append(kwargs.get("fixture_source_path"))
            observed_book_ids.append(
                {
                    str(row["book_id"])
                    for row in fixtures["books"]
                    if isinstance(row, dict) and "book_id" in row
                }
            )
        return run_episode(self, **kwargs)

    monkeypatch.setattr(ProcessWorker, "run_episode", recording_run_episode)

    def recording_expand_template(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        template = args[1]
        validated_templates.append((str(template["template_id"]), kwargs.get("held_out")))
        return expand_template(*args, **kwargs)

    monkeypatch.setattr(oracle_validation, "expand_template", recording_expand_template)

    benchmark_path = generate_bfcl(_write_config(tmp_path, pack, "state-held-out.yaml"))

    assert benchmark_path.is_file()
    assert observed_book_ids
    assert all(HELD_OUT_BOOK not in identifiers for identifiers in observed_book_ids)
    assert observed_fixture_sources
    # The wrapper sees both the outer projection request and its recursive execution
    # against the temporary mirror.
    assert pack / "fixtures.json" in observed_fixture_sources
    assert set(observed_fixture_sources) <= {pack / "fixtures.json", None}

    # The reserved template is still part of the pack contract. Validation compiles
    # it with template blocking disabled but fixture reservations left intact.
    report = json.loads(
        (benchmark_path.parent / "stage_cache" / ORACLE_VALIDATION_REPORT).read_text(
            encoding="utf-8"
        )
    )
    check = next(item for item in report["checks"] if item["id"] == 7)
    assert check["status"] == "pass"
    assert "held_out_templates_not_compiled" not in check
    assert report["gold_eligible"] is True
    reserved_policy = next(
        validation_policy
        for template_id, validation_policy in validated_templates
        if template_id == HELD_OUT_TEMPLATE
    )
    assert reserved_policy is not None
    assert reserved_policy.template_ids == ()
    assert fixture_ref("books", HELD_OUT_BOOK) in reserved_policy.fixture_refs


def test_a_policy_that_keeps_rows_in_state_still_compiles_the_reserved_template(
    held_out_run,
) -> None:
    _, output_dir = held_out_run
    report = json.loads((output_dir / "stage_cache" / ORACLE_VALIDATION_REPORT).read_text(encoding="utf-8"))
    check = next(item for item in report["checks"] if item["id"] == 7)

    # Stage 4 never binds the reserved template, but the pack must still be able to
    # state it: a private held-out run compiles it from this same pack.
    assert check["status"] == "pass"
    assert "held_out_templates_not_compiled" not in check
    assert report["gold_eligible"] is True


def test_reserved_template_validation_uses_the_state_each_policy_allows() -> None:
    in_state = HeldOutPolicy.from_normalized(_default_policy())
    out_of_state_data = _default_policy()
    out_of_state_data["policy"]["fixtures_in_backend_state"] = False
    out_of_state = HeldOutPolicy.from_normalized(out_of_state_data)

    # A private representative may use reserved rows only when the oracle still has
    # them. Otherwise only the template block opens and fixture reservations remain.
    assert _representative_held_out_policy(in_state, HELD_OUT_TEMPLATE) is None
    projected = _representative_held_out_policy(out_of_state, HELD_OUT_TEMPLATE)
    assert projected is not None
    assert projected.template_ids == ()
    assert projected.fixture_refs == out_of_state.fixture_refs
    assert _representative_held_out_policy(out_of_state, "public-template") is out_of_state


def _probe_pack(*, result_class: str, book_id: str, rows_in_state: bool = False) -> SimpleNamespace:
    policy = _default_policy()
    policy["policy"]["fixtures_in_backend_state"] = rows_in_state
    return SimpleNamespace(
        held_out=policy,
        validation_cases=[
            {
                "id": "probe_reserved_book",
                "tool": "get_book_status",
                "arguments": {"book_id": book_id},
                "expect": {"result_class": result_class, "error_code": None},
            }
        ],
    )


def test_a_probe_that_names_a_reserved_value_is_not_rejected_statically() -> None:
    pack = _probe_pack(result_class="success", book_id=HELD_OUT_BOOK)

    policy = held_out_policy(pack)

    # Validation cannot infer from an arbitrary argument name that the tool will
    # dereference this value. It must run the probe before assigning blame.
    assert policy.fixtures_in_backend_state is False


def test_a_probe_a_removed_row_can_still_satisfy_stays_valid() -> None:
    negative = _probe_pack(result_class="structured_error", book_id=HELD_OUT_BOOK)
    unreserved = _probe_pack(result_class="success", book_id="BK-100")
    # The same probe is honest while the rows stay in state, so the check must not fire.
    in_state = _probe_pack(result_class="success", book_id=HELD_OUT_BOOK, rows_in_state=True)

    assert held_out_policy(negative).fixtures_in_backend_state is False
    assert held_out_policy(unreserved).fixtures_in_backend_state is False
    assert held_out_policy(in_state).fixtures_in_backend_state is True


def test_not_found_probe_failure_names_matching_held_out_arguments() -> None:
    pack = _probe_pack(result_class="success", book_id=HELD_OUT_BOOK)
    policy = held_out_policy(pack)
    case = pack.validation_cases[0]

    context = _held_out_not_found_context(case, policy, "not_found")

    assert context == [
        {
            "argument": "book_id",
            "value": HELD_OUT_BOOK,
            "collection": "books",
            "fixture_ref": fixture_ref("books", HELD_OUT_BOOK),
        }
    ]
    assert _held_out_not_found_context(case, policy, "invalid_argument") == []


def _projection_inputs(*, fixtures_in_backend_state: bool) -> dict[str, Any]:
    return {
        "manifest": {"primary_keys": {"books": "book_id"}},
        "fixtures": {
            "books": [{"book_id": "BK-100"}, {"book_id": HELD_OUT_BOOK}, {"title": "unkeyed"}],
            "patrons": [{"patron_id": "P-1"}],
        },
        "held_out": {
            "fixtures": {"books": [HELD_OUT_BOOK]},
            "templates": [],
            "policy": {
                "fixtures_in_backend_state": fixtures_in_backend_state,
                "seed": 0,
            },
        },
    }


def test_a_policy_that_keeps_reserved_rows_in_state_projects_nothing() -> None:
    inputs = _projection_inputs(fixtures_in_backend_state=True)

    assert oracle_runtime_fixtures(**inputs) is inputs["fixtures"]


def test_isolating_oracle_state_drops_only_the_reserved_rows() -> None:
    inputs = _projection_inputs(fixtures_in_backend_state=False)

    projected = oracle_runtime_fixtures(**inputs)

    # A row the policy cannot identify stays: dropping it would withhold state the
    # pack never reserved.
    assert projected["books"] == [{"book_id": "BK-100"}, {"title": "unkeyed"}]
    assert projected["patrons"] == inputs["fixtures"]["patrons"]
    # Binding and publication enforcement still need the complete inventory.
    assert [row.get("book_id") for row in inputs["fixtures"]["books"]] == [
        "BK-100",
        HELD_OUT_BOOK,
        None,
    ]


def test_a_pack_without_a_policy_keeps_its_whole_seed_state() -> None:
    fixtures = {"books": [{"book_id": HELD_OUT_BOOK}]}

    assert oracle_runtime_fixtures(manifest={}, fixtures=fixtures, held_out=None) is fixtures
    assert oracle_runtime_fixtures(manifest={}, fixtures=None, held_out=None) is None


def test_run_invalidation_removes_the_normalized_held_out_policy(tmp_path: Path) -> None:
    config = SimpleNamespace(output_dir=str(tmp_path), expt_name="run")
    cache = tmp_path / "run" / "stage_cache"
    cache.mkdir(parents=True)
    normalized = cache / "held_out_normalized.json"
    normalized.write_text("{}\n", encoding="utf-8")

    pipeline._invalidate_final_outputs(config)

    assert not normalized.exists()


def test_missing_normalized_policy_stops_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        final_output,
    )

    pack = _prepare_pack(tmp_path, held_out=_default_policy())
    config_path = _write_config(tmp_path, pack, "missing-normalized.yaml")
    real_policy = final_output.held_out_policy

    def delete_normalized(loaded_pack):  # type: ignore[no-untyped-def]
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cache = (
            Path(config_data["output_dir"])
            / config_data["expt_name"]
            / "stage_cache"
        )
        (cache / "held_out_normalized.json").unlink(missing_ok=True)
        return real_policy(loaded_pack)

    monkeypatch.setattr(final_output, "held_out_policy", delete_normalized)

    with pytest.raises(ValueError, match="requires a valid.*held_out_normalized"):
        generate_bfcl(config_path)

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config_data["output_dir"]) / config_data["expt_name"]
    assert not (run_dir / "benchmark_raw.parquet").exists()
    assert not (run_dir / "benchmark.parquet").exists()
    assert not (run_dir / "run_manifest.json").exists()


def test_a_leak_that_reaches_publication_aborts_before_anything_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = _prepare_pack(tmp_path, held_out=_default_policy())
    config_path = _write_config(tmp_path, pack, "leaky.yaml")

    class _UnenforcedPolicy:
        """Stand in for a Stage 4 that records the policy but never applies it."""

        def __init__(self, policy: HeldOutPolicy) -> None:
            self._policy = policy

        def blocks_template(self, template_id: object) -> bool:
            return False

        def blocks_fixture(self, reference: str) -> bool:
            return False

        def as_lineage(self) -> dict[str, Any]:
            return self._policy.as_lineage()

    real_policy = expand_stage.held_out_policy
    monkeypatch.setattr(
        expand_stage,
        "held_out_policy",
        lambda pack: _UnenforcedPolicy(real_policy(pack)),
    )

    with pytest.raises(HeldOutLeakError, match="bind held-out material"):
        generate_bfcl(config_path)

    output_dir = Path(yaml.safe_load(config_path.read_text(encoding="utf-8"))["output_dir"])
    run_dir = output_dir / "bfcl_tiny_library_validation"
    assert not (run_dir / "benchmark.parquet").exists()
    assert not (run_dir / "benchmark_raw.parquet").exists()
    assert not (run_dir / "run_manifest.json").exists()

    scan = json.loads((run_dir / "stage_cache" / HELD_OUT_SCAN).read_text(encoding="utf-8"))
    assert scan["counts"]["rows_hit"] == len(scan["hits"])
    assert scan["counts"]["rows_hit"] > 0
    assert scan["counts"]["rows_blocked"] == scan["counts"]["rows_hit"]
    assert scan["counts"]["rows_dropped"] == 0
    assert scan["counts"]["rows_published"] == 0
    assert scan["counts"]["planned_rows_published"] > 0
    assert scan["action"] == "abort"
    matched = {
        (hit["matched_template_id"], tuple(hit["matched_fixture_refs"]))
        for hit in scan["hits"]
    }
    assert (HELD_OUT_TEMPLATE, ()) in matched or any(
        fixture_ref("books", HELD_OUT_BOOK) in references for _, references in matched
    )


def test_publication_refuses_a_run_whose_stage_four_left_no_evidence(tmp_path: Path) -> None:
    policy = HeldOutPolicy.from_normalized(_default_policy())
    config = SimpleNamespace(output_dir=str(tmp_path), expt_name="run")

    with pytest.raises(HeldOutLeakError, match="Stage 4 wrote no"):
        load_binding_report(config, policy)

    cache = tmp_path / "run" / "stage_cache"
    cache.mkdir(parents=True)
    valid_report = {
        "contract_version": "1.0",
        "policy": policy.as_lineage(),
        "blocked_templates": [],
        "blocked_fixture_refs": [],
        "counts": {
            "bind_attempts": 1,
            "templates_blocked": 0,
            "fixture_refs_blocked": 0,
            "tasks_expanded": 1,
        },
    }
    (cache / HELD_OUT_BINDINGS).write_text(
        json.dumps(
            {
                **valid_report,
                "policy": {**policy.as_lineage(), "version": "other"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(HeldOutLeakError, match="different held-out policy"):
        load_binding_report(config, policy)

    (cache / HELD_OUT_BINDINGS).write_text(
        json.dumps(
            {
                **valid_report,
                "blocked_fixture_refs": ["other.9"],
                "counts": {
                    **valid_report["counts"],
                    "fixture_refs_blocked": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(HeldOutLeakError, match="sorted unique subset"):
        load_binding_report(config, policy)

    (cache / HELD_OUT_BINDINGS).write_text(
        json.dumps(
            {
                **valid_report,
                "counts": {
                    **valid_report["counts"],
                    "bind_attempts": -1,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(HeldOutLeakError, match="non-negative integers"):
        load_binding_report(config, policy)

    (cache / HELD_OUT_BINDINGS).write_text(json.dumps(valid_report), encoding="utf-8")
    with pytest.raises(HeldOutLeakError, match="does not match Stage 4 output"):
        load_binding_report(config, policy, expected_tasks_expanded=2)


def _slot_pack(held_out: dict[str, Any] | None, *, rows: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        manifest={"pack_id": "p", "version": "1.0", "primary_keys": {"things": "thing_id"}},
        fixtures={"things": [{"thing_id": f"T-{index}"} for index in range(rows)]},
        tools=[],
        templates=[
            {
                "template_id": f"tpl-{index}",
                "category": "shared",
                "slots": {"thing_id": {"source": "fixture:things.thing_id"}},
            }
            for index in range(1)
        ],
        held_out=held_out,
    )


def test_a_slot_whose_every_row_is_reserved_stops_expansion(tmp_path: Path) -> None:
    pack = _slot_pack(
        {
            "version": "1",
            "fixtures": {"things": ["T-0", "T-1"]},
            "templates": [],
            "policy": {"fixtures_in_backend_state": True, "seed": 0},
        }
    )
    config = SimpleNamespace(
        task_generation={"tasks_per_category": 1},
        random_seed=1,
        output_dir=str(tmp_path),
        expt_name="run",
    )

    with pytest.raises(ExpansionError, match="reserves every row its filter matched"):
        run_expand(config, pack)


def test_a_budget_the_reservations_starve_is_a_pack_error(tmp_path: Path) -> None:
    pack = _slot_pack(
        {
            "version": "1",
            "fixtures": {"things": ["T-1"]},
            "templates": [],
            "policy": {"fixtures_in_backend_state": True, "seed": 0},
        },
        rows=3,
    )
    config = SimpleNamespace(
        task_generation={"tasks_per_category": 3},
        random_seed=1,
        output_dir=str(tmp_path),
        expt_name="run",
    )

    with pytest.raises(ExpansionError, match="templates and fixtures"):
        run_expand(config, pack)


def test_a_reserved_template_cannot_silently_shrink_its_category(tmp_path: Path) -> None:
    pack = SimpleNamespace(
        manifest={"pack_id": "p", "version": "1.0"},
        fixtures={},
        tools=[],
        templates=[
            {"template_id": "tpl-open", "category": "shared", "slots": {}},
            {"template_id": "tpl-held", "category": "shared", "slots": {}},
        ],
        held_out={
            "version": "1",
            "fixtures": {},
            "templates": ["tpl-held"],
            "policy": {"fixtures_in_backend_state": True, "seed": 0},
        },
    )
    config = SimpleNamespace(
        task_generation={"tasks_per_category": 2},
        random_seed=1,
        output_dir=str(tmp_path),
        expt_name="run",
    )

    with pytest.raises(ExpansionError, match="category 'shared' bound 1 of 2"):
        run_expand(config, pack)


def test_reserving_a_whole_category_is_a_shortfall(tmp_path: Path) -> None:
    pack = SimpleNamespace(
        manifest={"pack_id": "p", "version": "1.0"},
        fixtures={},
        tools=[],
        templates=[
            {"template_id": "tpl-open", "category": "open", "slots": {}},
            {"template_id": "tpl-held", "category": "reserved", "slots": {}},
        ],
        held_out={
            "version": "1",
            "fixtures": {},
            "templates": ["tpl-held"],
            "policy": {"fixtures_in_backend_state": True, "seed": 0},
        },
    )
    config = SimpleNamespace(
        task_generation={"tasks_per_category": 1},
        random_seed=1,
        output_dir=str(tmp_path),
        expt_name="run",
    )

    with pytest.raises(ExpansionError, match="category 'reserved' has no bindable template"):
        run_expand(config, pack)


def test_a_second_category_starved_by_the_same_rows_is_still_reported(tmp_path: Path) -> None:
    pack = SimpleNamespace(
        manifest={"pack_id": "p", "version": "1.0", "primary_keys": {"things": "thing_id"}},
        fixtures={
            "things": [{"thing_id": f"T-{index}"} for index in range(3)],
            "patrons": [{"patron_id": "P-1"}, {"patron_id": "P-2"}],
        },
        tools=[],
        templates=[
            {
                "template_id": "tpl-a",
                "category": "a",
                "slots": {
                    "thing_id": {"source": "fixture:things.thing_id"},
                    "patron_id": {"source": "fixture:patrons.patron_id"},
                },
            },
            {
                "template_id": "tpl-b",
                "category": "b",
                "slots": {"thing_id": {"source": "fixture:things.thing_id"}},
            },
        ],
        held_out={
            "version": "1",
            "fixtures": {"things": ["T-1", "T-2"]},
            "templates": [],
            "policy": {"fixtures_in_backend_state": True, "seed": 0},
        },
    )
    config = SimpleNamespace(
        task_generation={"tasks_per_category": 2},
        random_seed=1,
        output_dir=str(tmp_path),
        expt_name="run",
    )

    with pytest.raises(ExpansionError, match="category 'b' bound 1 of 2"):
        run_expand(config, pack)


def test_reserving_every_template_stops_expansion(tmp_path: Path) -> None:
    pack = _slot_pack(
        {
            "version": "1",
            "fixtures": {},
            "templates": ["tpl-0"],
            "policy": {"fixtures_in_backend_state": True, "seed": 0},
        }
    )
    config = SimpleNamespace(
        task_generation={"tasks_per_category": 1},
        random_seed=1,
        output_dir=str(tmp_path),
        expt_name="run",
    )

    with pytest.raises(ExpansionError, match="reserves every template"):
        run_expand(config, pack)
