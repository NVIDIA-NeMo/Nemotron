from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    EXPORT_FORMATS,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    validate_tool_definition,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import load_pack
from nemotron.steps.byob.runtime.benchmark_families.registry import list_families

BFCL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "bfcl" / "config"
BYOB_DIR = BFCL_CONFIG_DIR.parents[1]


def _copy_tiny_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    shutil.copytree(
        BYOB_DIR / "data" / "tiny_oracle_pack",
        pack,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return pack


def _write_tiny_config(tmp_path: Path, name: str, **overrides: Any) -> Path:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config_data.get(key), dict):
            config_data[key].update(value)
        else:
            config_data[key] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return path


def _edit_pack_yaml(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_list_families_includes_bfcl_and_mcq() -> None:
    assert list_families() == ["bfcl", "mcq"]


def test_all_bfcl_configs_pin_family() -> None:
    for name in (
        "tiny.yaml",
        "default.yaml",
        "smoke.example.yaml",
        "publication.example.yaml",
        "publication.paraphrase.example.yaml",
    ):
        data = yaml.safe_load((BFCL_CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert data["family"] == "bfcl", name


@pytest.mark.parametrize(
    ("historical_name", "maintained_name"),
    (
        ("banking_vn.gold.yaml", "publication.example.yaml"),
        (
            "banking_vn.gold.paraphrase.yaml",
            "publication.paraphrase.example.yaml",
        ),
    ),
)
def test_frozen_banking_readme_links_resolve_to_explicit_config_pointers(
    historical_name: str,
    maintained_name: str,
) -> None:
    pointer = BFCL_CONFIG_DIR / historical_name
    assert pointer.is_file()
    text = pointer.read_text(encoding="utf-8")
    assert maintained_name in text
    assert "not a second runnable copy" in text


@pytest.mark.parametrize("extension", ("x-mutates", "x-requires-confirmation"))
def test_tool_extensions_require_literal_booleans(extension: str) -> None:
    tool = {
        "type": "function",
        extension: "false",
        "function": {
            "name": "get_record",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    assert {
        "reason": "tool_extension_not_boolean",
        "field": extension,
        "value": "false",
    } in validate_tool_definition(tool)


def test_pack_rejects_canonical_backend_and_endpoint_files(tmp_path: Path) -> None:
    pack = _copy_tiny_pack(tmp_path)
    (pack / "endpoint_config.yaml").write_text("{}\n", encoding="utf-8")
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "dual-oracle.yaml",
            oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )

    with pytest.raises(
        ValueError,
        match="cannot contain both canonical backend.py and endpoint_config.yaml",
    ):
        load_pack(config)


def test_publication_example_can_bind_the_closest_uniform_bfcl_v1_scale() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
        DedupBalancingDecision,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
        balance_publication_set,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
        _select_round_robin,
        expand_template,
        group_by_category,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        render_task,
        resolve_render_contract,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        build_plan,
    )

    path = BFCL_CONFIG_DIR / "publication.example.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    templates = yaml.safe_load(
        (BYOB_DIR / "data" / "banking_vn_oracle_pack" / "task_templates.yaml").read_text(encoding="utf-8")
    )
    category_count = len({str(template["category"]) for template in templates})
    requested_samples = category_count * int(raw["task_generation"]["tasks_per_category"])

    config = BfclConfig.from_yaml(path)
    pack = load_pack(config)
    budget = int(config.task_generation["tasks_per_category"])
    candidate_budget = int(config.task_generation["candidate_tasks_per_category"])
    candidate_by_category = {
        category: _select_round_robin(
            [
                expand_template(
                    pack,
                    template,
                    candidate_budget,
                    int(config.random_seed or 0),
                )
                for template in grouped
            ],
            candidate_budget,
        )
        for category, grouped in group_by_category(pack.templates).items()
    }
    candidates = [task for category_tasks in candidate_by_category.values() for task in category_tasks]
    templates_by_id = {str(template["template_id"]): template for template in pack.templates}
    render_contract = resolve_render_contract(config, pack, templates_by_id)
    candidate_surfaces = {}
    for task in candidates:
        template = templates_by_id[str(task["template_id"])]
        plan = build_plan(template, task)
        task["num_tool_calls"] = plan["num_tool_calls"]
        task["is_multi_turn"] = plan["is_multi_turn"]
        candidate_surfaces[str(task["task_id"])] = render_task(
            pack,
            template,
            task,
            plan,
            language=render_contract["language"],
            prompt_bundle=render_contract["prompt_bundle"],
            tool_names=render_contract["tool_names"],
        )
    representative_decisions = [
        DedupBalancingDecision(
            task_id=str(task["task_id"]),
            selected=True,
            is_duplicate=False,
            selection_rank=index,
        )
        for index, task in enumerate(candidates)
    ]
    _, _, challenge_summary = balance_publication_set(
        config,
        candidates,
        candidate_surfaces,
        representative_decisions,
    )
    selected_by_category = {
        category: _select_round_robin(
            [
                expand_template(
                    pack,
                    template,
                    budget,
                    int(config.random_seed or 0),
                )
                for template in grouped
            ],
            budget,
        )
        for category, grouped in group_by_category(pack.templates).items()
    }
    selected = [task for category_tasks in selected_by_category.values() for task in category_tasks]
    # Out-of-scope templates do not share a slot vocabulary, so uniqueness is checked on
    # the whole binding rather than on the slots one template happens to declare.
    out_of_scope_cases = {
        (
            str(task["template_id"]),
            tuple(sorted((key, str(value)) for key, value in task["slots"].items())),
        )
        for task in selected_by_category["out_of_scope"]
    }
    create_dispute_tasks = expand_template(
        pack,
        templates_by_id["bn_create_dispute_single"],
        budget,
        int(config.random_seed or 0),
    )
    create_dispute_ids = {str(task["slots"]["transaction_id"]) for task in create_dispute_tasks}
    transactions = {str(row["transaction_id"]): row for row in pack.fixtures["transactions"]}
    open_dispute_transaction_ids = {
        str(row["transaction_id"])
        for row in pack.fixtures["disputes"]
        if row["status"] in {"open", "in_review", "awaiting_confirmation"}
    }

    assert category_count == 6
    assert requested_samples == 1_392
    assert abs(requested_samples - 1_390) == 2
    assert {category: len(tasks) for category, tasks in selected_by_category.items()} == {
        "balance_inquiry": 232,
        "transaction_status": 232,
        "transfer": 232,
        "qr_payment": 232,
        "dispute": 232,
        "out_of_scope": 232,
    }
    assert len(selected) == requested_samples
    assert len({str(task["task_id"]) for task in selected}) == requested_samples
    assert len(candidates) == 2_824
    assert all(len(tasks) >= budget for tasks in candidate_by_category.values())
    candidate_difficulty = {
        difficulty: sum(task["difficulty"] == difficulty for task in candidates)
        for difficulty in ("easy", "medium", "hard")
    }
    assert candidate_difficulty == {"easy": 753, "medium": 1_013, "hard": 1_058}
    assert candidate_difficulty["hard"] >= 626
    assert challenge_summary["selected_count"] == requested_samples
    assert challenge_summary["actual_counts"]["difficulty"] == {
        "easy": 348,
        "hard": 626,
        "medium": 418,
    }
    assert challenge_summary["actual_counts"]["turn_class"] == {
        "multi_turn": 418,
        "single_turn": 974,
    }
    assert challenge_summary["actual_counts"]["tool_call_count"] == {
        "0": 376,
        "1": 613,
        "2": 302,
        "3+": 101,
    }
    positive_call_total = 613 + 302 + 101
    assert 613 / positive_call_total == pytest.approx(0.60, abs=0.05)
    assert 302 / positive_call_total == pytest.approx(0.30, abs=0.05)
    assert 101 / positive_call_total == pytest.approx(0.10, abs=0.05)
    # The declared policy mix is the release's claim about what it actually tests, so
    # the shapes a candidate is most likely to fail are pinned rather than left to
    # whatever expansion inventory happened to produce.
    assert challenge_summary["actual_counts"]["turn_policy"] == {
        "clarify_only": 144,
        "confirmation": 98,
        "correction": 111,
        "dependent_call": 209,
        "irrelevant": 232,
        "missing_slot": 139,
        "multi_tool": 97,
        "negative_path": 97,
        "single_turn": 265,
    }
    # No two published rows call the same tools with the same arguments against the
    # same state, so 1,392 rows are 1,392 distinct behaviours rather than repeats.
    assert challenge_summary["group_diversity"]["execution_case_hash"] == {
        "unique": requested_samples,
        "max_reuse": 1,
        "cap": 1,
    }
    assert challenge_summary["group_diversity"]["intent"]["max_reuse"] <= 120
    assert challenge_summary["unmet_targets"] == []
    # Every row offers the whole catalog, so tool selection is a nine-way choice.
    assert {len(set(task["tools_present"] or [])) for task in candidates} == {len(pack.tools)}
    assert len(out_of_scope_cases) == 232
    assert len(create_dispute_tasks) == 72
    assert len(create_dispute_ids) == 18
    assert not create_dispute_ids & open_dispute_transaction_ids
    assert all(
        transactions[transaction_id]["dispute_eligible"] is True
        and transactions[transaction_id]["direction"] == "debit"
        and transactions[transaction_id]["status"] == "succeeded"
        for transaction_id in create_dispute_ids
    )
    assert config.config_status == "resolved"
    assert config.lineage.policy == "strict_separation"
    assert config.oracle_runtime.worker == "process"
    assert config.surface_quality_validation["enabled"] is True
    assert config.semantic_deduplication_config["enabled"] is True
    assert config.semantic_deduplication_config["remove_duplicates"] is False
    assert config.semantic_deduplication_config["unmet_target_policy"] == "abort"
    assert config.task_generation["target_published_tasks"] == requested_samples
    assert config.exports == {
        "bfcl_json": True,
        "nemo_evaluator_bundle": True,
    }


def test_publication_paraphrase_example_is_guarded_and_fail_closed() -> None:
    config = BfclConfig.from_yaml(BFCL_CONFIG_DIR / "publication.paraphrase.example.yaml")
    pack = load_pack(config)
    role = config.lineage.roles["paraphrase"]
    eligible = [template for template in pack.templates if (template.get("paraphrase") or {}).get("allowed") is True]

    assert config.lineage.policy == "strict_separation"
    assert role.enabled is True
    assert role.model_config["provider"] == "nvidia_inference_api"
    assert role.model_config["canonical_id"] == "nvidia-inference-api/azure/openai/gpt-5.6-sol"
    assert role.model_config["api_key_env"] == "NGC_API_KEY"
    # This route rejects top_p, so the profile must not declare it.
    assert "top_p" not in role.model_config["inference_parameters"]
    assert config.surface_generation["model_paraphrase_enabled"] is True
    assert config.surface_generation["paraphrases_per_template"] == 1
    assert len(eligible) >= 17
    assert all(template["paraphrase"]["max_variants"] == 1 for template in eligible)
    assert config.task_generation["target_published_tasks"] == 1_392
    assert config.semantic_deduplication_config["max_exact_surface_reuse"] == 8
    assert config.semantic_deduplication_config["min_exact_surface_ratio"] == 0.15
    assert config.semantic_deduplication_config["max_execution_case_reuse"] == 1
    assert config.semantic_deduplication_config["max_rows_per_intent"] == 120
    assert config.semantic_deduplication_config["unmet_target_policy"] == "abort"
    # A paraphrase run rewords the same executable cases, so the release states which
    # conversation shapes it buys with that scale rather than inheriting the mix.
    assert sum(config.task_generation["policy_mix"].values()) == pytest.approx(1.0)
    assert config.task_generation["policy_mix"]["clarify_only"] > 0.1
    assert config.task_generation["difficulty_mix"]["hard"] == 0.45


def test_candidate_category_budget_cannot_be_smaller_than_publication_budget(
    tmp_path: Path,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        "candidate-underflow.yaml",
        task_generation={
            "tasks_per_category": 4,
            "candidate_tasks_per_category": 3,
        },
    )

    with pytest.raises(
        ValueError,
        match="candidate_tasks_per_category must be greater than or equal",
    ):
        BfclConfig.from_yaml(path)


def test_tiny_config_loads_as_smoke(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    temp_config = tmp_path / "tiny.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = BfclConfig.from_yaml(temp_config)

    assert config.family == "bfcl"
    assert config.stage == "all"
    assert config.lineage.policy == "smoke_no_publication"
    assert config.oracle_runtime.worker == "process"
    assert config.oracle_runtime.tool_timeout_s == 5.0
    assert config.lineage.roles["paraphrase"].enabled is False
    assert config.surface_generation.get("paraphrases_per_template") == 0
    assert config.surface_quality_validation["contract_version"] == "1.1"
    assert config.oracle_pack.manifest_path.name == "manifest.yaml"


def test_config_rejects_unknown_surface_quality_contract_version(
    tmp_path: Path,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        "unknown-surface-contract.yaml",
        surface_quality_validation={"contract_version": "2.0"},
    )

    with pytest.raises(ValueError, match="contract_version must be '1.1'"):
        BfclConfig.from_yaml(path)


def test_config_locks_stage_eleven_contract_version(tmp_path: Path) -> None:
    config = BfclConfig.from_yaml(_write_tiny_config(tmp_path, "stage-eleven-default.yaml"))
    assert config.semantic_deduplication_config["contract_version"] == "1.0"
    assert config.semantic_deduplication_config["unmet_target_policy"] == "abort"

    path = _write_tiny_config(
        tmp_path,
        "unknown-stage-eleven-contract.yaml",
        semantic_deduplication_config={"contract_version": "2.0"},
    )
    with pytest.raises(ValueError, match="contract_version must be '1.0'"):
        BfclConfig.from_yaml(path)


@pytest.mark.parametrize("policy", ["continue", 1, ["abort"]])
def test_stage_eleven_rejects_unknown_unmet_target_policy(
    tmp_path: Path,
    policy: object,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        "invalid-unmet-target-policy.yaml",
        semantic_deduplication_config={"unmet_target_policy": policy},
    )

    with pytest.raises(ValueError, match="unmet_target_policy must be"):
        BfclConfig.from_yaml(path)


def test_enabled_stage_eleven_requires_stage_ten(tmp_path: Path) -> None:
    path = _write_tiny_config(
        tmp_path,
        "dedup-without-quality.yaml",
        semantic_deduplication_config={
            "enabled": True,
            "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
            "n_clusters": 20,
            "eps": 0.08,
            "remove_duplicates": True,
        },
    )
    with pytest.raises(
        ValueError,
        match="surface_quality_validation.enabled must be true",
    ):
        BfclConfig.from_yaml(path)

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "dedup-after-quality.yaml",
            surface_quality_validation={"enabled": True},
            semantic_deduplication_config={
                "enabled": True,
                "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
                "n_clusters": 20,
                "eps": 0.08,
                "remove_duplicates": True,
            },
        )
    )
    assert config.semantic_deduplication_config["enabled"] is True


@pytest.mark.parametrize(
    "missing_key",
    ["model_identifier", "n_clusters", "eps", "remove_duplicates"],
)
def test_enabled_stage_eleven_requires_complete_config(
    tmp_path: Path,
    missing_key: str,
) -> None:
    dedup = {
        "enabled": True,
        "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
        "n_clusters": 20,
        "eps": 0.08,
        "remove_duplicates": True,
    }
    del dedup[missing_key]
    path = _write_tiny_config(
        tmp_path,
        f"dedup-missing-{missing_key}.yaml",
        surface_quality_validation={"enabled": True},
        semantic_deduplication_config=dedup,
    )

    with pytest.raises(
        ValueError,
        match=rf"missing required keys.*{missing_key}",
    ):
        BfclConfig.from_yaml(path)


@pytest.mark.parametrize("eps", [0, -0.1, 1, 1.1])
def test_stage_eleven_requires_a_curator_cosine_distance_threshold(
    tmp_path: Path,
    eps: float,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        f"dedup-invalid-eps-{eps}.yaml",
        surface_quality_validation={"enabled": True},
        semantic_deduplication_config={
            "enabled": True,
            "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
            "n_clusters": 20,
            "eps": eps,
            "remove_duplicates": True,
        },
    )

    with pytest.raises(ValueError, match="eps must be between 0 and 1"):
        BfclConfig.from_yaml(path)


def test_stage_eleven_normalizes_the_model_identifier(tmp_path: Path) -> None:
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "dedup-normalized-model.yaml",
            surface_quality_validation={"enabled": True},
            semantic_deduplication_config={
                "enabled": True,
                "model_identifier": " sentence-transformers/all-MiniLM-L6-v2 ",
                "n_clusters": 20,
                "eps": 0.08,
                "remove_duplicates": True,
            },
        )
    )

    assert config.semantic_deduplication_config["model_identifier"] == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize(
    ("preference", "message"),
    [
        ([], "must be a non-empty list"),
        (["template", "template"], "must not repeat"),
        (["template", "oracle"], "unknown sources: oracle"),
    ],
)
def test_stage_eleven_validates_representative_source_preference(
    tmp_path: Path,
    preference: list[str],
    message: str,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        "dedup-source-preference.yaml",
        surface_quality_validation={"enabled": True},
        semantic_deduplication_config={
            "enabled": True,
            "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
            "n_clusters": 20,
            "eps": 0.08,
            "remove_duplicates": True,
            "representative_source_preference": preference,
        },
    )

    with pytest.raises(ValueError, match=message):
        BfclConfig.from_yaml(path)


def test_config_allows_deterministic_surface_quality_without_a_judge(
    tmp_path: Path,
) -> None:
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "deterministic-surface-quality.yaml",
            surface_quality_validation={
                "enabled": True,
                "drop_authority": False,
            },
        )
    )

    assert config.surface_quality_validation["enabled"] is True
    assert config.lineage.roles["surface_judge"].enabled is False


def test_config_rejects_judge_drop_authority_without_a_judge(
    tmp_path: Path,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        "judge-authority-without-judge.yaml",
        surface_quality_validation={"enabled": True, "drop_authority": True},
    )

    with pytest.raises(ValueError, match="drop_authority requires an enabled"):
        BfclConfig.from_yaml(path)


def test_default_config_loads() -> None:
    config = BfclConfig.from_yaml(BFCL_CONFIG_DIR / "default.yaml")
    assert config.family == "bfcl"
    assert config.lineage.policy == "strict_separation"


def test_resolved_config_rejects_replacement_tokens(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
    config_data["config_status"] = "resolved"
    config_data["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"REPLACE_ME_.*lineage\.roles"):
        BfclConfig.from_yaml(path)


def test_rejects_non_bfcl_family(tmp_path: Path) -> None:
    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["family"] = "mcq"
    config_data["output_dir"] = str(tmp_path / "output")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match="family"):
        BfclConfig.from_yaml(bad_path)


def test_rejects_unknown_config_keys_and_lineage_policies(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BFCL config has unknown keys: task_generaton"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "unknown-top-level.yaml",
                task_generaton={"tasks_per_category": 999},
            )
        )

    with pytest.raises(ValueError, match="lineage.policy must be one of"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "bad-policy.yaml",
                lineage={"policy": "strict_seperation"},
            )
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"surface_generation": []}, "surface_generation must be a mapping"),
        (
            {"lineage": {"roles": {"profile": {"enabled": "false"}}}},
            "lineage.roles.profile.enabled must be a boolean",
        ),
        (
            {"task_generation": {"tasks_per_category": "4"}},
            "task_generation.tasks_per_category must be an integer",
        ),
    ],
)
def test_config_rejects_values_that_would_be_silently_coerced(
    tmp_path: Path,
    override: dict[str, Any],
    message: str,
) -> None:
    path = _write_tiny_config(
        tmp_path,
        f"bad-type-{len(message)}.yaml",
        **override,
    )

    with pytest.raises(ValueError, match=message):
        BfclConfig.from_yaml(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_timeouts(tmp_path: Path, value: float) -> None:
    path = _write_tiny_config(
        tmp_path,
        "non-finite-timeout.yaml",
        oracle_runtime={"tool_timeout_s": value},
    )

    with pytest.raises(ValueError, match="must be finite"):
        BfclConfig.from_yaml(path)


def test_week4_distribution_and_dedup_contracts_parse_strictly(
    tmp_path: Path,
) -> None:
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "week4-contract.yaml",
            task_generation={
                "tasks_per_category": 34,
                "max_turns": 6,
                "max_tool_calls": 3,
                "difficulty_mix": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
                "turn_mix": {"single_turn": 0.6, "multi_turn": 0.4},
                "tool_call_count_mix": {"1": 0.75, "2": 0.2, "3+": 0.05},
                "max_intent_share": 0.50,
            },
            semantic_deduplication_config={
                "enabled": False,
                "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
                "n_clusters": 20,
                "eps": 0.08,
                "remove_duplicates": True,
            },
        )
    )

    assert config.task_generation["max_turns"] == 6
    assert config.task_generation["difficulty_mix"]["medium"] == 0.5
    assert config.task_generation["max_intent_share"] == 0.50
    assert config.semantic_deduplication_config["eps"] == 0.08


@pytest.mark.parametrize(
    ("task_generation", "message"),
    [
        (
            {"difficulty_mix": {"easy": 0.6, "hard": 0.3}},
            "probabilities must sum to 1",
        ),
        (
            {"turn_mix": {"single_turn": True, "multi_turn": 0.0}},
            "must be a number",
        ),
        (
            {"tool_call_count_mix": {"1": 1.1, "2": -0.1}},
            "must be between 0 and 1",
        ),
        (
            {"policy_mix": {"single_turn": 0.5, "not_a_policy": 0.5}},
            r"policy_mix has unknown keys: not_a_policy",
        ),
        (
            {"max_intent_share": 0.0},
            "must be greater than 0 and at most 1",
        ),
        (
            {"max_intent_share": 1.01},
            "must be greater than 0 and at most 1",
        ),
    ],
)
def test_week4_mix_contract_rejects_invalid_probabilities(
    tmp_path: Path,
    task_generation: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "invalid-week4-mix.yaml",
                task_generation=task_generation,
            )
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_rows_per_intent", 0),
        ("max_execution_case_reuse", 0),
        ("max_rows_per_intent", 1.5),
        ("max_execution_case_reuse", "one"),
    ],
)
def test_repetition_caps_must_be_positive_integers(
    tmp_path: Path,
    key: str,
    value: Any,
) -> None:
    # A cap of zero or a non-integer would silently make the constraint meaningless
    # rather than bounding repetition, so it is rejected at config load.
    with pytest.raises(ValueError, match=f"semantic_deduplication_config.{key}"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                f"invalid-{key}-{value}.yaml",
                semantic_deduplication_config={
                    "enabled": False,
                    "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
                    "n_clusters": 20,
                    "eps": 0.08,
                    "remove_duplicates": True,
                    key: value,
                },
            )
        )


def test_generation_mix_rejects_targets_missing_from_template_inventory(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "unavailable-difficulty.yaml",
            task_generation={"difficulty_mix": {"impossible": 1.0}},
        )
    )
    with pytest.raises(ValueError, match="unavailable template difficulties"):
        load_pack(config)


def test_strict_lineage_requires_distinct_canonical_model_identities(
    tmp_path: Path,
) -> None:
    role = {
        "enabled": True,
        "model_config": {
            "alias": "writer",
            "provider": "nvidia",
            "model": "model-a",
            "canonical_id": "source::model-a@revision",
        },
    }
    with pytest.raises(ValueError, match="distinct canonical model identities"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "duplicate-role-model.yaml",
                lineage={
                    "policy": "strict_separation",
                    "roles": {"profile": role, "paraphrase": role},
                },
            )
        )


def test_enabled_lineage_role_requires_canonical_identity_and_rejects_secrets(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical_id"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "missing-canonical.yaml",
                lineage={
                    "roles": {
                        "profile": {
                            "enabled": True,
                            "model_config": {"provider": "nvidia", "model": "profile"},
                        }
                    }
                },
            )
        )
    with pytest.raises(ValueError, match="looks like a secret"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "inline-secret.yaml",
                lineage={
                    "roles": {
                        "profile": {
                            "enabled": False,
                            "model_config": {"api_key": "must-not-enter-config"},
                        }
                    }
                },
            )
        )


def test_reference_benchmark_is_allowlisted_and_content_addressed(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "reference.jsonl"
    samples.write_text(
        '{"sample_id":"ref-1","language":"vi","messages":[{"role":"user","content":"Xin chào"}]}\n',
        encoding="utf-8",
    )
    content_hash = f"sha256:{hashlib.sha256(samples.read_bytes()).hexdigest()}"
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "reference-contract.yaml",
            oracle_runtime={"allowed_roots": [str(tmp_path), str(BYOB_DIR / "data")]},
            reference_benchmark={
                "name": "vi-style",
                "samples_path": str(samples),
                "content_hash": content_hash,
            },
        )
    )
    assert config.reference_benchmark is not None
    assert config.reference_benchmark.samples_path == samples

    with pytest.raises(ValueError, match="does not match"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "reference-hash-mismatch.yaml",
                oracle_runtime={"allowed_roots": [str(tmp_path), str(BYOB_DIR / "data")]},
                reference_benchmark={
                    "name": "vi-style",
                    "samples_path": str(samples),
                    "content_hash": f"sha256:{'0' * 64}",
                },
            )
        )

    samples.write_text(
        '{"sample_id":"ref-1","language":"vi","messages":[{"role":"user","content":"Xin chào"}],'
        '"expected_tool_calls":[]}\n',
        encoding="utf-8",
    )
    leaked_hash = f"sha256:{hashlib.sha256(samples.read_bytes()).hexdigest()}"
    with pytest.raises(ValueError, match="oracle-truth fields"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "reference-truth-leak.yaml",
                oracle_runtime={"allowed_roots": [str(tmp_path), str(BYOB_DIR / "data")]},
                reference_benchmark={
                    "name": "vi-style",
                    "samples_path": str(samples),
                    "content_hash": leaked_hash,
                },
            )
        )


def test_held_out_pack_contract_validates_ids_and_enters_fingerprint(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
        pack_fingerprint,
    )

    pack_root = _copy_tiny_pack(tmp_path)
    manifest_path = pack_root / "manifest.yaml"
    _edit_pack_yaml(manifest_path, lambda manifest: manifest.update({"held_out": "held_out.yaml"}))
    held_out_path = pack_root / "held_out.yaml"
    held_out = {
        "version": "0.1.0",
        "fixtures": {"books": ["BK-300"]},
        "templates": ["lib_status_single"],
        "policy": {"fixtures_in_backend_state": True, "seed": 42},
    }
    held_out_path.write_text(yaml.safe_dump(held_out), encoding="utf-8")
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "held-out-contract.yaml",
            oracle_pack={"manifest_path": str(manifest_path)},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )

    pack = load_pack(config)
    assert pack.held_out is not None
    assert pack.held_out["fixtures"]["books"] == ["BK-300"]
    assert pack.held_out["source"] == "held_out.yaml"
    fingerprint_before = pack_fingerprint(pack.paths)

    held_out["policy"]["seed"] = 43
    held_out_path.write_text(yaml.safe_dump(held_out), encoding="utf-8")
    assert pack_fingerprint(pack.paths) != fingerprint_before

    held_out["fixtures"]["books"] = ["BK-UNKNOWN"]
    held_out_path.write_text(yaml.safe_dump(held_out), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown primary ids"):
        load_pack(config)


def test_held_out_policy_version_rejects_boolean_yaml_scalars(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )

    pack_root = _copy_tiny_pack(tmp_path)
    manifest_path = pack_root / "manifest.yaml"
    _edit_pack_yaml(manifest_path, lambda manifest: manifest.update({"held_out": "held_out.yaml"}))
    (pack_root / "held_out.yaml").write_text(
        yaml.safe_dump(
            {
                "version": True,
                "fixtures": {},
                "templates": [],
                "policy": {"fixtures_in_backend_state": True, "seed": 0},
            }
        ),
        encoding="utf-8",
    )
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "held-out-boolean-version.yaml",
            oracle_pack={"manifest_path": str(manifest_path)},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )

    with pytest.raises(ValueError, match="version must be a non-empty string"):
        load_pack(config)


def test_held_out_rejects_primary_ids_that_collapse_after_normalization(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )

    pack_root = _copy_tiny_pack(tmp_path)
    manifest_path = pack_root / "manifest.yaml"
    _edit_pack_yaml(
        manifest_path,
        lambda manifest: manifest.update({"held_out": "held_out.yaml"}),
    )
    fixtures_path = pack_root / "fixtures.json"
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    fixtures["books"][0]["book_id"] = 1
    fixtures["books"][1]["book_id"] = "1"
    fixtures_path.write_text(json.dumps(fixtures), encoding="utf-8")
    (pack_root / "held_out.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "fixtures": {"books": [1]},
                "templates": [],
                "policy": {"fixtures_in_backend_state": True, "seed": 0},
            }
        ),
        encoding="utf-8",
    )
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "ambiguous-held-out.yaml",
            oracle_pack={"manifest_path": str(manifest_path)},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )

    with pytest.raises(ValueError, match="must be unique after scalar normalization"):
        load_pack(config)


def test_model_io_cache_key_is_stable_and_entries_are_immutable(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
        ImmutableModelIOCache,
        request_hash,
    )

    arguments = {
        "model_canonical": "Source::Model@Revision",
        "prompt_hash": "sha256:prompt",
        "model_input": {"text": "Xin chào", "protected": ["ACC-001"]},
        "inference_parameters": {"temperature": 0.0},
        "output_schema": {"type": "object"},
        "seed": 42,
    }
    key = request_hash(**arguments)
    assert key == request_hash(**arguments)
    assert key != request_hash(**{**arguments, "output_schema": {"type": "array"}})

    cache_path = tmp_path / "model_io.jsonl"
    cache = ImmutableModelIOCache(cache_path)
    cache.put(
        key,
        {"style_hints": ["concise"]},
        model_canonical=arguments["model_canonical"],
        input_hash="sha256:input",
    )
    assert cache.get(key) == {"style_hints": ["concise"]}
    with pytest.raises(ValueError, match="immutable"):
        cache.put(
            key,
            {"style_hints": ["different"]},
            model_canonical=arguments["model_canonical"],
            input_hash="sha256:input",
        )
    entry = json.loads(cache_path.read_text(encoding="utf-8"))
    entry["response"] = {"style_hints": ["tampered"]}
    cache_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid response_hash"):
        ImmutableModelIOCache(cache_path)


def test_reference_profile_normalizes_samples_and_reuses_model_cache(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )

    samples = tmp_path / "profile-samples.jsonl"
    samples.write_text(
        '{"sample_id":"ref-1","language":" VI ","messages":'
        '[{"role":"user","content":"Bạn kiểm tra giúp mình nhé."}],'
        '"tags":["polite","concise"]}\n',
        encoding="utf-8",
    )
    content_hash = f"sha256:{hashlib.sha256(samples.read_bytes()).hexdigest()}"
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "profile-enabled.yaml",
            oracle_runtime={"allowed_roots": [str(tmp_path), str(BYOB_DIR / "data")]},
            lineage={
                "roles": {
                    "profile": {
                        "enabled": True,
                        "model_config": {
                            "alias": "profile",
                            "provider": "nvidia",
                            "model": "profile-model",
                            "canonical_id": "source::profile-model@revision",
                            "inference_parameters": {"temperature": 0.0},
                        },
                    }
                }
            },
            reference_benchmark={
                "name": "vi-style",
                "samples_path": str(samples),
                "content_hash": content_hash,
            },
        )
    )
    calls: list[dict[str, Any]] = []

    def fake_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        calls.append(kwargs)
        request_id = kwargs["requests"][0]["request_id"]
        model_input = kwargs["requests"][0]["model_input"]
        assert "expected_tool_calls" not in model_input
        return {
            request_id: {
                "style_hints": ["Use conversational Vietnamese", "Be concise"],
                "avoid": ["Internal tool names"],
            }
        }

    first = run_reference_profile(config, model_runner=fake_runner)
    profile_path = Path(config.output_dir) / config.expt_name / "stage_cache" / "reference_profile.json"
    first_bytes = profile_path.read_bytes()
    second = run_reference_profile(
        config,
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit called the model")),
    )

    assert first == second
    assert profile_path.read_bytes() == first_bytes
    assert len(calls) == 1
    assert first["status"] == "completed"
    assert first["languages"] == ["vi"]
    assert first["profile_model_canonical"] == "source::profile-model@revision"
    rows = pq.read_table(profile_path.parent / "reference_samples.parquet").to_pylist()
    assert rows[0]["sample_id"] == "ref-1"
    assert rows[0]["language"] == "vi"
    assert rows[0]["tags"] == ["polite", "concise"]


def test_reference_profile_keeps_an_unusable_response_out_of_the_cache(
    tmp_path: Path,
) -> None:
    """A malformed response must stay retryable: the cache it would land in is immutable."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )

    samples = tmp_path / "profile-samples.jsonl"
    samples.write_text(
        '{"sample_id":"ref-1","language":"vi","messages":[{"role":"user","content":"Bạn kiểm tra giúp mình nhé."}]}\n',
        encoding="utf-8",
    )
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "profile-unusable.yaml",
            oracle_runtime={"allowed_roots": [str(tmp_path), str(BYOB_DIR / "data")]},
            lineage={
                "roles": {
                    "profile": {
                        "enabled": True,
                        "model_config": {
                            "alias": "profile",
                            "provider": "nvidia",
                            "model": "profile-model",
                            "canonical_id": "source::profile-model@revision",
                        },
                    }
                }
            },
            reference_benchmark={
                "name": "vi-style",
                "samples_path": str(samples),
                "content_hash": f"sha256:{hashlib.sha256(samples.read_bytes()).hexdigest()}",
            },
        )
    )
    assert config.lineage.roles["profile"].model_config["inference_parameters"] == {}

    def broken_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        return {kwargs["requests"][0]["request_id"]: {"style_hints": [" "]}}

    def working_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        return {
            kwargs["requests"][0]["request_id"]: {
                "style_hints": ["Use conversational Vietnamese"],
                "avoid": [],
            }
        }

    with pytest.raises(RuntimeError, match="style_hints"):
        run_reference_profile(config, model_runner=broken_runner)

    io_cache_path = Path(config.output_dir) / config.expt_name / "stage_cache" / "reference_profile_io_cache.jsonl"
    assert not io_cache_path.exists() or not io_cache_path.read_text(encoding="utf-8").strip()

    profile = run_reference_profile(config, model_runner=working_runner)

    assert profile["status"] == "completed"
    assert profile["style_hints"] == ["Use conversational Vietnamese"]


def test_profile_language_is_not_gated_when_no_template_consumes_it(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import LineageRole
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
        run_expand,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        run_paraphrase,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        run_render,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "unused-profile-language.yaml",
            lineage={
                "roles": {
                    "paraphrase": {
                        "enabled": True,
                        "model_config": {
                            "alias": "paraphrase",
                            "provider": "nvidia",
                            "model": "paraphrase-model",
                            "canonical_id": "source::paraphrase-model@revision",
                        },
                    }
                }
            },
            surface_generation={
                "model_paraphrase_enabled": True,
                "paraphrases_per_template": 1,
            },
        )
    )
    roles = dict(config.lineage.roles)
    roles["profile"] = LineageRole(
        enabled=True,
        model_config={
            "alias": "profile",
            "provider": "nvidia",
            "model": "profile-model",
            "canonical_id": "source::profile-model@revision",
            "inference_parameters": {},
        },
    )
    config.lineage = replace(
        config.lineage,
        profile_influenced_surface=True,
        roles=roles,
    )
    pack = load_pack(config)
    assert not any((template.get("paraphrase") or {}).get("allowed") is True for template in pack.templates)
    templates = {str(template["template_id"]): template for template in pack.templates}
    tasks = run_expand(config, pack)
    plans = run_state_machine(config, templates, tasks)
    surfaces, _ = run_render(config, pack, templates, tasks, plans)

    result = run_paraphrase(
        config,
        pack,
        templates,
        tasks,
        plans,
        surfaces,
        {
            "status": "completed",
            "languages": ["fr"],
            "style_hints": ["Use French phrasing"],
            "avoid": [],
            "output_hash": "sha256:profile",
        },
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("no eligible template should call the model")
        ),
    )

    assert result[3]["requested_candidates"] == 0
    assert result[3]["profile_consumed"] is False


def test_controlled_paraphrase_fans_out_only_guarded_variants_and_reuses_cache(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
        run_expand,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        run_paraphrase,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        run_render,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "paraphrase-enabled.yaml",
            lineage={
                "roles": {
                    "paraphrase": {
                        "enabled": True,
                        "model_config": {
                            "alias": "paraphrase",
                            "provider": "nvidia",
                            "model": "paraphrase-model",
                            "canonical_id": "source::paraphrase-model@revision",
                            "inference_parameters": {"temperature": 0.0},
                        },
                    }
                }
            },
            surface_generation={
                "model_paraphrase_enabled": True,
                "paraphrases_per_template": 2,
            },
        )
    )
    pack = load_pack(config)
    pack.templates[0]["paraphrase"] = {
        "allowed": True,
        "must_preserve": ["book_id"],
    }
    templates = {str(template["template_id"]): template for template in pack.templates}
    canonical_tasks = run_expand(config, pack)
    plans = run_state_machine(config, templates, canonical_tasks)
    surfaces, _ = run_render(
        config,
        pack,
        templates,
        canonical_tasks,
        plans,
    )
    profile = run_reference_profile(config)
    model_calls = 0

    def fake_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        nonlocal model_calls
        del args
        model_calls += 1
        responses = {}
        for request in kwargs["requests"]:
            contract = json.loads(request["model_input"])
            canonical = contract["canonical_user_turns"]
            protected = contract["must_preserve"][0]
            assert contract["style_avoid"] == []
            responses[request["request_id"]] = {
                "variants": [
                    {"user_turns": [f"{text} Please help." for text in canonical]},
                    {"user_turns": [text.replace(protected, "that book") for text in canonical]},
                ]
            }
        return responses

    cache = Path(config.output_dir) / config.expt_name / "stage_cache"

    def invalid_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        return {request["request_id"]: {"variants": []} for request in kwargs["requests"]}

    invalid_run = run_paraphrase(
        config,
        pack,
        templates,
        canonical_tasks,
        plans,
        surfaces,
        profile,
        model_runner=invalid_runner,
    )
    assert invalid_run[3]["accepted_candidates"] == 0
    io_cache_path = cache / "paraphrase_io_cache.jsonl"
    assert not io_cache_path.exists() or not io_cache_path.read_text(encoding="utf-8").strip()

    tasks, variant_plans, variant_surfaces, report = run_paraphrase(
        config,
        pack,
        templates,
        canonical_tasks,
        plans,
        surfaces,
        profile,
        model_runner=fake_runner,
    )
    accepted = [task for task in tasks if task["variant_index"] == 1]
    assert accepted
    assert all(task["task_id"] != task["base_task_id"] for task in accepted)
    assert all(
        task["slots"] == next(base["slots"] for base in canonical_tasks if base["task_id"] == task["base_task_id"])
        for task in accepted
    )
    assert report["requested_candidates"] == 2 * len(accepted)
    assert report["accepted_candidates"] == len(accepted)
    assert report["rejected_candidates"] == len(accepted)
    assert report["by_reason"]["must_preserve"] == len(accepted)
    assert all(
        variant_surfaces[str(task["task_id"])]["paraphrase_model_canonical"] == "source::paraphrase-model@revision"
        for task in accepted
    )
    assert all(
        variant_plans[str(task["task_id"])]["steps"] == plans[str(task["base_task_id"])]["steps"] for task in accepted
    )

    task_ids = {row["task_id"] for row in pq.read_table(cache / "task_instances.parquet").to_pylist()}
    plan_ids = {row["task_id"] for row in pq.read_table(cache / "conversation_plans.parquet").to_pylist()}
    render_ids = {row["task_id"] for row in pq.read_table(cache / "rendered_conversations.parquet").to_pylist()}
    assert task_ids == plan_ids == render_ids

    rerun = run_paraphrase(
        config,
        pack,
        templates,
        canonical_tasks,
        plans,
        surfaces,
        profile,
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit called the model")),
    )
    assert [task["task_id"] for task in rerun[0]] == [task["task_id"] for task in tasks]
    assert model_calls == 1


def test_style_plan_is_deterministic_distinct_per_variant_and_spread_over_axes() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        SURFACE_STYLE_AXES,
        style_plan,
    )

    task = {"seed": 7}
    plan = style_plan(task, 3)

    assert plan == style_plan(dict(task), 3)
    assert len(set(plan)) == 3
    assert set(plan) <= set(SURFACE_STYLE_AXES)
    assert style_plan({"seed": 0}, 0) == []
    # More variants than axes may only cycle; it must never raise.
    assert len(style_plan(task, len(SURFACE_STYLE_AXES) + 2)) == len(SURFACE_STYLE_AXES) + 2
    # A pack may declare its own register catalog, so the plan must honor it.
    assert style_plan({"seed": 1}, 2, ("first style", "second style")) == [
        "second style",
        "first style",
    ]
    with pytest.raises(ValueError, match="must not be empty"):
        style_plan(task, 1, ())

    # Bindings of one template differ only by seed, so the assignment has to spread
    # them across axes; otherwise every binding would be asked for the same rewrite.
    first_styles = {style_plan({"seed": seed}, 1)[0] for seed in range(200)}
    assert first_styles == set(SURFACE_STYLE_AXES)


def test_declared_surface_style_axes_are_validated_and_bound_variant_count(
    tmp_path: Path,
) -> None:
    from itertools import count

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        SURFACE_STYLE_AXES,
        resolve_style_axes,
    )

    written = count()

    def config_with(**surface: Any) -> BfclConfig:
        return BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                f"axes-{next(written)}.yaml",
                lineage={
                    "roles": {
                        "paraphrase": {
                            "enabled": True,
                            "model_config": {
                                "alias": "paraphrase",
                                "provider": "nvidia",
                                "model": "paraphrase-model",
                                "canonical_id": "source::paraphrase-model@revision",
                                "inference_parameters": {"temperature": 0.0},
                            },
                        }
                    }
                },
                surface_generation={"model_paraphrase_enabled": True, **surface},
            )
        )

    assert resolve_style_axes(config_with(paraphrases_per_template=1)) == SURFACE_STYLE_AXES
    declared = config_with(
        paraphrases_per_template=2,
        surface_style_axes=["  short and blunt  ", "long and formal"],
    )
    assert resolve_style_axes(declared) == ("short and blunt", "long and formal")

    # Asking for more variants than axes can only repeat an axis inside one binding,
    # which spends the model twice on the same rewrite.
    with pytest.raises(ValueError, match="cannot exceed the 20 declared surface style axes"):
        config_with(paraphrases_per_template=len(SURFACE_STYLE_AXES) + 1)
    with pytest.raises(ValueError, match="cannot exceed the 2 declared surface style axes"):
        config_with(
            paraphrases_per_template=3,
            surface_style_axes=["one", "two"],
        )
    with pytest.raises(ValueError, match="must be unique"):
        config_with(paraphrases_per_template=1, surface_style_axes=["same", "same"])
    with pytest.raises(ValueError, match="must be a non-empty list"):
        config_with(paraphrases_per_template=1, surface_style_axes=[])
    with pytest.raises(ValueError, match=r"surface_style_axes\[1\] must be a non-empty string"):
        config_with(paraphrases_per_template=1, surface_style_axes=["fine", "  "])


def test_paraphrase_asks_each_binding_for_its_own_surface_style(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
        run_expand,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        SURFACE_STYLE_AXES,
        run_paraphrase,
        style_plan,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        run_render,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "paraphrase-styles.yaml",
            lineage={
                "roles": {
                    "paraphrase": {
                        "enabled": True,
                        "model_config": {
                            "alias": "paraphrase",
                            "provider": "nvidia",
                            "model": "paraphrase-model",
                            "canonical_id": "source::paraphrase-model@revision",
                            "inference_parameters": {"temperature": 0.0},
                        },
                    }
                }
            },
            surface_generation={
                "model_paraphrase_enabled": True,
                "paraphrases_per_template": 2,
            },
        )
    )
    pack = load_pack(config)
    for template in pack.templates:
        template["paraphrase"] = {"allowed": True}
    templates = {str(template["template_id"]): template for template in pack.templates}
    canonical_tasks = run_expand(config, pack)
    plans = run_state_machine(config, templates, canonical_tasks)
    surfaces, _ = run_render(config, pack, templates, canonical_tasks, plans)
    profile = run_reference_profile(config)
    contracts: list[dict[str, Any]] = []

    def capturing_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        responses = {}
        for request in kwargs["requests"]:
            contract = json.loads(request["model_input"])
            contracts.append(contract)
            responses[request["request_id"]] = {
                "variants": [
                    {"user_turns": [f"{style}: {text}" for text in contract["canonical_user_turns"]]}
                    for style in contract["surface_styles"]
                ]
            }
        return responses

    run_paraphrase(
        config,
        pack,
        templates,
        canonical_tasks,
        plans,
        surfaces,
        profile,
        model_runner=capturing_runner,
    )

    assert contracts
    by_task = {str(task["task_id"]): task for task in canonical_tasks}
    requested_styles = [tuple(contract["surface_styles"]) for contract in contracts]
    assert all(len(styles) == 2 and len(set(styles)) == 2 for styles in requested_styles)
    assert all(set(styles) <= set(SURFACE_STYLE_AXES) for styles in requested_styles)
    assert sorted(requested_styles) == sorted(tuple(style_plan(task, 2)) for task in by_task.values())


def test_paraphrase_rejects_a_variant_that_repeats_another_variant(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
        run_expand,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        run_paraphrase,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        run_render,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "paraphrase-duplicates.yaml",
            lineage={
                "roles": {
                    "paraphrase": {
                        "enabled": True,
                        "model_config": {
                            "alias": "paraphrase",
                            "provider": "nvidia",
                            "model": "paraphrase-model",
                            "canonical_id": "source::paraphrase-model@revision",
                            "inference_parameters": {"temperature": 0.0},
                        },
                    }
                }
            },
            surface_generation={
                "model_paraphrase_enabled": True,
                "paraphrases_per_template": 2,
            },
        )
    )
    pack = load_pack(config)
    for template in pack.templates:
        template["paraphrase"] = {"allowed": True}
    templates = {str(template["template_id"]): template for template in pack.templates}
    canonical_tasks = run_expand(config, pack)
    plans = run_state_machine(config, templates, canonical_tasks)
    surfaces, _ = run_render(config, pack, templates, canonical_tasks, plans)
    profile = run_reference_profile(config)

    def duplicating_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        responses = {}
        for request in kwargs["requests"]:
            contract = json.loads(request["model_input"])
            repeated = {"user_turns": [f"cùng một cách diễn đạt: {text}" for text in contract["canonical_user_turns"]]}
            responses[request["request_id"]] = {"variants": [repeated, dict(repeated)]}
        return responses

    output_tasks, _, _, report = run_paraphrase(
        config,
        pack,
        templates,
        canonical_tasks,
        plans,
        surfaces,
        profile,
        model_runner=duplicating_runner,
    )

    accepted = [task for task in output_tasks if int(task.get("variant_index", 0)) > 0]
    assert accepted
    # Both variants pass the value guards, so only the repeat itself may be dropped.
    assert len(accepted) == len(canonical_tasks)
    assert report["rejected_candidates"] == len(canonical_tasks)
    assert report["by_reason"]["semantic_shape"] == len(canonical_tasks)
    assert any(
        detail["reason"] == "duplicate_variant_surface"
        for event in report["events"]
        for detail in event.get("detail", [])
    )


def test_one_failing_paraphrase_request_does_not_discard_its_batch(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
        run_expand,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        run_paraphrase,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
        run_render,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "paraphrase-batch-failure.yaml",
            ndd_batch_size=32,
            lineage={
                "roles": {
                    "paraphrase": {
                        "enabled": True,
                        "model_config": {
                            "alias": "paraphrase",
                            "provider": "nvidia",
                            "model": "paraphrase-model",
                            "canonical_id": "source::paraphrase-model@revision",
                            "inference_parameters": {"temperature": 0.0},
                        },
                    }
                }
            },
            surface_generation={
                "model_paraphrase_enabled": True,
                "paraphrases_per_template": 1,
            },
        )
    )
    pack = load_pack(config)
    for template in pack.templates:
        template["paraphrase"] = {"allowed": True}
    templates = {str(template["template_id"]): template for template in pack.templates}
    canonical_tasks = run_expand(config, pack)
    assert len(canonical_tasks) > 1
    plans = run_state_machine(config, templates, canonical_tasks)
    surfaces, _ = run_render(config, pack, templates, canonical_tasks, plans)
    profile = run_reference_profile(config)
    poisoned: list[str] = []
    call_sizes: list[int] = []

    def failing_runner(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        del args
        requests = kwargs["requests"]
        call_sizes.append(len(requests))
        # One request is permanently broken; the rest of its batch is healthy.
        broken = sorted(request["request_id"] for request in requests)[0]
        if len(requests) > 1 or requests[0]["request_id"] == broken:
            if not poisoned:
                poisoned.append(broken)
            if any(request["request_id"] == poisoned[0] for request in requests):
                raise RuntimeError("endpoint refused this request")
        responses = {}
        for request in requests:
            contract = json.loads(request["model_input"])
            responses[request["request_id"]] = {
                "variants": [{"user_turns": [f"diễn đạt khác: {text}" for text in contract["canonical_user_turns"]]}]
            }
        return responses

    output_tasks, _, _, report = run_paraphrase(
        config,
        pack,
        templates,
        canonical_tasks,
        plans,
        surfaces,
        profile,
        model_runner=failing_runner,
    )

    accepted = [task for task in output_tasks if int(task.get("variant_index", 0)) > 0]
    # Every binding but the broken one keeps its variant, and the retry is per request.
    assert len(accepted) == len(canonical_tasks) - 1
    assert max(call_sizes) > 1
    assert call_sizes.count(1) == len(canonical_tasks)
    assert report["by_reason"] == {"model_error": 1}
    assert [event["reason"] for event in report["events"]] == ["model_error"]
    assert report["events"][0]["detail"] == "RuntimeError"


def test_pipeline_checkpoints_post_replay_paraphrase_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl import model_runner
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def allow_first_template(templates: list[dict[str, Any]]) -> None:
        templates[0]["paraphrase"] = {
            "allowed": True,
            "must_preserve": ["book_id"],
        }

    _edit_pack_yaml(pack / "task_templates.yaml", allow_first_template)
    config_path = _write_tiny_config(
        tmp_path,
        "paraphrase-pipeline.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
        lineage={
            "roles": {
                "paraphrase": {
                    "enabled": True,
                    "model_config": {
                        "alias": "paraphrase",
                        "provider": "nvidia",
                        "model": "paraphrase-model",
                        "canonical_id": "source::paraphrase-model@revision",
                        "inference_parameters": {"temperature": 0.0},
                    },
                }
            }
        },
        surface_generation={
            "model_paraphrase_enabled": True,
            "paraphrases_per_template": 1,
        },
    )

    def fake_model(
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        del args
        return {
            request["request_id"]: {
                "variants": [
                    {
                        "user_turns": [
                            f"{turn} Please help. Algorithms."
                            for turn in json.loads(request["model_input"])["canonical_user_turns"]
                        ]
                    }
                ]
            }
            for request in kwargs["requests"]
        }

    monkeypatch.setattr(model_runner, "run_structured_model", fake_model)
    benchmark_path = generate_bfcl(config_path)
    rows = pq.read_table(benchmark_path).to_pylist()
    variants = [row for row in rows if row["variant_index"] == 1]

    assert variants == []
    manifest = json.loads((benchmark_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_mode"] == "smoke_no_publication"
    assert manifest["stage_counts"]["paraphrase_accepted"] == len(variants)
    assert manifest["paraphrase_rejections"]["by_reason"]["expected_result_leakage"] >= 1
    assert "paraphrase_io_cache" in manifest["artifacts"]
    assert "paraphrase_rejections" in manifest["artifacts"]
    replay_checkpoint = benchmark_path.parent / "stage_cache" / "checkpoints" / "executable_replay" / "manifest.json"
    checkpoint_document = json.loads(replay_checkpoint.read_text(encoding="utf-8"))
    assert "stage_cache/rendered_conversations.parquet" in checkpoint_document["produced_artifacts"]
    resumed = generate_bfcl(config_path, skip_until="final_output")
    assert [row["task_id"] for row in pq.read_table(resumed).to_pylist()] == [row["task_id"] for row in rows]


def test_post_replay_guard_rejects_expected_result_leakage(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        apply_expected_result_guards,
    )

    config = BfclConfig.from_yaml(_write_tiny_config(tmp_path, "result-leakage.yaml"))
    base_id = "base"
    variant_id = "variant"
    surfaces = {
        base_id: {
            "task_id": base_id,
            "base_task_id": base_id,
            "template_id": "tpl",
            "variant_index": 0,
            "source": "template",
            "language": "vi",
            "system_prompt_id": "prompt",
            "steps": [{"kind": "user", "content": "Kiểm tra tài khoản ACC-1."}],
            "guard_violations": [],
        },
        variant_id: {
            "task_id": variant_id,
            "base_task_id": base_id,
            "template_id": "tpl",
            "variant_index": 1,
            "source": "model",
            "language": "vi",
            "system_prompt_id": "prompt",
            "steps": [
                {
                    "kind": "user",
                    "content": "Tài khoản ACC-1 còn 500.000 đồng phải không?",
                }
            ],
            "guard_violations": [],
        },
    }
    tasks = [
        {"task_id": base_id, "template_id": "tpl", "variant_index": 0},
        {"task_id": variant_id, "template_id": "tpl", "variant_index": 1},
    ]
    report = {
        "accepted_candidates": 1,
        "rejected_candidates": 0,
        "by_reason": {},
        "by_template": {"tpl": {"requested": 1, "accepted": 1, "rejected": 0}},
        "events": [],
    }

    updated = apply_expected_result_guards(
        config,
        tasks,
        surfaces,
        {variant_id: {"passed": True, "results": [{"balance": 500000}]}},
        report,
    )

    assert updated["accepted_candidates"] == 0
    assert updated["rejected_candidates"] == 1
    assert surfaces[variant_id]["guard_violations"] == [{"guard": "expected_result_leakage", "value": "500000"}]


def test_config_rejects_output_nested_inside_pack_root(tmp_path: Path) -> None:
    pack = _copy_tiny_pack(tmp_path)
    config = _write_tiny_config(
        tmp_path,
        "nested-output.yaml",
        expt_name="generated",
        output_dir=str(pack),
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    with pytest.raises(ValueError, match="outside the oracle pack root"):
        BfclConfig.from_yaml(config)


@pytest.mark.parametrize(
    "expt_name",
    ["../escape", "nested/run", ".", "..", " padded "],
)
def test_config_rejects_an_expt_name_that_is_not_one_directory(tmp_path: Path, expt_name: str) -> None:
    """The run directory names the run, so it may not move the output somewhere else."""
    config = _write_tiny_config(tmp_path, "odd-expt-name.yaml", expt_name=expt_name)

    with pytest.raises(ValueError, match="single directory name"):
        BfclConfig.from_yaml(config)


def test_manifest_reports_replay_apart_from_surface_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guard-rejected row still replayed, and the counts must let a reader see that."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import render

    config = _write_tiny_config(tmp_path, "guard-split.yaml")
    # Validation certifies the pack's own surfaces first; the guard below stands in for
    # an instance whose paraphrase happens to break a guard during generation.
    prepare_bfcl(config)

    real_guards = render.check_surface_guards
    rejected: list[str] = []

    def reject_the_first_surface(template, task, user_texts, tool_names, **kwargs):  # type: ignore[no-untyped-def]
        violations = real_guards(template, task, user_texts, tool_names, **kwargs)
        if not violations and not rejected:
            rejected.append(str(task["task_id"]))
            return [{"guard": "must_preserve", "slot": "stand_in"}]
        return violations

    monkeypatch.setattr(render, "check_surface_guards", reject_the_first_surface)
    benchmark_path = generate_bfcl(config)

    manifest = json.loads((benchmark_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["stage_counts"]
    assert len(rejected) == 1
    assert counts["replay_passed"] == counts["expanded"]
    assert counts["surface_passed"] == counts["expanded"] - 1
    assert counts["published"] == counts["expanded"] - 1
    assert sum(manifest["surface_guard_rejections"]["by_template"].values()) == 1


def test_tiny_prepare_is_gold_eligible(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
    )

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    temp_config = tmp_path / "tiny.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    report_path = prepare_bfcl(temp_config)
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pack_id"] == "tiny_library"
    assert report["gold_eligible"] is True
    assert report["tier"] == "gold"
    assert derive_pack_tier(report) == (report["gold_eligible"], report["tier"])
    cache = tmp_path / "output" / "bfcl_tiny_library_validation" / "stage_cache"
    assert (cache / "tools_normalized.json").exists()
    assert (cache / "tools_normalized_internal.json").exists()


def test_prepare_rejects_a_template_whose_bound_call_breaks_schema(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    _edit_pack_yaml(
        pack / "task_templates.yaml",
        lambda templates: templates[0]["assistant_milestones"][0].update({"args": {"unexpected": "value"}}),
    )
    config = _write_tiny_config(
        tmp_path,
        "bad-template-arguments.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "representative_trace_schema_mismatch"
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_budget_that_cannot_keep_every_template(
    tmp_path: Path,
) -> None:
    """A category budget below its template count fails generation, so gold must see it."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    config = _write_tiny_config(
        tmp_path,
        "starved-budget.yaml",
        task_generation={"tasks_per_category": 1},
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "run_contract_failed"
    assert "tasks_per_category" in contract["failures"][0]["detail"]
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_pack_that_cannot_render_its_own_templates(
    tmp_path: Path,
) -> None:
    """Render is a hard contract, so a missing text block must not survive to generation."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    _edit_pack_yaml(
        pack / "manifest.yaml",
        lambda manifest: manifest["assistant_turn_templates"].pop("final_answer"),
    )
    config = _write_tiny_config(
        tmp_path,
        "unrenderable.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "representative_generation_failed"
    assert "final_answer" in contract["failures"][0]["detail"]
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_template_whose_surface_always_breaks_a_guard(
    tmp_path: Path,
) -> None:
    """A template that can publish no row is a defect, not an instance-level rejection."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def drop_slot_values(templates: list[dict[str, Any]]) -> None:
        for language in templates[0]["user_turn_templates"]:
            templates[0]["user_turn_templates"][language] = "Please help me with something."

    _edit_pack_yaml(pack / "task_templates.yaml", drop_slot_values)
    config = _write_tiny_config(
        tmp_path,
        "guard-always-fails.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert contract["failures"][0]["reason"] == "representative_surface_guard_violation"
    assert report["gold_eligible"] is False


def test_prepare_runs_a_representative_templates_assertions(tmp_path: Path) -> None:
    """An importable assertion that always fails must not receive a gold report."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    with (pack / "assertions.py").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\ndef always_fail(*, state, trace, task, ctx):\n"
            "    raise AssertionError('deliberate failure')\n"
            "ASSERTIONS = {name: always_fail for name in ASSERTIONS}\n"
        )
    config = _write_tiny_config(
        tmp_path,
        "failing-assertions.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    contract = next(check for check in report["checks"] if check["id"] == 7)
    assert contract["status"] == "fail"
    assert {failure["reason"] for failure in contract["failures"]} == {"representative_replay_failed"}
    assert report["gold_eligible"] is False


def test_prepare_rejects_a_non_function_tool_envelope(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tools[0]["type"] = "not-a-function"
    tools_path.write_text(json.dumps(tools), encoding="utf-8")
    config = _write_tiny_config(
        tmp_path,
        "bad-tool-envelope.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    alignment = next(check for check in report["checks"] if check["id"] == 3)
    assert alignment["status"] == "fail"
    assert alignment["failures"][0]["reason"] == "tool_type_not_function"


def test_success_coverage_requires_schema_valid_arguments(tmp_path: Path) -> None:
    """A backend accepting hidden arguments must not let those calls prove success."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def add_hidden_argument(cases: list[dict[str, Any]]) -> None:
        success = next(case for case in cases if case["id"] == "success_get_book_status")
        success["arguments"]["backend_only"] = True

    _edit_pack_yaml(pack / "validation_cases.yaml", add_hidden_argument)
    config = _write_tiny_config(
        tmp_path,
        "invalid-success-probe.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    coverage = next(check for check in report["checks"] if check["id"] == 5)
    reasons = {failure["reason"] for failure in coverage["failures"]}
    assert "successful_validation_case_schema_mismatch" in reasons
    assert "incomplete_validation_coverage" in reasons


def test_validation_rejects_pack_inputs_that_change_during_import(
    tmp_path: Path,
) -> None:
    """The fingerprint must describe one immutable set of inputs throughout validation."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    backend = pack / "backend.py"
    source = backend.read_text(encoding="utf-8")
    backend.write_text(
        source.replace(
            "from __future__ import annotations",
            "from __future__ import annotations\n"
            "from pathlib import Path\n"
            "with Path(__file__).with_name('imports.log').open('a') as handle:\n"
            "    handle.write('x')",
            1,
        ),
        encoding="utf-8",
    )
    config = _write_tiny_config(
        tmp_path,
        "mutating-pack.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    with pytest.raises(RuntimeError, match="changed while it was being validated"):
        prepare_bfcl(config)


def test_failed_rerun_does_not_leave_a_completed_manifest(tmp_path: Path) -> None:
    """A failed invocation must not leave a previous run looking current."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    config = _write_tiny_config(
        tmp_path,
        "rerun.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    benchmark = generate_bfcl(config)
    output = benchmark.parent
    assert (output / "run_manifest.json").exists()

    with (pack / "assertions.py").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\ndef always_fail(*, state, trace, task, ctx):\n"
            "    raise AssertionError('deliberate failure')\n"
            "ASSERTIONS = {name: always_fail for name in ASSERTIONS}\n"
        )
    with pytest.raises(RuntimeError, match="non-gold pack"):
        generate_bfcl(config)

    assert not (output / "run_manifest.json").exists()
    assert not (output / "benchmark.parquet").exists()
    assert not (output / "benchmark_raw.parquet").exists()


def test_prepare_rejects_bad_plans_and_missing_fixture_primary_keys(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    plan_root = tmp_path / "bad-plan"
    plan_root.mkdir()
    plan_pack = _copy_tiny_pack(plan_root)
    _edit_pack_yaml(
        plan_pack / "task_templates.yaml",
        lambda templates: templates[0].update(
            {
                "assistant_milestones": [
                    {"type": "ask_confirm"},
                    {"type": "tool_call", "tool": "get_book_status"},
                    {"type": "final_answer"},
                ]
            }
        ),
    )
    plan_config = _write_tiny_config(
        tmp_path,
        "bad-plan.yaml",
        oracle_pack={"manifest_path": str(plan_pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    plan_report = json.loads(prepare_bfcl(plan_config).read_text(encoding="utf-8"))
    assert any(
        failure.get("reason") == "invalid_conversation_plan" for failure in plan_report["checks"][0]["failures"]
    )

    key_root = tmp_path / "missing-key"
    key_root.mkdir()
    key_pack = _copy_tiny_pack(key_root)
    fixtures_path = key_pack / "fixtures.json"
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    fixtures["books"][1].pop("book_id")
    fixtures_path.write_text(json.dumps(fixtures), encoding="utf-8")
    _edit_pack_yaml(
        key_pack / "manifest.yaml",
        lambda manifest: manifest.setdefault("primary_keys", {}).update({"books": "book_id"}),
    )
    _edit_pack_yaml(
        key_pack / "task_templates.yaml",
        lambda templates: templates[0]["slots"]["book_id"].update({"source": "fixture:books.title"}),
    )
    key_config = _write_tiny_config(
        tmp_path,
        "missing-key.yaml",
        oracle_pack={"manifest_path": str(key_pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    key_report = json.loads(prepare_bfcl(key_config).read_text(encoding="utf-8"))
    assert any(
        failure.get("reason") == "fixture_row_missing_primary_key" for failure in key_report["checks"][1]["failures"]
    )


def test_thread_worker_cannot_claim_gold(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    config_data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    config_data["output_dir"] = str(tmp_path / "output")
    config_data["oracle_runtime"]["worker"] = "thread"
    temp_config = tmp_path / "thread.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))

    assert report["gold_eligible"] is False
    assert report["tier"] == "silver"
    isolation = next(check for check in report["extra_checks"] if check["id"] == "I1")
    assert isolation["status"] == "fail"
    timeout = next(check for check in report["extra_checks"] if check["id"] == "T1")
    assert timeout["status"] == "skipped"


@pytest.mark.parametrize("worker", ["process", "thread"])
def test_validation_checks_expected_result_fields(tmp_path: Path, worker: str) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def break_expected_status(cases: list[dict[str, Any]]) -> None:
        checkout = next(case for case in cases if case["id"] == "success_checkout_book")
        checkout["expect"]["status"] = "impossible_status"

    _edit_pack_yaml(pack / "validation_cases.yaml", break_expected_status)
    temp_config = _write_tiny_config(
        tmp_path,
        "bad-status.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)], "worker": worker},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(check for check in report["checks"] if check["id"] == 5)

    assert check["status"] == "fail"
    assert any(failure["reason"] == "result_field_mismatch" for failure in check["failures"])


@pytest.mark.parametrize("worker", ["process", "thread"])
def test_validation_cases_can_share_state_without_reset(tmp_path: Path, worker: str) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def chain_after_checkout(cases: list[dict[str, Any]]) -> None:
        index = next(i for i, case in enumerate(cases) if case["id"] == "success_checkout_book")
        cases.insert(
            index + 1,
            {
                "id": "observe_checkout_without_reset",
                "tool": "get_book_status",
                "arguments": {"book_id": "BK-100"},
                "expect": {"result_class": "success", "status": "on_loan"},
                "reset_before": False,
            },
        )

    _edit_pack_yaml(pack / "validation_cases.yaml", chain_after_checkout)
    temp_config = _write_tiny_config(
        tmp_path,
        "shared-state.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)], "worker": worker},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(check for check in report["checks"] if check["id"] == 5)

    assert check["status"] == "pass"
    determinism = next(check for check in report["extra_checks"] if check["id"] == "D1")
    assert determinism["status"] == "pass"


@pytest.mark.parametrize(
    "clock",
    ["not-a-timestamp", "2026-03-02T09:00:00"],
)
def test_rejects_unusable_clock(tmp_path: Path, clock: str) -> None:
    temp_config = _write_tiny_config(tmp_path, "clock.yaml", oracle_runtime={"clock": clock})

    with pytest.raises(ValueError, match="oracle_runtime.clock"):
        BfclConfig.from_yaml(temp_config)


def test_rejects_unquoted_yaml_timestamp(tmp_path: Path) -> None:
    text = (BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8")
    text = text.replace('"2026-03-02T09:00:00+07:00"', "2026-03-02T09:00:00+07:00")
    path = tmp_path / "unquoted-clock.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="quoted ISO-8601 string"):
        BfclConfig.from_yaml(path)


def test_rejects_a_schema_version_this_build_cannot_write(tmp_path: Path) -> None:
    """The manifest publishes this value, so it must name a shape the rows really have."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
        DEFAULT_BENCHMARK_SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="schema_version"):
        BfclConfig.from_yaml(_write_tiny_config(tmp_path, "future-schema.yaml", schema_version="9.9"))

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "known-schema.yaml",
            schema_version=DEFAULT_BENCHMARK_SCHEMA_VERSION,
        )
    )
    assert config.schema_version == DEFAULT_BENCHMARK_SCHEMA_VERSION


def test_rejects_non_positive_timeout(tmp_path: Path) -> None:
    temp_config = _write_tiny_config(tmp_path, "timeout.yaml", oracle_runtime={"tool_timeout_s": 0})

    with pytest.raises(ValueError, match="tool_timeout_s"):
        BfclConfig.from_yaml(temp_config)


def test_unevaluable_slot_filter_is_reported_not_raised(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def break_filter(templates: list[dict[str, Any]]) -> None:
        for template in templates:
            for slot in (template.get("slots") or {}).values():
                if str(slot.get("source", "")).startswith("fixture:"):
                    slot["filter"] = "status IN ['available']"
                    return
        raise AssertionError("tiny pack has no fixture-backed slot to break")

    _edit_pack_yaml(pack / "task_templates.yaml", break_filter)
    temp_config = _write_tiny_config(
        tmp_path,
        "bad-filter.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(check for check in report["checks"] if check["id"] == 2)

    assert report["gold_eligible"] is False
    assert check["status"] == "fail"
    assert any(failure["reason"] == "unevaluable_filter" for failure in check["failures"])


def test_malformed_range_source_fails_prepare_check(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def break_source(templates: list[dict[str, Any]]) -> None:
        templates[0]["slots"]["book_id"]["source"] = "range:{'min': 1}"

    _edit_pack_yaml(pack / "task_templates.yaml", break_source)
    config = _write_tiny_config(
        tmp_path,
        "bad-range.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    check = next(item for item in report["checks"] if item["id"] == 2)
    assert check["status"] == "fail"
    assert any(failure["reason"] == "invalid_source" for failure in check["failures"])


def test_generate_rejects_noncanonical_resume_stage(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.checkpoint import (
        CheckpointError,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
    )

    temp_config = _write_tiny_config(tmp_path, "resume.yaml")

    with pytest.raises(CheckpointError, match="unknown BFCL resume stage"):
        generate_bfcl(temp_config, skip_until="RENDER")


def test_unrunnable_validation_cases_are_skipped_not_passed(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "tool_the_backend_never_exposes",
                "description": "Declared in tools.json only.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    tools_path.write_text(json.dumps(tools, indent=2) + "\n", encoding="utf-8")

    temp_config = _write_tiny_config(
        tmp_path,
        "schema-drift.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    alignment = next(check for check in report["checks"] if check["id"] == 3)
    cases = next(check for check in report["checks"] if check["id"] == 5)

    assert alignment["status"] == "fail"
    assert cases["status"] == "skipped"
    assert cases["failures"][0]["reason"] == "not_run"
    assert report["gold_eligible"] is False


def test_gold_requires_a_mutating_tool_to_say_so(tmp_path: Path) -> None:
    """A grader treats a read-only tool differently, so the claim must match reality."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    mutating = [tool for tool in tools if tool.pop("x-mutates", False)]
    assert mutating, "the tiny pack must declare a mutating tool for this test to mean anything"
    tools_path.write_text(json.dumps(tools, indent=2) + "\n", encoding="utf-8")

    config = _write_tiny_config(
        tmp_path,
        "undeclared_mutation.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))

    mutation_check = next(check for check in report["extra_checks"] if check["id"] == "M1")
    assert mutation_check["status"] == "fail"
    assert mutation_check["failures"][0]["reason"] == "undeclared_mutation"
    assert report["gold_eligible"] is False


def test_gold_rejects_a_declared_mutation_that_no_success_probe_observes(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    tools_path = pack / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    read_only = next(tool for tool in tools if tool["function"]["name"] == "get_book_status")
    read_only["x-mutates"] = True
    tools_path.write_text(json.dumps(tools, indent=2) + "\n", encoding="utf-8")

    config = _write_tiny_config(
        tmp_path,
        "false-mutation.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))

    mutation_check = next(check for check in report["extra_checks"] if check["id"] == "M1")
    assert mutation_check["status"] == "fail"
    assert any(
        failure["reason"] == "declared_mutation_not_observed" and failure["tool"] == "get_book_status"
        for failure in mutation_check["failures"]
    )


def test_determinism_uses_observed_success_not_only_the_expect_label(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def omit_success_label(cases: list[dict[str, Any]]) -> None:
        case = next(item for item in cases if item["id"] == "success_get_book_status")
        case["expect"].pop("result_class")

    _edit_pack_yaml(pack / "validation_cases.yaml", omit_success_label)
    config = _write_tiny_config(
        tmp_path,
        "inferred-success.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))

    determinism = next(check for check in report["extra_checks"] if check["id"] == "D1")
    assert determinism["status"] == "pass"
    assert report["gold_eligible"] is True


def test_determinism_compares_state_even_when_result_is_stable(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    backend = pack / "backend.py"
    source = backend.read_text(encoding="utf-8")
    source = source.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nimport time\n",
        1,
    ).replace(
        '    if name == "get_book_status":\n        return _get_book_status(arguments)',
        '    if name == "get_book_status":\n'
        '        _STATE["_nondeterministic_nonce"] = time.time_ns()\n'
        "        return _get_book_status(arguments)",
    )
    backend.write_text(source, encoding="utf-8")
    config = _write_tiny_config(
        tmp_path,
        "nondeterministic-state.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    determinism = next(check for check in report["extra_checks"] if check["id"] == "D1")
    assert determinism["status"] == "fail"
    assert any(failure["reason"] == "nondeterministic" for failure in determinism["failures"])


def test_pack_load_checks_what_the_guards_depend_on(tmp_path: Path) -> None:
    """A slot with no visibility flag lands in neither the preserve nor the omit set."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        normalize_templates,
    )

    template = {
        "template_id": "tpl",
        "turn_policy": "single_turn",
        "slots": {"thing_id": {"source": "literal:['T-1']", "visible_in_first_turn": True}},
    }

    # The paraphrase block is optional; the run-wide guards still apply without it.
    assert normalize_templates([template])[0]["paraphrase"] == {}
    assert normalize_templates([{**template, "edge_signatures": [" rare_b ", "rare_a"]}])[0]["edge_signatures"] == [
        "rare_a",
        "rare_b",
    ]

    with pytest.raises(ValueError, match="edge_signatures must be unique"):
        normalize_templates([{**template, "edge_signatures": ["rare", " rare "]}])
    with pytest.raises(ValueError, match="edge_signatures must be a list"):
        normalize_templates([{**template, "edge_signatures": "rare"}])

    unflagged = {**template, "slots": {"thing_id": {"source": "literal:['T-1']"}}}
    with pytest.raises(ValueError, match="visible_in_first_turn"):
        normalize_templates([unflagged])

    with pytest.raises(ValueError, match="unknown turn_policy"):
        normalize_templates([{**template, "turn_policy": "single-turn"}])

    missing_policy = {key: value for key, value in template.items() if key != "turn_policy"}
    with pytest.raises(ValueError, match="unknown turn_policy None"):
        normalize_templates([missing_policy])

    duplicate_milestones = {
        **template,
        "assistant_milestones": [
            {"id": "lookup", "type": "tool_call", "tool": "lookup_asset"},
            {"id": "lookup", "type": "tool_call", "tool": "inspect_asset"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate milestone id 'lookup'"):
        normalize_templates([duplicate_milestones])

    dependent_any = {
        **template,
        "turn_policy": "dependent_call",
        "call_order": "any",
    }
    with pytest.raises(ValueError, match="dependent_call requires call_order: strict"):
        normalize_templates([dependent_any])


def test_a_pack_may_name_its_own_confirmation_protocol() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        confirmation_protocol,
    )

    assert confirmation_protocol({})["parameter"] == "confirm"
    renamed = confirmation_protocol({"confirmation": {"parameter": "xac_nhan"}})
    assert renamed["parameter"] == "xac_nhan"
    assert renamed["pending_status"] == "awaiting_confirmation"

    with pytest.raises(ValueError, match="unknown keys"):
        confirmation_protocol({"confirmation": {"parameter_name": "xac_nhan"}})


def test_pack_fingerprint_covers_files_the_backend_reads(tmp_path: Path) -> None:
    """A helper module or data file the backend imports changes what the oracle does."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        pack_fingerprint,
        resolve_pack_paths,
    )

    pack = _copy_tiny_pack(tmp_path)
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "fingerprint.yaml",
            oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )
    paths = resolve_pack_paths(config)
    before = pack_fingerprint(paths)

    (pack / "policy.json").write_text('{"late_fee": 1}\n', encoding="utf-8")
    with_helper = pack_fingerprint(paths)
    assert with_helper != before

    (pack / "policy.json").write_text('{"late_fee": 2}\n', encoding="utf-8")
    assert pack_fingerprint(paths) != with_helper

    # An unchecked-hash pyc can override edited source, so it must be rejected rather
    # than ignored outside the fingerprint.
    import py_compile
    import subprocess
    import sys

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
        PackTrustError,
    )

    helper = pack / "shadow.py"
    helper.write_text("VALUE = 'stale-bytecode'\n", encoding="utf-8")
    cache = pack / "__pycache__"
    cache.mkdir(exist_ok=True)
    compiled = cache / f"shadow.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(
        str(helper),
        cfile=str(compiled),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    helper.write_text("VALUE = 'reviewed-source'\n", encoding="utf-8")
    observed = subprocess.run(
        [sys.executable, "-c", "import shadow; print(shadow.VALUE)"],
        cwd=pack,
        check=True,
        capture_output=True,
        text=True,
    )
    assert observed.stdout.strip() == "stale-bytecode"
    with pytest.raises(PackTrustError, match="executable Python bytecode"):
        pack_fingerprint(paths)
    with pytest.raises(PackTrustError, match="executable Python bytecode"):
        load_pack(config)


def test_pack_fingerprint_framing_cannot_collide_on_embedded_nuls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pack_loader

    first = tmp_path / "first"
    second = tmp_path / "second"
    combined = tmp_path / "combined"
    first.write_bytes(b"b")
    second.write_bytes(b"d")
    combined.write_bytes(b"b\0c\0d")

    def ambiguous_serialization(entries: dict[str, Path]) -> bytes:
        return b"".join(name.encode() + b"\0" + path.read_bytes() + b"\0" for name, path in sorted(entries.items()))

    split_entries = {"a": first, "c": second}
    combined_entries = {"a": combined}
    assert ambiguous_serialization(split_entries) == ambiguous_serialization(combined_entries)

    monkeypatch.setattr(
        pack_loader,
        "pack_entries",
        lambda _paths: split_entries,
    )
    split_digest = pack_loader.pack_fingerprint(None)  # type: ignore[arg-type]
    monkeypatch.setattr(
        pack_loader,
        "pack_entries",
        lambda _paths: combined_entries,
    )
    assert pack_loader.pack_fingerprint(None) != split_digest  # type: ignore[arg-type]


def test_pack_file_hashes_cover_exactly_what_the_fingerprint_hashes(
    tmp_path: Path,
) -> None:
    """The map and the aggregate must agree on the hashed set, or drift reports lie.

    A file the fingerprint covers but the map omits produces drift nobody can
    name; one the map carries but the fingerprint ignores reports a change that
    never affected the pack.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        pack_entries,
        pack_file_hashes,
        resolve_pack_paths,
    )

    pack = _copy_tiny_pack(tmp_path)
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "file-hashes.yaml",
            oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )
    paths = resolve_pack_paths(config)

    assert set(pack_file_hashes(paths)) == set(pack_entries(paths))

    (pack / "policy.json").write_text('{"late_fee": 1}\n', encoding="utf-8")
    before = pack_file_hashes(paths)
    assert "tree/policy.json" in before

    (pack / "policy.json").write_text('{"late_fee": 2}\n', encoding="utf-8")
    after = pack_file_hashes(paths)
    assert after["tree/policy.json"] != before["tree/policy.json"]
    assert {name: h for name, h in after.items() if name != "tree/policy.json"} == {
        name: h for name, h in before.items() if name != "tree/policy.json"
    }


def test_declared_pack_inputs_name_only_what_the_manifest_declares(
    tmp_path: Path,
) -> None:
    """An undeclared file is hashed but is not an oracle input, and the two differ."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        declared_pack_inputs,
        pack_file_hashes,
        resolve_pack_paths,
    )

    pack = _copy_tiny_pack(tmp_path)
    (pack / "NOTES.md").write_text("how this pack came to be\n", encoding="utf-8")
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "declared-inputs.yaml",
            oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )
    paths = resolve_pack_paths(config)

    declared = declared_pack_inputs(paths)
    assert "tree/NOTES.md" in pack_file_hashes(paths)
    assert "tree/NOTES.md" not in declared
    assert "tree/manifest.yaml" in declared
    assert "tree/tools.json" in declared
    assert declared <= set(pack_file_hashes(paths))


def test_pack_fingerprint_refuses_symbolic_links_in_the_pack_tree(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
        PackTrustError,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        pack_fingerprint,
        resolve_pack_paths,
    )

    pack = _copy_tiny_pack(tmp_path)
    outside = tmp_path / "mutable-policy.json"
    outside.write_text('{"late_fee": 1}\n', encoding="utf-8")
    (pack / "policy.json").symlink_to(outside)
    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "symlink-fingerprint.yaml",
            oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(tmp_path)]},
        )
    )

    with pytest.raises(PackTrustError, match="must not contain symbolic links"):
        pack_fingerprint(resolve_pack_paths(config))


def test_pack_fingerprint_uses_semantic_names_for_external_files(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        ResolvedPackPaths,
        pack_fingerprint,
    )

    def paths_for(host: str) -> ResolvedPackPaths:
        root = tmp_path / host / "asset_pack"
        shutil.copytree(
            BYOB_DIR / "data" / "tiny_oracle_pack",
            root,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        external = tmp_path / host / "runtime"
        external.mkdir(parents=True)
        backend = external / "oracle.py"
        backend.write_bytes((root / "backend.py").read_bytes())
        return ResolvedPackPaths(
            pack_root=root,
            manifest_path=root / "manifest.yaml",
            tools_path=root / "tools.json",
            fixtures_path=root / "fixtures.json",
            templates_path=root / "task_templates.yaml",
            assertions_path=root / "assertions.py",
            validation_cases_path=root / "validation_cases.yaml",
            system_prompt_path=None,
            backend_path=backend,
            endpoint_config_path=None,
        )

    assert pack_fingerprint(paths_for("host-a")) == pack_fingerprint(paths_for("host-b"))


def test_pack_fingerprint_is_reproducible_across_python_hash_seeds(
    tmp_path: Path,
) -> None:
    import os
    import subprocess
    import sys

    pack = _copy_tiny_pack(tmp_path)
    script = f"""
from pathlib import Path
from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    pack_fingerprint,
    resolve_declared_pack_paths,
)
root = Path({str(pack)!r})
paths = resolve_declared_pack_paths(
    OraclePackRef(manifest_path=root / "manifest.yaml"),
    (root,),
)
print(pack_fingerprint(paths))
"""
    observed = []
    for seed in ("1", "8675309"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        observed.append(result.stdout.strip())
    assert observed[0] == observed[1]


def test_gold_requires_every_template_to_state_what_success_means(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def drop_assertions(templates: list[dict[str, Any]]) -> None:
        for template in templates:
            template["success_assertions"] = []

    _edit_pack_yaml(pack / "task_templates.yaml", drop_assertions)
    temp_config = _write_tiny_config(
        tmp_path,
        "no-assertions.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )

    report = json.loads(prepare_bfcl(temp_config).read_text(encoding="utf-8"))
    check = next(item for item in report["checks"] if item["id"] == 4)

    # An importable assertions module is not evidence: a template that names no
    # assertion has no statement of success, so replay could only confirm the trace ran.
    assert report["stats"]["n_assertions"] > 0
    assert check["status"] == "fail"
    assert {failure["reason"] for failure in check["failures"]} == {"template_without_success_assertion"}
    assert report["gold_eligible"] is False


def test_generate_revalidates_when_worker_changes(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )

    process_config = _write_tiny_config(tmp_path, "process.yaml")
    process_report = json.loads(prepare_bfcl(process_config).read_text(encoding="utf-8"))
    assert process_report["gold_eligible"] is True

    thread_config = _write_tiny_config(
        tmp_path,
        "thread.yaml",
        oracle_runtime={"worker": "thread"},
    )
    with pytest.raises(RuntimeError, match="non-gold pack"):
        generate_bfcl(thread_config)

    report_path = (
        tmp_path / "output" / "bfcl_tiny_library_validation" / "stage_cache" / "oracle_validation_report.json"
    )
    thread_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert thread_report["validation_config_fingerprint"] != process_report["validation_config_fingerprint"]
    isolation = next(check for check in thread_report["extra_checks"] if check["id"] == "I1")
    assert isolation["status"] == "fail"


def test_duplicate_template_ids_are_rejected(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)

    def duplicate_first(templates: list[dict[str, Any]]) -> None:
        templates.append(dict(templates[0]))

    _edit_pack_yaml(pack / "task_templates.yaml", duplicate_first)
    config = _write_tiny_config(
        tmp_path,
        "duplicate-template.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    with pytest.raises(ValueError, match="duplicate template_id"):
        prepare_bfcl(config)


def test_system_prompt_cannot_escape_pack_allowlist(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
        PackTrustError,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    (tmp_path / "secret.txt").write_text("not a pack artifact", encoding="utf-8")

    def point_outside(manifest: dict[str, Any]) -> None:
        manifest["system_prompt_path"] = "../secret.txt"

    _edit_pack_yaml(pack / "manifest.yaml", point_outside)
    config = _write_tiny_config(
        tmp_path,
        "prompt-escape.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(pack)]},
    )
    with pytest.raises(PackTrustError):
        prepare_bfcl(config)


def test_default_config_requests_only_supported_features() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        _unsupported_requests,
    )

    config = BfclConfig.from_yaml(BFCL_CONFIG_DIR / "default.yaml")

    assert config.stage == "all"
    assert _unsupported_requests(config) == []


def test_the_publication_template_names_no_example_pack() -> None:
    """A template that defaults to a bundled pack publishes it to anyone who forgets."""
    data = yaml.safe_load((BFCL_CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))

    assert str(data["oracle_pack"]["manifest_path"]).startswith("REPLACE_ME")


def test_generate_refuses_settings_it_would_otherwise_ignore(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        _unsupported_requests,
    )

    baseline = BfclConfig.from_yaml(_write_tiny_config(tmp_path, "baseline.yaml"))
    assert _unsupported_requests(baseline) == []

    with pytest.raises(ValueError, match="must be zero"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "paraphrases.yaml",
                surface_generation={"paraphrases_per_template": 3},
            )
        )

    with pytest.raises(ValueError, match="surface_generation has unknown keys: langauge"):
        BfclConfig.from_yaml(_write_tiny_config(tmp_path, "typo.yaml", surface_generation={"langauge": "en"}))

    with pytest.raises(ValueError, match="judge_advisory must be null"):
        BfclConfig.from_yaml(
            _write_tiny_config(
                tmp_path,
                "judge.yaml",
                lineage={"judge_advisory": True},
            )
        )

    batched = BfclConfig.from_yaml(_write_tiny_config(tmp_path, "batch.yaml", ndd_batch_size=8))
    assert _unsupported_requests(batched) == []


def test_every_declared_export_format_is_supported(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        _unsupported_requests,
    )

    for index, name in enumerate(EXPORT_FORMATS):
        config = BfclConfig.from_yaml(_write_tiny_config(tmp_path, f"export-{index}.yaml", exports={name: True}))
        assert _unsupported_requests(config) == []

    both = BfclConfig.from_yaml(
        _write_tiny_config(tmp_path, "exports-all.yaml", exports=dict.fromkeys(EXPORT_FORMATS, True))
    )
    assert _unsupported_requests(both) == []


def test_an_unknown_export_name_is_never_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exports has unknown keys: bfcl_jsno"):
        BfclConfig.from_yaml(_write_tiny_config(tmp_path, "export-typo.yaml", exports={"bfcl_jsno": True}))


def test_generate_revalidates_a_hand_edited_gold_report(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pipeline
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )

    config = _write_tiny_config(tmp_path, "tampered.yaml")
    prepare_bfcl(config)
    report_path = (
        tmp_path / "output" / "bfcl_tiny_library_validation" / "stage_cache" / "oracle_validation_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gold_eligible"] is True
    # Manufacture a passing payload while retaining public fingerprints. Generate
    # must replace it by executing validation, not trust locally editable statuses.
    report["checks"] = [{"id": "invented", "status": "pass", "failures": []}]
    report["extra_checks"] = []
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    # Drop the in-process verdict so this stands in for a separate `stage=generate` run.
    pipeline._VALIDATED_THIS_PROCESS.clear()

    assert generate_bfcl(config).exists()
    repaired = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(repaired["checks"]) == 7
    assert {check["id"] for check in repaired["extra_checks"]} == {
        "M1",
        "D1",
        "D2",
        "T1",
        "I1",
    }


def test_stage_all_validates_once_but_a_new_run_revalidates(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pipeline
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        oracle_validation,
    )

    config = _write_tiny_config(tmp_path, "revalidate.yaml")
    calls: list[str] = []
    real_run = oracle_validation.run_oracle_validation

    def counted(cfg, pack):  # type: ignore[no-untyped-def]
        calls.append(str(cfg.output_dir))
        return real_run(cfg, pack)

    oracle_validation.run_oracle_validation = counted
    try:
        prepare_bfcl(config)
        assert generate_bfcl(config).exists()
        assert len(calls) == 1
        pipeline._VALIDATED_THIS_PROCESS.clear()
        assert generate_bfcl(config).exists()
        assert len(calls) == 2
        # A release revalidation must not be answerable from this process's own memory,
        # otherwise a frozen pack would inherit a verdict computed before it was frozen.
        prepare_bfcl(config, force_validation=True)
        assert len(calls) == 3
    finally:
        oracle_validation.run_oracle_validation = real_run


def test_resolved_config_hash_input_is_portable_for_external_packs(
    tmp_path: Path,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
        _resolved_config,
    )

    machine_a = tmp_path / "host-a"
    machine_b = tmp_path / "host-b"
    config_a = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "portable-a.yaml",
            output_dir=str(machine_a / "output"),
            oracle_pack={"manifest_path": str(machine_a / "asset_pack" / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(machine_a / "asset_pack")]},
        )
    )
    config_b = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "portable-b.yaml",
            output_dir=str(machine_b / "output"),
            oracle_pack={"manifest_path": str(machine_b / "asset_pack" / "manifest.yaml")},
            oracle_runtime={"allowed_roots": [str(machine_b / "asset_pack")]},
        )
    )

    assert _resolved_config(config_a) == _resolved_config(config_b)


def test_surface_quality_settings_are_supported_by_generation(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        _unsupported_requests,
    )

    temp_config = _write_tiny_config(
        tmp_path,
        "judged.yaml",
        lineage={
            "judge_advisory": False,
            "roles": {
                "surface_judge": {
                    "enabled": True,
                    "model_config": {
                        "alias": "surface-judge",
                        "provider": "nvidia",
                        "model": "judge-model",
                        "canonical_id": "source::judge-model@revision",
                    },
                }
            },
        },
        surface_quality_validation={"enabled": True, "drop_authority": True},
    )

    config = BfclConfig.from_yaml(temp_config)
    assert _unsupported_requests(config) == []


def test_generate_refuses_ignored_task_generation_controls(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
    )

    temp_config = _write_tiny_config(
        tmp_path,
        "balanced.yaml",
        task_generation={"turn_mix": {"single_turn": 1.0}},
    )
    with pytest.raises(NotImplementedError, match="task_generation.turn_mix"):
        generate_bfcl(temp_config)


def test_stage_eleven_enables_implemented_balancing_controls(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        _unsupported_requests,
    )

    config = BfclConfig.from_yaml(
        _write_tiny_config(
            tmp_path,
            "stage-eleven-balancing.yaml",
            surface_quality_validation={"enabled": True},
            semantic_deduplication_config={
                "enabled": True,
                "model_identifier": "test/embedding",
                "n_clusters": 4,
                "eps": 0.08,
                "remove_duplicates": True,
            },
            task_generation={
                "difficulty_mix": {"easy": 1.0},
                "turn_mix": {"single_turn": 1.0},
                "tool_call_count_mix": {"1": 1.0},
                "max_turns": 3,
                "max_tool_calls": 2,
            },
        )
    )

    assert _unsupported_requests(config) == []


def test_generate_revalidates_when_pack_changed(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )

    pack = _copy_tiny_pack(tmp_path)
    temp_config = _write_tiny_config(
        tmp_path,
        "stale.yaml",
        oracle_pack={"manifest_path": str(pack / "manifest.yaml")},
        oracle_runtime={"allowed_roots": [str(tmp_path)]},
    )
    report_path = prepare_bfcl(temp_config)
    assert json.loads(report_path.read_text(encoding="utf-8"))["gold_eligible"] is True

    def drop_negative_cases(cases: list[dict[str, Any]]) -> None:
        cases[:] = [case for case in cases if case["id"].startswith("success_")]

    _edit_pack_yaml(pack / "validation_cases.yaml", drop_negative_cases)

    with pytest.raises(RuntimeError, match="non-gold pack"):
        generate_bfcl(temp_config)

    assert json.loads(report_path.read_text(encoding="utf-8"))["gold_eligible"] is False
