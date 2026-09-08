"""The eval config contract, resolved before any candidate is contacted."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    EVAL_CONFIG_SCHEMA_VERSION,
    BfclEvalConfig,
    CandidateIdentityError,
    EvalConfigPathError,
    EvalConfigSchemaError,
    MutableCandidateRevisionError,
    PublicationPolicyError,
    SecretInConfigError,
    UnsupportedEvalModeError,
    eval_config_reference,
    load_eval_config,
    load_eval_config_for_generation,
    resolved_eval_config_document,
    write_resolved_eval_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.cli_orchestration import (
    load_bfcl_eval_cli_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.config import describe_eval_config_error

BYOB_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"
BFCL_CONFIG_DIR = BYOB_DIR / "bfcl" / "config"
SHIPPED_EVAL_CONFIG = BFCL_CONFIG_DIR / "eval.default.yaml"
SHIPPED_SCORING_CONTRACT = BYOB_DIR / "references" / "bfcl-eval-scoring-contract.md"
RUN_EVALUATION_GUIDE = (
    Path(__file__).resolve().parents[3]
    / "docs/build-benchmarks/function-calling/how-to/run-evaluation.md"
)

IMMUTABLE_REVISION = "9f2c1b7d4e6a8c0b2d4f6a8c0e2b4d6f8a0c2e4b"
OTHER_REVISION = "1a3c5e7b9d0f2a4c6e8b0d2f4a6c8e0b2d4f6a8c"


def _hash(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_run_evaluation_copy_flow_resolves_repository_relative_paths(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate-a"
    destination.mkdir()
    eval_path = destination / "eval.yaml"
    cli_path = destination / "eval.cli.yaml"
    launcher_path = destination / "eval.launcher.yaml"
    shutil.copyfile(SHIPPED_EVAL_CONFIG, eval_path)
    shutil.copyfile(BFCL_CONFIG_DIR / "eval.cli.yaml", cli_path)
    shutil.copyfile(BFCL_CONFIG_DIR / "eval.launcher.yaml", launcher_path)

    evaluation = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    evaluation["scoring"]["contract"] = str(SHIPPED_SCORING_CONTRACT.resolve())
    eval_path.write_text(
        yaml.safe_dump(evaluation, sort_keys=False),
        encoding="utf-8",
    )
    direct = yaml.safe_load(cli_path.read_text(encoding="utf-8"))
    direct["eval_config_path"] = str(eval_path.resolve())
    cli_path.write_text(yaml.safe_dump(direct, sort_keys=False), encoding="utf-8")
    launcher = yaml.safe_load(launcher_path.read_text(encoding="utf-8"))
    launcher["eval_config_path"] = str(eval_path.resolve())
    launcher["launcher"]["launcher_base_config_path"] = str(
        (
            Path(__file__).resolve().parents[3]
            / "src/nemotron/steps/eval/model_eval/config/tiny_chat.yaml"
        ).resolve()
    )
    launcher_path.write_text(
        yaml.safe_dump(launcher, sort_keys=False),
        encoding="utf-8",
    )

    copied = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    loaded_cli = load_bfcl_eval_cli_config(cli_path)
    copied_launcher = yaml.safe_load(launcher_path.read_text(encoding="utf-8"))
    guide = RUN_EVALUATION_GUIDE.read_text(encoding="utf-8")

    assert Path(copied["scoring"]["contract"]).is_file()
    assert loaded_cli.eval_config_path == eval_path.resolve()
    assert Path(copied_launcher["eval_config_path"]).is_absolute()
    assert Path(
        copied_launcher["launcher"]["launcher_base_config_path"]
    ).is_file()
    assert 'REPO_ROOT="$(pwd)"' in guide
    assert "document[\"scoring\"][\"contract\"]" in guide
    assert "Resolve every path in the copied Launcher envelope" in guide


def _published_run(
    tmp_path: Path,
    *,
    run_id: str = "expt-20260819T090000000000Z-abc-1",
    gold_eligible: bool = True,
    oracle: bool = True,
    schema_version: str = "1.1",
    name: str = "published",
    manifest_extra: dict[str, Any] | None = None,
) -> Path:
    """Write a directory that looks like a committed Stage 12 publication tree."""
    run_dir = tmp_path / name / "expt"
    run_dir.mkdir(parents=True)
    pack_dir = tmp_path / name / "oracle_pack"
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "pack_id": "test_pack",
                "version": "1.0.0",
                "paths": {"backend": "backend.py"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (pack_dir / "backend.py").write_text("def reset():\n    return None\n", encoding="utf-8")
    table = b"parquet-bytes-for-" + run_id.encode()
    (run_dir / "benchmark.parquet").write_bytes(table)
    (run_dir / "benchmark_raw.parquet").write_bytes(b"raw-" + table)
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "lineage_policy": "strict_separation",
        "gold_eligible": gold_eligible,
        "tier": "gold" if gold_eligible else "silver",
        "pack": {
            "pack_id": "test_pack",
            "version": "1.0.0",
            "content_hash": _hash(b"test-pack-tree"),
        },
        "publication": {
            "published": {"file": "benchmark.parquet", "rows": 12, "content_hash": _hash(b"rows")},
        },
        "artifacts": {"benchmark_parquet": {"content_hash": _hash(table)}},
    }
    if oracle:
        manifest["oracle"] = {"kind": "python", "endpoint_metadata": None}
    manifest.update(manifest_extra or {})
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def _scoring_contract(tmp_path: Path, *, text: str = "argument match: schema then canonical\n") -> Path:
    contract = tmp_path / "contracts" / "eval_spec.md"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(text, encoding="utf-8")
    return contract


def _config_data(run_dir: Path, contract: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    pack_dir = run_dir.parent / "oracle_pack"
    source_oracle = (
        {
            "kind": manifest["oracle"]["kind"],
            "pack_manifest": str(pack_dir / "manifest.yaml"),
            "resource": str(pack_dir / "backend.py"),
        }
        if isinstance(manifest.get("oracle"), dict)
        else None
    )
    return {
        "schema_version": EVAL_CONFIG_SCHEMA_VERSION,
        "config_status": "resolved",
        "source_run_manifest": str(run_dir / "run_manifest.json"),
        "source_oracle": source_oracle,
        "translation_manifest": None,
        "eval": {"mode": ["trace"]},
        "scoring": {
            "contract": str(contract),
            "argument_matching": "schema_then_canonical",
            "insert_declared_defaults": True,
            "respect_call_order": True,
            "respect_call_group": True,
            "allow_llm_repair": False,
            "task_success": "all_applicable_gates",
        },
        "limits": {
            "max_turns": 5,
            "tool_timeout_s": 10.0,
            "candidate_timeout_s": 60.0,
            "episode_timeout_s": 300.0,
            "max_parallel_tasks": 1,
            "max_retries": 2,
        },
        "candidates": [
            {
                "alias": "candidate_a",
                "model": "nemotron-route-a",
                "provider": "nvidia",
                "provider_api_version": "v1",
                "api": {"base_url": "https://integrate.example.com/v1", "api_key_env": "NVIDIA_API_KEY"},
                "model_identity": {
                    "source": "huggingface",
                    "model": "org/model-a",
                    "revision": IMMUTABLE_REVISION,
                    "weights_digest": None,
                },
                "inference": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "seed": 42,
                    "tool_choice": "auto",
                },
            }
        ],
        "contamination": {
            "enforce": True,
            "on_violation": "fail_run",
            "comparison_set": "common_intersection",
        },
        "publication": {"requested": True, "require_same_task_ids": True},
        "outputs": {
            "output_dir": str(output_dir),
            "write_task_results": True,
            "write_eval_manifest": True,
            "cache_candidate_responses": True,
            "cache_tool_results": True,
        },
    }


def _write(config_dir: Path, data: dict[str, Any], name: str = "eval_config.yaml") -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def valid_config(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A resolved, publishable trace-only config plus its mutable source data."""
    run_dir = _published_run(tmp_path)
    contract = _scoring_contract(tmp_path)
    data = _config_data(run_dir, contract, tmp_path / "eval_out")
    return _write(tmp_path / "eval", data), data


def _load(tmp_path: Path, data: dict[str, Any], name: str = "eval_config.yaml") -> BfclEvalConfig:
    return load_eval_config(_write(tmp_path / "eval", data, name))


def test_a_resolved_trace_only_config_loads_frozen_and_publishable(
    valid_config: tuple[Path, dict[str, Any]],
) -> None:
    config = load_eval_config(valid_config[0])

    assert config.schema_version == EVAL_CONFIG_SCHEMA_VERSION
    assert config.settings.modes == ("trace",)
    assert config.publication_scope == "trace_only"
    assert config.publication_allowed is True
    assert config.non_publication_reasons == ()
    assert config.candidate_aliases == ("candidate_a",)
    assert config.eval_config_hash.startswith("sha256:")
    with pytest.raises(Exception):
        config.settings = None  # type: ignore[misc]


def test_executable_mode_is_canonically_ordered_and_needs_an_oracle(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["eval"]["mode"] = ["executable", "trace"]

    config = _load(tmp_path, data)

    assert config.settings.modes == ("trace", "executable")
    assert config.publication_scope == "trace_and_executable"
    assert config.source.oracle_kind == "python"


def test_held_out_eval_mode_requires_a_versioned_pin_and_private_outputs(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["eval"]["mode"] = ["held_out_eval"]
    data["held_out_eval"] = {
        "contract_version": "1.0",
        "policy_hash": "sha256:" + "7" * 64,
        "fixture_refs": ['["things","T-2"]'],
        "template_ids": ["tpl-held"],
        "seed": 17,
        "pack_version": "1.0.0",
        "max_tasks_per_template": 4,
    }
    data["publication"]["requested"] = False
    data["outputs"]["write_task_results"] = False
    data["outputs"]["cache_candidate_responses"] = False
    data["outputs"]["cache_tool_results"] = False

    config = _load(tmp_path, data)

    assert config.settings.held_out_eval is True
    assert config.settings.executable is True
    assert config.held_out_eval is not None
    assert config.held_out_eval.selection_mode == "both"

    data["outputs"]["cache_candidate_responses"] = True
    with pytest.raises(EvalConfigSchemaError, match="may not persist private"):
        _load(tmp_path, data, "unsafe-held-out.yaml")

    data["outputs"]["cache_candidate_responses"] = False
    data["contamination"]["comparison_set"] = "per_candidate"
    with pytest.raises(EvalConfigSchemaError, match="common seen task set"):
        _load(tmp_path, data, "incomparable-held-out.yaml")


def test_executable_mode_is_refused_when_the_source_declares_no_oracle(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path, oracle=False)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["eval"]["mode"] = ["trace", "executable"]

    with pytest.raises(EvalConfigSchemaError, match="needs a resolvable oracle"):
        _load(tmp_path, data)


def test_executable_mode_is_refused_when_only_the_manifest_oracle_label_exists(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["eval"]["mode"] = ["trace", "executable"]
    data["source_oracle"] = None

    with pytest.raises(EvalConfigSchemaError, match="needs a resolvable oracle"):
        _load(tmp_path, data)


def test_trace_only_mode_does_not_require_oracle_availability(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["source_oracle"] = None

    config = _load(tmp_path, data)

    assert config.settings.modes == ("trace",)
    assert config.source.oracle is None


def test_source_oracle_kind_must_match_the_source_run(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["source_oracle"]["kind"] = "endpoint"

    with pytest.raises(EvalConfigPathError, match="does not match oracle.kind"):
        _load(tmp_path, data)


def test_source_oracle_resource_must_exist_at_config_parse_time(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["source_oracle"]["resource"] = str(tmp_path / "missing-backend.py")

    with pytest.raises(EvalConfigPathError, match="does not exist"):
        _load(tmp_path, data)


def test_source_oracle_pack_identity_must_match_the_source_run(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    pack_manifest = Path(data["source_oracle"]["pack_manifest"])
    pack = yaml.safe_load(pack_manifest.read_text(encoding="utf-8"))
    pack["pack_id"] = "some_other_pack"
    pack_manifest.write_text(yaml.safe_dump(pack), encoding="utf-8")

    with pytest.raises(EvalConfigPathError, match="does not identify the pack"):
        _load(tmp_path, data)


def test_executable_publication_requires_a_gold_eligible_source(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path, gold_eligible=False)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["eval"]["mode"] = ["trace", "executable"]

    with pytest.raises(PublicationPolicyError, match="source.gold_eligible"):
        _load(tmp_path, data)

    data["publication"]["requested"] = False
    config = _load(tmp_path, data)
    assert config.publication_allowed is False
    assert "source.gold_eligible" in config.non_publication_reasons


def test_relative_paths_resolve_from_the_config_directory(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    contract = _scoring_contract(tmp_path)
    config_dir = tmp_path / "eval"
    config_dir.mkdir()
    data = _config_data(run_dir, contract, tmp_path / "eval_out")
    data["source_run_manifest"] = "../published/expt/run_manifest.json"
    data["scoring"]["contract"] = "../contracts/eval_spec.md"
    data["outputs"]["output_dir"] = "outputs"

    config = load_eval_config(_write(config_dir, data))

    assert config.source.run_manifest.path == run_dir / "run_manifest.json"
    assert config.scoring.contract.path == contract
    assert config.outputs.output_dir == config_dir / "outputs"


def test_an_unknown_key_is_never_silently_ignored(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["max_turns"] = 5

    with pytest.raises(EvalConfigSchemaError, match="unknown top-level key"):
        _load(tmp_path, data)


def test_an_unknown_section_key_is_rejected(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["scoring"]["partial_credit"] = True

    with pytest.raises(EvalConfigSchemaError, match="scoring: unknown key"):
        _load(tmp_path, data)


def test_a_missing_setting_is_an_error_rather_than_a_default(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    del data["limits"]["candidate_timeout_s"]

    with pytest.raises(EvalConfigSchemaError, match="missing required key"):
        _load(tmp_path, data)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("scoring", "allow_llm_repair", "false"),
        ("scoring", "respect_call_order", "true"),
        ("contamination", "enforce", "true"),
        ("publication", "requested", "yes"),
    ],
)
def test_a_quoted_boolean_never_becomes_a_boolean(
    tmp_path: Path,
    section: str,
    key: str,
    value: str,
) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data[section][key] = value

    with pytest.raises(EvalConfigSchemaError) as failure:
        _load(tmp_path, data)

    assert f"{section}.{key}" in str(failure.value)


def test_a_quoted_number_never_becomes_a_number(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["inference"]["temperature"] = "0.0"

    with pytest.raises(EvalConfigSchemaError, match="temperature"):
        _load(tmp_path, data)


@pytest.mark.parametrize(
    ("modes", "expected"),
    [
        ([], UnsupportedEvalModeError),
        (["trace", "trace"], UnsupportedEvalModeError),
        (["semantic"], UnsupportedEvalModeError),
        (["trace", 1], EvalConfigSchemaError),
        ("trace", EvalConfigSchemaError),
    ],
    ids=["empty", "repeated", "unknown", "not_a_string", "not_a_list"],
)
def test_modes_must_be_a_distinct_non_empty_list_of_known_modes(
    tmp_path: Path,
    modes: Any,
    expected: type[Exception],
) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["eval"]["mode"] = modes

    with pytest.raises(expected):
        _load(tmp_path, data)


def _second_candidate(**overrides: Any) -> dict[str, Any]:
    candidate = {
        "alias": "candidate_b",
        "model": "nemotron-route-b",
        "provider": "nvidia",
        "provider_api_version": "v1",
        "api": {"base_url": "https://integrate.example.com/v1", "api_key_env": "NVIDIA_API_KEY"},
        "model_identity": {
            "source": "huggingface",
            "model": "org/model-b",
            "revision": OTHER_REVISION,
            "weights_digest": None,
        },
        "inference": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "seed": 42,
            "tool_choice": "auto",
        },
    }
    candidate.update(overrides)
    return candidate


def test_two_candidates_may_not_share_an_alias(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"].append(_second_candidate(alias="candidate_a"))

    with pytest.raises(CandidateIdentityError, match="alias"):
        _load(tmp_path, data)


def test_two_candidates_may_not_resolve_to_the_same_weights(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    twin = _second_candidate()
    twin["model_identity"] = dict(data["candidates"][0]["model_identity"])
    data["candidates"].append(twin)

    with pytest.raises(CandidateIdentityError, match="same weights"):
        _load(tmp_path, data)


def test_case_sensitive_registry_identities_do_not_collapse(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    first_identity = data["candidates"][0]["model_identity"]
    first_identity["source"] = "custom"
    first_identity["model"] = "Org/Model"
    second = _second_candidate()
    second["model_identity"] = {
        "source": "custom",
        "model": "org/model",
        "revision": IMMUTABLE_REVISION,
        "weights_digest": None,
    }
    data["candidates"].append(second)

    config = _load(tmp_path, data)

    assert config.candidates[0].canonical_model_identity != config.candidates[1].canonical_model_identity


def test_a_config_without_candidates_has_nothing_to_evaluate(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"] = []

    with pytest.raises(CandidateIdentityError, match="no candidate"):
        _load(tmp_path, data)


@pytest.mark.parametrize(
    "revision",
    [
        "main",
        "MAIN",
        "latest",
        "master",
        "refs/heads/release-1",
        "prod",
    ],
)
def test_a_moving_revision_is_refused(tmp_path: Path, revision: str) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["model_identity"]["revision"] = revision

    with pytest.raises(MutableCandidateRevisionError, match="moving pointer"):
        _load(tmp_path, data)


@pytest.mark.parametrize("revision", ["feature/release-2026", "release-2026", "v1", "team-private-branch"])
def test_a_non_commit_revision_is_refused_even_when_it_is_not_on_a_denylist(
    tmp_path: Path,
    revision: str,
) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["model_identity"]["revision"] = revision

    with pytest.raises(MutableCandidateRevisionError, match="not a verifiable immutable commit"):
        _load(tmp_path, data)


def _unpinned(candidate: dict[str, Any]) -> dict[str, Any]:
    """Strip a candidate's pin, leaving the route as the only thing that names it."""
    candidate["model_identity"] = {
        "source": candidate["provider"],
        "model": candidate["model"],
        "revision": None,
        "weights_digest": None,
    }
    return candidate


def test_a_candidate_that_pins_nothing_is_scored_as_provider_managed(tmp_path: Path) -> None:
    """A hosted route that publishes no pin is recorded, not refused or invented."""
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    _unpinned(data["candidates"][0])
    data["publication"]["requested"] = False

    config = _load(tmp_path, data)

    identity = config.candidates[0].model_identity
    assert identity.assurance == "provider_managed"
    assert config.candidates[0].canonical_model_identity == "nvidia:nemotron-route-a@provider_managed"
    assert "candidates[candidate_a].model_identity" in config.non_publication_reasons
    assert config.publication_allowed is False


def test_publication_is_refused_while_a_candidate_pins_no_weights(tmp_path: Path) -> None:
    """The cost of not pinning is paid at load, not after a run has been scored."""
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    _unpinned(data["candidates"][0])

    with pytest.raises(PublicationPolicyError, match=r"candidates\[candidate_a\].model_identity"):
        _load(tmp_path, data)


def test_an_unpinned_candidate_may_not_name_weights_other_than_its_route(tmp_path: Path) -> None:
    """Without a pin, a free-text identity is a claim nothing in the run supports."""
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["model_identity"] = {
        "source": "openai",
        "model": "some-other-model",
        "revision": None,
        "weights_digest": None,
    }
    data["publication"]["requested"] = False

    with pytest.raises(CandidateIdentityError, match="must name the route that answered"):
        _load(tmp_path, data)


def test_two_unpinned_candidates_on_one_route_cannot_be_told_apart(tmp_path: Path) -> None:
    """Naming the route is what makes the duplicate check see one deployment."""
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    _unpinned(data["candidates"][0])
    twin = _second_candidate(model=data["candidates"][0]["model"])
    data["candidates"].append(_unpinned(twin))
    data["publication"]["requested"] = False

    with pytest.raises(CandidateIdentityError, match="same weights"):
        _load(tmp_path, data)


def test_a_config_declaring_the_previous_schema_still_loads_unchanged(tmp_path: Path) -> None:
    """1.2 widened the contract, so a 1.1 config keeps loading and keeps its version."""
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["schema_version"] = "1.1"

    config = _load(tmp_path, data)

    assert config.schema_version == "1.1"
    assert config.semantic_payload()["schema_version"] == "1.1"


@pytest.mark.parametrize(
    ("identity_change", "expected_message"),
    [
        ({"revision": None}, "schema 1.1 required every candidate to pin"),
        (
            {"revision": None, "weights_digest": f"bfcl-weight-manifest-v1:{'a' * 64}"},
            "schema 1.1 cannot read",
        ),
    ],
)
def test_the_previous_schema_refuses_what_only_the_current_one_added(
    tmp_path: Path,
    identity_change: dict[str, Any],
    expected_message: str,
) -> None:
    """A file that says 1.1 promises 1.1 readers it holds nothing they would refuse."""
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["schema_version"] = "1.1"
    data["publication"]["requested"] = False
    _unpinned(data["candidates"][0])
    data["candidates"][0]["model_identity"].update(identity_change)

    with pytest.raises(CandidateIdentityError, match=expected_message):
        _load(tmp_path, data)


def test_a_manifest_scoped_weights_digest_is_accepted_and_kept_distinct(tmp_path: Path) -> None:
    """The scheme travels with the digest so a later comparison can see the scope."""
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    digest = f"bfcl-weight-manifest-v1:{'b' * 64}"
    data["candidates"][0]["model_identity"]["revision"] = None
    data["candidates"][0]["model_identity"]["weights_digest"] = digest

    config = _load(tmp_path, data)

    assert config.candidates[0].canonical_model_identity.endswith(digest)
    assert config.publication_allowed is True


def test_a_pinned_identity_hashes_the_fields_it_hashed_before_assurance_existed(
    valid_config: tuple[Path, dict[str, Any]],
) -> None:
    """Assurance is derived, so recording it would refork every published hash."""
    config = load_eval_config(valid_config[0])

    payload = config.candidates[0].model_identity.semantic_payload()

    assert set(payload) == {"source", "model", "revision", "weights_digest", "canonical_id"}
    assert config.candidates[0].model_identity.assurance == "weights_pinned"


def test_a_weights_digest_must_use_the_pinned_format(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["model_identity"]["revision"] = None
    data["candidates"][0]["model_identity"]["weights_digest"] = "md5:deadbeef"

    with pytest.raises(EvalConfigSchemaError, match="weights_digest"):
        _load(tmp_path, data)


def test_a_digest_only_candidate_is_accepted(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["model_identity"]["revision"] = None
    data["candidates"][0]["model_identity"]["weights_digest"] = _hash(b"weights")

    config = _load(tmp_path, data)

    assert config.candidates[0].canonical_model_identity.endswith(_hash(b"weights"))


def test_a_credential_key_is_refused_before_anything_is_hashed(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["api"]["api_key"] = "nvapi-EXAMPLE_ONLY_not_a_real_key"

    with pytest.raises(SecretInConfigError) as failure:
        _load(tmp_path, data)

    assert "nvapi-" not in str(failure.value)
    assert "<redacted>" in str(failure.value)


def test_a_credential_value_is_refused_whatever_field_holds_it(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["provider_api_version"] = "nvapi-EXAMPLE_ONLY_wrong_field"

    with pytest.raises(SecretInConfigError) as failure:
        _load(tmp_path, data)

    assert "nvapi-EXAMPLE_ONLY_wrong_field" not in str(failure.value)


def test_a_missing_api_key_environment_variable_is_not_a_config_error(
    valid_config: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = load_eval_config(valid_config[0])

    assert config.candidates[0].api.api_key_env == "NVIDIA_API_KEY"


def test_a_base_url_may_not_carry_credentials(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["api"]["base_url"] = "https://user:pass@integrate.example.com/v1"

    with pytest.raises(EvalConfigSchemaError, match="base_url") as failure:
        _load(tmp_path, data)
    assert "user:pass" not in str(failure.value)
    assert "integrate.example.com" not in str(failure.value)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("contamination", "enforce", False),
        ("contamination", "on_violation", "exclude_row"),
        ("contamination", "comparison_set", "per_candidate"),
        ("scoring", "allow_llm_repair", True),
        ("scoring", "argument_matching", "canonical_only"),
        ("scoring", "task_success", "assertions_only"),
        ("scoring", "respect_call_order", False),
        ("scoring", "respect_call_group", False),
        ("outputs", "write_task_results", False),
        ("outputs", "write_eval_manifest", False),
        ("outputs", "cache_candidate_responses", False),
        ("outputs", "cache_tool_results", False),
    ],
)
def test_publication_is_refused_when_a_locked_gate_is_relaxed(
    tmp_path: Path,
    section: str,
    key: str,
    value: Any,
) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data[section][key] = value

    with pytest.raises(PublicationPolicyError) as failure:
        _load(tmp_path, data)
    assert f"{section}.{key}" in str(failure.value)

    data["publication"]["requested"] = False
    debug = _load(tmp_path, data)
    assert debug.publication_allowed is False
    assert f"{section}.{key}" in debug.non_publication_reasons


def test_a_debug_config_states_every_reason_it_may_not_publish(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["contamination"]["on_violation"] = "exclude_row"
    data["scoring"]["allow_llm_repair"] = True
    data["publication"]["requested"] = False

    config = _load(tmp_path, data)

    assert config.publication_allowed is False
    assert config.non_publication_reasons == (
        "scoring.allow_llm_repair",
        "contamination.on_violation",
        "publication.requested",
    )


def test_scoring_on_different_task_sets_is_never_allowed(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["publication"]["require_same_task_ids"] = False

    with pytest.raises(PublicationPolicyError, match="require_same_task_ids"):
        _load(tmp_path, data)


def test_a_template_config_is_not_runnable(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["config_status"] = "template"

    with pytest.raises(EvalConfigSchemaError, match="template is not runnable"):
        _load(tmp_path, data)


def test_a_leftover_placeholder_is_refused(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["model"] = "REPLACE_ME_SERVING_ROUTE"

    with pytest.raises(EvalConfigSchemaError, match="placeholder"):
        _load(tmp_path, data)


def test_the_source_must_be_a_manifest_not_a_benchmark_table(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["source_run_manifest"] = str(run_dir / "benchmark.parquet")

    with pytest.raises(EvalConfigPathError, match="instead of a manifest"):
        _load(tmp_path, data)


def test_a_benchmark_without_its_manifest_is_not_published_output(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    (run_dir / "run_manifest.json").unlink()

    with pytest.raises(EvalConfigPathError, match="does not exist"):
        _load(tmp_path, data)


def test_a_manifest_that_is_not_json_is_refused(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    (run_dir / "run_manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(EvalConfigPathError, match="not a JSON object"):
        _load(tmp_path, data)


def test_a_benchmark_schema_this_build_cannot_read_is_refused(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path, schema_version="9.9")
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")

    with pytest.raises(EvalConfigPathError, match="cannot read"):
        _load(tmp_path, data)


def test_a_manifest_without_the_published_table_hash_is_refused(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["benchmark_parquet"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")

    with pytest.raises(EvalConfigPathError, match="content hash"):
        _load(tmp_path, data)


def test_a_missing_published_table_is_refused(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    (run_dir / "benchmark.parquet").unlink()
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")

    with pytest.raises(EvalConfigPathError, match="does not exist"):
        _load(tmp_path, data)


def _translation_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "translated" / "translation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_a_translation_manifest_must_name_the_same_source_run(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["translation_manifest"] = str(_translation_manifest(tmp_path, {"source_run_id": "some-other-run"}))

    with pytest.raises(EvalConfigPathError, match="does not reference the source run"):
        _load(tmp_path, data)


@pytest.mark.parametrize("by", ["run_id", "manifest_hash"])
def test_a_translation_manifest_may_bind_by_run_id_or_manifest_hash(tmp_path: Path, by: str) -> None:
    run_dir = _published_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    run_id = json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"]
    payload = (
        {"source_run_id": run_id}
        if by == "run_id"
        else {"source_run_manifest_content_hash": _hash(manifest_path.read_bytes())}
    )
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["translation_manifest"] = str(_translation_manifest(tmp_path, payload))

    config = _load(tmp_path, data)

    assert config.source.translation_manifest is not None


@pytest.mark.parametrize("relation", ["inside_source", "contains_source", "is_source"])
def test_eval_output_may_not_overlap_the_source_publication_tree(tmp_path: Path, relation: str) -> None:
    run_dir = _published_run(tmp_path)
    output = {
        "inside_source": run_dir / "eval",
        "contains_source": run_dir.parent,
        "is_source": run_dir,
    }[relation]
    data = _config_data(run_dir, _scoring_contract(tmp_path), output)

    with pytest.raises(EvalConfigPathError, match="overlaps the source publication tree"):
        _load(tmp_path, data)


def test_eval_output_may_not_land_on_another_published_tree(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    other = _published_run(tmp_path, name="other", run_id="other-run")
    data = _config_data(run_dir, _scoring_contract(tmp_path), other)

    with pytest.raises(EvalConfigPathError, match="generation artifact"):
        _load(tmp_path, data)


def test_eval_output_may_not_be_an_existing_file(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    output_file = tmp_path / "occupied"
    output_file.write_text("not a directory", encoding="utf-8")
    data = _config_data(run_dir, _scoring_contract(tmp_path), output_file)

    with pytest.raises(EvalConfigPathError, match="exists but is not a directory"):
        _load(tmp_path, data)


def test_a_missing_scoring_contract_is_refused(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    contract = _scoring_contract(tmp_path)
    data = _config_data(run_dir, contract, tmp_path / "eval_out")
    contract.unlink()

    with pytest.raises(EvalConfigPathError, match="scoring.contract"):
        _load(tmp_path, data)


def test_an_episode_may_not_time_out_before_one_call_can_finish(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["limits"]["episode_timeout_s"] = 5.0

    with pytest.raises(EvalConfigSchemaError, match="episode_timeout_s"):
        _load(tmp_path, data)


@pytest.mark.parametrize(
    ("key", "value"),
    [("max_turns", 0), ("tool_timeout_s", 0), ("max_parallel_tasks", 0), ("max_retries", -1)],
)
def test_limits_are_bounded(tmp_path: Path, key: str, value: Any) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["limits"][key] = value

    with pytest.raises(EvalConfigSchemaError, match=key):
        _load(tmp_path, data)


@pytest.mark.parametrize(("key", "value"), [("top_p", 0.0), ("top_p", 1.5), ("max_tokens", 0), ("temperature", -0.1)])
def test_inference_parameters_are_bounded(tmp_path: Path, key: str, value: Any) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["inference"][key] = value

    with pytest.raises(EvalConfigSchemaError, match=key):
        _load(tmp_path, data)


def test_a_provider_only_inference_field_needs_a_versioned_namespace(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["inference"]["reasoning_effort"] = "high"

    with pytest.raises(EvalConfigSchemaError, match="unknown key"):
        _load(tmp_path, data)

    del data["candidates"][0]["inference"]["reasoning_effort"]
    data["candidates"][0]["inference"]["provider_extensions"] = {"nvidia": {"reasoning_effort": "high"}}
    with pytest.raises(EvalConfigSchemaError, match="versioned"):
        _load(tmp_path, data)

    data["candidates"][0]["inference"]["provider_extensions"] = {"nvidia.v1": {"reasoning_effort": "high"}}
    config = _load(tmp_path, data)
    assert config.candidates[0].inference.provider_extensions["nvidia.v1"]["reasoning_effort"] == "high"


def test_the_hash_is_the_same_for_two_loads_of_the_same_config(
    valid_config: tuple[Path, dict[str, Any]],
) -> None:
    first = load_eval_config(valid_config[0])
    second = load_eval_config(valid_config[0])

    assert first.eval_config_hash == second.eval_config_hash
    assert first.semantic_payload() == second.semantic_payload()


def test_moving_the_project_does_not_change_what_the_config_evaluates(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["source_run_manifest"] = "../published/expt/run_manifest.json"
    data["scoring"]["contract"] = "../contracts/eval_spec.md"
    data["outputs"]["output_dir"] = "outputs"
    original = load_eval_config(_write(tmp_path / "eval", data))

    moved_root = tmp_path / "moved"
    shutil.copytree(tmp_path, moved_root, ignore=shutil.ignore_patterns("moved"))
    moved = load_eval_config(moved_root / "eval" / "eval_config.yaml")

    assert moved.outputs.output_dir != original.outputs.output_dir
    assert moved.eval_config_hash == original.eval_config_hash


def test_the_output_directory_is_not_part_of_what_was_evaluated(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    baseline = _load(tmp_path, data)

    data["outputs"]["output_dir"] = str(tmp_path / "somewhere_else")
    assert _load(tmp_path, data).eval_config_hash == baseline.eval_config_hash


def test_candidate_order_is_presentation_not_meaning(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"].append(_second_candidate())
    forward = _load(tmp_path, data)

    data["candidates"] = list(reversed(data["candidates"]))
    reverse = _load(tmp_path, data)

    assert reverse.candidate_aliases == ("candidate_b", "candidate_a")
    assert reverse.eval_config_hash == forward.eval_config_hash


def _mutate_revision(data: dict[str, Any]) -> None:
    data["candidates"][0]["model_identity"]["revision"] = OTHER_REVISION


def _mutate_inference(data: dict[str, Any]) -> None:
    data["candidates"][0]["inference"]["temperature"] = 0.7


def _mutate_route(data: dict[str, Any]) -> None:
    data["candidates"][0]["api"]["base_url"] = "https://other.example.com/v1"


def _mutate_limits(data: dict[str, Any]) -> None:
    data["limits"]["max_turns"] = 6


def _mutate_modes(data: dict[str, Any]) -> None:
    data["eval"]["mode"] = ["trace", "executable"]


def _add_candidate(data: dict[str, Any]) -> None:
    data["candidates"].append(_second_candidate())


@pytest.mark.parametrize(
    "mutate",
    [_mutate_revision, _mutate_inference, _mutate_route, _mutate_limits, _mutate_modes, _add_candidate],
    ids=["revision", "inference", "route", "limits", "modes", "candidate_added"],
)
def test_anything_that_changes_the_measurement_changes_the_hash(tmp_path: Path, mutate: Any) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    baseline = _load(tmp_path, data).eval_config_hash

    mutate(data)

    assert _load(tmp_path, data).eval_config_hash != baseline


def test_editing_the_scoring_contract_changes_the_hash(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    contract = _scoring_contract(tmp_path)
    data = _config_data(run_dir, contract, tmp_path / "eval_out")
    baseline = _load(tmp_path, data).eval_config_hash

    contract.write_text("argument match: canonical only\n", encoding="utf-8")

    assert _load(tmp_path, data).eval_config_hash != baseline


def test_editing_the_oracle_execution_resource_changes_the_hash(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    baseline = _load(tmp_path, data).eval_config_hash
    backend = Path(data["source_oracle"]["resource"])

    backend.write_text("def reset():\n    return {'changed': True}\n", encoding="utf-8")

    assert _load(tmp_path, data).eval_config_hash != baseline


def test_evaluating_a_different_source_run_changes_the_hash(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    contract = _scoring_contract(tmp_path)
    data = _config_data(run_dir, contract, tmp_path / "eval_out")
    baseline = _load(tmp_path, data).eval_config_hash

    other = _published_run(tmp_path, name="second", run_id="second-run")
    data["source_run_manifest"] = str(other / "run_manifest.json")

    assert _load(tmp_path, data).eval_config_hash != baseline


def test_the_resolved_document_carries_the_hash_and_no_secret(
    valid_config: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    config = load_eval_config(valid_config[0])

    document = resolved_eval_config_document(config)
    target = config.outputs.output_dir / "audit" / "resolved_eval_config.json"
    content_hash = write_resolved_eval_config(config, target)

    assert document["eval_config_hash"] == config.eval_config_hash
    assert document["semantic_payload"]["candidates"][0]["api"]["api_key_env"] == "NVIDIA_API_KEY"
    text = target.read_text(encoding="utf-8")
    assert content_hash == _hash(text.encode("utf-8"))
    assert '"api_key"' not in json.dumps(document["semantic_payload"])
    # Paths are auditable but never hashed, so the payload cannot carry one.
    assert str(config.outputs.output_dir) not in json.dumps(document["semantic_payload"])
    assert document["resolved_paths"]["output_dir"] == str(config.outputs.output_dir)
    assert write_resolved_eval_config(config, target) == content_hash


def test_the_resolved_config_writer_cannot_overwrite_the_source_manifest(
    valid_config: tuple[Path, dict[str, Any]],
) -> None:
    config = load_eval_config(valid_config[0])
    source_manifest = config.source.run_manifest.path
    original = source_manifest.read_bytes()

    with pytest.raises(EvalConfigPathError, match="outside outputs.output_dir"):
        write_resolved_eval_config(config, source_manifest)

    assert source_manifest.read_bytes() == original


def test_the_resolved_config_writer_resolves_relative_paths_below_eval_output(
    valid_config: tuple[Path, dict[str, Any]],
) -> None:
    config = load_eval_config(valid_config[0])

    write_resolved_eval_config(config, "audit/resolved_eval_config.json")

    assert (config.outputs.output_dir / "audit" / "resolved_eval_config.json").is_file()


def test_the_shipped_template_names_the_shipped_contract_and_refuses_to_run() -> None:
    data = yaml.safe_load(SHIPPED_EVAL_CONFIG.read_text(encoding="utf-8"))

    assert data["schema_version"] == EVAL_CONFIG_SCHEMA_VERSION
    assert data["config_status"] == "template"
    contract = (SHIPPED_EVAL_CONFIG.parent / data["scoring"]["contract"]).resolve()
    assert contract == SHIPPED_SCORING_CONTRACT.resolve()
    assert contract.is_file()

    with pytest.raises(EvalConfigSchemaError, match="template is not runnable"):
        load_eval_config(SHIPPED_EVAL_CONFIG)


def test_the_error_summary_names_the_code_and_field_without_the_value(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    data["candidates"][0]["api"]["token"] = "sk-EXAMPLE_ONLY_not_a_real_key"

    with pytest.raises(SecretInConfigError) as failure:
        _load(tmp_path, data)

    summary = describe_eval_config_error(failure.value)
    assert summary.startswith("[secret_in_eval_config] candidates[0].api.token")
    assert "sk-EXAMPLE_ONLY" not in summary
    assert failure.value.as_report()["value"] == "<redacted>"


def _generation_config(tmp_path: Path, **overrides: Any) -> Path:
    data = yaml.safe_load((BFCL_CONFIG_DIR / "tiny.yaml").read_text(encoding="utf-8"))
    data["output_dir"] = str(tmp_path / "generated")
    data.update(overrides)
    path = tmp_path / "generation.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_generation_config_may_not_carry_two_eval_inputs(tmp_path: Path) -> None:
    path = _generation_config(
        tmp_path,
        eval_config_path="eval_config.yaml",
        eval={"schema_version": "1.1"},
    )

    with pytest.raises(ValueError, match="cannot both be set"):
        BfclConfig.from_yaml(path)


def test_an_inline_eval_block_normalizes_to_the_same_config_as_a_file(tmp_path: Path) -> None:
    run_dir = _published_run(tmp_path)
    data = _config_data(run_dir, _scoring_contract(tmp_path), tmp_path / "eval_out")
    from_file = load_eval_config(_write(tmp_path / "eval", data))

    generation = BfclConfig.from_yaml(_generation_config(tmp_path, eval=data))
    inline = load_eval_config_for_generation(generation)

    assert inline is not None
    assert inline.eval_config_hash == from_file.eval_config_hash
    assert eval_config_reference(generation) == ("inline", data)


def test_a_generation_config_without_an_eval_reference_resolves_to_nothing(tmp_path: Path) -> None:
    generation = BfclConfig.from_yaml(_generation_config(tmp_path))

    assert eval_config_reference(generation) is None
    assert load_eval_config_for_generation(generation) is None


def test_eval_inputs_stay_out_of_generation_lineage(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
        _generation_config as generation_payload,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
        _resolved_config as resolved_payload,
    )

    plain = BfclConfig.from_yaml(_generation_config(tmp_path))
    with_eval = BfclConfig.from_yaml(_generation_config(tmp_path, eval_config_path="configs/eval_config.yaml"))

    assert generation_payload(plain) == generation_payload(with_eval)
    assert resolved_payload(plain) == resolved_payload(with_eval)


def test_independent_config_faults_are_all_reported_in_one_refusal(
    valid_config: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """A preflight makes no candidate request, so stopping at the first fault only costs runs."""
    _, data = valid_config
    data["eval"]["mode"] = "trace"
    data["limits"]["max_turns"] = 0
    data["contamination"]["comparison_set"] = "whatever_survives"

    with pytest.raises(EvalConfigSchemaError) as refusal:
        _load(tmp_path, data, "many_faults.yaml")

    message = str(refusal.value)
    assert "eval.mode" in message
    assert "limits" in message
    assert "contamination" in message
    reported = refusal.value.as_report()
    assert [other["field"] for other in reported["other_violations"]] == [
        "limits.max_turns",
        "contamination.comparison_set",
    ]


def test_every_bad_candidate_is_reported_not_only_the_first(
    valid_config: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    _, data = valid_config
    data["candidates"].append(_second_candidate())
    data["candidates"][0]["model_identity"]["revision"] = "main"
    data["candidates"][1]["model_identity"]["revision"] = "main"

    with pytest.raises(CandidateIdentityError) as refusal:
        _load(tmp_path, data, "two_bad_candidates.yaml")

    fields = [refusal.value.field, *(other["field"] for other in refusal.value.as_report()["other_violations"])]
    assert [field.split(".")[0] for field in fields] == ["candidates[0]", "candidates[1]"]


def test_a_config_with_one_fault_reports_exactly_what_it_always_did(
    valid_config: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    _, data = valid_config
    data["limits"]["max_turns"] = 0

    with pytest.raises(EvalConfigSchemaError) as refusal:
        _load(tmp_path, data, "one_fault.yaml")

    assert refusal.value.other_violations == ()
    assert "also:" not in str(refusal.value)
    assert "other_violations" not in refusal.value.as_report()


def test_the_one_line_summary_admits_the_violations_it_is_not_showing(
    valid_config: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    _, data = valid_config
    data["limits"]["max_turns"] = 0
    data["contamination"]["comparison_set"] = "whatever_survives"

    with pytest.raises(EvalConfigSchemaError) as refusal:
        _load(tmp_path, data, "summary.yaml")

    summary = describe_eval_config_error(refusal.value)
    assert summary.count("\n") == 0
    assert "and 1 more: contamination.comparison_set" in summary
    assert "whatever_survives" not in summary
