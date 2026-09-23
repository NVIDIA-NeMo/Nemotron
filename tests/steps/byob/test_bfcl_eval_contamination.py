"""Candidate identity and the contamination gate.

Every test names a situation an operator can actually be in, and asserts the one
thing that must be true about it. The two failure directions are not symmetric:
clearing a candidate that wrote the rows invalidates the benchmark, while
refusing a candidate that did not costs an operator a pinned identity. The matrix
below covers both, and pays particular attention to the middle case — two
identities that cannot be compared — which must never be silently resolved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    CONTAMINATION_FAILURE_FILE,
    CONTAMINATION_REPORT_FILE,
    EVAL_CONFIG_SCHEMA_VERSION,
    SOURCE_VERIFICATION_CONTRACT_VERSION,
    BfclEvalConfig,
    CandidateContaminationError,
    ContaminationPlanDriftError,
    EligibleEvalPlan,
    EmptyEvaluationTaskSetError,
    ModelExposureError,
    ModelIdentityClaim,
    SourceChangedDuringEvalError,
    TranslationLineageError,
    UnresolvedContaminationError,
    VerifiedEvalSource,
    assert_plan_unchanged,
    compare_model_identity,
    describe_contamination_error,
    evaluate_contamination,
    load_eval_config,
    verify_eval_source,
    write_contamination_failure,
    write_contamination_report,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    PUBLICATION_BENCHMARK_TABLE,
    PUBLICATION_CONTRACT_VERSION,
    RAW_BENCHMARK_TABLE,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
    encode_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.translation import (
    translate_bfcl,
)

ORACLE_CLOCK = "2026-01-01T00:00:00+00:00"
IMMUTABLE_REVISION = "a" * 40
OTHER_REVISION = "b" * 40
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Balance of an account.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    }
]

# The route both the paraphraser and the "same model" candidate are served on.
SHARED_ROUTE = "org/surface-model"
SHARED_PROVIDER = "nvidia"
PARAPHRASE_LABEL = "nvidia/surface-model-v1"


def _hash(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_hash(path: Path) -> str:
    return _hash(path.read_bytes())


def _row(task_id: str, *, paraphrased_by: str | None = None, profiled: bool = False) -> dict[str, Any]:
    """One published row, optionally attributed to a surface model."""
    arguments = {"account_id": "1"}
    return {
        "task_id": task_id,
        "template_id": "tpl",
        "variant_index": 0,
        "messages": [
            {"role": "user", "content": "Balance of 1?", "tool_calls": None, "tool_call_id": None},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_balance", "arguments": canonical_json(arguments)},
                    }
                ],
                "tool_call_id": None,
            },
            {
                "role": "tool",
                "content": canonical_json({"balance": 10}),
                "tool_calls": None,
                "tool_call_id": "call_0",
            },
            {"role": "assistant", "content": "It is 10.", "tool_calls": None, "tool_call_id": None},
        ],
        "tools": canonical_json(TOOLS),
        "expected_tool_calls": [
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": encode_arguments(arguments),
            }
        ],
        "success_assertions": ["assert_balance_reported"],
        "fixture_refs": ['["accounts","1"]'],
        "intent": "check_balance",
        "category": "accounts",
        "difficulty": "easy",
        "required_tools": ["get_balance"],
        "required_tools_fingerprint": canonical_json(["get_balance"]),
        "tools_present": ["get_balance"],
        "turn_policy": "single_turn",
        "is_multi_turn": False,
        "num_tool_calls": 1,
        "call_order": "strict",
        "call_order_prefix": None,
        "system_prompt_id": "sp-1",
        "tier": "gold",
        "gold_eligible": True,
        "validated_by": ["schema", "replay", "assertions"],
        "pack_id": "test_pack",
        "pack_version": "1.0.0",
        "seed": 7,
        "paraphrase_model": "surface" if paraphrased_by else None,
        "paraphrase_model_canonical": paraphrased_by,
        "held_out_hit": None,
        "src": "test_pack:tpl",
        "metadata": canonical_json(
            {
                "language": "en",
                "expt_name": "expt",
                "base_task_id": None,
                "surface_source": "model" if paraphrased_by else "template",
                "profile_hash": _hash(b"profile") if profiled else None,
            }
        ),
    }


def _disabled_roles() -> dict[str, Any]:
    return {
        role: {
            "alias": None,
            "provider": None,
            "model_identity": None,
            "canonical_id": None,
            "config_hash": None,
            "enabled": False,
        }
        for role in ("profile", "paraphrase", "surface_judge")
    }


def _role(
    canonical_id: str,
    *,
    provider: str = SHARED_PROVIDER,
    model: str = SHARED_ROUTE,
    source: str | None = None,
    revision: str | None = None,
    weights_digest: str | None = None,
) -> dict[str, Any]:
    """One enabled lineage role, shaped exactly as Stage 12 records it."""
    return {
        "alias": "surface",
        "provider": provider,
        "model_identity": {
            "source": source,
            "model": model,
            "revision": revision,
            "weights_digest": weights_digest,
        },
        # Stage 12 lowercases the canonical id; the rows keep the pack's casing.
        "canonical_id": canonical_id.lower(),
        "config_hash": _hash(b"role-config"),
        "enabled": True,
    }


def _roles(**enabled: dict[str, Any]) -> dict[str, Any]:
    return {**_disabled_roles(), **enabled}


@dataclass
class Publication:
    run_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def benchmark_path(self) -> Path:
        return self.run_dir / PUBLICATION_BENCHMARK_TABLE

    @property
    def raw_path(self) -> Path:
        return self.run_dir / RAW_BENCHMARK_TABLE

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=benchmark_schema()), path)


def _publish(
    tmp_path: Path,
    *,
    name: str = "published",
    rows: list[dict[str, Any]] | None = None,
    models: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Publication:
    """A trace-only publication: no oracle pack, because none is evaluated here."""
    run_dir = tmp_path / name / "expt"
    run_dir.mkdir(parents=True)
    publication = Publication(run_dir=run_dir)
    published = rows if rows is not None else [_row("t__a"), _row("t__b")]
    _write_parquet(publication.benchmark_path, published)
    _write_parquet(publication.raw_path, published)
    manifest: dict[str, Any] = {
        "schema_version": "1.1",
        "run_id": "expt-20260819T090000000000Z-abc-1",
        "created_at": "2026-08-19T09:00:00+00:00",
        "oracle_clock": ORACLE_CLOCK,
        "generation_config_hash": _hash(b"generation-config"),
        "resolved_config_hash": _hash(b"resolved-config"),
        "lineage_policy": "strict_separation",
        "tier": "gold",
        "gold_eligible": True,
        "pack": {"pack_id": "test_pack", "version": "1.0.0", "content_hash": _hash(b"pack")},
        "oracle": {"kind": "python", "endpoint_metadata": None},
        "models": models if models is not None else _disabled_roles(),
        "held_out": {"contract_version": "1.0", "source": None, "evaluated": False},
        "publication": {
            "schema_version": PUBLICATION_CONTRACT_VERSION,
            "raw": {
                "file": RAW_BENCHMARK_TABLE,
                "rows": len(published),
                "content_hash": _file_hash(publication.raw_path),
                "contains": "schema_valid_and_replay_valid_rows",
            },
            "published": {
                "file": PUBLICATION_BENCHMARK_TABLE,
                "rows": len(published),
                "content_hash": _file_hash(publication.benchmark_path),
                "surface_gate": "surface_quality",
                "dedup_balancing_applied": False,
                "held_out_evaluated": False,
                "ordering": "raw_order",
            },
            "restated_fields": [],
            "verified": True,
        },
        "artifacts": {
            "benchmark_parquet": {"content_hash": _file_hash(publication.benchmark_path)},
            "benchmark_raw_parquet": {"content_hash": _file_hash(publication.raw_path)},
        },
    }
    manifest.update(extra or {})
    publication.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return publication


def _translate(
    publication: Publication,
    tmp_path: Path,
    *,
    model: dict[str, Any] | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> Path:
    """Write legacy-invalid evidence or run the registered translation adapter."""
    directory = tmp_path / "translated"
    if model is None:
        directory.mkdir(parents=True, exist_ok=True)
        table = directory / "benchmark_vi.parquet"
        table.write_bytes(publication.benchmark_path.read_bytes())
        task_ids = pq.read_table(table, columns=["task_id"]).column("task_id").to_pylist()
        document = {
            "schema_version": SOURCE_VERIFICATION_CONTRACT_VERSION,
            "source_run_id": publication.manifest()["run_id"],
            "language": "vi",
            "benchmark": {
                "file": table.name,
                "rows": len(task_ids),
                "content_hash": _file_hash(table),
            },
            "task_ids_hash": _hash(canonical_json(task_ids).encode("utf-8")),
        }
        manifest = directory / "translation_manifest.json"
        manifest.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    assert monkeypatch is not None
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    class StubTranslationPipeline:
        def __init__(self, _config: Any):
            pass

        def translate(self, dataframe: Any) -> Any:
            output = dataframe.copy()
            target = str(dataframe["target_language_code"].iloc[0])
            prefix = "Bản dịch: " if target == "vi" else "Backtranslation: "
            output["translation"] = output["text"].map(lambda text: prefix + text)
            return output

    def quality(dataframe: Any, _config: Any, **_kwargs: Any) -> Any:
        output = dataframe.copy()
        output["score_chrf"] = 100.0
        output["score_chrf_passed"] = True
        output["is_quality_metric_passed"] = True
        return output

    monkeypatch.setattr(translation, "TranslationPipeline", StubTranslationPipeline)
    monkeypatch.setattr(translation, "evaluate_text_quality_metrics", quality)
    config = {
        "family": "bfcl",
        "stage": "translate",
        "config_status": "resolved",
        "expt_name": "translated",
        "source_run_manifest": str(publication.manifest_path),
        "output_dir": str(tmp_path / "localized"),
        "source_language": "en",
        "target_language": "vi",
        "translate_tool_descriptions": False,
        "remove_low_quality": False,
        "translation_model_config": {
            "backend_type": "llm",
            "params": {
                "provider": model.get("provider", "test"),
                "model": model.get("model", "translator"),
                "canonical_id": model.get("canonical_id", model.get("model", "translator")),
                "source": model.get("source", "huggingface"),
                "revision": model.get("revision", IMMUTABLE_REVISION),
                "weights_digest": model.get("weights_digest"),
            },
        },
        "backtranslation_quality_metrics": [{"type": "chrf", "threshold": 0}],
    }
    config_path = tmp_path / "translation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return translate_bfcl(config_path).parent / "translation_manifest.json"


def _candidate(
    alias: str,
    *,
    provider: str = SHARED_PROVIDER,
    served_model: str = "candidate-route",
    source: str = "huggingface",
    model: str = "org/candidate",
    revision: str | None = IMMUTABLE_REVISION,
    weights_digest: str | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias,
        "model": served_model,
        "provider": provider,
        "provider_api_version": "v1",
        "api": {"base_url": "https://integrate.example.com/v1", "api_key_env": "NVIDIA_API_KEY"},
        "model_identity": {
            "source": source,
            "model": model,
            "revision": revision,
            "weights_digest": weights_digest,
        },
        "inference": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "seed": 42,
            "tool_choice": "auto",
        },
    }


def _config_data(
    publication: Publication,
    output_dir: Path,
    *,
    candidates: list[dict[str, Any]] | None = None,
    enforce: bool = True,
    on_violation: str = "fail_run",
    comparison_set: str = "common_intersection",
    publication_requested: bool = True,
    translation_manifest: Path | None = None,
) -> dict[str, Any]:
    contract = output_dir.parent / "contracts" / "eval_spec.md"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text("argument match: schema then canonical\n", encoding="utf-8")
    return {
        "schema_version": EVAL_CONFIG_SCHEMA_VERSION,
        "config_status": "resolved",
        "source_run_manifest": str(publication.manifest_path),
        "source_oracle": None,
        "translation_manifest": str(translation_manifest) if translation_manifest else None,
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
            "tool_timeout_s": 30.0,
            "candidate_timeout_s": 60.0,
            "episode_timeout_s": 120.0,
            "max_parallel_tasks": 1,
            "max_retries": 2,
        },
        "candidates": candidates or [_candidate("candidate_a")],
        "contamination": {
            "enforce": enforce,
            "on_violation": on_violation,
            "comparison_set": comparison_set,
        },
        "publication": {"requested": publication_requested, "require_same_task_ids": True},
        "outputs": {
            "output_dir": str(output_dir),
            "write_task_results": True,
            "write_eval_manifest": True,
            "cache_candidate_responses": True,
            "cache_tool_results": True,
        },
    }


def _load(tmp_path: Path, data: dict[str, Any], *, name: str = "eval") -> BfclEvalConfig:
    config_dir = tmp_path / name
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "eval_config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_eval_config(path)


def _gate(tmp_path: Path, **options: Any) -> tuple[VerifiedEvalSource, BfclEvalConfig, EligibleEvalPlan]:
    """Publish, verify, and gate in one step: the normal path for these tests."""
    publish_options = {key: options.pop(key) for key in ("name", "rows", "models", "extra") if key in options}
    publication = _publish(tmp_path, **publish_options)
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out", **options))
    source = verify_eval_source(config)
    return source, config, evaluate_contamination(config, source)


# --- the exposure inventory --------------------------------------------------


def test_a_benchmark_no_model_touched_records_no_exposure(tmp_path: Path) -> None:
    source, _, plan = _gate(tmp_path)

    assert source.exposures == ()
    assert plan.exposures == ()
    assert plan.common.task_ids == source.task_ids
    assert plan.publication_allowed is True
    assert plan.non_publication_reasons == ()


def test_a_paraphraser_is_scoped_to_the_rows_it_wrote(tmp_path: Path) -> None:
    source, _, plan = _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
        models=_roles(paraphrase=_role(PARAPHRASE_LABEL)),
    )

    assert len(source.exposures) == 1
    exposure = source.exposures[0]
    assert exposure.role == "paraphrase"
    assert exposure.scope == "paraphrased_rows"
    assert exposure.task_ids == ("t__a",)
    assert exposure.identity is not None
    assert exposure.identity.served_model == SHARED_ROUTE
    # The candidate is a different model, so the exposure costs it nothing.
    assert plan.common.task_ids == ("t__a", "t__b")


def test_a_profile_is_scoped_to_the_rows_it_shaped(tmp_path: Path) -> None:
    source, _, _ = _gate(
        tmp_path,
        rows=[_row("t__a", profiled=True), _row("t__b")],
        models=_roles(profile=_role("nvidia/profile-model")),
        extra={"profile_influenced_surface": True},
    )

    assert [(exposure.role, exposure.task_ids) for exposure in source.exposures] == [("profile", ("t__a",))]


def test_a_judge_reads_the_whole_published_surface(tmp_path: Path) -> None:
    source, _, _ = _gate(tmp_path, models=_roles(surface_judge=_role("nvidia/judge-model")))

    assert [(exposure.role, exposure.scope, exposure.task_ids) for exposure in source.exposures] == [
        ("surface_judge", "all_published_rows", ("t__a", "t__b"))
    ]


def test_a_judge_that_never_scored_a_surface_is_not_an_exposure(tmp_path: Path) -> None:
    source, _, _ = _gate(
        tmp_path,
        models=_roles(surface_judge=_role("nvidia/judge-model")),
        extra={"surface_quality_validation": {"enabled": False}},
    )

    assert source.exposures == ()


def test_a_profile_that_shaped_nothing_is_not_an_exposure(tmp_path: Path) -> None:
    source, _, _ = _gate(
        tmp_path,
        models=_roles(profile=_role("nvidia/profile-model")),
        extra={"profile_influenced_surface": False},
    )

    assert source.exposures == ()


def test_a_manifest_that_disagrees_with_its_rows_about_the_profile_is_refused(tmp_path: Path) -> None:
    publication = _publish(
        tmp_path,
        models=_roles(profile=_role("nvidia/profile-model")),
        extra={"profile_influenced_surface": True},
    )
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(ModelExposureError, match="profile_influenced_surface"):
        verify_eval_source(config)


def test_a_paraphrased_row_no_role_claims_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path, rows=[_row("t__a", paraphrased_by="nvidia/ghost-writer"), _row("t__b")])
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(ModelExposureError, match="ghost-writer"):
        verify_eval_source(config)


def test_a_profile_shaped_row_no_role_claims_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path, rows=[_row("t__a", profiled=True), _row("t__b")])
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(ModelExposureError, match="profile_hash"):
        verify_eval_source(config)


def test_an_enabled_role_that_names_no_model_is_refused(tmp_path: Path) -> None:
    anonymous = {
        "alias": None,
        "provider": None,
        "model_identity": None,
        "canonical_id": None,
        "config_hash": None,
        "enabled": True,
    }
    publication = _publish(tmp_path, models=_roles(paraphrase=anonymous))
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(ModelExposureError, match="names no model"):
        verify_eval_source(config)


def test_a_role_this_build_does_not_know_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path, models={**_disabled_roles(), "rewriter": _role("nvidia/rewriter")})
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))

    with pytest.raises(ModelExposureError, match="roles"):
        verify_eval_source(config)


# --- identity comparison ----------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "verdict"),
    [
        (
            ModelIdentityClaim(weights_digest=_hash(b"w")),
            ModelIdentityClaim(weights_digest=_hash(b"w")),
            "match",
        ),
        (
            ModelIdentityClaim(weights_digest=_hash(b"w")),
            ModelIdentityClaim(weights_digest=_hash(b"other")),
            "different",
        ),
        (
            ModelIdentityClaim(provider="nvidia", served_model="route-a"),
            ModelIdentityClaim(provider="NVIDIA", served_model="Route-A"),
            "match",
        ),
        (
            ModelIdentityClaim(weight_model="meta/Llama-3.3-70B-Instruct", revision=IMMUTABLE_REVISION),
            ModelIdentityClaim(weight_model="meta-llama/llama_3.3_70b_instruct", revision=IMMUTABLE_REVISION),
            "match",
        ),
        (
            ModelIdentityClaim(weight_model="meta/llama-3.3-70b", revision=IMMUTABLE_REVISION),
            ModelIdentityClaim(weight_model="meta/llama-3.3-70b", revision=OTHER_REVISION),
            "different",
        ),
        (
            ModelIdentityClaim(weight_model="meta/llama-3.3-70b"),
            ModelIdentityClaim(weight_model="meta/llama-3.3-70b", revision=IMMUTABLE_REVISION),
            "unknown",
        ),
        (
            ModelIdentityClaim(weight_model="nvidia/nemotron-4-340b"),
            ModelIdentityClaim(weight_model="meta/llama-3.3-70b", revision=IMMUTABLE_REVISION),
            "different",
        ),
        (
            # The same digest, recorded with and without its algorithm. Only the
            # candidate side is schema-checked as sha256:<hex>; a generation
            # manifest carries whatever the pack config wrote.
            ModelIdentityClaim(weights_digest=_hash(b"w").removeprefix("sha256:")),
            ModelIdentityClaim(weights_digest=_hash(b"w")),
            "match",
        ),
        (
            # Two digests of different things. Identical weights hash differently
            # under two algorithms, so this disagreement proves nothing.
            ModelIdentityClaim(weights_digest=f"blake3:{'c' * 64}"),
            ModelIdentityClaim(weights_digest=_hash(b"w")),
            "unknown",
        ),
        (
            # Same algorithm, different bytes: the one digest comparison that is
            # allowed to establish a separation.
            ModelIdentityClaim(weights_digest=_hash(b"w")),
            ModelIdentityClaim(weights_digest=_hash(b"other")),
            "different",
        ),
        (
            # One model name, two spellings of the registry it came from. A
            # generation manifest names the weight source in whatever words the
            # pack config used, so this is a mirror, a local copy, or the same
            # registry written twice — never a separation this may claim.
            ModelIdentityClaim(weight_source="hf", weight_model="meta/llama-3.3-70b"),
            ModelIdentityClaim(
                weight_source="huggingface",
                weight_model="meta/llama-3.3-70b",
                revision=IMMUTABLE_REVISION,
            ),
            "unknown",
        ),
        (None, ModelIdentityClaim(weight_model="meta/llama-3.3-70b"), "unknown"),
        (ModelIdentityClaim(), ModelIdentityClaim(weight_model="meta/llama-3.3-70b"), "unknown"),
    ],
)
def test_two_identities_are_compared_on_the_strongest_evidence_available(
    left: ModelIdentityClaim | None,
    right: ModelIdentityClaim | None,
    verdict: str,
) -> None:
    assert compare_model_identity(left, right) == verdict
    assert compare_model_identity(right, left) == verdict


# --- policy: fail_run -------------------------------------------------------


def test_a_candidate_that_paraphrased_the_rows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CandidateContaminationError) as error:
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
            models=_roles(paraphrase=_role(PARAPHRASE_LABEL)),
            candidates=[_candidate("candidate_a", served_model=SHARED_ROUTE)],
        )

    assert "as the paraphrase model over 1 of the 2 published row(s)" in str(error.value)
    assert describe_contamination_error(error.value).startswith("[eval_contamination_candidate_exposed]")


def test_every_contaminated_candidate_is_named_in_one_refusal(tmp_path: Path) -> None:
    """Two of the three candidates wrote rows. One run must report both.

    Refusing at the first collision would make an operator with four candidates
    discover them one re-run at a time, and each re-run costs a full source
    verification.
    """
    with pytest.raises(CandidateContaminationError) as error:
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by="model-a"), _row("t__b", profiled=True)],
            models=_roles(
                paraphrase=_role("model-a", model="route-a"),
                profile=_role("model-b", model="route-b"),
            ),
            extra={"profile_influenced_surface": True},
            candidates=[
                _candidate("candidate_a", served_model="route-a"),
                _candidate("candidate_b", served_model="route-b", model="org/candidate-b"),
                _candidate("candidate_c", served_model="route-c", model="org/candidate-c"),
            ],
        )

    message = str(error.value)
    assert "candidates[candidate_a, candidate_b]" in message
    assert "candidate_a as the paraphrase model" in message
    assert "candidate_b as the profile model" in message
    # The clean candidate is not accused of anything.
    assert "candidate_c" not in message


def test_a_candidate_that_shares_only_a_canonical_label_is_refused(tmp_path: Path) -> None:
    # The candidate's canonical id is derived; the role declares the same string
    # in lower case, which is how Stage 12 writes it.
    label = f"huggingface:org/candidate@{IMMUTABLE_REVISION}"
    with pytest.raises(CandidateContaminationError):
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by=label), _row("t__b")],
            models=_roles(paraphrase=_role(label, model="a-route-nobody-else-uses")),
        )


def test_a_digest_written_without_its_algorithm_still_names_the_same_weights(tmp_path: Path) -> None:
    """Generation recorded the digest as bare hex; the candidate pins sha256:<hex>.

    Nothing validates the generation side's spelling, so the two artifacts can
    describe one set of weights two ways. Reading that as two different models
    would clear the paraphraser of having written the row.
    """
    digest = _hash(b"shared-weights")
    with pytest.raises(CandidateContaminationError):
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
            models=_roles(paraphrase=_role(PARAPHRASE_LABEL, weights_digest=digest.removeprefix("sha256:"))),
            candidates=[_candidate("candidate_a", revision=None, weights_digest=digest)],
        )


def test_a_digest_from_another_algorithm_does_not_clear_a_candidate(tmp_path: Path) -> None:
    """The two sides hashed the same weights with different tools.

    Unequal digests only prove different weights when both digests measure the
    same thing. Here they do not, so the comparison falls back to the names and
    ends unresolved rather than clearing the candidate.
    """
    with pytest.raises(UnresolvedContaminationError):
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
            models=_roles(
                paraphrase=_role(
                    PARAPHRASE_LABEL,
                    model="org/candidate",
                    weights_digest=f"blake3:{'c' * 64}",
                )
            ),
            candidates=[_candidate("candidate_a", model="org/candidate", revision=None, weights_digest=_hash(b"w"))],
        )


def test_a_candidate_with_a_different_digest_is_cleared(tmp_path: Path) -> None:
    _, _, plan = _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
        models=_roles(
            paraphrase=_role(PARAPHRASE_LABEL, model=SHARED_ROUTE, weights_digest=_hash(b"generation-weights"))
        ),
        candidates=[
            _candidate(
                "candidate_a",
                served_model=SHARED_ROUTE,
                model=SHARED_ROUTE,
                revision=None,
                weights_digest=_hash(b"candidate-weights"),
            )
        ],
    )

    assert plan.candidates[0].collisions == ()
    assert plan.publication_allowed is True


# --- policy: exclude_row ----------------------------------------------------


def test_excluding_rows_leaves_the_rest_of_the_benchmark_scorable(tmp_path: Path) -> None:
    _, config, plan = _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
        models=_roles(paraphrase=_role(PARAPHRASE_LABEL)),
        candidates=[_candidate("candidate_a", served_model=SHARED_ROUTE)],
        on_violation="exclude_row",
        publication_requested=False,
    )

    candidate = plan.candidate("candidate_a")
    assert candidate.excluded_task_ids == ("t__a",)
    assert candidate.eligible_task_ids == ("t__b",)
    assert plan.evaluation_task_ids("candidate_a") == ("t__b",)
    assert plan.publication_allowed is False
    assert "contamination.excluded_rows:candidate_a" in plan.non_publication_reasons
    assert config.contamination.on_violation == "exclude_row"


def test_a_candidate_that_wrote_every_row_has_nothing_left_to_answer(tmp_path: Path) -> None:
    with pytest.raises(EmptyEvaluationTaskSetError, match="leaves nothing to score"):
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b", paraphrased_by=PARAPHRASE_LABEL)],
            models=_roles(paraphrase=_role(PARAPHRASE_LABEL)),
            candidates=[_candidate("candidate_a", served_model=SHARED_ROUTE)],
            on_violation="exclude_row",
            publication_requested=False,
        )


def test_two_candidates_that_wrote_different_halves_share_nothing(tmp_path: Path) -> None:
    with pytest.raises(EmptyEvaluationTaskSetError, match="answerable by every candidate"):
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by="model-a"), _row("t__b", profiled=True)],
            models=_roles(
                paraphrase=_role("model-a", model="route-a"),
                profile=_role("model-b", model="route-b"),
            ),
            extra={"profile_influenced_surface": True},
            candidates=[
                _candidate("candidate_a", served_model="route-a"),
                _candidate("candidate_b", served_model="route-b", model="org/candidate-b"),
            ],
            on_violation="exclude_row",
            publication_requested=False,
        )


# --- unresolved identity ----------------------------------------------------


def _unresolvable(tmp_path: Path, **options: Any) -> Any:
    """A role and a candidate that share a model name but pin different things."""
    return _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by="nvidia/shared-name"), _row("t__b")],
        models=_roles(paraphrase=_role("nvidia/shared-name", provider="other", model="org/candidate")),
        candidates=[_candidate("candidate_a")],
        **options,
    )


def test_an_unprovable_separation_refuses_a_publishable_run(tmp_path: Path) -> None:
    with pytest.raises(UnresolvedContaminationError, match="cannot be told apart"):
        _unresolvable(tmp_path)


def test_a_registry_spelled_two_ways_does_not_clear_the_model_it_names(tmp_path: Path) -> None:
    """The paraphraser and the candidate are one model, described by two configs.

    Generation wrote ``hf`` where evaluation wrote ``huggingface``, which is the
    kind of disagreement two independently written configs produce. It may cost
    the operator a pinned identity; it may not clear a candidate of having
    written the row it is about to be scored on.
    """
    with pytest.raises(UnresolvedContaminationError):
        _gate(
            tmp_path,
            rows=[_row("t__a", paraphrased_by="nvidia/paraphraser"), _row("t__b")],
            models=_roles(paraphrase=_role("nvidia/paraphraser", model="org/candidate", source="hf")),
            candidates=[_candidate("candidate_a", source="huggingface", model="org/candidate")],
        )


def test_an_unprovable_separation_is_recorded_in_a_debug_run(tmp_path: Path) -> None:
    _, _, plan = _unresolvable(tmp_path, publication_requested=False)

    candidate = plan.candidate("candidate_a")
    assert candidate.unresolved is True
    assert candidate.exposed is False
    # Suspicion never shrinks a task set on its own.
    assert candidate.excluded_task_ids == ()
    assert candidate.eligible_task_ids == ("t__a", "t__b")
    assert plan.publication_allowed is False
    assert "contamination.unresolved:candidate_a" in plan.non_publication_reasons


def test_enforcement_off_records_a_collision_without_acting_on_it(tmp_path: Path) -> None:
    _, _, plan = _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
        models=_roles(paraphrase=_role(PARAPHRASE_LABEL)),
        candidates=[_candidate("candidate_a", served_model=SHARED_ROUTE)],
        enforce=False,
        publication_requested=False,
    )

    candidate = plan.candidate("candidate_a")
    assert candidate.exposed is True
    assert candidate.excluded_task_ids == ()
    assert plan.publication_allowed is False
    assert "contamination.exposed:candidate_a" in plan.non_publication_reasons


# --- the comparable task set ------------------------------------------------


def test_every_candidate_answers_the_same_rows_under_common_intersection(tmp_path: Path) -> None:
    _, _, plan = _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by="model-a"), _row("t__b"), _row("t__c")],
        models=_roles(paraphrase=_role("model-a", model="route-a")),
        candidates=[
            _candidate("candidate_a", served_model="route-a"),
            _candidate("candidate_b", served_model="route-b", model="org/candidate-b"),
        ],
        on_violation="exclude_row",
        publication_requested=False,
    )

    assert plan.candidate("candidate_a").eligible_task_ids == ("t__b", "t__c")
    assert plan.candidate("candidate_b").eligible_task_ids == ("t__a", "t__b", "t__c")
    assert plan.common.task_ids == ("t__b", "t__c")
    assert plan.evaluation_task_ids("candidate_a") == ("t__b", "t__c")
    assert plan.evaluation_task_ids("candidate_b") == ("t__b", "t__c")


def test_per_candidate_scoring_keeps_each_candidates_own_rows(tmp_path: Path) -> None:
    _, _, plan = _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by="model-a"), _row("t__b"), _row("t__c")],
        models=_roles(paraphrase=_role("model-a", model="route-a")),
        candidates=[
            _candidate("candidate_a", served_model="route-a"),
            _candidate("candidate_b", served_model="route-b", model="org/candidate-b"),
        ],
        on_violation="exclude_row",
        comparison_set="per_candidate",
        publication_requested=False,
    )

    assert plan.evaluation_task_ids("candidate_a") == ("t__b", "t__c")
    assert plan.evaluation_task_ids("candidate_b") == ("t__a", "t__b", "t__c")
    assert plan.common.task_ids == ("t__b", "t__c")
    assert plan.publication_allowed is False


def test_the_task_set_keeps_publication_order(tmp_path: Path) -> None:
    _, source_config, plan = _gate(tmp_path, rows=[_row("t__b"), _row("t__a"), _row("t__c")])

    assert plan.common.task_ids == ("t__b", "t__a", "t__c")
    assert plan.candidates[0].eligible_task_ids == ("t__b", "t__a", "t__c")
    assert source_config.contamination.comparison_set == "common_intersection"


# --- the plan is an identity ------------------------------------------------


def test_the_plan_is_frozen(tmp_path: Path) -> None:
    _, _, plan = _gate(tmp_path)

    with pytest.raises(Exception):
        plan.common = None  # type: ignore[misc]


def test_the_same_decision_hashes_the_same_whatever_order_it_was_written_in(tmp_path: Path) -> None:
    first = _publish(tmp_path, name="one")
    second = _publish(tmp_path, name="two")
    candidates = [_candidate("candidate_a"), _candidate("candidate_b", model="org/candidate-b")]
    config_a = _load(
        tmp_path,
        _config_data(first, tmp_path / "out_a", candidates=candidates),
        name="eval_a",
    )
    config_b = _load(
        tmp_path,
        _config_data(second, tmp_path / "out_b", candidates=list(reversed(candidates))),
        name="eval_b",
    )

    plan_a = evaluate_contamination(config_a, verify_eval_source(config_a))
    plan_b = evaluate_contamination(config_b, verify_eval_source(config_b))

    assert plan_a.plan_identity == plan_b.plan_identity
    assert plan_a.candidate_aliases == ("candidate_a", "candidate_b")


def test_excluding_a_row_changes_the_plan_identity(tmp_path: Path) -> None:
    _, _, clean = _gate(tmp_path, rows=[_row("t__a"), _row("t__b")])
    _, _, reduced = _gate(
        tmp_path,
        name="second",
        rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
        models=_roles(paraphrase=_role(PARAPHRASE_LABEL)),
        candidates=[_candidate("candidate_a", served_model=SHARED_ROUTE)],
        on_violation="exclude_row",
        publication_requested=False,
    )

    assert clean.plan_identity != reduced.plan_identity


def test_a_gate_run_against_another_config_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    config = _load(tmp_path, _config_data(publication, tmp_path / "eval_out"))
    source = verify_eval_source(config)
    other = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", candidates=[_candidate("candidate_z")]),
        name="eval_other",
    )

    with pytest.raises(ContaminationPlanDriftError, match="not the config the source was verified against"):
        evaluate_contamination(other, source)


# --- artifacts --------------------------------------------------------------


def test_a_passing_gate_writes_a_citable_report(tmp_path: Path) -> None:
    _, config, plan = _gate(tmp_path)

    path, content_hash = write_contamination_report(config, plan)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == CONTAMINATION_REPORT_FILE
    assert content_hash.startswith("sha256:")
    assert document["status"] == "passed"
    assert document["plan_identity"] == plan.plan_identity
    assert document["common"]["task_ids"] == list(plan.common.task_ids)
    assert document["candidates"][0]["eligible_task_ids"] == list(plan.candidates[0].eligible_task_ids)
    assert "api_key" not in json.dumps(document)


def test_a_refusal_replaces_a_stale_passing_report(tmp_path: Path) -> None:
    _, config, plan = _gate(tmp_path)
    report_path, _ = write_contamination_report(config, plan)

    failure_path, _ = write_contamination_failure(
        config,
        CandidateContaminationError(
            "candidates[candidate_a]",
            "already read the rows it would be scored on",
            expected="a candidate that did not",
            recovery="evaluate another candidate",
        ),
    )

    assert failure_path.name == CONTAMINATION_FAILURE_FILE
    assert not report_path.exists()
    assert json.loads(failure_path.read_text(encoding="utf-8"))["error"]["code"] == (
        "eval_contamination_candidate_exposed"
    )


def test_a_new_pass_replaces_a_stale_refusal(tmp_path: Path) -> None:
    _, config, plan = _gate(tmp_path)
    failure_path, _ = write_contamination_failure(config, RuntimeError("boom"))

    write_contamination_report(config, plan)

    assert not failure_path.exists()


# --- handoff to the runner --------------------------------------------------


def test_an_unchanged_source_and_plan_authorize_the_run(tmp_path: Path) -> None:
    source, config, plan = _gate(tmp_path)

    assert assert_plan_unchanged(config, source, plan) is None


def test_a_benchmark_replaced_after_the_gate_stops_the_run(tmp_path: Path) -> None:
    source, config, plan = _gate(tmp_path)
    _write_parquet(source.benchmark.path, [_row("t__a")])

    with pytest.raises(SourceChangedDuringEvalError):
        assert_plan_unchanged(config, source, plan)


def test_a_plan_widened_after_the_gate_stops_the_run(tmp_path: Path) -> None:
    source, config, plan = _gate(
        tmp_path,
        rows=[_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
        models=_roles(paraphrase=_role(PARAPHRASE_LABEL)),
        candidates=[_candidate("candidate_a", served_model=SHARED_ROUTE)],
        on_violation="exclude_row",
        publication_requested=False,
    )
    widened = plan.model_copy(
        update={
            "candidates": (
                plan.candidates[0].model_copy(update={"eligible_task_ids": ("t__a", "t__b"), "excluded_task_ids": ()}),
            ),
            "common": plan.common.model_copy(update={"task_ids": ("t__a", "t__b")}),
        }
    )

    with pytest.raises(ContaminationPlanDriftError, match="no longer resolves"):
        assert_plan_unchanged(config, source, widened)


def test_a_plan_the_gate_would_never_produce_is_not_an_authorization(tmp_path: Path) -> None:
    """A plan carrying the right hashes, for a run the gate refuses outright.

    Under fail_run this config has no plan at all, so the only way to hold one is
    to have assembled it. The operator's problem is the plan in their hand, not
    the collision it does not mention, so this reports drift.
    """
    contaminated = {
        "rows": [_row("t__a", paraphrased_by=PARAPHRASE_LABEL), _row("t__b")],
        "models": _roles(paraphrase=_role(PARAPHRASE_LABEL)),
        "candidates": [_candidate("candidate_a", served_model=SHARED_ROUTE)],
    }
    debug_source, debug_config, debug_plan = _gate(
        tmp_path, name="debug", on_violation="exclude_row", publication_requested=False, **contaminated
    )
    strict_publication = _publish(tmp_path, name="strict", rows=contaminated["rows"], models=contaminated["models"])
    strict = _load(
        tmp_path,
        _config_data(strict_publication, tmp_path / "strict_out", candidates=contaminated["candidates"]),
        name="eval_strict",
    )
    strict_source = verify_eval_source(strict)
    forged = debug_plan.model_copy(
        update={
            "eval_config_hash": strict.eval_config_hash,
            "source_verification_identity": strict_source.verification_identity,
        }
    )

    with pytest.raises(ContaminationPlanDriftError, match="never a decision this config and this source"):
        assert_plan_unchanged(strict, strict_source, forged)

    # The legitimate debug plan still authorizes its own run.
    assert assert_plan_unchanged(debug_config, debug_source, debug_plan) is None


# --- translation ------------------------------------------------------------


def test_an_unnamed_translator_is_refused(tmp_path: Path) -> None:
    publication = _publish(tmp_path)
    manifest = _translate(publication, tmp_path)
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest, publication_requested=False),
    )
    with pytest.raises(TranslationLineageError, match="translation contract"):
        verify_eval_source(config)


def test_a_candidate_that_translated_the_benchmark_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _publish(tmp_path)
    manifest = _translate(
        publication,
        tmp_path,
        model={"provider": SHARED_PROVIDER, "model": "candidate-route"},
        monkeypatch=monkeypatch,
    )
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest),
    )
    source = verify_eval_source(config)

    with pytest.raises(CandidateContaminationError, match="translator model"):
        evaluate_contamination(config, source)


def test_a_named_translator_that_is_another_model_clears_the_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _publish(tmp_path)
    manifest = _translate(
        publication,
        tmp_path,
        model={"provider": "other", "model": "org/translator", "canonical_id": "other/translator"},
        monkeypatch=monkeypatch,
    )
    config = _load(
        tmp_path,
        _config_data(publication, tmp_path / "eval_out", translation_manifest=manifest),
    )
    source = verify_eval_source(config)

    plan = evaluate_contamination(config, source)

    assert plan.candidates[0].collisions == ()
    assert plan.publication_allowed is True
