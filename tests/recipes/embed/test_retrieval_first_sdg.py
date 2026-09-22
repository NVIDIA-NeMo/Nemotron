# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Recipe integration of the generic retrieval-first public producer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

pytest.importorskip("data_designer_retrieval_sdg.multimodal")

from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig, run_sdg
from nemotron.recipes.embed.stage0_sdg.plugin_adapter import build_retrieval_first_config


def config(tmp_path: Path, **updates) -> SDGConfig:
    return SDGConfig(
        sdg_workflow="retrieval_first",
        sources_file=tmp_path / "sources.jsonl",
        portable_export=True,
        output_dir=tmp_path / "output",
        contexts_file=tmp_path / "contexts.jsonl",
        qa_generation_model="configured-generator",
        quality_judge_model="configured-judge",
        nvidia_api_base_url="https://example.invalid/v1",
        sdg_credential_env="TEST_SDG_KEY",
        **updates,
    )


@pytest.mark.parametrize("field", ["sdg_options", "sdg_generator_options", "sdg_judge_options"])
def test_retrieval_first_options_are_rejected_by_legacy_workflow(field):
    with pytest.raises(ValidationError, match="require sdg_workflow='retrieval_first'"):
        SDGConfig(**{field: {"temperature": 0.5}})


def test_generic_configuration_mapping(tmp_path):
    mapped = build_retrieval_first_config(config(tmp_path, max_parallel_requests_for_gen=16, resume="always"))
    assert mapped.generator.model == "configured-generator"
    assert mapped.judge.model == "configured-judge"
    assert mapped.generator.credential_env == "TEST_SDG_KEY"
    assert mapped.sources_file == tmp_path / "sources.jsonl"
    assert mapped.contexts_file == tmp_path / "contexts.jsonl"
    assert mapped.output_dir == tmp_path / "output/multimodal"
    assert mapped.resume and mapped.concurrency == 16
    assert mapped.ratios.train == 0.8 and mapped.ratios.validation == 0


def test_all_generic_producer_controls_and_model_settings_are_forwarded(tmp_path):
    options = {
        "context_strategy": "document",
        "max_context_chars": 40000,
        "related_contexts_per_context": 2,
        "related_summary_similarity": 0.4,
        "judge_summaries": True,
        "summary_quality_threshold": 3,
        "summary_count": 50,
        "summary_near_duplicate_threshold": 0.95,
        "instructions_per_context": 1,
        "missing_response_attempts": 2,
        "require_verbatim_quotes": True,
        "relevance_threshold": 5,
        "self_sufficiency_threshold": 3,
        "group_near_duplicates": True,
        "instructions": [
            {
                "name": "comparison",
                "instruction": "Compare operating limits",
                "query_type": "comparison",
                "format": "question",
                "persona": "technician",
                "answerability": "multi-unit",
                "modality": "text_and_image",
            }
        ],
    }
    mapped = build_retrieval_first_config(
        config(
            tmp_path,
            sdg_options=options,
            sdg_generator_options={"temperature": 0.2, "max_tokens": 2048, "timeout": 90, "extra_body": {"top_k": 20}},
            sdg_judge_options={
                "temperature": 0,
                "credential_env": "JUDGE_KEY",
                "endpoint": "https://judge.invalid/v1",
            },
        )
    )
    serialized = mapped.model_dump(mode="json")
    for name, value in options.items():
        assert serialized[name] == value
    assert mapped.generator.temperature == 0.2 and mapped.generator.max_tokens == 2048
    assert mapped.generator.timeout == 90 and mapped.generator.extra_body == {"top_k": 20}
    assert mapped.judge.temperature == 0 and mapped.judge.credential_env == "JUDGE_KEY"


@pytest.mark.parametrize(
    "options",
    [{"output_dir": "elsewhere"}, {"seed": 7}, {"unknown": True}, {"summary_count": 2, "summary_fraction": 0.5}],
)
def test_producer_options_cannot_hide_errors_or_override_recipe_identity(tmp_path, options):
    with pytest.raises(ValueError):
        build_retrieval_first_config(config(tmp_path, sdg_options=options))


def test_independent_judge_credential_is_checked_before_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SDG_KEY", "placeholder")
    monkeypatch.delenv("JUDGE_KEY", raising=False)
    with pytest.raises(ValueError, match="JUDGE_KEY"):
        run_sdg(config(tmp_path, sdg_judge_options={"credential_env": "JUDGE_KEY"}))


@pytest.mark.parametrize("updates", [{"preview": True}, {"strict_visual": True}])
def test_unsupported_legacy_modes_do_not_silently_change_ea_run(tmp_path, updates):
    with pytest.raises(ValidationError):
        config(tmp_path, **updates)


def test_requires_canonical_input_and_portable_handoff():
    with pytest.raises(ValidationError, match="canonical"):
        SDGConfig(sdg_workflow="retrieval_first")


def test_ea_route_never_invokes_legacy_generation_or_downloads_default_corpus(tmp_path, monkeypatch):
    expected = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        "nemotron.recipes.embed.stage0_sdg.plugin_adapter.execute_retrieval_first", lambda cfg: expected
    )
    assert run_sdg(config(tmp_path)) == expected


def test_credentials_are_required_before_inference(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_SDG_KEY", raising=False)
    with pytest.raises(ValueError, match="credential environment"):
        run_sdg(config(tmp_path))
    with pytest.raises(ValueError, match="inline"):
        run_sdg(config(tmp_path, nvidia_api_key="placeholder"))


def test_recipe_validates_bundle_and_publishes_single_stage_handoff(tmp_path, monkeypatch):
    import data_designer_retrieval_sdg.multimodal as producer
    from data_designer_retrieval_sdg.multimodal.storage import write_json

    monkeypatch.setenv("TEST_SDG_KEY", "placeholder")
    bundle = tmp_path / "output/multimodal/bundle"
    bundle.mkdir(parents=True)
    (bundle / "source_records.jsonl").write_text("{}\n")
    write_json(bundle / "run_manifest.json", {"counts": {"views": {"image_and_text": {"accepted_queries": 1}}}})
    handoff = bundle / "generation_result.json"
    monkeypatch.setattr(producer, "run_multimodal_sdg", lambda cfg: handoff)
    inspected = []
    monkeypatch.setattr(
        "nemotron.recipes.retrieval_vl.inspect_vl_bundle", lambda *args, **kwargs: inspected.append((args, kwargs))
    )
    result = run_sdg(config(tmp_path))
    assert result == bundle / "source_records.jsonl"
    assert inspected[0][1] == {"view": "text_image", "require_local_artifacts": True, "verify_checksums": True}
    top = json.loads((tmp_path / "output/generation_result.json").read_text())
    assert top["portable_bundle_manifest"] == "multimodal/bundle/run_manifest.json"
    assert top["portable_bundle_sha256"]


def test_generic_semantic_planning_controls_are_forwarded(tmp_path):
    options = {
        "context_strategy": "sections",
        "section_size": 5,
        "combination_iterations": 20,
        "summary_embedding_model": "operator/summary-embedding",
        "summary_embedding_revision": "a" * 40,
        "summary_embedding_device": "cpu",
        "summary_count": 400,
        "persona": "A reader comparing maintenance procedures.",
    }
    mapped = build_retrieval_first_config(config(tmp_path, sdg_options=options))
    assert all(getattr(mapped, key) == value for key, value in options.items())
    assert mapped.sources_file == tmp_path / "sources.jsonl"
    assert mapped.contexts_file == tmp_path / "contexts.jsonl"


def test_summary_embedding_endpoint_is_independent_of_chat_provider(tmp_path):
    options = {
        "context_strategy": "sections",
        "combination_iterations": 20,
        "summary_embedding_model": "nvidia/nemotron-3-embed-1b",
        "summary_embedding_endpoint": "https://integrate.api.nvidia.com/v1",
        "summary_embedding_credential_env": "EMBEDDING_KEY",
        "summary_embedding_extra_body": {"input_type": "passage", "truncate": "NONE"},
    }
    mapped = build_retrieval_first_config(config(tmp_path, sdg_options=options))
    assert all(getattr(mapped, key) == value for key, value in options.items())
    assert mapped.generator.endpoint == "https://example.invalid/v1"
    assert mapped.generator.credential_env == "TEST_SDG_KEY"
