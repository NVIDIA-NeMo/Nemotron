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
