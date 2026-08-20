"""Contract tests for the released retrieval SDG package adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

pytest.importorskip("data_designer_retrieval_sdg")

from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig
from nemotron.recipes.embed.stage0_sdg.plugin_adapter import build_generation_config
from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig
from nemotron.recipes.embed.stage1_data_prep.plugin_adapter import build_conversion_config, execute_conversion


def test_generation_fields_have_explicit_ownership() -> None:
    package_fields = {
        "artifact_extraction_model",
        "artifact_extraction_provider",
        "artifact_path",
        "buffer_size",
        "bundle_size",
        "bundle_strategy",
        "corpus_id",
        "dataset_name",
        "embed_model",
        "embed_provider",
        "file_extensions",
        "log_level",
        "max_artifacts_per_type",
        "max_docs_per_bundle",
        "max_hops",
        "max_parallel_requests_for_gen",
        "min_complexity",
        "min_hops",
        "min_text_length",
        "multi_doc",
        "multi_doc_manifest",
        "num_files",
        "num_pairs",
        "num_sections",
        "nvidia_api_base_url",
        "output_dir",
        "qa_generation_model",
        "qa_generation_provider",
        "quality_judge_model",
        "quality_judge_provider",
        "query_counts",
        "reasoning_counts",
        "resume",
        "sentences_per_chunk",
        "similarity_threshold",
    }
    recipe_fields = {
        "artifact_root",
        "corpus_dir",
        "nvidia_api_key",
        "preview",
    }

    assert package_fields.isdisjoint(recipe_fields)
    assert package_fields | recipe_fields == set(SDGConfig.model_fields)


def test_conversion_fields_have_explicit_ownership() -> None:
    package_fields = {
        "conversion_seed",
        "corpus_id",
        "groups_json",
        "max_pos_docs",
        "output_dir",
        "quality_threshold",
        "sdg_input_path",
        "split_strategy",
        "test_ratio",
        "train_ratio",
        "use_group_id_in_eval",
        "val_ratio",
    }
    recipe_fields = {
        "artifact_recipe",
        "artifact_root",
        "attn_implementation",
        "base_model",
        "hard_neg_margin",
        "hard_negatives_to_mine",
        "mining_batch_size",
        "passage_max_length",
        "passage_prefix",
        "query_max_length",
        "query_prefix",
        "train_input_file",
        "trust_remote_code",
    }

    assert package_fields.isdisjoint(recipe_fields)
    assert package_fields | recipe_fields == set(DataPrepConfig.model_fields)


def test_generation_adapter_maps_complete_default_profile(tmp_path: Path) -> None:
    cfg = SDGConfig(
        corpus_dir=str(tmp_path / "corpus"),
        output_dir=tmp_path / "generated",
        artifact_path=tmp_path / "artifacts",
        nvidia_api_base_url="https://example.invalid/v1",
    )

    config, environment_variables = build_generation_config(cfg, tmp_path / "corpus")

    assert config.schema_version == 1
    assert config.dataset_name == "nv_pp_random"
    assert config.buffer_size == 200
    assert config.resume == "never"
    assert config.seed_source.recursive is True
    assert config.seed_source.file_extensions == [".txt", ".md", ".text", ""]
    assert config.pipeline.num_pairs == 7
    assert config.pipeline.query_counts == {"multi_hop": 3, "structural": 2, "contextual": 2}
    assert sum(config.pipeline.reasoning_counts.values()) == config.pipeline.num_pairs
    assert config.pipeline.artifact_extraction_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert config.pipeline.embed_model == "nvidia/nemotron-3-embed-1b"
    assert environment_variables == ("NVIDIA_API_KEY",)
    assert config.model_providers is not None
    provider = next(item for item in config.model_providers if item.name == "nvidia")
    assert provider.endpoint == "https://example.invalid/v1"
    assert provider.api_key == "NVIDIA_API_KEY"


def test_generation_adapter_records_environment_endpoint_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NVIDIA_API_BASE_URL", "https://environment.example.invalid/v1")
    cfg = SDGConfig(corpus_dir=str(tmp_path))

    _, environment_variables = build_generation_config(cfg, tmp_path)

    assert environment_variables == ("NVIDIA_API_KEY", "NVIDIA_API_BASE_URL")


def test_generation_adapter_maps_llama_models_without_a_plugin_profile(tmp_path: Path) -> None:
    llama_model = "nvidia/nemotron-3-nano-30b-a3b"
    cfg = SDGConfig(
        corpus_dir=str(tmp_path),
        artifact_extraction_model=llama_model,
        qa_generation_model=llama_model,
        quality_judge_model=llama_model,
    )

    config, _ = build_generation_config(cfg, tmp_path)

    assert config.pipeline.artifact_extraction_model == llama_model
    assert config.pipeline.qa_generation_model == llama_model
    assert config.pipeline.quality_judge_model == llama_model


def test_generation_adapter_maps_recipe_overrides(tmp_path: Path) -> None:
    manifest = tmp_path / "bundles.yaml"
    cfg = SDGConfig(
        corpus_dir=str(tmp_path / "corpus"),
        output_dir=tmp_path / "generated",
        artifact_path=tmp_path / "artifacts",
        dataset_name="custom-dataset",
        file_extensions=".rst,.txt",
        min_text_length=12,
        sentences_per_chunk=6,
        num_sections=2,
        num_files=4,
        max_artifacts_per_type=3,
        min_hops=2,
        max_hops=4,
        min_complexity=3,
        similarity_threshold=0.8,
        buffer_size=16,
        resume="if_possible",
        multi_doc=True,
        bundle_size=3,
        bundle_strategy="interleaved",
        max_docs_per_bundle=4,
        multi_doc_manifest=str(manifest),
        max_parallel_requests_for_gen=5,
        log_level="DEBUG",
    )

    config, _ = build_generation_config(cfg, tmp_path / "corpus")

    assert config.output_dir == tmp_path / "generated"
    assert config.artifact_path == tmp_path / "artifacts"
    assert config.dataset_name == "custom-dataset"
    assert config.buffer_size == 16
    assert config.resume == "if_possible"
    assert config.log_level == "DEBUG"
    assert config.seed_source.file_extensions == [".rst", ".txt"]
    assert config.seed_source.min_text_length == 12
    assert config.seed_source.sentences_per_chunk == 6
    assert config.seed_source.num_sections == 2
    assert config.seed_source.num_files == 4
    assert config.seed_source.multi_doc is True
    assert config.seed_source.bundle_size == 3
    assert config.seed_source.bundle_strategy == "interleaved"
    assert config.seed_source.max_docs_per_bundle == 4
    assert config.seed_source.multi_doc_manifest == str(manifest)
    assert config.pipeline.max_artifacts_per_type == 3
    assert config.pipeline.min_hops == 2
    assert config.pipeline.max_hops == 4
    assert config.pipeline.min_complexity == 3
    assert config.pipeline.similarity_threshold == 0.8
    assert config.pipeline.max_parallel_requests_for_gen == 5


def test_generation_adapter_requires_distributions_to_match_num_pairs(tmp_path: Path) -> None:
    cfg = SDGConfig(corpus_dir=str(tmp_path), num_pairs=8)

    with pytest.raises(ValidationError, match="must sum to num_pairs"):
        build_generation_config(cfg, tmp_path)


def test_conversion_adapter_maps_complete_recipe_config(tmp_path: Path) -> None:
    input_path = tmp_path / "stage0_sdg" / "nv_pp_random.jsonl"
    cfg = DataPrepConfig(
        sdg_input_path=input_path,
        output_dir=tmp_path / "stage1_data_prep",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        conversion_seed=13,
        max_pos_docs=4,
        use_group_id_in_eval=True,
        split_strategy="dedupped",
        groups_json=[tmp_path / "groups.json"],
    )

    config = build_conversion_config(cfg)

    assert config.schema_version == 1
    assert config.input_path == input_path
    assert config.output_dir == tmp_path / "stage1_data_prep"
    assert config.corpus_id == "nv_pp_random"
    assert config.train_ratio == 0.7
    assert config.val_ratio == 0.1
    assert config.seed == 13
    assert config.quality_threshold == 7.0
    assert config.max_pos_docs == 4
    assert config.use_group_id_in_eval is True
    assert config.split_strategy == "dedupped"
    assert config.groups_json == [tmp_path / "groups.json"]


def test_conversion_adapter_processes_plugin_jsonl_output(tmp_path: Path) -> None:
    input_path = tmp_path / "nv_pp_random.jsonl"
    records = [
        {
            "file_name": [f"nested/doc-{index}.txt"],
            "source_id": f"nested/doc-{index}.txt",
            "chunks": [{"chunk_id": 1, "text": f"document {index}"}],
            "deduplicated_qa_pairs": [{"question": f"Question {index}?", "segment_ids": [1]}],
            "qa_evaluations": {"evaluations": [{"overall": {"score": 9.0}}]},
        }
        for index in range(10)
    ]
    input_path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")
    cfg = DataPrepConfig(
        sdg_input_path=input_path,
        output_dir=tmp_path / "converted",
    )

    result = execute_conversion(cfg)

    assert result.train_file == tmp_path / "converted" / "train.json"
    assert result.train_file.exists()
    assert result.evaluation_dir == tmp_path / "converted" / "eval_beir"
    assert result.evaluation_dir.exists()
    assert result.training_examples == 8
    assert result.evaluation_queries == 2
    assert result.resolved_config_path is not None
    assert result.resolved_config_path.exists()
