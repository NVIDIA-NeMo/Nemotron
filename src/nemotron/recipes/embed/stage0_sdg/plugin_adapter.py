"""Translate the Nemotron SDG profile into the public retrieval SDG API."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer_retrieval_sdg import GenerationPreviewResult, GenerationResult, GenerationRunConfig

    from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig

RETRIEVAL_SDG_SCHEMA_VERSION = 1


def _effective_buffer_size(cfg: SDGConfig) -> int:
    if cfg.batch_size is None:
        return cfg.buffer_size

    warnings.warn(
        "batch_size is deprecated; use buffer_size. Data Designer now owns checkpointing and resume.",
        FutureWarning,
        stacklevel=2,
    )
    return cfg.batch_size


def build_generation_config(cfg: SDGConfig, corpus_dir: Path) -> tuple[GenerationRunConfig, tuple[str, ...]]:
    """Build the plugin's complete typed generation config from a recipe profile."""
    from data_designer_retrieval_sdg import (
        DocumentChunkerSeedSource,
        GenerationPipelineConfig,
        GenerationRunConfig,
        build_model_providers,
    )

    file_extensions = (
        [extension.strip() for extension in cfg.file_extensions.split(",") if extension.strip()]
        if cfg.file_extensions
        else [".txt", ".md", ".text", ""]
    )
    seed_source = DocumentChunkerSeedSource(
        path=str(corpus_dir),
        file_pattern="*",
        recursive=True,
        file_extensions=file_extensions,
        min_text_length=cfg.min_text_length,
        sentences_per_chunk=cfg.sentences_per_chunk,
        num_sections=cfg.num_sections,
        num_files=cfg.num_files,
        multi_doc=cfg.multi_doc,
        bundle_size=cfg.bundle_size,
        bundle_strategy=cfg.bundle_strategy,
        max_docs_per_bundle=cfg.max_docs_per_bundle,
        multi_doc_manifest=cfg.multi_doc_manifest,
    )
    pipeline = GenerationPipelineConfig(
        max_artifacts_per_type=cfg.max_artifacts_per_type,
        num_pairs=cfg.num_pairs,
        query_counts=cfg.query_counts,
        min_hops=cfg.min_hops,
        max_hops=cfg.max_hops,
        reasoning_counts=cfg.reasoning_counts,
        min_complexity=cfg.min_complexity,
        similarity_threshold=cfg.similarity_threshold,
        max_parallel_requests_for_gen=cfg.max_parallel_requests_for_gen,
        artifact_extraction_model=cfg.artifact_extraction_model,
        artifact_extraction_provider=cfg.artifact_extraction_provider,
        qa_generation_model=cfg.qa_generation_model,
        qa_generation_provider=cfg.qa_generation_provider,
        quality_judge_model=cfg.quality_judge_model,
        quality_judge_provider=cfg.quality_judge_provider,
        embed_model=cfg.embed_model,
        embed_provider=cfg.embed_provider,
    )

    model_providers = None
    environment_variables = ["NVIDIA_API_KEY"]
    if cfg.nvidia_api_base_url:
        model_providers, _ = build_model_providers(
            custom_provider_endpoint=cfg.nvidia_api_base_url,
            custom_provider_name="nvidia",
            custom_provider_type="openai",
            custom_provider_api_key="NVIDIA_API_KEY",
            custom_provider_fields={"endpoint", "name", "provider_type", "api_key"},
        )
        if "nvidia_api_base_url" not in cfg.model_fields_set:
            environment_variables.append("NVIDIA_API_BASE_URL")

    return (
        GenerationRunConfig(
            schema_version=RETRIEVAL_SDG_SCHEMA_VERSION,
            seed_source=seed_source,
            output_dir=cfg.output_dir.resolve(),
            artifact_path=cfg.artifact_path.resolve(),
            dataset_name=cfg.dataset_name or cfg.corpus_id,
            buffer_size=_effective_buffer_size(cfg),
            resume=cfg.resume,
            model_providers=model_providers,
            pipeline=pipeline,
            log_level=cfg.log_level,
        ),
        tuple(environment_variables),
    )


def execute_generation(
    cfg: SDGConfig,
    corpus_dir: Path,
) -> GenerationResult | GenerationPreviewResult:
    """Invoke the installed plugin after translating the recipe configuration."""
    from data_designer_retrieval_sdg import preview_generation, run_generation

    config, environment_variables = build_generation_config(cfg, corpus_dir)
    if cfg.preview:
        return preview_generation(config)
    return run_generation(config, environment_variables=environment_variables)
