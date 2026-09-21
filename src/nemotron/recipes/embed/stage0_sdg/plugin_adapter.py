"""Translate the Nemotron SDG profile into the public retrieval SDG API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer_retrieval_sdg import GenerationPreviewResult, GenerationResult, GenerationRunConfig

    from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig

RETRIEVAL_SDG_SCHEMA_VERSION = 1


def build_retrieval_first_config(cfg: SDGConfig):
    """Translate canonical recipe inputs into the generic EA producer, without corpus adapters."""
    from data_designer_retrieval_sdg.multimodal import ModelSettings, MultimodalSDGConfig
    from data_designer_retrieval_sdg.retrieval.models import SplitRatios

    if cfg.sources_file is None or not cfg.portable_export:
        raise ValueError("retrieval_first requires sources_file and portable_export")
    endpoint = cfg.nvidia_api_base_url or "https://integrate.api.nvidia.com/v1"
    return MultimodalSDGConfig(
        sources_file=cfg.sources_file.resolve(),
        contexts_file=cfg.contexts_file.resolve() if cfg.contexts_file else None,
        output_dir=cfg.output_dir.resolve() / "multimodal",
        dataset_id=cfg.corpus_id,
        generator=ModelSettings(
            model=cfg.qa_generation_model, endpoint=endpoint, credential_env=cfg.sdg_credential_env
        ),
        judge=ModelSettings(model=cfg.quality_judge_model, endpoint=endpoint, credential_env=cfg.sdg_credential_env),
        concurrency=cfg.max_parallel_requests_for_gen or 8,
        batch_size=cfg.sdg_batch_size,
        max_units_per_context=cfg.sdg_max_units_per_context,
        seed=cfg.export_seed,
        ratios=SplitRatios(
            train=cfg.export_train_ratio,
            validation=cfg.export_validation_ratio,
            evaluation=1 - cfg.export_train_ratio - cfg.export_validation_ratio,
        ),
        resume=cfg.resume != "never",
    )


def execute_retrieval_first(cfg: SDGConfig) -> Path:
    """Run public SDG, validate full portable views, and publish the recipe handoff."""
    import json
    import os

    from data_designer_retrieval_sdg.multimodal import run_multimodal_sdg

    from nemotron.recipes.embed.sdg_manifest import write_generation_manifest
    from nemotron.recipes.retrieval_vl import inspect_vl_bundle

    config = build_retrieval_first_config(cfg)
    if cfg.nvidia_api_key is not None:
        raise ValueError("retrieval_first accepts credentials through sdg_credential_env only, not inline config")
    if not os.environ.get(config.generator.credential_env):
        raise ValueError(f"Set the credential environment variable {config.generator.credential_env}")
    handoff = run_multimodal_sdg(config)
    bundle = handoff.parent
    manifest = json.loads((bundle / "run_manifest.json").read_text())
    for view, counts in manifest["counts"]["views"].items():
        if counts["accepted_queries"]:
            inspect_vl_bundle(
                bundle / "run_manifest.json",
                view="text_image" if view == "image_and_text" else view,
                require_local_artifacts=True,
                verify_checksums=True,
            )
    output = bundle / "source_records.jsonl"
    target = cfg.output_dir.resolve() / "generation_result.json"
    if target.exists():
        existing = json.loads(target.read_text())
        if (target.parent / existing["portable_bundle_manifest"]).resolve() != bundle / "run_manifest.json":
            raise ValueError("Existing Stage 0 handoff points to a different run; use a new output directory")
    write_generation_manifest(
        output_dir=cfg.output_dir, output_path=output, dataset_name=cfg.corpus_id, portable_bundle=bundle
    )
    print(f"Generic multimodal SDG completed: {target}")
    return output


def build_generation_config(cfg: SDGConfig, corpus_dir: Path) -> tuple[GenerationRunConfig, tuple[str, ...]]:
    """Build the plugin's complete typed generation config from a recipe profile."""
    from data_designer_retrieval_sdg import (
        DocumentChunkerSeedSource,
        GenerationPipelineConfig,
        GenerationRunConfig,
        RetrievalSourcesFile,
        build_model_providers,
    )

    file_extensions = (
        [extension.strip() for extension in cfg.file_extensions.split(",") if extension.strip()]
        if cfg.file_extensions
        else [".txt", ".md", ".text", ""]
    )
    seed_source = (
        RetrievalSourcesFile(path=corpus_dir)
        if cfg.sources_file is not None
        else DocumentChunkerSeedSource(
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
            buffer_size=cfg.buffer_size,
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
