"""MCQ benchmark-family orchestration.

The generic CLI dispatcher lives in `nemotron.steps.byob.scripts.runtime`.
This module owns the MCQ-specific stage order, cache paths, and final schema.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class McqGenerationStage(Enum):
    """MCQ generation stages in order."""

    GENERATION = 0
    JUDGEMENT = 1
    SEMANTIC_DEDUPLICATION = 2
    DISTRACTOR_EXPANSION = 3
    COVERAGE_CHECK = 4
    DISTRACTOR_VALIDITY_CHECK = 5
    SEMANTIC_OUTLIER_DETECTION = 6
    HALLUCINATION_EASINESS_DETECTION = 7
    FINAL_OUTPUT = 8


class McqTranslationStage(Enum):
    """MCQ translation stages in order."""

    TRANSLATION = 0
    BACKTRANSLATION = 1
    QUALITY_METRICS = 2
    FINAL_OUTPUT = 3


def _should_run(skip_until: str | None, stage: Enum) -> bool:
    """Return whether a stage should run after applying a skip point."""
    if skip_until is None:
        return True
    return stage.value >= type(stage)[skip_until].value


def _is_enabled(config_section: dict, *, default: bool = True) -> bool:
    """Return the optional stage switch while preserving legacy defaults."""
    return config_section.get("enabled", default)


def prepare_mcq_data(config_path: str | os.PathLike[str]):
    """Create the MCQ seed parquet from the configured source benchmark and corpus."""
    from nemotron.steps.byob.runtime.config import ByobConfig
    from nemotron.steps.byob.runtime.dataset import make_from_config

    config = ByobConfig.from_yaml(str(config_path))
    dataset = make_from_config(config)
    return dataset.sample_and_dump()


def generate_mcq(config_path: str | os.PathLike[str], *, skip_until: str | None = None) -> Path | None:
    """Run the MCQ benchmark generation pipeline and return the final benchmark path."""
    import pandas as pd

    from nemotron.steps.byob.runtime.benchmark_families.mcq.deduplication import TextSemanticDeduplicationMCQ
    from nemotron.steps.byob.runtime.benchmark_families.mcq.semantic_outlier import TextSemanticOutlierDetectionMCQ
    from nemotron.steps.byob.runtime.benchmark_families.mcq.stages import (
        check_distractor_validity,
        expand_distractors,
        filter_questions,
        generate_questions,
        judge_questions,
    )
    from nemotron.steps.byob.runtime.benchmark_families.mcq.text_coverage import TextCoverageMCQ
    from nemotron.steps.byob.runtime.benchmark_families.mcq.utils import (
        postprocess_distractor_expansion,
        postprocess_distractor_validity,
        postprocess_filtered_questions,
        postprocess_generated_questions,
        postprocess_judged_questions,
        prepare_distractor_expansion_seed_dataset,
        prepare_distractor_validity_seed_dataset,
        prepare_filtering_seed_dataset,
        prepare_generation_seed_dataset,
        prepare_judgement_seed_dataset,
    )
    from nemotron.steps.byob.runtime.config import ByobConfig
    from nemotron.steps.byob.runtime.data_designer_utils import batched_run

    config = ByobConfig.from_yaml(str(config_path))
    output_base = Path(config.output_dir) / config.expt_name
    stage_cache = output_base / "stage_cache"
    stage_cache.mkdir(parents=True, exist_ok=True)

    paths = {
        "generation": stage_cache / "generated_questions.parquet",
        "judgement": stage_cache / "judged_questions.parquet",
        "semantic_deduplication": stage_cache / "semantic_deduplicated_questions.parquet",
        "distractor_expansion": stage_cache / "expanded_distractors.parquet",
        "coverage": stage_cache / "coverage_check.parquet",
        "distractor_validity": stage_cache / "valid_distractors.parquet",
        "semantic_outlier_detection": stage_cache / "semantic_outlier_detection.parquet",
        "filtering": stage_cache / "filtered_questions.parquet",
    }
    output_path_raw = output_base / "benchmark_raw.parquet"
    output_path_final = output_base / "benchmark.parquet"

    if _should_run(skip_until, McqGenerationStage.GENERATION):
        seed_df_generation = prepare_generation_seed_dataset(config)
        dataset_out = batched_run(generate_questions, config, seed_df_generation, batch_size=config.ndd_batch_size)
        dataset_out.dropna(inplace=True)
        postprocess_generated_questions(dataset_out).to_parquet(paths["generation"])
        logger.info("Generated questions saved to %s", paths["generation"])
    last_output_path = paths["generation"]

    if _should_run(skip_until, McqGenerationStage.JUDGEMENT):
        dataset_in = pd.read_parquet(paths["generation"])
        seed_df_judgement = prepare_judgement_seed_dataset(config, dataset_in)
        dataset_out = batched_run(judge_questions, config, seed_df_judgement, batch_size=config.ndd_batch_size)
        postprocess_judged_questions(dataset_in, dataset_out).to_parquet(paths["judgement"])
        logger.info("Judged questions saved to %s", paths["judgement"])
    last_output_path = paths["judgement"]

    if _should_run(skip_until, McqGenerationStage.SEMANTIC_DEDUPLICATION):
        dataset_in = pd.read_parquet(last_output_path)
        if _is_enabled(config.semantic_deduplication_config):
            dataset_out = TextSemanticDeduplicationMCQ(config).run(dataset_in)
        else:
            dataset_out = dataset_in.copy()
            dataset_out["is_duplicate"] = False
        dataset_out.to_parquet(paths["semantic_deduplication"])
        logger.info("Semantic deduplication saved to %s", paths["semantic_deduplication"])
    last_output_path = paths["semantic_deduplication"]

    if config.do_distractor_expansion:
        if _should_run(skip_until, McqGenerationStage.DISTRACTOR_EXPANSION):
            dataset_in = pd.read_parquet(last_output_path)
            seed_df = prepare_distractor_expansion_seed_dataset(config, dataset_in)
            dataset_out = batched_run(expand_distractors, config, seed_df, batch_size=config.ndd_batch_size)
            postprocess_distractor_expansion(dataset_in, dataset_out, config).to_parquet(paths["distractor_expansion"])
            logger.info("Expanded distractors saved to %s", paths["distractor_expansion"])
        last_output_path = paths["distractor_expansion"]

    if config.do_coverage_check:
        if _should_run(skip_until, McqGenerationStage.COVERAGE_CHECK):
            dataset_in = pd.read_parquet(last_output_path)
            dataset_out = TextCoverageMCQ(config).analyze(dataset_in)
            dataset_out.to_parquet(paths["coverage"])
            logger.info("Coverage check saved to %s", paths["coverage"])
        last_output_path = paths["coverage"]

    if _should_run(skip_until, McqGenerationStage.DISTRACTOR_VALIDITY_CHECK):
        dataset_in = pd.read_parquet(last_output_path)
        seed_df = prepare_distractor_validity_seed_dataset(config, dataset_in)
        dataset_out = batched_run(check_distractor_validity, config, seed_df, batch_size=config.ndd_batch_size)
        postprocess_distractor_validity(dataset_in, dataset_out).to_parquet(paths["distractor_validity"])
        logger.info("Distractor validity saved to %s", paths["distractor_validity"])
    last_output_path = paths["distractor_validity"]

    if _should_run(skip_until, McqGenerationStage.SEMANTIC_OUTLIER_DETECTION):
        dataset_in = pd.read_parquet(last_output_path)
        if _is_enabled(config.semantic_outlier_detection_config):
            dataset_out = TextSemanticOutlierDetectionMCQ(config).detect(dataset_in)
        else:
            dataset_out = dataset_in.copy()
            dataset_out["answer_semantic_neighbours"] = None
            dataset_out["is_outlier"] = False
        dataset_out.to_parquet(paths["semantic_outlier_detection"])
        logger.info("Semantic outlier detection saved to %s", paths["semantic_outlier_detection"])
    last_output_path = paths["semantic_outlier_detection"]

    if _should_run(skip_until, McqGenerationStage.HALLUCINATION_EASINESS_DETECTION):
        dataset_in = pd.read_parquet(last_output_path)
        seed_df = prepare_filtering_seed_dataset(dataset_in, config)
        dataset_out = batched_run(filter_questions, config, seed_df, batch_size=config.ndd_batch_size)
        postprocess_filtered_questions(dataset_in, dataset_out, config).to_parquet(paths["filtering"])
        logger.info("Question filtering saved to %s", paths["filtering"])
    last_output_path = paths["filtering"]

    if not _should_run(skip_until, McqGenerationStage.FINAL_OUTPUT):
        return None

    dataset_final = pd.read_parquet(last_output_path)
    dataset_final.to_parquet(output_path_raw)
    if config.remove_hallucinated:
        dataset_final = dataset_final[~dataset_final["is_hallucination"]]
    if config.remove_easy:
        dataset_final = dataset_final[~dataset_final["is_easy"]]

    if len(dataset_final) == 0:
        logger.error("No questions left after filtering")
        return None

    dataset_final = dataset_final[
        ["id_question", "question_generated", "choices_generated", "answer_generated", "target_subject"]
    ]
    dataset_final.rename(
        columns={
            "id_question": "question_id",
            "question_generated": "question",
            "choices_generated": "options",
            "answer_generated": "answer",
            "target_subject": "category",
        },
        inplace=True,
    )
    dataset_final["answer_index"] = dataset_final["answer"].apply(lambda x: ord(x) - ord("A"))
    dataset_final["cot_content"] = "-"
    dataset_final["src"] = "-"
    dataset_final.to_parquet(output_path_final)
    logger.info("Final benchmark saved to %s", output_path_final)
    return output_path_final


def translate_mcq(config_path: str | os.PathLike[str], *, skip_until: str | None = None) -> Path | None:
    """Run the MCQ translation pipeline and return the final benchmark path."""
    import pandas as pd

    from nemotron.steps.byob.runtime.benchmark_families.mcq.utils import (
        postprocess_translated_questions,
        prepare_translation_seed_dataset,
    )
    from nemotron.steps.byob.runtime.config import ByobTranslationConfig
    from nemotron.steps.byob.runtime.translation.quality_metrics import evaluate_quality_metrics
    from nemotron.steps.byob.runtime.translation.translation import TranslationPipeline

    config = ByobTranslationConfig.from_yaml(str(config_path))
    output_base = Path(config.output_dir) / config.expt_name
    stage_cache = output_base / "stage_cache"
    stage_cache.mkdir(parents=True, exist_ok=True)

    output_path_translation = stage_cache / "translated_questions.parquet"
    output_path_backtranslation = stage_cache / "backtranslated_questions.parquet"
    output_path_quality_metrics = stage_cache / "quality_metrics.parquet"
    output_path_raw = output_base / "benchmark_raw.parquet"
    output_path_final = output_base / "benchmark.parquet"

    if _should_run(skip_until, McqTranslationStage.TRANSLATION):
        dataset_in = pd.read_parquet(config.dataset_path)
        seed_df = prepare_translation_seed_dataset(
            dataset_in,
            source_language=config.source_language,
            target_language=config.target_language,
            id_field="question_id",
            text_field="question",
            options_field="options",
        )
        translation_pipeline = TranslationPipeline(
            config=config,
        )
        dataset_out = translation_pipeline.translate(seed_df)
        dataset_out = postprocess_translated_questions(dataset_in, dataset_out)
        dataset_out.to_parquet(output_path_translation)
        logger.info("Translated questions saved to %s", output_path_translation)
    last_output_path = output_path_translation

    if _should_run(skip_until, McqTranslationStage.BACKTRANSLATION):
        dataset_in = pd.read_parquet(last_output_path)
        seed_df = prepare_translation_seed_dataset(
            dataset_in,
            source_language=config.target_language,
            target_language=config.source_language,
            id_field="question_id",
            text_field="question_translated",
            options_field="options_translated",
        )
        translation_pipeline = TranslationPipeline(
            config=config,
        )
        dataset_out = translation_pipeline.translate(seed_df)
        dataset_out = postprocess_translated_questions(
            dataset_in,
            dataset_out,
            id_field="question_id",
            text_field="question_translated",
            options_field="options_translated",
            answer_index_field="answer_index_translated",
            suffix="backtranslated",
        )
        dataset_out.to_parquet(output_path_backtranslation)
        logger.info("Back-translated questions saved to %s", output_path_backtranslation)
    last_output_path = output_path_backtranslation

    if _should_run(skip_until, McqTranslationStage.QUALITY_METRICS):
        dataset_in = pd.read_parquet(last_output_path)
        evaluate_quality_metrics(dataset_in, config).to_parquet(output_path_quality_metrics)
        logger.info("Quality metrics saved to %s", output_path_quality_metrics)
    last_output_path = output_path_quality_metrics

    if not _should_run(skip_until, McqTranslationStage.FINAL_OUTPUT):
        return None

    dataset_final = pd.read_parquet(last_output_path)
    dataset_final.to_parquet(output_path_raw)
    if config.remove_low_quality:
        dataset_final = dataset_final[dataset_final["is_quality_metric_passed"]]

    dataset_final = dataset_final[
        ["question_id", "question_translated", "options_translated", "answer_index_translated", "category"]
    ]
    dataset_final.rename(
        columns={
            "question_translated": "question",
            "options_translated": "options",
            "answer_index_translated": "answer_index",
        },
        inplace=True,
    )
    dataset_final["answer"] = dataset_final["answer_index"].apply(lambda x: chr(ord("A") + x))
    dataset_final["cot_content"] = "-"
    dataset_final["src"] = "-"
    dataset_final.to_parquet(output_path_final)
    logger.info("Translated benchmark saved to %s", output_path_final)
    return output_path_final
