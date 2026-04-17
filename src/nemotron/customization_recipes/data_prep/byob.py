# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Build Your Own Benchmark (BYOB) MCQ generation pipeline.

Generates multiple-choice questions from custom corpora using few-shot
learning, then judges, expands distractors, validates, and filters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_DD_MSG = (
    "data-designer is required for BYOB stages. "
    "Install with: pip install data-designer"
)

# ---------------------------------------------------------------------------
# Pydantic response models (ported from Speaker byob/mcq/response_model.py)
# ---------------------------------------------------------------------------


class MCQQuestion(BaseModel):
    """A single four-choice question."""

    question: str = Field(..., description="The question text")
    choice_a: str = Field(..., description="Choice A")
    choice_b: str = Field(..., description="Choice B")
    choice_c: str = Field(..., description="Choice C")
    choice_d: str = Field(..., description="Choice D")
    answer: Literal["A", "B", "C", "D"] = Field(..., description="Correct answer")


class MCQQuestionList(BaseModel):
    """Batch of generated questions."""

    questions: list[MCQQuestion] = Field(..., description="Generated questions")


class JudgeResult(BaseModel):
    """LLM judge output for a single question."""

    reason: str = Field(..., description="Reason for judgement")
    is_valid: bool = Field(..., description="Whether the question is valid")
    category: Literal["knowledge", "reasoning", "both"] = Field(
        ..., description="Question category"
    )


class DistractorExpansion(BaseModel):
    """Six additional distractor choices (E-J)."""

    choice_e: str = Field(..., description="Choice E")
    choice_f: str = Field(..., description="Choice F")
    choice_g: str = Field(..., description="Choice G")
    choice_h: str = Field(..., description="Choice H")
    choice_i: str = Field(..., description="Choice I")
    choice_j: str = Field(..., description="Choice J")


class DistractorValidityFourChoices(BaseModel):
    """Validity flags for 4-choice questions."""

    choice_a: Literal["Yes", "No"] = Field(...)
    choice_b: Literal["Yes", "No"] = Field(...)
    choice_c: Literal["Yes", "No"] = Field(...)
    choice_d: Literal["Yes", "No"] = Field(...)


class DistractorValidityTenChoices(BaseModel):
    """Validity flags for 10-choice questions."""

    choice_a: Literal["Yes", "No"] = Field(...)
    choice_b: Literal["Yes", "No"] = Field(...)
    choice_c: Literal["Yes", "No"] = Field(...)
    choice_d: Literal["Yes", "No"] = Field(...)
    choice_e: Literal["Yes", "No"] = Field(...)
    choice_f: Literal["Yes", "No"] = Field(...)
    choice_g: Literal["Yes", "No"] = Field(...)
    choice_h: Literal["Yes", "No"] = Field(...)
    choice_i: Literal["Yes", "No"] = Field(...)
    choice_j: Literal["Yes", "No"] = Field(...)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ByobConfig:
    """Configuration for the BYOB MCQ generation pipeline."""

    expt_name: str = ""
    output_dir: str = "data/byob"
    input_dir: str = ""
    language: str = "en"

    hf_dataset: str = "cais/mmlu"
    subset: str = ""
    split: str = "test"

    source_subjects: list[str] = field(default_factory=list)
    target_subjects: list[str] = field(default_factory=list)
    target_source_mapping: dict = field(default_factory=dict)

    few_shot_samples_per_query: int = 5
    queries_per_target_subject_document: int = 1
    num_questions_per_query: int = 5

    prompt_config: Optional[dict] = None
    generation_model_config: dict = field(default_factory=dict)
    judge_model_config: dict = field(default_factory=dict)

    do_distractor_expansion: bool = False
    distractor_expansion_model_config: dict = field(default_factory=dict)
    distractor_validity_model_config: dict = field(default_factory=dict)

    filtering_model_configs: dict = field(default_factory=lambda: {"easiness": [], "hallucination": []})
    easiness_threshold: float = 0.5
    hallucination_threshold: float = 0.5
    remove_hallucinated: bool = True
    remove_easy: bool = False

    ndd_batch_size: int = 1000
    random_seed: Optional[int] = None
    metadata_file: Optional[str] = None

    semantic_deduplication_config: dict = field(
        default_factory=lambda: {
            "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
            "n_clusters": 1,
            "eps": 0.07,
            "remove_duplicates": False,
        }
    )
    semantic_outlier_detection_config: dict = field(
        default_factory=lambda: {
            "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
            "n_neighbours_min": 1,
            "remove_outliers": False,
        }
    )
    chunking_config: dict = field(default_factory=lambda: {"window_size": None})
    do_coverage_check: bool = False
    coverage_check_config: dict = field(
        default_factory=lambda: {"window_size": None, "model_identifier": None}
    )

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "ByobConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(ByobConfig)
        merged = OmegaConf.merge(schema, cfg)
        return ByobConfig(**OmegaConf.to_container(merged, resolve=True))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_dd():
    try:
        from data_designer.essentials import DataDesigner  # noqa: F401
    except ImportError as exc:
        raise ImportError(_DD_MSG) from exc


def _run_dd_stage(
    config: ByobConfig,
    seed_df: "pd.DataFrame",
    model_config: dict,
    column_name: str,
    system_prompt: str,
    prompt: str,
    output_format: type[BaseModel],
    model_alias: str,
    stage_tag: str,
) -> "pd.DataFrame":
    """Generic DataDesigner stage runner used by all BYOB stages."""
    _require_dd()
    from data_designer.essentials import (
        DataDesigner,
        DataDesignerConfigBuilder,
        SeedConfig,
    )

    os.makedirs(f"{config.output_dir}/temp", exist_ok=True)
    seed_path = (
        f"{config.output_dir}/temp/"
        f"{config.expt_name}_{stage_tag}_{datetime.now():%Y%m%d%H%M%S}.csv"
    )
    seed_df.to_csv(seed_path, index=False)

    designer = DataDesigner(
        artifact_path=f"{config.output_dir}/{config.expt_name}/artifacts/data_designer"
    )
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(SeedConfig(dataset=seed_path))

    builder.add_column(
        name=column_name,
        column_type="llm-structured",
        system_prompt=system_prompt,
        prompt=prompt,
        output_format=output_format,
        model_alias=model_alias,
    )
    builder.validate()

    results = designer.create(config_builder=builder, num_records=len(seed_df))
    dataset = results.load_dataset()
    dataset.dropna(inplace=True)
    os.remove(seed_path)
    return dataset


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def generate_questions(config: ByobConfig, seed_df: "pd.DataFrame") -> "pd.DataFrame":
    """Generate MCQ questions using LLM few-shot prompting.

    Args:
        config: BYOB configuration.
        seed_df: Seed DataFrame with few-shot examples and target text.

    Returns:
        DataFrame with ``result`` column containing generated questions.
    """
    prompts = config.prompt_config or {}
    qa_cfg = prompts.get("qa_generation", {})
    sys_prompt = qa_cfg.get("system_prompt", "").format(
        num_few_shot_samples=config.few_shot_samples_per_query,
        num_questions=config.num_questions_per_query,
    )
    user_prompt = qa_cfg.get("prompt", "").format(
        num_questions=config.num_questions_per_query,
    )
    return _run_dd_stage(
        config,
        seed_df,
        config.generation_model_config,
        "result",
        sys_prompt,
        user_prompt,
        MCQQuestionList,
        config.generation_model_config.get("alias", "generator"),
        "generation",
    )


def judge_questions(config: ByobConfig, seed_df: "pd.DataFrame") -> "pd.DataFrame":
    """Judge quality and validity of generated questions.

    Returns:
        DataFrame with ``result`` column containing JudgeResult.
    """
    prompts = config.prompt_config or {}
    jcfg = prompts.get("question_judge", {})
    return _run_dd_stage(
        config,
        seed_df,
        config.judge_model_config,
        "result",
        jcfg.get("system_prompt", ""),
        jcfg.get("prompt", ""),
        JudgeResult,
        config.judge_model_config.get("alias", "judge"),
        "judge",
    )


def expand_distractors(config: ByobConfig, seed_df: "pd.DataFrame") -> "pd.DataFrame":
    """Expand from 4 to 10 answer choices.

    Returns:
        DataFrame with ``result_distractor_expansion`` column.
    """
    prompts = config.prompt_config or {}
    dcfg = prompts.get("distractor_expansion", {})
    return _run_dd_stage(
        config,
        seed_df,
        config.distractor_expansion_model_config,
        "result_distractor_expansion",
        dcfg.get("system_prompt", ""),
        dcfg.get("prompt", ""),
        DistractorExpansion,
        config.distractor_expansion_model_config.get("alias", "distractor_expander"),
        "distractor_expansion",
    )


def filter_questions(config: ByobConfig, dataset: "pd.DataFrame") -> "pd.DataFrame":
    """Filter questions for easiness and hallucination.

    Uses multiple LLM models to answer generated questions, then flags
    those that are too easy or hallucinated based on correct-answer ratios.

    Returns:
        DataFrame with response columns for each filter type and model.
    """
    _require_dd()
    from data_designer.essentials import (
        DataDesigner,
        DataDesignerConfigBuilder,
        SeedConfig,
    )

    num_choices = 10 if config.do_distractor_expansion else 4
    choices_text = "/".join(chr(ord("A") + i) for i in range(num_choices))

    prompts = config.prompt_config or {}
    sys_prompts = {
        "easiness": prompts.get("easiness_filter", {}).get("system_prompt", "").format(num_choices=num_choices),
        "hallucination": prompts.get("hallucination_filter", {}).get("system_prompt", "").format(num_choices=num_choices),
    }
    user_prompts = {
        "easiness": prompts.get("easiness_filter", {}).get("prompt", "").format(choices=choices_text),
        "hallucination": prompts.get("hallucination_filter", {}).get("prompt", "").format(choices=choices_text),
    }

    all_model_configs = [
        mc
        for ft in ("easiness", "hallucination")
        for mc in config.filtering_model_configs.get(ft, [])
    ]

    os.makedirs(f"{config.output_dir}/temp", exist_ok=True)
    seed_path = (
        f"{config.output_dir}/temp/"
        f"{config.expt_name}_filtering_{datetime.now():%Y%m%d%H%M%S}.csv"
    )
    dataset.to_csv(seed_path, index=False)

    designer = DataDesigner(
        artifact_path=f"{config.output_dir}/{config.expt_name}/artifacts/data_designer"
    )
    builder = DataDesignerConfigBuilder(model_configs=all_model_configs)
    builder.with_seed_dataset(SeedConfig(dataset=seed_path))

    for ft in ("easiness", "hallucination"):
        for mc in config.filtering_model_configs.get(ft, []):
            builder.add_column(
                name=f"response_{ft}_{mc['alias']}",
                column_type="llm-text",
                system_prompt=sys_prompts[ft],
                prompt=user_prompts[ft],
                model_alias=mc["alias"],
            )
    builder.validate()

    results = designer.create(config_builder=builder, num_records=len(dataset))
    df = results.load_dataset()
    os.remove(seed_path)
    return df


def check_distractor_validity(
    config: ByobConfig, dataset: "pd.DataFrame"
) -> "pd.DataFrame":
    """Verify that only the designated answer is correct.

    Returns:
        DataFrame with ``result_distractor_validity`` column.
    """
    num_choices = 10 if config.do_distractor_expansion else 4
    prompts = config.prompt_config or {}
    dv = prompts.get("distractor_validity", {})
    fmt = DistractorValidityTenChoices if num_choices == 10 else DistractorValidityFourChoices

    dataset = dataset.copy()
    dataset["num_choices"] = num_choices

    return _run_dd_stage(
        config,
        dataset,
        config.distractor_validity_model_config,
        "result_distractor_validity",
        dv.get("system_prompt", "").format(num_choices=num_choices),
        dv.get("prompt", ""),
        fmt,
        config.distractor_validity_model_config.get("alias", "validity_checker"),
        "distractor_validity",
    )


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def prepare_byob_seed(cfg: "DictConfig") -> dict:
    """Prepare a seed dataset for BYOB MCQ generation.

    Loads source data (from HuggingFace or a local directory) and writes
    a seed JSONL file that can be consumed by :func:`generate_byob_benchmark`.

    Args:
        cfg: OmegaConf DictConfig with BYOB parameters.

    Returns:
        Dict with ``seed_path`` and ``num_records``.
    """
    import pandas as pd

    byob_cfg = ByobConfig.from_omegaconf(cfg)
    os.makedirs(byob_cfg.output_dir, exist_ok=True)

    if byob_cfg.input_dir:
        seed_df = pd.read_json(byob_cfg.input_dir, lines=True)
    else:
        try:
            from datasets import load_dataset as hf_load
        except ImportError as exc:
            raise ImportError(
                "datasets is required for loading HuggingFace datasets. "
                "Install with: pip install datasets"
            ) from exc
        ds = hf_load(byob_cfg.hf_dataset, byob_cfg.subset or None, split=byob_cfg.split)
        seed_df = ds.to_pandas()

    seed_path = os.path.join(byob_cfg.output_dir, "seed_dataset.jsonl")
    seed_df.to_json(seed_path, orient="records", lines=True)
    log.info("BYOB seed dataset written: %s (%d records)", seed_path, len(seed_df))

    return {"seed_path": seed_path, "num_records": len(seed_df)}


def generate_byob_benchmark(cfg: "DictConfig") -> dict:
    """Run the full BYOB MCQ benchmark generation pipeline.

    Orchestrates generate -> judge -> expand distractors -> validate ->
    filter in sequence, writing the final benchmark to ``output_dir``.

    Args:
        cfg: OmegaConf DictConfig with BYOB parameters.

    Returns:
        Dict with ``output_dir``, ``num_questions``, and per-stage counts.
    """
    import pandas as pd

    byob_cfg = ByobConfig.from_omegaconf(cfg)
    result: Dict[str, object] = {"output_dir": byob_cfg.output_dir}

    os.makedirs(byob_cfg.output_dir, exist_ok=True)

    # Load seed data -- either from HuggingFace or local input_dir
    if byob_cfg.input_dir:
        seed_df = pd.read_json(byob_cfg.input_dir, lines=True)
    else:
        try:
            from datasets import load_dataset as hf_load
        except ImportError as exc:
            raise ImportError(
                "datasets is required for loading HuggingFace datasets. "
                "Install with: pip install datasets"
            ) from exc
        ds = hf_load(byob_cfg.hf_dataset, byob_cfg.subset or None, split=byob_cfg.split)
        seed_df = ds.to_pandas()

    log.info("BYOB pipeline: %d seed records", len(seed_df))

    # Stage 1: Generate MCQ questions
    log.info("[BYOB 1/5] Generating questions")
    gen_df = generate_questions(byob_cfg, seed_df)
    result["generated"] = len(gen_df)

    # Stage 2: Judge question quality
    log.info("[BYOB 2/5] Judging questions")
    judged_df = judge_questions(byob_cfg, gen_df)
    result["judged"] = len(judged_df)

    # Stage 3: Expand distractors (optional)
    if byob_cfg.do_distractor_expansion:
        log.info("[BYOB 3/5] Expanding distractors")
        judged_df = expand_distractors(byob_cfg, judged_df)
        result["expanded"] = len(judged_df)
    else:
        log.info("[BYOB 3/5] Distractor expansion skipped")

    # Stage 4: Validate distractor correctness
    log.info("[BYOB 4/5] Checking distractor validity")
    validated_df = check_distractor_validity(byob_cfg, judged_df)
    result["validated"] = len(validated_df)

    # Stage 5: Filter for easiness and hallucination
    log.info("[BYOB 5/5] Filtering questions")
    final_df = filter_questions(byob_cfg, validated_df)
    result["final"] = len(final_df)

    # Persist final benchmark
    output_path = os.path.join(byob_cfg.output_dir, "benchmark.jsonl")
    final_df.to_json(output_path, orient="records", lines=True)
    log.info("BYOB benchmark written: %s (%d questions)", output_path, len(final_df))

    result["num_questions"] = len(final_df)
    return result
