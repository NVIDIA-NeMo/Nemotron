# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer>=0.5.6",
#     "pydantic>=2",
#     "omegaconf>=2.3",
#     "pyyaml",
# ]
#
# [tool.runspec]
# schema = "1"
# name = "data/sdg/long-document/text-qa"
# image = "nvcr.io/nvidia/pytorch:25.12-py3"
# setup = "Inline PEP 723 deps resolved at runtime via `uv run --no-project`."
#
# [tool.runspec.run]
# launch = "direct"
# cmd = "uv run --no-project {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config"
# default = "03-text-qa"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Long-Document Understanding Text Question-Answering Recipe

Generate question-answer pairs from OCR-transcribed document text using a
reasoning LLM. For each seed record the pipeline:

  1. Samples a question type (multiple choice, true/false, short answer, numerical)
  2. Generates a structured question + answer pair grounded in the transcribed text
  3. Evaluates question relevance against the source text
  4. Evaluates answer correctness against the source text

Prerequisites:
    - A seed parquet file containing a `transcribed_texts` column with the
      OCR-transcribed document text (e.g. output of 02-nemotron-parse-ocr-sdg.py).
    - A vLLM-compatible deployment of the reasoning LLM
      (default: openai/gpt-oss-120b).
      Recommended vLLM launch flags:
        --tensor-parallel-size 2
        --reasoning-parser openai_gptoss

      Example launch script for 2× H100:
        docker run --gpus all \
            -p 8000:8000 \
            vllm/vllm-openai:latest \
            --model openai/gpt-oss-120b \
            --tensor-parallel-size 2 \
            --reasoning-parser openai_gptoss \
            --gpu-memory-utilization 0.80 \
            --max-model-len 32768

Run:
    # Basic usage (seed-path should point to the output of 02-nemotron-parse-ocr-sdg.py)
    uv run 03-text-qa-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path artifacts/nemotron_parse_ocr/parquet-files/*.parquet

    # Custom model and record count
    uv run 03-text-qa-sdg.py --vllm-endpoint http://localhost:8000/v1 --seed-path artifacts/nemotron_parse_ocr/parquet-files/*.parquet --num-records 100

    # For help message and available options
    uv run 03-text-qa-sdg.py --help
"""

import sys
from typing import Literal
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults
from pydantic import BaseModel, ConfigDict, Field

# Pull in the shared YAML+dotlist+Pydantic loader from the sibling helper
# (folder name has a dash so it isn't an importable package).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _recipe_config import load_recipe_config  # noqa: E402  (after sys.path mutation)

DEFAULT_REASONER_MODEL = "openai/gpt-oss-120b"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "03-text-qa.yaml"


class TextQAConfig(BaseModel):
    """Pydantic config for the long-document text-QA stage."""

    model_config = ConfigDict(extra="forbid")

    vllm_endpoint: str = Field(
        ...,
        description="Base URL of the vLLM server hosting the reasoning LLM.",
    )
    seed_path: Path = Field(
        ...,
        description="Path to a parquet file with a ``transcribed_texts`` column "
                    "(typically the output of 02-nemotron-parse-ocr-sdg.py).",
    )
    model_alias: str = Field(
        default="reasoner",
        description="Alias under which the reasoning LLM is published by the vLLM server.",
    )
    model_id: str = Field(
        default=DEFAULT_REASONER_MODEL,
        description="HF model id served by vLLM.",
    )
    num_records: int = Field(default=5, gt=0, description="Number of records to generate.")
    artifact_path: Path | None = Field(
        default=None,
        description="Optional Data Designer artifact directory.",
    )


VLLM_PROVIDER_NAME = "vllm"

# =============================================================================
# Structured output schemas
# =============================================================================


class QuestionAnswer(BaseModel):
    question: str = Field(..., description="The question to be answered.")
    answer: str = Field(..., description="The correct answer to the question.")


class QuestionRelevance(BaseModel):
    is_relevant: Literal["Relevant", "Irrelevant"] = Field(
        ...,
        description="The relevance of the question to the document content provided.",
    )


class AnswerCorrectness(BaseModel):
    is_correct: Literal["Correct", "Incorrect"] = Field(
        ..., description="Whether the answer is correct."
    )


# =============================================================================
# Prompt templates
# =============================================================================

PROMPT_QUESTION_ANSWER = """\
<question-type>
{{question_type}}
</question-type>

<context>
{{ transcribed_texts }}
</context>

You are an expert in creating challenging reasoning questions that require deep analysis \
and critical thinking. Your task is to examine the provided pages information and create a \
question that can only be answered by reviewing <context>.

Create a question & answer pair using <context> of type <question-type>.\
"""

PROMPT_QUESTION_RELEVANCE = """\
<context>
{{ transcribed_texts }}
</context>

<question>
{{ question_and_answer.question }}
</question>

Determine if the <question> is relevant to the <context>.\
"""

PROMPT_ANSWER_CORRECTNESS = """\
<context>
{{ transcribed_texts }}
</context>

<question>
{{ question_and_answer.question }}
</question>

<answer>
{{ question_and_answer.answer }}
</answer>

Determine if the <answer> to <question> is correct given <context>.\
"""


# =============================================================================
# Pipeline configuration
# =============================================================================

def build_config(
    seed_path: str = "seed.parquet",
    model_alias: str = "reasoner",
    model_id: str = DEFAULT_REASONER_MODEL,
) -> dd.DataDesignerConfigBuilder:
    model_configs = [
        dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=VLLM_PROVIDER_NAME,
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=32768,
                timeout=1200,
                extra_body={"reasoning_effort": "high"},
                max_parallel_requests=32,
            ),
        ),
    ]

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="question_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "multiple choice",
                    "true or false",
                    "short answer",
                    "numerical question",
                ],
            ),
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="question_and_answer",
            model_alias=model_alias,
            prompt=PROMPT_QUESTION_ANSWER,
            output_format=QuestionAnswer,
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="question_relevance",
            model_alias=model_alias,
            prompt=PROMPT_QUESTION_RELEVANCE,
            output_format=QuestionRelevance,
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="answer_correctness",
            model_alias=model_alias,
            prompt=PROMPT_ANSWER_CORRECTNESS,
            output_format=AnswerCorrectness,
        )
    )

    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    vllm_endpoint: str,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    model_providers = [
        dd.ModelProvider(
            name=VLLM_PROVIDER_NAME,
            endpoint=vllm_endpoint,
        ),
    ]
    data_designer = DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
    )
    data_designer.set_run_config(dd.RunConfig(progress_bar=True, disable_early_shutdown=True))
    results = data_designer.create(config_builder, num_records=num_records, dataset_name="text_qa")
    return results


def main(cfg: TextQAConfig | None = None) -> None:
    """Entry point. ``cfg`` is supplied when called from the Nemotron CLI;
    when called as a script we parse ``--config`` + dotlist overrides ourselves."""
    if cfg is None:
        cfg = load_recipe_config(DEFAULT_CONFIG_PATH, TextQAConfig)

    config_builder = build_config(
        seed_path=str(cfg.seed_path),
        model_alias=cfg.model_alias,
        model_id=cfg.model_id,
    )
    results = create_dataset(
        config_builder,
        num_records=cfg.num_records,
        vllm_endpoint=cfg.vllm_endpoint,
        artifact_path=cfg.artifact_path,
    )

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    results.load_analysis().to_report()


if __name__ == "__main__":
    main()
