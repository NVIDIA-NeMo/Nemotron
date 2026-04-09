# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic data generation (SDG) for the Nemotron-CC pipeline.

This script generates synthetic data from high-quality documents (buckets 18
and 19) using four LLM-based generation tasks from the Nemotron-CC paper:

  1. Diverse QA        — generate diverse question-answer pairs
  2. Distill           — condense text while preserving key information
  3. Extract Knowledge — rewrite as textbook/Wikipedia-style passages
  4. Knowledge List    — extract organized lists of factual information

Each task can be run independently via the --task flag, or all four can be
run sequentially with --task all.

The script reads bucketed parquet files from step 3 (quality classification),
preprocesses documents (splitting, filtering, joining segments), sends them
to an LLM for generation, postprocesses the results, and writes output to
parquet or JSONL.

This is a CPU-only pipeline — LLM inference happens via API calls to an
external endpoint (NVIDIA Integrate, or a self-hosted vLLM/TRT-LLM server).

Usage:
    # Run all SDG tasks on bucket 18/19 data
    python step_4-sdg.py \\
        --task all \\
        --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507

    # Run a single task
    python step_4-sdg.py \\
        --task diverse_qa \\
        --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507

    # Use a local/self-hosted endpoint
    python step_4-sdg.py \\
        --task distill \\
        --base-url http://localhost:8000/v1 \\
        --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507

See README.md in this directory for detailed usage instructions.
"""

import argparse
import json
import os
import time

from loguru import logger
from transformers import AutoTokenizer

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import (
    DistillStage,
    DiverseQAPostProcessingStage,
    DiverseQAStage,
    ExtractKnowledgeStage,
    KnowledgeListPostProcessingStage,
    KnowledgeListStage,
)
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
)
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.filters.heuristic import SubstringFilter
from nemo_curator.stages.text.filters.token import TokenCountFilter
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.stages.text.modifiers.string import (
    LineRemover,
    MarkdownRemover,
    QuotationRemover,
    Slicer,
)
from nemo_curator.stages.text.modules.joiner import DocumentJoiner
from nemo_curator.stages.text.modules.splitter import DocumentSplitter
from nemo_curator.tasks import DocumentBatch
from nemo_curator.tasks.utils import TaskPerfUtils


TASK_CONFIGS = {
    "diverse_qa": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": DIVERSE_QA_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 600,
    },
    "distill": {
        "system_prompt": NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
        "prompt_template": DISTILL_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 10,
        "max_input_tokens": 2000,
        "max_output_tokens": 1600,
    },
    "extract_knowledge": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1400,
        "max_output_tokens": 1400,
    },
    "knowledge_list": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": KNOWLEDGE_LIST_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 600,
    },
}

GENERATION_DEFAULTS = {
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": None,
    "max_output_tokens": 1600,
    "end_strings": "['</s>']",
}

HIGH_QUALITY_BUCKETS = [18, 19]


def _get_prefix_token_count(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt_template: str,
) -> int:
    """Calculate the number of tokens consumed by the prompt prefix."""
    if "{text}" in user_prompt_template:
        template_without_text = user_prompt_template.replace("{text}", "")
    else:
        template_without_text = user_prompt_template

    full_prefix = f"{system_prompt}\n{template_without_text}"
    tokens = tokenizer.encode(full_prefix)
    return len(tokens)


def _add_preprocessing_stages(
    pipeline: Pipeline,
    task_config: dict,
    tokenizer: AutoTokenizer,
    hf_token: str,
) -> Pipeline:
    """Add document splitting, filtering, and joining stages."""
    prefix_token_count = _get_prefix_token_count(
        tokenizer,
        task_config["system_prompt"],
        task_config["prompt_template"],
    )
    max_segment_tokens = task_config["max_input_tokens"] - prefix_token_count - 2

    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=tokenizer,
                hf_token=hf_token,
                min_tokens=task_config["min_document_tokens"],
            ),
            text_field="text",
            score_field="document_token_count",
        ),
    )
    pipeline.add_stage(
        DocumentSplitter(
            separator="\n",
            text_field="text",
            segment_id_field="segment_id",
        ),
    )

    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=tokenizer,
                hf_token=hf_token,
                max_tokens=max_segment_tokens,
            ),
            text_field="text",
            score_field="segment_token_count",
        ),
    )

    pipeline.add_stage(
        DocumentJoiner(
            separator="\n",
            text_field="text",
            segment_id_field="segment_id",
            document_id_field="id",
            max_length=max_segment_tokens,
            length_field="segment_token_count",
            drop_segment_id_field=False,
        ),
    )

    pipeline.add_stage(
        Filter(
            filter_fn=lambda x: x >= task_config["min_segment_tokens"],
            filter_field="segment_token_count",
        ),
    )

    return pipeline


def _add_diverse_qa_postprocessing(
    pipeline: Pipeline,
    tokenizer: AutoTokenizer,
    hf_token: str,
) -> Pipeline:
    """Add DiverseQA postprocessing stages."""
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, max_tokens=598),
            text_field="diverse_qa",
            score_field="rephrased_segment_token_count",
        ),
    )
    pipeline.add_stage(Modify(modifier_fn=MarkdownRemover(), input_fields="diverse_qa"))
    pipeline.add_stage(
        DiverseQAPostProcessingStage(
            input_field="text",
            qa_field="diverse_qa",
            tokenizer=tokenizer,
        ),
    )
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, min_tokens=100),
            text_field="diverse_qa",
            score_field="rephrased_document_token_count",
        ),
    )
    return pipeline


def _add_distill_postprocessing(
    pipeline: Pipeline,
    tokenizer: AutoTokenizer,
    hf_token: str,
) -> Pipeline:
    """Add Distill postprocessing stages."""
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, max_tokens=1598),
            text_field="distill",
            score_field="rephrased_segment_token_count",
        ),
    )
    pipeline.add_stage(Modify(modifier_fn=MarkdownRemover(), input_fields="distill"))
    pipeline.add_stage(
        ScoreFilter(
            SubstringFilter(substring="Paraphrased Text:", position="prefix"),
            text_field="distill",
            score_field="substring",
        ),
    )
    pipeline.add_stage(
        Modify(
            modifier_fn=Slicer(left="Paraphrased Text:", include_left=False, strip=True),
            input_fields="distill",
        ),
    )
    pipeline.add_stage(Modify(modifier_fn=QuotationRemover(), input_fields="distill"))
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, min_tokens=50),
            text_field="distill",
            score_field="rephrased_document_token_count",
        ),
    )
    return pipeline


def _add_extract_knowledge_postprocessing(
    pipeline: Pipeline,
    tokenizer: AutoTokenizer,
    hf_token: str,
) -> Pipeline:
    """Add ExtractKnowledge postprocessing stages."""
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, max_tokens=1398),
            text_field="extract_knowledge",
            score_field="rephrased_segment_token_count",
        ),
    )
    pipeline.add_stage(Modify(modifier_fn=MarkdownRemover(), input_fields="extract_knowledge"))
    pipeline.add_stage(
        Modify(
            modifier_fn=LineRemover(patterns=["Passage:", "Passage 1:", "Passage 2:", "Passage 3:"]),
            input_fields="extract_knowledge",
        ),
    )
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, min_tokens=50),
            text_field="extract_knowledge",
            score_field="rephrased_document_token_count",
        ),
    )
    return pipeline


def _add_knowledge_list_postprocessing(
    pipeline: Pipeline,
    tokenizer: AutoTokenizer,
    hf_token: str,
) -> Pipeline:
    """Add KnowledgeList postprocessing stages."""
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, max_tokens=598),
            text_field="knowledge_list",
            score_field="rephrased_segment_token_count",
        ),
    )
    pipeline.add_stage(Modify(modifier_fn=MarkdownRemover(), input_fields="knowledge_list"))
    pipeline.add_stage(
        KnowledgeListPostProcessingStage(input_field="knowledge_list"),
    )
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, min_tokens=50),
            text_field="knowledge_list",
            score_field="rephrased_document_token_count",
        ),
    )
    return pipeline


def build_pipeline(
    task_name: str,
    llm_client: AsyncOpenAIClient,
    generation_config: GenerationConfig,
    model_name: str,
    tokenizer: AutoTokenizer,
    hf_token: str,
    input_dir: str,
    output_dir: str,
    output_format: str,
) -> Pipeline:
    """Build a complete SDG pipeline for one task."""
    task_config = TASK_CONFIGS[task_name]

    pipeline = Pipeline(
        name=f"nemotron_cc_sdg_{task_name}",
        description=f"Nemotron-CC SDG: {task_name} on high-quality data",
    )

    input_paths = []
    for bucket in HIGH_QUALITY_BUCKETS:
        bucket_dir = os.path.join(input_dir, f"ensemble-max-int={bucket}")
        if os.path.isdir(bucket_dir):
            input_paths.append(bucket_dir)
        else:
            logger.warning(f"Bucket directory not found: {bucket_dir}")
    if not input_paths:
        msg = f"No bucket directories found in {input_dir} for buckets {HIGH_QUALITY_BUCKETS}"
        raise FileNotFoundError(msg)

    pipeline.add_stage(
        ParquetReader(
            file_paths=input_paths,
            read_kwargs={"engine": "pyarrow", "dtype_backend": "pyarrow"},
        )
    )

    @processing_stage(name="add-document-id")
    def add_document_id(batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        if "id" not in df.columns:
            df["id"] = range(len(df))
        batch.data = df
        return batch

    pipeline.add_stage(add_document_id)

    pipeline = _add_preprocessing_stages(pipeline, task_config, tokenizer, hf_token)

    if task_name == "diverse_qa":
        pipeline.add_stage(
            DiverseQAStage(
                client=llm_client,
                model_name=model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="diverse_qa",
            )
        )
        pipeline = _add_diverse_qa_postprocessing(pipeline, tokenizer, hf_token)

    elif task_name == "distill":
        pipeline.add_stage(
            DistillStage(
                client=llm_client,
                model_name=model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="distill",
            )
        )
        pipeline = _add_distill_postprocessing(pipeline, tokenizer, hf_token)

    elif task_name == "extract_knowledge":
        pipeline.add_stage(
            ExtractKnowledgeStage(
                client=llm_client,
                model_name=model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="extract_knowledge",
            )
        )
        pipeline = _add_extract_knowledge_postprocessing(pipeline, tokenizer, hf_token)

    elif task_name == "knowledge_list":
        pipeline.add_stage(
            KnowledgeListStage(
                client=llm_client,
                model_name=model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="knowledge_list",
            )
        )
        pipeline = _add_knowledge_list_postprocessing(pipeline, tokenizer, hf_token)

    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    if output_format == "jsonl":
        pipeline.add_stage(JsonlWriter(path=task_output_dir))
    else:
        pipeline.add_stage(ParquetWriter(path=task_output_dir))

    return pipeline


def _save_metrics(metrics: dict, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {file_path}")


def run_task(
    task_name: str,
    args: argparse.Namespace,
    llm_client: AsyncOpenAIClient,
    generation_config: GenerationConfig,
    tokenizer: AutoTokenizer,
) -> dict:
    """Run a single SDG task and return its metrics."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting SDG task: {task_name}")
    logger.info(f"{'=' * 60}")

    pipeline = build_pipeline(
        task_name=task_name,
        llm_client=llm_client,
        generation_config=generation_config,
        model_name=args.model_name,
        tokenizer=tokenizer,
        hf_token=args.hf_token,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
    )

    logger.info(pipeline.describe())

    if args.executor == "ray_data":
        executor = RayDataExecutor()
    else:
        executor = XennaExecutor()

    start_time = time.perf_counter()
    results = pipeline.run(executor)
    elapsed = time.perf_counter() - start_time

    logger.info(f"Task '{task_name}' completed in {elapsed:.1f}s ({elapsed / 60:.1f}m)")

    output_files = []
    if results:
        for result in results:
            if hasattr(result, "data") and result.data:
                for file_path in result.data:
                    output_files.append(file_path)
        logger.info(f"  Output files: {len(output_files)}")

    metrics = TaskPerfUtils.aggregate_task_metrics(results) if results else {}
    metrics["task"] = task_name
    metrics["elapsed_s"] = round(elapsed, 2)
    metrics["num_output_files"] = len(output_files)

    return metrics


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient(num_cpus=args.num_cpus, include_dashboard=False)
    ray_client.start()

    tokenizer_name = args.tokenizer if args.tokenizer else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    args.hf_token = os.environ.get("HF_TOKEN", "")

    api_key = args.api_key
    if not api_key:
        msg = (
            "API key is required. Set NVIDIA_API_KEY environment variable or use --api-key. "
            "Get your API key from https://build.nvidia.com/settings/api-keys"
        )
        raise ValueError(msg)

    llm_client = AsyncOpenAIClient(
        api_key=api_key,
        base_url=args.base_url,
        max_concurrent_requests=args.max_concurrent_requests,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        timeout=args.timeout,
    )

    generation_config = GenerationConfig(
        temperature=args.temperature if args.temperature is not None else GENERATION_DEFAULTS["temperature"],
        top_p=args.top_p if args.top_p is not None else GENERATION_DEFAULTS["top_p"],
        top_k=args.top_k if args.top_k is not None else GENERATION_DEFAULTS["top_k"],
        max_tokens=args.max_tokens if args.max_tokens is not None else GENERATION_DEFAULTS["max_output_tokens"],
        stop=args.end_strings if args.end_strings is not None else GENERATION_DEFAULTS["end_strings"],
        seed=args.seed,
    )

    if args.task == "all":
        tasks = list(TASK_CONFIGS.keys())
    else:
        tasks = [args.task]

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Nemotron-CC SDG Pipeline")
    logger.info(f"  Input:     {args.input_dir}")
    logger.info(f"  Output:    {args.output_dir}")
    logger.info(f"  Buckets:   {HIGH_QUALITY_BUCKETS}")
    logger.info(f"  Tasks:     {tasks}")
    logger.info(f"  Model:     {args.model_name}")
    logger.info(f"  Tokenizer: {tokenizer_name}")

    all_metrics = {}
    total_start = time.perf_counter()

    for task_name in tasks:
        task_metrics = run_task(
            task_name=task_name,
            args=args,
            llm_client=llm_client,
            generation_config=generation_config,
            tokenizer=tokenizer,
        )
        all_metrics[task_name] = task_metrics
        _save_metrics(
            task_metrics,
            os.path.join(args.output_dir, f"{task_name}_metrics.json"),
        )

    total_elapsed = time.perf_counter() - total_start
    all_metrics["total_elapsed_s"] = round(total_elapsed, 2)
    _save_metrics(all_metrics, os.path.join(args.output_dir, "sdg_metrics.json"))

    logger.info(f"All SDG tasks completed in {total_elapsed:.1f}s ({total_elapsed / 60:.1f}m)")

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nemotron-CC Step 4: Synthetic Data Generation (SDG). "
            "Runs LLM-based generation tasks on high-quality (bucket 18/19) data "
            "using Qwen3-30B-A3B-Instruct via API."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "diverse_qa", "distill", "extract_knowledge", "knowledge_list"],
        help="SDG task to run. Use 'all' to run all four tasks sequentially.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/quality_labeling/bucketed_results",
        help="Directory containing bucketed parquet files from step 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/sdg_output",
        help="Base output directory. Sub-directories are created per task.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["jsonl", "parquet"],
        help="Output format for generated data.",
    )

    parser.add_argument(
        "--executor",
        type=str,
        default="ray_data",
        choices=["xenna", "ray_data"],
        help="Pipeline executor backend.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=64,
        help="Number of CPUs for the local Ray cluster (default: all available).",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NVIDIA_API_KEY", ""),
        help="NVIDIA API key (or set NVIDIA_API_KEY environment variable).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="Base URL for the LLM API endpoint.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=3,
        help="Maximum number of concurrent API requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests.",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=1.0,
        help="Base delay between retries (in seconds).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Timeout in seconds for each LLM API request.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Name of the model to use for generation.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name/path. Defaults to --model-name.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0.0-2.0). Higher = more diverse.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter (0.0-1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate per sample (overrides per-task defaults).",
    )
    parser.add_argument(
        "--end-strings",
        type=str,
        default=None,
        help="End strings to stop generation.",
    )

    return parser


if __name__ == "__main__":
    args = attach_args().parse_args()
    main(args)
