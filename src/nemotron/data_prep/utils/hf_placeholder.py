# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""HuggingFace placeholder resolution for RL datasets.

The nvidia/Nemotron-3-Nano-RL-Training-Blend and
nvidia/Nemotron-3-Super-RL-Training-Blends datasets contain placeholder entries
for external datasets (DAPO, Skywork). These placeholders have an `_hf_placeholder`
field containing row indices and question templates that need to be resolved by
fetching the actual data from HuggingFace.

This module provides:
- Configuration for target datasets (HF repos, field paths, template types)
- Helper functions for template restoration (DAPO prefix/suffix, Skywork {question})
- Main resolution function that transforms placeholder records into full records
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Configuration for placeholder datasets that need resolution
# Maps dataset names (as they appear in the blend) to their HF source info

# Nano3 target datasets
# Reference: https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend/blob/main/create_nanov3_jsonl.py
NANO3_TARGET_DATASETS: dict[str, dict[str, Any]] = {
    "nano_v3_sft_profiled_dapo17k": {
        "hf_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "split": "train",
        "question_path": ["prompt", 0, "content"],
        "answer_path": ["reward_model", "ground_truth"],
        "template_type": "dapo",
    },
    "nano_v3_sft_profiled_skywork_no_omni": {
        "hf_dataset": "Skywork/Skywork-OR1-RL-Data",
        "split": "math",
        "question_path": ["prompt", 0, "content"],
        "answer_path": ["reward_model", "ground_truth"],
        "template_type": "skywork",
    },
}

# Super3 target datasets
# Reference: https://huggingface.co/datasets/nvidia/Nemotron-3-Super-RL-Training-Blends
SUPER3_TARGET_DATASETS: dict[str, dict[str, Any]] = {
    "super_v3_lcsft_step1000_dapo17k": {
        "hf_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "split": "train",
        "question_path": ["prompt", 0, "content"],
        "answer_path": ["reward_model", "ground_truth"],
        "template_type": "dapo",
    },
    "super_v3_lcsft_step1000_skyworks": {
        "hf_dataset": "Skywork/Skywork-OR1-RL-Data",
        "split": "math",
        "question_path": ["prompt", 0, "content"],
        "answer_path": ["reward_model", "ground_truth"],
        "template_type": "skywork",
    },
}

# Backward compatibility alias
TARGET_DATASETS = NANO3_TARGET_DATASETS


@dataclass
class PlaceholderConfig:
    """Configuration for a single placeholder dataset."""

    hf_dataset: str
    split: str
    question_path: list[str | int]
    answer_path: list[str | int]
    template_type: str  # "dapo" or "skywork"


def get_nested_value(record: dict, path: list[str | int]) -> Any:
    """Extract a value from a nested dict/list using a path.

    Args:
        record: The source record (dict or nested structure)
        path: List of keys/indices to traverse

    Returns:
        The value at the path, or None if not found

    Example:
        >>> get_nested_value({"a": [{"b": "value"}]}, ["a", 0, "b"])
        'value'
    """
    value: Any = record
    for key in path:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int):
            value = value[key] if 0 <= key < len(value) else None
        else:
            return None
    return value


def strip_dapo_prompt(hf_question: str) -> str:
    """Strip DAPO prompt wrapper from question text.

    DAPO questions from HuggingFace are wrapped in a prompt template like:
        "Solve the following math problem ... \\n\\n<question>\\n\\n..."
    This strips the wrapper to get the raw question before applying our template.

    Args:
        hf_question: The raw question text from the HF dataset (with DAPO wrapper)

    Returns:
        Question with DAPO wrapper stripped
    """
    # DAPO wraps questions with a prefix ending in \n\n and suffix starting with \n\n
    # The actual question is between these markers
    # If no wrapper found, return as-is
    parts = hf_question.split("\n\n")
    if len(parts) >= 3:
        # Return the middle parts (the actual question content)
        return "\n\n".join(parts[1:-1])
    return hf_question


def restore_dapo_question(hf_question: str, template: dict) -> str:
    """Restore DAPO question using prefix/suffix from template.

    Strips the DAPO prompt wrapper from the HF question first, then applies
    our template prefix/suffix. Strips trailing newlines from the result.

    DAPO templates have the structure:
    {
        "prefix": "... <some wrapper text>",
        "suffix": "<end wrapper text> ..."
    }

    Args:
        hf_question: The raw question text from the HF dataset
        template: Dict with "prefix" and/or "suffix" keys

    Returns:
        Full question with template applied
    """
    # Strip DAPO wrapper before applying template
    stripped = strip_dapo_prompt(hf_question)
    prefix = template.get("prefix", "")
    suffix = template.get("suffix", "")
    result = f"{prefix}{stripped}{suffix}"
    # Strip trailing newlines (upstream behavior)
    return result.rstrip("\n")


def restore_skywork_question(hf_question: str, template: str) -> str:
    """Restore Skywork question by replacing {question} placeholder.

    Skywork templates have {question} placeholders that need to be replaced
    with the actual question from HuggingFace.

    Args:
        hf_question: The raw question text from the HF dataset
        template: Template string containing {question} placeholder

    Returns:
        Full question with placeholder replaced
    """
    if "{question}" in template:
        return template.replace("{question}", hf_question)
    # Fallback if no placeholder found
    return hf_question


def get_answer(raw_answer: Any) -> Any:
    """Parse answer field, handling JSON-encoded strings.

    Upstream answers may be JSON strings like '"[42]"' or '"{\"key\": \"val\"}"'.
    This parses them and extracts the first element from arrays/objects.

    Args:
        raw_answer: The raw answer value (may be a JSON string)

    Returns:
        Parsed answer value
    """
    if not isinstance(raw_answer, str):
        return raw_answer
    try:
        parsed = json.loads(raw_answer)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        if isinstance(parsed, dict):
            # Return first value
            return next(iter(parsed.values()), parsed)
        return parsed
    except (json.JSONDecodeError, TypeError):
        return raw_answer


@dataclass
class HFPlaceholderResolver:
    """Resolver for HuggingFace placeholder records.

    Pre-loads external HF datasets using ray.data.from_huggingface() for
    scalable distributed loading, then materializes to PyArrow tables
    for efficient row-index access during placeholder resolution.

    Usage:
        >>> resolver = HFPlaceholderResolver.create()
        >>> resolved = resolver.resolve(placeholder_record)
    """

    tables: dict[str, Any]  # dataset name -> PyArrow Table for indexed access
    configs: dict[str, PlaceholderConfig]  # dataset name -> config

    @classmethod
    def create(cls, target_datasets: dict[str, dict] | None = None) -> "HFPlaceholderResolver":
        """Create a resolver with pre-loaded HF datasets via Ray.

        Uses ray.data.from_huggingface() for scalable dataset loading,
        then materializes to PyArrow tables for efficient row-index access.

        Args:
            target_datasets: Optional custom target dataset config.
                            Defaults to TARGET_DATASETS.

        Returns:
            Initialized resolver with loaded datasets
        """
        import pyarrow as pa

        if target_datasets is None:
            target_datasets = TARGET_DATASETS

        tables: dict[str, pa.Table] = {}
        configs: dict[str, PlaceholderConfig] = {}

        for name, cfg in target_datasets.items():
            config = PlaceholderConfig(
                hf_dataset=cfg["hf_dataset"],
                split=cfg["split"],
                question_path=cfg["question_path"],
                answer_path=cfg["answer_path"],
                template_type=cfg["template_type"],
            )
            configs[name] = config

            logger.info(f"Loading HF dataset: {config.hf_dataset} (split: {config.split})")
            try:
                # Load parquet files directly from HuggingFace Hub using huggingface_hub
                # This avoids datasets library internal API compatibility issues
                from huggingface_hub import HfFileSystem

                fs = HfFileSystem()

                # Try multiple patterns since HF datasets have varying structures:
                # 1. data/{split}-*.parquet (e.g., Skywork: data/math-00000-of-00001.parquet)
                # 2. data/*.parquet (e.g., DAPO: data/dapo-math-17k.parquet - single file)
                # 3. {split}/*.parquet (standard split directory)
                # 4. default/{split}/*.parquet (default config)
                patterns = [
                    f"datasets/{config.hf_dataset}/data/{config.split}-*.parquet",
                    f"datasets/{config.hf_dataset}/data/*.parquet",
                    f"datasets/{config.hf_dataset}/{config.split}/*.parquet",
                    f"datasets/{config.hf_dataset}/default/{config.split}/*.parquet",
                ]

                parquet_files = []
                for pattern in patterns:
                    parquet_files = fs.glob(pattern)
                    if parquet_files:
                        logger.info(f"Found parquet files with pattern: {pattern}")
                        break

                if parquet_files:
                    # Read all parquet files and concatenate
                    import pyarrow.parquet as pq

                    tables_list = []
                    for pq_file in parquet_files:
                        with fs.open(pq_file, "rb") as f:
                            table = pq.read_table(f)
                            tables_list.append(table)

                    if tables_list:
                        arrow_table = pa.concat_tables(tables_list)
                        tables[name] = arrow_table
                        logger.info(f"Loaded {len(arrow_table)} rows from {config.hf_dataset}")
                    else:
                        logger.warning(f"No data found in parquet files for {config.hf_dataset}")
                        tables[name] = None
                else:
                    logger.warning(
                        f"No parquet files found for {config.hf_dataset} split={config.split}"
                    )
                    tables[name] = None
            except Exception as e:
                logger.warning(f"Failed to load {config.hf_dataset}: {e}")
                tables[name] = None

        return cls(tables=tables, configs=configs)

    def get_loaded_datasets_info(self) -> list[dict[str, Any]]:
        """Return metadata about loaded external datasets for lineage tracking.

        Returns:
            List of dicts with dataset info (uri, name, split, num_rows, etc.)
            for each successfully loaded external dataset.
        """
        datasets_info = []
        for name, config in self.configs.items():
            table = self.tables.get(name)
            if table is not None:
                datasets_info.append({
                    "uri": f"hf://{config.hf_dataset}",
                    "name": name,
                    "split": config.split,
                    "num_rows": len(table),
                    "source_type": "hf_placeholder",
                })
        return datasets_info

    def resolve(self, record: dict) -> dict | None:
        """Resolve a placeholder record to its full content.

        Args:
            record: Record from source dataset with _hf_placeholder field

        Returns:
            Resolved record with full question/answer and responses_create_params,
            or None if resolution fails
        """
        placeholder = record.get("_hf_placeholder")
        if placeholder is None:
            return None  # Not a placeholder record

        dataset_name = record.get("dataset")
        if dataset_name not in self.configs:
            logger.debug(f"Unknown dataset in placeholder: {dataset_name}")
            return None

        config = self.configs[dataset_name]
        table = self.tables.get(dataset_name)
        if table is None:
            logger.warning(f"Dataset not loaded: {dataset_name}")
            return None

        # Get row index and template from placeholder
        row_idx = placeholder.get("row")
        question_template = placeholder.get("question_template")

        if row_idx is None:
            logger.warning(f"Missing row index in placeholder for {dataset_name}")
            return None

        if row_idx < 0 or row_idx >= len(table):
            logger.warning(
                f"Row index {row_idx} out of bounds for {dataset_name} "
                f"(size: {len(table)})"
            )
            return None

        # Fetch the actual record from PyArrow table
        try:
            # Slice single row and convert to dict
            row_table = table.slice(row_idx, 1)
            row_dict = row_table.to_pydict()
            # Convert from {col: [value]} to {col: value}
            hf_record = {k: v[0] if v else None for k, v in row_dict.items()}
        except Exception as e:
            logger.warning(f"Failed to fetch row {row_idx} from {dataset_name}: {e}")
            return None

        # Extract question and answer using configured paths
        hf_question = get_nested_value(hf_record, config.question_path)
        answer = get_nested_value(hf_record, config.answer_path)

        if hf_question is None:
            logger.warning(
                f"Could not extract question from {dataset_name} row {row_idx} "
                f"using path {config.question_path}"
            )
            return None

        # Restore full question using template
        raw_question = str(hf_question)
        if config.template_type == "dapo":
            if isinstance(question_template, dict):
                full_question = restore_dapo_question(raw_question, question_template)
            else:
                full_question = raw_question
        elif config.template_type == "skywork":
            if isinstance(question_template, str):
                full_question = restore_skywork_question(raw_question, question_template)
            else:
                full_question = raw_question
        else:
            full_question = raw_question

        # Parse answer (handles JSON-encoded strings)
        parsed_answer = get_answer(answer)

        # Build output record — preserve all original fields, remove placeholder
        resolved = dict(record)
        resolved.pop("_hf_placeholder", None)

        # For Skywork, question field is the raw HF question (not template-applied)
        # The template-applied version goes into responses_create_params
        if config.template_type == "skywork":
            resolved["question"] = raw_question
        else:
            resolved["question"] = full_question

        resolved["expected_answer"] = parsed_answer
        resolved["responses_create_params"] = {
            "input": [{"role": "user", "content": full_question}]
        }
        return resolved


def is_placeholder_record(record: dict) -> bool:
    """Check if a record is a placeholder that needs resolution.

    Args:
        record: Record to check

    Returns:
        True if record has _hf_placeholder field
    """
    return "_hf_placeholder" in record


__all__ = [
    "NANO3_TARGET_DATASETS",
    "SUPER3_TARGET_DATASETS",
    "TARGET_DATASETS",
    "PlaceholderConfig",
    "HFPlaceholderResolver",
    "get_nested_value",
    "strip_dapo_prompt",
    "restore_dapo_question",
    "restore_skywork_question",
    "get_answer",
    "is_placeholder_record",
]
