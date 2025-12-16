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

The nvidia/Nemotron-3-Nano-RL-Training-Blend dataset contains placeholder entries
for external datasets (DAPO, Skywork). These placeholders have an `_hf_placeholder`
field containing row indices and question templates that need to be resolved by
fetching the actual data from HuggingFace.

This module provides:
- Configuration for target datasets (HF repos, field paths, template types)
- Helper functions for template restoration (DAPO prefix/suffix, Skywork {question})
- Main resolution function that transforms placeholder records into full records
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Configuration for placeholder datasets that need resolution
# Maps dataset names (as they appear in the blend) to their HF source info
TARGET_DATASETS: dict[str, dict[str, Any]] = {
    "nano_v3_sft_profiled_dapo17k": {
        "hf_dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "split": "train",
        "question_path": ["prompt", 0, "content"],
        "answer_path": ["reward_model", "ground_truth"],
        "template_type": "dapo",
    },
    "nano_v3_sft_profiled_skywork_no_omni": {
        "hf_dataset": "Skywork/Skywork-OR1-RL-Data",
        "split": "train",
        # These paths need to be verified against the actual dataset structure
        "question_path": ["problem"],
        "answer_path": ["answer"],
        "template_type": "skywork",
    },
}


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


def restore_dapo_question(hf_question: str, template: dict) -> str:
    """Restore DAPO question using prefix/suffix from template.

    DAPO templates have the structure:
    {
        "prefix": "... <some wrapper text>",
        "suffix": "<end wrapper text> ..."
    }

    The original question from HF is wrapped with prefix and suffix.

    Args:
        hf_question: The raw question text from the HF dataset
        template: Dict with "prefix" and/or "suffix" keys

    Returns:
        Full question with template applied
    """
    prefix = template.get("prefix", "")
    suffix = template.get("suffix", "")
    return f"{prefix}{hf_question}{suffix}"


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


@dataclass
class HFPlaceholderResolver:
    """Resolver for HuggingFace placeholder records.

    Pre-loads external HF datasets and provides a method to resolve
    placeholder records to their full content.

    Usage:
        >>> resolver = HFPlaceholderResolver.create()
        >>> resolved = resolver.resolve(placeholder_record)
    """

    datasets: dict[str, Any]  # dataset name -> loaded HF Dataset
    configs: dict[str, PlaceholderConfig]  # dataset name -> config

    @classmethod
    def create(cls, target_datasets: dict[str, dict] | None = None) -> "HFPlaceholderResolver":
        """Create a resolver with pre-loaded HF datasets.

        Args:
            target_datasets: Optional custom target dataset config.
                            Defaults to TARGET_DATASETS.

        Returns:
            Initialized resolver with loaded datasets
        """
        from datasets import load_dataset

        if target_datasets is None:
            target_datasets = TARGET_DATASETS

        datasets: dict[str, Any] = {}
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
                datasets[name] = load_dataset(
                    config.hf_dataset,
                    split=config.split,
                )
                logger.info(f"Loaded {len(datasets[name])} rows from {config.hf_dataset}")
            except Exception as e:
                logger.warning(f"Failed to load {config.hf_dataset}: {e}")
                datasets[name] = None

        return cls(datasets=datasets, configs=configs)

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
        hf_dataset = self.datasets.get(dataset_name)
        if hf_dataset is None:
            logger.warning(f"Dataset not loaded: {dataset_name}")
            return None

        # Get row index and template from placeholder
        row_idx = placeholder.get("row")
        question_template = placeholder.get("question_template")

        if row_idx is None:
            logger.warning(f"Missing row index in placeholder for {dataset_name}")
            return None

        if row_idx < 0 or row_idx >= len(hf_dataset):
            logger.warning(
                f"Row index {row_idx} out of bounds for {dataset_name} "
                f"(size: {len(hf_dataset)})"
            )
            return None

        # Fetch the actual record from HuggingFace
        try:
            hf_record = hf_dataset[row_idx]
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
        if config.template_type == "dapo":
            if isinstance(question_template, dict):
                full_question = restore_dapo_question(str(hf_question), question_template)
            else:
                full_question = str(hf_question)
        elif config.template_type == "skywork":
            if isinstance(question_template, str):
                full_question = restore_skywork_question(str(hf_question), question_template)
            else:
                full_question = str(hf_question)
        else:
            full_question = str(hf_question)

        # Build output record
        return {
            "question": full_question,
            "expected_answer": answer,
            "responses_create_params": {
                "input": [{"role": "user", "content": full_question}]
            },
        }


def is_placeholder_record(record: dict) -> bool:
    """Check if a record is a placeholder that needs resolution.

    Args:
        record: Record to check

    Returns:
        True if record has _hf_placeholder field
    """
    return "_hf_placeholder" in record


__all__ = [
    "TARGET_DATASETS",
    "PlaceholderConfig",
    "HFPlaceholderResolver",
    "get_nested_value",
    "restore_dapo_question",
    "restore_skywork_question",
    "is_placeholder_record",
]
