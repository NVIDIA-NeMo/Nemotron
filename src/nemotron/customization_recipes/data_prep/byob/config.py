# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataclass configuration for the BYOB MCQ pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig, OmegaConf


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

    filtering_model_configs: dict = field(
        default_factory=lambda: {"easiness": [], "hallucination": []}
    )
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
        """Materialize a :class:`ByobConfig` from a DictConfig, ignoring unknown top-level keys."""
        from dataclasses import fields

        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            msg = f"Config must be a dict-like structure, got {type(raw)}"
            raise TypeError(msg)
        allowed = {f.name for f in fields(ByobConfig)}
        filtered = {k: v for k, v in raw.items() if k in allowed}
        schema = OmegaConf.structured(ByobConfig)
        merged = OmegaConf.merge(schema, OmegaConf.create(filtered))
        return ByobConfig(**OmegaConf.to_container(merged, resolve=True))
