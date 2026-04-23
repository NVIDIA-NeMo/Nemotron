# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BYOB stage-1: build a seed table (few-shot MMLU rows + target text) for MCQ generation."""

from __future__ import annotations

import logging
import os
import random
from typing import TYPE_CHECKING

import numpy as np

from nemotron.customization_recipes.data_prep.byob.config import ByobConfig
from nemotron.customization_recipes.data_prep.byob.mcq_dataset import McqByobDataset
from nemotron.customization_recipes.data_prep.byob.seed_config import prepare_byob_config_for_seed

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def prepare_byob_seed(cfg: "DictConfig") -> dict:
    """Sample few-shot examples from a HuggingFace MCQ set and pair with local target text.

    Writes ``{output_dir}/{expt_name}/seed.parquet`` in the same layout as the Speaker
    ``prepare_data`` / :meth:`McqByobDataset.sample_and_dump` pipeline.

    Returns:
        ``{"seed_path", "num_records", "expt_name"}`` with ``seed_path`` to the Parquet file.
    """
    byob_cfg = ByobConfig.from_omegaconf(cfg)
    byob_cfg = prepare_byob_config_for_seed(byob_cfg)
    if byob_cfg.random_seed is not None:
        np.random.seed(byob_cfg.random_seed)
        random.seed(byob_cfg.random_seed)

    dataset = McqByobDataset(byob_cfg)
    seed_df = dataset.sample_and_dump()
    out_path = os.path.join(byob_cfg.output_dir, byob_cfg.expt_name, "seed.parquet")
    logger.info("BYOB seed prep complete: %d rows at %s", len(seed_df), out_path)
    return {
        "seed_path": out_path,
        "num_records": int(len(seed_df)),
        "expt_name": byob_cfg.expt_name,
    }
