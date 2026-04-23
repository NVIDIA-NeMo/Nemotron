# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize and validate configuration for the BYOB MCQ seed preparation stage (ported from Speaker)."""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import replace
from typing import Any

import numpy as np

from nemotron.customization_recipes.data_prep.byob.config import ByobConfig
from nemotron.customization_recipes.data_prep.byob.constants import (
    ALLOWED_HF_DATASETS,
    HF_DATASET_TO_SUBSET,
)
from nemotron.customization_recipes.data_prep.byob.hf_utils import get_subjects

logger = logging.getLogger(__name__)


def _ensure_output_dirs(cfg: ByobConfig) -> None:
    if os.path.exists(cfg.output_dir):
        if not os.access(cfg.output_dir, os.W_OK):
            msg = f"Output directory {cfg.output_dir!r} is not writable"
            raise PermissionError(msg)
        logger.info("Output directory %s exists and is writable", cfg.output_dir)
    else:
        os.makedirs(cfg.output_dir, exist_ok=True)
        logger.info("Created output directory %s", cfg.output_dir)

    tag = os.path.join(cfg.output_dir, cfg.expt_name)
    if os.path.exists(tag):
        logger.warning(
            "Tag %s already exists under %s; seed files may be overwritten.",
            cfg.expt_name,
            cfg.output_dir,
        )
        if not os.access(tag, os.W_OK):
            msg = f"Experiment directory {tag!r} is not writable"
            raise PermissionError(msg)
    else:
        os.makedirs(tag, exist_ok=True)
        logger.info("Created experiment directory %s", tag)


def _validate_target_inputs(cfg: ByobConfig) -> None:
    for subject in cfg.target_source_mapping:
        path = os.path.join(cfg.input_dir, subject)
        is_dir = os.path.isdir(path)
        parquet_path = path + ".parquet"
        is_parquet = os.path.isfile(parquet_path)
        if not (is_dir or is_parquet):
            msg = (
                f"input_dir for target {subject!r} must be {path!r} (dir) or "
                f"{parquet_path!r} (file)"
            )
            raise FileNotFoundError(msg)
        if is_dir:
            if glob.glob(os.path.join(path, "*.txt")) == [] and not is_parquet:
                msg = f"Directory {path!r} must contain at least one .txt file"
                raise ValueError(msg)
        if is_dir and is_parquet:
            logger.warning("Both %s and %s exist; using the directory for .txt files.", path, parquet_path)


def expand_target_source_mapping(cfg: ByobConfig) -> dict[str, Any]:
    """Expand ``target_source_mapping`` to internal weight tensors (ported from Speaker)."""
    tsm: dict[str, Any] = {}
    for target_cfg in cfg.target_source_mapping:
        m = dict(cfg.target_source_mapping[target_cfg])
        if isinstance(m.get("subjects"), list):
            if not m["subjects"]:
                m["subjects"] = list(cfg.source_subjects)
            for src in m["subjects"]:
                if src not in cfg.source_subjects:
                    msg = (
                        f"Source subject {src!r} in target_source_mapping is not in "
                        f"source_subjects: {cfg.source_subjects!r}"
                    )
                    raise ValueError(msg)
            labels = m["subjects"]
            weights = np.ones(len(labels)) / len(labels)
        elif isinstance(m.get("subjects"), dict):
            labels = list(m["subjects"].keys())
            for src in labels:
                if src not in cfg.source_subjects:
                    msg = (
                        f"Source subject {src!r} in target_source_mapping is not in "
                        f"source_subjects: {cfg.source_subjects!r}"
                    )
                    raise ValueError(msg)
            weights = np.array(list(m["subjects"].values()), dtype=float)
            weights = weights / np.sum(weights)
        else:
            msg = f"Invalid `subjects` for target {target_cfg!r}"
            raise TypeError(msg)

        if cfg.metadata_file is None and "tags" in m:
            msg = "`tags` in target_source_mapping require `metadata_file`"
            raise ValueError(msg)
        if cfg.metadata_file is None:
            m.pop("tags", None)

        tags = m.get("tags", [""])
        if isinstance(tags, list):
            tag_labels = [tuple(t.split(",")) for t in tags]
            tag_weights = np.ones(len(tags)) / len(tags)
        elif isinstance(tags, dict):
            tag_labels = [tuple(k.split(",")) for k in tags]
            tag_weights = np.array(list(tags.values()), dtype=float)
            tag_weights = tag_weights / np.sum(tag_weights)
        else:
            msg = f"Invalid `tags` for target {target_cfg!r}"
            raise TypeError(msg)

        tsm[target_cfg] = {
            "source_subjects": labels,
            "source_weights": weights,
            "source_tags": tag_labels,
            "source_tag_weights": tag_weights,
        }
    return tsm


def prepare_byob_config_for_seed(cfg: ByobConfig) -> ByobConfig:
    """Return a config ready for :class:`McqByobDataset` (defaults, mapping expansion, I/O layout)."""
    if cfg.hf_dataset not in ALLOWED_HF_DATASETS:
        msg = f"hf_dataset {cfg.hf_dataset!r} is not allowed. Use one of: {ALLOWED_HF_DATASETS!r}"
        raise ValueError(msg)

    subset = cfg.subset
    if not subset or not str(subset).strip():
        subset = HF_DATASET_TO_SUBSET[cfg.hf_dataset]

    source_subjects = list(cfg.source_subjects) if cfg.source_subjects else []
    if not source_subjects:
        source_subjects = get_subjects(cfg.hf_dataset, subset, cfg.split)

    if not cfg.target_source_mapping:
        msg = "target_source_mapping is empty; list at least one target subject and corpus layout."
        raise ValueError(msg)

    working = replace(cfg, subset=subset, source_subjects=source_subjects)
    _validate_target_inputs(working)
    tsm_plain = expand_target_source_mapping(working)
    _ensure_output_dirs(working)
    return replace(working, target_source_mapping=tsm_plain)


def byob_config_from_omegaconf(omegacfg) -> ByobConfig:
    """Alias for :meth:`ByobConfig.from_omegaconf` (kept for call sites that prefer a function)."""
    return ByobConfig.from_omegaconf(omegacfg)
