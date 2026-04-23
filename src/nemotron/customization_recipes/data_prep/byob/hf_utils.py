# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face dataset helpers for BYOB seed prep (ported from Speaker)."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

try:
    import datasets
except ImportError:  # pragma: no cover
    datasets = None


def get_metadata(dataset_name: str) -> dict:
    """Return HF datasets-server ``splits`` response for a dataset name."""
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    api_url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_name}"
    response = requests.get(api_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def get_subsets(dataset_name: str) -> list[str]:
    """Return available subset/config names for a dataset."""
    metadata = get_metadata(dataset_name)
    return sorted({row["config"] for row in metadata["splits"]})


def get_subjects(dataset_name: str, subset: str, split: str) -> list[str]:
    """List subject/category values for a supported MCQ benchmark dataset.

    For ``cais/mmlu``, we exclude aggregate configs ``all`` and ``auxiliary_train``.
    """
    if datasets is None:
        raise ImportError("The `datasets` package is required. Install with: pip install datasets")

    if dataset_name == "cais/mmlu":
        subsets = [s for s in get_subsets(dataset_name) if s not in ("all", "auxiliary_train")]
        return sorted(subsets)
    if dataset_name == "TIGER-Lab/MMLU-Pro":
        ds = datasets.load_dataset(dataset_name, subset, split=split)
        return sorted(ds.to_pandas()["category"].unique().tolist())
    if dataset_name == "ai4bharat/MILU":
        ds = datasets.load_dataset(dataset_name, subset, split=split)
        return sorted(ds.to_pandas()["subject"].unique().tolist())
    if dataset_name in ("CohereLabs/Global-MMLU", "CohereLabs/Global-MMLU-Lite"):
        ds = datasets.load_dataset(dataset_name, subset, split=split)
        return sorted(ds.to_pandas()["subject"].unique().tolist())
    if dataset_name == "LinguaLift/IndicMMLU-Pro":
        ds = datasets.load_dataset(dataset_name, subset, split=split)
        return sorted(ds.to_pandas()["category"].unique().tolist())
    if dataset_name == "openai/MMMLU":
        ds = datasets.load_dataset(dataset_name, subset, split=split)
        return sorted(ds.to_pandas()["Subject"].unique().tolist())
    if dataset_name == "sarvamai/mmlu-indic":
        logger.info(
            "``sarvamai/mmlu-indic`` has no per-row subjects; using ``all`` as subject label."
        )
        return ["all"]
    if dataset_name == "Idavidrein/gpqa":
        ds = datasets.load_dataset(dataset_name, subset, split=split)
        return sorted(ds.to_pandas()["Subdomain"].unique().tolist())
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset(dataset_name: str, subset: str, split: str):
    """Load a split from Hugging Face ``datasets``."""
    if datasets is None:
        raise ImportError("The `datasets` package is required. Install with: pip install datasets")
    return datasets.load_dataset(dataset_name, subset, split=split)


def to_plain_mapping(obj: object) -> dict:
    """Convert OmegaConf DictConfig / other mappings to a plain ``dict`` recursively."""
    if not isinstance(obj, Mapping):
        return obj  # type: ignore[return-value]
    return {k: to_plain_mapping(v) if isinstance(v, Mapping) else v for k, v in obj.items()}
