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

"""Shared receipt scanning for pipeline finalize.

All three finalize functions (pretrain, SFT, RL) share the same receipt
scanning infrastructure: find plan_hash, glob receipts, filter+dedup.
Only aggregation differs per pipeline — that stays in the recipes.

Fixes nondeterministic plan_hash discovery: when multiple plan hashes
exist under one dataset (possible when inputs change), the most recently
created plan is selected instead of relying on fs.ls ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fsspec import AbstractFileSystem

from nemotron.data_prep.utils.filesystem import read_json

logger = logging.getLogger(__name__)


@dataclass
class DatasetReceipts:
    """Completed receipts for one dataset, ready for pipeline-specific aggregation."""

    plan_hash: str
    completed: list[dict[str, Any]]
    prefix: str  # "{run_dir}/datasets/{name}/{plan_hash}/shard"


def scan_dataset_receipts(
    run_dir: str,
    dataset_names: list[str],
    fs: AbstractFileSystem,
) -> dict[str, DatasetReceipts]:
    """Scan run directory for completed receipts across all datasets."""
    results: dict[str, DatasetReceipts] = {}
    for name in dataset_names:
        dataset_base = f"{run_dir}/datasets/{name}"
        plan_hash = _find_plan_hash(dataset_base, fs)
        if not plan_hash:
            continue
        receipts_dir = f"{dataset_base}/{plan_hash}/receipts"
        completed = _collect_completed(receipts_dir, plan_hash, fs)
        results[name] = DatasetReceipts(
            plan_hash=plan_hash,
            completed=completed,
            prefix=f"{dataset_base}/{plan_hash}/shard",
        )
    return results


def _find_plan_hash(dataset_base: str, fs: AbstractFileSystem) -> str | None:
    """Find the most recent plan_hash for a dataset.

    When multiple plan hashes exist (e.g. inputs changed between runs),
    selects the most recently created based on plan.json's created_at field.
    ISO 8601 timestamps sort lexicographically, so simple string sort works.
    """
    try:
        subdirs = [p for p in fs.ls(dataset_base) if fs.isdir(p)]
    except Exception:
        return None

    candidates: list[tuple[str, str]] = []
    for subdir in subdirs:
        plan_path = f"{subdir}/plan.json"
        try:
            if not fs.exists(plan_path):
                continue
            plan = read_json(fs, plan_path)
            created_at = plan.get("created_at", "")
            plan_hash = subdir.split("/")[-1]
            candidates.append((created_at, plan_hash))
        except Exception:
            continue

    if not candidates:
        return None
    candidates.sort(reverse=True)  # ISO 8601 sorts lexicographically
    return candidates[0][1]


def _collect_completed(
    receipts_dir: str,
    plan_hash: str,
    fs: AbstractFileSystem,
) -> list[dict[str, Any]]:
    """Collect completed receipts, deduplicating by shard_index."""
    try:
        receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
    except Exception:
        return []

    seen: set[int] = set()
    completed: list[dict[str, Any]] = []

    for path in receipt_files:
        try:
            r = read_json(fs, path)
            if r.get("status") != "completed" or r.get("plan_hash") != plan_hash:
                continue
            shard_index = int(r.get("shard_index", -1))
            if shard_index in seen:
                continue
            seen.add(shard_index)
            completed.append(r)
        except Exception as e:
            logger.warning(f"Failed to parse receipt {path}: {e}")
            continue

    return completed


__all__ = ["DatasetReceipts", "scan_dataset_receipts"]
