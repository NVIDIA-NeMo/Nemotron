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

"""Shared utilities for consistent stage name canonicalization.

This module provides a single source of truth for converting cosmos-xenna
stage names into canonical keys used in W&B metrics, Prometheus labels, etc.

Stage names in cosmos-xenna can take various forms:
- Plain: "PlanStage", "DownloadStage", "BinIdxTokenizationStage"
- Prefixed: "Stage 00 - PlanStage", "Stage 02 - BinIdxTokenizationStage"

This module normalizes all variations to consistent snake_case keys.
"""

from __future__ import annotations

import re

# Pattern to strip "Stage NN - " prefix
_STAGE_PREFIX_RE = re.compile(r"^Stage\s+\d+\s*-\s*")


def canonical_stage_id(name: str) -> str:
    """Convert a stage name to a canonical ID for use in metric keys.

    Examples:
        "PlanStage" -> "plan"
        "Stage 00 - PlanStage" -> "plan"
        "BinIdxTokenizationStage" -> "bin_idx_tokenization"
        "Stage 02 - BinIdxTokenizationStage" -> "bin_idx_tokenization"
        "PackedSftParquetStage" -> "packed_sft_parquet"

    Args:
        name: Raw stage name from cosmos-xenna

    Returns:
        Canonical snake_case stage ID (without "stage" suffix)
    """
    name = (name or "").strip()

    # Remove "Stage NN - " prefix if present
    name = _STAGE_PREFIX_RE.sub("", name)

    # Remove "Stage" suffix if present
    name = re.sub(r"Stage$", "", name).strip()

    # Convert CamelCase to snake_case
    # Insert underscore before uppercase letters that follow lowercase letters or digits
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

    # Replace non-alphanumeric with underscores
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)

    # Lowercase and clean up underscores
    name = name.lower().strip("_")
    name = re.sub(r"_+", "_", name)

    return name or "unknown"


# Mapping of canonical stage IDs to short display names for charts
# These are used in W&B chart titles and legends
STAGE_DISPLAY_NAMES: dict[str, str] = {
    "plan": "Plan",
    "sft_plan": "SFT Plan",
    "download": "Download",
    "bin_idx_tokenization": "Tokenize",
    "packed_sft_parquet": "Pack SFT",
}


def get_stage_display_name(canonical_id: str) -> str:
    """Get a human-readable display name for a canonical stage ID.

    Args:
        canonical_id: Canonical stage ID from canonical_stage_id()

    Returns:
        Human-readable name for display in charts
    """
    return STAGE_DISPLAY_NAMES.get(canonical_id, canonical_id.replace("_", " ").title())
