# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalize style references and create the profile used by surface generation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
    request_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.response_model import (
    ReferenceProfileResult,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    REFERENCE_SAMPLES,
    reference_samples_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)

PROFILE_VERSION = "1.0"
PROFILE_PROMPT_VERSION = "bfcl-reference-profile-v1"
PROFILE_SYSTEM_PROMPT = """You analyze conversation style only.
Return concise style_hints and avoid rules grounded in the supplied references.
Never infer tools, arguments, expected results, backend state, or assertions."""
PROFILE_PROMPT = """Analyze this canonical JSON payload of style-only conversation samples:
{{ model_input }}
Describe reusable surface-writing style. Do not repeat identifiers or facts from examples."""
ProfileRunner = Callable[..., dict[str, dict[str, Any]]]


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_samples(config: BfclConfig) -> list[dict[str, Any]]:
    reference = config.reference_benchmark
    if reference is None:
        return []
    rows: list[dict[str, Any]] = []
    for line in reference.samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                # BCP-47 language tags are case-insensitive. Canonicalizing here also
                # prevents harmless YAML/JSON whitespace from splitting one profile.
                "language": str(sample["language"]).strip().casefold(),
                "messages": canonical_json(sample["messages"]),
                "tags": [str(tag) for tag in sample.get("tags") or []],
                "source_hash": reference.content_hash,
            }
        )
    return rows


def _clean_rules(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"reference profile response {field} must be a list")
    rules: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(
                f"reference profile response {field} entries must be non-empty strings"
            )
        cleaned = item.strip()
        if cleaned not in rules:
            rules.append(cleaned)
    return rules


def run_reference_profile(
    config: BfclConfig,
    *,
    model_runner: ProfileRunner | None = None,
) -> dict[str, Any]:
    """Write normalized samples and resolve a cached style-only profile."""
    role = (config.lineage.roles or {}).get("profile")
    enabled = bool(role and role.enabled)
    cache = stage_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    sample_rows = _normalize_samples(config)
    write_stage_table(
        cache / REFERENCE_SAMPLES,
        sample_rows,
        reference_samples_schema(),
    )
    profile = {
        "profile_version": PROFILE_VERSION,
        "enabled": enabled,
        "status": "disabled",
        "languages": [],
        "sample_count": len(sample_rows),
        "style_hints": [],
        "avoid": [],
        "reference_content_hash": (
            config.reference_benchmark.content_hash
            if config.reference_benchmark is not None
            else None
        ),
        "profile_model_canonical": None,
        "prompt_hash": None,
        "output_hash": None,
    }
    if enabled:
        assert role is not None and role.model_config is not None
        model_config = role.model_config
        canonical_id = str(model_config["canonical_id"]).strip().lower()
        languages = sorted({row["language"] for row in sample_rows})
        if len(languages) != 1:
            raise ValueError(
                "an enabled reference profile requires samples in exactly one language; "
                f"found {', '.join(languages)}"
            )
        model_input = {
            "profile_version": PROFILE_VERSION,
            "samples": [
                {
                    "language": row["language"],
                    "messages": json.loads(row["messages"]),
                    "tags": row["tags"],
                }
                for row in sample_rows
            ],
        }
        input_json = canonical_json(model_input)
        input_hash = _sha256(input_json)
        prompt_hash = _sha256(
            PROFILE_PROMPT_VERSION + "\n" + PROFILE_SYSTEM_PROMPT + "\n" + PROFILE_PROMPT
        )
        key = request_hash(
            model_canonical=canonical_id,
            prompt_hash=prompt_hash,
            model_input=model_input,
            inference_parameters=dict(model_config.get("inference_parameters") or {}),
            output_schema=ReferenceProfileResult.model_json_schema(),
            seed=int(config.random_seed or 0),
        )
        io_cache = ImmutableModelIOCache(
            cache / "reference_profile_io_cache.jsonl"
        )
        cached = io_cache.get(key)
        response = cached
        if response is None:
            if model_runner is None:
                from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_runner import (
                    run_structured_model,
                )

                model_runner = run_structured_model
            responses = model_runner(
                config,
                stage_name="reference_profile",
                model_config=model_config,
                requests=[{"request_id": key, "model_input": input_json}],
                system_prompt=PROFILE_SYSTEM_PROMPT,
                prompt=PROFILE_PROMPT,
                output_format=ReferenceProfileResult,
            )
            response = responses[key]
        # Validate before writing: the cache is immutable, so a malformed response stored
        # here would fail every later run with no way back other than deleting the cache.
        style_hints = _clean_rules(response.get("style_hints"), "style_hints")
        avoid = _clean_rules(response.get("avoid", []), "avoid")
        if cached is None:
            io_cache.put(
                key,
                response,
                model_canonical=canonical_id,
                input_hash=input_hash,
            )
        profile.update(
            {
                "status": "completed",
                "languages": languages,
                "style_hints": style_hints,
                "avoid": avoid,
                "profile_model_canonical": canonical_id,
                "prompt_hash": prompt_hash,
            }
        )

    profile["output_hash"] = _sha256(
        canonical_json({key: value for key, value in profile.items() if key != "output_hash"})
    )
    path = cache / "reference_profile.json"
    _write_json_atomic(path, profile)
    logger.info("BFCL reference profile wrote %s", path)
    return profile
