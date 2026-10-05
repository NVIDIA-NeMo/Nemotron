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

"""Resolve and normalize oracle-pack artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    load_pack,
    project_model_facing_tools,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_oracle_pack(config: BfclConfig) -> LoadedPack:
    """Resolve pack, normalize artifacts, write stage_cache/."""
    pack = load_pack(config)
    cache = stage_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)

    tools_internal = pack.tools
    tools_model = project_model_facing_tools(tools_internal)

    _write_text_atomic(
        cache / "tools_normalized_internal.json",
        json.dumps(tools_internal, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        cache / "tools_normalized.json",
        json.dumps(tools_model, indent=2, sort_keys=True) + "\n",
    )

    fixtures_payload: dict[str, Any] = pack.fixtures or {}
    _write_text_atomic(
        cache / "fixtures_normalized.json",
        json.dumps(fixtures_payload, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        cache / "task_templates_normalized.yaml",
        yaml.safe_dump(pack.templates, sort_keys=False, allow_unicode=True),
    )
    _write_text_atomic(
        cache / "validation_cases_normalized.yaml",
        yaml.safe_dump(pack.validation_cases, sort_keys=False, allow_unicode=True),
    )
    _write_text_atomic(
        cache / "pack_manifest.json",
        json.dumps(pack.manifest, indent=2, sort_keys=True, default=str) + "\n",
    )
    if pack.held_out is not None:
        _write_text_atomic(
            cache / "held_out_normalized.json",
            json.dumps(pack.held_out, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
    else:
        (cache / "held_out_normalized.json").unlink(missing_ok=True)
    _write_text_atomic(
        cache / "pack_paths.json",
        json.dumps(
            {
                "pack_root": str(pack.paths.pack_root),
                "manifest_path": str(pack.paths.manifest_path),
                "tools_path": str(pack.paths.tools_path),
                "fixtures_path": str(pack.paths.fixtures_path) if pack.paths.fixtures_path else None,
                "templates_path": str(pack.paths.templates_path),
                "assertions_path": str(pack.paths.assertions_path),
                "validation_cases_path": str(pack.paths.validation_cases_path),
                "system_prompt_path": (
                    str(pack.paths.system_prompt_path)
                    if pack.paths.system_prompt_path
                    else None
                ),
                "backend_path": str(pack.paths.backend_path) if pack.paths.backend_path else None,
                "endpoint_config_path": (
                    str(pack.paths.endpoint_config_path)
                    if pack.paths.endpoint_config_path
                    else None
                ),
                "endpoint_ca_bundle_path": (
                    str(pack.paths.endpoint_ca_bundle_path)
                    if pack.paths.endpoint_ca_bundle_path
                    else None
                ),
                "held_out_path": (
                    str(pack.paths.held_out_path)
                    if pack.paths.held_out_path
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    logger.info("BFCL prepare wrote normalized pack artifacts to %s", cache)
    return pack
