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

"""One strict loader for reviewed companion ``tools.json`` artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    validate_tool_definition,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    CertificationRefusalCode,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    ToolEvidence,
    UntrustedText,
)

DEFAULT_MAX_CATALOG_BYTES = 10 * 1024 * 1024


class ReviewedCatalogError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        try:
            self.code = CertificationRefusalCode(code).value
        except ValueError as exc:
            raise ValueError(f"unknown reviewed catalog refusal code {code!r}") from exc
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


@dataclass(frozen=True)
class ReviewedToolCatalog:
    tools: tuple[ToolEvidence, ...]
    digest: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tool.published_name for tool in self.tools)


def load_reviewed_tool_catalog(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_CATALOG_BYTES,
) -> ReviewedToolCatalog:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        if path.stat().st_size > max_bytes:
            raise ReviewedCatalogError(
                "reviewed_schema_too_large",
                f"tools.json exceeds {max_bytes} bytes",
            )
    except OSError as exc:
        raise ReviewedCatalogError(
            "reviewed_schema_missing",
            "cannot stat reviewed tools.json",
        ) from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReviewedCatalogError(
                    "reviewed_schema_invalid",
                    f"tools.json repeats JSON key {key!r}",
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except ReviewedCatalogError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewedCatalogError(
            "reviewed_schema_invalid",
            f"cannot load tools.json: {type(exc).__name__}",
        ) from exc
    if not isinstance(document, list) or not document:
        raise ReviewedCatalogError(
            "reviewed_schema_invalid",
            "tools.json must be a non-empty array",
        )

    indexed: dict[str, ToolEvidence] = {}
    canonical: list[dict[str, Any]] = []
    for index, tool in enumerate(document):
        failures = validate_tool_definition(tool)
        if failures:
            raise ReviewedCatalogError(
                "reviewed_schema_invalid",
                f"tools.json[{index}] is invalid: {sha256_json(failures)}",
            )
        assert isinstance(tool, dict)
        function = tool["function"]
        assert isinstance(function, dict)
        if unknown := sorted(
            set(function) - {"description", "name", "parameters", "strict"}
        ):
            raise ReviewedCatalogError(
                "reviewed_schema_invalid",
                f"tools.json[{index}].function has unknown fields: {', '.join(unknown)}",
            )
        if unknown := sorted(
            set(tool)
            - {
                "function",
                "type",
                "x-mutates",
                "x-requires-confirmation",
            }
        ):
            raise ReviewedCatalogError(
                "reviewed_schema_invalid",
                f"tools.json[{index}] has unknown fields: {', '.join(unknown)}",
            )
        for annotation in ("x-mutates", "x-requires-confirmation"):
            if annotation in tool and not isinstance(tool[annotation], bool):
                raise ReviewedCatalogError(
                    "reviewed_schema_invalid",
                    f"tools.json[{index}].{annotation} must be boolean",
                )
        name = str(function["name"])
        if name in indexed:
            raise ReviewedCatalogError(
                "reviewed_schema_invalid",
                f"tools.json repeats function name {name!r}",
            )
        evidence = ToolEvidence(
            published_name=name,
            source_name=name,
            description=UntrustedText(
                untrusted_text=str(function.get("description", ""))
            ),
            parameter_schema=dict(function["parameters"]),
            output_schema=None,
            annotations=None,
            mutates=tool.get("x-mutates", False),
            requires_confirmation=tool.get("x-requires-confirmation", False),
            raw_digest=sha256_json(tool),
        )
        indexed[name] = evidence
        canonical.append(tool)
    ordered_names = sorted(indexed)
    return ReviewedToolCatalog(
        tools=tuple(indexed[name] for name in ordered_names),
        digest=sha256_json(
            sorted(canonical, key=lambda item: item["function"]["name"])
        ),
    )
