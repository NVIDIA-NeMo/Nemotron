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

"""Small Data Designer adapter for BFCL structured model calls."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig


def read_structured_responses(dataset_path: Path) -> list[dict[str, Any]]:
    """Read the rows a structured column wrote, without pandas' Arrow conversion.

    A struct field that is null in every row — a trace predicate names no argument, a
    draft blocked on nothing has an empty list — becomes a null-typed child that pandas'
    Arrow backend sizes to one element instead of the struct's length, so iterating the
    DataFrame Data Designer returns raises instead of yielding the rows. The parquet
    itself is well formed, which is why this reads the file rather than that frame.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_path, columns=["request_id", "response"])
    return table.to_pylist()


def run_structured_model(
    config: BfclConfig,
    *,
    stage_name: str,
    model_config: dict[str, Any],
    requests: Sequence[dict[str, str]],
    system_prompt: str,
    prompt: str,
    output_format: type[BaseModel],
) -> dict[str, dict[str, Any]]:
    """Execute one structured column and return responses by request id."""
    if not requests:
        return {}

    import pandas as pd
    from data_designer.config import DataDesignerConfigBuilder, LocalFileSeedSource
    from data_designer.interface import DataDesigner

    from nemotron.steps.byob.runtime.data_designer_utils import setup_model_config

    rows = list(requests)
    request_ids = [row["request_id"] for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"{stage_name} received duplicate request ids")

    run_dir = Path(config.output_dir) / config.expt_name
    temp_dir = run_dir / "artifacts" / "model_seeds"
    temp_dir.mkdir(parents=True, exist_ok=True)
    seed_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f"{stage_name}-",
            dir=temp_dir,
            delete=False,
            encoding="utf-8",
        ) as handle:
            seed_path = Path(handle.name)
            pd.DataFrame(rows).to_csv(handle, index=False)

        builder = DataDesignerConfigBuilder(
            model_configs=[setup_model_config(model_config)]
        )
        builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
        builder.add_column(
            name="response",
            column_type="llm-structured",
            system_prompt=system_prompt,
            prompt=prompt,
            output_format=output_format,
            model_alias=str(model_config["alias"]),
        )
        designer = DataDesigner(
            artifact_path=str(run_dir / "artifacts" / "data_designer")
        )
        designer.validate(builder)
        created = designer.create(
            config_builder=builder,
            num_records=len(rows),
        )
        records = read_structured_responses(
            created.artifact_storage.final_dataset_path
        )
    finally:
        if seed_path is not None:
            seed_path.unlink(missing_ok=True)

    responses: dict[str, dict[str, Any]] = {}
    for row in records:
        request_id = row.get("request_id")
        response = row.get("response")
        if isinstance(response, BaseModel):
            response = response.model_dump()
        if not isinstance(request_id, str) or not isinstance(response, dict):
            raise RuntimeError(
                f"{stage_name} returned a row without request_id/structured response"
            )
        responses[request_id] = response
    missing = sorted(set(request_ids) - set(responses))
    if missing:
        raise RuntimeError(
            f"{stage_name} returned no response for: {', '.join(missing)}"
        )
    return responses
