#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/synth/data_designer"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

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

"""Generate synthetic SFT or RL preference data with NeMo Data Designer.

Mirrors the upstream NVIDIA-NeMo/DataDesigner Python SDK: build a
``DataDesignerConfigBuilder`` from a declarative YAML column spec, then call
``client.preview(builder)`` (fast iteration) or ``client.create(builder, …)``
(full dataset).

Two configs ship out of the box:
  - ``default.yaml``  — SFT chat data (sampler ``persona`` × seed ``topic`` +
    LLM-generated ``user_query`` / ``assistant_response``).
  - ``rl_pref.yaml``  — DPO preference data (two LLM-generated responses + an
    LLM judge to label chosen / rejected).

Generation uses a remote inference endpoint, so this step needs no GPUs of its
own — only network access to the configured model service. Customisation lives
entirely in YAML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def build_columns(builder: Any, columns: list[dict[str, Any]], dd: Any) -> None:
    """Translate declarative column specs into typed Data Designer column configs.

    Supported ``type``s:
      - ``category``        — pick uniformly from a fixed list of values.
      - ``seed``            — surface a column from the seed dataset by name.
      - ``llm_text``        — generate free text via an LLM.
      - ``llm_structured``  — generate structured JSON via an LLM (provide ``output_format``).
      - ``llm_judge``       — alias for ``llm_structured``.
    """
    for spec in columns:
        kind = spec["type"]
        name = spec["name"]

        if kind == "category":
            builder.add_column(
                dd.SamplerColumnConfig(
                    name=name,
                    sampler_type=dd.SamplerType.CATEGORY,
                    params=dd.CategorySamplerParams(values=spec["values"]),
                )
            )

        elif kind == "seed":
            # The column name must match the field in the seed dataset.
            builder.add_column(dd.SeedDatasetColumnConfig(name=name))

        elif kind == "llm_text":
            builder.add_column(
                dd.LLMTextColumnConfig(
                    name=name,
                    model_alias=spec.get("model_alias", "nvidia-text"),
                    prompt=spec["prompt"],
                )
            )

        elif kind in ("llm_structured", "llm_judge"):
            builder.add_column(
                dd.LLMStructuredColumnConfig(
                    name=name,
                    model_alias=spec.get("model_alias", "nvidia-text"),
                    prompt=spec["prompt"],
                    output_format=spec["output_format"],
                )
            )

        else:
            raise ValueError(f"Unknown column type: {kind!r}")


def project_records(records: list[dict[str, Any]], projection: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Project Data Designer records into training-ready JSONL schemas."""
    if not projection:
        return records

    kind = projection.get("type")
    if kind == "structured_messages":
        source_field = projection.get("source_field", "conversation")
        messages_field = projection.get("messages_field", "messages")
        tools_field = projection.get("tools_field", "tools")
        metadata_fields = projection.get("metadata_fields") or []
        projected = []
        for record in records:
            source = record[source_field]
            if isinstance(source, str):
                source = json.loads(source)
            if not isinstance(source, dict):
                raise ValueError(f"{source_field!r} must be a mapping or JSON object string")

            item = {"messages": source[messages_field]}
            if tools_field in source:
                item["tools"] = source[tools_field]
            for field in metadata_fields:
                if field in record:
                    item[field] = record[field]
            projected.append(item)
        return projected

    if kind == "openai_messages":
        user_field = projection.get("user_field", "user_query")
        assistant_field = projection.get("assistant_field", "assistant_response")
        metadata_fields = projection.get("metadata_fields") or []
        projected = []
        for record in records:
            item = {
                "messages": [
                    {"role": "user", "content": record[user_field]},
                    {"role": "assistant", "content": record[assistant_field]},
                ]
            }
            for field in metadata_fields:
                if field in record:
                    item[field] = record[field]
            projected.append(item)
        return projected

    if kind == "dpo_preference":
        prompt_field = projection.get("prompt_field", "prompt")
        response_a_field = projection.get("response_a_field", "response_a")
        response_b_field = projection.get("response_b_field", "response_b")
        judge_field = projection.get("judge_field", "judge")
        winner_field = projection.get("winner_field", "winner")
        projected = []
        for record in records:
            judge = record.get(judge_field)
            if isinstance(judge, str):
                judge = json.loads(judge)
            if not isinstance(judge, dict):
                raise ValueError(f"{judge_field!r} must be a mapping or JSON object string")

            winner = str(judge.get(winner_field, "")).upper()
            if winner == "A":
                chosen = record[response_a_field]
                rejected = record[response_b_field]
            elif winner == "B":
                chosen = record[response_b_field]
                rejected = record[response_a_field]
            else:
                raise ValueError(f"Unsupported preference winner {winner!r}; expected 'A' or 'B'")

            projected.append(
                {
                    "prompt": record[prompt_field],
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        return projected

    raise ValueError(f"Unknown output_projection type: {kind!r}")


def main() -> None:
    config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    raw = apply_hydra_overrides(load_omegaconf_yaml(config_path), cli_overrides)
    cfg = OmegaConf.to_container(raw, resolve=True)

    columns = cfg.get("columns")
    if not columns:
        raise ValueError(f"{config_path}: config must declare a non-empty `columns:` list")

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deferred imports keep the module importable on dev hosts without
    # data_designer installed.
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    builder = dd.DataDesignerConfigBuilder()

    # Models — translate the YAML `models:` list into typed ModelConfig objects.
    # The builder ships with default model aliases; replace them when the YAML
    # declares the same alias so our endpoint / parameters win.
    for spec in cfg.get("models") or []:
        alias = spec["alias"]
        try:
            builder.delete_model_config(alias)
        except Exception:
            pass  # alias not yet registered — fine, just add it.

        params = spec.get("inference_parameters") or {}
        builder.add_model_config(
            dd.ModelConfig(
                alias=alias,
                model=spec["model"],
                provider=spec.get("provider"),
                skip_health_check=spec.get("skip_health_check", False),
                inference_parameters=dd.ChatCompletionInferenceParams(**params),
            )
        )

    seed = cfg.get("seed_dataset")
    if seed:
        strategy_name = seed.get("strategy", "shuffle").upper()
        builder.with_seed_dataset(
            dd.LocalFileSeedSource(path=seed["path"]),
            sampling_strategy=dd.SamplingStrategy[strategy_name],
        )

    build_columns(builder, columns, dd)

    client = DataDesigner()

    if cfg.get("preview", False):
        result = client.preview(builder)
        records = result.dataset.to_pandas().to_dict(orient="records")
        verb = "Preview"
    else:
        job = client.create(builder, num_records=cfg["num_records"])
        records = job.dataset.to_pandas().to_dict(orient="records")
        verb = "Generated"

    records = project_records(records, cfg.get("output_projection"))

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"{verb} {len(records)} records → {output_path}")


if __name__ == "__main__":
    main()
