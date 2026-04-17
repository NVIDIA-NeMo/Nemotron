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

"""
Synthetic data generation via NVIDIA DataDesigner.

Provides Pydantic conversation schemas and a thin wrapper for running
DataDesigner pipelines with OmegaConf configuration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

_DD_MSG = (
    "data-designer is required for synthetic data generation. "
    "Install with: pip install data-designer"
)

# ---------------------------------------------------------------------------
# Pydantic conversation schemas (ported from Speaker schema.py)
# ---------------------------------------------------------------------------


class FunctionCall(BaseModel):
    """A single function invocation within a tool call."""

    name: str = Field(..., description="Function name")
    arguments: str = Field(..., description="JSON-encoded arguments string")


class ToolCall(BaseModel):
    """An assistant tool call."""

    id: str = Field(..., description="Unique 9-char alphanumeric identifier")
    type: str = Field(default="function", description="Tool call type")
    function: FunctionCall = Field(..., description="Function call details")


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(..., description="user | assistant | tool | system")
    content: Optional[str] = Field(
        default=None, description="Text content (None for pure tool-call messages)"
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls made by the assistant"
    )


class Conversation(BaseModel):
    """A multi-turn conversation."""

    messages: List[Message] = Field(..., description="Ordered list of messages")


class ConversationList(BaseModel):
    """A batch of conversations (used as the LLM structured output format)."""

    conversations: List[Conversation] = Field(
        ..., description="List of generated conversations"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SDGConfig:
    """Configuration for synthetic data generation."""

    output_dir: str = "data/sdg"
    """Directory to write generated datasets."""

    seed_dataset: Optional[str] = None
    """Path to seed CSV/JSONL (DataDesigner SeedConfig)."""

    num_records: int = 100
    """Number of records to generate."""

    system_prompt: str = ""
    """System prompt for the LLM column."""

    user_prompt: str = ""
    """User-facing prompt template (may reference seed columns)."""

    column_name: str = "result"
    """Name of the generated output column."""

    column_type: str = "llm-structured"
    """DataDesigner column type (llm-structured | llm-text | sampler)."""

    output_format: str = "ConversationList"
    """Pydantic model name for structured output (resolved at runtime)."""

    model_configs: list[dict] = field(default_factory=list)
    """List of LLM model config dicts for DataDesigner."""

    model_alias: Optional[str] = None
    """Alias of the model to use for the generated column."""

    domain: Optional[str] = None
    """Target domain (e.g., "medical", "legal") — injected into system prompt."""

    language: Optional[str] = None
    """Target language (e.g., "Korean", "Hindi") — injected into system prompt."""

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "SDGConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(SDGConfig)
        merged = OmegaConf.merge(schema, cfg)
        return SDGConfig(**OmegaConf.to_container(merged, resolve=True))


# ---------------------------------------------------------------------------
# Schema registry for output_format resolution
# ---------------------------------------------------------------------------

_SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "FunctionCall": FunctionCall,
    "ToolCall": ToolCall,
    "Message": Message,
    "Conversation": Conversation,
    "ConversationList": ConversationList,
}


def register_schema(name: str, cls: type[BaseModel]) -> None:
    """Register a custom Pydantic model for use as an SDG output format."""
    _SCHEMA_REGISTRY[name] = cls


def resolve_schema(name: str) -> type[BaseModel]:
    """Look up a Pydantic model class by name."""
    if name not in _SCHEMA_REGISTRY:
        raise KeyError(
            f"Unknown output_format '{name}'. "
            f"Available: {list(_SCHEMA_REGISTRY.keys())}. "
            "Use register_schema() to add custom models."
        )
    return _SCHEMA_REGISTRY[name]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_sdg_pipeline(cfg: SDGConfig) -> "pd.DataFrame":
    """Run a DataDesigner synthetic data generation pipeline.

    Args:
        cfg: SDG configuration.

    Returns:
        pandas DataFrame with generated data.
    """
    try:
        from data_designer.essentials import (
            DataDesigner,
            DataDesignerConfigBuilder,
            SeedConfig,
        )
    except ImportError as exc:
        raise ImportError(_DD_MSG) from exc

    import pandas as pd

    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = str(output_path / "artifacts" / "data_designer")

    designer = DataDesigner(artifact_path=artifact_path)

    # Build model configs from dicts (DataDesigner expects its own config objects)
    model_cfgs = []
    for mc in cfg.model_configs:
        # DataDesigner provides setup_model_config or similar; pass dicts through.
        model_cfgs.append(mc)

    builder = DataDesignerConfigBuilder(model_configs=model_cfgs)

    if cfg.seed_dataset:
        builder.with_seed_dataset(SeedConfig(dataset=cfg.seed_dataset))

    output_format = resolve_schema(cfg.output_format)

    # Build system prompt, incorporating domain/language when provided
    system_prompt = cfg.system_prompt
    if cfg.domain or cfg.language:
        parts = []
        if cfg.language:
            parts.append(cfg.language)
        if cfg.domain:
            parts.append(cfg.domain)
        domain_lang_hint = (
            f"Generate {' '.join(parts)} conversations."
        )
        if system_prompt:
            system_prompt = f"{domain_lang_hint} {system_prompt}"
        else:
            system_prompt = domain_lang_hint

    builder.add_column(
        name=cfg.column_name,
        column_type=cfg.column_type,
        system_prompt=system_prompt,
        prompt=cfg.user_prompt,
        output_format=output_format,
        model_alias=cfg.model_alias,
    )
    builder.validate()

    log.info(
        "Running DataDesigner: records=%d, column=%s, format=%s",
        cfg.num_records,
        cfg.column_name,
        cfg.output_format,
    )

    job_results = designer.create(config_builder=builder, num_records=cfg.num_records)
    dataset = job_results.load_dataset()
    dataset.dropna(inplace=True)

    log.info("SDG complete: %d rows generated", len(dataset))
    return dataset


def generate_synthetic_data(cfg: "DictConfig") -> dict:
    """Orchestrate synthetic data generation from an OmegaConf config.

    This is a convenience wrapper around :func:`run_sdg_pipeline` that
    accepts a raw OmegaConf DictConfig (as produced by the run scripts),
    converts it to an :class:`SDGConfig`, runs the pipeline, persists the
    output, and returns a summary dict.

    Args:
        cfg: OmegaConf DictConfig with SDG parameters.

    Returns:
        Dict with ``output_dir`` and ``num_records`` keys.
    """
    sdg_cfg = SDGConfig.from_omegaconf(cfg)
    dataset = run_sdg_pipeline(sdg_cfg)

    output_path = Path(sdg_cfg.output_dir) / "synthetic_data.jsonl"
    dataset.to_json(str(output_path), orient="records", lines=True)
    log.info("Saved %d records to %s", len(dataset), output_path)

    return {"output_dir": sdg_cfg.output_dir, "num_records": len(dataset)}
