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

"""Thin adapter: bridges OmegaConf configs to nemotron.data_prep API.

Translates customization-recipe OmegaConf/YAML configurations into
``nemotron.data_prep`` objects and delegates all heavy lifting
(tokenization, packing, bin/idx writing, chat-template application,
thinking-token handling) to the production ``nemotron.data_prep``
pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclasses (OmegaConf interface for YAML configs)
# ---------------------------------------------------------------------------


@dataclass
class CPTConfig:
    """Configuration for Continued Pre-Training data preparation."""

    output_dir: str = "data/cpt"
    input_path: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_subset: Optional[str] = None
    hf_split: str = "train"
    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    text_field: str = "text"
    num_shards: int = 1
    max_samples: Optional[int] = None
    train_ratio: float = 0.90
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    add_bos: bool = False
    add_eos: bool = True
    min_doc_chars: Optional[int] = None
    max_doc_tokens: Optional[int] = None
    seed: int = 42
    batch_size: int = 1000
    recursive: bool = True

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "CPTConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(CPTConfig)
        merged = OmegaConf.merge(schema, cfg)
        return CPTConfig(**OmegaConf.to_container(merged, resolve=True))

    def to_data_blend(self):
        """Convert input source to a ``nemotron.data_prep.DataBlend``."""
        from nemotron.data_prep.blend import DataBlend, Dataset

        if self.hf_dataset:
            path = f"hf://{self.hf_dataset}"
            name = self.hf_dataset.replace("/", "_")
            ds = Dataset(
                name=name,
                path=path,
                split=self.hf_split,
                subset=self.hf_subset,
                text_field=self.text_field,
            )
        else:
            ds = Dataset(
                name=Path(self.input_path).stem,
                path=self.input_path,
                text_field=self.text_field,
            )
        return DataBlend(datasets=[ds])

    def to_tokenizer_config(self):
        """Convert to ``nemotron.data_prep.TokenizerConfig``."""
        from nemotron.data_prep.config import TokenizerConfig

        return TokenizerConfig(
            model=self.tokenizer_model,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
        )


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning data preparation."""

    output_dir: str = "data/sft"
    input_path: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_subset: Optional[str] = None
    hf_split: str = "train"
    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    pack_size: int = 4096
    train_ratio: float = 0.9
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    messages_field: str = "messages"
    conversations_field: Optional[str] = None
    seed: int = 42
    add_generation_prompt: bool = False
    recursive: bool = True
    packing_algorithm: str = "first_fit_decreasing"
    max_samples: Optional[int] = None
    enable_thinking: bool = False
    truncate_history_thinking: bool = True
    thinking_start_token: str = "<think>"
    thinking_end_token: str = "</think>"

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "SFTConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(SFTConfig)
        merged = OmegaConf.merge(schema, cfg)
        return SFTConfig(**OmegaConf.to_container(merged, resolve=True))

    def to_data_blend(self):
        """Convert input source to a ``nemotron.data_prep.DataBlend``."""
        from nemotron.data_prep.blend import DataBlend, Dataset

        if self.hf_dataset:
            path = f"hf://{self.hf_dataset}"
            name = self.hf_dataset.replace("/", "_")
            ds = Dataset(
                name=name,
                path=path,
                split=self.hf_split,
                subset=self.hf_subset,
            )
        else:
            ds = Dataset(
                name=Path(self.input_path).stem,
                path=self.input_path,
            )
        return DataBlend(datasets=[ds])

    def to_tokenizer_config(self):
        """Convert to ``nemotron.data_prep.TokenizerConfig``."""
        from nemotron.data_prep.config import TokenizerConfig

        return TokenizerConfig(model=self.tokenizer_model)


# ---------------------------------------------------------------------------
# Main entry points -- delegate to nemotron.data_prep pipelines
# ---------------------------------------------------------------------------


def prepare_cpt_data(cfg) -> dict:
    """Prepare data for Continued Pre-Training (Megatron bin/idx format).

    Accepts either a :class:`CPTConfig` dataclass or a raw
    ``omegaconf.DictConfig`` (which is auto-converted via
    ``CPTConfig.from_omegaconf``).

    Returns:
        Dict with ``output_dir`` and ``stats`` from the pipeline result.
    """
    from nemotron.data_prep.api import run_pretrain_pipeline

    if not isinstance(cfg, CPTConfig):
        cfg = CPTConfig.from_omegaconf(cfg)

    if cfg.hf_dataset and cfg.input_path:
        raise ValueError("Specify input_path or hf_dataset, not both.")
    if not cfg.hf_dataset and not cfg.input_path:
        raise ValueError("Must specify input_path or hf_dataset.")

    blend = cfg.to_data_blend()
    tokenizer = cfg.to_tokenizer_config()

    result = run_pretrain_pipeline(
        blend=blend,
        output_dir=cfg.output_dir,
        tokenizer=tokenizer,
        num_shards=cfg.num_shards,
        text_field_default=cfg.text_field,
        min_doc_chars=cfg.min_doc_chars,
        max_doc_tokens=cfg.max_doc_tokens,
        max_rows=cfg.max_samples,
        sample_seed=cfg.seed,
    )

    log.info("CPT prep complete: %s", cfg.output_dir)
    return {
        "output_dir": str(result.output_dir),
        "data_paths": result.data_paths,
        "stats": result.dataset_stats,
    }


def prepare_sft_data(cfg) -> dict:
    """Prepare data for Supervised Fine-Tuning (packed Parquet format).

    Accepts either a :class:`SFTConfig` dataclass or a raw
    ``omegaconf.DictConfig`` (which is auto-converted via
    ``SFTConfig.from_omegaconf``).

    Thinking-token support (``enable_thinking``, history truncation) is
    handled by the production ``nemotron.data_prep`` chat-template
    pipeline (see ``nemotron.data_prep.core.chat_template`` and the
    ``nano3.jinja`` template).

    Returns:
        Dict with ``output_dir`` and ``stats`` from the pipeline result.
    """
    from nemotron.data_prep.api import run_sft_pipeline

    if not isinstance(cfg, SFTConfig):
        cfg = SFTConfig.from_omegaconf(cfg)

    if cfg.hf_dataset and cfg.input_path:
        raise ValueError("Specify input_path or hf_dataset, not both.")
    if not cfg.hf_dataset and not cfg.input_path:
        raise ValueError("Must specify input_path or hf_dataset.")

    blend = cfg.to_data_blend()
    tokenizer = cfg.to_tokenizer_config()

    # Map the packing algorithm; nemotron.data_prep uses "first_fit_shuffle"
    # as default but customization configs may specify "first_fit_decreasing".
    algorithm = cfg.packing_algorithm

    # Use "nano3" chat template when thinking is enabled, which natively
    # supports enable_thinking and truncate_history_thinking via the
    # nano3.jinja template in nemotron.data_prep.
    chat_template = "nano3" if cfg.enable_thinking else None

    result = run_sft_pipeline(
        blend=blend,
        output_dir=cfg.output_dir,
        tokenizer=tokenizer,
        num_shards=1,
        pack_size=cfg.pack_size,
        algorithm=algorithm,
        messages_field_default=cfg.messages_field,
        max_rows=cfg.max_samples,
        sample_seed=cfg.seed,
        seed=cfg.seed,
        chat_template=chat_template,
    )

    log.info("SFT prep complete: %s", cfg.output_dir)
    return {
        "output_dir": str(result.output_dir),
        "data_paths": result.data_paths,
        "stats": result.dataset_stats,
    }
