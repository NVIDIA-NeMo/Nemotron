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

"""Public API facade for data preparation pipelines.

This module provides the stable public API for data preparation, including:
- Recipe entry points (run_pretrain_pipeline, run_sft_pipeline)
- Compatibility shims for legacy APIs (run_data_prep)

Supported Pipelines:
    - run_pretrain_pipeline: Tokenize to Megatron bin/idx format
    - run_sft_pipeline: Chat SFT to packed Parquet format

Usage:
    from nemotron.data_prep.api import run_pretrain_pipeline, run_sft_pipeline
    from nemotron.data_prep import DataBlend

    # Pretrain pipeline
    blend = DataBlend.load("pretrain_blend.json")
    result = run_pretrain_pipeline(
        blend=blend,
        output_dir="/output",
        tokenizer="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        num_shards=128,
    )

    # SFT pipeline
    blend = DataBlend.load("sft_blend.json")
    result = run_sft_pipeline(
        blend=blend,
        output_dir="/output",
        tokenizer="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        num_shards=64,
        chat_template="nano3",
    )

Legacy API:
    The `run_data_prep` function provides a compatibility shim that dispatches
    to the appropriate pipeline based on config.output.format. This is provided
    for backward compatibility with documentation examples; new code should use
    the recipe entry points directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NoReturn

from nemotron.data_prep.config import (
    BinIdxOutputConfig,
    ChatSftOutputConfig,
    FormatResult,
    JsonlOutputConfig,
    PackedOutputConfig,
    PipelineConfig,
    TokenizerConfig,
)

# Re-export recipe entry points as the primary public API
from nemotron.data_prep.recipes.pretrain import run_pretrain_pipeline
from nemotron.data_prep.recipes.sft import run_sft_pipeline

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend

__all__ = [
    # Primary API - recipe entry points
    "run_pretrain_pipeline",
    "run_sft_pipeline",
    # Compatibility shim
    "run_data_prep",
    # Legacy stub (raises NotImplementedError)
    "last_mile_process",
]


def run_data_prep(
    config: PipelineConfig,
    *,
    blend: "DataBlend",
) -> FormatResult:
    """Run data preparation pipeline based on output format configuration.

    This is a compatibility shim that dispatches to the appropriate recipe
    based on config.output.format. New code should use run_pretrain_pipeline
    or run_sft_pipeline directly.

    Supported formats:
        - BinIdxOutputConfig: Dispatches to run_pretrain_pipeline
        - ChatSftOutputConfig (parquet): Dispatches to run_sft_pipeline

    Args:
        config: Pipeline configuration with output format specification.
        blend: DataBlend specifying input datasets.

    Returns:
        FormatResult with run metadata, data paths, and statistics.

    Raises:
        ValueError: If tokenizer is required but not provided.
        NotImplementedError: If output format is not supported by this shim.

    Example:
        from nemotron.data_prep import DataBlend, PipelineConfig
        from nemotron.data_prep.config import OutputConfig, BinIdxOutputConfig, TokenizerConfig
        from nemotron.data_prep.api import run_data_prep

        blend = DataBlend.load("blend.json")
        config = PipelineConfig(
            tokenizer=TokenizerConfig(model="nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
            output=OutputConfig(
                dir=Path("./output"),
                format=BinIdxOutputConfig(num_shards=64),
            ),
        )
        result = run_data_prep(config, blend=blend)
    """
    output_format = config.output.format

    if isinstance(output_format, BinIdxOutputConfig):
        return _dispatch_binidx(config, blend)
    elif isinstance(output_format, ChatSftOutputConfig):
        return _dispatch_chat_sft(config, blend)
    elif isinstance(output_format, JsonlOutputConfig):
        raise NotImplementedError(
            "JSONL output is not supported via run_data_prep shim. "
            "Use the stage-specific data_prep.py scripts in nemotron/recipes/nano3/stage2_rl/ "
            "or call process_jsonl_shard_core directly for JSONL processing."
        )
    elif isinstance(output_format, PackedOutputConfig):
        raise NotImplementedError(
            "Packed .npy output is not supported via run_data_prep shim. "
            "Use ChatSftOutputConfig with packed_storage='parquet' for packed SFT data, "
            "or implement a custom pipeline using the packing modules."
        )
    else:
        raise NotImplementedError(
            f"Unsupported output format: {type(output_format).__name__}. "
            "Supported formats: BinIdxOutputConfig, ChatSftOutputConfig (parquet only)."
        )


def _dispatch_binidx(config: PipelineConfig, blend: "DataBlend") -> FormatResult:
    """Dispatch to pretrain pipeline for binidx format."""
    if config.tokenizer is None:
        raise ValueError(
            "TokenizerConfig is required for BinIdxOutputConfig. "
            "Set config.tokenizer = TokenizerConfig(model='...')"
        )

    output_format = config.output.format
    assert isinstance(output_format, BinIdxOutputConfig)

    # Determine num_shards from format config or deprecated output config
    num_shards = output_format.num_shards or config.output.num_shards
    if num_shards is None:
        raise ValueError(
            "num_shards is required for binidx format. "
            "Set config.output.format = BinIdxOutputConfig(num_shards=N) or "
            "config.output.num_shards = N"
        )

    # Determine dtype from format config or deprecated output config
    dtype = output_format.dtype
    if config.output.dtype is not None:
        dtype = config.output.dtype

    return run_pretrain_pipeline(
        blend=blend,
        output_dir=config.output.dir,
        tokenizer=config.tokenizer,
        num_shards=num_shards,
        dtype=dtype,
        min_doc_chars=config.output.min_doc_chars,
        max_doc_tokens=config.output.max_doc_tokens,
        max_rows=config.output.max_rows,
        sample=config.sample,
        sample_seed=config.sample_seed,
        force=config.force,
        observability=config.observability,
    )


def _dispatch_chat_sft(config: PipelineConfig, blend: "DataBlend") -> FormatResult:
    """Dispatch to SFT pipeline for chat_sft format."""
    if config.tokenizer is None:
        raise ValueError(
            "TokenizerConfig is required for ChatSftOutputConfig. "
            "Set config.tokenizer = TokenizerConfig(model='...')"
        )

    output_format = config.output.format
    assert isinstance(output_format, ChatSftOutputConfig)

    # Only parquet storage is supported via this shim
    if output_format.packed_storage != "parquet":
        raise NotImplementedError(
            f"ChatSftOutputConfig.packed_storage='{output_format.packed_storage}' "
            "is not supported via run_data_prep shim. "
            "Use packed_storage='parquet' or implement a custom pipeline."
        )

    # Determine num_shards from format config or deprecated output config
    num_shards = output_format.num_shards or config.output.num_shards
    if num_shards is None:
        raise ValueError(
            "num_shards is required for chat_sft format. "
            "Set config.output.format = ChatSftOutputConfig(num_shards=N) or "
            "config.output.num_shards = N"
        )

    return run_sft_pipeline(
        blend=blend,
        output_dir=config.output.dir,
        tokenizer=config.tokenizer,
        num_shards=num_shards,
        dtype=output_format.dtype,
        messages_field_default=output_format.messages_field,
        tools_field_default=output_format.tools_field,
        chat_template=output_format.chat_template,
        used_in_filter=output_format.used_in_filter,
        used_in_field=output_format.used_in_field,
        pack_size=output_format.pack_size,
        algorithm=output_format.algorithm,
        parquet_row_group_size=output_format.parquet_row_group_size,
        parquet_compression=output_format.parquet_compression,
        max_doc_tokens=config.output.max_doc_tokens,
        max_rows=config.output.max_rows,
        sample=config.sample,
        sample_seed=config.sample_seed,
        force=config.force,
        observability=config.observability,
    )


def last_mile_process(
    blend: "DataBlend",
    config: PipelineConfig,
    **kwargs: Any,
) -> NoReturn:
    """Legacy API stub - raises NotImplementedError with migration guidance.

    This function was documented as a generic format dispatcher but was never
    fully implemented. Use the recipe-specific APIs instead:

    - For pretrain (binidx): run_pretrain_pipeline()
    - For SFT (packed parquet): run_sft_pipeline()
    - For RL (JSONL): Use nemotron/recipes/nano3/stage2_rl/data_prep.py

    Raises:
        NotImplementedError: Always, with migration guidance.
    """
    raise NotImplementedError(
        "last_mile_process() is not implemented. Use the recipe-specific APIs:\n"
        "  - For pretrain (binidx): from nemotron.data_prep.api import run_pretrain_pipeline\n"
        "  - For SFT (packed parquet): from nemotron.data_prep.api import run_sft_pipeline\n"
        "  - For RL (JSONL): Use nemotron/recipes/nano3/stage2_rl/data_prep.py\n"
        "\n"
        "Or use run_data_prep(config, blend=blend) for format-based dispatch."
    )
