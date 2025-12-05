"""Data preparation for Megatron training.

Tokenizes raw text data from HuggingFace, S3, or local sources into
.bin/.idx format compatible with Megatron-Bridge and Megatron-Core.

Quick Start:
    from nemotron.data_prep import tokenize, DataBlend
    from nemotron.data_prep.config import PipelineConfig, TokenizerConfig, OutputConfig
    from pathlib import Path

    # Load data blend
    blend = DataBlend.load("data_blend.json")

    # Configure pipeline
    config = PipelineConfig(
        tokenizer=TokenizerConfig(model="meta-llama/Llama-3.2-1B"),
        output=OutputConfig(dir=Path("./output"), num_shards=128),
        split="99990,8,2",
    )

    # Run tokenization
    result = tokenize(blend, config)

    # Use output with Megatron-Bridge
    print(f"Blend file: {result.blend_path}")

Output Format:
    The generated blend.json is directly compatible with Megatron-Bridge's
    get_blend_fields_from_data_paths() function.
"""

from nemotron.data_prep.blend import DataBlend, Dataset
from nemotron.data_prep.config import PipelineConfig, TokenizerConfig, OutputConfig
from nemotron.data_prep.pipeline import tokenize, PipelineResult, SplitResult

__all__ = [
    # Input specification
    "DataBlend",
    "Dataset",
    # Configuration
    "PipelineConfig",
    "TokenizerConfig",
    "OutputConfig",
    # Execution
    "tokenize",
    # Results
    "PipelineResult",
    "SplitResult",
]
