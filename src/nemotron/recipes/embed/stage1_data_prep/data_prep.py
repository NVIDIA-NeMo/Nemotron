#!/usr/bin/env python3

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

"""Data Preparation for embedding fine-tuning.

Prepares training data from SDG output:
1. Convert to BEIR format (train/val/test split)
2. Mine hard negatives using base embedding model
3. Unroll multi-hop positives

Usage:
    # With default config
    nemotron embed prep -c default

    # With custom config
    nemotron embed prep -c /path/to/config.yaml

    # With CLI overrides
    nemotron embed prep -c default sdg_output_dir=/path/to/sdg
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from nemotron.kit.config_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class DataPrepConfig(RecipeSettings):
    """Data Preparation configuration.

    Converts SDG output to training format, mines hard negatives,
    and unrolls multi-hop positives.
    """

    model_config = ConfigDict(extra="forbid")

    corpus_id: str = Field(default="my_corpus", description="Corpus identifier (used for output naming).")
    sdg_output_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage0_sdg", description="Path to SDG output directory.")
    output_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage1_data_prep", description="Output directory for prepared training data.")

    # Model for hard negative mining
    base_model: str = Field(default="nvidia/llama-nemotron-embed-1b-v2", description="Base embedding model for hard negative mining.")

    # Quality filtering
    quality_threshold: float = Field(default=7.0, ge=0, le=10, description="Minimum quality score for Q&A pairs (0-10 scale).")

    # Train/val/test split ratios
    train_ratio: float = Field(default=0.8, gt=0, lt=1, description="Fraction of data for training.")
    val_ratio: float = Field(default=0.1, ge=0, lt=1, description="Fraction of data for validation.")
    test_ratio: float = Field(default=0.1, ge=0, lt=1, description="Fraction of data for testing.")

    @model_validator(mode="after")
    def _check_ratios_sum_to_one(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}")
        return self

    # Hard negative mining settings
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(default="sdpa", description="Attention implementation (sdpa, flash_attention_2, eager).")
    hard_negatives_to_mine: int = Field(default=5, gt=0, description="Number of hard negatives to mine per query.")
    hard_neg_margin: float = Field(default=0.95, gt=0, le=1, description="Margin for hard negative selection.")
    mining_batch_size: int = Field(default=128, gt=0, description="Batch size for mining.")
    query_max_length: int = Field(default=512, gt=0, description="Maximum query length for tokenization.")
    passage_max_length: int = Field(default=512, gt=0, description="Maximum passage length for tokenization.")
    query_prefix: str = Field(default="query:", description="Prefix for query inputs during mining.")
    passage_prefix: str = Field(default="passage:", description="Prefix for passage inputs during mining.")


def run_convert(cfg: DataPrepConfig) -> Path:
    """Convert SDG output to training format.

    Returns:
        Path to train_eval directory.
    """
    convert_script = STAGE_PATH / "scripts" / "convert_to_retriever_data.py"

    cmd = [
        sys.executable,
        str(convert_script),
        str(cfg.sdg_output_dir),
        "--corpus-id",
        cfg.corpus_id,
        "--output-dir",
        str(cfg.output_dir),
        "--quality-threshold",
        str(cfg.quality_threshold),
        "--train-ratio",
        str(cfg.train_ratio),
        "--val-ratio",
        str(cfg.val_ratio),
    ]

    print("🔄 Converting SDG output to training format...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: convert script failed with return code {result.returncode}")
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    return cfg.output_dir


def run_mining(cfg: DataPrepConfig) -> Path:
    """Mine hard negatives using base embedding model.

    Returns:
        Path to mined training file.
    """
    mining_script = STAGE_PATH / "scripts" / "mine_hard_negatives.py"
    mining_config = STAGE_PATH / "scripts" / "mining_config.yaml"

    train_file = cfg.output_dir / "train.json"
    output_file = cfg.output_dir / "train_mined.automodel.json"
    cache_dir = cfg.output_dir / "cache_embeddings"

    cmd = [
        sys.executable,
        str(mining_script),
        "--config",
        str(mining_config),
        "--mining.model_name_or_path",
        cfg.base_model,
        "--mining.train_qa_file_path",
        str(train_file),
        "--mining.train_file_output_path",
        str(output_file),
        "--mining.cache_embeddings_dir",
        str(cache_dir),
        "--mining.hard_neg_margin",
        str(cfg.hard_neg_margin),
        "--mining.hard_negatives_to_mine",
        str(cfg.hard_negatives_to_mine),
        "--mining.mining_batch_size",
        str(cfg.mining_batch_size),
        "--mining.query_prefix",
        cfg.query_prefix,
        "--mining.passage_prefix",
        cfg.passage_prefix,
        "--mining.attn_implementation",
        cfg.attn_implementation,
        "--mining.add_bos_token",
        "true",
        "--mining.add_eos_token",
        "false",
    ]

    print(f"\n⛏️  Mining hard negatives...")
    print(f"   Using model: {cfg.base_model}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: mining script failed with return code {result.returncode}")
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    return output_file


def run_unroll(cfg: DataPrepConfig) -> Path:
    """Unroll multi-hop positives.

    Returns:
        Path to final training file.
    """
    unroll_script = STAGE_PATH / "scripts" / "unroll_pos_docs.py"

    mined_file = cfg.output_dir / "train_mined.automodel.json"

    cmd = [
        sys.executable,
        str(unroll_script),
        str(mined_file),
    ]

    print("\n🔄 Unrolling multi-positive training examples...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: unroll script failed with return code {result.returncode}")
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    return cfg.output_dir / "train_mined.automodel_unrolled.json"


def run_data_prep(cfg: DataPrepConfig) -> Path:
    """Run full data preparation pipeline.

    Args:
        cfg: Data prep configuration.

    Returns:
        Path to final training data file.
    """
    print(f"📋 Data Preparation Pipeline")
    print(f"=" * 60)
    print(f"Corpus ID:      {cfg.corpus_id}")
    print(f"SDG Output:     {cfg.sdg_output_dir}")
    print(f"Output Dir:     {cfg.output_dir}")
    print(f"Base Model:     {cfg.base_model}")
    print(f"=" * 60)
    print()

    # Validate inputs
    if not cfg.sdg_output_dir.exists():
        print(f"Error: SDG output directory not found: {cfg.sdg_output_dir}", file=sys.stderr)
        print("       Please run stage0_sdg first.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Convert
    run_convert(cfg)

    # Step 2: Mine hard negatives
    run_mining(cfg)

    # Step 3: Unroll
    final_file = run_unroll(cfg)

    # Check eval set size
    eval_queries_path = cfg.output_dir / "eval_beir" / "queries.jsonl"
    if eval_queries_path.exists():
        eval_query_count = sum(1 for _ in open(eval_queries_path))
        if eval_query_count < 50:
            print(f"\nWarning: Eval set has only {eval_query_count} queries (recommended: 50+).",
                  file=sys.stderr)
            print(f"         Small eval sets produce noisy metrics. Consider:", file=sys.stderr)
            print(f"         - Adding more documents to your corpus", file=sys.stderr)
            print(f"         - Increasing num_pairs in SDG config", file=sys.stderr)
            print(f"         - Increasing test_ratio (currently {cfg.test_ratio})", file=sys.stderr)
        else:
            print(f"\n   Eval queries: {eval_query_count}")

    # Check training set size
    if final_file.exists():
        with open(final_file) as f:
            train_data = json.load(f)
        train_count = len(train_data.get("data", []))
        print(f"   Training examples: {train_count}")
        if train_count < 100:
            print(f"\nWarning: Only {train_count} training examples.",
                  file=sys.stderr)
            print(f"         Consider adding more documents or increasing num_pairs in SDG config.",
                  file=sys.stderr)

    print(f"\nData preparation complete!")
    print(f"   Training data: {final_file}")
    print(f"   Eval data:     {cfg.output_dir / 'eval_beir'}")

    return final_file


def main(cfg: DataPrepConfig | None = None) -> Path:
    """Entry point for data preparation.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Path to final training data file.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )

        try:
            cfg = load_config(config_path, cli_overrides, DataPrepConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_data_prep(cfg)


if __name__ == "__main__":
    main()
