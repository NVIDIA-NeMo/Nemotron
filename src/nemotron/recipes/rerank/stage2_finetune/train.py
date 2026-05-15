#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "rerank/finetune"
# image = "nvcr.io/nvidia/pytorch:25.12-py3"
# setup = "PyTorch pre-installed. Stage dependencies resolved via UV at runtime."
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 1
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

"""Fine-tuning script for cross-encoder reranking models.

Fine-tunes a reranking model using cross-entropy classification loss
with prepared training data (from embed stage1_data_prep).

Usage:
    # With default config
    nemotron rerank finetune -c default

    # With custom config
    nemotron rerank finetune -c /path/to/config.yaml

    # With CLI overrides
    nemotron rerank finetune -c default base_model=nvidia/llama-nemotron-rerank-1b-v2
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from types import MethodType
from typing import Literal

from pydantic import ConfigDict, Field

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class FinetuneConfig(RecipeSettings):
    """Fine-tuning configuration for cross-encoder reranking models."""

    model_config = ConfigDict(extra="forbid")

    # Model settings
    base_model: str = Field(default="nvidia/llama-nemotron-rerank-1b-v2", description="Base reranking model to fine-tune.")

    # Data paths
    train_data_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage1_prep/train_mined.automodel_unrolled.json", description="Path to training data file.")

    # Output settings
    checkpoint_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage2_finetune/checkpoints", description="Directory for saving checkpoints.")

    # Training hyperparameters
    num_epochs: int = Field(default=3, gt=0, description="Number of training epochs.")
    global_batch_size: int = Field(default=128, gt=0, description="Global batch size across all GPUs.")
    local_batch_size: int = Field(default=4, gt=0, description="Per-GPU batch size.")
    learning_rate: float = Field(default=3e-6, gt=0, description="Learning rate.")
    lr_warmup_steps: int = Field(default=100, ge=0, description="Learning rate warmup steps.")
    lr_decay_style: Literal["cosine", "linear"] = Field(default="cosine", description="LR decay schedule (cosine, linear).")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay for optimizer.")
    max_steps: int | None = Field(default=None, gt=0, description="Optional absolute maximum optimizer step to run to. Useful for short smoke tests and controlled resume runs.")
    force_fp32_parameters: bool = Field(default=True, description="Cast model parameters to fp32 before training so torch AdamW creates fp32 optimizer state.")

    # Model architecture
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] | None = Field(default=None, description="Attention implementation (sdpa, flash_attention_2, eager). None auto-detects.")
    train_n_passages: int = Field(default=5, ge=2, description="Number of passages per query during training (1 pos + n-1 neg).")
    num_labels: int = Field(default=1, ge=1, description="Number of classification labels.")
    temperature: float = Field(default=1.0, gt=0, description="Temperature for cross-entropy loss.")
    pooling: Literal["avg", "cls", "last"] = Field(default="avg", description="Pooling strategy.")

    # Tokenization
    rerank_max_length: int = Field(default=512, gt=0, description="Maximum sequence length for concatenated query+passage.")
    prompt_template: str = Field(default="question:{query} \n \n passage:{passage}", description="Template for formatting query-passage pairs.")

    # Checkpointing
    checkpoint_every_steps: int = Field(default=100, gt=0, description="Save checkpoint every N steps.")
    val_every_steps: int = Field(default=100, gt=0, description="Run validation every N steps.")
    restore_from: str | None = Field(default=None, description="Optional checkpoint directory or name to restore from.")


_ADAM_METADATA_KEYS = ("exp_avg", "exp_avg_sq", "step")


def _cast_model_parts_to_fp32(model_parts) -> None:
    """Cast model parameters in-place so torch AdamW keeps fp32 params/state."""
    print("Casting model parameters to fp32 before training")
    for model_part in model_parts:
        model_part.float()


def _optimizer_metadata_dtype_counts(metadata) -> dict[str, dict[str, int]]:
    """Summarize optimizer state dtypes from torch distributed checkpoint metadata."""
    counts: dict[str, dict[str, int]] = {}
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    for key, value in state_metadata.items():
        if not str(key).startswith("optim.state."):
            continue
        state_name = str(key).rsplit(".", 1)[-1]
        if state_name not in _ADAM_METADATA_KEYS:
            continue

        properties = getattr(value, "properties", None)
        dtype = getattr(properties, "dtype", None)
        if dtype is None:
            continue

        dtype_counts = counts.setdefault(state_name, {})
        dtype_name = str(dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    return counts


def _read_optimizer_metadata_dtype_counts(checkpoint_dir: Path) -> dict[str, dict[str, int]]:
    metadata_path = checkpoint_dir / "optim" / ".metadata"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Optimizer metadata not found: {metadata_path}")

    with metadata_path.open("rb") as f:
        return _optimizer_metadata_dtype_counts(pickle.load(f))


def _assert_optimizer_metadata_fp32(checkpoint_dir: Path) -> dict[str, dict[str, int]]:
    """Assert Adam optimizer checkpoint metadata contains only fp32 state tensors."""
    counts = _read_optimizer_metadata_dtype_counts(checkpoint_dir)
    missing = [state_name for state_name in _ADAM_METADATA_KEYS if state_name not in counts]
    non_fp32 = {
        state_name: dtype_counts
        for state_name, dtype_counts in counts.items()
        if set(dtype_counts) != {"torch.float32"}
    }
    if missing or non_fp32:
        raise AssertionError(
            "Expected resumed optimizer checkpoint metadata to be fp32. "
            f"missing={missing}, non_fp32={non_fp32}, counts={counts}"
        )
    return counts


def _install_fp32_restore_hook(recipe) -> None:
    """Load restored model in fp32 before loading optimizer state.

    Automodel default restore order is model plus optimizer load inside setup.
    If we cast after setup, optimizer state has already been attached and fused
    AdamW can fail because param, grad, and moment tensor lists no longer have
    matching dtypes. This hook keeps the load order explicit for fp32-state
    resumes: load model, cast params to fp32, then load optimizer and scheduler.
    """

    def load_checkpoint_with_fp32_cast(self, restore_from: str | None = None):
        if not self.checkpointer.config.enabled:
            if restore_from is not None:
                print("Enable checkpointing to resume from a checkpoint, skipping...", flush=True)
            return
        if restore_from is None:
            # Keep Automodel auto-detect path unchanged for non-explicit resumes.
            return original_load_checkpoint(restore_from)

        import torch

        is_rank_0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if str(restore_from).upper() == "LATEST":
            from nemo_automodel.recipes.base_recipe import _find_latest_checkpoint

            ckpt_dir = _find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
            if ckpt_dir is None:
                if is_rank_0:
                    print(
                        "restore_from=LATEST specified but no checkpoint found in "
                        f"{self.checkpointer.config.checkpoint_dir}. Starting fresh.",
                        flush=True,
                    )
                return
            ckpt_dir = str(ckpt_dir)
        elif os.path.sep not in str(restore_from) and not os.path.isabs(str(restore_from)):
            ckpt_dir = os.path.join(self.checkpointer.config.checkpoint_dir, str(restore_from))
        else:
            ckpt_dir = str(restore_from)

        self._validate_checkpoint_dir_exists(ckpt_dir, restore_from=str(restore_from), is_rank_0=is_rank_0)
        if is_rank_0:
            print(f"Loading checkpoint from {ckpt_dir}", flush=True)
            print("Restoring model before fp32 cast, then loading optimizer state", flush=True)

        model, optimizer, scheduler = self._load_checkpoint_tracked_state(ckpt_dir)
        self.checkpointer.load_model(model, os.path.join(ckpt_dir, "model"))
        _cast_model_parts_to_fp32(self.model_parts)
        self.checkpointer.load_optimizer(optimizer, model, ckpt_dir, scheduler)

    original_load_checkpoint = recipe.load_checkpoint
    recipe.load_checkpoint = MethodType(load_checkpoint_with_fp32_cast, recipe)

def _count_training_examples(train_data_path: Path) -> int:
    """Count the number of training examples in a training data file."""
    with open(train_data_path) as f:
        data = json.load(f)
    return len(data.get("data", []))


def _auto_scale_hyperparams(
    cfg: FinetuneConfig, num_examples: int
) -> tuple[int, int, int, int]:
    """Auto-scale training hyperparameters based on dataset size.

    Args:
        cfg: Fine-tuning configuration (with user-specified or default values).
        num_examples: Number of training examples.

    Returns:
        Tuple of (global_batch_size, num_epochs, checkpoint_every_steps, val_every_steps).
    """
    if cfg.global_batch_size == 128 and num_examples < 2000:
        global_batch_size = max(16, min(64, num_examples // 8))
    else:
        global_batch_size = cfg.global_batch_size

    steps_per_epoch = max(1, num_examples // global_batch_size)
    num_epochs = cfg.num_epochs
    total_steps = steps_per_epoch * num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)

    if total_steps < cfg.checkpoint_every_steps * 3:
        checkpoint_every_steps = max(1, total_steps // 3)
    else:
        checkpoint_every_steps = cfg.checkpoint_every_steps

    if total_steps < cfg.val_every_steps * 3:
        val_every_steps = max(1, total_steps // 3)
    else:
        val_every_steps = cfg.val_every_steps

    return global_batch_size, num_epochs, checkpoint_every_steps, val_every_steps


def run_finetune(cfg: FinetuneConfig) -> Path:
    """Run cross-encoder reranking model fine-tuning using nemo-automodel.

    Args:
        cfg: Fine-tuning configuration.

    Returns:
        Path to final checkpoint directory.
    """
    # Validate inputs
    if not cfg.train_data_path.exists():
        print(f"Error: Training data not found: {cfg.train_data_path}", file=sys.stderr)
        print("       Please run 'nemotron embed prep' first.", file=sys.stderr)
        sys.exit(1)

    # Count training examples
    num_examples = _count_training_examples(cfg.train_data_path)

    global_batch_size, num_epochs, ckpt_every, val_every = _auto_scale_hyperparams(
        cfg, num_examples
    )

    steps_per_epoch = max(1, num_examples // global_batch_size)
    total_steps = steps_per_epoch * num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)

    # Print training plan
    print(f"Training plan:")
    print(f"  Dataset:          {num_examples:,} examples")

    if global_batch_size != cfg.global_batch_size:
        print(f"  Batch size:       {global_batch_size} (auto-scaled from {cfg.global_batch_size} — dataset < 2000 examples)")
    else:
        print(f"  Batch size:       {global_batch_size}")

    print(f"  Epochs:           {num_epochs}")
    print(f"  Steps/epoch:      ~{steps_per_epoch}")
    print(f"  Total steps:      ~{total_steps}")
    if cfg.max_steps is not None:
        print(f"  Max steps:        {cfg.max_steps}")
    lr_warmup_steps = min(cfg.lr_warmup_steps, max(1, total_steps - 1))
    print(f"  LR schedule:      {cfg.lr_decay_style}, warmup={lr_warmup_steps}, peak={cfg.learning_rate}")
    print(f"  Checkpoint every: {ckpt_every} steps")
    print(f"  Validate every:   {val_every} steps")
    print()

    if total_steps < 50:
        print(f"Warning: Only ~{total_steps} total training steps. "
              f"Dataset may be too small for meaningful fine-tuning.", file=sys.stderr)
        print(f"         Consider adding more documents to your corpus.", file=sys.stderr)
        print()

    print(f"Base model:     {cfg.base_model}")
    print(f"Training data:  {cfg.train_data_path}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print()

    # Import nemo-automodel components
    try:
        from nemo_automodel.components.config.loader import load_yaml_config
        from nemo_automodel.recipes.retrieval import TrainCrossEncoderRecipe
    except ImportError as e:
        print(f"Error: Failed to import nemo-automodel. Is it installed?", file=sys.stderr)
        print(f"  Install with: pip install nemo-automodel", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load base config from nemo-automodel defaults
    base_config_path = STAGE_PATH / "crossencoder_base.yaml"
    automodel_cfg = load_yaml_config(str(base_config_path))

    # Apply overrides from our config
    # Model settings
    automodel_cfg.model.pretrained_model_name_or_path = cfg.base_model
    automodel_cfg.tokenizer.pretrained_model_name_or_path = cfg.base_model
    automodel_cfg.model.num_labels = cfg.num_labels
    automodel_cfg.model.temperature = cfg.temperature
    automodel_cfg.model.pooling = cfg.pooling

    # Auto-detect attention implementation if not explicitly set
    if cfg.attn_implementation is not None:
        attn_impl = cfg.attn_implementation
    else:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        print(f"  Attention:    {attn_impl} (auto-detected)")

    # Data settings
    automodel_cfg.dataloader.dataset.data_dir_list = [str(cfg.train_data_path)]
    automodel_cfg.dataloader.dataset.n_passages = cfg.train_n_passages
    automodel_cfg.dataloader.collate_fn.rerank_max_length = cfg.rerank_max_length
    automodel_cfg.dataloader.collate_fn.prompt_template = cfg.prompt_template

    # Training settings — use auto-scaled values
    automodel_cfg.step_scheduler.num_epochs = num_epochs
    automodel_cfg.step_scheduler.global_batch_size = global_batch_size
    automodel_cfg.step_scheduler.local_batch_size = cfg.local_batch_size
    automodel_cfg.step_scheduler.ckpt_every_steps = ckpt_every
    automodel_cfg.step_scheduler.val_every_steps = val_every
    if cfg.max_steps is not None:
        automodel_cfg.step_scheduler.max_steps = cfg.max_steps

    # Optimizer settings
    automodel_cfg.optimizer.lr = cfg.learning_rate
    automodel_cfg.optimizer.weight_decay = cfg.weight_decay
    automodel_cfg.lr_scheduler.lr_warmup_steps = lr_warmup_steps

    # Checkpoint settings
    automodel_cfg.checkpoint.checkpoint_dir = str(cfg.checkpoint_dir)
    if cfg.restore_from is not None:
        automodel_cfg.checkpoint.restore_from = cfg.restore_from

    # Create and run the cross-encoder recipe
    recipe = TrainCrossEncoderRecipe(automodel_cfg)
    if cfg.force_fp32_parameters and cfg.restore_from is not None:
        _install_fp32_restore_hook(recipe)
    recipe.setup()
    if cfg.force_fp32_parameters and cfg.restore_from is None:
        _cast_model_parts_to_fp32(recipe.model_parts)
    recipe.run_train_validation_loop()
    if cfg.force_fp32_parameters and cfg.restore_from is not None:
        counts = _assert_optimizer_metadata_fp32(cfg.checkpoint_dir / "LATEST")
        print(f"Verified resumed optimizer checkpoint metadata is fp32: {counts}")

    # Find the final checkpoint
    final_model_dir = cfg.checkpoint_dir / "LATEST" / "model" / "consolidated"

    print(f"\nFine-tuning complete!")
    print(f"   Checkpoint: {cfg.checkpoint_dir}")
    print(f"   Model:      {final_model_dir}")

    # Save artifact (registers with artifact registry if kit.init() was called)
    try:
        from nemotron.kit.artifacts.rerank import RerankModelArtifact

        artifact = RerankModelArtifact(
            path=final_model_dir,
            base_model=cfg.base_model,
            training_examples=num_examples,
            num_epochs=num_epochs,
            global_batch_size=global_batch_size,
            learning_rate=cfg.learning_rate,
            num_labels=cfg.num_labels,
        )
        artifact.save(name="rerank/model")
    except Exception:
        pass  # Artifact save is best-effort — don't break the pipeline

    return final_model_dir


def main(cfg: FinetuneConfig | None = None) -> Path:
    """Entry point for fine-tuning.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Path to final model checkpoint.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )

        try:
            cfg = load_config(config_path, cli_overrides, FinetuneConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_finetune(cfg)


if __name__ == "__main__":
    main()
