#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "rerank/finetune"
# image = "nvcr.io/nvidia/nemo-automodel:26.04"
# setup = "NeMo Automodel pre-installed. Stage dependencies resolved via UV at runtime."
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
# gpus_per_node = "gpu"
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
with prepared training data from rerank Stage 1 prep.

Usage:
    # With default config
    nemotron rerank finetune -c default

    # With custom config
    nemotron rerank finetune -c /path/to/config.yaml

    # With CLI overrides
    nemotron rerank finetune -c default base_model=nvidia/llama-nemotron-rerank-1b-v2
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides
from nemotron.recipes.rerank._trust import validate_trust_remote_code

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


def _is_rank_zero() -> bool:
    """Return True for the single process that should publish shared artifacts."""
    return os.environ.get("RANK", "0") == "0"


class FinetuneConfig(RecipeSettings):
    """Fine-tuning configuration for cross-encoder reranking models."""

    model_config = ConfigDict(extra="forbid")

    # Model settings
    base_model: str = Field(
        default="nvidia/llama-nemotron-rerank-1b-v2",
        description="Base reranking model to fine-tune.",
    )
    allow_untrusted_remote_code: bool = Field(
        default=False,
        description="Allow trust_remote_code for non-NVIDIA remote model refs.",
    )

    # Data paths
    train_data_path: Path = Field(
        default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage1_prep/train_mined.automodel_unrolled.json",
        description="Path to training data file.",
    )

    # Output settings
    checkpoint_dir: Path = Field(
        default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage2_finetune/checkpoints",
        description="Directory for saving checkpoints.",
    )

    # Training hyperparameters
    num_epochs: int = Field(default=3, gt=0, description="Number of training epochs.")
    global_batch_size: int = Field(default=128, gt=0, description="Global batch size across all GPUs.")
    local_batch_size: int = Field(default=4, gt=0, description="Per-GPU batch size.")
    learning_rate: float = Field(default=3e-6, gt=0, description="Learning rate.")
    lr_warmup_steps: int = Field(default=100, ge=0, description="Learning rate warmup steps.")
    lr_decay_style: Literal["cosine", "linear"] = Field(
        default="cosine",
        description="LR decay schedule (cosine, linear).",
    )
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay for optimizer.")
    optimizer_backend: Literal["auto", "fused_adam", "flash_adamw"] = Field(
        default="auto",
        description="Optimizer backend. 'auto' uses FusedAdam when available, otherwise FlashAdamW.",
    )
    flash_adamw_master_weight_bits: Literal[24, 32] = Field(
        default=32,
        description="Effective master-weight precision for FlashAdamW when Transformer Engine is unavailable.",
    )

    # Model architecture
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] | None = Field(
        default=None,
        description="Attention implementation (sdpa, flash_attention_2, eager). None auto-detects.",
    )
    train_n_passages: int = Field(
        default=5,
        ge=2,
        description="Number of passages per query during training (1 pos + n-1 neg).",
    )
    num_labels: int = Field(default=1, ge=1, description="Number of classification labels.")
    temperature: float = Field(default=1.0, gt=0, description="Temperature for cross-entropy loss.")
    pooling: Literal["avg", "cls", "last"] = Field(default="avg", description="Pooling strategy.")

    # Tokenization
    rerank_max_length: int = Field(
        default=512,
        gt=0,
        description="Maximum sequence length for concatenated query+passage.",
    )
    prompt_template: str = Field(
        default="question:{query} \n \n passage:{passage}",
        description="Template for formatting query-passage pairs.",
    )

    # Checkpointing
    checkpoint_every_steps: int = Field(default=100, gt=0, description="Save checkpoint every N steps.")
    val_every_steps: int = Field(default=100, gt=0, description="Run validation every N steps.")

    @model_validator(mode="after")
    def _check_remote_code_trust(self):
        validate_trust_remote_code(
            [self.base_model],
            allow_untrusted_remote_code=self.allow_untrusted_remote_code,
        )
        return self


def _count_training_examples(train_data_path: Path) -> int:
    """Count the number of training examples in a training data file."""
    with open(train_data_path) as f:
        data = json.load(f)
    return len(data.get("data", []))


def _warn_if_negatives_sparse(train_data_path: Path, train_n_passages: int) -> None:
    """Warn if training data has fewer negatives than train_n_passages requires."""
    needed = train_n_passages - 1
    with open(train_data_path) as f:
        data = json.load(f)
    records = data.get("data", [])
    if not records:
        return
    neg_counts = [len(r.get("neg_doc", [])) for r in records[:200]]
    median_neg = sorted(neg_counts)[len(neg_counts) // 2]
    if median_neg < needed:
        print(
            f"Warning: train_n_passages={train_n_passages} needs {needed} negatives per query,\n"
            f"         but training data has a median of {median_neg}.\n"
            f"         Consider increasing hard_negatives_to_mine in stage1 prep or\n"
            f"         reducing train_n_passages.",
            file=sys.stderr,
        )
        print()


def _auto_scale_hyperparams(cfg: FinetuneConfig, num_examples: int) -> tuple[int, int, int, int]:
    """Auto-scale training hyperparameters based on dataset size.

    Args:
        cfg: Fine-tuning configuration (with user-specified or default values).
        num_examples: Number of training examples.

    Returns:
        Tuple of (global_batch_size, num_epochs, checkpoint_every_steps, val_every_steps).
    """
    # Keep the user/default global batch size intact. Automodel requires it to
    # divide cleanly by the per-rank batch geometry; dataset-size-derived values
    # can become invalid on multi-GPU runs.
    global_batch_size = cfg.global_batch_size

    steps_per_epoch = max(1, num_examples // global_batch_size)
    num_epochs = cfg.num_epochs
    total_steps = steps_per_epoch * num_epochs

    if total_steps < cfg.checkpoint_every_steps * 3:
        checkpoint_every_steps = max(1, total_steps // 3)
    else:
        checkpoint_every_steps = cfg.checkpoint_every_steps

    if total_steps < cfg.val_every_steps * 3:
        val_every_steps = max(1, total_steps // 3)
    else:
        val_every_steps = cfg.val_every_steps

    return global_batch_size, num_epochs, checkpoint_every_steps, val_every_steps


def _can_import_fused_adam() -> tuple[bool, str | None]:
    """Return whether Transformer Engine FusedAdam is importable."""
    try:
        importlib.import_module("transformer_engine.pytorch.optimizers.fused_adam")
    except Exception as e:
        return False, str(e)
    return True, None


def _can_import_flash_adamw() -> tuple[bool, str | None]:
    """Return whether FlashAdamW is importable."""
    try:
        importlib.import_module("flashoptim")
    except Exception as e:
        return False, str(e)
    return True, None


def _get_fsdp_shard_mesh_size(mesh: Any) -> int | None:
    """Return the FSDP shard mesh size for a 1D FSDP or 2D HSDP mesh."""
    if mesh is None:
        return None
    try:
        mesh_ndim = int(getattr(mesh, "ndim", 1))
        shard_mesh_dim = 0 if mesh_ndim == 1 else mesh_ndim - 1
        return int(mesh.size(shard_mesh_dim))
    except Exception:
        return None


def _patch_flashoptim_fsdp2_shard_placement() -> None:
    """Shard small 2D heads on dim 1 so FlashOptim DCP checkpoints can save.

    FlashOptim wraps optimizer state tensors as DTensors before DCP save and
    currently requires the parameter shard dimension to divide evenly. The
    rerank score head is shaped [num_labels, hidden], commonly [1, hidden],
    which is uneven on FSDP's default dim-0 sharding across multi-GPU runs.
    Sharding those 2D parameters on dim 1 keeps FSDP semantics while making the
    optimizer state checkpointable.
    """
    try:
        import nemo_automodel.components.distributed.parallelizer as parallelizer
        from torch.distributed.tensor import Shard
    except Exception:
        return

    original_fully_shard = getattr(parallelizer, "fully_shard", None)
    if original_fully_shard is None or getattr(
        original_fully_shard,
        "_nemotron_flashoptim_shard_patch",
        False,
    ):
        return

    def fully_shard_with_flashoptim_placement(module: Any, *args: Any, **kwargs: Any) -> Any:
        mesh = kwargs.get("mesh")
        user_shard_placement_fn = kwargs.get("shard_placement_fn")
        shard_mesh_size = _get_fsdp_shard_mesh_size(mesh)

        def shard_placement_fn(param: Any) -> Any:
            if user_shard_placement_fn is not None:
                placement = user_shard_placement_fn(param)
                if placement is not None:
                    return placement

            if shard_mesh_size and getattr(param, "ndim", 0) >= 2:
                shape = tuple(param.shape)
                if shape[0] % shard_mesh_size != 0 and shape[1] % shard_mesh_size == 0:
                    return Shard(1)
            return None

        kwargs["shard_placement_fn"] = shard_placement_fn
        return original_fully_shard(module, *args, **kwargs)

    fully_shard_with_flashoptim_placement._nemotron_flashoptim_shard_patch = True
    parallelizer.fully_shard = fully_shard_with_flashoptim_placement


def _load_automodel_config(cfg: FinetuneConfig, config_node_cls: type) -> tuple[Any, str]:
    """Load Automodel YAML after choosing an optimizer that is importable here."""
    import yaml

    base_config_path = STAGE_PATH / "crossencoder_base.yaml"
    with open(base_config_path) as f:
        raw_config = yaml.safe_load(f)

    te_available, te_error = _can_import_fused_adam()
    flash_available, flash_error = _can_import_flash_adamw()
    optimizer_backend = cfg.optimizer_backend
    if optimizer_backend == "auto":
        optimizer_backend = "fused_adam" if te_available else "flash_adamw"

    if optimizer_backend == "fused_adam":
        if not te_available:
            print("Error: optimizer_backend=fused_adam requires Transformer Engine.", file=sys.stderr)
            if te_error:
                print(f"  Import error: {te_error}", file=sys.stderr)
            print(
                "  Use optimizer_backend=flash_adamw for local runs without Transformer Engine.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif optimizer_backend == "flash_adamw":
        if not flash_available:
            print("Error: optimizer_backend=flash_adamw requires flashoptim.", file=sys.stderr)
            if flash_error:
                print(f"  Import error: {flash_error}", file=sys.stderr)
            print("  Install flashoptim, or run in an environment with Transformer Engine FusedAdam.", file=sys.stderr)
            sys.exit(1)
        raw_config["optimizer"] = {
            "_target_": "flashoptim.FlashAdamW",
            "lr": raw_config.get("optimizer", {}).get("lr", cfg.learning_rate),
            "weight_decay": raw_config.get("optimizer", {}).get("weight_decay", cfg.weight_decay),
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "quantize": False,
            "compress_state_dict": False,
            "master_weight_bits": cfg.flash_adamw_master_weight_bits,
            "fused": True,
        }
        raw_config.setdefault("model", {})["torch_dtype"] = "bfloat16"
        raw_config["model"]["dtype"] = "bfloat16"

    return config_node_cls(raw_config), optimizer_backend


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
        print("       Please run 'nemotron rerank prep' first.", file=sys.stderr)
        sys.exit(1)

    # Count training examples and check negative passage availability
    num_examples = _count_training_examples(cfg.train_data_path)
    _warn_if_negatives_sparse(cfg.train_data_path, cfg.train_n_passages)

    global_batch_size, num_epochs, ckpt_every, val_every = _auto_scale_hyperparams(cfg, num_examples)

    steps_per_epoch = max(1, num_examples // global_batch_size)
    total_steps = steps_per_epoch * num_epochs

    # Print training plan
    print("Training plan:")
    print(f"  Dataset:          {num_examples:,} examples")

    if global_batch_size != cfg.global_batch_size:
        print(
            f"  Batch size:       {global_batch_size} "
            f"(auto-scaled from {cfg.global_batch_size} — dataset < 2000 examples)"
        )
    else:
        print(f"  Batch size:       {global_batch_size}")

    print(f"  Epochs:           {num_epochs}")
    print(f"  Steps/epoch:      ~{steps_per_epoch}")
    print(f"  Total steps:      ~{total_steps}")
    print(f"  LR schedule:      {cfg.lr_decay_style}, warmup={cfg.lr_warmup_steps}, peak={cfg.learning_rate}")
    print(f"  Checkpoint every: {ckpt_every} steps")
    print(f"  Validate every:   {val_every} steps")
    print()

    if total_steps < 50:
        print(
            f"Warning: Only ~{total_steps} total training steps. Dataset may be too small for meaningful fine-tuning.",
            file=sys.stderr,
        )
        print("         Consider adding more documents to your corpus.", file=sys.stderr)
        print()

    print(f"Base model:     {cfg.base_model}")
    print(f"Training data:  {cfg.train_data_path}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print()

    # Import nemo-automodel components
    try:
        from nemo_automodel.components.config.loader import ConfigNode
        from nemo_automodel.recipes.retrieval import TrainCrossEncoderRecipe
    except ImportError as e:
        print("Error: Failed to import nemo-automodel. Is it installed?", file=sys.stderr)
        print("  Install with: pip install nemo-automodel", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load base config from nemo-automodel defaults. ConfigNode resolves _target_
    # imports during construction, so optimizer selection must happen on raw YAML.
    automodel_cfg, optimizer_backend = _load_automodel_config(cfg, ConfigNode)
    if optimizer_backend == "flash_adamw":
        _patch_flashoptim_fsdp2_shard_placement()
    optimizer_detail = optimizer_backend
    if optimizer_backend == "flash_adamw":
        optimizer_detail = f"{optimizer_backend} (bf16 model, {cfg.flash_adamw_master_weight_bits}-bit master weights)"
    print(f"Optimizer:      {optimizer_detail}")
    print()

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
    automodel_cfg.model.attn_implementation = attn_impl

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

    # Optimizer settings
    automodel_cfg.optimizer.lr = cfg.learning_rate
    automodel_cfg.optimizer.weight_decay = cfg.weight_decay
    # Warmup must be strictly less than total decay steps
    lr_warmup_steps = min(cfg.lr_warmup_steps, max(1, total_steps - 1))
    automodel_cfg.lr_scheduler.lr_warmup_steps = lr_warmup_steps
    automodel_cfg.lr_scheduler.lr_decay_style = cfg.lr_decay_style

    # Checkpoint settings
    automodel_cfg.checkpoint.checkpoint_dir = str(cfg.checkpoint_dir)

    # Create and run the cross-encoder recipe
    recipe = TrainCrossEncoderRecipe(automodel_cfg)
    recipe.setup()
    recipe.run_train_validation_loop()

    # Find the final checkpoint
    final_model_dir = cfg.checkpoint_dir / "LATEST" / "model" / "consolidated"

    print("\nFine-tuning complete!")
    print(f"   Checkpoint: {cfg.checkpoint_dir}")
    print(f"   Model:      {final_model_dir}")

    # Save artifact (registers with artifact registry if kit.init() was called)
    if _is_rank_zero():
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
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)

        try:
            cfg = load_config(config_path, cli_overrides, FinetuneConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_finetune(cfg)


if __name__ == "__main__":
    main()
