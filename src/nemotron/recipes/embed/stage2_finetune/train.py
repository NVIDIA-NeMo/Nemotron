#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "embed/finetune"
# image = "nvcr.io/nvidia/nemo-automodel:26.06"
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

"""Fine-tuning script for embedding models.

Fine-tunes an embedding model using contrastive learning with prepared
training data (from stage1_data_prep).

Usage:
    # With default config
    nemotron embed finetune -c default

    # With custom config
    nemotron embed finetune -c /path/to/config.yaml

    # With CLI overrides
    nemotron embed finetune -c default model.pretrained_model_name_or_path=...

    # With an exact optimizer-step target
    nemotron embed finetune -c default num_epochs=null max_steps=1000
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides
from nemotron.recipes.embed.stage2_finetune.peft import (
    PeftConfig,
    merge_peft_checkpoint,
    validate_peft_base,
    validate_peft_runtime,
)
from nemotron.recipes.retrieval_vl import validate_vl_training_data

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


def _is_rank_zero() -> bool:
    """Return True for the single process that should publish shared artifacts."""
    return os.environ.get("RANK", "0") == "0"


class FinetuneConfig(RecipeSettings):
    """Fine-tuning configuration for embedding models."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: Path = Field(
        default_factory=lambda: _OUTPUT_BASE / "output/embed/nemotron-3-1b",
        description="Root directory for this model profile's pipeline artifacts.",
    )

    # Model settings
    base_model: str = Field(
        default="nvidia/Nemotron-3-Embed-1B-BF16",
        description="Base embedding model to fine-tune.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow Hugging Face custom model code. Required by the default Nemotron Embed model.",
    )
    tokenizer_force_default: bool = Field(
        default=False,
        description="Use AutoModel's Hugging Face tokenizer wrapper instead of model-type registry dispatch.",
    )
    model_family: Literal["text", "mistral3_vl"] = Field(
        default="text",
        description="AutoModel integration family. Use mistral3_vl for Mistral 3 vision-language retrieval models.",
    )
    peft: PeftConfig | None = Field(
        default=None,
        description="Optional native LoRA settings. Requires a local base with matching retrieval metadata.",
    )

    # Data paths
    train_data_path: Path = Field(
        default_factory=lambda data: data["artifact_root"] / "stage1_data_prep/train_mined.automodel_unrolled.json",
        description="Path to training data file.",
    )

    # Output settings
    checkpoint_dir: Path = Field(
        default_factory=lambda data: data["artifact_root"] / "stage2_finetune/checkpoints",
        description="Directory for saving checkpoints.",
    )

    # Training hyperparameters
    seed: int = Field(
        default=42,
        ge=0,
        le=4294967295,
        strict=True,
        description="Random seed for model training and dataset sampling (unsigned 32-bit integer).",
    )
    num_epochs: int | None = Field(
        default=3,
        gt=0,
        description="Number of training epochs. Set to null when max_steps is used.",
    )
    max_steps: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description=(
            "Absolute optimizer-step target. On resume, training completes the remaining steps. "
            "Requires num_epochs=null."
        ),
    )
    global_batch_size: int = Field(default=128, gt=0, description="Global batch size across all GPUs.")
    local_batch_size: int = Field(default=4, gt=0, description="Per-GPU batch size.")
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate.")
    lr_warmup_steps: int = Field(default=5, ge=0, description="Learning rate warmup steps.")
    lr_decay_style: Literal["cosine", "linear"] = Field(
        default="cosine",
        description="LR decay schedule (cosine, linear).",
    )
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay for optimizer.")
    optimizer_backend: Literal["auto", "fused_adam", "flash_adamw", "torch_adamw"] = Field(
        default="auto",
        description=(
            "Optimizer backend. 'auto' uses FusedAdam when available, otherwise FlashAdamW. "
            "'torch_adamw' explicitly uses native AdamW with FP32 parameters, compute, and moments."
        ),
    )
    flash_adamw_master_weight_bits: Literal[24, 32] | None = Field(
        default=32,
        description=(
            "Effective master-weight precision for FlashAdamW when Transformer Engine is unavailable. "
            "Set to None only to explicitly disable master-weight correction."
        ),
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
    pooling: Literal["avg", "cls", "last"] = Field(default="avg", description="Pooling strategy for embeddings.")
    l2_normalize: bool = Field(default=True, description="Whether to L2 normalize embeddings.")
    temperature: float = Field(default=0.02, gt=0, description="Temperature for contrastive loss.")
    is_causal: bool = Field(default=False, description="Whether attention is causal in the retrieval encoder.")
    do_distributed_inbatch_negative: bool = Field(
        default=False,
        description="Gather passages from every data-parallel rank as in-batch negatives.",
    )
    detach_distributed_inbatch_negatives: bool = Field(
        default=True,
        description="Detach gathered embeddings from remote ranks.",
    )

    # Tokenization
    query_max_length: int = Field(default=512, gt=0, description="Maximum query sequence length.")
    passage_max_length: int = Field(default=512, gt=0, description="Maximum passage sequence length.")
    query_prefix: str | None = Field(default="query: ", description="Prefix for query inputs.")
    passage_prefix: str | None = Field(default="passage: ", description="Prefix for passage inputs.")
    image_longest_edge: int | None = Field(
        default=1284, gt=0, description="Longest image edge presented to the processor."
    )
    pad_to_multiple_of: int = Field(default=8, gt=0, description="Pad sequence lengths to this multiple.")
    use_text_in_document: bool = Field(
        default=False,
        description="Include a document's text alongside its image when both are available.",
    )
    require_mined_negatives: bool = Field(
        default=False,
        description="Require hard-negative mining provenance and a finite score for every selected negative.",
    )

    # Checkpointing
    checkpoint_every_steps: int = Field(default=1000, gt=0, description="Save checkpoint every N steps.")
    val_every_steps: int = Field(default=1000, gt=0, description="Run validation every N steps.")
    auto_scale_checkpoint_intervals: bool = Field(
        default=False,
        description="Reduce checkpoint/validation intervals for small datasets.",
    )

    @field_validator("seed", mode="before")
    @classmethod
    def parse_seed_override(cls, value: Any) -> Any:
        """Parse CLI integer strings while retaining strict seed validation.

        Args:
            value: Seed supplied by configuration or a dotlist override.

        Returns:
            Parsed integer for a string, otherwise the unchanged value.

        Raises:
            ValueError: If a string does not represent an integer.
        """
        return int(value) if isinstance(value, str) else value

    @field_validator("max_steps", mode="before")
    @classmethod
    def parse_max_steps_override(cls, value: Any) -> Any:
        """Parse CLI integer strings while retaining strict integer validation.

        Args:
            value: Maximum step target supplied by configuration or a dotlist override.

        Returns:
            Parsed integer for a string, otherwise the unchanged value.

        Raises:
            ValueError: If a string does not represent an integer.
        """
        return int(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_training_budget(self) -> FinetuneConfig:
        """Require exactly one epoch-based or step-based training budget.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If both or neither training budget is configured.
        """
        if (self.num_epochs is None) == (self.max_steps is None):
            raise ValueError("Exactly one of num_epochs and max_steps must be set")
        return self


def _automodel_collator_prefix(prefix: str) -> str:
    """Adapt a complete prompt prefix for AutoModel's collator.

    ``BiEncoderCollator`` inserts one ASCII space between a non-empty prefix
    and its text. Recipe configs include that separator so mining, training,
    and evaluation expose the same prefix contract; remove exactly that one
    space before handing the value to the collator.
    """
    return prefix.removesuffix(" ")


def _repair_vllm_sentence_transformers_metadata(model_dir: Path, cfg: FinetuneConfig) -> None:
    """Write metadata that vLLM can use for the same pooling/prompt contract as training."""
    modules_path = model_dir / "modules.json"
    modules = json.loads(modules_path.read_text())
    module_types = {
        "": "sentence_transformers.models.Transformer",
        "1_Pooling": "sentence_transformers.models.Pooling",
        "2_Normalize": "sentence_transformers.models.Normalize",
    }
    for module in modules:
        path = module.get("path")
        if path in module_types:
            module["type"] = module_types[path]
    modules_path.write_text(json.dumps(modules, indent=2) + "\n")

    model_config = json.loads((model_dir / "config.json").read_text())
    text_config = model_config.get("text_config", {})
    embedding_dimension = text_config.get("hidden_size", model_config.get("hidden_size"))
    pooling_config = {
        "word_embedding_dimension": embedding_dimension,
        "pooling_mode_cls_token": cfg.pooling == "cls",
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": cfg.pooling == "avg",
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": cfg.pooling == "last",
        "include_prompt": True,
    }
    pooling_path = model_dir / "1_Pooling" / "config.json"
    pooling_path.parent.mkdir(parents=True, exist_ok=True)
    pooling_path.write_text(json.dumps(pooling_config, indent=2) + "\n")

    sentence_transformers_path = model_dir / "config_sentence_transformers.json"
    sentence_transformers_config = json.loads(sentence_transformers_path.read_text())
    prompts = sentence_transformers_config.setdefault("prompts", {})
    if cfg.query_prefix is not None:
        prompts["query"] = _automodel_collator_prefix(cfg.query_prefix)
    if cfg.passage_prefix is not None:
        prompts["document"] = _automodel_collator_prefix(cfg.passage_prefix)
    sentence_transformers_path.write_text(json.dumps(sentence_transformers_config, indent=2) + "\n")


def _wandb_config_from_env() -> dict[str, Any] | None:
    """Build AutoModel's native W&B block from the run environment."""
    enabled = os.environ.get("WANDB_ENABLED", "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    project = os.environ.get("WANDB_PROJECT")
    if not project:
        raise ValueError("WANDB_ENABLED requires WANDB_PROJECT")

    config: dict[str, Any] = {"project": project}
    optional_fields = {
        "entity": "WANDB_ENTITY",
        "group": "WANDB_GROUP",
        "job_type": "WANDB_JOB_TYPE",
        "dir": "WANDB_DIR",
    }
    for field, env_name in optional_fields.items():
        value = os.environ.get(env_name)
        if value:
            config[field] = value

    run_name = os.environ.get("WANDB_NAME")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if run_name:
        config["name"] = f"{run_name}-{slurm_job_id}" if slurm_job_id else run_name

    tags = [tag.strip() for tag in os.environ.get("WANDB_TAGS", "").split(",") if tag.strip()]
    if tags:
        config["tags"] = tags

    return config


def _count_training_examples(train_data_path: Path) -> int:
    """Count the number of training examples in a training data file.

    Args:
        train_data_path: Path to training JSON file.

    Returns:
        Number of training examples.
    """
    with open(train_data_path) as f:
        data = json.load(f)
    return len(data.get("data", []))


def _warn_if_negatives_sparse(train_data_path: Path, train_n_passages: int) -> None:
    """Warn if training data has fewer negatives than train_n_passages requires."""
    needed = train_n_passages - 1  # 1 positive + (n-1) negatives
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


def _auto_scale_hyperparams(cfg: FinetuneConfig, num_examples: int) -> tuple[int, int | None, int | None, int, int]:
    """Auto-scale training hyperparameters based on dataset size.

    Adjusts batch size, epochs, checkpoint frequency, and validation
    frequency when the user hasn't overridden defaults and the dataset
    is small enough to benefit.

    Args:
        cfg: Fine-tuning configuration (with user-specified or default values).
        num_examples: Number of training examples.

    Returns:
        Tuple of (global_batch_size, num_epochs, max_steps, checkpoint_every_steps,
        val_every_steps).
    """
    # --- Batch size ---
    # Default is 128; auto-scale down for small datasets so we get more steps
    if cfg.global_batch_size == 128 and num_examples < 2000:
        global_batch_size = max(16, min(64, num_examples // 8))
    else:
        global_batch_size = cfg.global_batch_size

    steps_per_epoch = max(1, num_examples // global_batch_size)
    total_steps = cfg.max_steps if cfg.max_steps is not None else steps_per_epoch * cfg.num_epochs

    # --- Checkpoint / validation frequency ---
    # Default is 100; cap so we get at least 3 checkpoints unless the
    # caller explicitly chooses sparse intervals for storage-constrained runs.
    if cfg.auto_scale_checkpoint_intervals and total_steps < cfg.checkpoint_every_steps * 3:
        checkpoint_every_steps = max(1, total_steps // 3)
    else:
        checkpoint_every_steps = cfg.checkpoint_every_steps

    if cfg.auto_scale_checkpoint_intervals and total_steps < cfg.val_every_steps * 3:
        val_every_steps = max(1, total_steps // 3)
    else:
        val_every_steps = cfg.val_every_steps

    return global_batch_size, cfg.num_epochs, cfg.max_steps, checkpoint_every_steps, val_every_steps


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


def _assert_optimizer_metadata_fp32(checkpoint_dir: Path) -> dict[str, int]:
    """Verify both Adam moments in a checkpoint produced by this training run.

    Args:
        checkpoint_dir: Trusted local checkpoint written by the current recipe.

    Returns:
        Number of FP32 tensors for each Adam moment.

    Raises:
        AssertionError: A moment is missing or has a non-FP32 dtype.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(checkpoint_dir / "optim").read_metadata()
    counts = {"exp_avg": 0, "exp_avg_sq": 0}
    for key, value in metadata.state_dict_metadata.items():
        prefix, _, name = str(key).rpartition(".")
        if not prefix.startswith("optim.state.") or name not in counts:
            continue
        dtype = getattr(getattr(value, "properties", None), "dtype", None)
        if dtype != torch.float32:
            raise AssertionError(f"Expected FP32 Adam moment {key}, got {dtype}")
        counts[name] += 1
    if not all(counts.values()):
        raise AssertionError(f"Missing Adam moment tensors: {counts}")
    return counts


def _load_automodel_config(cfg: FinetuneConfig, config_node_cls: type) -> tuple[Any, str]:
    """Load Automodel YAML after choosing an optimizer that is importable here."""
    import yaml

    base_config_name = (
        "mistral3_vl_biencoder_base.yaml" if cfg.model_family == "mistral3_vl" else "biencoder_base.yaml"
    )
    base_config_path = STAGE_PATH / base_config_name
    with open(base_config_path) as f:
        raw_config = yaml.safe_load(f)

    raw_config["seed"] = cfg.seed
    if cfg.model_family == "text" and cfg.tokenizer_force_default:
        raw_config["tokenizer"]["force_default"] = True
    if cfg.peft is not None:
        raw_config["peft"] = {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            **cfg.peft.model_dump(),
            "use_memory_efficient_lora": False,
            "use_triton": False,
        }
        raw_config["checkpoint"]["save_consolidated"] = False
    dataset = raw_config["dataset"] if cfg.model_family == "mistral3_vl" else raw_config["dataloader"]["dataset"]
    dataset["seed"] = cfg.seed

    wandb_config = _wandb_config_from_env()
    if wandb_config is not None:
        raw_config["wandb"] = wandb_config

    optimizer_backend = cfg.optimizer_backend
    te_available, te_error = (False, None)
    flash_available, flash_error = (False, None)
    if optimizer_backend in {"auto", "fused_adam"}:
        te_available, te_error = _can_import_fused_adam()
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
        flash_available, flash_error = _can_import_flash_adamw()
        if not flash_available:
            print("Error: optimizer_backend=flash_adamw requires flashoptim.", file=sys.stderr)
            if flash_error:
                print(f"  Import error: {flash_error}", file=sys.stderr)
            print(
                "  Install flashoptim, or run in an environment with Transformer Engine FusedAdam.",
                file=sys.stderr,
            )
            sys.exit(1)
        flash_optimizer = {
            "_target_": "flashoptim.FlashAdamW",
            "lr": raw_config.get("optimizer", {}).get("lr", cfg.learning_rate),
            "weight_decay": raw_config.get("optimizer", {}).get("weight_decay", cfg.weight_decay),
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "quantize": True,
            "compress_state_dict": False,
            "master_weight_bits": cfg.flash_adamw_master_weight_bits,
            "fused": True,
        }
        raw_config["optimizer"] = flash_optimizer
        raw_config.setdefault("model", {})["torch_dtype"] = "bfloat16"
    elif optimizer_backend == "torch_adamw":
        raw_config["optimizer"] = {
            "_target_": "torch.optim.AdamW",
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "fused": True,
        }
        raw_config.setdefault("model", {})["torch_dtype"] = "float32"
        raw_config.setdefault("distributed", {})["mp_policy"] = {
            "_target_": "torch.distributed.fsdp.MixedPrecisionPolicy",
            "param_dtype": "float32",
            "reduce_dtype": "float32",
            "output_dtype": "float32",
        }

    return config_node_cls(raw_config), optimizer_backend


def run_finetune(cfg: FinetuneConfig) -> Path:
    """Run embedding model fine-tuning using nemo-automodel.

    Args:
        cfg: Fine-tuning configuration.

    Returns:
        Path to final checkpoint directory.
    """
    # Validate inputs
    if cfg.peft is not None:
        validate_peft_runtime()
        validate_peft_base(cfg)
        export_path = cfg.checkpoint_dir / "LATEST" / "model" / "consolidated"
        if export_path.exists() or export_path.is_symlink():
            raise FileExistsError(f"PEFT export already exists: {export_path}")
    if not cfg.train_data_path.exists():
        print(f"Error: Training data not found: {cfg.train_data_path}", file=sys.stderr)
        print("       Please run stage1_data_prep first.", file=sys.stderr)
        sys.exit(1)

    if cfg.model_family == "mistral3_vl":
        validate_vl_training_data(
            cfg.train_data_path,
            required_negatives=cfg.train_n_passages - 1,
            require_mined_negatives=cfg.require_mined_negatives,
        )

    # Count training examples and check negative passage availability
    num_examples = _count_training_examples(cfg.train_data_path)
    _warn_if_negatives_sparse(cfg.train_data_path, cfg.train_n_passages)

    global_batch_size, num_epochs, max_steps, ckpt_every, val_every = _auto_scale_hyperparams(cfg, num_examples)

    steps_per_epoch = max(1, num_examples // global_batch_size)
    total_steps = max_steps if max_steps is not None else steps_per_epoch * num_epochs

    # Print training plan
    print("Training plan:")
    print(f"  Dataset:          {num_examples:,} examples")

    if global_batch_size != cfg.global_batch_size:
        print(
            f"  Batch size:       {global_batch_size} "
            f"(auto-scaled from {cfg.global_batch_size} - dataset < 2000 examples)"
        )
    else:
        print(f"  Batch size:       {global_batch_size}")
        if num_examples < 2000 and cfg.global_batch_size != 128:
            print("                    (note: auto-scaling skipped because batch size was explicitly set)")

    if max_steps is not None:
        print("  Epochs:           derived from exact step budget")
        print(f"  Max steps:        {max_steps}")
    else:
        print(f"  Epochs:           {num_epochs}")

    print(f"  Steps/epoch:      ~{steps_per_epoch}")
    total_steps_prefix = "" if max_steps is not None else "~"
    print(f"  Total steps:      {total_steps_prefix}{total_steps}")
    print(f"  LR schedule:      {cfg.lr_decay_style}, warmup={cfg.lr_warmup_steps}, peak={cfg.learning_rate}")
    print(f"  Checkpoint every: {ckpt_every} steps")
    print(f"  Validate every:   {val_every} steps")
    print()

    if total_steps < 50:
        print(
            f"Warning: Only {total_steps_prefix}{total_steps} total training steps. "
            "Dataset may be too small for meaningful fine-tuning.",
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
        from nemo_automodel.recipes.retrieval import TrainBiEncoderRecipe
    except ImportError as e:
        print("Error: Failed to import nemo-automodel. Is it installed?", file=sys.stderr)
        print("  Install with: pip install nemo-automodel", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load base config from nemo-automodel defaults. ConfigNode resolves _target_
    # imports during construction, so optimizer selection must happen on raw YAML.
    automodel_cfg, optimizer_backend = _load_automodel_config(cfg, ConfigNode)
    optimizer_detail = optimizer_backend
    if optimizer_backend == "flash_adamw":
        if cfg.flash_adamw_master_weight_bits is None:
            optimizer_detail = f"{optimizer_backend} (master weights disabled)"
        else:
            optimizer_detail = (
                f"{optimizer_backend} (bf16 model, {cfg.flash_adamw_master_weight_bits}-bit master weights)"
            )
    print(f"Optimizer:      {optimizer_detail}")
    print()

    # Apply overrides from our config
    # Model settings
    automodel_cfg.model.pretrained_model_name_or_path = cfg.base_model
    automodel_cfg.tokenizer.pretrained_model_name_or_path = cfg.base_model
    if cfg.model_family == "text":
        automodel_cfg.model.trust_remote_code = cfg.trust_remote_code
        automodel_cfg.tokenizer.trust_remote_code = cfg.trust_remote_code
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
    if cfg.model_family == "mistral3_vl":
        automodel_cfg.dataset.data_dir_list = [str(cfg.train_data_path)]
        automodel_cfg.dataset.n_passages = cfg.train_n_passages
        automodel_cfg.dataset.use_text_in_document = cfg.use_text_in_document
        automodel_cfg.tokenizer.q_max_length = cfg.query_max_length
        automodel_cfg.tokenizer.p_max_length = cfg.passage_max_length
        if cfg.query_prefix is not None:
            automodel_cfg.tokenizer.query_prefix = _automodel_collator_prefix(cfg.query_prefix)
        if cfg.passage_prefix is not None:
            automodel_cfg.tokenizer.passage_prefix = _automodel_collator_prefix(cfg.passage_prefix)
        automodel_cfg.tokenizer.pad_to_multiple_of = cfg.pad_to_multiple_of
        if cfg.image_longest_edge is not None:
            automodel_cfg.tokenizer.image_longest_edge = cfg.image_longest_edge
    else:
        automodel_cfg.dataloader.dataset.data_dir_list = [str(cfg.train_data_path)]
        automodel_cfg.dataloader.dataset.n_passages = cfg.train_n_passages
        automodel_cfg.dataloader.collate_fn.q_max_len = cfg.query_max_length
        automodel_cfg.dataloader.collate_fn.p_max_len = cfg.passage_max_length
        automodel_cfg.dataloader.collate_fn.query_prefix = _automodel_collator_prefix(cfg.query_prefix)
        automodel_cfg.dataloader.collate_fn.passage_prefix = _automodel_collator_prefix(cfg.passage_prefix)

    # Training settings — use auto-scaled values
    automodel_cfg.step_scheduler.num_epochs = num_epochs
    automodel_cfg.step_scheduler.max_steps = max_steps
    automodel_cfg.step_scheduler.global_batch_size = global_batch_size
    automodel_cfg.step_scheduler.local_batch_size = cfg.local_batch_size
    automodel_cfg.step_scheduler.ckpt_every_steps = ckpt_every
    automodel_cfg.step_scheduler.val_every_steps = val_every

    # Optimizer settings
    automodel_cfg.optimizer.lr = cfg.learning_rate
    automodel_cfg.optimizer.weight_decay = cfg.weight_decay
    automodel_cfg.lr_scheduler.lr_warmup_steps = cfg.lr_warmup_steps
    automodel_cfg.lr_scheduler.lr_decay_style = cfg.lr_decay_style

    # Model architecture
    automodel_cfg.model.pooling = cfg.pooling
    automodel_cfg.model.l2_normalize = cfg.l2_normalize
    if cfg.model_family == "mistral3_vl":
        automodel_cfg.model.is_causal = cfg.is_causal
        automodel_cfg.model.do_distributed_inbatch_negative = cfg.do_distributed_inbatch_negative
        automodel_cfg.model.detach_distributed_inbatch_negatives = cfg.detach_distributed_inbatch_negatives
    automodel_cfg.temperature = cfg.temperature

    # Checkpoint settings
    automodel_cfg.checkpoint.checkpoint_dir = str(cfg.checkpoint_dir)

    # Create and run the bi-encoder recipe
    recipe = TrainBiEncoderRecipe(automodel_cfg)
    recipe.setup()
    if optimizer_backend == "torch_adamw":
        for model_part in recipe.model_parts:
            model_part.float()
    recipe.run_train_validation_loop()

    if _is_rank_zero() and optimizer_backend == "torch_adamw":
        counts = _assert_optimizer_metadata_fp32(cfg.checkpoint_dir / "LATEST")
        print(f"Verified optimizer checkpoint Adam moments are FP32: {counts}")

    # Find the final checkpoint
    final_model_dir = cfg.checkpoint_dir / "LATEST" / "model" / "consolidated"
    if _is_rank_zero() and cfg.peft is not None:
        merge_peft_checkpoint(cfg, final_model_dir)
    elif _is_rank_zero() and cfg.model_family == "mistral3_vl":
        _repair_vllm_sentence_transformers_metadata(final_model_dir, cfg)

    print("\nFine-tuning complete!")
    print(f"   Checkpoint: {cfg.checkpoint_dir}")
    print(f"   Model:      {final_model_dir}")

    # Save artifact (registers with artifact registry if kit.init() was called)
    if _is_rank_zero():
        try:
            from nemotron.kit.artifacts.embed import EmbedModelArtifact

            artifact = EmbedModelArtifact(
                path=final_model_dir,
                base_model=cfg.base_model,
                training_examples=num_examples,
                num_epochs=num_epochs,
                max_steps=max_steps,
                global_batch_size=global_batch_size,
                learning_rate=cfg.learning_rate,
                temperature=cfg.temperature,
            )
            artifact.save(name="embed/model")
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
