# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Advisory checkpoint resource checks for full Nemotron 3.5 fine-tuning."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_GIB = 1024**3
# BF16 tensor bytes measured at the specified public model revisions.
_VALIDATED_MODELS = {
    "nvidia/Nemotron-3.5-Embed-1B-BF16": (
        "c0f2e8ab5367f5090e6eed608f607bafe76d707b",
        3_109_421_056,
    ),
    "nvidia/Nemotron-3.5-Rerank-8B-BF16": (
        "39aa01525864f736e63f1d5742fc4cf59e754abd",
        16_762_318_848,
    ),
}


def _model_bytes(base_model: str, revision: str | None) -> tuple[int, str] | None:
    """Read small local/cached metadata only; never download model artifacts."""
    try:
        model_dir = Path(base_model)
        if not model_dir.is_dir():
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(base_model, "model.safetensors.index.json", revision=revision)
            if not isinstance(cached, str):
                raise FileNotFoundError
            model_dir = Path(cached).parent
        config = json.loads((model_dir / "config.json").read_text())
        dtype = config.get("dtype", config.get("torch_dtype"))
        if dtype is None:
            text_config = config.get("text_config", {})
            dtype = text_config.get("dtype", text_config.get("torch_dtype"))
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        size = index["metadata"]["total_size"]
        if dtype in ("bfloat16", "float16") and isinstance(size, int) and not isinstance(size, bool) and size > 0:
            return size, "local or cached 16-bit model metadata"
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    validated = _VALIDATED_MODELS.get(base_model)
    if validated is not None and revision in (None, validated[0]):
        return validated[1], f"validated revision {validated[0]} (same architecture assumed)"
    return None


def _available_host_memory() -> int | None:
    """Bound Linux MemAvailable by remaining cgroup v2 memory where readable."""
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(fields["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        return None
    cgroup_root = Path("/sys/fs/cgroup")
    candidates = {cgroup_root}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative = Path(line[3:].lstrip("/"))
                if ".." not in relative.parts:
                    current = cgroup_root / relative
                    candidates.update((current, *current.parents))
    except OSError:
        pass
    for directory in candidates:
        if directory != cgroup_root and cgroup_root not in directory.parents:
            continue
        try:
            maximum = int((directory / "memory.max").read_text().strip())
            used = int((directory / "memory.current").read_text().strip())
        except (OSError, ValueError):
            continue
        available = min(available, max(0, maximum - used))
    return available


def _optimizer_bytes(model_bytes: int, config: Mapping[str, Any]) -> int | None:
    """Estimate tensor state for the optimizers configured by these recipes."""
    target = str(config.get("_target_", ""))
    parameters = model_bytes // 2
    # ConfigNode resolves the public re-export, then serializes its defining module.
    flash_adamw = target in ("flashoptim.FlashAdamW", "flashoptim.optimizers.FlashAdamW")
    if flash_adamw and config.get("compress_state_dict") is False:
        # Quantized moments are dequantized to FP32 for uncompressed saves.
        # Unquantized moments retain the BF16 parameter dtype.
        moment_bytes = 8 if config.get("quantize", True) else 4
        master_bits = config.get("master_weight_bits", 24)
        if master_bits not in (None, 24, 32):
            return None
        # FSDP serialization casts 24-bit uint8 correction buffers to the BF16
        # parameter dtype; both 24-bit and 32-bit modes save two bytes per value.
        correction_bytes = 0 if master_bits is None else 2
        return parameters * (moment_bytes + correction_bytes)
    if target == "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam":
        return parameters * (8 + (4 if config.get("master_weights", False) else 0))
    return None


def report_checkpoint_resources(
    checkpoint_dir: Path,
    *,
    base_model: str,
    model_revision: str | None,
    optimizer_config: Mapping[str, Any],
    emit: bool = True,
) -> None:
    """Report pre-load headroom; estimates exclude caches, data, and other jobs.

    Call on every rank, with ``emit`` enabled only on rank zero. These estimates
    are advisory: optimizer buffers, filesystem overhead, protected best
    checkpoints, and concurrent jobs can require more disk or host memory.
    """
    destination = checkpoint_dir.expanduser().resolve()
    existing = destination
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    try:
        free = shutil.disk_usage(existing).free
    except OSError as error:
        if emit:
            print(f"Warning: checkpoint disk preflight unavailable for {destination}: {error}")
        return
    if free == 0:
        raise ValueError(f"Checkpoint filesystem has no available disk space: {destination}")
    validated = _VALIDATED_MODELS.get(base_model)
    if validated is not None and model_revision == validated[0] and free < validated[1]:
        raise ValueError(
            f"Checkpoint filesystem has {free / _GIB:.1f} GiB available, less than the validated "
            f"model payload alone ({validated[1] / _GIB:.1f} GiB), before optimizer state or export. "
            f"Use a checkpoint_dir on a larger filesystem: {destination}"
        )
    if not emit:
        return
    print(f"Checkpoint resources: {destination}")
    print(f"  Available disk: {free / _GIB:.1f} GiB on {existing}")
    memory = _available_host_memory()
    if memory is not None:
        print(f"  Available host RAM: {memory / _GIB:.1f} GiB (including readable cgroup v2 limits)")
    metadata = _model_bytes(base_model, model_revision)
    optimizer_bytes = _optimizer_bytes(metadata[0], optimizer_config) if metadata else None
    if metadata is None or optimizer_bytes is None:
        print("  Checkpoint estimate unavailable for this model/optimizer override; verify resources before training.")
        return
    weights, source = metadata
    # The final checkpoint contains both distributed model shards and an export.
    checkpoint_bytes = 2 * weights + optimizer_bytes
    has_previous = (destination / "LATEST").exists()
    additional = checkpoint_bytes if has_previous else 2 * checkpoint_bytes
    print(f"  Estimated full checkpoint with consolidated export: {checkpoint_bytes / _GIB:.1f} GiB")
    print(f"  Estimated additional disk through the next save/retention cycle: {additional / _GIB:.1f} GiB")
    print(f"  Estimate source: {source}. Assumes full BF16 fine-tuning.")
    print("  A previous checkpoint remains until the next save completes, even with max_recent_checkpoints=1.")
    print("  Existing files are already deducted from available disk; caches, data, and protected best checkpoints")
    print("  require additional capacity. These estimates are not a resource guarantee.")
    print(
        f"  Default writer may stage about {optimizer_bytes / _GIB:.1f} GiB of optimizer tensors across local ranks,"
    )
    print("  plus model state and serialization overhead; leave additional host RAM available.")
    if free < additional:
        print("Warning: available disk is below estimated checkpoint headroom; use a larger checkpoint_dir.")
    if memory is not None and memory < optimizer_bytes:
        print("Warning: available host RAM is below estimated optimizer staging; checkpoint saving may fail.")
