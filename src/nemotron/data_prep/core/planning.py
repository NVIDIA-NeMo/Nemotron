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

"""Shard plan creation with size-balanced file assignment."""

import hashlib
import heapq
import json
import logging
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ray
from fsspec import AbstractFileSystem

from nemotron.data_prep.config import (
    DatasetConfig,
    FileInfo,
    InternalOutputConfig,
    InternalTokenizerConfig,
    ShardAssignment,
    ShardPlan,
)
from nemotron.data_prep.utils.discovery import discover_input_files
from nemotron.data_prep.utils.filesystem import read_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanRequest:
    """Inputs required to create a shard plan."""

    dataset_config: DatasetConfig
    num_shards: int
    config_hash: str
    tokenizer_config: InternalTokenizerConfig | None = None
    output_config: InternalOutputConfig | None = None
    transform_fingerprint: str | None = None


def create_size_balanced_assignments(
    files: list[FileInfo],
    num_shards: int,
) -> list[ShardAssignment]:
    """
    Deterministically assign files to shards with size balancing.

    Algorithm: greedy bin-packing with heap (or round-robin if sizes unavailable)
    - Sort files by (size desc, path asc) for determinism
    - If sizes available: use min-heap to find shard with smallest total (O(n log k))
    - If sizes unavailable (all 0): use round-robin for even distribution
    - Tie-break by shard index for determinism
    """
    # Sort: largest files first, then by path for determinism
    sorted_files = sorted(files, key=lambda f: (-f.size, f.path))

    # Initialize assignments
    assignments = [
        ShardAssignment(shard_index=i, files=[], total_bytes=0) for i in range(num_shards)
    ]

    # Check if file sizes are available
    has_sizes = any(f.size > 0 for f in files)

    if has_sizes:
        # Use min-heap for O(n log k) assignment instead of O(n * k)
        # Heap entries: (total_bytes, shard_index) - shard_index for tie-breaking
        heap: list[tuple[int, int]] = [(0, i) for i in range(num_shards)]
        heapq.heapify(heap)

        for file_info in sorted_files:
            total_bytes, shard_idx = heapq.heappop(heap)
            assignments[shard_idx].files.append(file_info)
            assignments[shard_idx].total_bytes += file_info.size
            heapq.heappush(heap, (total_bytes + file_info.size, shard_idx))
    else:
        # Round-robin assignment when sizes are unavailable
        for i, file_info in enumerate(sorted_files):
            shard_idx = i % num_shards
            assignments[shard_idx].files.append(file_info)
            # Still accumulate total_bytes even if sizes are 0 (for consistency)
            assignments[shard_idx].total_bytes += file_info.size

    # Sort files within each shard by path for deterministic processing order
    for assignment in assignments:
        assignment.files.sort(key=lambda f: f.path)

    return assignments


def _is_local_path(model: str) -> bool:
    """Check if model refers to a local filesystem path."""
    return (
        model.startswith("/")
        or model.startswith("./")
        or model.startswith("../")
        or Path(model).exists()
    )


def resolve_tokenizer(config: InternalTokenizerConfig) -> dict:
    """Resolve tokenizer to immutable revision."""
    result = {
        "type": config.type,
        "model": config.model,
        "add_eos": config.add_eos,
        "add_bos": config.add_bos,
        "trust_remote_code": config.trust_remote_code,
    }

    if config.type == "huggingface":
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        is_local = _is_local_path(config.model)

        if is_local:
            # Local model - no revision needed
            result["resolved_revision"] = "local"
            revision_for_tokenizer = None
        else:
            # HuggingFace model - resolve to immutable SHA
            api = HfApi()
            try:
                model_info = api.model_info(config.model, revision=config.revision)
                result["resolved_revision"] = model_info.sha
                revision_for_tokenizer = model_info.sha
            except Exception:
                # api.model_info() failed but this is a HF model, not local
                # Use the user-specified revision (or None for default)
                result["resolved_revision"] = config.revision
                revision_for_tokenizer = config.revision

        # Get vocab size
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=revision_for_tokenizer,
            trust_remote_code=config.trust_remote_code,
        )
        result["vocab_size"] = len(tokenizer)

    elif config.type == "sentencepiece":
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=config.model)
        result["resolved_revision"] = "local"
        result["vocab_size"] = sp.vocab_size()

    return result


def compute_source_fingerprint(files: list[FileInfo], dataset_config: DatasetConfig) -> str:
    """
    Compute fingerprint from file list and dataset identity.

    Includes:
    - File path, size, etag
    - mtime for local files (detects in-place modifications)
    - version_id for S3/GCS versioned objects
    - HF repo_id and revision for HF files (dataset identity)
    """
    components = []

    # Include dataset identity for HF sources
    if dataset_config.path.startswith("hf://"):
        hf_path = dataset_config.path[5:]
        components.append(f"hf_repo:{hf_path}")
        # First file has the resolved revision
        if files and files[0].hf_revision:
            components.append(f"hf_revision:{files[0].hf_revision}")

    for f in sorted(files, key=lambda x: x.path):
        # Build comprehensive fingerprint component
        parts = [f.path, str(f.size), f.etag or ""]

        # Add mtime for local files (stronger fingerprint)
        if f.mtime is not None:
            parts.append(f"mtime:{f.mtime}")

        # Add version_id for S3/GCS versioned objects
        if f.version_id is not None:
            parts.append(f"ver:{f.version_id}")

        # Add HF identity
        if f.hf_repo_id is not None:
            parts.append(f"hf:{f.hf_repo_id}@{f.hf_revision}")

        components.append(":".join(parts))

    content = "\n".join(components)
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


def create_plan(request: PlanRequest, fs: AbstractFileSystem) -> ShardPlan:
    """Create deterministic shard plan for tokenizer and non-tokenizer pipelines."""
    files = discover_input_files(request.dataset_config, fs)
    if not files:
        raise ValueError(f"No input files found for {request.dataset_config.name}")

    resolved_tokenizer = (
        resolve_tokenizer(request.tokenizer_config)
        if request.tokenizer_config is not None
        else {"type": "none"}
    )
    source_fingerprint = compute_source_fingerprint(files, request.dataset_config)
    assignments = create_size_balanced_assignments(files, request.num_shards)
    determinism_constraints = _build_determinism_constraints(
        has_tokenizer=request.tokenizer_config is not None,
        transform_fingerprint=request.transform_fingerprint,
    )

    plan_content = json.dumps(
        {
            "dataset_name": request.dataset_config.name,
            "num_shards": request.num_shards,
            "source_fingerprint": source_fingerprint,
            "resolved_tokenizer": resolved_tokenizer,
            "determinism_constraints": determinism_constraints,
            "config_hash": request.config_hash,
            "file_paths": sorted(f.path for f in files),
        },
        sort_keys=True,
    )
    plan_hash = hashlib.sha256(plan_content.encode()).hexdigest()[:16]

    return ShardPlan(
        version="1.0",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        plan_hash=plan_hash,
        dataset_name=request.dataset_config.name,
        num_shards=request.num_shards,
        source_fingerprint=source_fingerprint,
        config_hash=request.config_hash,
        determinism_constraints=determinism_constraints,
        resolved_tokenizer=resolved_tokenizer,
        file_assignments=assignments,
    )


def _build_determinism_constraints(
    *,
    has_tokenizer: bool,
    transform_fingerprint: str | None,
) -> dict[str, str]:
    if has_tokenizer:
        import tokenizers
        import transformers

        return {
            "ray_version": ray.__version__,
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
            "input_file_order": "size_desc_path_asc",
            "processing_order": "sequential_within_shard",
        }

    import pyarrow

    constraints: dict[str, str] = {
        "ray_version": ray.__version__,
        "pyarrow_version": pyarrow.__version__,
        "processing_order": "sequential_within_shard",
    }
    if transform_fingerprint is not None:
        constraints["transform_fingerprint"] = transform_fingerprint
    return constraints


def verify_binidx_output(receipt: dict, shard_dir: str, fs: AbstractFileSystem) -> bool:
    """Pretrain: bin/idx files exist for non-empty shards."""
    if int(receipt.get("stats", {}).get("num_sequences", 0) or 0) == 0:
        return True
    files = receipt.get("files", {}) or {}
    bin_path = ((files.get("bin") or {}).get("path")) or ""
    idx_path = ((files.get("idx") or {}).get("path")) or ""
    if not bin_path or not idx_path:
        return False
    return fs.exists(f"{shard_dir}/{bin_path}") and fs.exists(f"{shard_dir}/{idx_path}")


def verify_jsonl_output(receipt: dict, shard_dir: str, fs: AbstractFileSystem) -> bool:
    """JSONL: output file exists for non-empty shards."""
    if int(receipt.get("stats", {}).get("num_records", 0) or 0) == 0:
        return True
    output_file = receipt.get("output_file")
    if not output_file:
        return False
    return fs.exists(f"{shard_dir}/{output_file}")


def verify_parquet_output(receipt: dict, shard_dir: str, fs: AbstractFileSystem) -> bool:
    """SFT: parquet file exists for non-empty shards."""
    if int(receipt.get("stats", {}).get("num_sequences", 0) or 0) == 0:
        return True
    parquet_path = ((receipt.get("files", {}).get("parquet") or {}).get("path")) or ""
    if not parquet_path:
        return False
    return fs.exists(f"{shard_dir}/{parquet_path}")


def create_shard_plan(
    dataset_config: DatasetConfig,
    output_config: InternalOutputConfig,
    tokenizer_config: InternalTokenizerConfig,
    config_hash: str,
    fs: AbstractFileSystem,
) -> ShardPlan:
    """Backward-compatible wrapper for tokenizer-based plan creation."""
    return create_plan(
        PlanRequest(
            dataset_config=dataset_config,
            num_shards=output_config.num_shards,
            config_hash=config_hash,
            tokenizer_config=tokenizer_config,
            output_config=output_config,
        ),
        fs,
    )


def get_pending_shards(
    plan: ShardPlan,
    receipts_dir: str,
    fs: AbstractFileSystem,
    verify_output: Callable[[dict, str, AbstractFileSystem], bool] | None = None,
) -> list[int]:
    """Determine which shard indices still need processing."""
    completed_indices: set[int] = set()
    shard_dir = str(Path(receipts_dir).parent)
    verifier = verify_output or verify_binidx_output

    try:
        receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
    except Exception:
        receipt_files = []

    for receipt_path in receipt_files:
        try:
            receipt = read_json(fs, receipt_path)
            if receipt.get("plan_hash") != plan.plan_hash:
                continue
            if receipt.get("status") != "completed":
                continue
            if verifier and not verifier(receipt, shard_dir, fs):
                continue
            completed_indices.add(int(receipt["shard_index"]))
        except Exception as e:
            logger.warning(f"Failed to parse receipt {receipt_path}: {e}")

    all_indices = set(range(plan.num_shards))
    return sorted(all_indices - completed_indices)


def get_sampled_shard_indices(
    num_shards: int,
    dataset_name: str,
    sample_spec: str | int,
    seed: int = 42,
) -> set[int]:
    """
    Deterministically select shard indices for sampling.

    Preserves "skip compute" for non-selected shards.
    """
    # Derive per-dataset seed using hashlib for cross-run determinism
    # (Python's hash() is randomized by default)
    seed_str = f"{seed}:{dataset_name}"
    dataset_seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(dataset_seed)

    all_indices = list(range(num_shards))

    if isinstance(sample_spec, str) and sample_spec.endswith("%"):
        # Percentage of shards
        fraction = float(sample_spec.rstrip("%")) / 100
        k = max(1, int(num_shards * fraction))
    else:
        # Fixed count of shards
        k = min(int(sample_spec), num_shards)

    # Deterministic selection
    selected = set(rng.sample(all_indices, k))
    return selected


def apply_shard_sampling(
    pending_indices: list[int],
    plan: ShardPlan,
    sample_spec: str | int | None,
    seed: int,
) -> list[int]:
    """Filter pending indices by sampling."""
    if sample_spec is None:
        return pending_indices

    sampled = get_sampled_shard_indices(
        plan.num_shards,
        plan.dataset_name,
        sample_spec,
        seed,
    )

    return [i for i in pending_indices if i in sampled]


def create_jsonl_shard_plan(
    *,
    dataset_config: DatasetConfig,
    num_shards: int,
    config_hash: str,
    fs: AbstractFileSystem,
    transform_fingerprint: str,
) -> ShardPlan:
    """Backward-compatible wrapper for non-tokenizer JSONL plan creation."""
    return create_plan(
        PlanRequest(
            dataset_config=dataset_config,
            num_shards=num_shards,
            config_hash=config_hash,
            tokenizer_config=None,
            transform_fingerprint=transform_fingerprint,
        ),
        fs,
    )


def get_pending_jsonl_shards(
    plan: ShardPlan,
    receipts_dir: str,
    fs: AbstractFileSystem,
) -> list[int]:
    """Backward-compatible wrapper for JSONL pending scan."""
    return get_pending_shards(plan, receipts_dir, fs, verify_output=verify_jsonl_output)


def serialize_shard_plan(plan: ShardPlan) -> dict:
    """Serialize ShardPlan to JSON-serializable dict."""
    result = asdict(plan)
    # Convert FileInfo objects in assignments
    for assignment in result["file_assignments"]:
        assignment["files"] = [
            asdict(f) if hasattr(f, "__dict__") else f for f in assignment["files"]
        ]
    return result
