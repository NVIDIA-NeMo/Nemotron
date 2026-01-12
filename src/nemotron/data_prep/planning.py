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
from dataclasses import asdict
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
from nemotron.data_prep.discovery import discover_input_files
from nemotron.data_prep.filesystem import read_json

logger = logging.getLogger(__name__)


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


def create_shard_plan(
    dataset_config: DatasetConfig,
    output_config: InternalOutputConfig,
    tokenizer_config: InternalTokenizerConfig,
    config_hash: str,
    fs: AbstractFileSystem,
) -> ShardPlan:
    """Create deterministic shard plan."""
    import tokenizers
    import transformers

    # Discover input files
    files = discover_input_files(dataset_config, fs)

    if not files:
        raise ValueError(f"No input files found for {dataset_config.name}")

    # Clamp requested shards to avoid empty assignments
    # (each shard should have at least one file assigned)
    requested_shards = output_config.num_shards
    effective_shards = max(1, min(requested_shards, len(files)))

    if effective_shards < requested_shards:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Dataset '{dataset_config.name}' has {len(files)} files but "
            f"{requested_shards} shards requested. Using {effective_shards} shards "
            f"to avoid empty outputs. Consider using 'shard_size' instead of "
            f"explicit 'num_shards' for blends with varied dataset sizes."
        )

    # Resolve tokenizer to immutable revision
    resolved_tokenizer = resolve_tokenizer(tokenizer_config)

    # Compute fingerprints (includes dataset identity for HF sources)
    source_fingerprint = compute_source_fingerprint(files, dataset_config)

    # Create size-balanced assignments
    assignments = create_size_balanced_assignments(files, effective_shards)

    # Determinism constraints
    determinism_constraints = {
        "ray_version": ray.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "input_file_order": "size_desc_path_asc",
        "processing_order": "sequential_within_shard",
    }

    # Compute plan hash
    plan_content = json.dumps(
        {
            "dataset_name": dataset_config.name,
            "num_shards": effective_shards,
            "source_fingerprint": source_fingerprint,
            "resolved_tokenizer": resolved_tokenizer,
            "determinism_constraints": determinism_constraints,
            "config_hash": config_hash,
            "file_paths": sorted([f.path for f in files]),
        },
        sort_keys=True,
    )
    plan_hash = hashlib.sha256(plan_content.encode()).hexdigest()[:16]

    return ShardPlan(
        version="1.0",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        plan_hash=plan_hash,
        dataset_name=dataset_config.name,
        num_shards=effective_shards,
        source_fingerprint=source_fingerprint,
        config_hash=config_hash,
        determinism_constraints=determinism_constraints,
        resolved_tokenizer=resolved_tokenizer,
        file_assignments=assignments,
    )


def get_pending_shards(
    plan: ShardPlan,
    receipts_dir: str,
    fs: AbstractFileSystem,
) -> list[int]:
    """Determine which shard indices still need processing."""
    completed_indices: set[int] = set()

    try:
        receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
    except FileNotFoundError:
        receipt_files = []
    except Exception:
        receipt_files = []

    for receipt_path in receipt_files:
        try:
            receipt = read_json(fs, receipt_path)

            # Verify receipt belongs to current plan
            if receipt.get("plan_hash") != plan.plan_hash:
                continue

            if receipt.get("status") != "completed":
                continue

            shard_index = receipt["shard_index"]

            # For non-empty shards, verify files exist
            if receipt["stats"]["num_sequences"] > 0:
                shard_dir = str(Path(receipts_dir).parent)
                bin_path = f"{shard_dir}/{receipt['files']['bin']['path']}"
                idx_path = f"{shard_dir}/{receipt['files']['idx']['path']}"

                if not (fs.exists(bin_path) and fs.exists(idx_path)):
                    continue

            completed_indices.add(shard_index)

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


def serialize_shard_plan(plan: ShardPlan) -> dict:
    """Serialize ShardPlan to JSON-serializable dict."""
    result = asdict(plan)
    # Convert FileInfo objects in assignments
    for assignment in result["file_assignments"]:
        assignment["files"] = [
            asdict(f) if hasattr(f, "__dict__") else f for f in assignment["files"]
        ]
    return result
