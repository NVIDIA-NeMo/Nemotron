#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Test script to investigate memory usage in the ChatSFT packing stage.
# Creates synthetic spool data and exercises the pack pipeline to identify OOM sources.
#
# Usage:
#   uv run python scripts/test_pack_memory.py --num-sequences 150000 --avg-seq-length 2000
#
# This mimics a single shard from the full SFT pipeline (~440M tokens per shard).

from __future__ import annotations

import argparse
import gc
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


def log_memory(label: str) -> float:
    """Log current memory usage and return RSS in GB."""
    try:
        import psutil
        process = psutil.Process()
        rss_gb = process.memory_info().rss / (1024**3)
        print(f"[Memory] {label}: RSS={rss_gb:.2f} GB")
        return rss_gb
    except ImportError:
        print(f"[Memory] {label}: psutil not available")
        return 0.0


def create_synthetic_spool(
    spool_dir: str,
    num_sequences: int,
    avg_seq_length: int,
    length_std: int = 500,
    seed: int = 42,
) -> dict:
    """Create synthetic spool data mimicking real tokenized sequences."""
    from fsspec.implementations.local import LocalFileSystem
    from nemotron.data_prep.packing.spool import SequenceSpoolPaths, SequenceSpoolWriter

    print(f"\n=== Creating synthetic spool ===")
    print(f"  Sequences: {num_sequences:,}")
    print(f"  Avg length: {avg_seq_length}")
    print(f"  Spool dir: {spool_dir}")

    log_memory("Before spool creation")

    fs = LocalFileSystem()
    paths = SequenceSpoolPaths.for_root(spool_dir)
    writer = SequenceSpoolWriter(fs=fs, paths=paths)

    rng = np.random.default_rng(seed)
    total_tokens = 0

    # Generate sequences in batches to avoid memory spikes
    batch_size = 10000
    for batch_start in range(0, num_sequences, batch_size):
        batch_end = min(batch_start + batch_size, num_sequences)
        batch_count = batch_end - batch_start

        # Generate random lengths for this batch
        lengths = rng.normal(avg_seq_length, length_std, batch_count).astype(int)
        lengths = np.clip(lengths, 100, avg_seq_length * 3)  # Clamp to reasonable range

        for length in lengths:
            # Generate random tokens and mask
            input_ids = rng.integers(0, 50000, size=length, dtype=np.int32)
            loss_mask = rng.integers(0, 2, size=length, dtype=np.uint8)
            writer.append(input_ids, loss_mask)
            total_tokens += length

        if (batch_end % 50000) == 0 or batch_end == num_sequences:
            print(f"  Created {batch_end:,}/{num_sequences:,} sequences ({total_tokens:,} tokens)")

    log_memory("After writing sequences")

    manifest = writer.finalize(extra_manifest={
        "shard_id": "test_shard",
        "shard_index": 0,
        "pack_size": 4096,
        "algorithm": "first_fit_decreasing",
        "dtype": "int32",
        "tokenization_stats": {
            "num_input_rows": num_sequences,
            "num_output_sequences": num_sequences,
        }
    })

    log_memory("After spool finalize")

    return {
        "num_sequences": num_sequences,
        "total_tokens": total_tokens,
        "spool_dir": spool_dir,
    }


def test_pack_from_spool(
    spool_dir: str,
    output_dir: str,
    pack_size: int = 4096,
    algorithm: str = "first_fit_decreasing",
) -> dict:
    """Test the packing stage using the actual core function."""
    from fsspec.implementations.local import LocalFileSystem
    from nemotron.data_prep.chat_sft_shard_core import process_chat_sft_pack_from_spool_core

    print(f"\n=== Testing pack from spool (using actual core function) ===")
    log_memory("Start of pack test")

    fs = LocalFileSystem()
    os.makedirs(output_dir, exist_ok=True)
    receipts_dir = os.path.join(output_dir, ".receipts")
    os.makedirs(receipts_dir, exist_ok=True)

    print("\n--- Calling process_chat_sft_pack_from_spool_core ---")
    log_memory("Before core function")

    stats = process_chat_sft_pack_from_spool_core(
        shard_index=0,
        output_dir=output_dir,
        receipts_dir=receipts_dir,
        spool_dir=spool_dir,
        output_fs=fs,
        pack_size=pack_size,
        algorithm=algorithm,
        dtype=np.dtype("int32"),
        seed=42,
    )

    log_memory("After core function returned")

    # Additional cleanup
    gc.collect()
    log_memory("After gc.collect()")

    npy_path = os.path.join(output_dir, "shard_000000.npy")
    npy_size = os.path.getsize(npy_path) if os.path.exists(npy_path) else 0

    return {
        "num_sequences": stats.get("num_sequences", 0),
        "num_bins": stats.get("num_packed_sequences", 0),
        "npy_size": npy_size,
        "total_tokens": stats.get("total_tokens", 0),
    }


def test_ray_simulation(
    spool_dir: str,
    output_dir: str,
    pack_size: int = 4096,
) -> None:
    """Simulate the Ray pipeline behavior with memory logging."""
    import ray

    print(f"\n=== Simulating Ray pipeline behavior ===")
    log_memory("Before ray.init()")

    # Initialize Ray locally
    if not ray.is_initialized():
        ray.init(num_cpus=4, object_store_memory=2 * 1024**3)  # 2GB object store
    log_memory("After ray.init()")

    # Run the pack as a Ray task (simulates actor behavior)
    @ray.remote
    def pack_task(spool_dir: str, output_dir: str, pack_size: int) -> dict:
        return test_pack_from_spool(spool_dir, output_dir, pack_size)

    print("\n--- Running pack as Ray task ---")
    result_ref = pack_task.remote(spool_dir, output_dir, pack_size)
    result = ray.get(result_ref)
    log_memory("After ray.get() of pack task")

    # Simulate what happens after pipeline
    print("\n--- Simulating post-pipeline ---")
    del result_ref
    log_memory("After del result_ref")

    gc.collect()
    log_memory("After gc.collect()")

    # Wait to see if delayed cleanup causes issues
    print("\n--- Waiting 30s to observe delayed memory behavior ---")
    for i in range(6):
        time.sleep(5)
        log_memory(f"After {(i+1)*5}s wait")

    ray.shutdown()
    log_memory("After ray.shutdown()")


def test_multiple_shards(
    num_shards: int,
    num_sequences: int,
    avg_seq_length: int,
    pack_size: int = 4096,
) -> None:
    """Test processing multiple shards sequentially to check memory accumulation."""
    print(f"\n{'='*60}")
    print(f"Testing {num_shards} sequential shards")
    print(f"{'='*60}")

    log_memory("Initial state before multi-shard test")

    temp_dir = tempfile.mkdtemp(prefix="multi_shard_test_")

    try:
        for shard_idx in range(num_shards):
            print(f"\n--- Shard {shard_idx + 1}/{num_shards} ---")
            spool_dir = os.path.join(temp_dir, f"spool_{shard_idx}")
            output_dir = os.path.join(temp_dir, f"output_{shard_idx}")

            # Create spool
            create_synthetic_spool(
                spool_dir=spool_dir,
                num_sequences=num_sequences,
                avg_seq_length=avg_seq_length,
            )
            log_memory(f"After creating spool {shard_idx}")

            # Process shard
            test_pack_from_spool(spool_dir, output_dir, pack_size)
            log_memory(f"After processing shard {shard_idx}")

            # Force cleanup
            gc.collect()
            log_memory(f"After gc.collect() for shard {shard_idx}")

    finally:
        print(f"\nCleaning up temp dir: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    log_memory("Final state after multi-shard test")


def main():
    parser = argparse.ArgumentParser(description="Test memory usage in ChatSFT packing")
    parser.add_argument("--num-sequences", type=int, default=150000,
                        help="Number of sequences to generate (default: 150000, ~1 shard)")
    parser.add_argument("--avg-seq-length", type=int, default=2000,
                        help="Average sequence length in tokens (default: 2000)")
    parser.add_argument("--pack-size", type=int, default=4096,
                        help="Pack size for packing (default: 4096)")
    parser.add_argument("--with-ray", action="store_true",
                        help="Also test with Ray to simulate actor behavior")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files for inspection")
    parser.add_argument("--multi-shard", type=int, default=0,
                        help="Test multiple shards sequentially (0=disabled)")
    args = parser.parse_args()

    print("=" * 60)
    print("ChatSFT Packing Memory Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Sequences: {args.num_sequences:,}")
    print(f"  Avg length: {args.avg_seq_length}")
    print(f"  Expected tokens: ~{args.num_sequences * args.avg_seq_length:,}")
    print(f"  Pack size: {args.pack_size}")
    print(f"  With Ray: {args.with_ray}")
    print(f"  Multi-shard: {args.multi_shard}")

    log_memory("Initial state")

    # Multi-shard test mode
    if args.multi_shard > 0:
        test_multiple_shards(
            num_shards=args.multi_shard,
            num_sequences=args.num_sequences,
            avg_seq_length=args.avg_seq_length,
            pack_size=args.pack_size,
        )
        return

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="pack_memory_test_")
    spool_dir = os.path.join(temp_dir, "spool")
    output_dir = os.path.join(temp_dir, "output")

    try:
        # Create synthetic spool
        spool_info = create_synthetic_spool(
            spool_dir=spool_dir,
            num_sequences=args.num_sequences,
            avg_seq_length=args.avg_seq_length,
        )
        print(f"\nSpool created: {spool_info['total_tokens']:,} tokens")

        gc.collect()
        log_memory("After spool creation + gc")

        if args.with_ray:
            # Test with Ray
            test_ray_simulation(spool_dir, output_dir, args.pack_size)
        else:
            # Test without Ray
            pack_info = test_pack_from_spool(spool_dir, output_dir, args.pack_size)
            print(f"\nPacking complete: {pack_info['num_bins']:,} bins")

        print("\n" + "=" * 60)
        print("Test complete!")
        log_memory("Final state")

    finally:
        if not args.keep_temp:
            print(f"\nCleaning up temp dir: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nKeeping temp dir: {temp_dir}")


if __name__ == "__main__":
    main()
