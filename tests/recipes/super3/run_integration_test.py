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

"""Super3 integration test pipeline.

Runs the full Super3 integration test end-to-end: data preparation followed
by training with a tiny (~7M param) Super3-architecture model on a single GPU.

The pipeline chains stages via wandb artifacts:
  1. data prep pretrain  (--config tiny)  → super3-pretrain-data-tiny artifact
  2. data prep sft       (--config tiny)  → super3-sft-data artifact
  3. data prep rl rlvr                    → super3-rl-rlvr1-data artifact
  4. pretrain             (--config test)  → super3-pretrain-model-tiny artifact
  5. sft                  (--config test)  → super3-sft-model-tiny artifact
  6. rl preflight         (--config test)  → validates RL infra (Ray, env vars,
                                             artifact resolution, GPU, imports)

Data-prep stages run on CPU nodes (Ray), training stages run on GPU nodes
(torchrun), and RL preflight runs on a GPU node inside the nemo-rl container
(Ray). They may require separate env.toml profiles.

Usage::

    # Run full pipeline
    python tests/recipes/super3/run_integration_test.py \\
        --data-profile cpu-cluster --train-profile gpu-cluster

    # Skip data prep (artifacts already exist from a previous run)
    python tests/recipes/super3/run_integration_test.py \\
        --train-profile gpu-cluster --skip-data-prep

    # Dry-run: print commands without executing
    python tests/recipes/super3/run_integration_test.py \\
        --data-profile cpu-cluster --train-profile gpu-cluster --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import NoReturn


# ── Stage definitions ────────────────────────────────────────────────────────

STAGES: list[dict] = [
    # Phase 1: Data preparation (CPU / Ray)
    {
        "name": "data-prep-pretrain",
        "phase": "data",
        "cmd": ["nemotron", "super3", "data", "prep", "pretrain", "--config", "tiny"],
    },
    {
        "name": "data-prep-sft",
        "phase": "data",
        "cmd": ["nemotron", "super3", "data", "prep", "sft", "--config", "tiny"],
    },
    {
        "name": "data-prep-rl",
        "phase": "data",
        "cmd": ["nemotron", "super3", "data", "prep", "rl", "rlvr"],
    },
    # Phase 2: Training (GPU / torchrun)
    {
        "name": "pretrain",
        "phase": "train",
        "cmd": ["nemotron", "super3", "pretrain", "--config", "test"],
    },
    {
        "name": "sft",
        "phase": "train",
        "cmd": ["nemotron", "super3", "sft", "--config", "test"],
    },
    # Phase 3: RL infrastructure preflight (GPU / Ray)
    {
        "name": "rl-preflight",
        "phase": "rl",
        "cmd": ["nemotron", "super3", "rl", "rlvr", "--config", "test"],
    },
]


# ── Runner ───────────────────────────────────────────────────────────────────


def _build_cmd(stage: dict, profiles: dict[str, str]) -> list[str]:
    """Build the full CLI command for a stage, adding --run <profile>."""
    profile = profiles[stage["phase"]]
    return [*stage["cmd"], "--run", profile]


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        description="Run the Super3 integration test pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-profile",
        help="env.toml profile for data-prep stages (CPU/Ray)",
    )
    parser.add_argument(
        "--train-profile",
        help="env.toml profile for training stages (GPU/torchrun)",
    )
    parser.add_argument(
        "--rl-profile",
        help="env.toml profile for RL preflight (GPU/Ray, defaults to --train-profile)",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation (use existing wandb artifacts)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training stages (only run data prep)",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip RL preflight check",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Validate args
    if not args.skip_data_prep and not args.data_profile:
        parser.error("--data-profile is required (or use --skip-data-prep)")
    if not args.skip_training and not args.train_profile:
        parser.error("--train-profile is required (or use --skip-training)")
    rl_profile = args.rl_profile or args.train_profile
    if not args.skip_rl and not rl_profile:
        parser.error("--rl-profile or --train-profile is required (or use --skip-rl)")

    profiles = {
        "data": args.data_profile,
        "train": args.train_profile,
        "rl": rl_profile,
    }

    # Filter stages
    stages = [
        s for s in STAGES
        if not (args.skip_data_prep and s["phase"] == "data")
        and not (args.skip_training and s["phase"] == "train")
        and not (args.skip_rl and s["phase"] == "rl")
    ]

    if not stages:
        print("Nothing to run (all stages skipped).")
        sys.exit(0)

    # Run stages sequentially
    for i, stage in enumerate(stages, 1):
        cmd = _build_cmd(stage, profiles)
        header = f"[{i}/{len(stages)}] {stage['name']}"
        print(f"\n{'=' * 60}")
        print(f"{header}")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n✗ Stage '{stage['name']}' failed (exit code {result.returncode})")
            sys.exit(result.returncode)

        print(f"\n✓ Stage '{stage['name']}' completed successfully")

    if args.dry_run:
        print("\n(dry-run: no commands were executed)")
    else:
        print(f"\n{'=' * 60}")
        print("✓ All stages completed successfully")
        print(f"{'=' * 60}")

    sys.exit(0)


if __name__ == "__main__":
    main()
