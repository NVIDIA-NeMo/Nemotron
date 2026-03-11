#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "super3/rl-test"
# image = "nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super"
# setup = """
# Preflight check — validates RL infrastructure inside the container
# without running actual GRPO training.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run python {script} --config {config}"
# workdir = "/opt/nemo-rl"
#
# [tool.runspec.config]
# dir = "./config"
# default = "test"
# format = "omegaconf"
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

"""RL infrastructure preflight check for Nemotron Super3.

Validates that the RL execution environment is correctly configured without
running actual GRPO training. Designed to be submitted as a RayJob via the CLI::

    nemotron super3 rl rlvr --config test --run <profile>

Checks performed:
  1. Config parsing & OmegaConf resolution
  2. Artifact resolution (run.data, run.model → real paths)
  3. Ray cluster initialization
  4. GPU availability (torch.cuda)
  5. Key package imports (nemo_rl, vllm)
  6. Expected environment variables
  7. Container filesystem (mounts, workdir)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


# ── Utilities ────────────────────────────────────────────────────────────────

class PreflightResult:
    """Tracks pass/fail/skip for each check."""

    def __init__(self):
        self.results: list[tuple[str, str, str]] = []  # (name, status, detail)

    def ok(self, name: str, detail: str = ""):
        self.results.append((name, "PASS", detail))
        print(f"  ✓ {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, detail: str = ""):
        self.results.append((name, "FAIL", detail))
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))

    def skip(self, name: str, detail: str = ""):
        self.results.append((name, "SKIP", detail))
        print(f"  ○ {name}" + (f" — {detail}" if detail else ""))

    @property
    def passed(self) -> int:
        return sum(1 for _, s, _ in self.results if s == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for _, s, _ in self.results if s == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for _, s, _ in self.results if s == "SKIP")


# ── Checks ───────────────────────────────────────────────────────────────────


def check_config_parsing(config_path: str, r: PreflightResult) -> dict | None:
    """Load and resolve the YAML config, returning the resolved dict."""
    try:
        from nemo_rl.utils.config import load_config
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)

        config = load_config(config_path)
        r.ok("config_load", f"loaded from {config_path}")
    except Exception as e:
        r.fail("config_load", str(e))
        return None

    # Register artifact resolvers so ${art:...} references resolve
    try:
        from nemo_runspec.config.resolvers import clear_artifact_cache, register_resolvers_from_config

        clear_artifact_cache()
        register_resolvers_from_config(
            config,
            artifacts_key="run",
            mode="pre_init",
            pre_init_patch_http_digest=False,
        )
        r.ok("artifact_resolvers", "registered")
    except Exception as e:
        r.fail("artifact_resolvers", str(e))

    # Resolve all interpolations
    try:
        resolved = OmegaConf.to_container(config, resolve=True)
        r.ok("config_resolve", "all interpolations resolved")
        return resolved
    except Exception as e:
        r.fail("config_resolve", str(e))
        return None


def check_artifacts(config: dict, r: PreflightResult) -> None:
    """Verify artifact paths exist on the filesystem."""
    run_cfg = config.get("run", {})

    # Check data artifact
    data_cfg = config.get("data", {})
    # Support both v0.4 (flat) and v0.5 (nested) formats
    if "train" in data_cfg and isinstance(data_cfg["train"], dict):
        train_path = data_cfg["train"].get("data_path")
        val_path = data_cfg.get("validation", {}).get("data_path")
    else:
        train_path = data_cfg.get("train_jsonl_fpath")
        val_path = data_cfg.get("validation_jsonl_fpath")

    for name, path in [("data.train", train_path), ("data.val", val_path)]:
        if path and os.path.exists(path):
            r.ok(f"artifact:{name}", path)
        elif path:
            r.fail(f"artifact:{name}", f"path does not exist: {path}")
        else:
            r.skip(f"artifact:{name}", "not configured")

    # Check model artifact (initial_checkpoint or policy.model_name)
    model_name = config.get("policy", {}).get("model_name", "")
    initial_ckpt = config.get("initial_checkpoint", "")
    model_path = initial_ckpt or model_name
    if model_path and os.path.exists(model_path):
        r.ok("artifact:model", model_path)
    elif model_path and model_path.startswith("/"):
        r.fail("artifact:model", f"path does not exist: {model_path}")
    elif model_path:
        # Could be a HF model ID — skip filesystem check
        r.skip("artifact:model", f"HF model ID: {model_path}")
    else:
        r.skip("artifact:model", "not configured")


def check_ray(r: PreflightResult) -> None:
    """Initialize and shut down a Ray cluster."""
    try:
        import ray

        ray.init(ignore_reinit_error=True)
        resources = ray.cluster_resources()
        gpu_count = resources.get("GPU", 0)
        cpu_count = resources.get("CPU", 0)
        ray.shutdown()
        r.ok("ray_init", f"CPUs={cpu_count}, GPUs={gpu_count}")
    except ImportError:
        r.fail("ray_init", "ray not installed")
    except Exception as e:
        r.fail("ray_init", str(e))


def check_gpu(r: PreflightResult) -> None:
    """Check GPU availability via torch."""
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count > 0 else "unknown"
            r.ok("gpu", f"{count} GPU(s): {name}")
        else:
            r.fail("gpu", "torch.cuda.is_available() is False")
    except ImportError:
        r.fail("gpu", "torch not installed")
    except Exception as e:
        r.fail("gpu", str(e))


def check_imports(r: PreflightResult) -> None:
    """Check that key packages can be imported."""
    packages = [
        ("nemo_rl", "nemo_rl"),
        ("vllm", "vllm"),
        ("nemo_rl.algorithms.grpo", "grpo"),
        ("nemo_rl.distributed.virtual_cluster", "virtual_cluster"),
        ("nemo_rl.environments.nemo_gym", "nemo_gym"),
    ]
    for module, label in packages:
        try:
            __import__(module)
            r.ok(f"import:{label}")
        except ImportError as e:
            r.fail(f"import:{label}", str(e))
        except Exception as e:
            r.fail(f"import:{label}", f"{type(e).__name__}: {e}")


def check_env_vars(r: PreflightResult) -> None:
    """Check expected environment variables are set."""
    # Core NRL/vLLM env vars from config
    expected = [
        "NRL_WG_USE_RAY_REF",
        "NRL_VLLM_USE_V1",
        "NRL_IGNORE_VERSION_MISMATCH",
        "VLLM_ATTENTION_BACKEND",
        "OMP_NUM_THREADS",
    ]
    for var in expected:
        val = os.environ.get(var)
        if val is not None:
            r.ok(f"env:{var}", val)
        else:
            r.fail(f"env:{var}", "not set")

    # Optional sandbox vars (only if sandbox is configured)
    sandbox_vars = ["SANDBOX_CONTAINER", "NEMO_SKILLS_SANDBOX_PORT"]
    for var in sandbox_vars:
        val = os.environ.get(var)
        if val is not None:
            r.ok(f"env:{var}", val)
        else:
            r.skip(f"env:{var}", "not set (sandbox not configured)")

    # W&B credentials (needed for artifact resolution)
    for var in ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]:
        val = os.environ.get(var)
        if val:
            r.ok(f"env:{var}", val[:4] + "..." if var == "WANDB_API_KEY" else val)
        else:
            r.skip(f"env:{var}", "not set")


def check_filesystem(r: PreflightResult) -> None:
    """Check expected container mounts and paths exist."""
    paths = [
        ("/opt/nemo-rl", "nemo-rl workdir"),
        ("/lustre", "lustre mount"),
    ]
    for path, label in paths:
        if os.path.exists(path):
            r.ok(f"fs:{label}", path)
        else:
            r.skip(f"fs:{label}", f"{path} not found")

    # Check apptainer availability (needed for SWE stages)
    if shutil.which("apptainer"):
        result = subprocess.run(
            ["apptainer", "--version"], capture_output=True, text=True
        )
        r.ok("apptainer", result.stdout.strip())
    else:
        r.skip("apptainer", "not installed (only needed for SWE stages)")


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL infrastructure preflight check")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "config", "test.yaml"
    )

    print("=" * 60)
    print("Super3 RL Preflight Check")
    print("=" * 60)

    r = PreflightResult()

    print("\n[1/6] Config parsing & resolution")
    config = check_config_parsing(config_path, r)

    print("\n[2/6] Artifact resolution")
    if config:
        check_artifacts(config, r)
    else:
        r.skip("artifacts", "skipped (config failed to load)")

    print("\n[3/6] Ray cluster")
    check_ray(r)

    print("\n[4/6] GPU availability")
    check_gpu(r)

    print("\n[5/6] Package imports")
    check_imports(r)

    print("\n[6/6] Environment variables")
    check_env_vars(r)

    print("\n[7/7] Filesystem & tools")
    check_filesystem(r)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {r.passed} passed, {r.failed} failed, {r.skipped} skipped")
    print(f"{'=' * 60}")

    if r.failed > 0:
        print("\nPreflight FAILED — see failures above")
        sys.exit(1)
    else:
        print("\nPreflight OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
