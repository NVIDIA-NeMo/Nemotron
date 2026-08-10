#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "lightning35/eval"
# image = "nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_5_lightning"
# setup = """
# The NeMo-RL container bundles vLLM and NeMo Gym (/opt/nemo-rl/3rdparty/Gym-workspace/Gym).
# This stage serves the checkpoint with vLLM in-job and drives `gym eval` against it.
# """
#
# [tool.runspec.run]
# cmd = "python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 8
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

"""Nemotron 3.5 Lightning evaluation (stage3) with NeMo Gym.

NeMo Evaluator is deprecated for the instruct-model benchmark suite in favor
of NeMo Gym. This stage reproduces the Gym-native evaluation flow used for
the published Lightning results (see Gym `scripts/more/reproducibility.md`):

1. Serve the HF checkpoint with vLLM using the Lightning serving settings
   (nemotron_v3 reasoning parser, qwen3_coder tool parser, fp32 mamba cache,
   optional MTP speculative decoding).
2. Wait until the OpenAI-compatible endpoint is healthy.
3. For each configured benchmark: `gym eval prepare` then `gym eval run`
   against the endpoint (`--model-type vllm_model`).
4. Collect each benchmark's `*_aggregate_metrics.json` into a single
   summary.json for the stage.

Base-model tasks (lm-evaluation-harness / RULER) are NOT covered here; those
still run through nemo-evaluator-launcher.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf

from nemotron.kit.train_script import load_omegaconf_yaml, parse_config_and_overrides

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


# =============================================================================
# vLLM serving
# =============================================================================


def build_vllm_command(serving: dict) -> list[str]:
    """Build the `vllm serve` command from the serving config section."""
    cmd = [
        "vllm",
        "serve",
        str(serving["model_path"]),
        "--host",
        str(serving.get("host", "127.0.0.1")),
        "--port",
        str(serving.get("port", 8000)),
        "--tensor-parallel-size",
        str(serving.get("tensor_parallel_size", 4)),
        "--gpu-memory-utilization",
        str(serving.get("gpu_memory_utilization", 0.9)),
        "--kv-cache-dtype",
        str(serving.get("kv_cache_dtype", "auto")),
        "--mamba-ssm-cache-dtype",
        str(serving.get("mamba_ssm_cache_dtype", "float32")),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        str(serving.get("tool_call_parser", "qwen3_coder")),
        "--reasoning-parser",
        str(serving.get("reasoning_parser", "nemotron_v3")),
    ]
    if serving.get("enable_expert_parallel", True):
        cmd.append("--enable-expert-parallel")
    if serving.get("data_parallel_size", 1) > 1:
        cmd.extend(["--data-parallel-size", str(serving["data_parallel_size"])])
    if serving.get("speculative_config"):
        spec = serving["speculative_config"]
        cmd.extend(["--speculative-config", spec if isinstance(spec, str) else json.dumps(spec)])
    if serving.get("max_model_len"):
        cmd.extend(["--max-model-len", str(serving["max_model_len"])])
    for extra in serving.get("extra_args") or []:
        cmd.extend(shlex.split(str(extra)))
    return cmd


def wait_for_endpoint(base_url: str, timeout_s: int) -> None:
    """Poll the OpenAI-compatible /models endpoint until it responds."""
    deadline = time.time() + timeout_s
    url = f"{base_url}/models"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
                if resp.status == 200:
                    logger.info(f"vLLM endpoint healthy: {url}")
                    return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_error = e
        time.sleep(10)
    raise TimeoutError(f"vLLM endpoint {url} not healthy after {timeout_s}s: {last_error}")


# =============================================================================
# Gym evaluation
# =============================================================================


def run_benchmark(
    benchmark: str | dict,
    *,
    gym_cfg: dict,
    base_url: str,
    model_path: str,
    output_dir: Path,
) -> dict | None:
    """Run one Gym benchmark; return its aggregate metrics (or None on failure).

    A benchmark entry is either a name ("gpqa") or a mapping with per-benchmark
    Gym overrides, mirroring the reference scripts in Gym scripts/more/:
        {name: scicode, overrides: ["++...test_data_fpath=/data/test_data.h5"]}
    """
    if isinstance(benchmark, dict):
        name = str(benchmark["name"])
        per_benchmark_overrides = [str(o) for o in benchmark.get("overrides") or []]
    else:
        name = str(benchmark)
        per_benchmark_overrides = []

    workdir = gym_cfg.get("workdir")
    gym_cmd = shlex.split(str(gym_cfg.get("command", "gym")))
    output_jsonl = output_dir / f"{name}.jsonl"

    prepare_cmd = [*gym_cmd, "eval", "prepare", "--benchmark", name]
    run_cmd = [
        *gym_cmd,
        "eval",
        "run",
        "--benchmark",
        name,
        "--model-type",
        str(gym_cfg.get("model_type", "vllm_model")),
        "--model-url",
        base_url,
        "--model-api-key",
        str(gym_cfg.get("api_key", "dummy")),
        "--model",
        model_path,
        "--output",
        str(output_jsonl),
        "--split",
        str(gym_cfg.get("split", "benchmark")),
        "--concurrency",
        str(gym_cfg.get("concurrency", 128)),
    ]
    if gym_cfg.get("limit"):
        run_cmd.extend(["--limit", str(gym_cfg["limit"])])
    if gym_cfg.get("num_repeats"):
        run_cmd.extend(["--num-repeats", str(gym_cfg["num_repeats"])])
    # Reference-standard overrides applied to every benchmark, then
    # per-benchmark overrides, then ad-hoc extras (later wins in Gym).
    for override in gym_cfg.get("common_overrides") or []:
        run_cmd.append(str(override))
    run_cmd.extend(per_benchmark_overrides)
    for override in gym_cfg.get("extra_overrides") or []:
        run_cmd.append(str(override))

    for label, cmd in (("prepare", prepare_cmd), ("run", run_cmd)):
        logger.info(f"[{name}] gym eval {label}: {shlex.join(cmd)}")
        result = subprocess.run(cmd, cwd=workdir)  # noqa: S603
        if result.returncode != 0:
            logger.error(f"[{name}] gym eval {label} failed (exit {result.returncode})")
            return None

    metrics_path = output_dir / f"{name}_aggregate_metrics.json"
    if not metrics_path.exists():
        logger.error(f"[{name}] missing aggregate metrics: {metrics_path}")
        return None
    with open(metrics_path) as f:
        return json.load(f)


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    """Entry point for Nemotron 3.5 Lightning Gym evaluation."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if cli_overrides:
        # nemo_rl ships in the eval container (same image as stage2 RL)
        from nemo_rl.utils.config import parse_hydra_overrides

        config = parse_hydra_overrides(config, cli_overrides)

    # Register nemotron artifact resolver for ${art:...} interpolations
    from nemo_runspec.config.resolvers import clear_artifact_cache, register_resolvers_from_config

    clear_artifact_cache()
    register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )
    cfg = OmegaConf.to_container(config, resolve=True)

    serving = cfg["serving"]
    gym_cfg = cfg["gym"]
    benchmarks = list(gym_cfg.get("benchmarks") or [])
    if not benchmarks:
        logger.error("No benchmarks configured (gym.benchmarks is empty)")
        sys.exit(1)

    output_dir = Path(gym_cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(serving["model_path"])
    base_url = f"http://{serving.get('host', '127.0.0.1')}:{serving.get('port', 8000)}/v1"

    vllm_cmd = build_vllm_command(serving)
    logger.info(f"Starting vLLM: {shlex.join(vllm_cmd)}")
    vllm_log = open(output_dir / "vllm_serve.log", "w")  # noqa: SIM115
    vllm_proc = subprocess.Popen(vllm_cmd, stdout=vllm_log, stderr=subprocess.STDOUT)  # noqa: S603

    summary: dict[str, dict | None] = {}
    try:
        wait_for_endpoint(base_url, int(serving.get("startup_timeout_s", 3600)))
        for benchmark in benchmarks:
            name = benchmark["name"] if isinstance(benchmark, dict) else str(benchmark)
            summary[name] = run_benchmark(
                benchmark,
                gym_cfg=gym_cfg,
                base_url=base_url,
                model_path=model_path,
                output_dir=output_dir,
            )
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
        vllm_log.close()

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote evaluation summary: {summary_path}")

    failed = [name for name, metrics in summary.items() if metrics is None]
    for name, metrics in summary.items():
        status = "FAILED" if metrics is None else "ok"
        logger.info(f"  {name}: {status}")
    if failed:
        logger.error(f"{len(failed)}/{len(benchmarks)} benchmarks failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
