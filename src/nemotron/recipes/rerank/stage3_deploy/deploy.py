#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "rerank/deploy"
# setup = "Local-only Docker wrapper. Launches a NIM container for inference."
#
# [tool.runspec.run]
# launch = "direct"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
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

"""Deploy script for NIM reranking service with custom model.

Launches the NVIDIA NIM container with a custom ONNX/TensorRT model
exported from stage2_export. The NIM provides a ranking API with
the custom fine-tuned reranking model.

Usage:
    # With default config (launches NIM in foreground)
    nemotron rerank deploy -c default

    # With custom model path
    nemotron rerank deploy -c default model_dir=/path/to/onnx

    # Detached mode (background)
    nemotron rerank deploy -c default detach=true
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from typing import Annotated

from pydantic import BeforeValidator, ConfigDict, Field

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class DeployConfig(RecipeSettings):
    """Deployment configuration for NIM reranking service."""

    model_config = ConfigDict(extra="forbid")

    # Container settings
    nim_image: str = Field(default="nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0", description="NIM container image to use.")
    container_name: str = Field(default="nemotron-rerank-nim", description="Name for the Docker container.")

    # Model settings
    model_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage2_export/onnx", description="Path to custom model directory (ONNX or TensorRT).")
    use_onnx: bool = Field(default=True, description="Use ONNX model instead of TensorRT.")

    # Container paths
    container_model_path: str = Field(default="/opt/nim/custom_model", description="Path inside container where model will be mounted.")
    container_cache_path: str = Field(default="/opt/nim/.cache", description="Path inside container for NIM cache.")

    # Network settings
    host_port: int = Field(default=8001, ge=1, le=65535, description="Port to expose on host.")
    container_port: int = Field(default=8000, ge=1, le=65535, description="Port inside container.")

    # Resource settings
    gpus: Annotated[str, BeforeValidator(str)] = Field(default="all", description="Number of GPUs to use for the container.")
    shm_size: str = Field(default="2gb", description="Shared memory size.")

    # Runtime settings
    detach: bool = Field(default=False, description="Run container in detached mode.")
    remove_on_exit: bool = Field(default=True, description="Remove container when it exits.")
    health_check_timeout: int = Field(default=120, gt=0, description="Timeout in seconds for health check.")
    health_check_interval: int = Field(default=5, gt=0, description="Interval in seconds between health checks.")

    # Environment
    ngc_api_key_env: str = Field(default="NGC_API_KEY", description="Environment variable name for NGC API key.")


def check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_nvidia_docker() -> bool:
    """Check if NVIDIA Container Runtime is available."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
        )
        return "nvidia" in result.stdout.lower()
    except FileNotFoundError:
        return False


def stop_existing_container(container_name: str) -> None:
    """Stop and remove existing container with the same name."""
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)


def build_docker_command(cfg: DeployConfig) -> list[str]:
    """Build the Docker run command."""
    cmd = ["docker", "run"]

    if cfg.detach:
        cmd.append("-d")
    else:
        cmd.extend(["-it"])

    cmd.extend(["--name", cfg.container_name])

    if cfg.remove_on_exit and not cfg.detach:
        cmd.append("--rm")

    cmd.extend(["--gpus", cfg.gpus])
    cmd.extend(["--shm-size", cfg.shm_size])
    cmd.extend(["-u", "root"])
    cmd.extend(["-p", f"{cfg.host_port}:{cfg.container_port}"])

    ngc_key = os.environ.get(cfg.ngc_api_key_env)
    if ngc_key:
        cmd.extend(["-e", f"NGC_API_KEY={ngc_key}"])
    else:
        print(f"Warning: {cfg.ngc_api_key_env} not set. NIM may not authenticate properly.")

    cmd.extend(["-e", f"NIM_CUSTOM_MODEL={cfg.container_model_path}"])

    model_dir_abs = cfg.model_dir.resolve()
    cmd.extend(["-v", f"{model_dir_abs}:{cfg.container_model_path}:ro"])

    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    nim_cache = Path(cache_dir) / "nim"
    nim_cache.mkdir(parents=True, exist_ok=True)
    cmd.extend(["-v", f"{nim_cache}:{cfg.container_cache_path}"])

    cmd.append(cfg.nim_image)

    return cmd


def wait_for_health(cfg: DeployConfig) -> bool:
    """Wait for NIM to become healthy."""
    import urllib.request
    import urllib.error

    health_url = f"http://localhost:{cfg.host_port}/v1/health/ready"
    start_time = time.time()

    print(f"   Waiting for NIM to become healthy (timeout: {cfg.health_check_timeout}s)...")

    while time.time() - start_time < cfg.health_check_timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        time.sleep(cfg.health_check_interval)
        elapsed = int(time.time() - start_time)
        print(f"   ... still waiting ({elapsed}s)")

    return False


def run_deploy(cfg: DeployConfig) -> dict:
    """Run NIM reranker deployment."""
    print(f"NIM Reranking Service Deployment")
    print(f"=" * 60)
    print(f"NIM image:       {cfg.nim_image}")
    print(f"Container name:  {cfg.container_name}")
    print(f"Model directory: {cfg.model_dir}")
    print(f"Host port:       {cfg.host_port}")
    print(f"GPUs:            {cfg.gpus}")
    print(f"Detached:        {cfg.detach}")
    print(f"=" * 60)
    print()

    if not check_docker():
        print("Error: Docker is not installed or not running.")
        sys.exit(1)

    if not check_nvidia_docker():
        print("Warning: NVIDIA Container Runtime may not be available.")

    if not cfg.model_dir.exists():
        print(f"Error: Model directory not found: {cfg.model_dir}")
        print("       Please run stage2_export first.")
        sys.exit(1)

    model_files = list(cfg.model_dir.glob("*.onnx")) + list(cfg.model_dir.glob("*.plan"))
    if not model_files:
        print(f"Warning: No ONNX or TensorRT files found in {cfg.model_dir}")

    print(f"Stopping existing container (if any)...")
    stop_existing_container(cfg.container_name)

    docker_cmd = build_docker_command(cfg)
    print(f"Starting NIM container...")
    print(f"   Command: {' '.join(docker_cmd)}")
    print()

    result = {
        "container_name": cfg.container_name,
        "host_port": cfg.host_port,
        "model_dir": str(cfg.model_dir),
        "api_url": f"http://localhost:{cfg.host_port}/v1/ranking",
    }

    if cfg.detach:
        proc = subprocess.run(docker_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error starting container: {proc.stderr}")
            sys.exit(1)

        container_id = proc.stdout.strip()
        result["container_id"] = container_id
        print(f"   Container ID: {container_id[:12]}")

        if wait_for_health(cfg):
            print()
            print(f"NIM is ready!")
            print(f"   API endpoint: {result['api_url']}")
            print()
            print(f"   Test with:")
            print(f"   curl -X POST http://localhost:{cfg.host_port}/v1/ranking \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"model\": \"nvidia/llama-nemotron-rerank-1b-v2\", \"query\": {{\"text\": \"what is AI?\"}}, \"passages\": [{{\"text\": \"AI is artificial intelligence\"}}]}}'")
            print()
            print(f"   Stop with: docker stop {cfg.container_name}")
        else:
            print()
            print(f"Error: Health check timed out after {cfg.health_check_timeout}s.", file=sys.stderr)
            print(f"   Check logs with: docker logs {cfg.container_name}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"   Running in foreground. Press Ctrl+C to stop.")
        print()

        def signal_handler(signum, frame):
            print("\n   Shutting down...")
            stop_existing_container(cfg.container_name)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            subprocess.run(docker_cmd)
        except KeyboardInterrupt:
            pass

    return result


def main(cfg: DeployConfig | None = None) -> dict:
    """Entry point for deployment."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )

        try:
            cfg = load_config(config_path, cli_overrides, DeployConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_deploy(cfg)


if __name__ == "__main__":
    main()
