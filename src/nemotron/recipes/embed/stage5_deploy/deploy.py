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

"""Deploy script for NIM embedding service with custom model.

Launches the NVIDIA NIM container with a custom ONNX/TensorRT model
exported from stage4_export. The NIM provides an OpenAI-compatible
embeddings API with the custom fine-tuned model.

Usage:
    # With default config (launches NIM in foreground)
    nemotron embed deploy -c default

    # With custom model path
    nemotron embed deploy -c default model_dir=/path/to/onnx

    # Detached mode (background)
    nemotron embed deploy -c default detach=true
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

from nemotron.kit.config_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class DeployConfig(RecipeSettings):
    """Deployment configuration for NIM embedding service."""

    model_config = ConfigDict(extra="forbid")

    # Container settings
    nim_image: str = Field(default="nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.10.1", description="NIM container image to use.")
    container_name: str = Field(default="nemotron-embed-nim", description="Name for the Docker container.")

    # Model settings
    model_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage4_export/onnx", description="Path to custom model directory (ONNX or TensorRT).")
    use_onnx: bool = Field(default=True, description="Use ONNX model instead of TensorRT.")

    # Container paths
    container_model_path: str = Field(default="/opt/nim/custom_model", description="Path inside container where model will be mounted.")
    container_cache_path: str = Field(default="/opt/nim/.cache", description="Path inside container for NIM cache.")

    # Network settings
    host_port: int = Field(default=8000, ge=1, le=65535, description="Port to expose on host.")
    container_port: int = Field(default=8000, ge=1, le=65535, description="Port inside container.")

    # Resource settings
    gpus: Annotated[str, BeforeValidator(str)] = Field(default="all", description="Number of GPUs to use for the container. (e.g., 'all', 1).")
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
    subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "rm", container_name],
        capture_output=True,
    )


def build_docker_command(cfg: DeployConfig) -> list[str]:
    """Build the Docker run command.

    Args:
        cfg: Deployment configuration.

    Returns:
        List of command arguments.
    """
    cmd = ["docker", "run"]

    # Interactive/detached mode
    if cfg.detach:
        cmd.append("-d")
    else:
        cmd.extend(["-it"])

    # Container name
    cmd.extend(["--name", cfg.container_name])

    # Remove on exit
    if cfg.remove_on_exit and not cfg.detach:
        cmd.append("--rm")

    # GPU allocation
    cmd.extend(["--gpus", cfg.gpus])

    # Shared memory
    cmd.extend(["--shm-size", cfg.shm_size])

    # Run as root (required for NIM)
    cmd.extend(["-u", "root"])

    # Port mapping
    cmd.extend(["-p", f"{cfg.host_port}:{cfg.container_port}"])

    # NGC API key
    ngc_key = os.environ.get(cfg.ngc_api_key_env)
    if ngc_key:
        cmd.extend(["-e", f"NGC_API_KEY={ngc_key}"])
    else:
        print(f"Warning: {cfg.ngc_api_key_env} not set. NIM may not authenticate properly.")

    # Custom model environment variable
    cmd.extend(["-e", f"NIM_CUSTOM_MODEL={cfg.container_model_path}"])

    # Volume mounts
    # Model directory
    model_dir_abs = cfg.model_dir.resolve()
    cmd.extend(["-v", f"{model_dir_abs}:{cfg.container_model_path}:ro"])

    # Cache directory (optional, uses host cache)
    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    nim_cache = Path(cache_dir) / "nim"
    nim_cache.mkdir(parents=True, exist_ok=True)
    cmd.extend(["-v", f"{nim_cache}:{cfg.container_cache_path}"])

    # Container image
    cmd.append(cfg.nim_image)

    return cmd


def wait_for_health(cfg: DeployConfig) -> bool:
    """Wait for NIM to become healthy.

    Args:
        cfg: Deployment configuration.

    Returns:
        True if healthy, False if timeout.
    """
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
    """Run NIM deployment.

    Args:
        cfg: Deployment configuration.

    Returns:
        Dictionary with deployment info.
    """
    print(f"🚀 NIM Embedding Service Deployment")
    print(f"=" * 60)
    print(f"NIM image:       {cfg.nim_image}")
    print(f"Container name:  {cfg.container_name}")
    print(f"Model directory: {cfg.model_dir}")
    print(f"Host port:       {cfg.host_port}")
    print(f"GPUs:            {cfg.gpus}")
    print(f"Detached:        {cfg.detach}")
    print(f"=" * 60)
    print()

    # Check prerequisites
    if not check_docker():
        print("Error: Docker is not installed or not running.")
        sys.exit(1)

    if not check_nvidia_docker():
        print("Warning: NVIDIA Container Runtime may not be available.")

    # Validate model directory
    if not cfg.model_dir.exists():
        print(f"Error: Model directory not found: {cfg.model_dir}")
        print("       Please run stage4_export first.")
        sys.exit(1)

    # Check for model files
    model_files = list(cfg.model_dir.glob("*.onnx")) + list(cfg.model_dir.glob("*.plan"))
    if not model_files:
        print(f"Warning: No ONNX or TensorRT files found in {cfg.model_dir}")

    # Stop any existing container with same name
    print(f"📦 Stopping existing container (if any)...")
    stop_existing_container(cfg.container_name)

    # Build Docker command
    docker_cmd = build_docker_command(cfg)
    print(f"📦 Starting NIM container...")
    print(f"   Command: {' '.join(docker_cmd)}")
    print()

    result = {
        "container_name": cfg.container_name,
        "host_port": cfg.host_port,
        "model_dir": str(cfg.model_dir),
        "api_url": f"http://localhost:{cfg.host_port}/v1/embeddings",
    }

    if cfg.detach:
        # Run in background
        proc = subprocess.run(docker_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error starting container: {proc.stderr}")
            sys.exit(1)

        container_id = proc.stdout.strip()
        result["container_id"] = container_id
        print(f"   Container ID: {container_id[:12]}")

        # Wait for health
        if wait_for_health(cfg):
            print()
            print(f"✅ NIM is ready!")
            print(f"   API endpoint: {result['api_url']}")
            print()
            print(f"   Test with:")
            print(f"   curl -X POST http://localhost:{cfg.host_port}/v1/embeddings \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"input\": [\"hello world\"], \"model\": \"nvidia/llama-3.2-nv-embedqa-1b-v2\", \"input_type\": \"query\"}}'")
            print()
            print(f"   Stop with: docker stop {cfg.container_name}")
        else:
            print()
            print(f"⚠️  Health check timeout. Container may still be starting.")
            print(f"   Check logs with: docker logs {cfg.container_name}")
    else:
        # Run in foreground (interactive)
        print(f"   Running in foreground. Press Ctrl+C to stop.")
        print()

        # Set up signal handler for clean shutdown
        def signal_handler(signum, frame):
            print("\n   Shutting down...")
            stop_existing_container(cfg.container_name)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run interactively
        try:
            subprocess.run(docker_cmd)
        except KeyboardInterrupt:
            pass

    return result


def main(cfg: DeployConfig | None = None) -> dict:
    """Entry point for deployment.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Dictionary with deployment info.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
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
