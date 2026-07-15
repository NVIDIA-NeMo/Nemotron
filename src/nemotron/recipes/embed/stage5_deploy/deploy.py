#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "embed/deploy"
# setup = "Local-only Docker wrapper. Launches a NIM or vLLM container for inference."
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

"""Deploy an embedding service with a custom model checkpoint.

The default backend launches NVIDIA NIM. Retriever Embedding NIM 2.1.0 and
later consume a Hugging Face-style safetensors directory through
``NIM_ENGINE_MODEL_PATH``. Older NIM images use the ``NIM_CUSTOM_MODEL`` contract for
exported ONNX/TensorRT artifacts. The optional vLLM backend serves the same
Hugging Face checkpoint and is evaluated through vLLM's ``/v2/embed`` API.

Usage:
    # With default config (launches NIM in foreground)
    nemotron embed deploy -c default

    # With custom model path
    nemotron embed deploy -c default model_dir=/path/to/model

    # Detached mode (background)
    nemotron embed deploy -c default detach=true

    # Serve the default checkpoint with NVIDIA's vLLM container
    nemotron embed deploy -c default backend=vllm
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class DeployConfig(RecipeSettings):
    """Deployment configuration for a NIM or vLLM embedding service."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: Path = Field(
        default_factory=lambda: _OUTPUT_BASE / "output/embed/nemotron-3-1b",
        description="Root directory for this model profile's pipeline artifacts.",
    )

    # Container settings
    backend: Literal["nim", "vllm"] = Field(default="nim", description="Serving backend to launch.")
    nim_image: str | None = Field(
        default_factory=lambda: os.environ.get("NEMOTRON3_EMBED_NIM_IMAGE"),
        description="NIM container image to use for the NIM backend.",
    )
    vllm_image: str = Field(
        default="nvcr.io/nvidia/vllm:26.06-py3",
        description="vLLM container image to use for the vLLM backend.",
    )
    nim_model: str = Field(
        default="nvidia/nemotron-3-embed-1b",
        description="Model identifier advertised by the selected embedding service.",
    )
    container_name: str = Field(default="nemotron-embed", description="Name for the Docker container.")

    # Model settings
    model_dir: Path = Field(
        default_factory=lambda data: Path(
            os.environ.get(
                "NEMOTRON3_EMBED_DEPLOY_CHECKPOINT",
                data["artifact_root"] / "stage2_finetune/checkpoints/LATEST/model/consolidated",
            )
        ),
        description="Path to custom model artifacts on the host.",
    )
    use_onnx: bool = Field(
        default=False,
        description="NIM_CUSTOM_MODEL artifact selector. Ignored when model_path_env selects a direct checkpoint.",
    )
    model_path_env: Literal["NIM_CUSTOM_MODEL", "NIM_ENGINE_MODEL_PATH", "NIM_MODEL_PATH"] = Field(
        default="NIM_ENGINE_MODEL_PATH",
        description="NIM artifact selector; NIM_ENGINE_MODEL_PATH is the default for Retriever NIM 2.1.0+.",
    )
    expected_model_fingerprint: dict[str, int] | None = Field(
        default_factory=lambda: {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "vocab_size": 131072,
        },
        description="Optional architecture fingerprint required by the selected NIM image.",
    )

    # Container paths
    container_model_path: str = Field(
        default="/model", description="Path inside container where model will be mounted."
    )
    container_cache_path: str = Field(default="/opt/cache", description="Path inside container for NIM cache.")
    vllm_container_cache_path: str = Field(
        default="/root/.cache/huggingface",
        description="Path inside the vLLM container for Hugging Face cache data.",
    )

    # Network settings
    host_port: int = Field(default=8000, ge=1, le=65535, description="Port to expose on host.")
    container_port: int = Field(default=8000, ge=1, le=65535, description="Port inside container.")
    max_seq_len: int | None = Field(
        default=None,
        gt=0,
        description="Optional NIM maximum sequence length override; unset uses the image default.",
    )
    pipeline_id: str | None = Field(
        default=None,
        description="Optional exact NIM runtime pipeline identifier.",
    )
    # Resource settings
    gpus: Annotated[str, BeforeValidator(str)] = Field(
        default="all", description="Number of GPUs to use for the container. (e.g., 'all', 1)."
    )
    shm_size: str = Field(default="16gb", description="Shared memory size.")

    # Runtime settings
    detach: bool = Field(default=False, description="Run container in detached mode.")
    remove_on_exit: bool = Field(default=True, description="Remove container when it exits.")
    health_check_timeout: int = Field(default=600, gt=0, description="Timeout in seconds for health check.")
    health_check_interval: int = Field(default=5, gt=0, description="Interval in seconds between health checks.")

    # Environment
    ngc_api_key_env: str = Field(default="NGC_API_KEY", description="Environment variable name for NGC API key.")
    forward_ngc_api_key: bool = Field(
        default=False,
        description=(
            "Forward NGC_API_KEY into the container. Local NIM_ENGINE_MODEL_PATH artifacts "
            "do not require it; enable for NIM_CUSTOM_MODEL or model-download workflows when needed."
        ),
    )
    allow_unhealthy_detached: bool = Field(
        default=False,
        description="Return success after a detached health timeout instead of failing.",
    )

    @model_validator(mode="after")
    def _require_selected_backend_image(self) -> DeployConfig:
        # ``load_pydantic_config`` preserves OmegaConf's quoted empty fallback
        # as the literal string ``''``. Treat it as unset while leaving the raw
        # interpolation intact for direct YAML schema validation.
        if self.nim_image in {"", "''", '""', "null", "None"}:
            self.nim_image = None
        if self.backend == "nim" and (self.nim_image is None or not self.nim_image.strip()):
            raise ValueError("NEMOTRON3_EMBED_NIM_IMAGE must be set when backend=nim")
        if self.backend == "vllm" and not self.vllm_image.strip():
            raise ValueError("vllm_image must be set when backend=vllm")
        return self

    @property
    def container_image(self) -> str:
        """Return the image selected by the serving backend."""
        if self.backend == "vllm":
            return self.vllm_image
        assert self.nim_image is not None
        return self.nim_image


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

    # NGC API key is not needed when all model artifacts are already mounted.
    if cfg.backend == "nim" and cfg.forward_ngc_api_key:
        ngc_key = os.environ.get(cfg.ngc_api_key_env)
        if ngc_key:
            cmd.extend(["-e", "NGC_API_KEY"])
        else:
            print(f"Warning: {cfg.ngc_api_key_env} not set. NIM may not authenticate properly.")

    if cfg.backend == "nim":
        # NIM 2.1.0+ uses engine-prefixed variables for Hugging Face checkpoints.
        # The compatible image supplies NIM_ENGINE_MODEL_NAME. Keep NIM_MODEL_NAME
        # for existing 2.0 deployments that override this setting.
        if cfg.model_path_env == "NIM_MODEL_PATH":
            cmd.extend(["-e", f"NIM_MODEL_NAME={cfg.nim_model}"])
        cmd.extend(["-e", f"{cfg.model_path_env}={cfg.container_model_path}"])

        if cfg.max_seq_len is not None:
            max_seq_len_env = (
                "NIM_PIPELINE_MAX_SEQ_LEN"
                if cfg.model_path_env == "NIM_ENGINE_MODEL_PATH"
                else "NIM_MAX_SEQ_LEN"
            )
            cmd.extend(["-e", f"{max_seq_len_env}={cfg.max_seq_len}"])
        if cfg.pipeline_id:
            cmd.extend(["-e", f"NIM_PIPELINE_ID={cfg.pipeline_id}"])

    # Volume mounts
    # Model directory
    model_dir_abs = cfg.model_dir.resolve()
    cmd.extend(["-v", f"{model_dir_abs}:{cfg.container_model_path}:ro"])

    # Cache directory (optional, uses host cache)
    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_name = "huggingface" if cfg.backend == "vllm" else "nim"
    host_cache = Path(cache_dir) / cache_name
    host_cache.mkdir(parents=True, exist_ok=True)
    container_cache = cfg.vllm_container_cache_path if cfg.backend == "vllm" else cfg.container_cache_path
    cmd.extend(["-v", f"{host_cache}:{container_cache}"])

    # Container image
    cmd.append(cfg.container_image)

    if cfg.backend == "vllm":
        cmd.extend(
            [
                "vllm",
                "serve",
                cfg.container_model_path,
                "--served-model-name",
                cfg.nim_model,
                "--host",
                "0.0.0.0",
                "--port",
                str(cfg.container_port),
            ]
        )

    return cmd


def build_docker_environment(cfg: DeployConfig) -> dict[str, str]:
    """Build the Docker client environment without placing secrets in argv."""
    env = os.environ.copy()
    ngc_key = os.environ.get(cfg.ngc_api_key_env) if cfg.forward_ngc_api_key else None
    if ngc_key:
        # Docker receives ``-e NGC_API_KEY`` and copies this value into the
        # container without exposing it in the command line or command log.
        env["NGC_API_KEY"] = ngc_key
    return env


def model_artifact_errors(cfg: DeployConfig) -> list[str]:
    """Return missing or incompatible model artifact diagnostics."""
    if cfg.backend == "vllm" or cfg.model_path_env in {"NIM_ENGINE_MODEL_PATH", "NIM_MODEL_PATH"}:
        errors: list[str] = []
        config_path = cfg.model_dir / "config.json"
        if not config_path.is_file():
            errors.append("missing config.json")
        if not any(cfg.model_dir.glob("*.safetensors")):
            errors.append("missing *.safetensors")
        if not any((cfg.model_dir / filename).is_file() for filename in ("tokenizer.json", "tokenizer_config.json")):
            errors.append("missing tokenizer.json or tokenizer_config.json")

        if config_path.is_file() and cfg.expected_model_fingerprint:
            try:
                model_config = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"cannot read config.json: {error}")
            else:
                for field, expected in cfg.expected_model_fingerprint.items():
                    actual = model_config.get(field)
                    if actual != expected:
                        errors.append(f"config.json {field}={actual!r} (expected {expected!r})")
        return errors

    artifact_pattern = "*.onnx" if cfg.use_onnx else "*.plan"
    if not any(cfg.model_dir.glob(artifact_pattern)):
        return [f"missing {artifact_pattern}"]
    return []


def wait_for_health(cfg: DeployConfig) -> bool:
    """Wait for the embedding service to become healthy.

    Args:
        cfg: Deployment configuration.

    Returns:
        True if healthy, False if timeout.
    """
    import http.client
    import urllib.error
    import urllib.request

    health_path = "/health" if cfg.backend == "vllm" else "/v1/health/ready"
    health_url = f"http://localhost:{cfg.host_port}{health_path}"
    start_time = time.time()

    print(f"   Waiting for embedding service to become healthy (timeout: {cfg.health_check_timeout}s)...")

    while time.time() - start_time < cfg.health_check_timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            http.client.HTTPException,
            ConnectionError,
            OSError,
            TimeoutError,
        ):
            pass

        time.sleep(cfg.health_check_interval)
        elapsed = int(time.time() - start_time)
        print(f"   ... still waiting ({elapsed}s)")

    return False


def run_deploy(cfg: DeployConfig) -> dict:
    """Run an embedding service deployment.

    Args:
        cfg: Deployment configuration.

    Returns:
        Dictionary with deployment info.
    """
    print("🚀 Embedding Service Deployment")
    print("=" * 60)
    print(f"Backend:         {cfg.backend}")
    print(f"Container image: {cfg.container_image}")
    print(f"API model:       {cfg.nim_model}")
    print(f"Container name:  {cfg.container_name}")
    print(f"Model directory: {cfg.model_dir}")
    print(f"Host port:       {cfg.host_port}")
    print(f"GPUs:            {cfg.gpus}")
    print(f"Detached:        {cfg.detach}")
    print("=" * 60)
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
        print("       Set model_dir to the artifacts required by the selected backend.")
        sys.exit(1)

    # Check for model files expected by the selected artifact contract.
    artifact_errors = model_artifact_errors(cfg)
    if artifact_errors:
        print(f"Error: Model artifacts are not compatible with the selected backend in {cfg.model_dir}:")
        for error in artifact_errors:
            print(f"       - {error}")
        sys.exit(1)

    # Stop any existing container with same name
    print("📦 Stopping existing container (if any)...")
    stop_existing_container(cfg.container_name)

    # Build Docker command
    docker_cmd = build_docker_command(cfg)
    docker_env = build_docker_environment(cfg)
    print("📦 Starting embedding service container...")
    print(f"   Command: {' '.join(docker_cmd)}")
    print()

    result = {
        "backend": cfg.backend,
        "container_name": cfg.container_name,
        "host_port": cfg.host_port,
        "model_dir": str(cfg.model_dir),
        "api_url": (
            f"http://localhost:{cfg.host_port}/v2/embed"
            if cfg.backend == "vllm"
            else f"http://localhost:{cfg.host_port}/v1/embeddings"
        ),
    }

    if cfg.detach:
        # Run in background
        proc = subprocess.run(docker_cmd, capture_output=True, text=True, env=docker_env)
        if proc.returncode != 0:
            print(f"Error starting container: {proc.stderr}")
            sys.exit(1)

        container_id = proc.stdout.strip()
        result["container_id"] = container_id
        print(f"   Container ID: {container_id[:12]}")

        # Wait for health
        if wait_for_health(cfg):
            print()
            print("✅ Embedding service is ready!")
            print(f"   API endpoint: {result['api_url']}")
            print()
            print("   Test with:")
            print(f"   curl -X POST {result['api_url']} \\")
            print("     -H 'Content-Type: application/json' \\")
            if cfg.backend == "vllm":
                print(f'     -d \'{{"texts": ["hello world"], "model": "{cfg.nim_model}", "input_type": "query"}}\'')
            else:
                print(f'     -d \'{{"input": ["hello world"], "model": "{cfg.nim_model}", "input_type": "query"}}\'')
            print()
            print(f"   Stop with: docker stop {cfg.container_name}")
        else:
            print()
            message = (
                "Health check timeout: detached embedding service did not become ready. "
                f"Check logs with: docker logs {cfg.container_name}"
            )
            if not cfg.allow_unhealthy_detached:
                raise RuntimeError(message)
            print(f"⚠️  {message}")
    else:
        # Run in foreground (interactive)
        print("   Running in foreground. Press Ctrl+C to stop.")
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
            subprocess.run(docker_cmd, env=docker_env)
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
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)

        try:
            cfg = load_config(config_path, cli_overrides, DeployConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_deploy(cfg)


if __name__ == "__main__":
    main()
