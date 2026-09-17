#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "rerank/deploy"
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

"""Deploy a reranking service with NIM or vLLM.

By default, launches the NVIDIA Reranking NIM container. The exported Stage 4
ONNX model directory is mounted into the container and exposed through
NIM_CUSTOM_MODEL. The NIM discovers the mounted custom model at startup and
creates its runtime manifest automatically. Preview VL checkpoints use the
vLLM backend and mount a consolidated Hugging Face checkpoint directly.

Usage:
    # Launch with the Stage 4 ONNX export in foreground
    nemotron rerank deploy -c default

    # Detached mode (background)
    nemotron rerank deploy -c default detach=true

    # Use a TensorRT export instead
    nemotron rerank deploy -c default model_dir=/path/to/stage4_export/tensorrt

    # Serve the image default model
    nemotron rerank deploy -c default model_dir=null

    # Serve the Nemotron 3.5 VL preview with vLLM
    nemotron rerank deploy -c mistral3-vl
"""

from __future__ import annotations

import json
import os
import shlex
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

_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))
_CONTAINER_LABEL_KEY = "nemotron.recipe"
_CONTAINER_LABEL_VALUE = "rerank"


def _default_container_user() -> str:
    """Return the current UID for Docker cache ownership, when available."""
    return str(os.getuid()) if hasattr(os, "getuid") else ""


class DeployConfig(RecipeSettings):
    """Deployment configuration for a NIM or vLLM reranking service."""

    model_config = ConfigDict(extra="forbid")

    # Container settings
    model_family: Literal["llama_text", "mistral3_vl"] = Field(
        default="llama_text",
        description="Model-family deployment contract.",
    )
    backend: Literal["nim", "vllm"] = Field(default="nim", description="Serving backend to launch.")
    nim_image: str = Field(
        default="nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0",
        description="NIM container image to use.",
    )
    vllm_image: str = Field(
        default="nvcr.io/nvidia/vllm:26.06-py3",
        description="vLLM container image to use for the vLLM backend.",
    )
    served_model_name: str = Field(
        default="nvidia/llama-nemotron-rerank-1b-v2",
        description="Model identifier advertised by vLLM.",
    )
    vllm_runner: Literal["pooling"] = Field(default="pooling", description="vLLM runner.")
    vllm_max_model_len: int | None = Field(default=None, gt=0, description="Optional vLLM context limit.")
    vllm_hf_overrides: dict[str, object] | None = Field(
        default=None,
        description="Optional Hugging Face config overrides passed to vLLM.",
    )
    vllm_trust_remote_code: bool = Field(
        default=False,
        description="Allow the vLLM server to load checkpoint-provided Python code.",
    )
    vllm_enforce_eager: bool = Field(default=False, description="Disable CUDA graph capture in vLLM.")
    container_name: str = Field(default="nemotron-rerank-nim", description="Name for the Docker container.")
    replace_existing: bool = Field(default=False, description="Replace an existing container created by this recipe.")

    # Optional NIM model selection. A custom ONNX/TensorRT export directory is
    # mounted and advertised via NIM_CUSTOM_MODEL. NIM discovers the artifact and
    # creates its runtime manifest during startup.
    model_dir: Path | None = Field(
        default_factory=lambda: _OUTPUT_BASE / "output/rerank/stage4_export/onnx",
        description="Optional host directory containing an exported ONNX or TensorRT model artifact.",
    )

    # Container paths
    container_model_path: str = Field(
        default="/opt/nim/custom_model",
        description="Container mount path for the custom exported model artifact.",
    )
    container_cache_path: str = Field(default="/opt/nim/.cache", description="Path inside container for NIM cache.")
    vllm_container_cache_path: str = Field(
        default="/root/.cache/huggingface",
        description="Path inside the vLLM container for Hugging Face cache data.",
    )

    # Network settings
    bind_address: str = Field(default="127.0.0.1", description="Host interface to bind the NIM HTTP port to.")
    host_port: int = Field(default=8000, ge=1, le=65535, description="Port to expose on host.")
    container_port: int = Field(default=8000, ge=1, le=65535, description="Port inside container.")

    # Resource settings
    gpus: Annotated[str, BeforeValidator(str)] = Field(
        default="all",
        description="Number of GPUs to use for the container.",
    )
    shm_size: str = Field(default="16gb", description="Shared memory size.")
    container_user: str = Field(
        default_factory=_default_container_user,
        description="User ID to run as inside the container. Empty string uses the image default.",
    )

    # Runtime settings
    detach: bool = Field(default=False, description="Run container in detached mode.")
    remove_on_exit: bool = Field(default=True, description="Remove container when it exits.")
    keep_failed_container: bool = Field(
        default=False, description="Keep detached containers after health-check failure."
    )
    health_check_timeout: int = Field(default=120, gt=0, description="Timeout in seconds for health check.")
    health_check_interval: int = Field(default=5, gt=0, description="Interval in seconds between health checks.")

    # Environment
    ngc_api_key_env: str = Field(
        default="NGC_API_KEY",
        description="Host environment variable name for the NGC API key.",
    )

    @model_validator(mode="after")
    def _validate_backend(self) -> DeployConfig:
        if self.model_family == "mistral3_vl" and self.backend != "vllm":
            raise ValueError("model_family=mistral3_vl currently supports backend=vllm only")
        if self.backend == "vllm" and self.model_dir is None:
            raise ValueError("model_dir must be set when backend=vllm")
        if self.backend == "vllm" and not self.vllm_image.strip():
            raise ValueError("vllm_image must be set when backend=vllm")
        return self

    @property
    def container_image(self) -> str:
        return self.vllm_image if self.backend == "vllm" else self.nim_image


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


def _docker_label(container_name: str, label_key: str) -> str | None:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            f'{{{{ index .Config.Labels "{label_key}" }}}}',
            container_name,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def stop_existing_container(container_name: str, replace_existing: bool) -> None:
    """Stop and remove an existing recipe-owned container with the same name."""
    label = _docker_label(container_name, _CONTAINER_LABEL_KEY)
    if label is None:
        return
    if label != _CONTAINER_LABEL_VALUE:
        print(
            f"Error: container {container_name!r} already exists but was not created by this recipe.",
            file=sys.stderr,
        )
        print("       Stop or rename it manually before deploying.", file=sys.stderr)
        sys.exit(1)
    if not replace_existing:
        print(f"Error: container {container_name!r} already exists.", file=sys.stderr)
        print("       Set replace_existing=true to replace recipe-owned containers.", file=sys.stderr)
        sys.exit(1)
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)


def _format_command(cmd: list[str]) -> str:
    """Return a shell-escaped command string with no secret values embedded."""
    return " ".join(shlex.quote(part) for part in cmd)


def _client_host(bind_address: str) -> str:
    """Return the host clients should use for the configured bind address."""
    if bind_address in {"", "0.0.0.0"}:
        return "localhost"
    if bind_address == "::":
        return "[::1]"
    if ":" in bind_address and not bind_address.startswith("["):
        return f"[{bind_address}]"
    return bind_address


def _api_base_url(cfg: DeployConfig) -> str:
    return f"http://{_client_host(cfg.bind_address)}:{cfg.host_port}"


def build_docker_command(cfg: DeployConfig) -> tuple[list[str], dict[str, str]]:
    """Build the Docker run command and environment."""
    cmd = ["docker", "run"]
    docker_env = os.environ.copy()

    if cfg.detach:
        cmd.append("-d")
    elif sys.stdin.isatty() and sys.stdout.isatty():
        cmd.extend(["-it"])

    cmd.extend(["--name", cfg.container_name])
    cmd.extend(["--label", f"{_CONTAINER_LABEL_KEY}={_CONTAINER_LABEL_VALUE}"])

    if cfg.remove_on_exit and not cfg.detach:
        cmd.append("--rm")

    cmd.extend(["--gpus", cfg.gpus])
    cmd.extend(["--shm-size", cfg.shm_size])
    if cfg.container_user:
        cmd.extend(["-u", cfg.container_user])
    cmd.extend(["-p", f"{cfg.bind_address}:{cfg.host_port}:{cfg.container_port}"])

    if cfg.backend == "nim":
        ngc_key = os.environ.get(cfg.ngc_api_key_env)
        if ngc_key:
            docker_env["NGC_API_KEY"] = ngc_key
            cmd.extend(["-e", "NGC_API_KEY"])
        else:
            print(f"Warning: {cfg.ngc_api_key_env} not set. NIM may not authenticate properly.")
        cmd.extend(["-e", f"NIM_HTTP_API_PORT={cfg.container_port}"])
        cmd.extend(["-e", f"NIM_CACHE_PATH={cfg.container_cache_path}"])
    if cfg.model_dir is not None:
        model_dir_abs = cfg.model_dir.resolve()
        cmd.extend(["-v", f"{model_dir_abs}:{cfg.container_model_path}:ro"])
        if cfg.backend == "nim":
            cmd.extend(["-e", f"NIM_CUSTOM_MODEL={cfg.container_model_path}"])

    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_name = "huggingface" if cfg.backend == "vllm" else "nim"
    host_cache = Path(cache_dir) / cache_name
    host_cache.mkdir(parents=True, exist_ok=True)
    container_cache = cfg.vllm_container_cache_path if cfg.backend == "vllm" else cfg.container_cache_path
    cmd.extend(["-v", f"{host_cache}:{container_cache}"])

    cmd.append(cfg.container_image)
    if cfg.backend == "vllm":
        cmd.extend(
            [
                "vllm",
                "serve",
                cfg.container_model_path,
                "--served-model-name",
                cfg.served_model_name,
                "--runner",
                cfg.vllm_runner,
                "--host",
                "0.0.0.0",
                "--port",
                str(cfg.container_port),
            ]
        )
        if cfg.vllm_max_model_len is not None:
            cmd.extend(["--max-model-len", str(cfg.vllm_max_model_len)])
        if cfg.vllm_hf_overrides is not None:
            cmd.extend(["--hf-overrides", json.dumps(cfg.vllm_hf_overrides, separators=(",", ":"))])
        if cfg.vllm_trust_remote_code:
            cmd.append("--trust-remote-code")
        if cfg.vllm_enforce_eager:
            cmd.append("--enforce-eager")

    return cmd, docker_env


def wait_for_health(cfg: DeployConfig) -> bool:
    """Wait for the selected reranking service to become healthy."""
    import urllib.error
    import urllib.request

    health_path = "/health" if cfg.backend == "vllm" else "/v1/health/ready"
    health_url = f"{_api_base_url(cfg)}{health_path}"
    start_time = time.time()

    print(f"   Waiting for {cfg.backend.upper()} to become healthy (timeout: {cfg.health_check_timeout}s)...")

    while time.time() - start_time < cfg.health_check_timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            pass

        time.sleep(cfg.health_check_interval)
        elapsed = int(time.time() - start_time)
        print(f"   ... still waiting ({elapsed}s)")

    return False


def run_deploy(cfg: DeployConfig) -> dict:
    """Run reranker deployment."""
    print(f"{cfg.backend.upper()} Reranking Service Deployment")
    print("=" * 60)
    print(f"Container image: {cfg.container_image}")
    print(f"Container name:  {cfg.container_name}")
    print(f"Host bind:       {cfg.bind_address}:{cfg.host_port}")
    print(f"Container port:  {cfg.container_port}")
    print(f"GPUs:            {cfg.gpus}")
    print(f"Detached:        {cfg.detach}")
    if cfg.model_dir is not None:
        print(f"Custom model:    {cfg.model_dir}")
    else:
        print("Model:           image default")
    print("=" * 60)
    print()

    if not check_docker():
        print("Error: Docker is not installed or not running.")
        sys.exit(1)

    if not check_nvidia_docker():
        print("Warning: NVIDIA Container Runtime may not be available.")

    if cfg.model_dir is not None and not cfg.model_dir.exists():
        print(f"Error: model_dir not found: {cfg.model_dir}", file=sys.stderr)
        print(
            "       Run rerank export first, or set model_dir=null to serve the image default model.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Stopping existing recipe-owned container (if any)...")
    stop_existing_container(cfg.container_name, cfg.replace_existing)

    docker_cmd, docker_env = build_docker_command(cfg)
    print(f"Starting {cfg.backend} container...")
    print(f"   Command: {_format_command(docker_cmd)}")
    print()

    result = {
        "container_name": cfg.container_name,
        "host_port": cfg.host_port,
        "api_url": f"{_api_base_url(cfg)}/v1/rerank" if cfg.backend == "vllm" else f"{_api_base_url(cfg)}/v1/ranking",
    }
    if cfg.model_dir is not None:
        result["model_dir"] = str(cfg.model_dir)

    if cfg.detach:
        proc = subprocess.run(docker_cmd, capture_output=True, text=True, env=docker_env)
        if proc.returncode != 0:
            print(f"Error starting container: {proc.stderr}")
            sys.exit(1)

        container_id = proc.stdout.strip()
        result["container_id"] = container_id
        print(f"   Container ID: {container_id[:12]}")

        if wait_for_health(cfg):
            print()
            print(f"{cfg.backend.upper()} is ready!")
            print(f"   API endpoint: {result['api_url']}")
            print()
            print("   Test with:")
            print(f"   curl -X POST {result['api_url']}")
            print("     -H 'Content-Type: application/json'")
            if cfg.backend == "vllm":
                sample_payload = json.dumps(
                    {
                        "model": cfg.served_model_name,
                        "query": "what is AI?",
                        "documents": ["AI is artificial intelligence"],
                        "top_n": 1,
                    }
                )
            else:
                sample_payload = (
                    '{"model": "nvidia/llama-nemotron-rerank-1b-v2", '
                    '"query": {"text": "what is AI?"}, '
                    '"passages": [{"text": "AI is artificial intelligence"}], '
                    '"truncate": "END"}'
                )
            print(f"     -d '{sample_payload}'")
            print()
            print(f"   Stop with: docker stop {cfg.container_name}")
        else:
            print()
            print(f"Error: Health check timed out after {cfg.health_check_timeout}s.", file=sys.stderr)
            if cfg.keep_failed_container:
                print(f"   Check logs with: docker logs {cfg.container_name}", file=sys.stderr)
            else:
                print(f"   Removing failed container: {cfg.container_name}", file=sys.stderr)
                stop_existing_container(cfg.container_name, True)
            sys.exit(1)
    else:
        print("   Running in foreground. Press Ctrl+C to stop.")
        print()

        def signal_handler(signum, frame):
            print("\n   Shutting down...")
            stop_existing_container(cfg.container_name, True)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            proc = subprocess.run(docker_cmd, env=docker_env)
        except KeyboardInterrupt:
            proc = None
        if proc is not None and proc.returncode != 0:
            sys.exit(proc.returncode)

    return result


def main(cfg: DeployConfig | None = None) -> dict:
    """Entry point for deployment."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)

        try:
            cfg = load_config(config_path, cli_overrides, DeployConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_deploy(cfg)


if __name__ == "__main__":
    main()
