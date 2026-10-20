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

The default NIM backend mounts the Stage 4 ONNX/TensorRT export through
NIM_CUSTOM_MODEL. The Nemotron 3.5 profile serves the consolidated Stage 2
checkpoint with vLLM, retaining its classifier head and query-document template.
Set model_dir=null with that profile to serve the public base checkpoint.
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
    """Deployment configuration for a NIM or Nemotron 3.5 vLLM service."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: Path = Field(
        default_factory=lambda: _OUTPUT_BASE / "output/rerank",
        description="Root directory for rerank checkpoints and evaluation outputs.",
    )

    # Container settings
    backend: Literal["nim", "vllm"] = Field(default="nim", description="Serving backend to launch.")
    vllm_image: str = Field(default="vllm/vllm-openai:v0.31.0", description="vLLM container image.")
    vllm_model: str = Field(
        default="nvidia/Nemotron-3.5-Rerank-8B-BF16",
        description="Served model name and Hugging Face model ID when model_dir is null.",
    )
    model_revision: str | None = Field(default=None, description="Hugging Face revision when model_dir is null.")
    vllm_max_model_len: int = Field(default=4096, gt=0, description="Maximum combined query/document length.")
    vllm_max_num_seqs: int = Field(default=4, gt=0, description="Maximum concurrent vLLM sequences.")
    vllm_gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1, description="vLLM GPU memory fraction.")
    nim_image: str = Field(
        default="nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0",
        description="NIM container image to use.",
    )
    container_name: str = Field(default="nemotron-rerank-nim", description="Name for the Docker container.")
    replace_existing: bool = Field(default=False, description="Replace an existing container created by this recipe.")

    # NIM consumes Stage 4 exports; vLLM consumes native Stage 2 checkpoints.
    model_dir: Path | None = Field(
        default_factory=lambda data: (
            data["artifact_root"]
            / (
                "stage2_finetune/checkpoints/LATEST/model/consolidated"
                if data["backend"] == "vllm"
                else "stage4_export/onnx"
            )
        ),
        description="Local NIM export or vLLM consolidated checkpoint; null selects the backend base model.",
    )

    # Container paths
    container_model_path: str = Field(
        default="/opt/nim/custom_model",
        description="Container mount path for local model artifacts.",
    )
    container_cache_path: str = Field(default="/opt/nim/.cache", description="Path inside container for NIM cache.")

    vllm_container_cache_path: str = Field(
        default="/opt/cache/huggingface", description="Hugging Face cache path inside the vLLM container."
    )

    # Network settings
    bind_address: str = Field(default="127.0.0.1", description="Host interface to bind the service HTTP port to.")
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
        if self.backend == "vllm" and (not self.vllm_image.strip() or not self.vllm_model.strip()):
            raise ValueError("vllm_image and vllm_model must be nonempty when backend=vllm")
        return self

    @property
    def container_image(self) -> str:
        return self.vllm_image if self.backend == "vllm" else self.nim_image


# vLLM's scoring endpoints require an explicit template. Extract the checkpoint's
# template inside the serving environment so deploy does not need Transformers.
_VLLM_LAUNCH = """import os, sys, tempfile
from transformers import AutoTokenizer
model, revision = sys.argv[1:3]
tokenizer = AutoTokenizer.from_pretrained(model, revision=revision or None)
template = tokenizer.get_chat_template()
if not template:
    raise ValueError("The reranker checkpoint has no query-document chat template")
with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as output:
    output.write(template)
    template_path = output.name
os.execvp("vllm", ["vllm", "serve", model, "--chat-template", template_path, *sys.argv[3:]])
"""


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

    if cfg.model_dir is not None:
        cmd.extend(["-v", f"{cfg.model_dir.resolve()}:{cfg.container_model_path}:ro"])

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
    host_cache = cache_dir / ("huggingface" if cfg.backend == "vllm" else "nim")
    host_cache.mkdir(parents=True, exist_ok=True)
    container_cache = cfg.vllm_container_cache_path if cfg.backend == "vllm" else cfg.container_cache_path
    cmd.extend(["-v", f"{host_cache}:{container_cache}"])

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
            cmd.extend(["-e", f"NIM_CUSTOM_MODEL={cfg.container_model_path}"])
        cmd.append(cfg.nim_image)
    else:
        # The image's HOME is /root, but the recipe runs as the host UID.
        # Keep model/runtime caches writable without changing cache ownership.
        cmd.extend(
            [
                "-e",
                f"HF_HOME={container_cache}",
                "-e",
                f"XDG_CACHE_HOME={container_cache}",
                "-e",
                f"XDG_CONFIG_HOME={container_cache}/config",
                "-e",
                f"FLASHINFER_WORKSPACE_BASE={container_cache}",
                "-e",
                f"TRITON_CACHE_DIR={container_cache}/triton",
                "-e",
                f"CUDA_CACHE_PATH={container_cache}/cuda",
            ]
        )
        if os.environ.get("HF_TOKEN"):
            cmd.extend(["-e", "HF_TOKEN"])
        cmd.extend(["--entrypoint", "python3", cfg.vllm_image, "-c", _VLLM_LAUNCH])
        model = cfg.container_model_path if cfg.model_dir is not None else cfg.vllm_model
        revision = cfg.model_revision if cfg.model_dir is None else None
        cmd.extend([model, revision or ""])
        cmd.extend(
            [
                "--served-model-name",
                cfg.vllm_model,
                "--runner",
                "pooling",
                "--dtype",
                "bfloat16",
                "--pooler-config",
                json.dumps({"seq_pooling_type": "MEAN", "use_activation": False}),
                "--chat-template-content-format",
                "openai",
                "--max-model-len",
                str(cfg.vllm_max_model_len),
                "--max-num-seqs",
                str(cfg.vllm_max_num_seqs),
                "--gpu-memory-utilization",
                str(cfg.vllm_gpu_memory_utilization),
                "--limit-mm-per-prompt",
                json.dumps({"image": 1}),
                "--host",
                "0.0.0.0",
                "--port",
                str(cfg.container_port),
            ]
        )
        if revision:
            cmd.extend(["--revision", revision])

    return cmd, docker_env


def wait_for_health(cfg: DeployConfig) -> bool:
    """Wait for the selected service to become healthy."""
    import urllib.error
    import urllib.request

    health_path = "/health" if cfg.backend == "vllm" else "/v1/health/ready"
    health_url = f"{_api_base_url(cfg)}{health_path}"
    start_time = time.time()

    print(f"   Waiting for {cfg.backend} to become healthy (timeout: {cfg.health_check_timeout}s)...")

    while time.time() - start_time < cfg.health_check_timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            pass

        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", cfg.container_name],
            capture_output=True,
            text=True,
        )
        if state.returncode != 0 or state.stdout.strip() == "false":
            print(f"   Container exited before becoming healthy. Check: docker logs {cfg.container_name}")
            return False

        time.sleep(cfg.health_check_interval)
        elapsed = int(time.time() - start_time)
        print(f"   ... still waiting ({elapsed}s)")

    return False


def run_deploy(cfg: DeployConfig) -> dict:
    """Run reranker deployment with the selected backend."""
    print(f"{cfg.backend} Reranking Service Deployment")
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
        print(f"Base model:      {cfg.vllm_model if cfg.backend == 'vllm' else 'image default'}")
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
            "       Select an existing model directory, or set model_dir=null to serve the base model.",
            file=sys.stderr,
        )
        sys.exit(1)

    if cfg.backend == "vllm" and cfg.model_dir is not None:
        required = ("config.json", "tokenizer_config.json")
        missing = [name for name in required if not (cfg.model_dir / name).is_file()]
        if not any(cfg.model_dir.glob("*.safetensors")):
            missing.append("*.safetensors")
        if missing:
            print(f"Error: incomplete consolidated checkpoint: {', '.join(missing)}", file=sys.stderr)
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
        "api_url": f"{_api_base_url(cfg)}{'/classify' if cfg.backend == 'vllm' else '/v1/ranking'}",
        "backend": cfg.backend,
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
            print(f"{cfg.backend} is ready!")
            print(f"   API endpoint: {result['api_url']}")
            print()
            print("   Test with:")
            if cfg.backend == "vllm":
                sample_payload = json.dumps(
                    {
                        "model": cfg.vllm_model,
                        "messages": [
                            {"role": "query", "content": [{"type": "text", "text": "What is AI?"}]},
                            {
                                "role": "document",
                                "content": [{"type": "text", "text": "AI is artificial intelligence."}],
                            },
                        ],
                        "add_special_tokens": False,
                        "use_activation": False,
                    }
                )
            else:
                sample_payload = json.dumps(
                    {
                        "model": "nvidia/llama-nemotron-rerank-1b-v2",
                        "query": {"text": "what is AI?"},
                        "passages": [{"text": "AI is artificial intelligence"}],
                        "truncate": "END",
                    }
                )
            print(
                f"   curl -X POST {shlex.quote(result['api_url'])} "
                f"-H 'Content-Type: application/json' -d {shlex.quote(sample_payload)}"
            )
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
