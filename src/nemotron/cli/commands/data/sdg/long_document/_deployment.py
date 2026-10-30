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

"""Auto-deploy support for long-document SDG recipes.

When ``--serve`` is passed to a producer stage (``ocr``, ``text-qa``, …),
the CLI:

  1. Loads a ``DeploymentConfig`` from ``deployment/<name>.yaml`` (schema
     mirrors Evaluator-launcher's ``vllm.yaml`` so the two orgs can converge
     later).  The default ``<name>`` per stage is hard-coded in
     ``STAGE_DEFAULT_DEPLOYMENT`` below; operators override via
     ``--serve-config <name>``.
  2. Composes a multi-task ``nemo_run.Experiment``: a ``serve_task`` on the
     GPU partition that brings vLLM up + publishes its endpoint to a
     sentinel file on shared storage, and a ``client_task`` (the existing
     recipe path) that waits on that sentinel before injecting
     ``vllm_endpoint=<url>`` into the recipe config.

The serve task watches for ``<sentinel>.done`` and clean-shuts vLLM when the
client trap writes it, which fires on the client's normal completion or any
non-fatal failure.  Slurm walltime backstops abnormal exits (OOM,
preemption).

This module is intentionally self-contained — no imports from the recipe
scripts — so it loads at CLI startup without requiring ``data_designer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Per-stage default deployment lookup
# --------------------------------------------------------------------------- #

# Maps the SDG stage name (matches the runspec's `name` tail) to the
# deployment-config YAML stem under
# ``recipes/data/sdg/long-document/deployment/``.  When a user passes
# ``--serve`` without ``--serve-config``, this is the deployment that gets
# brought up.
STAGE_DEFAULT_DEPLOYMENT: dict[str, str] = {
    "ocr": "nemotron-parse-v1.1",
    "text-qa": "gpt-oss-120b",
    "page-classification": "qwen3-vl-30b",
    "visual-qa": "qwen3-vl-235b",
    "single-page-qa": "qwen3-vl-235b",
    "windowed-qa": "qwen3-vl-235b",
    "whole-document-qa": "qwen3-vl-235b",
}


# --------------------------------------------------------------------------- #
# Pydantic schema (mirrors Evaluator-launcher's vllm.yaml)
# --------------------------------------------------------------------------- #


class DeploymentConfig(BaseModel):
    """Validated deployment-config for a long-document SDG model serve task."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Identity
    name: str = Field(..., description="Stable identifier for this deployment config (used in run dirs and logs).")
    hf_model_handle: str = Field(..., description="HuggingFace model id served via vLLM.")
    served_model_name: str = Field(..., description="Name vLLM publishes on /v1/* endpoints.")

    # Container
    image: str = Field(..., description="vLLM-capable container image.")

    # vLLM tuning
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Pin a specific port for vLLM.  Default (null) picks a free "
                    "ephemeral port at runtime — robust against shared-node "
                    "collisions (DCGM exporter / Prometheus on 8000, etc.).",
    )
    tensor_parallel_size: int = Field(default=1, gt=0)
    pipeline_parallel_size: int = Field(default=1, gt=0)
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    max_model_len: int | None = Field(default=None, description="vLLM --max-model-len; omit to use the model's default.")
    max_num_seqs: int | None = Field(default=None, description="vLLM --max-num-seqs.")
    trust_remote_code: bool = Field(default=False)
    extra_args: str = Field(default="", description="Verbatim extra `vllm serve` flags.")

    # Container-side env / preamble
    env_vars: dict[str, str] = Field(default_factory=dict)
    pre_cmd: str = Field(default="", description="Optional shell preamble run inside the container before `vllm serve`.")

    # Optional chat template (some models — e.g. Nemotron-Parse — require one)
    chat_template_jinja: str | None = Field(default=None, description="Inline Jinja chat template. Written to a tmp file and passed via --chat-template.")

    # Slurm resources for the serve task
    nodes: int = Field(default=1, gt=0)
    gpus_per_node: int = Field(default=1, gt=0)
    walltime: str = Field(default="08:00:00", description="Slurm --time for the serve job.")
    partition: str | None = Field(default=None, description="Slurm partition; falls back to env.toml's `sdg_serve_partition` when null.")


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def deployment_config_dir() -> Path:
    """Absolute path to the deployment-config directory."""
    # This module lives at <repo>/src/nemotron/cli/commands/data/sdg/long_document/_deployment.py
    # The deployment YAMLs live at <repo>/src/nemotron/recipes/data/sdg/long-document/deployment/
    return Path(__file__).resolve().parents[5] / "recipes" / "data" / "sdg" / "long-document" / "deployment"


def load_deployment_config(name: str, overrides: list[str] | None = None) -> DeploymentConfig:
    """Load ``deployment/<name>.yaml`` and apply Hydra-style dotlist overrides.

    Args:
        name: deployment-config stem (e.g. ``"nemotron-parse-v1.1"``).
        overrides: optional list of ``key=value`` overrides (e.g.
            ``["tensor_parallel_size=4", "extra_args=--reasoning-parser deepseek_r1"]``).

    Returns:
        Validated ``DeploymentConfig``.

    Raises:
        FileNotFoundError: if ``deployment/<name>.yaml`` does not exist.
    """
    yaml_path = deployment_config_dir() / f"{name}.yaml"
    if not yaml_path.exists():
        available = sorted(p.stem for p in deployment_config_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"Deployment config '{name}' not found at {yaml_path}.\n"
            f"Available: {', '.join(available) if available else '(none)'}"
        )

    yaml_cfg = OmegaConf.load(yaml_path)
    override_cfg = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})
    merged = OmegaConf.merge(yaml_cfg, override_cfg)
    container: dict[str, Any] = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]
    return DeploymentConfig(**container)


# --------------------------------------------------------------------------- #
# Bash template generators
# --------------------------------------------------------------------------- #


def build_serve_bash(cfg: DeploymentConfig, sentinel_path: str) -> str:
    """Emit the bash that the serve task runs inside its container.

    Behaviour:
      1. (Optional) write the chat template to a tmp file.
      2. (Optional) run ``pre_cmd``.
      3. Start ``vllm serve`` in the background.
      4. Poll ``/health`` until 200; bail if vLLM dies first.
      5. Write ``http://<host>:<port>/v1`` to ``sentinel_path``.
      6. Wait for ``<sentinel_path>.done`` to appear; SIGTERM vLLM and exit.

    The resulting string is suitable for use as a ``run.Script`` body or
    embedded into an sbatch wrapper.
    """
    template_arg = ""
    template_setup = ""
    if cfg.chat_template_jinja:
        template_setup = (
            "TEMPLATE_PATH=$(mktemp /tmp/sdg_chat_template.XXXXXX.jinja)\n"
            f"cat > \"$TEMPLATE_PATH\" <<'__SDG_CHAT_TEMPLATE__'\n"
            f"{cfg.chat_template_jinja.rstrip()}\n"
            "__SDG_CHAT_TEMPLATE__\n"
        )
        template_arg = '--chat-template "$TEMPLATE_PATH"'

    env_export = "\n".join(f'export {k}={_shell_quote(v)}' for k, v in cfg.env_vars.items())

    serve_args = [
        f'--tensor-parallel-size {cfg.tensor_parallel_size}',
        f'--pipeline-parallel-size {cfg.pipeline_parallel_size}',
        f'--gpu-memory-utilization {cfg.gpu_memory_utilization}',
        '--port "$PORT"',
        f'--served-model-name {_shell_quote(cfg.served_model_name)}',
    ]
    if cfg.max_model_len is not None:
        serve_args.append(f'--max-model-len {cfg.max_model_len}')
    if cfg.max_num_seqs is not None:
        serve_args.append(f'--max-num-seqs {cfg.max_num_seqs}')
    if cfg.trust_remote_code:
        serve_args.append('--trust-remote-code')
    if cfg.extra_args:
        serve_args.append(cfg.extra_args)
    if template_arg:
        serve_args.append(template_arg)
    serve_args_str = " \\\n    ".join(serve_args)

    # Resolve the port: pinned in the deployment YAML, or dynamically picked
    # at runtime by binding an ephemeral socket and reading its port.  The
    # dynamic path is the default — pinning a port only works on dedicated
    # nodes where collisions are impossible.
    if cfg.port is None:
        port_setup = (
            "PORT=$(python3 -c \"import socket; s=socket.socket(); "
            "s.bind(('', 0)); print(s.getsockname()[1]); s.close()\")\n"
            "if ! [ \"$PORT\" -gt 0 ] 2>/dev/null; then\n"
            "    echo \"[serve] failed to pick a free port\" >&2\n"
            "    exit 1\n"
            "fi\n"
            "echo \"[serve] selected dynamic port $PORT\"\n"
        )
    else:
        port_setup = f'PORT={cfg.port}\necho "[serve] using pinned port $PORT"\n'

    return f"""#!/bin/bash
set -euo pipefail

SENTINEL={_shell_quote(sentinel_path)}
SENTINEL_DONE="${{SENTINEL}}.done"
mkdir -p "$(dirname "$SENTINEL")"

# Clean up any stale sentinel from a prior run with the same id.
rm -f "$SENTINEL" "$SENTINEL_DONE"

{env_export}

{template_setup}
{cfg.pre_cmd}

{port_setup}

# Start vLLM in the background on the resolved port.
vllm serve {_shell_quote(cfg.hf_model_handle)} \\
    {serve_args_str} &
VLLM_PID=$!

cleanup() {{
    echo "[serve] cleanup: SIGTERM vllm pid=$VLLM_PID"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    rm -f "$SENTINEL" "$SENTINEL_DONE"
}}
trap cleanup EXIT INT TERM

# Determine the externally-reachable host (slurm node hostname is reachable
# inside the cluster's network).
HOST=$(hostname -f)
ENDPOINT="http://${{HOST}}:${{PORT}}/v1"
HEALTH_URL="http://${{HOST}}:${{PORT}}/health"
MODELS_URL="http://${{HOST}}:${{PORT}}/v1/models"

echo "[serve] vLLM starting on $ENDPOINT (pid=$VLLM_PID)"

# Wait for vLLM to be ready: /health returns 200 AND /v1/models lists our
# served_model_name.  Even on a dynamically-selected port, double-checking
# the model name protects against the unlikely case that another process
# grabbed the port between our pick and vLLM's bind.
EXPECTED_MODEL={_shell_quote(cfg.served_model_name)}
HEALTH_TIMEOUT_SEC=1800   # 30 min
WAITED=0
while true; do
    HEALTH_CODE=$(curl -fsS -o /dev/null -w "%{{http_code}}" "$HEALTH_URL" 2>/dev/null || echo 000)
    if [ "$HEALTH_CODE" = "200" ]; then
        # /health passed — verify the model list contains our model.
        MODELS_BODY=$(curl -fsS "$MODELS_URL" 2>/dev/null || echo '')
        if echo "$MODELS_BODY" | grep -qF "\"$EXPECTED_MODEL\""; then
            echo "[serve] vLLM ready after ${{WAITED}}s on $ENDPOINT (model '$EXPECTED_MODEL' registered)"
            break
        fi
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[serve] vLLM died before becoming healthy (last /health=$HEALTH_CODE)"
        exit 1
    fi
    if [ "$WAITED" -ge "$HEALTH_TIMEOUT_SEC" ]; then
        echo "[serve] vLLM did not become ready within $HEALTH_TIMEOUT_SEC seconds"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

# Publish the endpoint atomically (write to .tmp + mv) so the client never
# reads a half-written file.
echo "$ENDPOINT" > "${{SENTINEL}}.tmp"
mv "${{SENTINEL}}.tmp" "$SENTINEL"
echo "[serve] published endpoint to $SENTINEL: $ENDPOINT"

# Wait until the client signals completion via <sentinel>.done.
while true; do
    if [ -f "$SENTINEL_DONE" ]; then
        echo "[serve] received done signal at $SENTINEL_DONE"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[serve] vLLM died unexpectedly"
        exit 1
    fi
    sleep 30
done
"""


def build_client_preamble(sentinel_path: str, *, max_wait_secs: int = 1800) -> str:
    """Emit a bash preamble for the client task that:

    - polls the sentinel for the serve endpoint
    - exports VLLM_ENDPOINT for the recipe to consume
    - traps EXIT so the serve task gets a clean shutdown signal

    The returned snippet is intended to be prepended to whatever command
    invokes the recipe.
    """
    return f"""SENTINEL={_shell_quote(sentinel_path)}
SENTINEL_DONE="${{SENTINEL}}.done"

# Clean signal to serve task on any exit.
__sdg_signal_done() {{
    touch "$SENTINEL_DONE" 2>/dev/null || true
}}
trap __sdg_signal_done EXIT INT TERM

# Wait for the serve task to publish its endpoint.
echo "[client] waiting for serve endpoint at $SENTINEL"
WAITED=0
while [ ! -s "$SENTINEL" ]; do
    if [ "$WAITED" -ge {max_wait_secs} ]; then
        echo "[client] timed out after {max_wait_secs}s waiting for $SENTINEL"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
VLLM_ENDPOINT=$(cat "$SENTINEL")
export VLLM_ENDPOINT
echo "[client] using endpoint: $VLLM_ENDPOINT"
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _shell_quote(value: str) -> str:
    """Quote *value* for safe inclusion in bash, including ``$`` and quotes."""
    if not value:
        return "''"
    if all(c.isalnum() or c in "@%+=:,./-_" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


__all__ = [
    "DeploymentConfig",
    "STAGE_DEFAULT_DEPLOYMENT",
    "build_client_preamble",
    "build_serve_bash",
    "deployment_config_dir",
    "load_deployment_config",
]
