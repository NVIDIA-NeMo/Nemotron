# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared NeMo-RL entry logic for every Super3.5 RL sub-stage.

Every stage of the Super 3.5 RL reference run (RLVR, the specialist teachers and
both MOPD phases) runs the same upstream entry point,
``examples/nemo_gym/run_grpo_nemo_gym.py``. MOPD is GRPO with an
``on_policy_distillation`` block, so no stage needs its own training loop.

This module only does what the upstream script cannot:

1. Resolve ``${art:...}`` references (policy, teachers, data) into concrete
   paths. Every other interpolation -- notably ``${oc.env:...}`` secrets such
   as judge API keys -- is left for NeMo-RL to resolve when it loads the
   config, so secrets never land in the handed-off YAML or the logs.
2. Turn checkpoint artifacts into the form NeMo-RL expects: HF checkpoints for
   the policy and MOPD teachers (converting Megatron-Bridge SFT checkpoints and
   NeMo-RL ``step_N`` checkpoints on first use), and the native weights dir for
   ``policy.pretrained_checkpoint`` warm starts.
3. Strip Nemotron-only keys and run the upstream script in-process, so the
   W&B checkpoint-artifact patch stays active.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any

# Upstream NeMo-RL entry point used by every Super3.5 RL stage.
NEMO_RL_ENTRYPOINT = "examples/nemo_gym/run_grpo_nemo_gym.py"

# Keys consumed here that NeMo-RL's MasterConfig does not know about.
_NEMOTRON_ONLY_KEYS = ("run", "convert_megatron_to_hf", "converted_checkpoint_dir")


def parse_args(description: str) -> tuple[argparse.Namespace, list[str]]:
    """Parse ``--config`` plus Hydra-style overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    return parser.parse_known_args()


# =============================================================================
# Checkpoint layout
# =============================================================================


def megatron_weights_dir(path: str | Path) -> Path:
    """Return the dir holding ``iter_*`` subdirs.

    NeMo-RL checkpoints keep Megatron weights under ``step_N/policy/weights``;
    Megatron-Bridge checkpoints keep ``iter_*`` at the top level.
    """
    path = Path(path)
    weights = path / "policy" / "weights"
    return weights if weights.is_dir() else path


def latest_iter_dir(path: str | Path) -> Path | None:
    """Return the newest ``iter_*`` dir of a Megatron checkpoint, if any."""
    root = megatron_weights_dir(path)
    if not root.is_dir():
        return None
    iters = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.startswith("iter_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    return iters[-1] if iters else None


def is_megatron_checkpoint(path: str | Path) -> bool:
    """Return True if ``path`` looks like a Megatron (not HF) checkpoint."""
    path = Path(path)
    if not path.is_dir() or (path / "config.json").exists():
        return False
    return (path / "latest_checkpointed_iteration.txt").exists() or latest_iter_dir(path) is not None


def to_hf_checkpoint(path: str, hf_architecture: str, output_root: str | None = None) -> str:
    """Return an HF checkpoint for ``path``, converting a Megatron one on first use.

    HF dirs and hub ids pass through unchanged. A previous conversion is reused
    from ``<path>/hf`` (or ``<output_root>/<name>_hf``).
    """
    src = Path(path)
    if not is_megatron_checkpoint(src):
        return path
    out = Path(output_root) / f"{src.name}_hf" if output_root else src / "hf"
    if (out / "config.json").exists():
        print(f"Reusing HF conversion of {src} at {out}")
        return str(out)

    iter_dir = latest_iter_dir(src)
    print(f"Converting {iter_dir} -> {out} (architecture from {hf_architecture})")
    if (src / "policy" / "weights").is_dir():
        # NeMo-RL step_N checkpoint: same exporter as NeMo-RL's convert_megatron_to_hf.py.
        from nemo_rl.models.megatron.community_import import export_model_from_megatron

        export_model_from_megatron(
            hf_model_name=hf_architecture,
            input_path=str(iter_dir),
            output_path=str(out),
            hf_tokenizer_path=hf_architecture,
        )
    else:
        # Megatron-Bridge checkpoint (e.g. super35 sft output).
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(hf_architecture, trust_remote_code=True)
        bridge.export_ckpt(megatron_path=str(iter_dir), hf_path=str(out))
    return str(out)


def prepare_checkpoints(config: dict, run_cfg: dict) -> None:
    """Rewrite checkpoint paths in ``config`` into the form NeMo-RL loads."""
    if not run_cfg.get("convert_megatron_to_hf", config.get("convert_megatron_to_hf", True)):
        return
    hf_architecture = run_cfg["hf_architecture"]
    output_root = config.get("converted_checkpoint_dir")

    policy = config["policy"]
    policy["model_name"] = to_hf_checkpoint(policy["model_name"], hf_architecture, output_root)

    # Warm starts load native weights, not HF.
    pretrained = policy.get("pretrained_checkpoint")
    if isinstance(pretrained, dict) and pretrained.get("path"):
        pretrained["path"] = str(megatron_weights_dir(pretrained["path"]))

    # MOPD teachers are served by vLLM from HF checkpoints. Task routes that
    # alias a canonical teacher are still interpolations here and follow it.
    teachers = (config.get("on_policy_distillation") or {}).get("teacher_model_by_agent_name") or {}
    for agent, target in teachers.items():
        if isinstance(target, str) and "${" not in target:
            teachers[agent] = to_hf_checkpoint(target, hf_architecture, output_root)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_artifact_refs(node: Any, config, path: str = "") -> Any:
    """Resolve only leaves containing ``${art:...}``; keep other interpolations."""
    from omegaconf import OmegaConf

    if isinstance(node, dict):
        return {k: _resolve_artifact_refs(v, config, f"{path}.{k}" if path else str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_artifact_refs(v, config, f"{path}[{i}]") for i, v in enumerate(node)]
    if isinstance(node, str) and "${art:" in node:
        return OmegaConf.select(config, path, throw_on_missing=True)
    return node


def load_config(config_path: str, overrides: list[str]) -> tuple[dict, dict]:
    """Load the YAML and apply overrides.

    Returns ``(config, run_cfg)``: the NeMo-RL config with artifact references
    resolved and every other interpolation intact, and the plain ``run:``
    section (artifact names, output artifact, HF architecture).
    """
    from omegaconf import OmegaConf

    from nemo_runspec.config.resolvers import clear_artifact_cache, register_resolvers_from_config

    config = OmegaConf.load(config_path)
    if overrides:
        print(f"Overrides: {overrides}")
        config = OmegaConf.merge(config, OmegaConf.from_dotlist([o.lstrip("+") for o in overrides]))

    clear_artifact_cache()
    register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )
    raw = OmegaConf.to_container(config, resolve=False)
    run_cfg = raw.get("run", {})
    resolved = _resolve_artifact_refs({k: v for k, v in raw.items() if k not in _NEMOTRON_ONLY_KEYS}, config)
    for key in ("convert_megatron_to_hf", "converted_checkpoint_dir"):
        if key in raw:
            resolved[key] = raw[key]
    return resolved, run_cfg


def run(stage_name: str, default_config: str) -> None:
    """Resolve the stage config and run NeMo-RL's GRPO/MOPD entry point.

    The W&B output artifact name comes from ``run.output_artifact`` so that
    variants sharing one script (e.g. RLHF teacher 1 vs 2, MOPD phase 1 vs 2)
    register distinct artifacts.
    """
    from omegaconf import OmegaConf

    from nemotron.kit.wandb_kit import (
        patch_nemo_rl_checkpoint_logging,
        patch_wandb_http_handler_skip_digest_verification,
        patch_wandb_local_file_handler_skip_digest_verification,
        patch_wandb_runid_for_seeded_random,
    )

    args, overrides = parse_args(f"Run Nemotron Super3.5 {stage_name} with NeMo-RL")
    config, run_cfg = load_config(args.config or default_config, overrides)

    patch_wandb_http_handler_skip_digest_verification()
    patch_wandb_local_file_handler_skip_digest_verification()
    patch_wandb_runid_for_seeded_random()
    patch_nemo_rl_checkpoint_logging(artifact_name=run_cfg["output_artifact"])

    prepare_checkpoints(config, run_cfg)
    for key in _NEMOTRON_ONLY_KEYS:
        config.pop(key, None)

    workdir = Path(os.environ.get("NEMO_RL_DIR", "/opt/nemo-rl"))
    entrypoint = workdir / NEMO_RL_ENTRYPOINT
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"NeMo-RL entry point not found at {entrypoint}. "
            "Run inside the Super3.5 RL container or set NEMO_RL_DIR."
        )

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"{stage_name}-", delete=False) as f:
        OmegaConf.save(OmegaConf.create(config), f.name)
        resolved_path = f.name
    print(f"Resolved {stage_name} config written to {resolved_path}")

    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    sys.argv = [str(entrypoint), "--config", resolved_path]
    runpy.run_path(str(entrypoint), run_name="__main__")
