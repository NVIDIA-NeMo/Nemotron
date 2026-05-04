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

"""Shared launcher utilities for Model Optimizer based steps."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

FlagStyle = Literal["hyphen", "underscore"]


def exec_torchrun_script(
    *,
    default_config: Path,
    upstream_script: str,
    forwarded_fields: Iterable[str],
    flag_style: FlagStyle = "hyphen",
    default_nproc_per_node: int = 8,
) -> None:
    """Load YAML, translate selected keys into CLI flags, and exec torchrun."""
    config_path, cli_overrides = parse_config_and_overrides(default_config=default_config)
    raw = apply_hydra_overrides(load_omegaconf_yaml(config_path), cli_overrides)
    cfg = OmegaConf.to_container(raw, resolve=True)

    script_cfg = _optional_mapping(cfg.get("script"), "script")
    script = str(script_cfg.get("path") or cfg.get("upstream_script") or upstream_script)
    resolved_flag_style = _resolve_flag_style(script_cfg.get("flag_style", cfg.get("flag_style", flag_style)))

    torchrun = dict(cfg.get("torchrun") or {})
    nproc = int(
        torchrun.get(
            "nproc_per_node", cfg.get("nproc_per_node", os.environ.get("LOCAL_WORLD_SIZE", default_nproc_per_node))
        )
    )

    script_args = to_cli_args(cfg, forwarded_fields=forwarded_fields, flag_style=resolved_flag_style)
    if os.environ.get("WORLD_SIZE") and os.environ.get("RANK"):
        # The step itself was launched by the backend's distributed launcher.
        # Reuse that process group instead of nesting a per-node torchrun, which
        # would split multi-node jobs into independent single-node worlds.
        cmd = [sys.executable, script, *script_args]
    else:
        cmd = ["torchrun", f"--nproc_per_node={nproc}"]
        for key in ("nnodes", "node_rank", "master_addr", "master_port"):
            value = torchrun.get(key, cfg.get(key))
            if value is not None:
                cmd.append(f"--{key}={value}")
        cmd.extend([script, *script_args])
    print(f"$ {shlex.join(cmd)}", flush=True)
    wandb_cfg = _optional_mapping(cfg.get("wandb_wrapper"), "wandb_wrapper")
    if _wandb_wrapper_enabled(wandb_cfg):
        raise SystemExit(_run_with_wandb_wrapper(cmd, cfg, script, script_args, wandb_cfg))
    os.execvp(cmd[0], cmd)


def to_cli_args(
    cfg: dict[str, Any],
    *,
    forwarded_fields: Iterable[str],
    flag_style: FlagStyle,
) -> list[str]:
    """Translate YAML-controlled script args to argparse-compatible CLI arguments.

    Preferred shape:

    ```yaml
    args:
      hf_model_id: model/name
      trust_remote_code: true
    extra_args: ["--new-upstream-flag", "value"]
    ```

    ``forwarded_fields`` keeps older flat configs working and lets a flat Hydra
    override such as ``hf_model_id=...`` override ``args.hf_model_id``.
    """
    args: list[str] = []
    merged_args = dict(_optional_mapping(cfg.get("args"), "args"))
    for key in forwarded_fields:
        if key in cfg and cfg[key] is not None:
            merged_args[key] = cfg[key]
    for key, value in merged_args.items():
        if value is not None:
            _append_flag(args, key, value, flag_style)
    args.extend(str(item) for item in (cfg.get("extra_args") or []))
    return args


def _append_flag(args: list[str], key: str, value: Any, flag_style: FlagStyle) -> None:
    flag = "--" + (key.replace("_", "-") if flag_style == "hyphen" else key)
    if isinstance(value, bool):
        if value:
            args.append(flag)
        return
    if isinstance(value, (list, tuple)):
        args.append(flag)
        args.extend(str(item) for item in value)
        return
    if isinstance(value, dict):
        args.extend([flag, json.dumps(value)])
        return
    args.extend([flag, str(value)])


def _optional_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping when set")
    return value


def _resolve_flag_style(value: Any) -> FlagStyle:
    if value not in ("hyphen", "underscore"):
        raise ValueError("flag_style must be 'hyphen' or 'underscore'")
    return value


def _wandb_wrapper_enabled(wandb_cfg: Mapping[str, Any]) -> bool:
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return False
    rank = _distributed_rank()
    if rank not in (None, 0):
        os.environ.setdefault("WANDB_MODE", "disabled")
        return False
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] wrapper disabled: WANDB_API_KEY is not set", flush=True)
        return False
    project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT")
    if not project:
        print("[wandb] wrapper disabled: no project configured", flush=True)
        return False
    return True


def _distributed_rank() -> int | None:
    """Return the global distributed rank when launched under torchrun/MPI/Slurm."""
    for key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _run_with_wandb_wrapper(
    cmd: list[str],
    cfg: dict[str, Any],
    script: str,
    script_args: list[str],
    wandb_cfg: Mapping[str, Any],
) -> int:
    project = str(wandb_cfg.get("project") or os.environ["WANDB_PROJECT"])
    entity = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
    name = wandb_cfg.get("name") or os.environ.get("WANDB_NAME")

    run = None
    started = time.time()
    try:
        import wandb

        run = wandb.init(
            project=project,
            entity=str(entity) if entity else None,
            name=str(name) if name else None,
            job_type=str(wandb_cfg.get("job_type") or Path(script).stem),
            config=_wandb_safe_config(cfg),
        )
        wandb.summary["upstream_script"] = script
        wandb.summary["command"] = shlex.join(cmd)
        wandb.summary["script_args"] = " ".join(script_args)
        for key, value in _output_path_summary(cfg).items():
            wandb.summary[key] = value
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] wrapper init failed ({type(exc).__name__}: {exc}); continuing without W&B", flush=True)

    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - started

    if run is not None:
        import wandb

        status = "success" if result.returncode == 0 else "failed"
        wandb.log({"exit_code": result.returncode, "duration_seconds": elapsed})
        wandb.summary["status"] = status
        wandb.summary["exit_code"] = result.returncode
        wandb.summary["duration_seconds"] = elapsed
        run.finish(exit_code=result.returncode)
    return result.returncode


def _wandb_safe_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "script": cfg.get("script"),
        "args": cfg.get("args"),
        "torchrun": cfg.get("torchrun"),
        "extra_args": cfg.get("extra_args"),
    }


def _output_path_summary(cfg: dict[str, Any]) -> dict[str, str]:
    args = _optional_mapping(cfg.get("args"), "args")
    keys = ("output_hf_path", "megatron_save_path", "output_dir", "hf_export_path")
    return {f"arg_{key}": str(args[key]) for key in keys if args.get(key)}
