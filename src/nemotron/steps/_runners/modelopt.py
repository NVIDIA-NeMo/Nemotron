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
from collections.abc import Iterable
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

    script = str(cfg.get("upstream_script") or upstream_script)
    torchrun = dict(cfg.get("torchrun") or {})
    nproc = int(
        torchrun.get(
            "nproc_per_node", cfg.get("nproc_per_node", os.environ.get("LOCAL_WORLD_SIZE", default_nproc_per_node))
        )
    )

    cmd = ["torchrun", f"--nproc_per_node={nproc}"]
    for key in ("nnodes", "node_rank", "master_addr", "master_port"):
        value = torchrun.get(key, cfg.get(key))
        if value is not None:
            cmd.append(f"--{key}={value}")

    script_args = to_cli_args(cfg, forwarded_fields=forwarded_fields, flag_style=flag_style)
    cmd.extend([script, *script_args])
    print(f"$ {shlex.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)


def to_cli_args(
    cfg: dict[str, Any],
    *,
    forwarded_fields: Iterable[str],
    flag_style: FlagStyle,
) -> list[str]:
    """Translate named YAML fields to argparse-compatible CLI arguments."""
    args: list[str] = []
    for key in forwarded_fields:
        if key not in cfg or cfg[key] is None:
            continue
        _append_flag(args, key, cfg[key], flag_style)
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
