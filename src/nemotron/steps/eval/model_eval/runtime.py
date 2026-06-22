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

"""Runtime helpers for `eval/model_eval`."""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nemo_runspec.config import clear_artifact_cache, register_resolvers_from_config
from nemo_runspec.evaluator import (
    ensure_wandb_host_env,
    get_non_task_args,
    inject_wandb_env_mappings,
    maybe_auto_squash_evaluator,
    needs_wandb,
    parse_task_flags,
    save_eval_configs,
)
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

_STEP_ONLY_KEYS = {
    "dry_run",
    "output_dir",
    "task_filters",
}


def run_model_eval(*, default_config: Path) -> None:
    config_path, cfg, overrides = _load_config(default_config)
    passthrough = _passthrough_args(overrides)
    _validate_passthrough(passthrough)

    launcher_cfg, dry_run, configured_tasks = _build_launcher_config(cfg)
    task_filters = parse_task_flags(passthrough) or configured_tasks
    eval_path = _save_launcher_config(config_path, cfg, launcher_cfg)

    try:
        from nemo_evaluator_launcher.api.functional import run_eval
    except ImportError:
        print("Error: nemo-evaluator-launcher is required for evaluation", file=sys.stderr)
        print("Install with: uv sync --extra evaluator", file=sys.stderr)
        raise SystemExit(1)

    invocation_id = run_eval(launcher_cfg, dry_run=dry_run, tasks=task_filters)
    print(f"launcher_config: {eval_path}")
    if invocation_id:
        print(f"launcher_invocation_id: {invocation_id}")
        print(f"status_command: nemo-evaluator-launcher status {invocation_id}")
        print(f"logs_command: nemo-evaluator-launcher logs {invocation_id}")


def _load_config(default_config: Path) -> tuple[Path, DictConfig, list[str]]:
    config_path, overrides = parse_config_and_overrides(default_config=default_config)
    cfg = apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides)
    return Path(config_path), cfg, overrides


def _build_launcher_config(cfg: DictConfig) -> tuple[DictConfig, bool, list[str] | None]:
    dry_run = bool(cfg.get("dry_run", False))
    output_dir = cfg.get("output_dir")
    task_filters = cfg.get("task_filters")

    _maybe_auto_squash(cfg, dry_run=dry_run)

    if needs_wandb(cfg):
        ensure_wandb_host_env()

    clear_artifact_cache()
    register_resolvers_from_config(cfg, artifacts_key="run", mode="pre_init")
    launcher_dict = dict(OmegaConf.to_container(cfg, resolve=True))
    launcher_dict.pop("run", None)
    for key in _STEP_ONLY_KEYS:
        launcher_dict.pop(key, None)

    if output_dir:
        launcher_dict.setdefault("execution", {})
        launcher_dict["execution"].setdefault("output_dir", output_dir)

    launcher_cfg = OmegaConf.create(launcher_dict)
    if needs_wandb(launcher_cfg):
        ensure_wandb_host_env()
        inject_wandb_env_mappings(launcher_cfg)

    return launcher_cfg, dry_run, list(task_filters) if task_filters else None


def _passthrough_args(overrides: list[str]) -> list[str]:
    """Return non-Hydra passthrough args from direct step.py invocation."""
    return [arg for arg in overrides if arg != "--" and "=" not in arg]


def _validate_passthrough(passthrough: list[str]) -> None:
    extra_args = get_non_task_args(passthrough)
    if extra_args:
        print(
            f"Error: Unknown arguments: {' '.join(extra_args)}\nOnly -t/--task flags are supported for passthrough.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _maybe_auto_squash(cfg: DictConfig, *, dry_run: bool) -> None:
    run = cfg.get("run")
    if not isinstance(run, DictConfig):
        return

    mode = str(run.get("mode", "local"))
    force_squash = bool(run.get("force_squash", False))
    maybe_auto_squash_evaluator(
        cfg,
        mode=mode,
        dry_run=dry_run,
        force_squash=force_squash,
    )


def _save_launcher_config(
    config_path: Path,
    cfg: DictConfig,
    launcher_cfg: DictConfig,
) -> Path:
    if config_path.name == "train.yaml":
        eval_path = config_path.with_name("eval.yaml")
    else:
        _, eval_path = save_eval_configs(cfg, "eval/model_eval")

    OmegaConf.save(launcher_cfg, eval_path)
    return eval_path
