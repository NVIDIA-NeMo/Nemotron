from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nemotron.kit.artifact import ArtifactInput


@dataclass(frozen=True)
class StageCommandMeta:
    name: str
    config_type: type
    handler: Callable[[Any], Any]
    artifacts: dict[str, ArtifactInput]
    defaults_fn: Callable[..., Any] | None = None
    kwargs_schema: type | None = None
    entry_module: str | None = None
    ray: bool = False


def run_stage_tui(meta: StageCommandMeta) -> None:
    from nemotron.kit.tui.app import NemotronTUIApp

    NemotronTUIApp(meta).run()


def maybe_run_stage_tui(app: Any) -> bool:
    """Run the stage TUI for bare nano3 stage invocations.

    This is intentionally Nano3-stage focused and only intercepts:
        nemotron nano3 pretrain
        nemotron nano3 sft
        nemotron nano3 rl

    It does not intercept nested commands like `data prep ...`.

    Returns:
        True if the TUI took over, otherwise False.
    """
    args = sys.argv[1:]
    if len(args) != 1:
        return False

    cmd = args[0]
    if cmd not in {"pretrain", "sft", "rl"}:
        return False

    # Look up leaf command in App._commands.
    # App._commands has historically been a 7-tuple, but now includes a trailing
    # `script_path` (8-tuple). Be tolerant to tuple shape.
    for command in getattr(app, "_commands", []):
        if not command:
            continue

        name = command[0]
        if name != cmd:
            continue

        config_type = command[1]
        handler = command[2]
        artifacts = command[4] if len(command) > 4 else None
        defaults_fn = command[5] if len(command) > 5 else None
        kwargs_schema = command[6] if len(command) > 6 else None

        entry_module = {
            "pretrain": "nemotron.recipes.nano3.stage0_pretrain.train",
            "sft": "nemotron.recipes.nano3.stage1_sft.train",
            "rl": "nemotron.recipes.nano3.stage2_rl.train",
        }[cmd]

        meta = StageCommandMeta(
            name=cmd,
            config_type=config_type,
            handler=handler,
            artifacts=artifacts or {},
            defaults_fn=defaults_fn,
            kwargs_schema=kwargs_schema,
            entry_module=entry_module,
            ray=(cmd == "rl"),
        )
        run_stage_tui(meta)
        return True

    return False


__all__ = [
    "StageCommandMeta",
    "maybe_run_stage_tui",
    "run_stage_tui",
]
