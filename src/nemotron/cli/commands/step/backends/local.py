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

"""Local backend — runs the step.py in a subprocess on this host."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

from nemo_runspec.execution import execute_local
from nemotron.cli.commands.step.backends.base import JobContext


class LocalBackend:
    """Runs locally via subprocess. Honours runspec ``cmd`` and ``launch``."""

    name = "local"

    def submit(self, ctx: JobContext) -> None:
        # No explicit cmd OR torchrun launch → let the shared helper own
        # distributed rendezvous setup (LOCAL_RANK, master port, etc.).
        if ctx.spec.run.cmd is None or ctx.spec.run.launch == "torchrun":
            execute_local(
                str(ctx.script_path),
                ctx.train_path,
                ctx.passthrough,
                torchrun=(ctx.spec.run.launch == "torchrun"),
                env_vars=ctx.env_vars,
                startup_commands=ctx.startup_commands,
            )
            return

        # Author-supplied cmd template (ray / uv-with-extras / …) — verbatim.
        cmd = ctx.spec.run.cmd.format(script=str(ctx.script_path), config=str(ctx.train_path))
        if ctx.passthrough:
            cmd = f"{cmd} {shlex.join(ctx.passthrough)}"
        if ctx.startup_commands:
            cmd = " && ".join([*ctx.startup_commands, cmd])

        env = {**os.environ, **(ctx.env_vars or {})}
        print(f"$ {cmd}", flush=True)
        rc = subprocess.run(["bash", "-lc", cmd], env=env).returncode
        if rc != 0:
            sys.exit(rc)
