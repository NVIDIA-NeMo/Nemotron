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

"""Slurm backend — nemo-run SlurmExecutor + SourcePackager via SSH tunnel."""

from __future__ import annotations

import shlex
import time

import typer

from nemo_runspec.execution import (
    create_executor,
    create_slurm_executor,
    prepend_startup_to_cmd,
)
from nemo_runspec.packaging import (
    REMOTE_CONFIG,
    REMOTE_SCRIPT,
    CodePackager,
    SelfContainedPackager,
)
from nemotron.cli.commands.steps.backends.base import JobContext

_CURATOR_RUNTIME_MODULE = "nemotron.steps._bootstrap.curator_runtime"
_REMOTE_SRC_DIR = "/nemo_run/code/src"


class SlurmBackend:
    """Submit to a Slurm cluster (attached or detached) via nemo-run."""

    name = "slurm"

    def submit(self, ctx: JobContext) -> None:
        try:
            import nemo_run as run
        except ImportError:
            typer.echo("nemo-run is required (pip install nemo-run).", err=True)
            raise typer.Exit(1)

        from nemo_runspec.run import (
            patch_nemo_run_ray_template_for_cpu,
            patch_nemo_run_rsync_accept_new_host_keys,
        )
        patch_nemo_run_rsync_accept_new_host_keys()
        patch_nemo_run_ray_template_for_cpu()

        if ctx.spec.run.launch == "ray":
            self._submit_ray(ctx)
            return

        packager_cls = CodePackager if self._uses_code_packager(ctx) else SelfContainedPackager
        packager = packager_cls(
            script_path=str(ctx.script_path),
            train_path=str(ctx.train_path),
        )
        executor = create_executor(
            env=ctx.env,
            env_vars=ctx.env_vars,
            packager=packager,
            attached=ctx.attached,
            force_squash=ctx.force_squash,
            default_image=ctx.spec.image,
            script_resources=ctx.spec.resources,
        )

        # nemo-run's slurm template only escapes Script args correctly via
        # the torchrun launcher path. Run everything through bash -lc so the
        # runspec cmd reaches the worker intact regardless of launch mode.
        # When the author didn't supply a cmd, build a torchrun wrap inline so
        # multi-GPU jobs land with WORLD_SIZE = nproc_per_node × nodes.
        cmd = self._build_cmd(ctx)
        if ctx.passthrough:
            cmd = f"{cmd} {shlex.join(ctx.passthrough)}"
        if ctx.startup_commands:
            cmd = prepend_startup_to_cmd(ctx.startup_commands, cmd)
        task = run.Script(path="bash", args=["-lc", cmd])

        with run.Experiment(ctx.job_name) as exp:
            exp.add(task, executor=executor, name=ctx.job_name)
            exp.run(detach=not ctx.attached)

    def _submit_ray(self, ctx: JobContext) -> None:
        from nemo_run.run.ray.job import RayJob

        # Ray-launched steps need real modules on workers (cloudpickle records
        # function ``__module__`` and Ray's worker resolves it via import). The
        # SelfContainedPackager inlines functions into ``main.py`` whose
        # ``__module__`` becomes ``__main__`` — workers can't find them.
        # CodePackager ships the repo as a proper tree.
        packager = CodePackager(
            script_path=str(ctx.script_path),
            train_path=str(ctx.train_path),
        )
        executor = create_slurm_executor(
            env=ctx.env,
            env_vars=ctx.env_vars,
            packager=packager,
            attached=ctx.attached,
            force_squash=ctx.force_squash,
            default_image=ctx.spec.image,
            script_resources=ctx.spec.resources,
            launcher=None,
        )

        cmd = self._build_cmd(ctx)
        if ctx.passthrough:
            cmd = f"{cmd} {shlex.join(ctx.passthrough)}"
        if ctx.startup_commands:
            cmd = prepend_startup_to_cmd(ctx.startup_commands, cmd)

        ray_job = RayJob(name=ctx.job_name, executor=executor)
        ray_job.start(command=cmd, workdir="")
        if ctx.attached:
            final_state = self._wait_for_ray_job(ray_job)
            if final_state in {"FAILED", "STOPPED", "CANCELLED", "TIMEOUT", "NOT_FOUND"}:
                raise RuntimeError(f"Ray job {ctx.job_name} ended in {final_state}")

    @staticmethod
    def _build_cmd(ctx: JobContext) -> str:
        """Format the worker-side bash command from ``ctx.spec.run``.

        Honors author-supplied ``cmd`` verbatim. When it's None, picks an
        invocation based on ``launch``: torchrun for distributed training
        (so WORLD_SIZE matches the slurm allocation), bare ``python``
        otherwise.
        """
        command_template = SlurmBackend._command_template(ctx)
        if command_template is not None:
            command = command_template.format(script=REMOTE_SCRIPT, config=REMOTE_CONFIG)
            if SlurmBackend._command_uses_curator_runtime(command_template):
                return SlurmBackend._with_remote_src_pythonpath(command)
            return command
        if ctx.spec.run.launch == "torchrun":
            # nemo-run's torchrun launcher is set on the executor and handles
            # the actual srun-side wrap; on this code path we just feed the
            # plain script + args through ``bash -lc``.
            return f"python {REMOTE_SCRIPT} --config {REMOTE_CONFIG}"
        return f"python {REMOTE_SCRIPT} --config {REMOTE_CONFIG}"

    @staticmethod
    def _env_get(env: object, key: str, default: object = None) -> object:
        if env is None:
            return default
        if hasattr(env, "get"):
            return env.get(key, default)
        return getattr(env, key, default)

    @staticmethod
    def _uses_code_packager(ctx: JobContext) -> bool:
        """Data prep steps start Ray internally, so workers need importable modules."""
        return ctx.step_id.startswith("data_prep/") or SlurmBackend._uses_curator_runtime(ctx)

    @staticmethod
    def _uses_curator_runtime(ctx: JobContext) -> bool:
        command_template = SlurmBackend._command_template(ctx)
        return SlurmBackend._command_uses_curator_runtime(command_template)

    @staticmethod
    def _command_template(ctx: JobContext) -> str | None:
        if ctx.spec.run.cmd is not None:
            return ctx.spec.run.cmd
        run_command = SlurmBackend._env_get(ctx.env, "run_command")
        return run_command if isinstance(run_command, str) and run_command else None

    @staticmethod
    def _command_uses_curator_runtime(command: object) -> bool:
        return isinstance(command, str) and _CURATOR_RUNTIME_MODULE in command

    @staticmethod
    def _with_remote_src_pythonpath(command: str) -> str:
        return f"export PYTHONPATH={_REMOTE_SRC_DIR}${{PYTHONPATH:+:$PYTHONPATH}}; {command}"

    @staticmethod
    def _wait_for_ray_job(ray_job: object, *, poll_seconds: int = 30) -> str:
        """Wait for Slurm Ray without tailing remote logs to the submit terminal."""
        terminal_states = {
            "SUCCEEDED",
            "COMPLETED",
            "FAILED",
            "STOPPED",
            "CANCELLED",
            "TIMEOUT",
            "NOT_FOUND",
        }
        last_state: str | None = None
        while True:
            status = ray_job.status(display=False)  # type: ignore[attr-defined]
            state = SlurmBackend._ray_status_state(status)
            if state != last_state:
                typer.echo(f"[ray] state={state}")
                last_state = state
            if state in terminal_states:
                return state
            time.sleep(poll_seconds)

    @staticmethod
    def _ray_status_state(status: object) -> str:
        if isinstance(status, dict):
            return str(
                status.get("state")
                or status.get("status")
                or status.get("job_status")
                or "UNKNOWN"
            )
        return str(getattr(status, "value", status))


# Public alias so the registry can import a Backend, not the module.
__all__ = ["SlurmBackend"]
