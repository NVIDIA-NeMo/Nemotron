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
from nemotron.cli.commands.step.backends.base import JobContext


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
            ray_job.logs(follow=True)

    @staticmethod
    def _build_cmd(ctx: JobContext) -> str:
        """Format the worker-side bash command from ``ctx.spec.run``.

        Honors author-supplied ``cmd`` verbatim. When it's None, picks an
        invocation based on ``launch``: torchrun for distributed training
        (so WORLD_SIZE matches the slurm allocation), bare ``python``
        otherwise.
        """
        if ctx.spec.run.cmd is not None:
            return ctx.spec.run.cmd.format(script=REMOTE_SCRIPT, config=REMOTE_CONFIG)
        if ctx.spec.run.launch == "torchrun":
            # nemo-run's torchrun launcher is set on the executor and handles
            # the actual srun-side wrap; on this code path we just feed the
            # plain script + args through ``bash -lc``.
            return f"python {REMOTE_SCRIPT} --config {REMOTE_CONFIG}"
        return f"python {REMOTE_SCRIPT} --config {REMOTE_CONFIG}"

    @staticmethod
    def _uses_code_packager(ctx: JobContext) -> bool:
        """Prep steps start Ray internally, so workers need importable modules."""
        return ctx.step_id.startswith("prep/")


# Public alias so the registry can import a Backend, not the module.
__all__ = ["SlurmBackend"]
