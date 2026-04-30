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

"""Cloud backend — Lepton + DGX Cloud.

Most steps pick their cloud submission shape from the runspec's ``launch``
field:

* ``launch = "ray"``   → :func:`execute_cloud_ray` builds a multi-pod
  RayCluster (head + workers) and submits a RayJob. Required by NeMo-RL,
  which expects a single shared Ray cluster spanning every GPU pod.
* otherwise            → :func:`execute_cloud` submits as a Lepton
  distributed workload; ``execute_cloud`` handles its own torchrun wrap
  for ``launch = "torchrun"`` and a bare python invocation otherwise.

Preparation steps are the exception: Xenna owns its own Ray initialization
inside the worker pod, so cloud prep runs as a plain inline workload even when
the local/Slurm runspec uses ``launch = "ray"``.

Both paths share the chunked-env-var source transport (no PatternPackager).
"""

from __future__ import annotations

from nemo_runspec.execution import execute_cloud, execute_cloud_ray
from nemotron.cli.commands.step.backends.base import JobContext


class CloudBackend:
    """Submit to NVIDIA DGX Cloud Lepton or DGX Cloud via nemo-run."""

    name = "cloud"

    def submit(self, ctx: JobContext) -> None:
        rel_script = self._pod_relative_script(str(ctx.script_path))

        if ctx.spec.run.launch == "ray" and not self._uses_inline_cloud(ctx):
            execute_cloud_ray(
                rel_script,
                ctx.train_path,
                env=ctx.env,
                env_vars=ctx.env_vars,
                passthrough=ctx.passthrough,
                attached=ctx.attached,
                default_image=ctx.spec.image,
                script_resources=ctx.spec.resources,
                startup_commands=ctx.startup_commands,
                run_command=ctx.spec.run.cmd,
            )
            return

        # Plain distributed-workload path. Prep steps intentionally arrive here
        # even with launch="ray": Xenna starts Ray inside the single cloud pod,
        # avoiding a Lepton RayCluster whose workers may not share Python deps.
        execute_cloud(
            rel_script,
            ctx.train_path,
            env=ctx.env,
            env_vars=ctx.env_vars,
            passthrough=ctx.passthrough,
            attached=ctx.attached,
            default_image=ctx.spec.image,
            script_resources=ctx.spec.resources,
            startup_commands=ctx.startup_commands,
            run_command=ctx.spec.run.cmd,
            launch=None if self._uses_inline_cloud(ctx) else ctx.spec.run.launch,
        )

    @staticmethod
    def _uses_inline_cloud(ctx: JobContext) -> bool:
        """Return True for steps that should not create cloud RayClusters."""
        return ctx.step_id.startswith("prep/")

    @staticmethod
    def _pod_relative_script(script_path: str) -> str:
        """Strip the local repo root so the cloud pod's cwd resolves the script.

        Drivers see e.g. ``/home/.../src/nemotron/steps/prep/sft_packing/step.py``
        but the pod's workspace is the repo root, so we want
        ``src/nemotron/steps/prep/sft_packing/step.py`` instead.
        """
        for marker in ("/src/nemotron/", "src/nemotron/"):
            if marker in script_path:
                tail = script_path.split(marker, 1)[1]
                return f"src/nemotron/{tail}"
        return script_path
