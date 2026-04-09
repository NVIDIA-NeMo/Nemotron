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

# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""NeMo-Run patches for Ray CPU templates and rsync host key handling."""

from __future__ import annotations

import os


def patch_nemo_run_ray_template_for_cpu() -> None:
    """Patch nemo-run Ray template to properly handle CPU-only partitions.

    The default nemo_run Ray template hardcodes gpus_per_node=8 and calculates
    CPUs as 16*gpus_per_node, which results in 0 CPUs for CPU-only partitions.

    This patch modifies the template location to use our custom template that
    auto-detects CPUs from SLURM environment variables.
    """
    import tempfile
    from pathlib import Path

    try:
        # Use 'from ... import' syntax to avoid issues with 'run' being shadowed
        # by the nemo_run.run function when using 'import nemo_run.run.ray.slurm'
        from nemo_run.run.ray import slurm as slurm_mod
    except Exception:
        return

    if getattr(slurm_mod, "_nemotron_cpu_template_patched", False):
        return

    # Get the path to our custom template
    custom_template_dir = Path(__file__).parent / "templates"
    custom_template_name = "ray_cpu.sub.j2"

    # Check if our custom template exists
    template_path = custom_template_dir / custom_template_name
    if not template_path.exists():
        return

    def patched_create(
        self,
        pre_ray_start_commands=None,
        dryrun=False,
        command=None,
        workdir=None,
        command_groups=None,
    ):
        """Patched create that uses custom CPU-aware Ray template."""
        name = self.name
        executor = self.executor
        cluster_dir = os.path.join(executor.tunnel.job_dir, name)

        # Use custom template for CPU-aware Ray cluster
        ray_sbatch = slurm_mod.SlurmRayRequest(
            name=name,
            cluster_dir=cluster_dir,
            template_name=custom_template_name,
            template_dir=str(custom_template_dir),
            executor=executor,
            pre_ray_start_commands=pre_ray_start_commands,
            command=command,
            workdir=workdir,
            command_groups=command_groups,
            launch_cmd=["sbatch", "--requeue", "--parsable", "--dependency=singleton"],
        ).materialize()

        if dryrun:
            slurm_mod.logger.debug(f"Dry run: Ray cluster '{name}'")
            print(ray_sbatch)
            return None

        slurm_mod.logger.info(f"Creating Ray cluster '{name}'")
        # Check if a cluster with this name already exists
        try:
            status = self.status()
        except Exception as e:
            # Slurm controller may be temporarily unavailable (e.g., backup controller
            # in standby mode). Proceed with safe defaults rather than failing.
            slurm_mod.logger.warning(
                f"Ray cluster '{name}': failed to query Slurm status; "
                f"proceeding with safe defaults: {e}"
            )
            status = {"job_id": None, "state": "UNKNOWN"}

        if status["job_id"] is not None:
            job_state = status["state"]
            if job_state in ["PENDING", "RUNNING", "CONFIGURING"]:
                slurm_mod.logger.debug(
                    f"Ray cluster '{name}' already exists with ID {status['job_id']} "
                    f"and is currently in {job_state} state. "
                    f"Skipping creation."
                )
                return None
            elif job_state not in [
                "COMPLETING",
                "COMPLETED",
                "CANCELLED",
                "FAILED",
                "TIMEOUT",
                "NOT_FOUND",
            ]:
                slurm_mod.logger.warning(
                    f"Ray cluster '{name}' exists with ID {status['job_id']} "
                    f"in state {job_state}. Creating new cluster anyway."
                )

        # Submit to SLURM - same logic as original nemo-run
        executor.tunnel.connect()
        executor.tunnel.run(f"mkdir -p {cluster_dir}")

        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
            f.write(ray_sbatch)
            f.flush()
            os.fsync(f.fileno())
            ray_sbatch_path = f.name
            executor.tunnel.put(ray_sbatch_path, os.path.join(cluster_dir, "ray.sub"))

        sbatch_cmd = ["sbatch", "--parsable", os.path.join(cluster_dir, "ray.sub")]
        job_id = executor.tunnel.run(" ".join(sbatch_cmd)).stdout.strip()

        # Store job_id in cluster_map
        self.cluster_map[name] = job_id

        slurm_mod.logger.info(f"Slurm job for Ray cluster '{name}' created with ID {job_id}")

        return job_id

    slurm_mod.SlurmRayCluster.create = patched_create
    slurm_mod._nemotron_cpu_template_patched = True


def patch_nemo_run_rsync_accept_new_host_keys() -> None:
    """Patch nemo-run rsync to avoid hanging on first-time host key prompts.

    nemo-run's SSH tunnel uses Paramiko for its control connection, but the
    rsync step shells out to the system `ssh`, which can block waiting for an
    interactive StrictHostKeyChecking prompt.

    We set `StrictHostKeyChecking=accept-new` unless the caller already
    provided a StrictHostKeyChecking option.
    """

    try:
        import nemo_run.core.tunnel.rsync as rsync_mod
    except Exception:
        return

    if getattr(rsync_mod.rsync, "_nemotron_patched", False):
        return

    orig = rsync_mod.rsync

    def patched(*args, **kwargs):
        ssh_opts = kwargs.get("ssh_opts", "") or ""
        if "StrictHostKeyChecking" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + "-o StrictHostKeyChecking=accept-new"
        if "BatchMode" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + "-o BatchMode=yes"
        if "PreferredAuthentications" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + (
                "-o PreferredAuthentications=publickey"
            )
        if "ConnectTimeout" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + "-o ConnectTimeout=30"
        kwargs["ssh_opts"] = ssh_opts

        rsync_opts = kwargs.get("rsync_opts", "") or ""
        # Note: --info=progress2 removed because older rsync versions on some clusters don't support it
        if "--timeout" not in rsync_opts:
            rsync_opts = (rsync_opts + " " if rsync_opts else "") + "--timeout=60"
        # Use --delete for faster incremental syncs (removes stale files on remote)
        if "--delete" not in rsync_opts:
            rsync_opts = (rsync_opts + " " if rsync_opts else "") + "--delete"
        kwargs["rsync_opts"] = rsync_opts

        # Default exclusions for our repo (avoid syncing large non-runtime dirs).
        # Users can override by passing `exclude=...` explicitly.
        # Note: Use patterns anchored at root (e.g., "/artifacts") to avoid
        # excluding source directories like src/nemotron/kit/artifacts.
        kwargs.setdefault(
            "exclude",
            (
                ".git",
                ".venv",
                "__pycache__",
                ".ruff_cache",
                ".pytest_cache",
                ".mypy_cache",
                ".nemotron",
                ".conductor",
                "/output",
                "/outputs",
                "/artifacts",
                "/wandb",
                "usage-cookbook",
                "use-case-examples",
            ),
        )

        # Show progress/errors instead of looking hung.
        kwargs.setdefault("hide_output", False)

        return orig(*args, **kwargs)

    patched._nemotron_patched = True  # type: ignore[attr-defined]
    rsync_mod.rsync = patched  # type: ignore[assignment]

    # Patch already-imported call sites that `from ... import rsync`.
    try:
        import nemo_run.run.experiment as exp

        exp.rsync = patched  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import nemo_run.run.ray.slurm as slurm

        slurm.rsync = patched  # type: ignore[assignment]
    except Exception:
        pass
