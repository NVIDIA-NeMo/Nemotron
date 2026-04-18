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

"""NeMo-Run patches for Ray CPU templates, rsync host key handling, and cloud executor Ray backends."""

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


def _make_configs_excluding_copy_fn(original_signature: str):
    """Build a ``copy_directory_data_command`` replacement that skips ``configs/``.

    Lepton returns ``list[str]`` (``["sh", "-c", cmd]``); DGXCloud returns a
    single ``str`` (command body only). The ``original_signature`` switch
    picks the right return shape.
    """
    import base64
    import os
    import subprocess
    import tempfile

    def _build_cmd(local_dir_path: str, dest_path: str) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            tarball_path = os.path.join(temp_dir, "archive.tar.gz")
            # Exclude ``configs/`` — only nemo-run's local state lives there
            # and the pod entrypoint never reads it. Skipping it keeps the
            # resulting single argv string under the kernel's 128 KiB
            # ``MAX_ARG_STRLEN`` limit even when env-var chunks inflate the
            # serialized executor config.
            subprocess.run(
                f"tar --exclude='./configs' -czf {tarball_path} -C {local_dir_path} .",
                shell=True,
                check=True,
            )
            with open(tarball_path, "rb") as file:
                encoded_data = base64.b64encode(file.read()).decode("utf-8")
        return (
            f"rm -rf {dest_path} && mkdir -p {dest_path} && "
            f"echo {encoded_data} | base64 -d > {dest_path}/archive.tar.gz && "
            f"tar -xzf {dest_path}/archive.tar.gz -C {dest_path} && "
            f"rm {dest_path}/archive.tar.gz"
        )

    if original_signature == "list":
        def patched(self, local_dir_path, dest_path):
            return ["sh", "-c", _build_cmd(local_dir_path, dest_path)]
    else:
        def patched(self, local_dir_path, dest_path):
            return _build_cmd(local_dir_path, dest_path)
    return patched


def patch_cloud_data_mover_skip_configs() -> None:
    """Shrink the inline-base64 data-mover payload on Lepton *and* DGXCloud.

    Both executors ship the whole ``job_dir`` (including
    ``configs/executor.yaml`` — which re-serializes every env var, potentially
    hundreds of KiB once source chunks get injected) to remote storage via a
    helper pod whose ``command`` is ``sh -c "echo <base64> | base64 -d > ..."``.
    That single argv string is bounded by the kernel's ``MAX_ARG_STRLEN``
    (128 KiB), so a moderately large ``configs/`` causes ``exec: argument list
    too long``.

    The pod-side launch script never reads ``configs/``; only
    ``launch_script.sh`` matters. Excluding ``configs/`` from the tarball
    keeps the data-mover command small while preserving correctness.
    """
    try:
        from nemo_run.core.execution import lepton as lep_mod
    except Exception:
        lep_mod = None  # type: ignore[assignment]

    try:
        from nemo_run.core.execution import dgxcloud as dgx_mod
    except Exception:
        dgx_mod = None  # type: ignore[assignment]

    if lep_mod and not getattr(lep_mod.LeptonExecutor, "_nemotron_data_mover_patched", False):
        lep_mod.LeptonExecutor.copy_directory_data_command = (
            _make_configs_excluding_copy_fn("list")
        )
        lep_mod.LeptonExecutor._nemotron_data_mover_patched = True

    if dgx_mod and not getattr(dgx_mod.DGXCloudExecutor, "_nemotron_data_mover_patched", False):
        dgx_mod.DGXCloudExecutor.copy_directory_data_command = (
            _make_configs_excluding_copy_fn("str")
        )
        dgx_mod.DGXCloudExecutor._nemotron_data_mover_patched = True


