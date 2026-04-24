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

    ``original_signature`` selects the return shape: ``"list"`` for Lepton
    (``["sh", "-c", cmd]``) and ``"str"`` for DGXCloud (command body only).
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


def patch_dgxcloud_accept_legacy_kwargs() -> None:
    """Map legacy ``app_id``/``app_secret`` → ``client_id``/``client_secret``.

    Pre-PR#480 nemo-run named the auth fields ``app_id``/``app_secret``; the
    current fork uses ``client_id``/``client_secret``. Cached fiddle Configs
    with the old names otherwise produce ``TypeError: unexpected keyword
    argument 'app_id'`` noise on every status poll.
    """
    try:
        from nemo_run.core.execution import dgxcloud as dgx_mod
    except Exception:
        return

    cls = dgx_mod.DGXCloudExecutor
    if getattr(cls, "_nemotron_legacy_kwargs_patched", False):
        return

    orig_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        if "app_id" in kwargs:
            kwargs.setdefault("client_id", kwargs.pop("app_id"))
        if "app_secret" in kwargs:
            kwargs.setdefault("client_secret", kwargs.pop("app_secret"))
        return orig_init(self, *args, **kwargs)

    cls.__init__ = patched_init
    cls._nemotron_legacy_kwargs_patched = True


def patch_dgxcloud_strip_source_chunks_from_exports() -> None:
    """Keep ``_NEMOTRON_SRC_CHUNK_*`` out of DGXCloud's ``torchrun_job.sh``.

    ``DGXCloudRequest.materialize()`` normally bakes every env var as an
    ``export KEY=VAL`` line. With ~400 KiB of source chunks, that file
    blows up and ``move_data`` chunks it into dozens of 10 KiB workloads
    (run:AI's Args cap) — a ~12 min submission.

    Chunks are already delivered via the Job spec's ``environmentVariables``
    field, so we strip them from the export block. Net: one data-mover
    workload, ~1 s submission; pod still sees the vars via ``os.environ``.
    """
    try:
        from nemo_run.core.execution import dgxcloud as dgx_mod
    except Exception:
        return

    if getattr(dgx_mod.DGXCloudRequest, "_nemotron_exports_patched", False):
        return

    orig_materialize = dgx_mod.DGXCloudRequest.materialize

    def materialize(self):
        # Snapshot + filter env_vars *before* materialize() reads them. Both
        # fields are dataclass attrs, so we can swap and restore.
        saved_exec_env = self.executor.env_vars
        saved_extra_env = self.extra_env
        self.executor.env_vars = {
            k: v for k, v in saved_exec_env.items() if not k.startswith("_NEMOTRON_SRC_")
        }
        self.extra_env = {
            k: v for k, v in saved_extra_env.items() if not k.startswith("_NEMOTRON_SRC_")
        }
        try:
            return orig_materialize(self)
        finally:
            self.executor.env_vars = saved_exec_env
            self.extra_env = saved_extra_env

    dgx_mod.DGXCloudRequest.materialize = materialize
    dgx_mod.DGXCloudRequest._nemotron_exports_patched = True


def patch_cloud_data_mover_skip_configs() -> None:
    """Exclude ``configs/`` from Lepton's and DGXCloud's data-mover tarball.

    Both executors ship ``job_dir`` via a helper pod whose command is
    ``sh -c "echo <base64> | base64 -d > …"`` — a single argv bounded by
    ``MAX_ARG_STRLEN`` (128 KiB). ``configs/executor.yaml`` re-serializes
    every env var (hundreds of KiB with source chunks), which blows past
    that limit as ``exec: argument list too long``.

    The pod-side launch script reads only ``launch_script.sh``, so dropping
    ``configs/`` keeps the command small without affecting correctness.
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


