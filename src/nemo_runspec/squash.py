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

"""Container squash utilities for Slurm execution.

Handles converting container images or archives to squash files on remote
clusters using enroot. Uses deterministic naming to avoid re-squashing
existing images.

Also hosts the small precedence helpers that build-context Slurm
submissions share (squash + omni3 build dispatcher + family-specific
build.py wrappers). Precedence:

    partition:  build_partition  >  run_partition  >  partition
    time:       build_time       >  time                      (default fallback)
    image:      build_image      >  caller's default

Prefer these helpers over re-implementing the precedence inline — see the
``refactor(runspec): support archive container schemes`` commit for
context. For nemo-run SlurmExecutor kwargs (different shape from
salloc argv), use the three resolve_* helpers individually.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Precedence helpers for build-context Slurm submissions
# ---------------------------------------------------------------------------


def resolve_build_partition(env_config: Mapping[str, Any] | None) -> str | None:
    """Select the Slurm partition for a container-build / enroot-import job.

    Precedence: ``build_partition`` > ``run_partition`` > ``partition``.
    Returns ``None`` when no partition is set (caller decides the default).
    """
    if env_config is None:
        return None
    return (
        env_config.get("build_partition")
        or env_config.get("run_partition")
        or env_config.get("partition")
    )


def resolve_build_time(env_config: Mapping[str, Any] | None, default: str = "04:00:00") -> str:
    """Select the Slurm walltime for a container-build job.

    Precedence: ``build_time`` > ``time`` > caller-provided ``default``.
    """
    if env_config is None:
        return default
    return env_config.get("build_time") or env_config.get("time", default)


def resolve_build_image(env_config: Mapping[str, Any] | None, default: str) -> str:
    """Select the container image used by the build job itself.

    Precedence: ``build_image`` > caller-provided ``default``. Empty/None
    values in env_config fall through to the default.
    """
    if env_config is None:
        return default
    return env_config.get("build_image") or default


def build_salloc_args(
    env_config: Mapping[str, Any] | None,
    *,
    default_time: str = "04:00:00",
    include_gpus: bool = True,
) -> list[str]:
    """Build the salloc argv list for a single-node build/enroot-import job.

    Canonical shape shared by ``ensure_squashed_image`` and ``kit squash``.
    For nemo-run SlurmExecutor constructions (different kwargs shape), call
    the three resolve_* helpers individually.
    """
    env_config = env_config or {}
    account = env_config.get("account")
    partition = resolve_build_partition(env_config)
    time_limit = resolve_build_time(env_config, default_time)
    gpus_per_node = env_config.get("gpus_per_node") if include_gpus else None

    args: list[str] = []
    if account:
        args.append(f"--account={account}")
    if partition:
        args.append(f"--partition={partition}")
    args.extend(["--nodes=1", "--ntasks-per-node=1"])
    if gpus_per_node:
        args.append(f"--gpus-per-node={gpus_per_node}")
    args.append(f"--time={time_limit}")
    return args

_SUPPORTED_SCHEMES = (
    "docker://",
    "dockerd://",
    "podman://",
    "docker-archive://",
    "oci-archive://",
)
_ARCHIVE_SCHEMES = ("docker-archive://", "oci-archive://")
_DEFAULT_SCHEME = "docker://"


def normalize_container_source(container: str) -> str:
    """Normalize a container reference to an enroot-compatible URI."""
    if container.startswith(_SUPPORTED_SCHEMES):
        return container
    return f"{_DEFAULT_SCHEME}{container}"


def _split_container_source(container: str) -> tuple[str | None, str]:
    """Split a normalized container URI into scheme and payload."""
    for scheme in _SUPPORTED_SCHEMES:
        if container.startswith(scheme):
            return scheme, container.removeprefix(scheme)
    return None, container


def container_to_sqsh_name(container: str) -> str:
    """Convert a container reference to a deterministic squash filename.

    Replaces any characters that can't be used in filenames with underscores.
    Archive URIs use only the archive basename so remote paths do not leak into
    the sqsh filename.

    Args:
        container: Container image or archive URI.

    Returns:
        Safe squash filename (e.g., "nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh")

    Examples:
        >>> container_to_sqsh_name("nvcr.io/nvidian/nemo:25.11-nano-v3.rc2")
        'nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh'
        >>> container_to_sqsh_name("rayproject/ray:nightly-extra-py312-cpu")
        'rayproject_ray_nightly_extra_py312_cpu.sqsh'
    """
    normalized = normalize_container_source(container)
    scheme, payload = _split_container_source(normalized)
    if scheme in _ARCHIVE_SCHEMES:
        payload = payload.rsplit("/", 1)[-1]

    # Replace any non-alphanumeric characters (except underscore) with underscore
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", payload)
    # Collapse multiple underscores into one
    safe_name = re.sub(r"_+", "_", safe_name)
    # Strip leading/trailing underscores
    safe_name = safe_name.strip("_")
    return f"{safe_name}.sqsh"


def check_sqsh_exists(tunnel: Any, remote_path: str) -> bool:
    """Check if a squash file exists on the remote cluster.

    Args:
        tunnel: nemo-run SSHTunnel instance
        remote_path: Full path to the squash file

    Returns:
        True if file exists, False otherwise
    """
    result = tunnel.run(f"test -f {remote_path} && echo exists", hide=True, warn=True)
    return result.ok and "exists" in result.stdout


def get_squash_path(container_image: str, remote_job_dir: str) -> str:
    """Get the path to the squashed container image.

    Creates a deterministic filename based on the container reference.
    For example: nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 ->
    nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh

    Args:
        container_image: Container image or archive URI.
        remote_job_dir: Remote directory for squashed images

    Returns:
        Full path to squashed image file
    """
    sqsh_name = container_to_sqsh_name(container_image)
    return f"{remote_job_dir}/{sqsh_name}"


def ensure_squashed_image(
    tunnel: Any,
    container_image: str,
    remote_job_dir: str,
    env_config: dict,
    *,
    force: bool = False,
) -> str:
    """Ensure the container image is squashed on the remote cluster.

    Checks if a squashed version exists, and if not, creates it using enroot
    on a compute node via salloc.

    Args:
        tunnel: SSHTunnel instance (already connected)
        container_image: Container image or archive URI to squash
        remote_job_dir: Remote directory for squashed images
        env_config: Environment config with Slurm settings
        force: If True, re-squash even if file already exists

    Returns:
        Path to the squashed image file
    """
    sqsh_path = get_squash_path(container_image, remote_job_dir)

    # Check if squashed image already exists (unless force is set)
    if not force:
        with console.status("[bold blue]Checking for squashed image..."):
            result = tunnel.run(f"test -f {sqsh_path} && echo exists", hide=True, warn=True)

        if result.ok and "exists" in result.stdout:
            console.print(
                f"[green]✓[/green] Using existing squashed image: [cyan]{sqsh_path}[/cyan]"
            )
            return sqsh_path

    # Need to create the squashed image
    if force:
        console.print("[yellow]![/yellow] Force re-squash requested, removing existing file...")
        tunnel.run(f"rm -f {sqsh_path}", hide=True)
    else:
        console.print("[yellow]![/yellow] Squashed image not found, creating...")
    console.print(f"  [dim]Image:[/dim] {container_image}")
    console.print(f"  [dim]Output:[/dim] {sqsh_path}")
    console.print()

    # Ensure directory exists
    tunnel.run(f"mkdir -p {remote_job_dir}", hide=True)

    # Build salloc command to run enroot import on a compute node
    # (login nodes don't have enough memory for enroot import).
    # build_salloc_args applies the canonical build-context precedence
    # (build_partition > run_partition > partition; build_time > time).
    salloc_args = build_salloc_args(env_config)

    # Set up writable enroot paths (default /raid/enroot may not be user-writable)
    enroot_runtime = f"{remote_job_dir}/.enroot"
    enroot_env = (
        f"export ENROOT_RUNTIME_PATH={enroot_runtime} "
        f"ENROOT_CACHE_PATH={enroot_runtime}/cache "
        f"ENROOT_DATA_PATH={enroot_runtime}/data && "
        f"mkdir -p {enroot_runtime}/cache {enroot_runtime}/data && "
    )
    container_source = normalize_container_source(container_image)
    enroot_cmd = f"{enroot_env}enroot import --output {sqsh_path} {container_source}"
    cmd = f"salloc {' '.join(salloc_args)} srun --export=ALL bash -c '{enroot_cmd}'"

    # Run enroot import via salloc (this can take a while)
    console.print(
        "[bold blue]Allocating compute node and importing container "
        "(this may take several minutes)...[/bold blue]"
    )
    console.print(f"[dim]$ {cmd}[/dim]")
    console.print()
    result = tunnel.run(cmd, hide=False, warn=True)

    if not result.ok:
        raise RuntimeError(
            f"Failed to squash container image.\n"
            f"Command: {cmd}\n"
            f"Error: {result.stderr or 'Unknown error'}"
        )

    console.print(f"[green]✓[/green] Created squashed image: [cyan]{sqsh_path}[/cyan]")
    return sqsh_path
