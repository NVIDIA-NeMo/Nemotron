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

Handles converting Docker images to squash files on remote clusters
using enroot. Uses deterministic naming to avoid re-squashing existing images.
"""

from __future__ import annotations

import re
from typing import Any

from rich.console import Console

console = Console()


def container_to_sqsh_name(container: str) -> str:
    """Convert container image name to deterministic squash filename.

    Replaces any characters that can't be used in filenames with underscores.

    Args:
        container: Docker image name (e.g., "nvcr.io/nvidian/nemo:25.11-nano-v3.rc2")

    Returns:
        Safe squash filename (e.g., "nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh")

    Examples:
        >>> container_to_sqsh_name("nvcr.io/nvidian/nemo:25.11-nano-v3.rc2")
        'nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh'
        >>> container_to_sqsh_name("rayproject/ray:nightly-extra-py312-cpu")
        'rayproject_ray_nightly_extra_py312_cpu.sqsh'
    """
    # Replace any non-alphanumeric characters (except underscore) with underscore
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", container)
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

    Creates a deterministic filename based on the container image name.
    For example: nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 -> nemo-25.11-nano-v3.rc2.sqsh

    Args:
        container_image: Docker container image (e.g., nvcr.io/nvidian/nemo:25.11-nano-v3.rc2)
        remote_job_dir: Remote directory for squashed images

    Returns:
        Full path to squashed image file
    """
    # Extract image name and tag for readable filename
    # nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 -> nemo:25.11-nano-v3.rc2
    image_name = container_image.split("/")[-1]
    # nemo:25.11-nano-v3.rc2 -> nemo-25.11-nano-v3.rc2.sqsh
    sqsh_name = image_name.replace(":", "-") + ".sqsh"

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
        container_image: Docker container image to squash
        remote_job_dir: Remote directory for squashed images
        env_config: Environment config with slurm settings (account, partition, time)
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
    # (login nodes don't have enough memory for enroot import)
    account = env_config.get("account")
    partition = env_config.get("run_partition") or env_config.get("partition")
    time_limit = env_config.get("time", "04:00:00")
    gpus_per_node = env_config.get("gpus_per_node")

    salloc_args = []
    if account:
        salloc_args.append(f"--account={account}")
    if partition:
        salloc_args.append(f"--partition={partition}")
    salloc_args.append("--nodes=1")
    salloc_args.append("--ntasks-per-node=1")
    if gpus_per_node:
        salloc_args.append(f"--gpus-per-node={gpus_per_node}")
    salloc_args.append(f"--time={time_limit}")

    # Set up writable enroot paths (default /raid/enroot may not be user-writable)
    enroot_runtime = f"{remote_job_dir}/.enroot"
    enroot_env = (
        f"export ENROOT_RUNTIME_PATH={enroot_runtime} "
        f"ENROOT_CACHE_PATH={enroot_runtime}/cache "
        f"ENROOT_DATA_PATH={enroot_runtime}/data && "
        f"mkdir -p {enroot_runtime}/cache {enroot_runtime}/data && "
    )
    enroot_cmd = f"{enroot_env}enroot import --output {sqsh_path} docker://{container_image}"
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
