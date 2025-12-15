# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Setup Slurm cluster by importing Docker images to squash files.

Usage:
    uv run -m nemotron.kit.setup_slurm_cluster dlw

This script:
1. Reads a cluster profile from run.toml
2. Connects to the cluster via SSH tunnel
3. Imports Docker images to squash files using enroot
4. Updates run.toml with new profile entries
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Images to import: squash filename -> Docker image
IMAGES = {
    "ray.sqsh": "rayproject/ray:nightly-extra-py312-cpu",
    "nano-3.sqsh": "nvcr.io/nvidian/nemo:25.11-nano-v3.rc1",
    "nano-3-rl.sqsh": "nvcr.io/nvidian/nemo-rl:25.11-nano-v3.rc1",
}

console = Console()


@dataclass
class ImportResult:
    """Result of an image import operation."""

    sqsh_name: str
    image: str
    status: str  # "imported", "skipped", "failed"
    message: str | None = None


def find_run_config() -> Path | None:
    """Find run.toml in cwd or walking up to project root."""
    for path in [Path.cwd(), *Path.cwd().parents]:
        run_file = path / "run.toml"
        if run_file.exists():
            return run_file
        if (path / "pyproject.toml").exists():
            break
    return None


def check_file_exists(tunnel: Any, remote_path: str) -> bool:
    """Check if a file exists on the remote cluster."""
    result = tunnel.run(f"test -f {remote_path} && echo exists", hide=True, warn=True)
    return result.ok and "exists" in result.stdout


def import_image(
    tunnel: Any,
    sqsh_name: str,
    image: str,
    remote_job_dir: str,
) -> ImportResult:
    """Import a Docker image to squash file on remote cluster.

    Args:
        tunnel: SSHTunnel instance.
        sqsh_name: Output squash filename.
        image: Docker image to import.
        remote_job_dir: Remote directory for output.

    Returns:
        ImportResult with status and details.
    """
    remote_path = f"{remote_job_dir}/{sqsh_name}"

    # Check if already exists
    if check_file_exists(tunnel, remote_path):
        return ImportResult(
            sqsh_name=sqsh_name,
            image=image,
            status="skipped",
            message="Already exists",
        )

    # Run enroot import
    cmd = f"enroot import --output {remote_path} docker://{image}"
    result = tunnel.run(cmd, hide=False, warn=True)

    if result.ok:
        return ImportResult(
            sqsh_name=sqsh_name,
            image=image,
            status="imported",
        )
    else:
        return ImportResult(
            sqsh_name=sqsh_name,
            image=image,
            status="failed",
            message=result.stderr or "Unknown error",
        )


def update_run_toml(
    config_path: Path,
    cluster_name: str,
    remote_job_dir: str,
    cpu_partition: str,
    gpu_partition: str,
) -> list[str]:
    """Update run.toml with new profile entries.

    Args:
        config_path: Path to run.toml.
        cluster_name: Base cluster name (e.g., "dlw").
        remote_job_dir: Remote job directory path.
        cpu_partition: Partition for CPU/prep jobs.
        gpu_partition: Partition for GPU jobs.

    Returns:
        List of created profile names.
    """
    import tomlkit

    with open(config_path, "r") as f:
        doc = tomlkit.load(f)

    created_profiles = []

    # Define profiles to create
    profiles = {
        f"{cluster_name}-prep": {
            "extends": cluster_name,
            "partition": cpu_partition,
            "nodes": 1,
            "ntasks_per_node": 1,
            "gpus_per_node": 0,
            "container_image": f"{remote_job_dir}/ray.sqsh",
        },
        f"{cluster_name}-megatron": {
            "extends": cluster_name,
            "partition": gpu_partition,
            "nodes": 1,
            "ntasks_per_node": 8,
            "gpus_per_node": 8,
            "mem": "0",
            "exclusive": True,
            "container_image": f"{remote_job_dir}/nano-3.sqsh",
        },
        f"{cluster_name}-rl": {
            "extends": cluster_name,
            "partition": gpu_partition,
            "nodes": 1,
            "ntasks_per_node": 8,
            "gpus_per_node": 8,
            "mem": "0",
            "exclusive": True,
            "container_image": f"{remote_job_dir}/nano-3-rl.sqsh",
        },
    }

    for profile_name, settings in profiles.items():
        # Create or update profile
        if profile_name not in doc:
            doc.add(tomlkit.nl())
            doc.add(profile_name, tomlkit.table())

        for key, value in settings.items():
            doc[profile_name][key] = value

        created_profiles.append(profile_name)

    with open(config_path, "w") as f:
        tomlkit.dump(doc, f)

    return created_profiles


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup Slurm cluster by importing Docker images to squash files.",
        prog="nemotron.kit.setup_slurm_cluster",
    )
    parser.add_argument(
        "cluster",
        help="Cluster profile name from run.toml (e.g., 'dlw')",
    )
    parser.add_argument(
        "--cpu-partition",
        default=None,
        help="Partition for prep/CPU jobs (default: from profile or 'cpu_long')",
    )
    parser.add_argument(
        "--gpu-partition",
        default=None,
        help="Partition for GPU jobs (default: from profile or 'batch')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args(argv)

    # Find run.toml
    config_path = find_run_config()
    if config_path is None:
        console.print("[red bold]Error:[/red bold] No run.toml found")
        return 1

    console.print(f"Using config: [cyan]{config_path}[/cyan]")
    console.print()

    # Load cluster profile
    from nemotron.kit.run import load_run_profile

    try:
        profile = load_run_profile(args.cluster, config_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        return 1

    # Validate required settings
    if not profile.host or not profile.user:
        console.print(
            f"[red bold]Error:[/red bold] Profile '{args.cluster}' missing host or user for SSH"
        )
        return 1

    if not profile.remote_job_dir:
        console.print(
            f"[red bold]Error:[/red bold] Profile '{args.cluster}' missing remote_job_dir"
        )
        return 1

    # Determine partitions
    cpu_partition = args.cpu_partition or profile.partition or "cpu_long"
    gpu_partition = args.gpu_partition or profile.partition or "batch"

    # Print configuration
    table = Table(title="Cluster Configuration", show_header=False)
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Profile", f"[cyan]{args.cluster}[/cyan]")
    table.add_row("Host", profile.host)
    table.add_row("User", profile.user)
    table.add_row("Remote dir", profile.remote_job_dir)
    table.add_row("CPU partition", cpu_partition)
    table.add_row("GPU partition", gpu_partition)
    console.print(table)
    console.print()

    if args.dry_run:
        console.print("[yellow]Dry-run mode - no changes will be made[/yellow]")
        console.print()
        console.print("[bold]Would import images:[/bold]")
        for sqsh_name, image in IMAGES.items():
            console.print(f"  {sqsh_name} <- {image}")
        console.print()
        console.print("[bold]Would create profiles:[/bold]")
        console.print(f"  {args.cluster}-prep, {args.cluster}-megatron, {args.cluster}-rl")
        return 0

    # Connect to cluster
    from nemo_run import SSHTunnel

    tunnel = SSHTunnel(
        host=profile.host,
        user=profile.user,
        job_dir=profile.remote_job_dir,
        identity=profile.identity,
    )

    with console.status("[bold blue]Connecting to cluster..."):
        tunnel.connect()

    console.print("[green]Connected![/green]")
    console.print()

    # Ensure remote directory exists
    with console.status("[bold blue]Creating remote directory..."):
        tunnel.run(f"mkdir -p {profile.remote_job_dir}", hide=True)

    # Import images
    console.print("[bold]Importing images...[/bold]")
    console.print()

    results: list[ImportResult] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        for sqsh_name, image in IMAGES.items():
            task_id = progress.add_task(f"[cyan]{sqsh_name}[/cyan]", total=100)

            # Check if exists
            progress.update(task_id, description=f"[cyan]{sqsh_name}[/cyan] - checking")
            remote_path = f"{profile.remote_job_dir}/{sqsh_name}"

            if check_file_exists(tunnel, remote_path):
                results.append(
                    ImportResult(
                        sqsh_name=sqsh_name,
                        image=image,
                        status="skipped",
                        message="Already exists",
                    )
                )
                progress.update(
                    task_id,
                    description=f"[dim]{sqsh_name}[/dim] - [yellow]skipped[/yellow]",
                    completed=100,
                )
                continue

            # Import
            progress.update(task_id, description=f"[cyan]{sqsh_name}[/cyan] - importing")
            cmd = f"enroot import --output {remote_path} docker://{image}"
            result = tunnel.run(cmd, hide=False, warn=True)

            if result.ok:
                results.append(
                    ImportResult(
                        sqsh_name=sqsh_name,
                        image=image,
                        status="imported",
                    )
                )
                progress.update(
                    task_id,
                    description=f"[green]{sqsh_name}[/green] - [green]complete[/green]",
                    completed=100,
                )
            else:
                results.append(
                    ImportResult(
                        sqsh_name=sqsh_name,
                        image=image,
                        status="failed",
                        message=result.stderr or "Unknown error",
                    )
                )
                progress.update(
                    task_id,
                    description=f"[red]{sqsh_name}[/red] - [red]failed[/red]",
                    completed=100,
                )

    console.print()

    # Cleanup tunnel
    tunnel.cleanup()

    # Update run.toml
    with console.status("[bold blue]Updating run.toml..."):
        created_profiles = update_run_toml(
            config_path=config_path,
            cluster_name=args.cluster,
            remote_job_dir=profile.remote_job_dir,
            cpu_partition=cpu_partition,
            gpu_partition=gpu_partition,
        )

    # Summary
    console.print()
    imported = sum(1 for r in results if r.status == "imported")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")

    summary = Table(box=None, show_header=False, padding=(0, 2))
    summary.add_column("Key", style="dim")
    summary.add_column("Value")

    summary.add_row("Images imported", f"[green]{imported}[/green]")
    if skipped > 0:
        summary.add_row("Images skipped", f"[yellow]{skipped}[/yellow] (already exist)")
    if failed > 0:
        summary.add_row("Images failed", f"[red]{failed}[/red]")

    summary.add_row("Profiles created", f"[cyan]{', '.join(created_profiles)}[/cyan]")
    summary.add_row("Config updated", str(config_path))

    if failed > 0:
        console.print(
            Panel(summary, title="[bold yellow]Setup Complete (with errors)[/bold yellow]", border_style="yellow")
        )
        return 1
    else:
        console.print(
            Panel(summary, title="[bold green]Setup Complete[/bold green]", border_style="green")
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
