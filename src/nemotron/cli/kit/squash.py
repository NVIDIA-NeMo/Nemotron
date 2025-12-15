"""Squash command - convert Docker images to squash files on remote clusters.

Usage:
    nemotron kit squash dlw nvcr.io/nvidian/nemo:25.11-nano-v3.rc2
    nemotron kit squash dlw --all  # squash all containers from config
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemotron.kit.cli.env import load_env_profile
from nemotron.kit.cli.squash import check_sqsh_exists, container_to_sqsh_name

console = Console()


def squash(
    profile: str = typer.Argument(
        ...,
        help="Env profile name from env.toml (e.g., 'dlw')",
    ),
    container: Optional[str] = typer.Argument(
        None,
        help="Docker image to squash (e.g., 'nvcr.io/nvidian/nemo:25.11-nano-v3.rc2')",
    ),
    dry_run: bool = typer.Option(
        False,
        "-d",
        "--dry-run",
        help="Show what would be done without executing",
    ),
) -> None:
    """Convert Docker images to squash files on remote cluster.

    Connects to the cluster via SSH and uses enroot to import Docker images
    as squash files. Uses deterministic naming so existing images are skipped.

    Examples:
        nemotron kit squash dlw nvcr.io/nvidian/nemo:25.11-nano-v3.rc2
        nemotron kit squash dlw rayproject/ray:nightly-extra-py312-cpu
        nemotron kit squash dlw nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 --dry-run
    """
    # Load env profile
    try:
        env_config = load_env_profile(profile)
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise typer.Exit(1)

    # Validate profile has required fields
    host = env_config.get("host")
    user = env_config.get("user")
    remote_job_dir = env_config.get("remote_job_dir")

    if not host or not user:
        console.print(
            f"[red bold]Error:[/red bold] Profile '{profile}' missing host or user for SSH"
        )
        raise typer.Exit(1)

    if not remote_job_dir:
        console.print(
            f"[red bold]Error:[/red bold] Profile '{profile}' missing remote_job_dir"
        )
        raise typer.Exit(1)

    if not container:
        console.print("[red bold]Error:[/red bold] Container image is required")
        console.print("\nUsage: nemotron kit squash <profile> <container>")
        console.print("Example: nemotron kit squash dlw nvcr.io/nvidian/nemo:25.11-nano-v3.rc2")
        raise typer.Exit(1)

    # Generate squash filename
    sqsh_name = container_to_sqsh_name(container)
    remote_path = f"{remote_job_dir}/{sqsh_name}"

    # Show configuration
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Profile", f"[cyan]{profile}[/cyan]")
    table.add_row("Host", f"{user}@{host}")
    table.add_row("Container", container)
    table.add_row("Output", remote_path)
    console.print(Panel(table, title="[bold]Squash Configuration[/bold]", expand=False))
    console.print()

    if dry_run:
        console.print("[yellow]Dry-run mode - no changes will be made[/yellow]")
        console.print()
        console.print(f"Would run on {host}:")
        console.print(f"  enroot import --output {remote_path} docker://{container}")
        return

    # Connect to cluster
    try:
        import nemo_run as run
    except ImportError:
        console.print("[red bold]Error:[/red bold] nemo-run is required for squash")
        console.print("Install with: pip install nemo-run")
        raise typer.Exit(1)

    with console.status("[bold blue]Connecting to cluster..."):
        tunnel = run.SSHTunnel(
            host=host,
            user=user,
            job_dir=remote_job_dir,
        )
        tunnel.connect()

    console.print("[green]Connected![/green]")
    console.print()

    # Check if already exists
    if check_sqsh_exists(tunnel, remote_path):
        console.print(f"[yellow]Squash file already exists:[/yellow] {remote_path}")
        console.print("[dim]Skipping import.[/dim]")
        tunnel.cleanup()
        return

    # Ensure remote directory exists
    with console.status("[bold blue]Creating remote directory..."):
        tunnel.run(f"mkdir -p {remote_job_dir}", hide=True)

    # Run enroot import
    console.print(f"[bold]Importing container...[/bold]")
    console.print(f"  {container}")
    console.print(f"  -> {remote_path}")
    console.print()
    console.print("[dim]This may take several minutes...[/dim]")
    console.print()

    cmd = f"enroot import --output {remote_path} docker://{container}"
    result = tunnel.run(cmd, hide=False, warn=True)

    tunnel.cleanup()

    if result.ok:
        console.print()
        console.print(Panel(
            f"[green]Successfully imported:[/green]\n{remote_path}",
            title="[bold green]Complete[/bold green]",
            border_style="green",
            expand=False,
        ))
    else:
        console.print()
        console.print(Panel(
            f"[red]Failed to import container[/red]\n{result.stderr or 'Unknown error'}",
            title="[bold red]Error[/bold red]",
            border_style="red",
            expand=False,
        ))
        raise typer.Exit(1)
