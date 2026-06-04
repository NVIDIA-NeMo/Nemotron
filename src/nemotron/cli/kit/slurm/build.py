# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Build command - build a recipe stage's Dockerfile into a .sqsh on a Slurm cluster.

The build counterpart of ``nemotron kit slurm squash``: ``squash`` imports an
existing image into a squash file; ``build`` adds a ``podman build`` of a
recipe-owned Dockerfile in front of the same enroot-import-on-a-compute-node
path. Slurm-only and explicit about it (that is why it lives under
``kit slurm``); driven by an ``env.toml`` profile via nemo_runspec.

The build logic itself lives in the shared, recipe-agnostic
``build_container.sh`` next to this file; this command only resolves inputs from
the profile + stage registry and submits it through the SSH tunnel.

Usage:
    nemotron kit slurm build dlw --recipe ultra3 --stage pretrain
    nemotron kit slurm build dlw --dockerfile <path> --output my.sqsh
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemo_runspec.env import load_env_profile
from nemo_runspec.squash import build_salloc_args, check_sqsh_exists

console = Console()

# Build-time podman/enroot container (pyxis launches it on the compute node).
PODMAN_IMAGE = "docker://quay.io#podman/stable:v5.3"

# Stage registry — consolidates the per-recipe STAGES dicts the retired
# ultra3/omni3 build.py dispatchers used to own. Maps recipe -> stage alias ->
# (stage_dir, build-time image tag, output .sqsh basename).
RECIPES: dict[str, dict[str, tuple[str, str, str]]] = {
    "ultra3": {
        "pretrain": ("stage0_pretrain", "nemotron/ultra3-pretrain:latest", "ultra3-pretrain.sqsh"),
        "sft": ("stage1_sft", "nemotron/ultra3-sft:latest", "ultra3-sft.sqsh"),
    },
    "omni3": {
        "sft": ("stage0_sft", "nemotron/omni3-sft:latest", "omni3-sft.sqsh"),
        "rl": ("stage1_rl", "nemotron/omni3-rl:latest", "omni3-rl.sqsh"),
    },
}


def build(
    profile: str = typer.Argument(..., help="Env profile name from env.toml (e.g., 'dlw')."),
    recipe: str | None = typer.Option(None, "--recipe", help="Recipe to build (e.g. 'ultra3', 'omni3')."),
    stage: str | None = typer.Option(None, "--stage", help="Stage alias within the recipe (e.g. 'pretrain', 'sft')."),
    dockerfile: str | None = typer.Option(None, "--dockerfile", help="Generic: remote abs path to a Dockerfile (overrides --recipe/--stage)."),
    output: str | None = typer.Option(None, "--output", help="Generic: output .sqsh basename (with --dockerfile)."),
    repo_root: str | None = typer.Option(None, "--repo-root", help="Remote path to the checked-out repo on the cluster. Default: <remote_job_dir>/Nemotron."),
    build_arg: list[str] = typer.Option([], "--build-arg", help="Docker build-arg, repeatable (e.g. MEGATRON_BRIDGE_BRANCH=...)."),
    dry_run: bool = typer.Option(False, "-d", "--dry-run", help="Show the resolved salloc/srun command without executing."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if the .sqsh already exists."),
) -> None:
    """Build a recipe stage's Dockerfile into an enroot .sqsh on a Slurm cluster.

    Examples:
        nemotron kit slurm build dlw --recipe ultra3 --stage pretrain \\
            --build-arg MEGATRON_BRIDGE_BRANCH=<branch> --build-arg MEGATRON_CORE_BRANCH=<branch>
        nemotron kit slurm build dlw --dockerfile /repo/.../Dockerfile --output ultra3-pretrain.sqsh
    """
    try:
        env_config = load_env_profile(profile)
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise typer.Exit(1)

    host = env_config.get("host")
    user = env_config.get("user")
    remote_job_dir = env_config.get("remote_job_dir")
    if not host or not user:
        console.print(f"[red bold]Error:[/red bold] Profile '{profile}' missing host or user for SSH")
        raise typer.Exit(1)
    if not remote_job_dir:
        console.print(f"[red bold]Error:[/red bold] Profile '{profile}' missing remote_job_dir")
        raise typer.Exit(1)

    # Remote repo checkout (holds the Dockerfile, build context, and the shared
    # build_container.sh). Like `squash`, this command does not transport code;
    # clone or rsync the repo onto a cluster-visible filesystem first.
    repo = repo_root or str(PurePosixPath(remote_job_dir) / "Nemotron")
    inner = str(PurePosixPath(repo) / "src/nemotron/cli/kit/slurm/build_container.sh")

    # Resolve build inputs: explicit --dockerfile/--output, else --recipe/--stage.
    if dockerfile:
        df = dockerfile
        context = str(PurePosixPath(dockerfile).parent)
        sqsh_name = output or (PurePosixPath(dockerfile).parent.name + ".sqsh")
        image_tag = f"nemotron/{PurePosixPath(sqsh_name).stem}:latest"
        manifest_key = PurePosixPath(sqsh_name).stem
    else:
        if not recipe or not stage:
            console.print("[red bold]Error:[/red bold] provide --recipe and --stage, or --dockerfile/--output.")
            console.print("\nKnown recipes/stages:")
            for r, stages in RECIPES.items():
                console.print(f"  {r}: {', '.join(stages)}")
            raise typer.Exit(1)
        if recipe not in RECIPES or stage not in RECIPES[recipe]:
            console.print(f"[red bold]Error:[/red bold] unknown recipe/stage '{recipe}/{stage}'.")
            raise typer.Exit(1)
        stage_dir, image_tag, sqsh_name = RECIPES[recipe][stage]
        df = str(PurePosixPath(repo) / f"src/nemotron/recipes/{recipe}/{stage_dir}/Dockerfile")
        context = str(PurePosixPath(df).parent)
        manifest_key = f"{recipe}-{stage_dir}"

    # Output + cache layout on the cluster.
    build_cache_dir = env_config.get("build_cache_dir") or str(PurePosixPath(remote_job_dir) / "nemotron-cache")
    containers_dir = str(PurePosixPath(build_cache_dir) / "containers")
    sqsh = str(PurePosixPath(containers_dir) / sqsh_name)
    manifest = str(PurePosixPath(containers_dir) / "manifest.yaml")
    build_args = " ".join(f"--build-arg {a}" for a in build_arg)

    # Assemble the remote command: salloc a compute node, then srun a pyxis
    # podman/stable container with the repo + cache mounted, running the shared
    # inner build script with explicit inputs.
    inner_env = (
        f"DOCKERFILE={shlex.quote(df)} "
        f"CONTEXT={shlex.quote(context)} "
        f"IMAGE_TAG={shlex.quote(image_tag)} "
        f"SQSH={shlex.quote(sqsh)} "
        f"BUILD_CACHE_DIR={shlex.quote(build_cache_dir)} "
        f"MANIFEST={shlex.quote(manifest)} "
        f"MANIFEST_KEY={shlex.quote(manifest_key)} "
        f"BUILD_ARGS={shlex.quote(build_args)}"
    )
    container_cmd = f"{inner_env} bash {shlex.quote(inner)}"
    srun = (
        f"srun --export=ALL --container-image={PODMAN_IMAGE} "
        f"--container-mounts={repo}:{repo},{build_cache_dir}:{build_cache_dir} "
        f"--no-container-mount-home bash -lc {shlex.quote(container_cmd)}"
    )
    salloc_args = build_salloc_args(env_config)
    cmd = f"salloc {' '.join(salloc_args)} {srun}"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Profile", f"[cyan]{profile}[/cyan]")
    table.add_row("Host", f"{user}@{host}")
    table.add_row("Repo (remote)", repo)
    table.add_row("Dockerfile", df)
    table.add_row("Image tag", image_tag)
    table.add_row("Output", sqsh)
    table.add_row("Manifest", f"{manifest} (key: {manifest_key})")
    table.add_row("Build args", build_args or "<none>")
    console.print(Panel(table, title="[bold]Build Configuration[/bold]", expand=False))
    console.print()

    if dry_run:
        console.print("[yellow]Dry-run mode - no changes will be made[/yellow]")
        console.print(f"Would run on {host}:")
        console.print(f"  [dim]{cmd}[/dim]")
        return

    try:
        import nemo_run as run
    except ImportError:
        console.print("[red bold]Error:[/red bold] nemo-run is required for build")
        console.print("Install with: pip install nemo-run")
        raise typer.Exit(1)

    with console.status("[bold blue]Connecting to cluster..."):
        tunnel = run.SSHTunnel(host=host, user=user, job_dir=remote_job_dir)
        tunnel.connect()
    console.print("[green]Connected![/green]")
    console.print()

    if not force and check_sqsh_exists(tunnel, sqsh):
        console.print(f"[yellow]Squash file already exists:[/yellow] {sqsh}")
        console.print("[dim]Skipping build. Use --force to rebuild.[/dim]")
        tunnel.cleanup()
        return

    tunnel.run(f"mkdir -p {containers_dir}", hide=True)
    if force:
        tunnel.run(f"rm -f {sqsh}", hide=True)

    console.print("[bold]Allocating compute node and building container...[/bold]")
    console.print(f"  {df}")
    console.print(f"  -> {sqsh}")
    console.print(f"[dim]$ {cmd}[/dim]")
    console.print("[dim]This may take many minutes...[/dim]")
    console.print()

    result = tunnel.run(cmd, hide=False, warn=True)
    tunnel.cleanup()

    if result.ok:
        console.print(
            Panel(
                f"[green]Built and imported:[/green]\n{sqsh}\n\n"
                f"[dim]run with: ... run.env.container_image={sqsh}[/dim]",
                title="[bold green]Complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                f"[red]Build failed[/red]\n{result.stderr or 'Unknown error'}",
                title="[bold red]Error[/bold red]",
                border_style="red",
                expand=False,
            )
        )
        raise typer.Exit(1)
