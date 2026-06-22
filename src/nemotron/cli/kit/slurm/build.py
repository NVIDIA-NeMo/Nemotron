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

The build counterpart of ``nemotron kit slurm squash``. It runs as one detached
Slurm job with two steps in a single allocation:

  1. podman build + ``podman push`` to the profile's ``build_registry`` (inside a
     pyxis ``podman/stable`` container);
  2. ``enroot import docker://<registry>#<image>`` on the HOST (no container) ->
     ``.sqsh`` + manifest.

Why the round-trip: a nested unprivileged ``enroot import`` inside a pyxis
container cannot create the user namespace it needs to unpack (Permission
denied), and the host has no podman and no archive-import support. Pushing to a
registry and importing ``docker://`` on the host is the proven path that
``kit squash`` already uses. Set ``build_registry`` per profile in ``env.toml``.

Code transport is just two small files (the recipe Dockerfile and the shared
``build_container.sh``) via ``tunnel.put`` — the Dockerfile clones everything
else from git, so the build context is self-contained.

Usage:
    nemotron kit slurm build dlw --recipe ultra3 --stage pretrain \
        --build-arg MEGATRON_BRIDGE_BRANCH=nemotron_3_ultra
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path, PurePosixPath

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemo_runspec.env import load_env_profile
from nemo_runspec.squash import resolve_build_cache_dir, resolve_build_partition, resolve_build_time

console = Console()

# Build-time podman container (pyxis launches it on the compute node).
PODMAN_IMAGE = "docker://quay.io#podman/stable:v5.3"

# Stage registry — consolidates the per-recipe STAGES dicts the retired
# ultra3/omni3 build.py dispatchers owned. recipe -> stage alias ->
# (stage_dir, image basename, output .sqsh basename).
RECIPES: dict[str, dict[str, tuple[str, str, str]]] = {
    "ultra3": {
        "pretrain": ("stage0_pretrain", "ultra3-pretrain", "ultra3-pretrain.sqsh"),
        "sft": ("stage1_sft", "ultra3-sft", "ultra3-sft.sqsh"),
    },
    "omni3": {
        "sft": ("stage0_sft", "omni3-sft", "omni3-sft.sqsh"),
        "rl": ("stage1_rl", "omni3-rl", "omni3-rl.sqsh"),
    },
}


def _local_repo_root() -> Path:
    # repo root is five levels up: .../<root>/src/nemotron/cli/kit/slurm/build.py
    return Path(__file__).resolve().parents[5]


def _render_sbatch(*, exports: dict[str, str], podman_image: str, mounts: str, inner: str,
                   containers_dir: str, sqsh: str, enroot_uri: str, manifest: str, manifest_key: str) -> str:
    """Render the orchestration sbatch body: container build+push -> host import -> manifest."""
    export_lines = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in exports.items())
    return f"""#!/bin/bash
set -euo pipefail
{export_lines}

echo "[kit-build] step 1/2: podman build + push (pyxis container)"
srun --export=ALL --mpi=none \\
    --container-image={shlex.quote(podman_image)} \\
    --container-mounts={shlex.quote(mounts)} \\
    --no-container-mount-home \\
    bash {shlex.quote(inner)}

echo "[kit-build] step 2/2: host enroot import {enroot_uri}"
mkdir -p {shlex.quote(containers_dir)}
rm -f {shlex.quote(sqsh)}
srun --export=ALL enroot import --output {shlex.quote(sqsh)} {shlex.quote(enroot_uri)}
ls -la {shlex.quote(sqsh)}

SHA256=$(sha256sum {shlex.quote(sqsh)} | awk '{{print $1}}')
BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
touch {shlex.quote(manifest)}
if grep -q "^{manifest_key}:" {shlex.quote(manifest)} 2>/dev/null; then
    sed -i "/^{manifest_key}:/,+3d" {shlex.quote(manifest)}
fi
cat >> {shlex.quote(manifest)} <<MANIFEST_EOF
{manifest_key}:
  ref: {sqsh}
  sha256: $SHA256
  built: $BUILT_AT
MANIFEST_EOF
echo "[kit-build] manifest updated: {manifest} (key: {manifest_key})"
echo "[kit-build] run with:  ... run.env.container_image={sqsh}"
echo KIT_BUILD_DONE
"""


def build(
    profile: str = typer.Argument(..., help="Env profile name from env.toml (e.g., 'dlw')."),
    recipe: str | None = typer.Option(None, "--recipe", help="Recipe to build (e.g. 'ultra3', 'omni3')."),
    stage: str | None = typer.Option(None, "--stage", help="Stage alias within the recipe (e.g. 'pretrain', 'sft')."),
    build_arg: list[str] = typer.Option([], "--build-arg", help="Docker build-arg, repeatable (e.g. MEGATRON_BRIDGE_BRANCH=nemotron_3_ultra)."),
    tag: str = typer.Option("latest", "--tag", help="Image tag to push/import."),
    dry_run: bool = typer.Option(False, "-d", "--dry-run", help="Show the resolved plan + sbatch without submitting."),
) -> None:
    """Build a recipe stage into an enroot .sqsh via registry round-trip on a Slurm cluster.

    Example:
        nemotron kit slurm build dlw --recipe ultra3 --stage pretrain \\
            --build-arg MEGATRON_BRIDGE_BRANCH=nemotron_3_ultra
    """
    try:
        env_config = load_env_profile(profile)
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise typer.Exit(1)

    host = env_config.get("host")
    user = env_config.get("user")
    remote_job_dir = env_config.get("remote_job_dir")
    registry = env_config.get("build_registry")
    if not host or not user or not remote_job_dir:
        console.print(f"[red bold]Error:[/red bold] Profile '{profile}' missing host/user/remote_job_dir.")
        raise typer.Exit(1)
    if not registry:
        console.print(
            f"[red bold]Error:[/red bold] Profile '{profile}' has no [bold]build_registry[/bold]. "
            "Set it in env.toml, e.g. build_registry = \"gitlab-master.nvidia.com:5005/<repo>\"."
        )
        raise typer.Exit(1)

    if not recipe or not stage or recipe not in RECIPES or stage not in RECIPES.get(recipe, {}):
        console.print("[red bold]Error:[/red bold] provide a known --recipe and --stage.")
        for r, stages in RECIPES.items():
            console.print(f"  {r}: {', '.join(stages)}")
        raise typer.Exit(1)

    stage_dir, image_basename, sqsh_name = RECIPES[recipe][stage]
    manifest_key = f"{recipe}-{stage_dir}"
    local_root = _local_repo_root()
    local_dockerfile = local_root / f"src/nemotron/recipes/{recipe}/{stage_dir}/Dockerfile"
    local_inner = local_root / "src/nemotron/cli/kit/slurm/build_container.sh"
    if not local_dockerfile.is_file():
        console.print(f"[red bold]Error:[/red bold] Dockerfile not found locally: {local_dockerfile}")
        raise typer.Exit(1)

    # Registry refs: podman push uses "<registry>/<image>:<tag>"; enroot import
    # uses "docker://<host>#<repo>/<image>:<tag>". Docker/podman repository names
    # must be lowercase (GitLab lowercases registry paths too).
    registry = registry.lower()
    image_basename = image_basename.lower()
    registry_host, _, registry_repo = registry.partition("/")
    image_ref = f"{registry}/{image_basename}:{tag}"
    enroot_uri = f"docker://{registry_host}#{registry_repo}/{image_basename}:{tag}"

    build_cache_dir = str(resolve_build_cache_dir(env_config, Path(remote_job_dir) / "nemotron-cache"))
    containers_dir = str(PurePosixPath(build_cache_dir) / "containers")
    sqsh = str(PurePosixPath(containers_dir) / sqsh_name)
    manifest = str(PurePosixPath(containers_dir) / "manifest.yaml")
    staging = str(PurePosixPath(remote_job_dir) / "kit-build" / manifest_key)
    inner = str(PurePosixPath(staging) / "build_container.sh")
    df_remote = str(PurePosixPath(staging) / "Dockerfile")
    build_args = " ".join(f"--build-arg {a}" for a in build_arg)
    partition = resolve_build_partition(env_config)
    walltime = resolve_build_time(env_config, "04:00:00")

    exports = {
        "DOCKERFILE": df_remote,
        "CONTEXT": staging,
        "IMAGE_REF": image_ref,
        "BUILD_ARGS": build_args,
    }

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Profile", f"[cyan]{profile}[/cyan]")
    table.add_row("Host", f"{user}@{host}")
    table.add_row("Registry push", image_ref)
    table.add_row("Host import", enroot_uri)
    table.add_row("Output", f"{sqsh}  (manifest key: {manifest_key})")
    table.add_row("Partition / time", f"{partition} / {walltime}")
    table.add_row("Build args", build_args or "<none>")
    console.print(Panel(table, title="[bold]Build Configuration[/bold]", expand=False))
    console.print()

    if dry_run:
        mounts = f"{staging}:{staging},<auth.json bridged from enroot creds>"
        console.print("[yellow]Dry-run — sbatch body that would be submitted:[/yellow]")
        console.print(_render_sbatch(exports=exports, podman_image=PODMAN_IMAGE, mounts=mounts, inner=inner,
                                     containers_dir=containers_dir, sqsh=sqsh, enroot_uri=enroot_uri,
                                     manifest=manifest, manifest_key=manifest_key))
        return

    try:
        import nemo_run as run
    except ImportError:
        console.print("[red bold]Error:[/red bold] nemo-run is required for build")
        raise typer.Exit(1)

    from nemo_runspec.execution import materialize_podman_auth_from_enroot

    tunnel = run.SSHTunnel(host=host, user=user, job_dir=remote_job_dir)
    tunnel.connect()
    tunnel.run(f"mkdir -p {staging} {containers_dir}", hide=True)

    # Bridge host enroot credentials -> podman auth.json so the in-container
    # `podman push` authenticates to the registry.
    mounts = f"{staging}:{staging}"
    try:
        auth_path = materialize_podman_auth_from_enroot(tunnel, f"{build_cache_dir}/.auth")
        if auth_path:
            mounts += f",{auth_path}:/root/.config/containers/auth.json:ro"
        else:
            console.print("[yellow]Note:[/yellow] no podman auth bridged; push may need REGISTRY_TOKEN.")
    except Exception as exc:
        console.print(f"[dim]podman auth bridge skipped: {exc}[/dim]")

    # Ship the two build inputs and the generated orchestration script.
    tunnel.put(str(local_dockerfile), df_remote)
    tunnel.put(str(local_inner), inner)
    sbatch_body = _render_sbatch(exports=exports, podman_image=PODMAN_IMAGE, mounts=mounts, inner=inner,
                                 containers_dir=containers_dir, sqsh=sqsh, enroot_uri=enroot_uri,
                                 manifest=manifest, manifest_key=manifest_key)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False)
    tmp.write(sbatch_body)
    tmp.close()
    sbatch_remote = str(PurePosixPath(staging) / "build.sbatch")
    tunnel.put(tmp.name, sbatch_remote)
    Path(tmp.name).unlink(missing_ok=True)

    submit = (
        f"sbatch --account={shlex.quote(env_config.get('account') or '')} "
        f"--partition={shlex.quote(partition or '')} --time={shlex.quote(walltime)} "
        f"--nodes=1 --ntasks-per-node=1 --job-name=kit-build-{manifest_key} "
        f"--output={staging}/build_%j.out {sbatch_remote}"
    )
    console.print(f"[bold]Submitting build to {host} ({partition})...[/bold]")
    result = tunnel.run(submit, hide=False, warn=True)
    tunnel.cleanup()

    if result.ok:
        console.print(
            Panel(
                f"[green]Build submitted.[/green]\nLogs: {staging}/build_<jobid>.out\n"
                f"On success: {sqsh}\n\n[dim]run with: ... run.env.container_image={sqsh}[/dim]",
                title="[bold green]Submitted[/bold green]",
                border_style="green",
                expand=False,
            )
        )
    else:
        console.print(f"[red bold]Submit failed:[/red bold] {result.stderr or 'unknown'}")
        raise typer.Exit(1)
