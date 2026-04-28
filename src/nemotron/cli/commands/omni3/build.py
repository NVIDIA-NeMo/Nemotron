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

"""Build container images for Omni3 stages.

The dispatcher here owns the *build process* (podman → enroot → squashfs).
Each stage's recipe folder owns only the **Dockerfile** that describes what
goes into the container; the dispatcher synthesizes the bash command that
turns that Dockerfile into a runtime-mountable ``.sqsh`` and submits it
via nemo-run's ``run.Script(inline=...)``.

This split keeps recipe folders purely declarative and centralizes
build-policy decisions (build base image, enroot version pin, output
naming) in one place.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nemo_runspec.env import parse_env
from nemo_runspec.execution import (
    clone_git_repos_via_tunnel,
    get_startup_commands,
    materialize_podman_auth_from_enroot,
    prepend_startup_to_cmd,
)
from nemo_runspec.packaging import CodePackager
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.squash import (
    resolve_build_cache_dir,
    resolve_build_image,
    resolve_build_partition,
    resolve_build_time,
)

console = Console()

RECIPE_ROOT = Path("src/nemotron/recipes/omni3")
# Pyxis/enroot needs ``docker://<registry>#<repo>:<tag>`` for non-Docker-Hub
# registries; bare ``quay.io/...`` strings get misrouted to docker.io.
BUILD_IMAGE = "docker://quay.io#podman/stable:v5.3"
# Enroot version installed inside the build container at runtime to do
# the ``podman://<tag>`` → ``.sqsh`` conversion. Pinning gives a stable
# build-artifact format across submissions; the cluster's host-side
# enroot version doesn't have to match because pyxis just mounts the
# resulting squashfs (a universal Linux filesystem). Override with
# ``NEMOTRON_BUILD_ENROOT_VERSION`` on the submitter's environment if a
# specific cluster needs a different version.
ENROOT_VERSION = os.environ.get("NEMOTRON_BUILD_ENROOT_VERSION", "3.5.0")
# Local default for the host-side build cache. For remote builds, set
# ``build_cache_dir`` in your env.toml profile to a cluster-visible path
# (typically on Lustre) — the laptop's ``Path.home()`` is not visible to
# compute nodes.
DEFAULT_BUILD_CACHE_DIR = Path.home() / ".cache" / "nemotron"
CONTAINER_CACHE_PATH = "/nemotron-cache"
# In-container path where the rsync'd repo lives (nemo-run mounts the
# CodePackager output here). The Dockerfile path inside the build
# container is ``{REMOTE_CODE_ROOT}/{stage_dir}/Dockerfile``.
REMOTE_CODE_ROOT = "/nemo_run/code"
# Build base image's resource needs — CPU-only single-node container
# build. Was previously read from each recipe's ``build.py`` runspec
# header; consolidated here since every omni3 stage has the same shape.
BUILD_NODES = 1
BUILD_GPUS_PER_NODE = 0


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------
# Maps each user-visible stage name to:
#   - the recipe directory (relative to repo root)
#   - the OCI image tag the build produces (transient, only used inside
#     the build container as the bridge between podman storage and enroot)
#   - the basename of the output ``.sqsh`` placed under
#     ``<build_cache_dir>/containers/``
#
# Adding a new stage is one row here plus a Dockerfile in the recipe
# folder; nothing else.


STAGES: dict[str, dict[str, str]] = {
    "stage0_sft": {
        "image_tag": "nemotron/omni3-sft:latest",
        "sqsh_name": "omni3-sft.sqsh",
    },
    "stage1_rl": {
        "image_tag": "nemotron/omni3-rl:latest",
        "sqsh_name": "omni3-rl.sqsh",
    },
}
STAGE_ALIASES: dict[str, str] = {
    "sft": "stage0_sft",
    "rl": "stage1_rl",
}


def _resolve_stage(stage: str) -> tuple[str, Path, str, str]:
    """Resolve a user-visible stage name to (canonical_name, dir, image_tag, sqsh_name)."""
    canonical = STAGE_ALIASES.get(stage, stage)
    if canonical not in STAGES:
        valid = sorted({*STAGE_ALIASES, *STAGES})
        raise typer.BadParameter(
            f"Unknown omni3 build stage '{stage}'. Expected one of: {', '.join(valid)}.",
        )
    stage_dir = RECIPE_ROOT / canonical
    if not stage_dir.is_dir():
        raise typer.BadParameter(
            f"Stage directory missing on disk: {stage_dir}",
        )
    if not (stage_dir / "Dockerfile").is_file():
        raise typer.BadParameter(
            f"Stage '{canonical}' is missing {stage_dir / 'Dockerfile'}.",
        )
    return canonical, stage_dir, STAGES[canonical]["image_tag"], STAGES[canonical]["sqsh_name"]


def _build_cache_mount(env) -> tuple[Path, str]:
    """Return ``(host_cache_dir, mount_string)`` for the build cache."""
    host_cache = resolve_build_cache_dir(env, DEFAULT_BUILD_CACHE_DIR)
    return host_cache, f"{host_cache}:{CONTAINER_CACHE_PATH}"


def _make_build_script(
    *,
    stage_dir: Path,
    image_tag: str,
    sqsh_name: str,
    enroot_version: str,
    extra_podman_args: list[str] | None = None,
) -> str:
    """Render the inline bash script that runs inside the build container.

    Phases:
      1. Install enroot at runtime (no custom build base image needed).
      2. ``podman build`` the recipe's Dockerfile.
      3. ``enroot import podman://<tag>`` → squashfs in the cache mount.

    The output ``.sqsh`` lands in the container at
    ``{CONTAINER_CACHE_PATH}/containers/<sqsh>`` which the host sees at
    ``<build_cache_dir>/containers/<sqsh>`` via the bind mount.
    """
    extra = " ".join(shlex.quote(a) for a in (extra_podman_args or []))
    extra_inline = f" {extra}" if extra else ""
    dockerfile_in_container = f"{REMOTE_CODE_ROOT}/{stage_dir}/Dockerfile"
    context_in_container = f"{REMOTE_CODE_ROOT}/{stage_dir}"
    output_in_container = f"{CONTAINER_CACHE_PATH}/containers/{sqsh_name}"
    rpm_url_base = f"https://github.com/NVIDIA/enroot/releases/download/v{enroot_version}"
    # Two dnf5/libdnf5 bugs on Fedora 41 force a two-phase install:
    #
    #   1. dnf5 crashes with ``std::length_error`` from
    #      ``basic_string::_M_replace_aux`` while installing/upgrading
    #      ncurses (any non-tty stdout triggers it). ``--quiet`` and
    #      ``script -qc`` PTY tricks don't avoid it.
    #
    #   2. enroot's only ncurses-tainted dep is ``parallel`` (via
    #      ``perl-Term-Cap``), which enroot uses for parallel layer
    #      downloads from registries. Our build does
    #      ``enroot import podman://...`` which reads from local
    #      podman storage — no parallel network fetch needed.
    #
    # So: install jq + squashfs-tools (small, ncurses-clean) via dnf,
    # then ``rpm --nodeps`` enroot itself, skipping the parallel
    # requirement. enroot's import-from-podman path runs fine without
    # parallel; if a future use case needs cross-layer parallelism,
    # add ``parallel`` here and accept the dnf workaround complexity.
    return f"""set -euo pipefail
export TERM=dumb NO_COLOR=1
echo "[omni3-build] installing enroot v{enroot_version} runtime deps ..."
# dnf5 on Fedora 41 has a string-handling bug in its progress reporter
# that crashes the *final* transaction step (``std::length_error`` from
# ``basic_string::_M_replace_aux``) regardless of which packages are
# being installed — files land on disk, then bookkeeping segfaults.
# We tolerate the non-zero exit and verify the binaries directly.
# Excluding ``ncurses`` keeps the dep set minimal (avoids the
# ``parallel → perl-Term-Cap → ncurses`` chain we don't need).
set +e
dnf install -y --quiet --exclude=ncurses jq squashfs-tools
dnf_rc=$?
set -e
if ! command -v jq >/dev/null 2>&1 || ! command -v mksquashfs >/dev/null 2>&1; then
    echo "[omni3-build] dnf install actually failed (rc=$dnf_rc); deps missing." >&2
    exit 1
fi
echo "[omni3-build] dnf install rc=$dnf_rc (non-zero is the dnf5 transaction-finish bug; files installed); deps present."
echo "[omni3-build] fetching + rpm --nodeps installing enroot ..."
mkdir -p /tmp/enroot-rpms
curl -fsSL -o /tmp/enroot-rpms/enroot.rpm \\
    {rpm_url_base}/enroot-{enroot_version}-1.el8.x86_64.rpm
curl -fsSL -o /tmp/enroot-rpms/enroot-caps.rpm \\
    {rpm_url_base}/enroot+caps-{enroot_version}-1.el8.x86_64.rpm
rpm -i --nodeps /tmp/enroot-rpms/enroot.rpm /tmp/enroot-rpms/enroot-caps.rpm
# enroot's default cache/data/runtime dirs land in $HOME (/root inside
# the pyxis container), which is a small tmpfs and fills immediately
# during ``enroot import`` of a 30 GB image. Redirect everything to
# the mounted cache dir on Lustre.
export ENROOT_CACHE_PATH={CONTAINER_CACHE_PATH}/enroot/cache
export ENROOT_DATA_PATH={CONTAINER_CACHE_PATH}/enroot/data
export ENROOT_RUNTIME_PATH={CONTAINER_CACHE_PATH}/enroot/runtime
export ENROOT_TEMP_PATH={CONTAINER_CACHE_PATH}/enroot/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH"
enroot version
echo "[omni3-build] podman build -t {image_tag} ..."
podman build{extra_inline} -f {dockerfile_in_container} -t {image_tag} {context_in_container}
echo "[omni3-build] enroot import podman://{image_tag} -> {output_in_container}"
mkdir -p {CONTAINER_CACHE_PATH}/containers
rm -f {output_in_container}
enroot import --output {output_in_container} podman://{image_tag}
ls -la {output_in_container}
echo "[omni3-build] done."
"""


def _show_build_plan(
    *,
    stage_canonical: str,
    stage_dir: Path,
    image_tag: str,
    sqsh_name: str,
    cfg,
    env,
) -> None:
    _, build_cache_mount = _build_cache_mount(env)
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Stage", stage_canonical)
    table.add_row("Stage dir", str(stage_dir))
    table.add_row("Image tag (build-time)", image_tag)
    table.add_row("Output sqsh", sqsh_name)
    table.add_row("Mode", cfg.mode)
    if env is not None:
        table.add_row("Profile", cfg.profile or "")
        table.add_row("Build base image", resolve_build_image(env, BUILD_IMAGE))
        table.add_row("Enroot version", ENROOT_VERSION)
        table.add_row("Partition", resolve_build_partition(env) or "")
        table.add_row("Time", resolve_build_time(env, "02:00:00"))
    else:
        table.add_row("Build base image", "local host execution")
    table.add_row("Cache mount", build_cache_mount)
    if cfg.passthrough:
        table.add_row("Extra podman args", " ".join(cfg.passthrough))
    console.print(table)

    # Heads-up for the typical first-time stumble: a CPU-only build
    # submitted to a profile whose ``run_partition`` requires GPUs.
    if env is not None and BUILD_GPUS_PER_NODE == 0 and not env.get("build_partition"):
        console.print(
            "[yellow]Note:[/yellow] this build is CPU-only "
            "(gpus_per_node=0) but no [bold]build_partition[/bold] is "
            "set in your env.toml profile. The dispatcher will fall "
            "back to [bold]run_partition[/bold] / [bold]partition[/bold], "
            "which on training clusters is typically GPU-only and will "
            "be rejected by sbatch. Set "
            "[bold]build_partition = \"<cpu-partition>\"[/bold] in the "
            "profile to fix.",
        )
    console.print()


def _execute_local(stage_dir: Path, image_tag: str, passthrough: list[str]) -> None:
    """Local fallback: run ``podman build`` only.

    Doesn't produce a ``.sqsh`` (most laptops don't have enroot). Useful
    for iterating on the Dockerfile before submitting a remote build.
    """
    cmd = ["podman", "build", "-f", str(stage_dir / "Dockerfile")]
    cmd.extend(passthrough)
    cmd.extend(["-t", image_tag, str(stage_dir)])
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        typer.echo(
            f"Built local image {image_tag}; no .sqsh produced "
            "(use --run/--batch <profile> for the slurm build pipeline).",
        )
    raise typer.Exit(result.returncode)


def _execute_slurm(
    *,
    stage_canonical: str,
    stage_dir: Path,
    image_tag: str,
    sqsh_name: str,
    passthrough: list[str],
    attached: bool,
    env,
) -> None:
    """Slurm/pyxis build path.

    Submits the inline build bash script as a single-task slurm job
    inside a pyxis-launched ``quay.io/podman/stable`` container. Pyxis-
    specific steps (pulling podman creds out of enroot's netrc, mounting
    them as docker auth.json) live here.
    """
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemo_runspec.run import patch_nemo_run_rsync_accept_new_host_keys

    patch_nemo_run_rsync_accept_new_host_keys()

    def _get(key: str, default=None):
        if env is None:
            return default
        return env.get(key, default) if hasattr(env, "get") else getattr(env, key, default)

    tunnel = None
    remote_job_dir = _get("remote_job_dir")
    if _get("tunnel") == "ssh":
        tunnel = run.SSHTunnel(
            host=_get("host", "localhost"),
            user=_get("user"),
            job_dir=remote_job_dir,
        )

    # CodePackager rsyncs the repo to ``$remote_job_dir/<exp>/code``,
    # which gets mounted at ``REMOTE_CODE_ROOT`` inside the container.
    # We don't need a real script anchor (we're inlining bash); a
    # placeholder train_path keeps CodePackager happy. The Dockerfile
    # itself is the natural anchor for ``script_path``.
    tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    try:
        tmp_config.write("{}\n")
        tmp_config.close()

        packager = CodePackager(
            script_path=str(stage_dir / "Dockerfile"),
            train_path=tmp_config.name,
        )

        host_cache, build_cache_mount = _build_cache_mount(env)

        git_mounts = []
        podman_auth_mount: str | None = None
        if tunnel and remote_job_dir:
            tunnel.connect()
            git_mounts = clone_git_repos_via_tunnel(tunnel, remote_job_dir)
            tunnel.run(f"mkdir -p {shlex.quote(str(host_cache))}", hide=True)
            auth_path = materialize_podman_auth_from_enroot(
                tunnel,
                f"{host_cache}/.auth",
            )
            if auth_path:
                podman_auth_mount = f"{auth_path}:/root/.config/containers/auth.json:ro"

        raw_mounts = list(_get("mounts") or [])
        mounts = [m for m in raw_mounts if not m.startswith("__auto_mount__:")]
        mounts.extend(git_mounts)
        mounts.append("/lustre:/lustre")
        mounts.append(build_cache_mount)
        if podman_auth_mount:
            mounts.append(podman_auth_mount)

        executor = run.SlurmExecutor(
            account=_get("account"),
            partition=resolve_build_partition(env),
            nodes=BUILD_NODES,
            ntasks_per_node=1,
            gpus_per_node=BUILD_GPUS_PER_NODE,
            cpus_per_task=_get("build_cpus", 16),
            time=resolve_build_time(env, "02:00:00"),
            container_image=resolve_build_image(env, BUILD_IMAGE),
            container_mounts=mounts,
            tunnel=tunnel,
            packager=packager,
            mem=_get("build_mem") or _get("mem"),
            env_vars={"NEMOTRON_CACHE_DIR": CONTAINER_CACHE_PATH},
            launcher=None,
        )

        build_script_inline = _make_build_script(
            stage_dir=stage_dir,
            image_tag=image_tag,
            sqsh_name=sqsh_name,
            enroot_version=ENROOT_VERSION,
            extra_podman_args=passthrough,
        )
        startup_commands = get_startup_commands(env)
        if startup_commands:
            build_script_inline = prepend_startup_to_cmd(
                startup_commands, build_script_inline
            )

        script_task = run.Script(inline=build_script_inline, entrypoint="bash")

        recipe_name = f"omni3-{stage_canonical}-build"
        with run.Experiment(recipe_name) as exp:
            exp.add(script_task, executor=executor, name=recipe_name)
            exp.run(detach=not attached)
    finally:
        Path(tmp_config.name).unlink(missing_ok=True)


def build(
    ctx: typer.Context,
    stage: str = typer.Argument(..., help="Stage name to build (for example: sft, rl)."),
) -> None:
    """Build an Omni3 stage container via podman + enroot on the cluster."""
    cfg = parse_recipe_config(ctx)
    stage_canonical, stage_dir, image_tag, sqsh_name = _resolve_stage(stage)
    env = parse_env(cfg.ctx)

    _show_build_plan(
        stage_canonical=stage_canonical,
        stage_dir=stage_dir,
        image_tag=image_tag,
        sqsh_name=sqsh_name,
        cfg=cfg,
        env=env,
    )

    if cfg.dry_run:
        return

    if cfg.mode == "local":
        _execute_local(stage_dir, image_tag, cfg.passthrough)
        return

    # Remote execution: dispatch by env.toml ``executor`` so non-slurm
    # executors (lepton, k8s, …) added in future PRs can plug in their
    # own branch without going through pyxis-specific machinery
    # (auth bridging, /lustre mounts, etc.).
    executor_kind = (env.get("executor") if env is not None else None) or "slurm"
    if executor_kind == "slurm":
        _execute_slurm(
            stage_canonical=stage_canonical,
            stage_dir=stage_dir,
            image_tag=image_tag,
            sqsh_name=sqsh_name,
            passthrough=cfg.passthrough,
            attached=cfg.attached,
            env=env,
        )
    else:
        typer.echo(
            f"Error: omni3 build does not yet support executor "
            f"'{executor_kind}'. Supported: slurm, local.",
            err=True,
        )
        raise typer.Exit(1)
