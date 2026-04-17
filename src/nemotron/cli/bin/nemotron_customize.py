#!/usr/bin/env python3

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

"""Nemotron Customize — Command Dispatcher for multi-container deployments.

Routes ``nemotron customize <subcommand>`` to the correct Docker container in
the multi-container Compose setup.  Inspired by the Speaker ``speaker-run``
dispatcher pattern (kipraveen/speaker-run branch).

Subcommand → Container mapping
------------------------------
  translate   → nemotron-curator     (Translation for data preparation)
  data-prep   → nemotron-curator     (NeMo Curator for data processing)
  sdg         → nemotron-curator     (DataDesigner for synthetic generation)
  byob        → nemotron-curator     (BYOB MCQ pipeline uses NeMo Curator)
  cpt         → nemotron-trainer     (CPT needs NeMo + Megatron-Bridge)
  sft         → nemotron-trainer     (SFT needs NeMo + Megatron-Bridge)
  rl          → nemotron-trainer     (RL needs NeMo + Ray)
  eval        → nemotron-evaluator   (Uses nemo-evaluator-launcher)
  quantize    → nemotron-trainer     (Needs model loading + TensorRT)

Usage (from orchestrator container or host)::

    nemotron-customize data-prep -c default
    nemotron-customize sft -c default --run MY-CLUSTER
    nemotron-customize eval -c default -it

The script discovers sibling containers via Docker and forwards the full
command line, including all config flags and overrides.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import NoReturn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Subcommand → target service name (as defined in docker-compose.yaml)
COMMAND_ROUTING: dict[str, str] = {
    "translate": "nemotron-curator",
    "data-prep": "nemotron-curator",
    "sdg": "nemotron-curator",
    "byob": "nemotron-curator",
    "cpt": "nemotron-trainer",
    "sft": "nemotron-trainer",
    "rl": "nemotron-trainer",
    "eval": "nemotron-evaluator",
    "quantize": "nemotron-trainer",
}

#: Environment variables to forward to target containers
FORWARDED_ENV_VARS: list[str] = [
    "NGC_API_KEY",
    "NVIDIA_API_KEY",
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
]

#: Default compose project name (matches ``name:`` in docker-compose.yaml)
DEFAULT_PROJECT_NAME = "nemotron-customize"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("nemotron-customize")


def _setup_logging() -> None:
    """Configure console logging with a clean format."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------


def _is_inside_container() -> bool:
    """Return True if we are running inside a Docker container."""
    if os.environ.get("NEMOTRON_CONTAINER"):
        return True
    if os.path.exists("/.dockerenv"):
        return True
    # cgroup-based check (works on most Docker runtimes)
    try:
        with open("/proc/1/cgroup", "r") as fh:
            return "docker" in fh.read() or "containerd" in fh.read()
    except (FileNotFoundError, PermissionError):
        return False


def _docker_available() -> bool:
    """Return True if the ``docker`` CLI is available."""
    return shutil.which("docker") is not None


def _compose_available() -> bool:
    """Return True if ``docker compose`` (v2 plugin) is available."""
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Container Discovery
# ---------------------------------------------------------------------------


def _get_project_name() -> str:
    """Return the Compose project name used for container naming."""
    return os.environ.get("COMPOSE_PROJECT_NAME", DEFAULT_PROJECT_NAME)


def _resolve_container_name(service: str) -> str | None:
    """Resolve the running container name for a Compose *service*.

    Docker Compose v2 names containers as ``{project}-{service}-{replica}``.
    We try several patterns and return the first that is actually running.

    Returns:
        The container name string, or ``None`` if no running container found.
    """
    project = _get_project_name()

    # Candidate container names, in order of preference
    candidates = [
        f"{project}-{service}-1",          # docker compose v2 default
        f"{project}_{service}_1",          # docker compose v1 legacy
        service,                           # bare service name (user-defined)
    ]

    for name in candidates:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip() == "true":
                return name
        except FileNotFoundError:
            return None

    return None


def _container_is_running(service: str) -> tuple[bool, str | None]:
    """Check if the service container is running.

    Returns:
        Tuple of (is_running, container_name).
    """
    name = _resolve_container_name(service)
    return (name is not None, name)


# ---------------------------------------------------------------------------
# Command Building
# ---------------------------------------------------------------------------


def _build_env_flags() -> list[str]:
    """Build ``-e KEY=VALUE`` flags for environment variables to forward."""
    flags: list[str] = []
    for var in FORWARDED_ENV_VARS:
        value = os.environ.get(var)
        if value:
            flags.extend(["-e", f"{var}={value}"])
    return flags


def _build_docker_exec_cmd(
    container_name: str,
    subcommand: str,
    passthrough_args: list[str],
    *,
    interactive: bool = False,
) -> list[str]:
    """Build the full ``docker exec`` command line.

    Args:
        container_name: Name of the target container.
        subcommand: The customize subcommand (e.g. ``sft``, ``eval``).
        passthrough_args: Remaining CLI arguments to forward.
        interactive: Whether to allocate a TTY (``-it``).

    Returns:
        List of command tokens suitable for ``subprocess.run()``.
    """
    cmd = ["docker", "exec"]

    if interactive:
        cmd.append("-it")

    # Forward environment variables
    cmd.extend(_build_env_flags())

    # Target container + inner command
    cmd.append(container_name)
    cmd.extend(["nemotron", "customize", subcommand])
    cmd.extend(passthrough_args)

    return cmd


def _build_compose_exec_cmd(
    service: str,
    subcommand: str,
    passthrough_args: list[str],
    *,
    interactive: bool = False,
) -> list[str]:
    """Build a ``docker compose exec`` command as fallback.

    This is used when we cannot resolve the container name directly (e.g.
    non-standard project names).

    Args:
        service: Compose service name (e.g. ``nemotron-curator``).
        subcommand: The customize subcommand.
        passthrough_args: Remaining CLI arguments to forward.
        interactive: Whether to allocate a TTY.

    Returns:
        List of command tokens.
    """
    cmd = ["docker", "compose", "exec"]

    if not interactive:
        cmd.append("-T")  # compose exec is interactive by default

    # Forward environment variables
    for var in FORWARDED_ENV_VARS:
        value = os.environ.get(var)
        if value:
            cmd.extend(["-e", f"{var}={value}"])

    cmd.append(service)
    cmd.extend(["nemotron", "customize", subcommand])
    cmd.extend(passthrough_args)

    return cmd


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> tuple[str | None, bool, list[str]]:
    """Parse dispatcher-level arguments from *argv*.

    Extracts the subcommand and ``-it``/``--interactive`` flag, leaving
    everything else as passthrough arguments for the target container.

    Args:
        argv: Raw argument list (``sys.argv[1:]``).

    Returns:
        Tuple of (subcommand, interactive, passthrough_args).
        subcommand is ``None`` if not provided or ``--help`` requested.
    """
    interactive = False
    passthrough: list[str] = []
    subcommand: str | None = None

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in ("-it", "--interactive"):
            interactive = True
        elif arg in ("-h", "--help") and subcommand is None:
            # Help at the dispatcher level
            return None, False, []
        elif subcommand is None and not arg.startswith("-"):
            subcommand = arg
        else:
            passthrough.append(arg)

        i += 1

    return subcommand, interactive, passthrough


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _print_usage() -> None:
    """Print dispatcher usage / help text."""
    print(
        "nemotron-customize — Command dispatcher for Nemotron multi-container setup\n"
        "\n"
        "Usage:\n"
        "  nemotron-customize <subcommand> [options...]\n"
        "  nemotron customize  <subcommand> [options...]   (via nemotron CLI)\n"
        "\n"
        "Subcommands (routed to the correct container automatically):\n"
    )
    # Print routing table
    for cmd, svc in COMMAND_ROUTING.items():
        print(f"  {cmd:<12}  ->  {svc}")
    print(
        "\n"
        "Options:\n"
        "  -it, --interactive   Attach interactive TTY to target container\n"
        "  -h, --help           Show this help message\n"
        "\n"
        "All other arguments are forwarded to the target container's\n"
        "``nemotron customize <subcommand>`` command.\n"
        "\n"
        "Examples:\n"
        "  nemotron-customize data-prep -c default\n"
        "  nemotron-customize sft -c default --run MY-CLUSTER train.train_iters=5000\n"
        "  nemotron-customize eval -c default -it\n"
        "\n"
        "Environment:\n"
        "  COMPOSE_PROJECT_NAME    Override compose project name (default: nemotron-customize)\n"
        "  NEMOTRON_ORCHESTRATOR   Set to '1' to enable dispatcher mode (auto-set in orchestrator)\n"
        "  NGC_API_KEY, HF_TOKEN, OPENAI_API_KEY, WANDB_API_KEY — forwarded to target containers\n"
    )


def dispatch(argv: list[str] | None = None) -> NoReturn:
    """Main dispatcher entry point.

    Parses the subcommand from *argv*, resolves the target container, and
    executes the command via ``docker exec``.  Falls back to local execution
    if Docker is not available.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.
    """
    _setup_logging()

    if argv is None:
        argv = sys.argv[1:]

    subcommand, interactive, passthrough = _parse_args(argv)

    # -----------------------------------------------------------------------
    # Help / no subcommand
    # -----------------------------------------------------------------------
    if subcommand is None:
        _print_usage()
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Validate subcommand
    # -----------------------------------------------------------------------
    if subcommand not in COMMAND_ROUTING:
        logger.error(
            "Unknown subcommand '%s'. Valid subcommands: %s",
            subcommand,
            ", ".join(sorted(COMMAND_ROUTING)),
        )
        sys.exit(1)

    target_service = COMMAND_ROUTING[subcommand]

    # -----------------------------------------------------------------------
    # Check Docker availability
    # -----------------------------------------------------------------------
    if not _docker_available():
        logger.warning(
            "Docker CLI not found. Falling back to local execution. "
            "Install Docker or run inside the orchestrator container."
        )
        # Fall back: run the command locally (assumes deps are installed)
        local_cmd = ["nemotron", "customize", subcommand, *passthrough]
        logger.info("Executing locally: %s", " ".join(local_cmd))
        result = subprocess.run(local_cmd)
        sys.exit(result.returncode)

    # -----------------------------------------------------------------------
    # Resolve target container
    # -----------------------------------------------------------------------
    is_running, container_name = _container_is_running(target_service)

    if is_running and container_name:
        # Primary path: docker exec with resolved container name
        cmd = _build_docker_exec_cmd(
            container_name,
            subcommand,
            passthrough,
            interactive=interactive,
        )
        logger.info(
            "Dispatching '%s' to container %s (%s)",
            subcommand,
            container_name,
            target_service,
        )
    elif _compose_available():
        # Fallback: docker compose exec (lets compose resolve the container)
        cmd = _build_compose_exec_cmd(
            target_service,
            subcommand,
            passthrough,
            interactive=interactive,
        )
        logger.info(
            "Dispatching '%s' via docker compose exec to service %s",
            subcommand,
            target_service,
        )
    else:
        # Container not running and compose not available
        logger.error(
            "Container %s is not running. "
            "Run 'docker compose up -d' first.\n"
            "\n"
            "  cd deploy/nemotron/customization_recipes\n"
            "  docker compose up -d\n",
            target_service,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main() -> NoReturn:
    """Console script entry point for ``nemotron-customize``."""
    dispatch()


if __name__ == "__main__":
    main()
