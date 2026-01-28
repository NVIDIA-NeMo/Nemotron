"""GPU-tier smoke tests for embed recipe — local and Docker modes.

Validates the stage-specific uv/pyproject.toml dependency system that
each embed recipe stage (1-3) relies on for both local and Docker execution.

Tier 0 (``TestCliImportChain``, ``@pytest.mark.integration``):
    Validates that ``nemotron embed <cmd> --help`` works for each CLI
    command, catching broken import chains in the nemotron package.

Tier 1 (``TestLocalUvSync``, ``@pytest.mark.gpu``):
    Exercises ``uv sync --frozen --project <stage_dir>`` in an isolated venv,
    then validates key imports and script parseability.

Tier 2 (``TestDockerRunUv``, ``@pytest.mark.docker``):
    Runs ``run_uv.py`` inside the production NGC container to validate the
    exclude-dependencies injection + ``uv sync`` mechanism.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import EMBED_DIR, STAGES

# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------
STAGE_NAMES = list(STAGES.keys())

# CLI subcommands that map to embed recipe stages
EMBED_CLI_COMMANDS = ["sdg", "prep", "finetune", "eval", "export", "deploy"]


# ===================================================================
# Tier 0: CLI import chain — catches broken imports in nemotron pkg
# ===================================================================
@pytest.mark.integration
class TestCliImportChain:
    """Validate that ``nemotron embed <cmd> --help`` succeeds for each command.

    This catches broken import chains (e.g. deleted modules still referenced
    by imports) that wouldn't surface in stage-venv or data-format tests.
    """

    @pytest.mark.parametrize("cmd", EMBED_CLI_COMMANDS)
    def test_embed_cli_help(self, cmd: str) -> None:
        """``nemotron embed <cmd> --help`` exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "nemotron", "embed", cmd, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"'nemotron embed {cmd} --help' failed (rc={result.returncode}):\n"
            f"--- stderr ---\n{result.stderr}"
        )

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_uv_run_with_injection(self, stage_name: str) -> None:
        """``uv run --with <root> --project <stage> python -c "import ..."`` exits 0.

        The CLI launches stage scripts via ``uv run --with <root_package>
        --project <stage_dir> python <entry_script>``.  This builds a wheel
        from the root project and installs it alongside stage deps.  If any
        module is missing from the built wheel (e.g. a deleted subpackage
        still referenced by imports), this test will catch it.
        """
        import shutil

        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv not found on PATH")

        stage_dir = EMBED_DIR / stage_name
        code_root = EMBED_DIR.parents[3]

        # Import the config_loader — this triggers the full nemotron.kit
        # import chain (artifact → artifacts, pydantic_settings, etc.)
        result = subprocess.run(
            [
                uv,
                "run",
                "--no-cache",
                "--with",
                str(code_root),
                "--project",
                str(stage_dir),
                "python",
                "-c",
                "from nemotron.kit.config_loader import parse_config_and_overrides",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"uv run --with injection failed for {stage_name}:\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )


# ===================================================================
# Tier 1: Local uv sync — validates pyproject.toml + uv.lock
# ===================================================================
@pytest.mark.gpu
class TestLocalUvSync:
    """Validate that ``uv sync`` resolves each stage's dependencies locally.

    Tests run ``uv sync`` and imports as subprocesses against isolated stage
    venvs — the test runner itself does not need torch or CUDA.
    """

    # -- uv sync succeeds ---------------------------------------------------

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_uv_sync_succeeds(
        self, stage_name: str, stage_venv_factory
    ) -> None:
        """``uv sync --frozen`` exits 0 for the stage's pyproject.toml."""
        python = stage_venv_factory(stage_name)
        assert python.is_file(), f"venv python not created for {stage_name}"

    # -- key imports work ---------------------------------------------------

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_key_imports(
        self, stage_name: str, stage_venv_factory
    ) -> None:
        """Each stage's key packages are importable in the synced venv."""
        python = stage_venv_factory(stage_name)
        key_imports = STAGES[stage_name]["key_imports"]

        for pkg in key_imports:
            result = subprocess.run(
                [str(python), "-c", f"import {pkg}"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, (
                f"Failed to import {pkg!r} in {stage_name} venv:\n"
                f"--- stderr ---\n{result.stderr}"
            )

    # -- entry script is parseable ------------------------------------------

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_stage_script_parseable(
        self, stage_name: str, stage_venv_factory
    ) -> None:
        """The stage entry script parses without syntax errors."""
        python = stage_venv_factory(stage_name)
        script = EMBED_DIR / stage_name / STAGES[stage_name]["entry_script"]
        assert script.is_file(), f"Entry script not found: {script}"

        result = subprocess.run(
            [
                str(python),
                "-c",
                f"import ast; ast.parse(open({str(script)!r}).read())",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Syntax error in {script.name} for {stage_name}:\n"
            f"--- stderr ---\n{result.stderr}"
        )


# ===================================================================
# Tier 2: Docker run_uv.py — validates container-mode dependency flow
# ===================================================================
def _docker_image_available(image: str) -> bool:
    """Return True if *image* is already pulled locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.gpu
@pytest.mark.docker
class TestDockerRunUv:
    """Validate the Docker-mode uv mechanism (run_uv.py) inside NGC containers."""

    @pytest.fixture(autouse=True)
    def _require_docker(self, docker_available: bool) -> None:
        if not docker_available:
            pytest.skip("Docker with NVIDIA runtime not available")

    # -- run_uv.py --help succeeds in container -----------------------------

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_docker_uv_sync(self, stage_name: str) -> None:
        """``run_uv.py --help`` inside the production container exits 0.

        This validates that:
        1. The container image is pullable/available
        2. run_uv.py's exclude-dependencies injection works
        3. ``uv sync`` succeeds inside the container
        4. The target script is found and accepts --help
        """
        meta = STAGES[stage_name]
        image = meta["container"]

        if not _docker_image_available(image):
            pytest.skip(
                f"Container image {image} not available locally — "
                f"pull with: docker pull {image}"
            )

        code_root = EMBED_DIR.parents[3]  # src/nemotron/recipes/embed -> repo root
        stage_dir_in_container = (
            f"/nemo_run/code/src/nemotron/recipes/embed/{stage_name}"
        )

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{code_root}:/nemo_run/code",
                image,
                "python3",
                f"{stage_dir_in_container}/run_uv.py",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"run_uv.py --help failed for {stage_name} in {image}:\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    # -- end-to-end with synthetic data in container ------------------------

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_docker_stage_with_synthetic_data(
        self,
        stage_name: str,
        convert_output_dir: Path,
    ) -> None:
        """Mount synthetic data and run the stage entry script via run_uv.py.

        Validates end-to-end data flow in Docker mode.  Each stage gets
        different args to exercise a minimal but real invocation.
        """
        meta = STAGES[stage_name]
        image = meta["container"]

        if not _docker_image_available(image):
            pytest.skip(
                f"Container image {image} not available locally — "
                f"pull with: docker pull {image}"
            )

        code_root = EMBED_DIR.parents[3]
        stage_dir_in_container = (
            f"/nemo_run/code/src/nemotron/recipes/embed/{stage_name}"
        )

        # Build stage-specific args
        args = _synthetic_data_args(stage_name, convert_output_dir)
        if args is None:
            pytest.skip(
                f"No synthetic data invocation defined for {stage_name}"
            )

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{code_root}:/nemo_run/code",
                "-v",
                f"{convert_output_dir}:/data",
                image,
                "python3",
                f"{stage_dir_in_container}/run_uv.py",
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, (
            f"Docker run failed for {stage_name}:\n"
            f"--- stdout ---\n{result.stdout[-2000:]}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )


def _synthetic_data_args(
    stage_name: str, convert_output_dir: Path
) -> list[str] | None:
    """Return CLI args for a minimal run of the given stage with synthetic data.

    Returns None if no invocation is defined for the stage (will skip).
    """
    if stage_name == "stage1_data_prep":
        return [
            "--data_dir=/data",
            "--output_dir=/tmp/stage1_out",
            "--corpus_id=test_corpus",
        ]
    elif stage_name == "stage2_finetune":
        return [
            "--train_data=/data/train.json",
            "--num_epochs=0",
        ]
    elif stage_name == "stage3_eval":
        return [
            "--eval_data=/data/eval_beir",
            "--batch_size=1",
        ]
    return None
