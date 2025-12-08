"""Tests for nemotron.kit.pipeline (PipelineConfig, run_pipeline)."""

import subprocess
import sys
from dataclasses import fields
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from nemotron.kit import PipelineConfig, Step, run_pipeline
from nemotron.kit.pipeline import (
    generate_sbatch_script,
    generate_pipeline_commands,
    run_local,
)


def create_mock_module(name: str, file_path: str | None = None) -> ModuleType:
    """Create a mock module for testing."""
    module = ModuleType(name)
    module.__name__ = name
    module.__file__ = file_path or f"/path/to/{name.replace('.', '/')}.py"
    return module


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()

        assert config.launcher == "local"
        assert config.nproc_per_node == 8
        assert config.executor == "local"
        assert config.nodes == 1
        assert config.time == "04:00:00"
        assert config.job_name == "pipeline"
        assert config.dry_run is False
        assert config.verbose is True

    def test_custom_values(self):
        """Test PipelineConfig with custom values."""
        config = PipelineConfig(
            launcher="sbatch",
            nproc_per_node=4,
            account="my_account",
            partition="batch",
            nodes=2,
            time="08:00:00",
            job_name="custom_job",
        )

        assert config.launcher == "sbatch"
        assert config.nproc_per_node == 4
        assert config.account == "my_account"
        assert config.partition == "batch"
        assert config.nodes == 2
        assert config.time == "08:00:00"
        assert config.job_name == "custom_job"

    def test_slurm_settings(self):
        """Test Slurm-specific settings."""
        config = PipelineConfig(
            launcher="nemo-run",
            executor="slurm",
            account="account",
            partition="gpu",
            container_image="/path/to/container.sqsh",
            mounts=["/data:/data"],
        )

        assert config.executor == "slurm"
        assert config.container_image == "/path/to/container.sqsh"
        assert config.mounts == ["/data:/data"]

    def test_ssh_tunnel_settings(self):
        """Test SSH tunnel settings."""
        config = PipelineConfig(
            tunnel="ssh",
            host="cluster.example.com",
            user="myuser",
            remote_job_dir="/remote/jobs",
            identity="~/.ssh/id_rsa",
        )

        assert config.tunnel == "ssh"
        assert config.host == "cluster.example.com"
        assert config.user == "myuser"
        assert config.remote_job_dir == "/remote/jobs"
        assert config.identity == "~/.ssh/id_rsa"

    def test_env_vars(self):
        """Test environment variables."""
        config = PipelineConfig(
            env_vars=["CUDA_VISIBLE_DEVICES=0,1", "NCCL_DEBUG=INFO"],
        )

        assert len(config.env_vars) == 2
        assert "CUDA_VISIBLE_DEVICES=0,1" in config.env_vars


class TestRunPipeline:
    """Tests for run_pipeline function."""

    def test_empty_steps_returns_error(self):
        """Test that empty steps list returns error code."""
        config = PipelineConfig()

        result = run_pipeline(config, [])

        assert result == 1

    def test_dry_run_local(self):
        """Test dry run with local launcher."""
        config = PipelineConfig(dry_run=True, verbose=False)
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module)]

        result = run_pipeline(config, steps)

        assert result == 0

    def test_unknown_launcher_returns_error(self):
        """Test that unknown launcher returns error."""
        config = PipelineConfig()
        # Manually set invalid launcher to bypass type checking
        object.__setattr__(config, "launcher", "invalid")

        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module)]

        result = run_pipeline(config, steps)

        assert result == 1


class TestRunLocal:
    """Tests for run_local function."""

    def test_dry_run_returns_zero(self):
        """Test dry run returns 0 without executing."""
        config = PipelineConfig(dry_run=True, verbose=False)
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module)]

        result = run_local(config, steps)

        assert result == 0

    def test_empty_steps_returns_zero(self):
        """Test empty steps list returns 0."""
        config = PipelineConfig()

        result = run_local(config, [])

        assert result == 0

    @patch("subprocess.Popen")
    def test_single_step_execution(self, mock_popen):
        """Test single step execution."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = None
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        config = PipelineConfig(verbose=False)
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module)]

        result = run_local(config, steps)

        assert result == 0
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_step_failure_returns_error(self, mock_popen):
        """Test that step failure returns error code."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = None
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        config = PipelineConfig(verbose=False)
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module)]

        result = run_local(config, steps)

        assert result == 1


class TestGenerateSbatchScript:
    """Tests for generate_sbatch_script function."""

    def test_basic_script_generation(self):
        """Test basic sbatch script generation."""
        config = PipelineConfig(
            account="test_account",
            partition="gpu",
            nodes=2,
            nproc_per_node=8,
            time="04:00:00",
            job_name="test_job",
        )
        module = create_mock_module("test.training")
        steps = [Step(name="training", module=module, torchrun=True)]

        script = generate_sbatch_script(config, steps)

        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=test_job" in script
        assert "#SBATCH --account=test_account" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --time=04:00:00" in script
        assert "srun" in script

    def test_script_with_env_vars(self):
        """Test script generation with environment variables."""
        config = PipelineConfig(
            account="account",
            partition="gpu",
            env_vars=["MY_VAR=value", "OTHER=123"],
        )
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module)]

        script = generate_sbatch_script(config, steps)

        assert "export MY_VAR=value" in script
        assert "export OTHER=123" in script

    def test_script_with_container(self):
        """Test script generation with container."""
        config = PipelineConfig(
            account="account",
            partition="gpu",
            container_image="/path/to/container.sqsh",
            mounts=["/data:/data"],
        )
        module = create_mock_module("test.training")
        steps = [Step(name="training", module=module, torchrun=True)]

        script = generate_sbatch_script(config, steps)

        assert "--container-image=/path/to/container.sqsh" in script
        assert "--container-mounts=/data:/data" in script


class TestGeneratePipelineCommands:
    """Tests for generate_pipeline_commands function."""

    def test_single_step_command(self):
        """Test command generation for single step."""
        config = PipelineConfig()
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module, torchrun=True)]

        commands = generate_pipeline_commands(config, steps)

        assert 'echo "Step: test"' in commands
        assert "srun" in commands
        assert "test.step" in commands

    def test_multiple_step_piping(self):
        """Test command generation for multiple steps with piping."""
        config = PipelineConfig()
        module1 = create_mock_module("step1")
        module2 = create_mock_module("step2")
        steps = [
            Step(name="step1", module=module1, torchrun=True),
            Step(name="step2", module=module2, torchrun=True),
        ]

        commands = generate_pipeline_commands(config, steps)

        # Should have piping logic
        assert "STEP_OUTPUT=" in commands
        assert 'echo "$STEP_OUTPUT"' in commands

    def test_extra_args_included(self):
        """Test that extra args are included in commands."""
        config = PipelineConfig()
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module, torchrun=True)]

        commands = generate_pipeline_commands(
            config, steps, extra_args=["--batch-size", "64"]
        )

        assert "--batch-size" in commands
        assert "64" in commands

    def test_error_handling_in_script(self):
        """Test that error handling is included."""
        config = PipelineConfig()
        module = create_mock_module("test.step")
        steps = [Step(name="test", module=module)]

        commands = generate_pipeline_commands(config, steps)

        assert "if [ $? -ne 0 ]" in commands
        assert "exit 1" in commands
