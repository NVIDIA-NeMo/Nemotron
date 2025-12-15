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

"""Pipeline orchestration for nemotron.kit.

Provides pipeline execution with multiple launcher backends:
- local: Subprocess piping with torchrun (default, for local development)
- nemo-run: NeMo-Run framework with LocalExecutor or SlurmExecutor
- sbatch: Generate and submit raw Slurm batch script

Example:
    >>> from nemotron.kit import Step
    >>> from nemotron.kit import PipelineConfig, run_pipeline
    >>>
    >>> steps = [
    ...     Step(name="data_prep", module=data_prep),
    ...     Step(name="training", module=training, torchrun=True),
    ... ]
    >>>
    >>> config = PipelineConfig(launcher="local", nproc_per_node=8)
    >>> exit_code = run_pipeline(config, steps)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Literal

from nemotron.kit.step import Step

Launcher = Literal["local", "nemo-run", "sbatch"]
"""Execution launcher backend: local, nemo-run, or sbatch."""


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestrator.

    Attributes:
        launcher: Execution backend ("local", "nemo-run", or "sbatch")
        nproc_per_node: Number of processes (GPUs) per node
        executor: NeMo-Run executor type ("local" or "slurm")
        account: Slurm account name
        partition: Slurm partition name
        nodes: Number of nodes for distributed training
        time: Slurm job time limit (HH:MM:SS)
        container_image: Container image path for Slurm jobs
        mounts: Container mount points (e.g., '/data:/data')
        job_name: Slurm job name
        tunnel: Tunnel type for NeMo-Run Slurm executor
        host: SSH host for remote job submission
        user: SSH user for remote job submission
        remote_job_dir: Remote directory for job files (SSH tunnel)
        identity: SSH identity file path
        env_vars: Environment variables (KEY=VALUE format)
        dry_run: Print commands/scripts without executing
        detach: Don't wait for job completion (nemo-run only)
        verbose: Print progress messages
    """

    # Launcher selection
    launcher: Launcher = "local"
    """Execution backend: local, nemo-run, or sbatch."""

    # Local/torchrun settings
    nproc_per_node: int = 8
    """Number of processes (GPUs) per node."""

    # Slurm settings (nemo-run and sbatch)
    executor: Literal["local", "slurm"] = "local"
    """NeMo-Run executor type."""

    account: str | None = None
    """Slurm account name."""

    partition: str | None = None
    """Slurm partition name."""

    nodes: int = 1
    """Number of nodes for distributed training."""

    time: str = "04:00:00"
    """Slurm job time limit (HH:MM:SS)."""

    container_image: str | None = None
    """Container image path for Slurm jobs."""

    mounts: list[str] = field(default_factory=list)
    """Container mount points (e.g., '/data:/data')."""

    job_name: str = "pipeline"
    """Slurm job name."""

    # NeMo-Run specific
    tunnel: Literal["local", "ssh"] = "local"
    """Tunnel type for NeMo-Run Slurm executor."""

    host: str | None = None
    """SSH host for remote job submission."""

    user: str | None = None
    """SSH user for remote job submission."""

    remote_job_dir: str | None = None
    """Remote directory for job files (SSH tunnel)."""

    identity: str | None = None
    """SSH identity file path."""

    # Environment
    env_vars: list[str] = field(default_factory=list)
    """Environment variables (KEY=VALUE format)."""

    # Common
    dry_run: bool = False
    """Print commands/scripts without executing."""

    detach: bool = False
    """Don't wait for job completion (nemo-run only)."""

    verbose: bool = True
    """Print progress messages."""


def run_pipeline(
    config: PipelineConfig,
    steps: list[Step],
    extra_args: list[str] | None = None,
) -> int:
    """Run a pipeline of steps.

    Args:
        config: Pipeline configuration.
        steps: List of steps to execute.
        extra_args: Additional args forwarded to steps.

    Returns:
        Exit code (0 = success).
    """
    if not steps:
        _log_error("No steps to run")
        return 1

    # Show plan
    step_names = [s.name for s in steps]
    _log_info(f"Pipeline: {' -> '.join(step_names)}")
    _log_info(f"Launcher: {config.launcher}")

    # Dispatch to launcher
    if config.launcher == "local":
        return run_local(config, steps, extra_args)
    elif config.launcher == "nemo-run":
        return run_nemo_run(config, steps, extra_args)
    elif config.launcher == "sbatch":
        return run_sbatch(config, steps, extra_args)
    else:
        _log_error(f"Unknown launcher: {config.launcher}")
        return 1


# =============================================================================
# Local Launcher (Subprocess Piping)
# =============================================================================


def run_local(
    config: PipelineConfig,
    steps: list[Step],
    extra_args: list[str] | None = None,
) -> int:
    """Execute pipeline locally with subprocess piping.

    Each step's stdout is piped to the next step's stdin.
    stderr is passed through for progress/logs.
    """
    if not steps:
        return 0

    processes: list[subprocess.Popen] = []

    try:
        prev_stdout = None

        for i, step in enumerate(steps):
            cmd = step.get_command(config.nproc_per_node)
            if extra_args:
                cmd.extend(extra_args)

            if config.verbose:
                _log_step_start(step.name, cmd)

            if config.dry_run:
                continue

            proc = subprocess.Popen(
                cmd,
                stdin=prev_stdout,
                stdout=subprocess.PIPE if i < len(steps) - 1 else None,
                stderr=None,  # Pass through to terminal
            )
            processes.append(proc)

            # Close previous stdout in parent to allow SIGPIPE
            if prev_stdout is not None:
                prev_stdout.close()

            prev_stdout = proc.stdout

        if config.dry_run:
            return 0

        # Wait for all processes
        exit_codes = []
        for proc in processes:
            proc.wait()
            exit_codes.append(proc.returncode)

        # Check for failures
        for step, code in zip(steps, exit_codes):
            if code != 0:
                _log_step_error(step.name, code)
                return code

        return 0

    except Exception as e:
        _log_error(f"Pipeline failed: {e}")
        for proc in processes:
            try:
                proc.kill()
            except Exception:
                pass
        return 1


# =============================================================================
# NeMo-Run Launcher
# =============================================================================


def run_nemo_run(
    config: PipelineConfig,
    steps: list[Step],
    extra_args: list[str] | None = None,
) -> int:
    """Execute pipeline using NeMo-Run framework."""
    # Handle dry-run without requiring nemo-run
    if config.dry_run:
        _log_info("NeMo-Run configuration:")
        _log_info(f"  Executor: {config.executor}")
        _log_info(f"  Steps: {[s.name for s in steps]}")
        _log_info(f"  GPUs/node: {config.nproc_per_node}")
        if config.executor == "slurm":
            _log_info(f"  Account: {config.account}")
            _log_info(f"  Partition: {config.partition}")
            _log_info(f"  Nodes: {config.nodes}")
            _log_info(f"  Time: {config.time}")
            if config.container_image:
                _log_info(f"  Container: {config.container_image}")
            if config.tunnel == "ssh":
                _log_info(f"  SSH tunnel: {config.user}@{config.host}")
        return 0

    try:
        import nemo_run as run
    except ImportError:
        _log_error(
            "nemo-run not installed. Install with: pip install nemo-run\n"
            "Or use --launcher local or --launcher sbatch"
        )
        return 1

    # Parse environment variables
    env_vars = {}
    for env in config.env_vars:
        if "=" in env:
            key, value = env.split("=", 1)
            env_vars[key] = value

    # Build executor
    if config.executor == "local":
        executor = run.LocalExecutor(
            ntasks_per_node=config.nproc_per_node,
            launcher="torchrun",
            env_vars=env_vars,
        )
    else:  # slurm
        # Validate required args
        if not config.account:
            _log_error("--account required for Slurm executor")
            return 1
        if not config.partition:
            _log_error("--partition required for Slurm executor")
            return 1

        # Build tunnel
        if config.tunnel == "ssh":
            if not config.host or not config.user:
                _log_error("--host and --user required for SSH tunnel")
                return 1
            tunnel = run.SSHTunnel(
                host=config.host,
                user=config.user,
                job_dir=config.remote_job_dir,
                identity=config.identity,
            )
        else:
            tunnel = run.LocalTunnel()

        executor = run.SlurmExecutor(
            account=config.account,
            partition=config.partition,
            nodes=config.nodes,
            ntasks_per_node=config.nproc_per_node,
            gpus_per_node=config.nproc_per_node,
            time=config.time,
            mem="0",
            exclusive=True,
            container_image=config.container_image,
            container_mounts=config.mounts,
            tunnel=tunnel,
            env_vars=env_vars,
        )

    # Build and run experiment
    with run.Experiment(config.job_name) as exp:
        # Inject experiment_id for artifact aliasing across tasks
        executor.env_vars["NEMO_EXPERIMENT_ID"] = exp._id

        for step in steps:
            # Build script path from module's __file__
            script_path = step.module_file

            task = run.Script(path=script_path)
            if extra_args:
                task.args = extra_args

            exp.add(task, executor=executor, name=step.name)

        exp.run(detach=config.detach, tail_logs=not config.detach)

    return 0


# =============================================================================
# Sbatch Launcher
# =============================================================================


def run_sbatch(
    config: PipelineConfig,
    steps: list[Step],
    extra_args: list[str] | None = None,
) -> int:
    """Generate and submit Slurm batch script."""
    # Validate required args
    if not config.account:
        _log_error("--account required for sbatch")
        return 1
    if not config.partition:
        _log_error("--partition required for sbatch")
        return 1

    script = generate_sbatch_script(config, steps, extra_args)

    if config.dry_run:
        print(script)
        return 0

    # Write to temp file and submit
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, prefix="pipeline_") as f:
        f.write(script)
        script_path = f.name

    _log_info(f"Generated script: {script_path}")
    _log_info("Submitting to Slurm...")

    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        _log_info(result.stdout.strip())
    else:
        _log_error(f"sbatch failed: {result.stderr}")

    return result.returncode


def generate_sbatch_script(
    config: PipelineConfig,
    steps: list[Step],
    extra_args: list[str] | None = None,
) -> str:
    """Generate sbatch script for pipeline."""
    # Build SBATCH directives
    directives = [
        f"#SBATCH --job-name={config.job_name}",
        f"#SBATCH --account={config.account}",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks-per-node={config.nproc_per_node}",
        f"#SBATCH --gpus-per-node={config.nproc_per_node}",
        f"#SBATCH --time={config.time}",
        "#SBATCH --exclusive",
    ]

    # Build environment exports
    env_exports = [
        "export TORCH_NCCL_AVOID_RECORD_STREAMS=1",
        "export NCCL_NVLS_ENABLE=0",
    ]

    for env in config.env_vars:
        if "=" in env:
            env_exports.append(f"export {env}")

    # Build pipeline commands
    pipeline_commands = generate_pipeline_commands(config, steps, extra_args)

    return f"""#!/bin/bash
{chr(10).join(directives)}

# Environment setup
{chr(10).join(env_exports)}

# Optional: Uncomment for debugging
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

echo "Starting pipeline on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Pipeline execution
{pipeline_commands}

echo "Pipeline complete"
"""


def generate_pipeline_commands(
    config: PipelineConfig,
    steps: list[Step],
    extra_args: list[str] | None = None,
) -> str:
    """Generate pipeline execution commands for sbatch script."""
    commands = []
    extra_args_str = " ".join(extra_args) if extra_args else ""

    for i, step in enumerate(steps):
        srun_cmd = step.get_srun_command(
            config.nproc_per_node,
            config.container_image,
            config.mounts,
        )

        if extra_args_str:
            srun_cmd = f"{srun_cmd} {extra_args_str}"

        # For piped steps, capture output to temp file
        if i == 0 and len(steps) > 1:
            # First step: output to temp file for next step
            commands.append(f'echo "Step: {step.name}"')
            commands.append(f"STEP_OUTPUT=$({srun_cmd})")
            commands.append(f'if [ $? -ne 0 ]; then echo "Step {step.name} failed"; exit 1; fi')
        elif i < len(steps) - 1:
            # Middle steps: read from prev, output for next
            commands.append(f'echo "Step: {step.name}"')
            commands.append(f'STEP_OUTPUT=$(echo "$STEP_OUTPUT" | {srun_cmd})')
            commands.append(f'if [ $? -ne 0 ]; then echo "Step {step.name} failed"; exit 1; fi')
        else:
            # Last step (or single step): normal execution
            commands.append(f'echo "Step: {step.name}"')
            if len(steps) > 1:
                commands.append(f'echo "$STEP_OUTPUT" | {srun_cmd}')
            else:
                commands.append(srun_cmd)
            commands.append(f'if [ $? -ne 0 ]; then echo "Step {step.name} failed"; exit 1; fi')

    return "\n".join(commands)


# =============================================================================
# Logging Helpers
# =============================================================================


def _log_step_start(name: str, cmd: list[str]) -> None:
    """Log step start to stderr."""
    cmd_str = " ".join(cmd)
    sys.stderr.write(f"\n{'=' * 60}\n")
    sys.stderr.write(f"[pipeline] Starting: {name}\n")
    sys.stderr.write(f"[pipeline] Command: {cmd_str}\n")
    sys.stderr.write(f"{'=' * 60}\n\n")
    sys.stderr.flush()


def _log_step_error(name: str, code: int) -> None:
    """Log step failure to stderr."""
    sys.stderr.write(f"\n[pipeline] ERROR: Step '{name}' failed with exit code {code}\n")
    sys.stderr.flush()


def _log_error(msg: str) -> None:
    """Log error to stderr."""
    sys.stderr.write(f"[pipeline] ERROR: {msg}\n")
    sys.stderr.flush()


def _log_info(msg: str) -> None:
    """Log info to stderr."""
    sys.stderr.write(f"[pipeline] {msg}\n")
    sys.stderr.flush()
