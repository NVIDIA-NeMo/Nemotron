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

"""Tests for nemotron.kit.step (Step class)."""

import sys
from types import ModuleType

from nemotron.kit import Step


def create_mock_module(name: str, file_path: str | None = None) -> ModuleType:
    """Create a mock module for testing."""
    module = ModuleType(name)
    module.__name__ = name
    module.__file__ = file_path
    return module


class TestStep:
    """Tests for Step dataclass."""

    def test_step_creation(self):
        """Test basic Step creation."""
        module = create_mock_module("test.module", "/path/to/module.py")
        step = Step(name="test_step", module=module)

        assert step.name == "test_step"
        assert step.module == module
        assert step.torchrun is False  # default

    def test_step_with_torchrun(self):
        """Test Step with torchrun=True."""
        module = create_mock_module("train.module", "/path/to/train.py")
        step = Step(name="training", module=module, torchrun=True)

        assert step.name == "training"
        assert step.torchrun is True

    def test_module_name(self):
        """Test module_name property."""
        module = create_mock_module("nemotron.recipes.nano3.data_prep")
        step = Step(name="data_prep", module=module)

        assert step.module_name == "nemotron.recipes.nano3.data_prep"

    def test_module_file(self):
        """Test module_file property."""
        module = create_mock_module("test.module", "/path/to/test/module.py")
        step = Step(name="test", module=module)

        assert step.module_file == "/path/to/test/module.py"

    def test_module_file_none(self):
        """Test module_file when __file__ is None."""
        module = create_mock_module("builtin.module", None)
        step = Step(name="builtin", module=module)

        assert step.module_file is None

    def test_get_command_python(self):
        """Test get_command for non-torchrun step."""
        module = create_mock_module("my.data_prep")
        step = Step(name="data_prep", module=module, torchrun=False)

        cmd = step.get_command()

        assert cmd == [sys.executable, "-m", "my.data_prep"]

    def test_get_command_torchrun(self):
        """Test get_command for torchrun step."""
        module = create_mock_module("my.training")
        step = Step(name="training", module=module, torchrun=True)

        cmd = step.get_command(nproc_per_node=8)

        assert cmd == [
            "torchrun",
            "--nproc_per_node=8",
            "-m",
            "my.training",
        ]

    def test_get_command_custom_nproc(self):
        """Test get_command with custom nproc_per_node."""
        module = create_mock_module("my.training")
        step = Step(name="training", module=module, torchrun=True)

        cmd = step.get_command(nproc_per_node=4)

        assert "--nproc_per_node=4" in cmd

    def test_get_srun_command_basic(self):
        """Test get_srun_command without container."""
        module = create_mock_module("my.training")
        step = Step(name="training", module=module, torchrun=True)

        cmd = step.get_srun_command(nproc_per_node=8)

        assert "srun" in cmd
        assert "--mpi=pmix" in cmd
        assert "torchrun" in cmd
        assert "--nproc_per_node=$SLURM_GPUS_PER_NODE" in cmd
        assert "-m" in cmd
        assert "my.training" in cmd

    def test_get_srun_command_with_container(self):
        """Test get_srun_command with container image."""
        module = create_mock_module("my.training")
        step = Step(name="training", module=module, torchrun=True)

        cmd = step.get_srun_command(
            nproc_per_node=8,
            container_image="/path/to/container.sqsh",
            mounts=["/data:/data", "/scratch:/scratch"],
        )

        assert "--container-image=/path/to/container.sqsh" in cmd
        assert "--container-mounts=/data:/data" in cmd
        assert "--container-mounts=/scratch:/scratch" in cmd

    def test_get_srun_command_returns_string(self):
        """Test that get_srun_command returns a string (for shell)."""
        module = create_mock_module("my.training")
        step = Step(name="training", module=module, torchrun=True)

        cmd = step.get_srun_command()

        assert isinstance(cmd, str)
        assert " " in cmd  # Multiple parts joined


class TestStepEquality:
    """Tests for Step equality and hashing."""

    def test_steps_equal(self):
        """Test that identical steps are equal."""
        module = create_mock_module("test.module")
        step1 = Step(name="test", module=module, torchrun=True)
        step2 = Step(name="test", module=module, torchrun=True)

        assert step1 == step2

    def test_steps_not_equal_name(self):
        """Test that steps with different names are not equal."""
        module = create_mock_module("test.module")
        step1 = Step(name="test1", module=module)
        step2 = Step(name="test2", module=module)

        assert step1 != step2

    def test_steps_not_equal_torchrun(self):
        """Test that steps with different torchrun are not equal."""
        module = create_mock_module("test.module")
        step1 = Step(name="test", module=module, torchrun=True)
        step2 = Step(name="test", module=module, torchrun=False)

        assert step1 != step2
