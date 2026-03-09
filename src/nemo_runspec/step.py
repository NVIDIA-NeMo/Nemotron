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

"""Pipeline step definition.

Defines the Step class used to represent executable pipeline steps.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType


@dataclass
class Step:
    """A pipeline step that can be executed.

    Steps wrap Python modules that can be run as scripts. They support
    both single-process (python) and distributed (torchrun) execution.

    Example:
        >>> from nemo_runspec.step import Step
        >>> from nemotron.recipes.nano3.stage0_pretrain import data_prep, training
        >>>
        >>> steps = [
        ...     Step(name="data_prep", module=data_prep),
        ...     Step(name="training", module=training, torchrun=True),
        ... ]
    """

    name: str
    """Step name (e.g., 'data_prep', 'training')."""

    module: ModuleType
    """Python module containing the step."""

    torchrun: bool = False
    """Whether to run with torchrun for distributed execution."""

    @property
    def module_name(self) -> str:
        """Get the module's fully qualified name."""
        return self.module.__name__

    @property
    def module_file(self) -> str | None:
        """Get the module's file path."""
        return self.module.__file__

    def get_command(self, nproc_per_node: int = 8) -> list[str]:
        """Build command to execute this step.

        Args:
            nproc_per_node: Number of processes per node (for torchrun).

        Returns:
            Command as list of strings.
        """
        if self.torchrun:
            return [
                "torchrun",
                f"--nproc_per_node={nproc_per_node}",
                "-m",
                self.module_name,
            ]
        return [sys.executable, "-m", self.module_name]

    def get_srun_command(
        self,
        nproc_per_node: int = 8,
        container_image: str | None = None,
        mounts: list[str] | None = None,
    ) -> str:
        """Build srun command for Slurm execution.

        Args:
            nproc_per_node: Number of processes per node.
            container_image: Container image path.
            mounts: Container mount points.

        Returns:
            Command as string for shell execution.
        """
        parts = ["srun", "--mpi=pmix"]

        if container_image:
            parts.append(f"--container-image={container_image}")
            if mounts:
                for mount in mounts:
                    parts.append(f"--container-mounts={mount}")

        parts.extend(
            [
                "torchrun",
                "--nproc_per_node=$SLURM_GPUS_PER_NODE",
                "--nnodes=$SLURM_JOB_NUM_NODES",
                "--node_rank=$SLURM_PROCID",
                '--master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
                "--master_port=29500",
                "-m",
                self.module_name,
            ]
        )

        return " ".join(parts)
