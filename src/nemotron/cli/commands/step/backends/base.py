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

"""Shared types — the ``Backend`` protocol and the immutable ``JobContext``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class JobContext:
    """Everything a backend needs to submit one job — captured in one struct.

    Built once by ``step run`` from the runspec + env profile + rendered
    config. Backends never poke at globals; they read from this.
    """

    step_id: str                       # e.g. "prep/sft_packing"
    script_path: Path                  # absolute path to the local step.py
    train_path: Path                   # absolute path to the rendered train.yaml
    spec: Any                          # nemo_runspec.Runspec
    env: Any                           # OmegaConf DictConfig or dict
    env_vars: dict[str, str]
    passthrough: list[str]
    startup_commands: list[str]
    attached: bool                     # --run vs --batch
    force_squash: bool

    @property
    def job_name(self) -> str:
        """Stable, RFC-1123-ish job slug derived from the step id."""
        slug = self.step_id.replace("/", "-").replace("_", "-").lower()
        return slug.strip("-") or "nemotron-step"


class Backend(Protocol):
    """A submission target — local, slurm, lepton, dgxcloud, …

    Backends own three concerns:
      1. Choosing the right packager for their transport.
      2. Building the executor (Local / Slurm / Lepton / DGXCloud).
      3. Submitting the job and (optionally) waiting for it.
    """

    name: str

    def submit(self, ctx: JobContext) -> None:
        """Run / submit the job. Blocks for attached, returns for detached."""
        ...
