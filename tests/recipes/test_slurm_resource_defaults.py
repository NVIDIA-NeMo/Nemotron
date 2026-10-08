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

"""Regression tests for recipe resource defaults in remote CLI launchers."""

from importlib import import_module
from unittest.mock import MagicMock

import nemo_run as run
import pytest
from omegaconf import OmegaConf

from nemo_runspec import run as runspec_run


@pytest.mark.parametrize(
    "command",
    ["lightning35.pretrain", "lightning35.sft", "lightning35.eval", "nano3.pretrain", "nano3.sft"],
)
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="recipe-defaults"),
        pytest.param({"nodes": 3}, id="override-nodes"),
        pytest.param({"gpus_per_node": 4}, id="override-gpus"),
        pytest.param({"nodes": 3, "gpus_per_node": 4, "ntasks_per_node": 2}, id="override-all"),
    ],
)
def test_remote_launcher_slurm_resources(command, overrides, monkeypatch, tmp_path):
    """Exercise the real executor factory without packaging or submitting jobs."""
    module = import_module(f"nemotron.cli.commands.{command}")
    experiment_factory = MagicMock()
    monkeypatch.setattr(run, "Experiment", experiment_factory)
    monkeypatch.setattr(runspec_run, "patch_nemo_run_rsync_accept_new_host_keys", lambda: None)
    monkeypatch.setattr(runspec_run, "patch_nemo_run_ray_template_for_cpu", lambda: None)

    module._execute_remote(
        train_path=tmp_path / "config.yaml",
        env=OmegaConf.create({"executor": "slurm", "account": "test", "partition": "gpu", **overrides}),
        passthrough=[],
        attached=False,
        env_vars={},
        startup_commands=None,
        force_squash=False,
    )

    experiment = experiment_factory.return_value.__enter__.return_value
    experiment.add.assert_called_once()
    executor = experiment.add.call_args.kwargs["executor"]
    expected_gpus = overrides.get("gpus_per_node", module.SPEC.resources.gpus_per_node)
    assert isinstance(executor, run.SlurmExecutor)
    assert executor.nodes == overrides.get("nodes", module.SPEC.resources.nodes)
    assert executor.gpus_per_node == expected_gpus
    assert executor.ntasks_per_node == overrides.get("ntasks_per_node", expected_gpus)
    experiment.run.assert_called_once_with(detach=True)
