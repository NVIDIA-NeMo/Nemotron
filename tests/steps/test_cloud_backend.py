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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import nemotron.cli.commands.step.backends.cloud as cloud_mod
from nemotron.cli.commands.step.backends.base import JobContext
from nemotron.cli.commands.step.backends.cloud import CloudBackend


def _ctx(step_id: str, *, launch: str = "ray") -> JobContext:
    return JobContext(
        step_id=step_id,
        script_path=Path("/repo/src/nemotron/steps") / step_id / "step.py",
        train_path=Path("/tmp/train.yaml"),
        spec=SimpleNamespace(
            run=SimpleNamespace(
                launch=launch,
                cmd="python {script} --config {config}",
            ),
            image="test-image",
            resources=SimpleNamespace(nodes=1, gpus_per_node=0),
        ),
        env={"executor": "lepton"},
        env_vars={},
        passthrough=[],
        startup_commands=[],
        attached=False,
        force_squash=False,
    )


def test_prep_ray_step_uses_inline_cloud_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_execute_cloud(*_args, **kwargs):
        calls.append(("inline", kwargs))

    def fail_execute_cloud_ray(*_args, **_kwargs):
        raise AssertionError("prep steps must not use cloud RayCluster submission")

    monkeypatch.setattr(cloud_mod, "execute_cloud", fake_execute_cloud)
    monkeypatch.setattr(cloud_mod, "execute_cloud_ray", fail_execute_cloud_ray)

    CloudBackend().submit(_ctx("prep/sft_packing"))

    assert len(calls) == 1
    assert calls[0][0] == "inline"
    assert calls[0][1]["launch"] is None


def test_non_prep_ray_step_keeps_raycluster_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fail_execute_cloud(*_args, **_kwargs):
        raise AssertionError("ray training steps should use cloud RayCluster submission")

    def fake_execute_cloud_ray(*_args, **kwargs):
        calls.append(("ray", kwargs))

    monkeypatch.setattr(cloud_mod, "execute_cloud", fail_execute_cloud)
    monkeypatch.setattr(cloud_mod, "execute_cloud_ray", fake_execute_cloud_ray)

    CloudBackend().submit(_ctx("rl/nemo_rl/dpo"))

    assert len(calls) == 1
    assert calls[0][0] == "ray"
