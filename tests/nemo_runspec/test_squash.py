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

"""Tests for nemo_runspec.squash."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nemo_runspec.squash import (
    build_salloc_args,
    container_to_sqsh_name,
    ensure_squashed_image,
    resolve_build_image,
    resolve_build_partition,
    resolve_build_time,
)


@dataclass
class _Result:
    ok: bool = True
    stdout: str = ""
    stderr: str = ""


class _FakeTunnel:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, cmd: str, hide: bool = False, warn: bool = False):
        self.commands.append(cmd)
        return _Result()


@pytest.mark.parametrize(
    ("container", "expected_name"),
    [
        (
            "docker://nvcr.io/nvidian/nemo:25.11-nano-v3.rc2",
            "nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh",
        ),
        (
            "dockerd://rayproject/ray:nightly-extra-py312-cpu",
            "rayproject_ray_nightly_extra_py312_cpu.sqsh",
        ),
        (
            "podman://quay.io/podman/stable:v5.3",
            "quay_io_podman_stable_v5_3.sqsh",
        ),
        (
            "docker-archive:///home/test/.cache/nemotron/containers/omni3-sft.tar",
            "omni3_sft_tar.sqsh",
        ),
        (
            "oci-archive:///home/test/.cache/nemotron/containers/omni3-rl.tar",
            "omni3_rl_tar.sqsh",
        ),
    ],
)
def test_container_to_sqsh_name_handles_supported_schemes(container: str, expected_name: str):
    assert container_to_sqsh_name(container) == expected_name


@pytest.mark.parametrize(
    ("container", "expected_source"),
    [
        ("docker://nvcr.io/nvidian/nemo:25.11-nano-v3.rc2", "docker://nvcr.io/nvidian/nemo:25.11-nano-v3.rc2"),
        ("dockerd://rayproject/ray:nightly-extra-py312-cpu", "dockerd://rayproject/ray:nightly-extra-py312-cpu"),
        ("podman://quay.io/podman/stable:v5.3", "podman://quay.io/podman/stable:v5.3"),
        (
            "docker-archive:///home/test/.cache/nemotron/containers/omni3-sft.tar",
            "docker-archive:///home/test/.cache/nemotron/containers/omni3-sft.tar",
        ),
        (
            "oci-archive:///home/test/.cache/nemotron/containers/omni3-rl.tar",
            "oci-archive:///home/test/.cache/nemotron/containers/omni3-rl.tar",
        ),
    ],
)
def test_ensure_squashed_image_passes_through_supported_schemes(
    container: str,
    expected_source: str,
):
    tunnel = _FakeTunnel()

    sqsh_path = ensure_squashed_image(
        tunnel,
        container,
        "/remote/jobs",
        {"account": "acct", "build_partition": "cpu", "build_time": "02:00:00"},
        force=True,
    )

    assert sqsh_path == f"/remote/jobs/{container_to_sqsh_name(container)}"
    assert "--partition=cpu" in tunnel.commands[-1]
    assert "--time=02:00:00" in tunnel.commands[-1]
    assert f"enroot import --output {sqsh_path} {expected_source}" in tunnel.commands[-1]


def test_ensure_squashed_image_defaults_to_docker_scheme_for_bare_images():
    tunnel = _FakeTunnel()

    sqsh_path = ensure_squashed_image(
        tunnel,
        "nvcr.io/nvidian/nemo:25.11-nano-v3.rc2",
        "/remote/jobs",
        {"partition": "batch", "time": "04:00:00"},
        force=True,
    )

    assert sqsh_path == "/remote/jobs/nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh"
    assert "--partition=batch" in tunnel.commands[-1]
    assert "--time=04:00:00" in tunnel.commands[-1]
    assert (
        "enroot import --output /remote/jobs/nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh "
        "docker://nvcr.io/nvidian/nemo:25.11-nano-v3.rc2"
    ) in tunnel.commands[-1]


class TestResolveBuildPartition:
    def test_build_partition_wins(self):
        env = {"build_partition": "cpu", "run_partition": "interactive", "partition": "batch"}
        assert resolve_build_partition(env) == "cpu"

    def test_falls_back_to_run_partition(self):
        env = {"run_partition": "interactive", "partition": "batch"}
        assert resolve_build_partition(env) == "interactive"

    def test_falls_back_to_partition(self):
        assert resolve_build_partition({"partition": "batch"}) == "batch"

    def test_none_when_nothing_set(self):
        assert resolve_build_partition({}) is None
        assert resolve_build_partition(None) is None


class TestResolveBuildTime:
    def test_build_time_wins(self):
        env = {"build_time": "02:00:00", "time": "08:00:00"}
        assert resolve_build_time(env) == "02:00:00"

    def test_falls_back_to_time(self):
        assert resolve_build_time({"time": "08:00:00"}) == "08:00:00"

    def test_default_applied(self):
        assert resolve_build_time({}) == "04:00:00"
        assert resolve_build_time(None, default="06:00:00") == "06:00:00"


class TestResolveBuildImage:
    def test_build_image_wins(self):
        env = {"build_image": "custom/podman:v1"}
        assert resolve_build_image(env, "quay.io/podman/stable:v5.3") == "custom/podman:v1"

    def test_default_used_when_unset(self):
        assert resolve_build_image({}, "quay.io/podman/stable:v5.3") == "quay.io/podman/stable:v5.3"
        assert resolve_build_image(None, "quay.io/podman/stable:v5.3") == "quay.io/podman/stable:v5.3"


class TestBuildSallocArgs:
    def test_minimal_config(self):
        args = build_salloc_args({})
        assert "--nodes=1" in args
        assert "--ntasks-per-node=1" in args
        assert "--time=04:00:00" in args

    def test_full_config(self):
        env = {
            "account": "dl-algo",
            "build_partition": "cpu",
            "build_time": "02:00:00",
            "gpus_per_node": 8,
        }
        args = build_salloc_args(env)
        assert "--account=dl-algo" in args
        assert "--partition=cpu" in args
        assert "--time=02:00:00" in args
        assert "--gpus-per-node=8" in args

    def test_include_gpus_false(self):
        args = build_salloc_args({"gpus_per_node": 8}, include_gpus=False)
        assert not any("gpus-per-node" in a for a in args)

    def test_build_partition_precedence_applied(self):
        # Regression: the three-layer precedence that used to be inlined
        # six times across the repo now lives in one helper.
        env = {
            "build_partition": "cpu",
            "run_partition": "interactive",
            "partition": "batch",
        }
        args = build_salloc_args(env)
        assert "--partition=cpu" in args
        assert "--partition=interactive" not in args
        assert "--partition=batch" not in args
