# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The code is the tested agent: model-facing hashes and world contents match each frozen version's manifest."""

import json

import pytest

import world
from investigator import check_config
from investigator.config import DEFAULT, VERSIONS
from investigator.version import manifest_file


@pytest.mark.parametrize("version", VERSIONS)
def test_model_facing_content_matches_the_frozen_manifest(version):
    assert check_config(version) == {item: True for item in check_config(version)}


def test_the_default_is_a_frozen_version():
    assert DEFAULT["name"] in VERSIONS and json.loads(manifest_file().read_text())["name"] == DEFAULT["name"]


@pytest.mark.parametrize("version", VERSIONS)
def test_world_contents_match_the_frozen_manifest(version):
    frozen = json.loads(manifest_file(version).read_text())["world_content_sha256"]
    for scenario, digest in frozen.items():
        world.ensure(scenario)
        assert world.content_sha256(scenario) == digest, scenario


def test_recordings_were_made_with_a_frozen_configuration():
    from investigator import load_recording

    for name in ("model-switch-visual",):
        manifest = load_recording(name).manifest
        assert manifest["agent_version"] in VERSIONS and manifest["matches_frozen"], name
