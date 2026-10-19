# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replaying a recording's model replies through the harness reproduces the run: same requests, same results."""

import json

import pytest

from investigator import config as agent_config
from investigator import sandbox
from investigator.harness import investigate
from investigator.runs import load_recording, load_run
from recorded_model import RecordedModel, seed_images

RECORDINGS = ["model-switch-visual"]


@pytest.mark.parametrize("name", RECORDINGS)
def test_harness_reproduces_recording(name, tmp_path):
    recording = load_recording(name)
    seed_images(recording)
    model = RecordedModel(recording)
    config = agent_config.for_version(recording.manifest["agent_version"])
    replay = load_run(investigate(recording.scenario, run_dir=tmp_path / name, chat=model, config=config))

    assert model.mismatches == [], f"requests differ from the recorded conversation at model calls {model.mismatches}"
    assert model.calls == len(model.replies)
    for recorded, replayed in zip(recording.tool_log, replay.tool_log, strict=True):
        assert (recorded["name"], recorded["arguments"]) == (replayed["name"], replayed["arguments"])
        if recorded["name"] == "run_python" and sandbox.available() is None:
            continue  # no sandbox here, so the code was not run
        assert json.dumps(recorded["result"], sort_keys=True) == json.dumps(replayed["result"], sort_keys=True), recorded["name"]
    assert replay.report == recording.report
    assert replay.proposals == recording.proposals
