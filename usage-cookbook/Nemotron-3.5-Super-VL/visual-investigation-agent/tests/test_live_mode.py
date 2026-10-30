# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live mode validates endpoint capabilities before it starts the investigation."""

import pytest

import investigator
from investigator import endpoint
from investigator.runs import load_recording


class _DisplayHandle:
    def __init__(self):
        self.updates = []

    def update(self, value):
        self.updates.append(value)


def test_live_entrypoint_validates_then_uses_the_same_run_loader(monkeypatch):
    recording = load_recording("model-switch-visual")
    handle = _DisplayHandle()
    events = []

    monkeypatch.setattr(endpoint, "check", lambda: [
        ("Text", True, "ready"),
        ("Image", True, "read 47"),
        ("Tool call", True, "called echo"),
    ])
    monkeypatch.setattr("IPython.display.display", lambda *args, **kwargs: handle)

    def fake_investigate(scenario, on_step):
        events.append(scenario)
        return recording.folder

    monkeypatch.setattr(investigator, "investigate", fake_investigate)
    live = investigator.run_live(recording.scenario)

    assert events == [recording.scenario]
    assert live.report == recording.report
    assert handle.updates


def test_live_entrypoint_stops_when_capability_validation_fails(monkeypatch):
    monkeypatch.setattr(endpoint, "check", lambda: [
        ("Image", False, "image input unsupported"),
    ])
    monkeypatch.setattr(
        investigator,
        "investigate",
        lambda *args, **kwargs: pytest.fail("investigation started after failed endpoint validation"),
    )

    with pytest.raises(RuntimeError, match="Live endpoint validation failed.*Image"):
        investigator.run_live("model-switch-visual")
