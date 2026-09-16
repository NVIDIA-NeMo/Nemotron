# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher mode returns on submission, so the exit code is not the result."""

from __future__ import annotations

import tomllib
from pathlib import Path

STEP = Path(__file__).resolve().parents[2] / "src" / "nemotron" / "steps" / "eval" / "model_eval"


def test_submission_is_announced() -> None:
    assert "nemotron_step_status: SUBMITTED" in (STEP / "runtime.py").read_text(encoding="utf-8")


def test_mode_documents_the_caveat() -> None:
    manifest = tomllib.loads((STEP / "step.toml").read_text(encoding="utf-8"))
    mode = next(p for p in manifest["parameters"] if p["name"] == "mode")
    assert "never gate CI on that exit code" in mode["description"]
