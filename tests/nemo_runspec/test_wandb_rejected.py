# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""wandb raises AuthenticationError for an unreachable server too, and telling
the user to re-login does not fix a network fault."""

from __future__ import annotations

import sys
import types

import pytest
from wandb.errors import AuthenticationError

from nemo_runspec.execution import _detect_wandb_api_key


def _stub_wandb(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    wandb = types.ModuleType("wandb")
    wandb.api = types.SimpleNamespace(api_key="k")

    def _api(timeout=None):
        raise error

    wandb.Api = _api
    monkeypatch.setitem(sys.modules, "wandb", wandb)


def test_unreachable_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_wandb(monkeypatch, AuthenticationError("Unable to connect to https://api.wandb.ai to verify API token."))
    assert _detect_wandb_api_key({}) == "k"


def test_rejected_key_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_wandb(monkeypatch, AuthenticationError("401 Unauthorized"))
    with pytest.raises(RuntimeError, match="authentication failed"):
        _detect_wandb_api_key({})
