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

"""A traceback escaping a CLI command must not print credentials.

Typer renders every frame's locals into an unhandled traceback, and the
submission path carries HF_TOKEN and WANDB_API_KEY through locals. Keeping a
credential out of any one local does not fix that; the rendering does.

Runs in a subprocess because Typer prints through its own console, which an
in-process capture does not intercept.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HF_CANARY = "hf-CANARY-aaaaaaaaaaaaaaaaaaaaaaaa"
WB_CANARY = "wandb-CANARY-bbbbbbbbbbbbbbbbbbbbbbbb"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Raises inside _detect_wandb_api_key with both credentials live, which is the
# shape of a real submission failure.
PROBE = '''
import os, sys, types

HF, WB = os.environ["HF_CANARY"], os.environ["WB_CANARY"]
os.environ["HF_TOKEN"] = HF

wandb = types.ModuleType("wandb")


class _Api:
    def __init__(self, timeout=None):
        pass

    @property
    def viewer(self):
        raise RuntimeError("401 Unauthorized")


wandb.api = types.SimpleNamespace(api_key=WB)
wandb.Api = _Api
sys.modules["wandb"] = wandb

hub = types.ModuleType("huggingface_hub")
hub.get_token = lambda: HF
sys.modules["huggingface_hub"] = hub

from nemo_runspec.execution import build_env_vars
from nemotron.cli.bin.nemotron import app


@app.command("leakprobe")
def leakprobe():
    build_env_vars(
        {"run": {"wandb": {"project": "p", "entity": "e"}}},
        {"remote_job_dir": "/tmp/x"},
    )


sys.argv = ["nemotron", "leakprobe"]
app()
'''


@pytest.fixture(scope="module")
def traceback_output() -> str:
    env = dict(
        os.environ,
        HF_CANARY=HF_CANARY,
        WB_CANARY=WB_CANARY,
        COLUMNS="400",  # narrow terminals wrap values and hide them from a count
    )
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return proc.stdout + proc.stderr


def test_probe_actually_raised(traceback_output: str) -> None:
    """Guard against a vacuous pass.

    If the command failed earlier than the credential path -- a missing profile,
    an import error -- nothing would be rendered and the assertions below would
    pass on a leaking build.
    """
    assert "401 Unauthorized" in traceback_output, traceback_output[-2000:]


def test_hf_token_absent(traceback_output: str) -> None:
    assert traceback_output.count(HF_CANARY) == 0


def test_wandb_key_absent(traceback_output: str) -> None:
    assert traceback_output.count(WB_CANARY) == 0
