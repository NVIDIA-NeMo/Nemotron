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

import importlib
import sys
import types


def test_patch_wandb_init_for_lineage_registers_artifacts_and_tags(monkeypatch):
    import nemotron.kit.wandb as wb

    wb = importlib.reload(wb)

    used: list[str] = []

    class FakeRun:
        def __init__(self):
            self.tags = []

        def use_artifact(self, qname: str):
            used.append(qname)

    fake_run = FakeRun()

    def fake_init(*args, **kwargs):
        fake_wandb.run = fake_run
        return fake_run

    fake_wandb = types.SimpleNamespace(run=None, init=fake_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wb.patch_wandb_init_for_lineage(
        artifact_qualified_names=["ent/proj/DataBlendsArtifact-pretrain:v5"],
        tags=["pretrain"],
    )

    fake_wandb.init()

    assert used == ["ent/proj/DataBlendsArtifact-pretrain:v5"]
    assert "pretrain" in fake_run.tags
