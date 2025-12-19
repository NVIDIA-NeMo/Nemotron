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

from omegaconf import OmegaConf

from nemotron.kit.recipe_loader import extract_recipe_config


def test_extract_recipe_config_defaults_when_missing_recipe():
    cfg = OmegaConf.create({"x": 1})
    target, kwargs = extract_recipe_config(cfg, default_target="a.b.c")
    assert target == "a.b.c"
    assert kwargs == {}


def test_extract_recipe_config_reads_target_and_kwargs():
    cfg = OmegaConf.create(
        {
            "recipe": {
                "_target_": "m.n.func",
                "alpha": 1,
                "beta": "x",
            }
        }
    )
    target, kwargs = extract_recipe_config(cfg, default_target="a.b.c")
    assert target == "m.n.func"
    assert kwargs == {"alpha": 1, "beta": "x"}
