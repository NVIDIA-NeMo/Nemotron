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


class _FakeModelConfig:
    """Minimal stand-in for a Megatron-Bridge model provider."""

    def __init__(self, cp=1, tp=1, sequence_parallel=False):
        self.context_parallel_size = cp
        self.tensor_model_parallel_size = tp
        self.sequence_parallel = sequence_parallel


def test_derive_pad_seq_to_mult_defaults_to_one():
    from nemotron.kit.recipe_loader import derive_pad_seq_to_mult

    assert derive_pad_seq_to_mult(None, None) == 1
    assert derive_pad_seq_to_mult(None, _FakeModelConfig()) == 1


def test_derive_pad_seq_to_mult_honors_explicit_value():
    from nemotron.kit.recipe_loader import derive_pad_seq_to_mult

    assert derive_pad_seq_to_mult(16, _FakeModelConfig()) == 16
    # Explicit value is never weakened, only strengthened via lcm
    assert derive_pad_seq_to_mult(16, _FakeModelConfig(cp=2)) == 16
    assert derive_pad_seq_to_mult(3, _FakeModelConfig(cp=2)) == 12


def test_derive_pad_seq_to_mult_context_parallel():
    from nemotron.kit.recipe_loader import derive_pad_seq_to_mult

    # CP requires 2 * CP (load-balanced CP halves)
    assert derive_pad_seq_to_mult(None, _FakeModelConfig(cp=2)) == 4
    assert derive_pad_seq_to_mult(None, _FakeModelConfig(cp=4)) == 8


def test_derive_pad_seq_to_mult_sequence_parallel_adds_tp_constraint():
    from nemotron.kit.recipe_loader import derive_pad_seq_to_mult

    # CP=2, TP=3, SP=true -> lcm(2*CP, CP*TP) = lcm(4, 6) = 12
    assert derive_pad_seq_to_mult(None, _FakeModelConfig(cp=2, tp=3, sequence_parallel=True)) == 12
    # SP without TP>1 adds nothing beyond the CP constraint
    assert derive_pad_seq_to_mult(None, _FakeModelConfig(cp=2, tp=1, sequence_parallel=True)) == 4
    # TP>1 without SP adds nothing beyond the CP constraint
    assert derive_pad_seq_to_mult(None, _FakeModelConfig(cp=2, tp=4, sequence_parallel=False)) == 4
    # Bridge parity: explicit 5 with CP=1/TP=2/SP -> lcm(5, 2) = 10
    assert derive_pad_seq_to_mult(5, _FakeModelConfig(cp=1, tp=2, sequence_parallel=True)) == 10


def test_derive_pad_seq_to_mult_includes_eval_context_parallel():
    from types import SimpleNamespace

    from nemotron.kit.recipe_loader import derive_pad_seq_to_mult

    # Bridge parity: CP=2/TP=3/SP with eval CP=4 -> lcm(4, 8, 6, 12) = 24
    dist = SimpleNamespace(eval_context_parallel_size=4)
    assert derive_pad_seq_to_mult(None, _FakeModelConfig(cp=2, tp=3, sequence_parallel=True), dist) == 24
    # eval CP unset behaves like train-only topology
    dist_none = SimpleNamespace(eval_context_parallel_size=None)
    assert derive_pad_seq_to_mult(None, _FakeModelConfig(cp=2, tp=3, sequence_parallel=True), dist_none) == 12
