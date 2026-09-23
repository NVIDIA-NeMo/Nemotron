# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The splice must add exactly `extension_size` tokens.

Fertility and BPB are only comparable between arms built at the same budget, so
an arm that over- or under-fills silently invalidates the comparison it exists
to support. Both directions have regressed before, hence a test for each.

`continued_bpe` imports heavy optional dependencies at module scope, so the
three pure functions under test are loaded directly from the source file.
"""

from __future__ import annotations

import ast
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402

PURE_FUNCTIONS = ("_rank_merges", "_apply_merges", "_apply_bpe_extension_backend")


def _load_splice() -> Callable[..., Any]:
    source = (
        Path(__file__).resolve().parents[3] / "src/nemotron/steps/tokenizer_extension/extend/continued_bpe.py"
    ).read_text()
    tree = ast.parse(source)
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in PURE_FUNCTIONS]
    assert len(wanted) == len(PURE_FUNCTIONS), "continued_bpe no longer defines the splice helpers"
    namespace: dict[str, Any] = {"Tokenizer": Tokenizer, "json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[])), "<splice>", "exec"), namespace)
    return namespace["_apply_bpe_extension_backend"]


def _bpe(corpus: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False))
    return tokenizer


@pytest.fixture(scope="module")
def corpus_fixture() -> tuple[Tokenizer, dict[str, int], list[str], list[str]]:
    """A base tokenizer plus novel candidates, in trained order and shuffled.

    The base is alphabet-only so that candidates genuinely need new merge rules;
    the shuffled ordering breaks the BPE guarantee that a token's constituent
    pieces precede it, which is what makes a candidate cost more than one row.
    """
    rng = random.Random(2)
    base = _bpe(["a b c d e f g h"], vocab_size=8)
    corpus = [
        " ".join("".join(rng.choice("abcdefgh") for _ in range(rng.randint(6, 12))) for _ in range(40))
        for _ in range(600)
    ]
    trained = _bpe(corpus, vocab_size=1500)
    raw = json.loads(trained.to_str())["model"]["merges"]
    merges = raw if isinstance(raw[0], str) else [f"{a} {b}" for a, b in raw]
    novel = [t for t in trained.get_vocab() if t not in base.get_vocab()]
    in_order = sorted(novel, key=lambda t: trained.get_vocab()[t])
    shuffled = list(novel)
    rng.shuffle(shuffled)
    return base, trained.get_vocab(), merges, [in_order, shuffled]


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 10, 30, 100, 300])
def test_splice_adds_exactly_the_budget_in_trained_order(corpus_fixture, budget):
    base, _, merges, (in_order, _) = corpus_fixture
    splice = _load_splice()
    candidates = {tok: i for i, tok in enumerate(in_order)}
    result = splice(base, candidates, merges, budget, False)
    assert result.get_vocab_size() - base.get_vocab_size() == budget


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 10, 30, 100, 300])
def test_splice_adds_exactly_the_budget_when_dependencies_are_out_of_order(corpus_fixture, budget):
    """Out-of-order candidates make a token cost more than one row.

    Guards both regressions at once: the budget check used to sit only between
    tokens (overshoot), and then stopped at the first token that did not fit
    rather than skipping it (underfill).
    """
    base, _, merges, (_, shuffled) = corpus_fixture
    splice = _load_splice()
    candidates = {tok: i for i, tok in enumerate(shuffled)}
    result = splice(base, candidates, merges, budget, False)
    assert result.get_vocab_size() - base.get_vocab_size() == budget


def test_every_spliced_token_is_reachable(corpus_fixture):
    """A token added without its merge chain would sit in the vocab unemittable."""
    base, _, merges, (_, shuffled) = corpus_fixture
    splice = _load_splice()
    candidates = {tok: i for i, tok in enumerate(shuffled)}
    result = splice(base, candidates, merges, 50, False)
    model = json.loads(result.to_str())["model"]
    vocab, raw = model["vocab"], model["merges"]
    pairs = {tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in raw}
    produced = {left + right for left, right in pairs}
    for token in vocab:
        if len(token) > 1:
            assert token in produced, f"{token!r} is in the vocab but no merge rule produces it"


def test_budget_larger_than_candidate_pool_is_capped(corpus_fixture):
    """Asking for more than exists must not loop or invent tokens."""
    base, _, merges, (in_order, _) = corpus_fixture
    splice = _load_splice()
    candidates = {tok: i for i, tok in enumerate(in_order)}
    result = splice(base, candidates, merges, len(in_order) + 5_000, False)
    added = result.get_vocab_size() - base.get_vocab_size()
    assert 0 < added <= len(in_order)
