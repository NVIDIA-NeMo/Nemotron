# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expand keeps stock ByteLevel identities, matching and artifact portability."""

from __future__ import annotations

import ast
import copy
import json
import pickle
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402


def _helpers():
    path = Path(__file__).resolve().parents[3] / "src/nemotron/steps/tokenizer_extension/extend/extension.py"
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in ("_expand_pattern", "_build_expand")
    ]
    assert len(nodes) == 2
    namespace = {
        "json": json,
        "copy": copy,
        "re": re,
        "AddedToken": AddedToken,
        "Regex": Regex,
        "Tokenizer": Tokenizer,
        "pre_tokenizers": pre_tokenizers,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_build_expand"], namespace["_expand_pattern"]


def _base(*, ignore_merges: bool = True, reserved: bool = False) -> PreTrainedTokenizerFast:
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], ignore_merges=ignore_merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    if reserved:
        backend.add_special_tokens([AddedToken("<special>", special=True)])
        backend.add_tokens([AddedToken("raw-added", normalized=False)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)


def _trained(surfaces: list[str]) -> SimpleNamespace:
    encoder = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    keys = ["".join(piece for piece, _ in encoder.pre_tokenize_str(surface)) for surface in surfaces]
    return SimpleNamespace(
        get_vocab=lambda: {key: rank for rank, key in enumerate(keys)},
        convert_tokens_to_string=lambda tokens: decoders.ByteLevel().decode(tokens),
    )


def _decode(tok: PreTrainedTokenizerFast, text: str) -> str:
    return tok.decode(
        tok.encode(text, add_special_tokens=False), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


@pytest.mark.parametrize(
    "surface",
    [
        "ă",
        "bào",
        "café",
        "cafe\u0301",
        "�",
        "你好",
        "नमस्ते",
        "العربية",
        "עברית",
        "ภาษาไทย",
        "한글",
        "👨‍👩‍👧",
        "  bào",
        "a\tb",
        "a\nb",
        "a+b(c)?[x].*",
        "a\x00\x03",
    ],
)
def test_unicode_whitespace_and_literal_regex_roundtrip(surface: str, tmp_path: Path) -> None:
    base = _base()
    before = base.backend_tokenizer.to_str()
    result, _, count = _helpers()[0](base, _trained([surface]), 1)
    assert count == 1
    assert result.encode(surface, add_special_tokens=False) == [len(base)]
    probes = [surface, "A\t" + surface + "  " + surface + "\nB", "ă Äĥ café �\x03", "  \t\n"]
    for tok in [result, pickle.loads(pickle.dumps(result))]:
        assert all(_decode(tok, text) == text for text in probes)
    result.save_pretrained(tmp_path)
    loaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True, trust_remote_code=False)
    assert all(_decode(loaded, text) == text for text in probes)
    assert base.backend_tokenizer.to_str() == before


def test_old_ids_decoder_merges_normalizer_and_specials_survive() -> None:
    base = _base(reserved=True)
    before = json.loads(base.backend_tokenizer.to_str())
    result, _, count = _helpers()[0](base, _trained(["ă", "bào", "你好"]), 3)
    after = json.loads(result.backend_tokenizer.to_str())
    assert count == 3
    assert all(result.get_vocab()[key] == idx for key, idx in base.get_vocab().items())
    for field in ("decoder", "normalizer", "added_tokens", "post_processor"):
        assert after[field] == before[field]
    assert after["model"]["merges"] == before["model"]["merges"]
    assert result.encode("<special>", add_special_tokens=False) == base.encode("<special>", add_special_tokens=False)
    rng = random.Random(17)
    for _ in range(250):
        ids = [rng.randrange(len(base)) for _ in range(rng.randrange(1, 20))]
        assert result.decode(ids, skip_special_tokens=False) == base.decode(ids, skip_special_tokens=False)
    reloaded = Tokenizer.from_str(result.backend_tokenizer.to_str())
    assert reloaded.get_vocab() == result.get_vocab()
