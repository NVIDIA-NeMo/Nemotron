# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nemotron.steps.tokenizer_extension.init_embeddings.checkpoint import (
    normalize_untied_weight_metadata,
    save_pretrained,
)


class Model:
    def __init__(self, *, tied: bool = False) -> None:
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self._tied_weights_keys = ["lm_head.weight"]
        self.input = torch.nn.Embedding(4, 3)
        self.output = self.input if tied else torch.nn.Embedding(4, 3)
        self.saved_to = None

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output

    def save_pretrained(self, output_dir) -> None:
        self.saved_to = output_dir


def test_transformers_5_normalizes_legacy_metadata_for_proven_untied_weights() -> None:
    model = Model()

    assert normalize_untied_weight_metadata(model, "5.17.0") is True
    assert model._tied_weights_keys == {}


def test_transformers_4_keeps_the_list_metadata_it_expects() -> None:
    model = Model()

    assert normalize_untied_weight_metadata(model, "4.57.6") is False
    assert model._tied_weights_keys == ["lm_head.weight"]


@pytest.mark.parametrize(
    ("tied_config", "share_storage"),
    [(True, False), (False, True)],
)
def test_transformers_5_never_discards_real_tie_metadata(tied_config: bool, share_storage: bool) -> None:
    model = Model(tied=tied_config)
    if share_storage:
        model.config.tie_word_embeddings = False
        model.output = model.input

    with pytest.raises(RuntimeError, match="refusing"):
        normalize_untied_weight_metadata(model, "5.17.0")


def test_transformers_5_fails_closed_when_storage_cannot_be_checked() -> None:
    class Weight:
        @staticmethod
        def untyped_storage():
            raise RuntimeError("unavailable")

    model = Model()
    model.input = SimpleNamespace(weight=Weight())
    model.output = SimpleNamespace(weight=Weight())

    with pytest.raises(RuntimeError, match="Cannot verify"):
        normalize_untied_weight_metadata(model, "5.17.0")


def test_save_wrapper_delegates_after_the_guard(monkeypatch, tmp_path) -> None:
    model = Model()
    monkeypatch.setattr(
        "nemotron.steps.tokenizer_extension.init_embeddings.checkpoint.normalize_untied_weight_metadata",
        lambda _model: False,
    )

    save_pretrained(model, tmp_path)

    assert model.saved_to == tmp_path
