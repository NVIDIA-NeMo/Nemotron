# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe controls survive AutoModel's checkpoint-encoder API migration."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nemotron.recipes.embed.mining_encoder import CheckpointMiningEncoderConfig


@pytest.mark.parametrize("use_images,use_text", [(True, True), (True, False), (False, True), (False, False)])
def test_native_encoder_preserves_processor_and_modality_controls(monkeypatch, use_images, use_text):
    captured = {}

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured.update(path=path, options=kwargs)
            return cls()

    class Encoder:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def encode_documents(self, documents, *, batch_size):
            return documents, batch_size

    automodel = ModuleType("nemo_automodel")
    automodel.NeMoAutoModelBiEncoder = SimpleNamespace()
    native = ModuleType("nemo_automodel.recipes.retrieval.mining_encoder")
    native.CheckpointMiningEncoder = Encoder
    processor_module = ModuleType("smoke_processor")
    processor_module.Processor = Processor
    monkeypatch.setitem(sys.modules, "nemo_automodel", automodel)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, processor_module.__name__, processor_module)
    model = SimpleNamespace(
        model=SimpleNamespace(retrieval_processor_target="smoke_processor.Processor"),
        config=SimpleNamespace(_commit_hash="revision", name_or_path="checkpoint"),
    )
    config = CheckpointMiningEncoderConfig(
        q_max_length=64,
        p_max_length=256,
        query_prefix="",
        image_longest_edge=224,
        use_images=use_images,
        use_text_in_document=use_text,
    )
    encoder = config.build(model=model, device="cpu")
    assert captured == {
        "path": "checkpoint",
        "options": {
            "q_max_length": 64,
            "p_max_length": 256,
            "query_prefix": "",
            "image_longest_edge": 224,
            "revision": "revision",
        },
    }
    image = object()
    documents = [{"image": image, "text": "body", "title": "title"}, {"text": "text only", "image": ""}]
    result, batch_size = encoder.encode_documents(documents, batch_size=2)
    assert batch_size == 2
    assert result[0].get("image") is (image if use_images else None)
    assert result[0]["text"] == ("" if use_images and not use_text else "body")
    assert result[1]["text"] == "text only"
    assert documents[0] == {"image": image, "text": "body", "title": "title"}
