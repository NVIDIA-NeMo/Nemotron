# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Exercise modality preservation through the public training dataset adapter."""

from __future__ import annotations

import sys
from functools import partial
from io import BytesIO
from types import ModuleType
from typing import Any, ClassVar

import pytest
from PIL import Image, UnidentifiedImageError

from nemotron.recipes.retrieval_vl import DataDesignerRetrievalDatasetConfig


class SourceCorpus:
    """Stand in for the optional backend's corpus storage, not its adapter."""

    document: ClassVar[dict[str, Any]] = {}

    def get_document_by_id(self, identifier: str) -> dict[str, Any]:
        """Return an isolated source record for the adapter to consume."""
        return self.document.copy()


def materialize_document(registry: dict, corpus_class: str, **kwargs: Any) -> dict[str, Any]:
    """Ask the backend-selected corpus adapter for the original source unit."""
    return registry[corpus_class]().get_document_by_id("unit")


def load_document(document: dict[str, Any], monkeypatch: pytest.MonkeyPatch, corpus_class: str) -> dict[str, Any]:
    """Build the real recipe adapter against a minimal optional-backend boundary."""
    backend = ModuleType("nemo_automodel.components.datasets.llm.retrieval_dataset")
    backend.DATASETS = {name: SourceCorpus for name in ("ColPaliDataset", "WikiSSNQDataset")}
    backend.ColPaliDataset = backend.WikiSSNQDataset = SourceCorpus
    backend.make_retrieval_dataset = partial(materialize_document, backend.DATASETS, corpus_class)
    package = ModuleType("nemo_automodel.components.datasets.llm")
    package.retrieval_dataset = backend
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(SourceCorpus, "document", document)
    result = DataDesignerRetrievalDatasetConfig().build()
    assert all(value is SourceCorpus for value in backend.DATASETS.values())
    return result


@pytest.mark.parametrize("corpus_class", ["ColPaliDataset", "WikiSSNQDataset"])
@pytest.mark.parametrize("image_fields", [{}, {"image": None}, {"image": ""}, {"image": {"bytes": None}}])
def test_text_only_source_has_no_fabricated_image(corpus_class, image_fields, monkeypatch) -> None:
    document = load_document({"text": "Supplied source text", **image_fields}, monkeypatch, corpus_class)
    assert document["text"] == "Supplied source text"
    assert document["image"] == ""


@pytest.mark.parametrize("text_fields", [{}, {"text": None}, {"text": ""}])
def test_image_only_source_has_no_fabricated_text(text_fields, monkeypatch) -> None:
    image = Image.new("RGB", (2, 3), color="red")
    document = load_document({"image": image, **text_fields}, monkeypatch, "WikiSSNQDataset")
    assert document["image"] is image
    assert document["text"] == ""


@pytest.mark.parametrize("representation", [bytes, bytearray, memoryview, "mapping"])
def test_binary_images_preserve_source_pixels_and_text(representation, monkeypatch) -> None:
    image = Image.new("RGB", (2, 3), color="red")
    stream = BytesIO()
    image.save(stream, format="PNG")
    value = {"bytes": stream.getvalue()} if representation == "mapping" else representation(stream.getvalue())
    document = load_document({"image": value, "text": "Caption"}, monkeypatch, "WikiSSNQDataset")
    assert document["image"].tobytes() == image.tobytes()
    assert document["image"].size == image.size
    assert document["text"] == "Caption"


@pytest.mark.parametrize("image_bytes", [b"", b"not an image"])
def test_corrupt_image_is_not_silently_dropped(image_bytes, monkeypatch) -> None:
    with pytest.raises(UnidentifiedImageError):
        load_document({"image": image_bytes, "text": "Source"}, monkeypatch, "WikiSSNQDataset")


@pytest.mark.parametrize("image_fields", [{}, {"path": "page.png"}, {"bytes": None, "path": "page.png"}])
def test_path_only_or_malformed_image_is_not_silently_dropped(image_fields, monkeypatch) -> None:
    with pytest.raises(ValueError, match="embedded bytes"):
        load_document({"image": image_fields, "text": "Source"}, monkeypatch, "WikiSSNQDataset")
