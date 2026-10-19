# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Exercise plugin-shaped Parquet through the optional native AutoModel dataset."""

from __future__ import annotations

import json
import pickle
from dataclasses import fields
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError


@pytest.fixture
def adapter():
    backend = pytest.importorskip("nemo_automodel.components.datasets.llm.retrieval_dataset")
    if not hasattr(backend, "RetrievalDatasetConfig"):
        pytest.skip("Requires the recipe's pinned AutoModel dataset API")
    from nemotron.recipes._retrieval.data import NullableImageDatasetConfig

    return backend, NullableImageDatasetConfig


def _training_file(tmp_path: Path, documents: list[dict], corpus_class: str = "WikiSSNQDataset") -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    schema = pa.schema(
        [("docid", pa.string()), ("image_filename", pa.string()), ("text", pa.string()), ("image", pa.binary())]
    )
    rows = [dict(docid=str(i), image_filename=str(i), **doc) for i, doc in enumerate(documents)]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), corpus / "train.parquet")
    (corpus / "merlin_metadata.json").write_text(json.dumps({"class": corpus_class, "corpus_id": "source"}))
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {"path": "corpus"},
                "data": [
                    {
                        "question_id": "q1",
                        "question": "What is shown?",
                        "corpus_id": "source",
                        "pos_doc": [{"id": str(i)} for i in range(len(documents))],
                        "neg_doc": [],
                    }
                ],
            }
        )
    )
    return path


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 3), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("model_type", ["bi_encoder", "cross_encoder"])
@pytest.mark.parametrize("data_type", ["train", "eval"])
@pytest.mark.parametrize("modality", ["text", "image", "image_and_text"])
def test_native_dataset_preserves_present_and_absent_modalities(tmp_path, adapter, model_type, data_type, modality):
    backend, config_class = adapter
    text = None if modality == "image" else "Supplied source text"
    image = None if modality == "text" else _image_bytes()
    path = _training_file(tmp_path, [{"text": text, "image": image}])
    original_registry = backend.DATASETS.copy()
    dataset = config_class(
        data_dir_list=str(path), model_type=model_type, data_type=data_type, n_passages=1, use_text_in_document=True
    ).build()
    # Read through the actual lazy native transform, not just the wrapper.
    row = dataset[0]
    actual_text = row["doc_text"][0] if model_type == "bi_encoder" else row["doc_text"]
    actual_image = row["doc_image"][0] if model_type == "bi_encoder" else row["doc_image"]
    assert actual_text == (text or "")
    if image is None:
        assert actual_image == ""
    else:
        assert actual_image.size == (2, 3)
        assert actual_image.getpixel((0, 0)) == (255, 0, 0)
    assert backend.DATASETS == original_registry


def test_native_null_image_regression_and_positive_cycling(tmp_path, adapter):
    backend, config_class = adapter
    path = _training_file(
        tmp_path, [{"text": "Text only", "image": None}, {"text": "Caption", "image": _image_bytes()}]
    )
    kwargs = dict(data_dir_list=str(path), n_passages=1, use_text_in_document=True, cycle_positive_docs=True)
    with pytest.raises(AttributeError, match="convert"):
        backend.RetrievalDatasetConfig(**kwargs).build()[0]
    dataset = config_class(**kwargs).build()
    assert dataset[0]["doc_text"] == ["Text only"]
    dataset = pickle.loads(pickle.dumps(dataset))
    dataset.set_epoch(1)
    assert dataset[0]["doc_text"] == ["Caption"]
    assert dataset[0]["doc_image"][0].size == (2, 3)


@pytest.mark.parametrize("corpus_class", ["ColPaliDataset", "WikiSSNQDataset"])
def test_corrupt_binary_images_are_rejected_by_automodel(tmp_path, adapter, corpus_class):
    _, config_class = adapter
    path = _training_file(tmp_path, [{"text": "Source", "image": b"not an image"}], corpus_class)
    dataset = config_class(data_dir_list=str(path), n_passages=1).build()
    with pytest.raises(UnidentifiedImageError):
        dataset[0]


def test_adapter_keeps_native_typed_configuration(adapter):
    backend, config_class = adapter
    from nemo_automodel.components.datasets.loader import make_dataset_config

    config = make_dataset_config(config_class, {"n_passages": 4, "cycle_positive_docs": True})
    assert isinstance(config, backend.RetrievalDatasetConfig)
    assert fields(config) == fields(backend.RetrievalDatasetConfig)
    assert config.n_passages == 4
    assert config.cycle_positive_docs is True


def test_image_view_uses_native_colpali_corpus(tmp_path, adapter):
    _, config_class = adapter
    path = _training_file(tmp_path, [{"text": None, "image": _image_bytes()}], "ColPaliDataset")
    row = config_class(data_dir_list=str(path), n_passages=1).build()[0]
    assert row["doc_text"] == [""]
    assert row["doc_image"][0].getpixel((0, 0)) == (255, 0, 0)
