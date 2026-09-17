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

"""Contracts shared by multimodal retrieval recipes."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

EXPECTED_CONSUMER_CONTRACT = "nemotron-vl-retrieval-v2"
ViewName = Literal["text", "image", "text_image"]


class RetrievalVLBundleError(ValueError):
    """Raised when a multimodal retrieval bundle violates its handoff contract."""


@dataclass(frozen=True)
class VLBundlePaths:
    """Resolved paths for one view in a DataDesigner multimodal bundle."""

    root: Path
    manifest: Path
    view: ViewName
    train: Path
    validation: Path
    train_corpus: Path
    validation_corpus: Path
    evaluation_corpus: Path
    synthetic_eval: Path
    gold_eval: Path | None
    local_artifacts_verified: bool = False


def _decode_document_image(document: dict[str, Any]) -> dict[str, Any]:
    """Decode image cells and preserve absent modalities for AutoModel.

    AutoModel represents absent text and images with empty strings, not nulls.
    Present text and decoded images remain unchanged; binary images are detached
    from their input buffer before reaching the training dataset.
    """
    from io import BytesIO

    from PIL import Image

    if document.get("text") is None:
        document["text"] = ""
    value = document.get("image")
    if isinstance(value, dict):
        if "bytes" not in value or (value["bytes"] is None and value.get("path")):
            raise ValueError("Image records require embedded bytes; path-only image references are unsupported")
        value = value.get("bytes")
    if value is None:
        document["image"] = ""
    elif isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(value))) as image:
            document["image"] = image.copy()
    return document


@dataclass
class DataDesignerRetrievalDatasetConfig:
    """AutoModel dataset config for DataDesignerPlugins binary-image Parquet."""

    data_dir_list: Any = None
    model_type: str = "bi_encoder"
    data_type: str = "train"
    n_passages: int = 5
    eval_negative_size: int | None = None
    seed: int = 42
    do_shuffle: bool = False
    max_train_samples: int | None = None
    train_data_select_offset: int = 0
    use_dataset_instruction: bool = False
    cycle_positive_docs: bool = False
    use_text_in_document: bool = False

    def build(self) -> Any:
        """Build through AutoModel while decoding plugin-emitted image bytes."""
        from nemo_automodel.components.datasets.llm import retrieval_dataset

        class DecodedColPaliDataset(retrieval_dataset.ColPaliDataset):
            def get_document_by_id(self, id: str) -> dict[str, Any]:
                return _decode_document_image(super().get_document_by_id(id))

        class DecodedWikiSSNQDataset(retrieval_dataset.WikiSSNQDataset):
            def get_document_by_id(self, id: str) -> dict[str, Any]:
                return _decode_document_image(super().get_document_by_id(id))

        replacements = {
            "ColPaliDataset": DecodedColPaliDataset,
            "WikiSSNQDataset": DecodedWikiSSNQDataset,
        }
        originals = {name: retrieval_dataset.DATASETS[name] for name in replacements}
        retrieval_dataset.DATASETS.update(replacements)
        try:
            return retrieval_dataset.make_retrieval_dataset(
                data_dir_list=self.data_dir_list,
                model_type=self.model_type,
                data_type=self.data_type,
                n_passages=self.n_passages,
                eval_negative_size=self.eval_negative_size,
                seed=self.seed,
                do_shuffle=self.do_shuffle,
                max_train_samples=self.max_train_samples,
                train_data_select_offset=self.train_data_select_offset,
                use_dataset_instruction=self.use_dataset_instruction,
                cycle_positive_docs=self.cycle_positive_docs,
                use_text_in_document=self.use_text_in_document,
            )
        finally:
            retrieval_dataset.DATASETS.update(originals)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalVLBundleError(f"Unable to read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise RetrievalVLBundleError(f"Expected a JSON object in {path}")
    return value


def _contained_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or ".." in Path(relative_path).parts:
        raise RetrievalVLBundleError(f"Invalid bundle artifact path: {relative_path!r}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RetrievalVLBundleError(f"Bundle artifact escapes its root: {relative_path}") from error
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect_vl_bundle(
    manifest_path: Path,
    *,
    view: ViewName = "text_image",
    require_local_artifacts: bool = True,
    verify_checksums: bool = False,
) -> VLBundlePaths:
    """Validate and resolve a DataDesignerPlugins multimodal retrieval bundle."""
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = _load_json(manifest_path)

    if view not in ("text", "image", "text_image"):
        raise RetrievalVLBundleError(f"Unsupported retrieval view: {view!r}")
    if manifest.get("schema_version") == 2:
        from nemotron.recipes.retrieval_bundle_v2 import inspect_unified_bundle

        return inspect_unified_bundle(
            manifest_path,
            manifest,
            view=view,
            require_local_artifacts=require_local_artifacts,
            verify_checksums=verify_checksums,
        )

    expected_values = {
        "schema_version": 3,
        "status": "completed",
        "portable": True,
        "consumer_contract_identity": EXPECTED_CONSUMER_CONTRACT,
        "consumer_contract_verified": True,
        "relevance_annotation_scope": "known_positives_only",
        "negative_mining_performed": False,
        "unlisted_document_disposition": "unjudged",
        "relevance_judgements_complete": False,
    }
    for field, expected in expected_values.items():
        actual = manifest.get(field)
        if actual != expected:
            raise RetrievalVLBundleError(f"Unsupported bundle contract: {field} must be {expected!r}, got {actual!r}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RetrievalVLBundleError("Bundle manifest must contain an artifacts list")

    records: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            raise RetrievalVLBundleError("Every artifact must have a relative_path")
        relative_path = record["relative_path"]
        if Path(relative_path).is_absolute():
            raise RetrievalVLBundleError(f"Invalid portable artifact path: {relative_path}")
        _contained_path(root, relative_path)
        if relative_path in records:
            raise RetrievalVLBundleError(f"Duplicate artifact path in manifest: {relative_path}")
        records[relative_path] = record

    required_files = [
        f"views/{view}/train.json",
        f"views/{view}/validation.json",
        f"synthetic_eval/{view}/queries.jsonl",
        f"synthetic_eval/{view}/corpus.jsonl",
        f"synthetic_eval/{view}/qrels/test.tsv",
    ]
    gold_eval_available = manifest.get("gold_evaluation_status") == "available"
    if gold_eval_available and manifest.get("gold_eval_integrity_verified") is not True:
        raise RetrievalVLBundleError("Available gold evaluation must have verified integrity")
    if gold_eval_available:
        required_files.extend(
            [
                f"gold_eval/{view}/queries.jsonl",
                f"gold_eval/{view}/corpus.jsonl",
                f"gold_eval/{view}/qrels/test.tsv",
            ]
        )

    missing_records = [relative_path for relative_path in required_files if relative_path not in records]
    if missing_records:
        raise RetrievalVLBundleError(f"Manifest is missing required {view} artifacts: {missing_records}")

    required_paths = [_contained_path(root, relative_path) for relative_path in required_files]
    corpus_paths = [
        _contained_path(root, f"views/{view}/corpus/train"),
        _contained_path(root, f"views/{view}/corpus/validation"),
        _contained_path(root, f"views/{view}/corpus/evaluation"),
    ]
    if require_local_artifacts:
        missing_paths = [path for path in [*required_paths, *corpus_paths] if not path.exists()]
        if missing_paths:
            raise RetrievalVLBundleError(f"Bundle is incomplete on disk; missing: {missing_paths}")

    if verify_checksums:
        for relative_path, record in records.items():
            path = _contained_path(root, relative_path)
            if not path.is_file():
                raise RetrievalVLBundleError(f"Cannot checksum non-file artifact: {path}")
            expected_checksum = record.get("checksum")
            actual_checksum = _sha256(path)
            if actual_checksum != expected_checksum:
                raise RetrievalVLBundleError(
                    f"Checksum mismatch for {relative_path}: expected {expected_checksum}, got {actual_checksum}"
                )
            for field in ("bytes", "size_bytes"):
                if field in record and path.stat().st_size != record[field]:
                    raise RetrievalVLBundleError(f"Size mismatch for {relative_path}")
        for directory in corpus_paths:
            for path in directory.rglob("*"):
                if path.is_file() and path.relative_to(root).as_posix() not in records:
                    raise RetrievalVLBundleError(f"Unlisted corpus artifact: {path}")

    gold_eval = root / f"gold_eval/{view}" if gold_eval_available else None
    return VLBundlePaths(
        root=root,
        manifest=manifest_path,
        view=view,
        train=root / f"views/{view}/train.json",
        validation=root / f"views/{view}/validation.json",
        train_corpus=corpus_paths[0],
        validation_corpus=corpus_paths[1],
        evaluation_corpus=corpus_paths[2],
        synthetic_eval=root / f"synthetic_eval/{view}",
        gold_eval=gold_eval,
        local_artifacts_verified=require_local_artifacts and verify_checksums,
    )


def prepare_vl_training_data(
    manifest_path: Path,
    *,
    output_dir: Path,
    view: ViewName = "text_image",
) -> Path:
    """Copy a verified training view into a new recipe preparation directory.

    Args:
        manifest_path: Immutable producer bundle manifest.
        output_dir: New directory; existing outputs are never overwritten.
        view: Recipe view name, mapped explicitly for the producer contract.

    Returns:
        Training JSON accepted by Stage 1 ``train_input_file``. Its corpus is
        local to the output directory, and generated positives remain partial.
    """
    paths = inspect_vl_bundle(manifest_path, view=view, verify_checksums=True)
    output_dir = output_dir.resolve()
    if output_dir == paths.root or paths.root in output_dir.parents:
        raise RetrievalVLBundleError("Preparation output must be outside the producer bundle")
    payload = _load_json(paths.train)
    for record in payload["data"]:
        if record.get("neg_doc") != [] or record.get("negative_mining_performed", False) is not False:
            raise RetrievalVLBundleError("Preparation requires positive-only records before mining")
        record.update(
            negative_mining_performed=False,
            relevance_judgements_complete=False,
            unlisted_document_disposition="unjudged",
            relevance_annotation_scope="known_positives_only",
        )
    payload["corpus"]["path"] = "corpus/train"
    payload["source_bundle"] = {
        "schema_version": _load_json(paths.manifest)["schema_version"],
        "manifest_sha256": _sha256(paths.manifest),
        "producer_view": paths.train.parent.name,
        "evaluation_provenance": "synthetic; independent evaluation supplied separately",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(paths.train_corpus, output_dir / "corpus/train")
    result = output_dir / "train.json"
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return result


def validate_vl_training_data(
    train_data_path: Path,
    *,
    required_negatives: int,
    require_mined_negatives: bool,
) -> int:
    """Validate records without overstating model-mined negatives as judgements."""
    if required_negatives < 0:
        raise ValueError("required_negatives must be non-negative")

    payload = _load_json(train_data_path)
    if not isinstance(payload.get("corpus"), dict):
        raise RetrievalVLBundleError("Training data must contain a corpus object")
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise RetrievalVLBundleError("Training data must contain a non-empty data list")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RetrievalVLBundleError(f"Training record {index} must be an object")
        if not record.get("pos_doc"):
            raise RetrievalVLBundleError(f"Training record {index} has no positive document")
        negatives = record.get("neg_doc")
        if not isinstance(negatives, list) or len(negatives) < required_negatives:
            count = len(negatives) if isinstance(negatives, list) else 0
            raise RetrievalVLBundleError(
                f"Training record {index} has {count} negatives; {required_negatives} are required"
            )
        if require_mined_negatives:
            if record.get("negative_mining_performed") is not True:
                raise RetrievalVLBundleError(
                    f"Training record {index} must set negative_mining_performed=True after downstream mining"
                )
            for negative_index, negative in enumerate(negatives[:required_negatives]):
                score = negative.get("score") if isinstance(negative, dict) else None
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                    raise RetrievalVLBundleError(
                        f"Training record {index} negative {negative_index} has no finite mining score"
                    )

    return len(records)


def mine_vllm_hard_negatives(
    train_data_path: Path,
    output_path: Path,
    *,
    api_url: str,
    model: str,
    batch_size: int,
    hard_negatives_to_mine: int,
    hard_neg_margin: float,
    expected_dimension: int | None = None,
    use_images: bool = True,
    api_truncate: Literal["NONE", "START", "END"] | None = None,
) -> Path:
    """Mine scored VL negatives from a DataDesigner Parquet corpus via vLLM."""
    import numpy as np
    import pyarrow.parquet as pq

    from nemotron.recipes.embed.stage3_eval.eval import NIMEmbeddingModel

    payload = _load_json(train_data_path)
    corpus = payload.get("corpus")
    if not isinstance(corpus, dict) or not isinstance(corpus.get("path"), str):
        raise RetrievalVLBundleError("Training data corpus.path must be a string")
    corpus_path = _contained_path(train_data_path.parent, corpus["path"])
    parquet_files = sorted(corpus_path.glob("*.parquet"))
    if not parquet_files:
        raise RetrievalVLBundleError(f"No Parquet corpus shards found under {corpus_path}")

    available = set(pq.read_schema(parquet_files[0]).names)
    id_columns = available.intersection({"docid", "id", "image_filename"})
    if len(id_columns) != 1:
        raise RetrievalVLBundleError("Corpus must have one supported document ID column")
    id_column = id_columns.pop()
    has_images = use_images and "image" in available
    if "text" not in available and not has_images:
        raise RetrievalVLBundleError("Corpus has no usable text or enabled images")
    columns = [id_column, *(["text"] if "text" in available else []), *(["image"] if has_images else [])]
    table = pq.read_table(parquet_files, columns=columns)
    doc_ids = table[id_column].to_pylist()
    if len(doc_ids) != len(set(doc_ids)):
        raise RetrievalVLBundleError("Corpus contains duplicate document IDs")
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise RetrievalVLBundleError("Training data must contain a non-empty data list")

    client = NIMEmbeddingModel(
        api_url=api_url,
        model=model,
        batch_size=batch_size,
        timeout=600,
        expected_dimension=expected_dimension,
        api_backend="vllm",
        api_truncate=api_truncate,
    )
    query_embeddings = client.encode_queries([str(record.get("question", "")) for record in records])

    document_inputs: list[str | dict[str, Any]] = []
    texts = table["text"].to_pylist() if "text" in available else [""] * len(doc_ids)
    images = table["image"].to_pylist() if has_images else [None] * len(doc_ids)
    for text, image in zip(texts, images, strict=True):
        document_text = str(text or "").strip()
        if image:
            raw_image = image.get("bytes") if isinstance(image, dict) else image
            encoded = base64.b64encode(bytes(raw_image)).decode("ascii")
            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            ]
            if document_text:
                content.append({"type": "text", "text": document_text})
            document_inputs.append({"content": content})
        else:
            document_inputs.append(document_text)

    document_batches = []
    for start in range(0, len(document_inputs), batch_size):
        batch = document_inputs[start : start + batch_size]
        document_batches.extend(client._encode_batch(batch, input_type="passage"))
    document_embeddings = np.asarray(document_batches, dtype=np.float32)
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        query_dimension = query_embeddings.shape[1]
        document_dimension = document_embeddings.shape[1]
        raise RetrievalVLBundleError(
            f"Query dimension {query_dimension} does not match corpus dimension {document_dimension}"
        )

    scores = query_embeddings @ document_embeddings.T
    for row_index, record in enumerate(records):
        positives = record.get("pos_doc")
        if not isinstance(positives, list) or not positives:
            raise RetrievalVLBundleError(f"Training record {row_index} has no positive document")
        try:
            positive_indices = [doc_index[positive["id"]] for positive in positives]
        except (KeyError, TypeError) as error:
            raise RetrievalVLBundleError(f"Training record {row_index} references an unknown positive") from error
        positive_scores = [float(scores[row_index, index]) for index in positive_indices]
        for positive, score in zip(positives, positive_scores, strict=True):
            positive["score"] = score

        candidate_scores = scores[row_index].copy()
        candidate_scores[positive_indices] = -np.inf
        candidate_scores[candidate_scores > min(positive_scores) * hard_neg_margin] = -np.inf
        candidate_indices = np.argsort(-candidate_scores, kind="stable")
        selected = [index for index in candidate_indices if np.isfinite(candidate_scores[index])][
            :hard_negatives_to_mine
        ]
        if len(selected) < hard_negatives_to_mine:
            raise RetrievalVLBundleError(
                f"Training record {row_index} yielded only {len(selected)} finite hard negatives"
            )
        record["neg_doc"] = [{"id": doc_ids[index], "score": float(candidate_scores[index])} for index in selected]
        record["negative_scores"] = [float(candidate_scores[index]) for index in selected]
        record["negative_mining_performed"] = True

    payload["corpus"]["path"] = str(corpus_path.resolve())
    payload["mining"] = {
        "backend": "vllm",
        "model": model,
        "api_url": api_url,
        "batch_size": batch_size,
        "hard_negatives_to_mine": hard_negatives_to_mine,
        "hard_neg_margin": hard_neg_margin,
        "uses_document_images": has_images,
        "embedding_dimension": int(document_embeddings.shape[1]),
        "relevance_semantics": "retrieval-mined candidates remain unjudged",
    }
    if api_truncate is not None:
        payload["mining"]["api_truncate"] = api_truncate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path
