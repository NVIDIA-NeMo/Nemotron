"""CPU-tier integration tests for embed recipe stage 1 data preparation.

Validates that *convert_to_retriever_data.py* and *unroll_pos_docs.py*
produce correctly formatted outputs from synthetic SDG data, and that
all cross-stage data contracts are satisfied.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from .conftest import CORPUS_ID

pytestmark = pytest.mark.integration


# ===================================================================
# Helpers
# ===================================================================
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ===================================================================
# TestConvertOutputStructure — filesystem layout
# ===================================================================
class TestConvertOutputStructure:
    """Verify that all expected files and directories exist after convert."""

    def test_train_json_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "train.json").is_file()

    def test_val_json_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "val.json").is_file()

    def test_corpus_dir_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "corpus").is_dir()

    def test_corpus_parquet_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "corpus" / "train.parquet").is_file()

    def test_corpus_metadata_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "corpus" / "merlin_metadata.json").is_file()

    def test_eval_beir_dir_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "eval_beir").is_dir()

    def test_eval_corpus_jsonl_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "eval_beir" / "corpus.jsonl").is_file()

    def test_eval_queries_jsonl_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "eval_beir" / "queries.jsonl").is_file()

    def test_eval_qrels_exists(self, convert_output_dir: Path) -> None:
        assert (convert_output_dir / "eval_beir" / "qrels" / "test.tsv").is_file()


# ===================================================================
# TestTrainJsonFormat — train.json schema for stage 2
# ===================================================================
class TestTrainJsonFormat:
    """Validate that train.json conforms to the NeMo Retriever training format."""

    @pytest.fixture()
    def train_data(self, convert_output_dir: Path) -> dict:
        return _load_json(convert_output_dir / "train.json")

    def test_top_level_keys(self, train_data: dict) -> None:
        assert set(train_data.keys()) == {"corpus", "data"}

    def test_corpus_has_path(self, train_data: dict) -> None:
        assert "path" in train_data["corpus"]

    def test_data_non_empty(self, train_data: dict) -> None:
        assert len(train_data["data"]) > 0

    def test_record_keys(self, train_data: dict) -> None:
        required = {"question_id", "question", "corpus_id", "pos_doc", "neg_doc"}
        for rec in train_data["data"]:
            assert required.issubset(rec.keys()), f"Missing keys in {rec.get('question_id')}"

    def test_question_ids_unique(self, train_data: dict) -> None:
        ids = [r["question_id"] for r in train_data["data"]]
        assert len(ids) == len(set(ids))

    def test_corpus_id_matches(self, train_data: dict) -> None:
        for rec in train_data["data"]:
            assert rec["corpus_id"] == CORPUS_ID

    def test_pos_doc_is_list_of_id_dicts(self, train_data: dict) -> None:
        for rec in train_data["data"]:
            assert isinstance(rec["pos_doc"], list)
            assert len(rec["pos_doc"]) > 0
            for doc in rec["pos_doc"]:
                assert "id" in doc

    def test_all_pos_doc_ids_in_corpus(self, train_data: dict, convert_output_dir: Path) -> None:
        corpus_df = pd.read_parquet(convert_output_dir / "corpus" / "train.parquet")
        corpus_ids = set(corpus_df["id"])
        for rec in train_data["data"]:
            for doc in rec["pos_doc"]:
                assert doc["id"] in corpus_ids, (
                    f"pos_doc id {doc['id']} not found in corpus parquet"
                )


# ===================================================================
# TestValJsonFormat — val.json mirrors train format
# ===================================================================
class TestValJsonFormat:
    """Validate that val.json has the same structure as train.json."""

    @pytest.fixture()
    def val_data(self, convert_output_dir: Path) -> dict:
        return _load_json(convert_output_dir / "val.json")

    def test_top_level_keys(self, val_data: dict) -> None:
        assert set(val_data.keys()) == {"corpus", "data"}

    def test_data_non_empty(self, val_data: dict) -> None:
        assert len(val_data["data"]) > 0

    def test_shares_corpus_path(self, val_data: dict, convert_output_dir: Path) -> None:
        train_data = _load_json(convert_output_dir / "train.json")
        assert val_data["corpus"]["path"] == train_data["corpus"]["path"]


# ===================================================================
# TestCorpusParquet — corpus/train.parquet
# ===================================================================
class TestCorpusParquet:
    """Validate the corpus parquet file read by stage 2 training."""

    @pytest.fixture()
    def corpus_df(self, convert_output_dir: Path) -> pd.DataFrame:
        return pd.read_parquet(convert_output_dir / "corpus" / "train.parquet")

    def test_readable(self, corpus_df: pd.DataFrame) -> None:
        assert len(corpus_df) > 0

    def test_has_required_columns(self, corpus_df: pd.DataFrame) -> None:
        assert {"id", "text"}.issubset(set(corpus_df.columns))

    def test_ids_match_pattern(self, corpus_df: pd.DataFrame) -> None:
        pattern = re.compile(r"^d_[0-9a-f]{16}$")
        for doc_id in corpus_df["id"]:
            assert pattern.match(doc_id), f"Unexpected id format: {doc_id}"

    def test_no_empty_texts(self, corpus_df: pd.DataFrame) -> None:
        for text in corpus_df["text"]:
            assert text.strip(), "Found empty text in corpus"

    def test_no_duplicate_ids(self, corpus_df: pd.DataFrame) -> None:
        assert corpus_df["id"].nunique() == len(corpus_df)


# ===================================================================
# TestCorpusMetadata — corpus/merlin_metadata.json
# ===================================================================
class TestCorpusMetadata:
    """Validate the corpus metadata file."""

    @pytest.fixture()
    def metadata(self, convert_output_dir: Path) -> dict:
        return _load_json(convert_output_dir / "corpus" / "merlin_metadata.json")

    def test_has_corpus_id(self, metadata: dict) -> None:
        assert metadata["corpus_id"] == CORPUS_ID

    def test_has_class(self, metadata: dict) -> None:
        assert "class" in metadata


# ===================================================================
# TestBeirEvalFormat — eval_beir/ for stage 3 evaluation
# ===================================================================
class TestBeirEvalFormat:
    """Validate the BEIR-formatted evaluation output."""

    @pytest.fixture()
    def corpus_records(self, convert_output_dir: Path) -> list[dict]:
        return _load_jsonl(convert_output_dir / "eval_beir" / "corpus.jsonl")

    @pytest.fixture()
    def query_records(self, convert_output_dir: Path) -> list[dict]:
        return _load_jsonl(convert_output_dir / "eval_beir" / "queries.jsonl")

    @pytest.fixture()
    def qrels_lines(self, convert_output_dir: Path) -> list[str]:
        path = convert_output_dir / "eval_beir" / "qrels" / "test.tsv"
        return path.read_text().splitlines()

    # --- corpus.jsonl ---
    def test_corpus_record_keys(self, corpus_records: list[dict]) -> None:
        required = {"_id", "text", "title", "metadata"}
        for rec in corpus_records:
            assert required.issubset(rec.keys()), f"Missing keys in corpus record {rec.get('_id')}"

    def test_corpus_non_empty(self, corpus_records: list[dict]) -> None:
        assert len(corpus_records) > 0

    # --- queries.jsonl ---
    def test_query_record_keys(self, query_records: list[dict]) -> None:
        required = {"_id", "text", "metadata"}
        for rec in query_records:
            assert required.issubset(rec.keys())

    def test_query_metadata_has_file_name(self, query_records: list[dict]) -> None:
        for rec in query_records:
            assert "file_name" in rec["metadata"]

    def test_query_metadata_has_segment_ids(self, query_records: list[dict]) -> None:
        for rec in query_records:
            assert "segment_ids" in rec["metadata"]

    # --- qrels/test.tsv ---
    def test_qrels_header(self, qrels_lines: list[str]) -> None:
        assert qrels_lines[0] == "query-id\tcorpus-id\tscore"

    def test_qrels_has_data_rows(self, qrels_lines: list[str]) -> None:
        assert len(qrels_lines) > 1, "qrels has no data rows"

    def test_qrels_all_scores_are_one(self, qrels_lines: list[str]) -> None:
        for line in qrels_lines[1:]:
            parts = line.split("\t")
            assert parts[2] == "1", f"Unexpected score: {parts[2]}"

    def test_qrels_query_ids_valid(
        self, query_records: list[dict], qrels_lines: list[str]
    ) -> None:
        valid_qids = {r["_id"] for r in query_records}
        for line in qrels_lines[1:]:
            qid = line.split("\t")[0]
            assert qid in valid_qids, f"qrels query-id {qid} not in queries.jsonl"

    def test_qrels_corpus_ids_valid(
        self, corpus_records: list[dict], qrels_lines: list[str]
    ) -> None:
        valid_cids = {r["_id"] for r in corpus_records}
        for line in qrels_lines[1:]:
            cid = line.split("\t")[1]
            assert cid in valid_cids, f"qrels corpus-id {cid} not in corpus.jsonl"


# ===================================================================
# TestUnrollOutput — unrolled training data
# ===================================================================
class TestUnrollOutput:
    """Validate the output of unroll_pos_docs.py."""

    @pytest.fixture()
    def unrolled_data(self, unroll_output_path: Path) -> dict:
        return _load_json(unroll_output_path)

    @pytest.fixture()
    def original_data(self, convert_output_dir: Path) -> dict:
        return _load_json(convert_output_dir / "train.json")

    def test_top_level_keys(self, unrolled_data: dict) -> None:
        assert set(unrolled_data.keys()) == {"corpus", "data"}

    def test_corpus_unchanged(self, unrolled_data: dict, original_data: dict) -> None:
        assert unrolled_data["corpus"] == original_data["corpus"]

    def test_every_record_has_single_pos_doc(self, unrolled_data: dict) -> None:
        for rec in unrolled_data["data"]:
            assert len(rec["pos_doc"]) == 1, (
                f"Record {rec['question_id']} has {len(rec['pos_doc'])} pos_docs"
            )

    def test_ids_have_suffix_for_multi_pos(self, unrolled_data: dict) -> None:
        suffixed = [r for r in unrolled_data["data"] if re.search(r"_\d+$", r["question_id"])]
        assert len(suffixed) > 0, "Expected some unrolled records with _N suffixes"

    def test_record_count_ge_original(
        self, unrolled_data: dict, original_data: dict
    ) -> None:
        assert len(unrolled_data["data"]) >= len(original_data["data"])

    def test_data_non_empty(self, unrolled_data: dict) -> None:
        assert len(unrolled_data["data"]) > 0


# ===================================================================
# TestCrossStageCompatibility — highest-value checks
# ===================================================================
class TestCrossStageCompatibility:
    """Verify that data contracts between stages are satisfied end-to-end."""

    def test_train_pos_docs_resolve_to_corpus_text(
        self, convert_output_dir: Path
    ) -> None:
        """Every pos_doc ID in train.json maps to a non-empty text in corpus parquet."""
        train = _load_json(convert_output_dir / "train.json")
        corpus_df = pd.read_parquet(convert_output_dir / "corpus" / "train.parquet")
        id_to_text = dict(zip(corpus_df["id"], corpus_df["text"]))

        for rec in train["data"]:
            for doc in rec["pos_doc"]:
                assert doc["id"] in id_to_text, (
                    f"pos_doc {doc['id']} from {rec['question_id']} missing in corpus"
                )
                assert id_to_text[doc["id"]].strip(), (
                    f"pos_doc {doc['id']} has empty text in corpus"
                )

    def test_beir_qrels_reference_valid_ids(self, convert_output_dir: Path) -> None:
        """All query-id and corpus-id values in qrels exist in their JSONL files."""
        eval_dir = convert_output_dir / "eval_beir"

        corpus_ids = {
            r["_id"] for r in _load_jsonl(eval_dir / "corpus.jsonl")
        }
        query_ids = {
            r["_id"] for r in _load_jsonl(eval_dir / "queries.jsonl")
        }

        qrels_path = eval_dir / "qrels" / "test.tsv"
        lines = qrels_path.read_text().splitlines()[1:]  # skip header

        for line in lines:
            qid, cid, _ = line.split("\t")
            assert qid in query_ids, f"qrels query-id {qid} not in queries.jsonl"
            assert cid in corpus_ids, f"qrels corpus-id {cid} not in corpus.jsonl"
