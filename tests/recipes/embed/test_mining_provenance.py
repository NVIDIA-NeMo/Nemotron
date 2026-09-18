# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for exact source-to-miner provenance restoration."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemotron.recipes.embed.stage1_data_prep.mining_provenance import restore_mining_provenance
from nemotron.recipes.embed.stage1_data_prep.scripts.unroll_pos_docs import unroll_training_data


@pytest.fixture
def inputs(tmp_path):
    row = {
        "question_id": "q",
        "question": "Which evidence?",
        "corpus_id": "c",
        "query_group_id": "family",
        "source_page_ids": ["p", "p2", "context"],
        "pos_doc": [{"id": "p", "score": 2}, {"id": "p2", "score": 1}],
        "neg_doc": [],
        "negative_scores": [],
        "custom": {"language": "en"},
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"data": [row], "corpus": {"path": "corpus"}}))
    mined_row = {key: copy.deepcopy(row[key]) for key in ("question_id", "question", "corpus_id", "pos_doc")}
    mined_row["pos_doc"][0]["score"] = 0.8
    mined_row["pos_doc"][1]["score"] = 0.7
    mined_row["neg_doc"] = [{"id": "negative", "score": 0.5}]
    mined = tmp_path / "mined.json"
    mined.write_text(json.dumps({"data": [mined_row], "mining": {"args": {"model": "base"}}}))
    return source, mined


def test_metadata_and_graded_multipage_positives_survive_unrolling(inputs):
    source, mined = inputs
    restore_mining_provenance(source, mined)
    result = json.loads(mined.read_text())
    row = result["data"][0]
    assert row["query_group_id"] == "family"
    assert row["custom"] == {"language": "en"}
    assert row["source_page_ids"] == ["p", "p2", "context"]
    assert [(p["score"], p["relevance_grade"]) for p in row["pos_doc"]] == [(0.8, 2), (0.7, 1)]
    assert row["neg_doc"] == [{"id": "negative", "score": 0.5}]
    assert "negative_scores" not in row
    assert result["mining"]["args"]["model"] == "base"
    unrolled = unroll_training_data([row])
    assert len(unrolled) == 2
    assert all(r["query_group_id"] == "family" and set(r["all_pos_doc_ids"]) == {"p", "p2"} for r in unrolled)


@pytest.mark.parametrize(
    "change", ["text", "corpus", "positive", "duplicate_positive", "collision", "missing", "duplicate"]
)
def test_invalid_join_fails_without_modifying_output(inputs, change):
    source, mined = inputs
    output = json.loads(mined.read_text())
    row = output["data"][0]
    if change == "text":
        row["question"] = "Changed"
    elif change == "corpus":
        row["corpus_id"] = "different"
    elif change == "positive":
        row["pos_doc"].pop()
    elif change == "duplicate_positive":
        row["pos_doc"].append(row["pos_doc"][0])
    elif change == "collision":
        row["neg_doc"][0]["id"] = "p2"
    elif change == "missing":
        output["data"] = []
    else:
        output["data"].append(copy.deepcopy(row))
    mined.write_text(json.dumps(output))
    before = mined.read_bytes()
    with pytest.raises(ValueError):
        restore_mining_provenance(source, mined)
    assert mined.read_bytes() == before


@pytest.mark.parametrize("rank_zero", [True, False])
def test_actual_native_entrypoint_restores_only_after_run_on_rank_zero(rank_zero):
    # Execute the entrypoint's real body without importing CUDA/AutoModel.
    import nemotron.recipes.embed.stage1_data_prep.mining_provenance as module

    script = Path(module.__file__).parent / "scripts/mine_hard_negatives.py"
    main = next(
        node
        for node in ast.parse(script.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    recipe = Mock(
        dist_env=SimpleNamespace(is_main=rank_zero), train_qa_file_path="source", train_file_output_path="output"
    )
    restore = Mock()
    namespace = {
        "Path": Path,
        "parse_args_and_load_config": Mock(),
        "MineHardNegativesRecipe": Mock(return_value=recipe),
        "restore_mining_provenance": restore,
    }
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(script), "exec"), namespace)
    namespace["main"]()
    recipe.setup.assert_called_once()
    recipe.run.assert_called_once()
    assert restore.call_count == int(rank_zero)
