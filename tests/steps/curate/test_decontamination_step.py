# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""S4 end to end, on the CPU path.

The GPU candidate pass is Curator's and is not exercised here. What is exercised
is everything the step adds around it: the exact-identity pass, verification,
the removal direction, and the wording of the report — which is where the
overclaim would live if it lived anywhere.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from nemotron.steps.curate.nemo_curator.runtime import decon
from nemotron.steps.curate.nemo_curator.scripts import run_decontamination as step

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "nemo_curator", "decontamination")

FILLER = " ".join(f"sentence {i} about a general and unrelated topic" for i in range(30))


def write(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    return str(path)


def config(tmp_path: Path, train: list[dict], holdout: list[dict], **overrides) -> dict:
    cfg = {
        "train_glob": write(tmp_path / "train.jsonl", train),
        "holdout_glob": write(tmp_path / "holdout.jsonl", holdout),
        "output_dir": str(tmp_path / "out"),
        "id_field": "id",
        "text_field": "text",
        "threshold": 0.8,
        "skip_similarity": True,
        "work_dir": str(tmp_path / "work"),
    }
    cfg.update(overrides)
    return cfg


def kept_ids(cfg: dict) -> set[str]:
    path = Path(cfg["output_dir"]) / "train_decontaminated.jsonl"
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}


# -- static -------------------------------------------------------------------


def test_decontamination_step_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/decontamination",
        expected_launch="python",
        expected_default_config="default",
    )


def test_it_is_the_only_curate_step_that_declares_a_gpu() -> None:
    """The similarity pass is GPU-backed; the other curate steps are not."""
    source = (STEP_DIR / "step.py").read_text(encoding="utf-8")

    assert "gpus_per_node = 1" in source


def test_the_gpu_path_has_a_packaged_runtime_extra() -> None:
    # STEP_DIR is one level deeper than it was: steps live under nemo_curator/
    # now. Walk up to the pyproject rather than counting, so the next move does
    # not silently point this at the wrong file.
    root = STEP_DIR
    while not (root / "pyproject.toml").is_file() and root != root.parent:
        root = root.parent
    with (root / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)

    requirements = project["project"]["optional-dependencies"]["curate-gpu"]
    runtime = project["tool"]["nemotron"]["runtime"]["curate-gpu"]
    assert any("deduplication_cuda12" in requirement for requirement in requirements)
    assert runtime["extras"] == ["curate-gpu"]
    assert "cudf" in runtime["required-imports"]


def test_the_manifest_states_the_threat_model_boundary() -> None:
    """A reviewer must be able to find the limit without reading the code."""
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    text = manifest["step"]["description"] + " ".join(s["then"] for s in manifest.get("strategies", []))
    assert "substring" in text.lower()


def test_the_tiny_config_runs_without_a_gpu() -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / "tiny.yaml").read_text(encoding="utf-8"))

    assert cfg["skip_similarity"] is True


def test_nothing_in_the_step_claims_the_holdout_is_clean() -> None:
    """The overclaim, blocked across every user-visible surface."""
    for path in (
        STEP_DIR / "README.md",
        STEP_DIR / "step.toml",
        STEP_DIR / "config" / "default.yaml",
        Path(step.__file__),
    ):
        text = path.read_text(encoding="utf-8").lower()
        # The README names the phrase in order to reject it.
        assert "verified clean" not in text.replace('never "holdout verified clean"', "")


# -- the exact identity pass --------------------------------------------------


def test_the_same_page_is_removed_even_with_no_textual_overlap(tmp_path) -> None:
    """The leak the similarity pass cannot see, end to end."""
    holdout = [{"id": "h1", "url": "https://example.test/a", "text": "Original wording. " + FILLER}]
    train = [
        {
            "id": "t1",
            "url": "http://www.Example.test/a?utm_source=x",
            "text": "Completely rewritten. " + FILLER.replace("sentence", "line"),
        },
        {"id": "t2", "url": "https://other.test/b", "text": "Unrelated. " + FILLER.replace("sentence", "item")},
    ]
    cfg = config(tmp_path, train, holdout)

    report = step.run(cfg, "2026-08-25T00:00:00Z")

    assert kept_ids(cfg) == {"t2"}
    assert report["group_overlap"]["shared_group_count"] == 1
    assert report["result"]["removed_by_group_identity_only"] == 1


def test_a_clean_corpus_loses_nothing(tmp_path) -> None:
    holdout = [{"id": "h1", "url": "https://example.test/a", "text": "Holdout content. " + FILLER}]
    train = [
        {
            "id": f"t{i}",
            "url": f"https://other.test/{i}",
            "text": f"Training document {i}. " + FILLER.replace("sentence", f"item{i}"),
        }
        for i in range(3)
    ]
    cfg = config(tmp_path, train, holdout)

    report = step.run(cfg, "2026-08-25T00:00:00Z")

    assert len(kept_ids(cfg)) == 3
    assert report["result"]["train_documents_removed"] == 0


# -- direction ----------------------------------------------------------------


def test_the_holdout_is_never_written(tmp_path) -> None:
    holdout = [{"id": "h1", "url": "https://example.test/a", "text": "Holdout. " + FILLER}]
    train = [{"id": "t1", "url": "https://example.test/a", "text": "Same page. " + FILLER}]
    cfg = config(tmp_path, train, holdout)

    step.run(cfg, "2026-08-25T00:00:00Z")

    written = {p.name for p in Path(cfg["output_dir"]).iterdir()}
    assert written == {"train_decontaminated.jsonl", "decontamination_report.json"}


def test_the_report_records_that_the_holdout_was_not_modified(tmp_path) -> None:
    holdout = [{"id": "h1", "text": "Holdout. " + FILLER}]
    train = [{"id": "t1", "text": "Training. " + FILLER}]

    report = step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")

    assert report["splits"]["holdout"]["modified"] is False
    assert report["splits"]["holdout"]["fingerprint"].startswith("sha256:")


# -- the report ---------------------------------------------------------------


def test_the_report_states_the_threat_model_and_the_claim(tmp_path) -> None:
    holdout = [{"id": "h1", "text": "Holdout. " + FILLER}]
    train = [{"id": "t1", "text": "Training. " + FILLER}]

    report = step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")

    assert "substring contamination" in report["threat_model"]
    assert "not a statement that the holdout is uncontaminated" in report["claim"].lower()


def test_skipping_similarity_says_it_was_not_measured(tmp_path) -> None:
    """Reporting zero overlap for a comparison that never ran would be a lie."""
    holdout = [{"id": "h1", "text": "Holdout. " + FILLER}]
    train = [{"id": "t1", "text": "Training. " + FILLER}]

    report = step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")

    assert "NOT measured" in report["similarity"]["note"]
    assert report["similarity"]["candidate_pairs"] == 0


def test_the_report_records_the_shingling_and_normalization(tmp_path) -> None:
    """Two runs that normalized differently did not measure the same thing."""
    holdout = [{"id": "h1", "text": "Holdout. " + FILLER}]
    train = [{"id": "t1", "text": "Training. " + FILLER}]

    params = step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")["parameters"]

    assert params["shingle_kind"] == decon.DEFAULT_SHINGLE_KIND
    assert params["shingle_size"] == decon.DEFAULT_SHINGLE_SIZE
    assert set(params["normalization"]) == {
        "nfc",
        "casefold",
        "collapse_whitespace",
        "strip_punctuation",
    }


def test_the_removal_counts_separate_the_two_mechanisms(tmp_path) -> None:
    """Which pass removed a document is the thing a reader wants to know."""
    holdout = [{"id": "h1", "url": "https://example.test/a", "text": "Holdout. " + FILLER}]
    train = [
        {"id": "t1", "url": "https://example.test/a", "text": "Rewritten. " + FILLER.replace("sentence", "line")},
        {"id": "t2", "url": "https://other.test/x", "text": "Clean. " + FILLER.replace("sentence", "item")},
    ]

    result = step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")["result"]

    assert result["removed_by_group_identity_only"] == 1
    assert result["removed_by_similarity_only"] == 0
    assert result["removed_by_both"] == 0


# -- refusals -----------------------------------------------------------------


def test_a_missing_id_is_refused(tmp_path) -> None:
    holdout = [{"id": "h1", "text": FILLER}]
    train = [{"text": FILLER}]

    with pytest.raises(step.ConfigError, match="no 'id'"):
        step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")


def test_a_duplicate_id_is_refused(tmp_path) -> None:
    """A removal report keyed on a non-unique id cannot say what it removed."""
    holdout = [{"id": "h1", "text": FILLER}]
    train = [{"id": "t1", "text": FILLER}, {"id": "t1", "text": FILLER}]

    with pytest.raises(step.ConfigError, match="more than once"):
        step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")


def test_incomparable_key_fields_refuse_before_output(tmp_path) -> None:
    train = [
        {
            "id": "train-id",
            "url": "https://train.example/document",
            "text": "Training. " + FILLER,
        }
    ]
    holdout = [{"id": "holdout-id", "text": "Holdout. " + FILLER}]
    cfg = config(tmp_path, train, holdout)

    with pytest.raises(step.ConfigError, match="keyed off a field"):
        step.run(cfg, "2026-08-25T00:00:00Z")

    assert not Path(cfg["output_dir"], "train_decontaminated.jsonl").exists()
    assert not Path(cfg["output_dir"], "decontamination_report.json").exists()


def test_an_impossible_threshold_is_refused(tmp_path) -> None:
    holdout = [{"id": "h1", "text": FILLER}]
    train = [{"id": "t1", "text": FILLER}]

    with pytest.raises(step.ConfigError, match="threshold"):
        step.run(config(tmp_path, train, holdout, threshold=0), "2026-08-25T00:00:00Z")


def test_an_unmatched_glob_is_refused(tmp_path) -> None:
    holdout = [{"id": "h1", "text": FILLER}]
    train = [{"id": "t1", "text": FILLER}]
    cfg = config(tmp_path, train, holdout, holdout_glob=str(tmp_path / "none" / "*.jsonl"))

    with pytest.raises(step.ConfigError, match="holdout_glob matched no files"):
        step.run(cfg, "2026-08-25T00:00:00Z")


def test_the_similarity_pass_is_not_imported_when_skipped(tmp_path, monkeypatch) -> None:
    """The CPU path must not need cudf present to run."""

    def boom(*args, **kwargs):
        raise AssertionError("the GPU path ran while skip_similarity was set")

    monkeypatch.setattr(step, "candidate_pairs", boom)
    holdout = [{"id": "h1", "text": FILLER}]
    train = [{"id": "t1", "text": FILLER}]

    step.run(config(tmp_path, train, holdout), "2026-08-25T00:00:00Z")


def test_missing_gpu_dependencies_name_the_install_extra(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "nemo_curator.stages.deduplication.fuzzy.workflow", None)

    with pytest.raises(step.ConfigError, match=r"nemotron\[curate-gpu\]"):
        step.candidate_pairs(
            {"work_dir": str(tmp_path / "work")},
            [{"id": "t1", "text": FILLER}],
            [{"id": "h1", "text": FILLER}],
            "id",
            "text",
        )


def test_cross_split_pairs_use_named_lsh_columns_not_column_order(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    output = tmp_path / "lsh" / "band_0-band_5"
    output.mkdir(parents=True)
    pd.DataFrame(
        {
            "unrelated": ["ignored", "ignored"],
            step.CURATOR_BUCKET_FIELD: ["bucket-a", "bucket-b"],
            step.CURATOR_ID_FIELD: [[0, 1, 2], [1, 2]],
        }
    ).to_parquet(output / "part.0.parquet")

    pairs = step.read_cross_split_pairs(
        tmp_path / "lsh",
        {
            "holdout:h1": step.HOLDOUT,
            "train:t1": step.TRAIN,
            "train:t2": step.TRAIN,
        },
        {0: "holdout:h1", 1: "train:t1", 2: "train:t2"},
    )

    assert pairs == [("t1", "h1"), ("t2", "h1")]


def test_records_without_text_and_non_objects_are_counted(tmp_path) -> None:
    holdout = [{"id": "h1", "text": "Holdout. " + FILLER}]
    train_path = tmp_path / "custom_train.jsonl"
    train_path.write_text(
        "\n".join(
            (
                json.dumps({"id": "t1", "text": "Training. " + FILLER}),
                json.dumps({"id": "missing-text"}),
                "42",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = config(tmp_path, [], holdout, train_glob=str(train_path))

    report = step.run(cfg, "2026-08-25T00:00:00Z")

    assert report["splits"]["train"]["documents"] == 1
    assert report["splits"]["train"]["skipped_missing_text"] == 1
    assert report["splits"]["train"]["skipped_non_mapping"] == 1


# -- verification wiring ------------------------------------------------------


def test_a_verified_near_duplicate_is_removed(tmp_path, monkeypatch) -> None:
    """The similarity path, with Curator's candidate generator stubbed out."""
    text = "A distinctive holdout document body. " + FILLER
    holdout = [{"id": "h1", "text": text}]
    train = [{"id": "t1", "text": text}, {"id": "t2", "text": "Different. " + FILLER.replace("sentence", "item")}]

    monkeypatch.setattr(step, "candidate_pairs", lambda *a, **k: [("t1", "h1"), ("t2", "h1")])
    cfg = config(tmp_path, train, holdout, skip_similarity=False)

    report = step.run(cfg, "2026-08-25T00:00:00Z")

    assert kept_ids(cfg) == {"t2"}, "t2 was a candidate but is not a duplicate"
    assert report["result"]["removed_by_similarity_only"] == 1
    assert report["similarity"]["candidate_pairs"] == 2
    assert report["similarity"]["verified_duplicates"] == 1


def test_an_unverified_candidate_is_reported_and_kept(tmp_path, monkeypatch) -> None:
    holdout = [{"id": "h1", "text": "short"}]
    train = [{"id": "t1", "text": "short"}]

    monkeypatch.setattr(step, "candidate_pairs", lambda *a, **k: [("t1", "h1")])
    cfg = config(tmp_path, train, holdout, skip_similarity=False)

    report = step.run(cfg, "2026-08-25T00:00:00Z")

    assert kept_ids(cfg) == {"t1"}
    assert report["similarity"]["unverifiable_pairs"] == 1
    assert report["similarity"]["unverifiable"][0]["verifiable"] is False


def test_removed_pairs_carry_their_similarity_as_evidence(tmp_path, monkeypatch) -> None:
    """A removal report without the number that justified it is not evidence."""
    text = "A distinctive holdout document body. " + FILLER
    holdout = [{"id": "h1", "text": text}]
    train = [{"id": "t1", "text": text}]

    monkeypatch.setattr(step, "candidate_pairs", lambda *a, **k: [("t1", "h1")])
    report = step.run(config(tmp_path, train, holdout, skip_similarity=False), "2026-08-25T00:00:00Z")

    pair = report["similarity"]["removed_pairs"][0]
    assert pair["train_id"] == "t1" and pair["holdout_id"] == "h1"
    assert pair["similarity"] >= 0.8
