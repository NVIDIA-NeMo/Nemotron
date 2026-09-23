# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""S3 end to end: what the step writes, and what it refuses to write.

``test_subset.py`` covers the algorithm. These tests cover the artifacts, because
the guarantee that matters to a user is about the corpora on disk, not about the
plan that predicted them.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest
import yaml

from nemotron.steps.curate.nemo_curator.scripts import run_subset

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "nemo_curator", "subset")


def corpus(tmp_path: Path, n: int = 150) -> Path:
    """Three unequal sources, tied lengths, and a coarse score column."""
    path = tmp_path / "corpus.jsonl"
    rows = []
    for i in range(n):
        source = ["web", "web", "wiki", "news"][i % 4]
        length = [8, 8, 8, 25, 25, 90][i % 6]
        rows.append(
            {
                "id": f"doc-{i:04d}",
                "source": source,
                "text": " ".join(["word"] * length),
                "__quality": round((i % 5) / 5, 2),
                "url": f"https://example.test/{i}",
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def config(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "input_glob": str(corpus(tmp_path)),
        "output_dir": str(tmp_path / "out"),
        "text_field": "text",
        "id_field": "id",
        "source_field": "source",
        "token_budgets": [600, 1500, 4000],
        "tokenizer": None,
        "token_cache": None,
        "quality_score_field": None,
        "length_bands": [16, 50],
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def tier_ids(out: Path, budget: int, unit: str = "words") -> set[str]:
    path = out / f"budget_{budget}_{unit}" / "subset.jsonl"
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}


# -- static manifest ----------------------------------------------------------


def test_subset_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/subset",
        expected_launch="python",
        expected_default_config="default",
    )


def test_the_manifest_declares_both_analysis_artifacts() -> None:
    """A corpus with no plan and no report is not auditable later."""
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    produced = {p["type"] for p in manifest["produces"]}
    assert {"subset_plan", "subset_report", "filtered_jsonl"} <= produced


def test_the_tiny_config_needs_no_tokenizer_download() -> None:
    cfg = yaml.safe_load((STEP_DIR / "config" / "tiny.yaml").read_text(encoding="utf-8"))

    assert cfg["tokenizer"] is None, "a smoke config must not require model weights"


def test_the_packaged_fixture_has_unequal_sources() -> None:
    """Equal sources would let a stratification bug pass unnoticed."""
    path = STEP_DIR / "data" / "tiny" / "corpus.jsonl"
    sources: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        source = json.loads(line)["source"]
        sources[source] = sources.get(source, 0) + 1

    assert len(sources) >= 3
    assert len(set(sources.values())) > 1


# -- what a run writes --------------------------------------------------------


def test_a_run_writes_a_plan_a_report_and_one_corpus_per_tier(tmp_path) -> None:
    cfg = config(tmp_path)

    run_subset.run(cfg, "2026-08-25T00:00:00Z")

    out = Path(cfg["output_dir"])
    assert (out / "plan.json").is_file()
    assert (out / "subset_report.json").is_file()
    for budget in cfg["token_budgets"]:
        assert (out / f"budget_{budget}_words" / "subset.jsonl").is_file()


def test_the_written_tiers_nest(tmp_path) -> None:
    """The claim restated against the files, not the plan that predicted them."""
    cfg = config(tmp_path)
    run_subset.run(cfg, "2026-08-25T00:00:00Z")
    out = Path(cfg["output_dir"])

    budgets = cfg["token_budgets"]
    for smaller, larger in zip(budgets, budgets[1:], strict=False):
        assert tier_ids(out, smaller) <= tier_ids(out, larger)


def test_records_pass_through_unchanged(tmp_path) -> None:
    """A subset that quietly dropped a column would break the next step."""
    cfg = config(tmp_path)
    run_subset.run(cfg, "2026-08-25T00:00:00Z")

    line = (Path(cfg["output_dir"]) / "budget_4000_words" / "subset.jsonl").read_text(encoding="utf-8").splitlines()[0]

    assert set(json.loads(line)) == {"id", "source", "text", "__quality", "url"}


def test_the_plan_is_written_before_any_tier(tmp_path, monkeypatch) -> None:
    """A plan you can only read after the corpus exists cannot be used to reject it."""
    cfg = config(tmp_path)
    out = Path(cfg["output_dir"])

    real = run_subset.subset.materialize
    seen: dict[str, bool] = {}

    def spy(*args, **kwargs):
        seen["plan_exists"] = (out / "plan.json").is_file()
        seen["tiers_exist"] = any(out.glob("budget_*"))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_subset.subset, "materialize", spy)
    run_subset.run(cfg, "2026-08-25T00:00:00Z")

    assert seen["plan_exists"]
    assert not seen["tiers_exist"]


def test_the_report_states_the_unit_and_the_tokenizer(tmp_path) -> None:
    """A budget without a unit is a number nobody can reproduce."""
    cfg = config(tmp_path)
    report = run_subset.run(cfg, "2026-08-25T00:00:00Z")

    assert report["unit"] == "words"
    assert report["tokenizer"]["name"] == "whitespace"
    for tier in report["tiers"]:
        assert tier["unit"] == "words"


def test_every_tier_reports_shortfall_and_deviation(tmp_path) -> None:
    report = run_subset.run(config(tmp_path), "2026-08-25T00:00:00Z")

    for tier in report["tiers"]:
        assert tier["achieved_tokens"] <= tier["budget"]
        assert tier["token_shortfall"] == tier["budget"] - tier["achieved_tokens"]
        assert tier["per_stratum_deviation"]


def test_the_report_says_the_budget_is_a_ceiling(tmp_path) -> None:
    report = run_subset.run(config(tmp_path), "2026-08-25T00:00:00Z")

    assert "at most" in report["interpretation"]


def test_two_runs_of_the_same_config_select_the_same_documents(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run_subset.run(config(a), "2026-08-25T00:00:00Z")
    run_subset.run(config(b), "2026-08-25T00:00:00Z")

    assert tier_ids(Path(a / "out"), 1500) == tier_ids(Path(b / "out"), 1500)


# -- refusals -----------------------------------------------------------------


def test_a_missing_id_field_is_refused(tmp_path) -> None:
    with pytest.raises(run_subset.ConfigError, match="id_field is required"):
        run_subset.run(config(tmp_path, id_field=None), "2026-08-25T00:00:00Z")


def test_a_tokenizer_without_a_revision_is_refused(tmp_path) -> None:
    """Counts from two revisions are not comparable, so the pin is not optional."""
    with pytest.raises(run_subset.ConfigError, match="revision"):
        run_subset.run(
            config(tmp_path, tokenizer={"name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"}),
            "2026-08-25T00:00:00Z",
        )


@pytest.mark.parametrize("budgets", [[1.5], ["100"], [True], [0]])
def test_token_budgets_must_really_be_positive_integers(tmp_path, budgets) -> None:
    with pytest.raises(run_subset.ConfigError, match="positive integers"):
        run_subset.run(config(tmp_path, token_budgets=budgets), "2026-08-25T00:00:00Z")


@pytest.mark.parametrize("bands", [[16.5, 50], ["16", "50"], [True, 50], [50, 16], [16, 16]])
def test_length_bands_must_really_be_increasing_positive_integers(tmp_path, bands) -> None:
    with pytest.raises(run_subset.ConfigError, match="length_bands"):
        run_subset.run(config(tmp_path, length_bands=bands), "2026-08-25T00:00:00Z")


def test_a_score_field_absent_from_the_data_is_refused(tmp_path) -> None:
    with pytest.raises(run_subset.subset.SubsetError, match="__missing"):
        run_subset.run(config(tmp_path, quality_score_field="__missing"), "2026-08-25T00:00:00Z")


def test_non_finite_quality_scores_are_refused_by_the_step(tmp_path) -> None:
    path = tmp_path / "non-finite.jsonl"
    path.write_text(
        json.dumps({"id": "bad", "source": "web", "text": "one two", "__q": float("nan")}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(run_subset.subset.SubsetError, match="non-finite"):
        run_subset.run(
            config(tmp_path, input_glob=str(path), quality_score_field="__q"),
            "2026-08-25T00:00:00Z",
        )


def test_non_positive_token_counts_are_refused_by_the_step(tmp_path, monkeypatch) -> None:
    class BrokenCounter:
        name = "broken"
        revision = "test"
        hits = 0
        misses = 0

        @staticmethod
        def count(_doc_id, _text):
            return 0

        @staticmethod
        def save():
            return None

    monkeypatch.setattr(run_subset, "build_counter", lambda _cfg: (BrokenCounter(), "tokens"))

    with pytest.raises(run_subset.subset.SubsetError, match="non-positive token count"):
        run_subset.run(config(tmp_path), "2026-08-25T00:00:00Z")


def test_an_unmatched_glob_is_an_error(tmp_path) -> None:
    with pytest.raises(run_subset.ConfigError, match="matched no files"):
        run_subset.run(
            config(tmp_path, input_glob=str(tmp_path / "nothing" / "*.jsonl")),
            "2026-08-25T00:00:00Z",
        )


def test_a_corpus_with_no_usable_records_is_an_error(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")

    with pytest.raises(run_subset.ConfigError, match="no usable documents"):
        run_subset.run(config(tmp_path, input_glob=str(path)), "2026-08-25T00:00:00Z")


def test_duplicate_ids_stop_the_run_before_anything_is_written(tmp_path) -> None:
    path = tmp_path / "dupes.jsonl"
    rows = [{"id": "same", "source": "web", "text": "word " * 20} for _ in range(10)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = config(tmp_path, input_glob=str(path))

    with pytest.raises(run_subset.subset.SubsetError, match="not unique"):
        run_subset.run(cfg, "2026-08-25T00:00:00Z")

    assert not any(Path(cfg["output_dir"]).glob("budget_*"))


def test_unparsable_lines_are_counted_not_ignored(tmp_path) -> None:
    source = corpus(tmp_path)
    damaged = tmp_path / "damaged.jsonl"
    damaged.write_text(source.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")

    report = run_subset.run(config(tmp_path, input_glob=str(damaged)), "2026-08-25T00:00:00Z")

    assert report["corpus"]["unparsable_lines"] == 1


# -- the token cache ----------------------------------------------------------


def test_a_cache_from_another_tokenizer_is_ignored(tmp_path, caplog) -> None:
    cache = tmp_path / "counts.json"
    cache.write_text(
        json.dumps(
            {
                "cache_version": run_subset.CACHE_VERSION,
                "tokenizer": "some/other-model",
                "revision": "abc",
                "counts": {"doc-0000": 999_999},
            }
        ),
        encoding="utf-8",
    )

    counter = run_subset.TokenCounter("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "main", cache)

    assert counter._counts == {}, "reusing counts from another tokenizer is silent corruption"


@pytest.mark.parametrize("contents", ["[]", "{not json"])
def test_a_malformed_token_cache_is_ignored(tmp_path, contents, caplog) -> None:
    cache = tmp_path / "counts.json"
    cache.write_text(contents, encoding="utf-8")

    counter = run_subset.TokenCounter("m", "r", cache)

    assert counter._counts == {}
    assert "ignored" in caplog.text


def test_a_cache_from_the_same_tokenizer_is_reused(tmp_path) -> None:
    text = "ignored"
    cache = tmp_path / "counts.json"
    cache.write_text(
        json.dumps(
            {
                "cache_version": run_subset.CACHE_VERSION,
                "tokenizer": "m",
                "revision": "r",
                "counts": {
                    "doc-0000": {
                        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "tokens": 42,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    counter = run_subset.TokenCounter("m", "r", cache)

    assert counter.count("doc-0000", text) == 42
    assert counter.hits == 1 and counter.misses == 0


def test_token_cache_invalidates_when_text_changes(tmp_path, monkeypatch) -> None:
    old_text = "old content"
    cache = tmp_path / "counts.json"
    cache.write_text(
        json.dumps(
            {
                "cache_version": run_subset.CACHE_VERSION,
                "tokenizer": "m",
                "revision": "r",
                "counts": {
                    "doc-0000": {
                        "content_sha256": hashlib.sha256(old_text.encode()).hexdigest(),
                        "tokens": 42,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    counter = run_subset.TokenCounter("m", "r", cache)

    class Filter:
        @staticmethod
        def score_document(_text):
            return 7

    monkeypatch.setattr(counter, "_load", lambda: Filter())

    assert counter.count("doc-0000", "new content") == 7
    assert counter.hits == 0 and counter.misses == 1


def test_the_cache_round_trips(tmp_path) -> None:
    cache = tmp_path / "counts.json"
    first = run_subset.WordCounter()
    written = run_subset.TokenCounter("m", "r", cache)
    text = "x"
    written._counts = {
        "a": {
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokens": 7,
        }
    }
    written.save()

    assert run_subset.TokenCounter("m", "r", cache).count("a", text) == 7
    assert first.count("a", "one two three") == 3


# -- vocabulary ---------------------------------------------------------------


def test_the_step_does_not_claim_to_maximize() -> None:
    """The contract is nesting; a word implying otherwise is the overclaim."""
    for path in (
        Path(run_subset.__file__),
        STEP_DIR / "README.md",
        STEP_DIR / "step.toml",
        STEP_DIR / "config" / "default.yaml",
    ):
        assert "maximiz" not in path.read_text(encoding="utf-8").lower(), path


# -- the tokenizer pin --------------------------------------------------------


def test_the_tokenizer_is_passed_as_a_model_name_not_a_tokenizer_object() -> None:
    """``tokenizer=`` takes a loaded AutoTokenizer; a string there is inert.

    TokenCountFilter accepts either a loaded ``tokenizer`` or an ``hf_model_name``
    string. Passing the model name under ``tokenizer=`` is accepted by the
    constructor and then fails once per document as
    ``str.encode(encoding=<the document text>)``. Worse, ``load_tokenizer`` only
    reads ``transformers_init_kwargs`` when ``hf_model_name`` is set, so the
    revision pin — the reason a revision is recorded at all — would never be
    applied and two incomparable subsets would look comparable.
    """
    import inspect

    source = inspect.getsource(run_subset.TokenCounter._load)

    assert "hf_model_name=self.name" in source
    assert "tokenizer=self.name" not in source


def test_the_revision_travels_in_transformers_init_kwargs() -> None:
    """TokenCountFilter has no revision parameter; this is where the pin lands."""
    import inspect

    source = inspect.getsource(run_subset.TokenCounter._load)

    assert 'transformers_init_kwargs={"revision": self.revision}' in source


def test_token_count_filter_still_accepts_these_arguments() -> None:
    """Pinned against the real signature wherever Curator is installed."""
    module = pytest.importorskip("nemo_curator.stages.text.filters.token.token_count")
    import inspect

    params = inspect.signature(module.TokenCountFilter.__init__).parameters

    for name in ("hf_model_name", "transformers_init_kwargs", "min_tokens", "max_tokens"):
        assert name in params, f"TokenCountFilter no longer accepts {name}"
    assert "revision" not in params, (
        "TokenCountFilter grew a revision parameter; the transformers_init_kwargs detour is no longer necessary"
    )


# -- the second nesting check must read the disk --------------------------------
#
# The step verifies nesting twice, and the pair was described as "the first
# catches a planning defect, the second a writing one". It did not: the second
# check was handed `wanted`, the same planned set the first had already checked,
# so it re-verified arithmetic instead of what reached disk. A tier that silently
# failed to write a document still reported nesting_verified: true.


def test_the_written_check_reads_the_ids_it_actually_wrote(tmp_path) -> None:
    """A document planning selected but writing dropped must not pass unnoticed.

    Asserted on the mechanism because the defect is only observable when the
    write diverges from the plan, which needs a broken writer to produce. What
    can be checked cheaply is that the second verification is not handed the
    first one's input.
    """
    source = (STEP_DIR.parent / "scripts" / "run_subset.py").read_text(encoding="utf-8")

    assert "written_ids[budget] = wanted" not in source, (
        "the written-tier check must collect the ids it wrote, not the planned set the first check already verified"
    )


def test_a_tier_that_writes_fewer_documents_than_planned_is_refused(tmp_path, monkeypatch) -> None:
    real_materialize = run_subset.subset.materialize

    def include_an_unaddressable_id(plan, rows):
        results = real_materialize(plan, rows)
        for result in results.values():
            result.doc_ids.append("not-present-in-the-corpus")
        return results

    monkeypatch.setattr(run_subset.subset, "materialize", include_an_unaddressable_id)
    cfg = config(tmp_path)

    with pytest.raises(run_subset.subset.SubsetError, match="planned .* but wrote"):
        run_subset.run(cfg, "2026-08-25T00:00:00Z")

    output = Path(cfg["output_dir"])
    assert not (output / "plan.json").exists()
    assert not (output / "subset_report.json").exists()
    assert not list(output.glob("budget_*"))


def test_a_failed_write_leaves_no_partial_tiers_or_success_reports(tmp_path, monkeypatch) -> None:
    real_verify = run_subset.subset.verify_nesting
    calls = 0

    def fail_second_verification(results):
        nonlocal calls
        calls += 1
        if calls == 2:
            return ["forced write failure"]
        return real_verify(results)

    monkeypatch.setattr(run_subset.subset, "verify_nesting", fail_second_verification)
    cfg = config(tmp_path)

    with pytest.raises(run_subset.subset.NestingViolationError, match="written tiers"):
        run_subset.run(cfg, "2026-08-25T00:00:00Z")

    output = Path(cfg["output_dir"])
    assert not (output / "plan.json").exists()
    assert not (output / "subset_report.json").exists()
    assert not list(output.glob("budget_*"))


def test_cleanup_removes_this_run_s_tiers_and_leaves_the_rest(tmp_path) -> None:
    """`budget_*` swept away tiers this run was never going to write.

    A directory holding budget_500000000_tokens from last week is not stale
    merely because today's config asks for budget_2000000_tokens. The tier names
    follow from the config alone — the unit is decided by whether a tokenizer is
    set — so the cleanup can name what it owns instead of globbing for it.
    """
    for name in ("budget_500000000_tokens", "budget_2000000_tokens", "budget_9_words"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "subset.jsonl").write_text("{}\n")
    (tmp_path / "plan.json").write_text("{}")

    cfg = {"token_budgets": [2000000], "tokenizer": {"name": "x", "revision": "y"}}
    assert run_subset._tier_dir_names(cfg) == ["budget_2000000_tokens"]

    run_subset._reset_output_artifacts(tmp_path, run_subset._tier_dir_names(cfg))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["budget_500000000_tokens", "budget_9_words"]


def test_the_tier_unit_follows_the_tokenizer(tmp_path) -> None:
    """Names must match what _run writes, or cleanup silently misses them."""
    assert run_subset._tier_dir_names({"token_budgets": [7], "tokenizer": None}) == ["budget_7_words"]
    assert run_subset._tier_dir_names({"token_budgets": [7]}) == ["budget_7_words"]
    assert run_subset._tier_dir_names({"token_budgets": [7], "tokenizer": {"name": "n", "revision": "r"}}) == [
        "budget_7_tokens"
    ]


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"input_glob": "/nonexistent/dir/*.jsonl"}, id="input_glob-matches-nothing"),
        pytest.param({"input_glob": None}, id="input_glob-absent"),
        pytest.param({"id_field": None}, id="id_field-missing"),
        pytest.param({"seed": "zero"}, id="seed-not-an-integer"),
        pytest.param({"tokenizer": {"name": "x"}}, id="tokenizer-without-revision"),
    ],
)
def test_a_config_error_leaves_the_previous_run_intact(tmp_path, override) -> None:
    """The delete must not run before the configuration is known to be good.

    Validation covered token_budgets and length_bands only, so every other config
    error still raised from inside _run — after the tiers had been removed. A
    mistyped input_glob destroyed the previous run on its way to reporting
    itself, which is the failure the review asked to prevent.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in ("budget_600_words", "budget_1500_words"):
        (output_dir / name).mkdir()
        (output_dir / name / "subset.jsonl").write_text('{"id": "a"}\n')
    (output_dir / "plan.json").write_text("{}")
    (output_dir / "subset_report.json").write_text("{}")
    before = sorted(p.name for p in output_dir.iterdir())

    cfg = {
        "output_dir": str(output_dir),
        "token_budgets": [600, 1500],
        "tokenizer": None,
        "input_glob": str(tmp_path / "data" / "*.jsonl"),
        "id_field": "id",
        "text_field": "text",
    }
    cfg.update(override)
    if override.get("input_glob") is None and "input_glob" in override:
        cfg.pop("input_glob")

    with pytest.raises(run_subset.ConfigError):
        run_subset.run(cfg)

    assert sorted(p.name for p in output_dir.iterdir()) == before


def test_a_padded_output_dir_is_not_silently_relative(tmp_path, monkeypatch) -> None:
    """Validating a stripped string and then using the unstripped one.

    `"  /data/out"` passed the emptiness check and then became a *relative* path
    whose first component is two spaces, created under the caller's working
    directory while /data/out was never touched — reinstating the very hazard
    the required-output_dir message says it prevents.
    """
    monkeypatch.chdir(tmp_path)
    wanted = tmp_path / "wanted"

    with pytest.raises(run_subset.ConfigError):
        run_subset.run(
            {
                "output_dir": f"  {wanted}  ",
                "token_budgets": [600],
                "tokenizer": None,
                "input_glob": str(tmp_path / "absent" / "*.jsonl"),
                "id_field": "id",
                "text_field": "text",
            }
        )

    assert [p.name for p in tmp_path.iterdir() if p.name.strip() != p.name] == []
