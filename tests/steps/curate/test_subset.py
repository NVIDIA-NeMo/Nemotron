# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""S3: the nesting guarantee, and the ways it is normally lost.

Nesting is easy to claim and easy to break silently. Each test below either
demonstrates the property on data with the ties and duplicates that break it, or
demonstrates a plausible alternative implementation failing so the choice made
here is visible rather than asserted.
"""

from __future__ import annotations

import random

import pytest

from nemotron.steps.curate.nemo_curator.runtime import subset
from nemotron.steps.curate.nemo_curator.runtime.subset import ScanRow

BUDGETS = [8_000, 25_000, 60_000, 140_000]


def corpus(n: int = 300, *, tied: bool = True, seed: int = 11) -> list[ScanRow]:
    """A corpus with heavy ties in token count — the case that breaks sorting.

    ``sort_values`` on a column with many equal values is not a total order, so
    membership at the cut boundary becomes run-dependent. Distinct values would
    hide exactly the defect worth testing for.
    """
    rng = random.Random(seed)
    sources = ["web", "wiki", "news"]
    rows = []
    for i in range(n):
        tokens = rng.choice([64, 64, 64, 100, 100, 700, 700, 3000]) if tied else 50 + i
        rows.append(
            ScanRow(
                doc_id=f"doc-{i:04d}",
                source=sources[i % len(sources)],
                tokens=tokens,
                score=round(rng.random(), 3),
            )
        )
    return rows


def tiers(rows, budgets=BUDGETS, **kwargs):
    plan = subset.build_plan(rows, budgets, **kwargs)
    return plan, subset.materialize(plan, rows)


# -- the guarantee ------------------------------------------------------------


def test_every_tier_nests_in_every_larger_tier() -> None:
    _, results = tiers(corpus())

    assert subset.verify_nesting(results) == []
    for smaller, larger in zip(BUDGETS, BUDGETS[1:], strict=False):
        assert set(results[smaller].doc_ids) <= set(results[larger].doc_ids)


def test_nesting_holds_with_a_score_field_full_of_ties() -> None:
    """Deciles of a coarse score column put many documents on a cut boundary."""
    rows = [ScanRow(f"doc-{i:04d}", ["web", "wiki"][i % 2], 100, score=round(i % 4 / 4, 2)) for i in range(200)]

    _, results = tiers(rows, score_field="__q")

    assert subset.verify_nesting(results) == []


def test_a_tier_never_exceeds_its_budget() -> None:
    _, results = tiers(corpus())

    for budget, result in results.items():
        assert result.achieved_tokens <= budget


def test_the_plan_is_the_same_across_runs() -> None:
    a, _ = tiers(corpus())
    b, _ = tiers(corpus())

    assert a.to_dict() == b.to_dict()


def test_a_different_seed_selects_different_documents() -> None:
    """Otherwise the seed is decorative and the sample cannot be varied."""
    _, a = tiers(corpus(), seed=1)
    _, b = tiers(corpus(), seed=2)

    assert set(a[BUDGETS[0]].doc_ids) != set(b[BUDGETS[0]].doc_ids)


def test_tiers_grow_with_the_budget() -> None:
    _, results = tiers(corpus())

    counts = [len(results[b].doc_ids) for b in BUDGETS]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1], "a larger budget that adds nothing is not a tier"


# -- why the apportionment method was chosen ----------------------------------


def test_largest_remainder_exhibits_the_alabama_paradox() -> None:
    """The method one reaches for first, failing on the property that matters.

    Not a hypothetical: a stratum losing a unit when the budget *grows* means a
    document leaves a larger tier, which is exactly a nesting violation.
    """
    weights = {"a": 6, "b": 6, "c": 2}

    paradox = any(
        subset.largest_remainder(bigger, weights)[s] < subset.largest_remainder(smaller, weights)[s]
        for smaller, bigger in zip(range(1, 60), range(2, 61), strict=True)
        for s in weights
    )

    assert paradox, "if this stops holding the comparison below proves nothing"


def test_the_apportionment_used_is_house_monotone() -> None:
    weights = {"a": 6, "b": 6, "c": 2}

    for smaller, bigger in zip(range(1, 400), range(2, 401), strict=True):
        low, high = subset.apportion(smaller, weights), subset.apportion(bigger, weights)
        for stratum in weights:
            assert high[stratum] >= low[stratum], f"{stratum} shrank from {smaller} to {bigger}"


def test_apportionment_is_house_monotone_on_ragged_weights() -> None:
    """Equal weights are the easy case; a long tail of small strata is not."""
    weights = {f"s{i}": w for i, w in enumerate([1, 1, 2, 3, 5, 8, 13, 21, 34, 500, 900])}

    for budget in range(1, 300):
        low, high = subset.apportion(budget, weights), subset.apportion(budget + 1, weights)
        for stratum in weights:
            assert high[stratum] >= low[stratum]


def test_apportionment_hits_the_budget_exactly_when_it_can() -> None:
    weights = {"a": 100, "b": 50, "c": 25}

    for budget in (1, 7, 60, 174, 175):
        assert sum(subset.apportion(budget, weights).values()) == budget


def test_apportionment_never_exceeds_a_stratums_own_size() -> None:
    weights = {"a": 3, "b": 1000}

    assert subset.apportion(900, weights)["a"] <= 3


# -- the counterexample that forced the contract ------------------------------


def test_filling_the_budget_and_nesting_cannot_both_hold() -> None:
    """The 4/3/2 case, as executable evidence rather than a claim in a doc.

    The fullest selection for a budget of 4 is {4}; for 5 it is {3,2}. Since
    {4} ⊄ {3,2}, an implementation that always packed the budget would have to
    break nesting. This is why shortfall is reported instead.
    """
    rows = [ScanRow("a", "s", 4), ScanRow("b", "s", 3), ScanRow("c", "s", 2)]

    _, results = tiers(rows, [4, 5])

    assert subset.verify_nesting(results) == []
    assert results[5].token_shortfall > 0, "the budget is not packed, and that is the contract"


def test_shortfall_is_reported_rather_than_redistributed() -> None:
    """One large document blocks its stratum; the tokens must not move elsewhere."""
    rows = [ScanRow("big", "a", 10_000)] + [ScanRow(f"small-{i}", "b", 10) for i in range(50)]

    _, results = tiers(rows, [600])
    result = results[600]

    assert result.token_shortfall > 0
    assert result.achieved_tokens <= 600
    assert any(v < 0 for v in result.per_stratum_deviation.values())


def test_a_run_that_cannot_reach_the_budget_still_nests() -> None:
    rows = [ScanRow(f"d{i}", "a", 100) for i in range(10)]

    _, results = tiers(rows, [500, 900, 9000])

    assert subset.verify_nesting(results) == []
    assert results[9000].token_shortfall > 0


def test_a_budget_too_small_for_the_stratification_says_so() -> None:
    """The tier would otherwise come back empty with nothing to explain it.

    Stratifying more finely makes this worse, which is the opposite of what
    someone tuning the stratum key expects, so it has to be said out loud.
    """
    plan = subset.build_plan(corpus(300), [400], score_field="__q")

    assert any("shortest document" in w for w in plan.warnings)
    assert any("400" in w for w in plan.warnings), "the warning must name the budget"


def test_a_workable_budget_does_not_warn_about_starvation() -> None:
    plan = subset.build_plan(corpus(300), [140_000])

    assert not any("shortest document" in w for w in plan.warnings)


def test_a_budget_larger_than_the_corpus_warns_and_takes_everything() -> None:
    rows = [ScanRow(f"d{i}", "a", 100) for i in range(10)]

    plan, results = tiers(rows, [50_000])

    assert plan.warnings, "silently returning a small corpus for a huge budget is a trap"
    assert len(results[50_000].doc_ids) == len(rows)


# -- identity -----------------------------------------------------------------


def test_a_duplicate_id_fails_with_a_count_and_examples() -> None:
    rows = corpus(20) + [ScanRow("doc-0000", "web", 64), ScanRow("doc-0001", "web", 64)]

    with pytest.raises(subset.SubsetError) as excinfo:
        subset.build_plan(rows, BUDGETS)

    message = str(excinfo.value)
    assert "2 repeated" in message
    assert "doc-0000" in message and "doc-0001" in message


def test_an_empty_id_fails() -> None:
    rows = corpus(10) + [ScanRow("  ", "web", 64)]

    with pytest.raises(subset.SubsetError, match="empty id_field"):
        subset.build_plan(rows, BUDGETS)


def test_many_duplicates_are_summarised_not_dumped() -> None:
    rows = [ScanRow(f"dup-{i % 20}", "web", 10) for i in range(200)]

    with pytest.raises(subset.SubsetError, match="more"):
        subset.build_plan(rows, BUDGETS)


# -- the score field ----------------------------------------------------------


def test_a_score_field_set_but_absent_is_an_error_not_a_silent_downgrade() -> None:
    rows = [ScanRow(f"d{i}", "a", 100, score=None) for i in range(10)]

    with pytest.raises(subset.SubsetError) as excinfo:
        subset.build_plan(rows, BUDGETS, score_field="__q")

    message = str(excinfo.value)
    assert "__q" in message
    assert "annotate" in message, "the error should say where a score column comes from"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_quality_scores_are_refused(score) -> None:
    rows = [ScanRow("finite", "a", 100, score=0.5), ScanRow("bad", "a", 100, score=score)]

    with pytest.raises(subset.SubsetError, match="non-finite"):
        subset.build_plan(rows, BUDGETS, score_field="__q")


@pytest.mark.parametrize("tokens", [0, -1])
def test_non_positive_token_counts_are_refused(tokens) -> None:
    rows = [ScanRow("valid", "a", 10), ScanRow("bad", "a", tokens)]

    with pytest.raises(subset.SubsetError, match="non-positive token count"):
        subset.build_plan(rows, BUDGETS)


def test_length_bands_must_be_strictly_increasing() -> None:
    with pytest.raises(subset.SubsetError, match="strictly increasing"):
        subset.build_plan(corpus(10), BUDGETS, length_bands=(512, 128))


def test_without_a_score_field_the_stratum_key_is_source_and_length() -> None:
    plan = subset.build_plan(corpus(60), BUDGETS)

    for key in plan.strata:
        assert len(key.split("|")) == 2


def test_with_a_score_field_deciles_join_the_stratum_key() -> None:
    plan = subset.build_plan(corpus(300), BUDGETS, score_field="__q")

    assert all(len(key.split("|")) == 3 for key in plan.strata)
    assert any(key.endswith("d0") for key in plan.strata)


def test_length_bands_separate_short_from_long_documents() -> None:
    """A subset with the right source mix and the wrong length mix is not neutral."""
    rows = [ScanRow(f"s{i}", "a", 10) for i in range(50)] + [ScanRow(f"l{i}", "a", 5000) for i in range(50)]

    plan = subset.build_plan(rows, [10_000])

    assert len(plan.strata) == 2


# -- reporting ----------------------------------------------------------------


def test_every_tier_reports_the_five_required_figures() -> None:
    _, results = tiers(corpus())

    for result in results.values():
        reported = result.to_dict()
        for key in (
            "achieved_tokens",
            "token_shortfall",
            "per_stratum_deviation",
            "documents_refilled",
            "strata_exhausted",
        ):
            assert key in reported


def test_a_missing_document_is_counted_as_a_refill() -> None:
    """The plan can be inspected and re-run; the corpus may have moved on."""
    rows = corpus(60)
    plan = subset.build_plan(rows, [200_000])
    dropped = plan.strata[next(iter(sorted(plan.strata)))][0]

    results = subset.materialize(plan, [r for r in rows if r.doc_id != dropped])

    assert results[200_000].documents_refilled >= 1


def test_verify_nesting_actually_detects_a_violation() -> None:
    """A checker that cannot fail proves nothing about the runs that pass it."""
    _, results = tiers(corpus())
    results[BUDGETS[-1]].doc_ids = results[BUDGETS[-1]].doc_ids[:1]

    assert subset.verify_nesting(results)


# -- vocabulary ---------------------------------------------------------------


def test_the_module_does_not_claim_to_maximize() -> None:
    """The contract is nesting. A word implying otherwise is the overclaim."""
    import inspect

    source = inspect.getsource(subset).lower()

    assert "maximiz" not in source.replace("maximises", "")
