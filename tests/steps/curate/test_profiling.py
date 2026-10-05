# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Distribution, retention, and co-occurrence measurement."""

from __future__ import annotations

import numpy as np
import pytest

from nemotron.steps.curate.nemo_curator.runtime import profiling as p
from nemotron.steps.curate.nemo_curator.runtime import registry as r

# -- threshold semantics ------------------------------------------------------


def test_a_max_gate_keeps_low_scores() -> None:
    mask = p.keep_mask(np.array([0.1, 0.5, 0.9]), "max", 0.5)

    assert list(mask) == [True, True, False]


def test_a_min_gate_keeps_high_scores() -> None:
    mask = p.keep_mask(np.array([0.1, 0.5, 0.9]), "min", 0.5)

    assert list(mask) == [False, True, True]


def test_an_interval_gate_keeps_the_middle() -> None:
    mask = p.keep_mask(np.array([1.0, 5.0, 50.0]), "interval", 2.0, 10.0)

    assert list(mask) == [False, True, False]


def test_a_categorical_signal_cannot_be_swept() -> None:
    with pytest.raises(ValueError, match="cannot sweep"):
        p.keep_mask(np.array([1.0]), "categorical", 0.5)


# -- direction verification ---------------------------------------------------


class _Agrees:
    """A filter whose keep_document matches a declared max direction."""

    def __init__(self, **kwargs):
        self.cutoff = next(iter(kwargs.values()))

    def keep_document(self, score):
        return score <= self.cutoff


class _Disagrees(_Agrees):
    """The failure this check exists for: parameter says max, comparison is >=."""

    def keep_document(self, score):
        return score >= self.cutoff


def _signal(factory) -> r.Signal:
    return r.Signal(
        name="probe",
        factory=factory,
        direction="max",
        units="ratio",
        grid=r.Grid(0.0, 1.0, 8),
        threshold_params=("max_ratio",),
    )


def test_a_matching_direction_verifies_quietly() -> None:
    p.verify_direction(_signal(_Agrees), [0.1, 0.3, 0.5, 0.7, 0.9])


def test_a_contradicting_keep_document_stops_the_run() -> None:
    """Retention derived from the wrong comparison would be silently inverted."""
    with pytest.raises(p.DirectionMismatchError, match="keep_document disagrees"):
        p.verify_direction(_signal(_Disagrees), [0.1, 0.3, 0.5, 0.7, 0.9])


def test_a_filter_that_cannot_be_built_is_reported_not_swallowed() -> None:
    def explode(**kwargs):
        raise RuntimeError("needs a model")

    with pytest.raises(p.DirectionMismatchError, match="could not construct"):
        p.verify_direction(_signal(explode), [0.1, 0.5])


def test_verification_of_an_empty_score_set_is_a_no_op() -> None:
    p.verify_direction(_signal(_Disagrees), [])


# -- distribution -------------------------------------------------------------


def test_quantiles_of_an_empty_sample_are_nan() -> None:
    out = p.weighted_quantiles([], [0.5])

    assert np.isnan(out["p50"])


def test_weighting_moves_the_median_toward_the_heavier_observations() -> None:
    values = [1.0, 1.0, 10.0, 10.0]

    unweighted = p.weighted_quantiles(values, [0.5])["p50"]
    weighted = p.weighted_quantiles(values, [0.5], weights=[1, 1, 100, 100])["p50"]

    assert weighted > unweighted


def test_non_finite_scores_are_dropped_rather_than_poisoning_the_quantiles() -> None:
    out = p.weighted_quantiles([1.0, float("nan"), 3.0], [0.5])

    assert out["p50"] == 2.0


def test_macro_and_micro_differ_when_sources_differ_in_size() -> None:
    """The whole reason both views are reported. A fixture where they agree proves nothing."""
    scores = p.SignalScores(
        name="probe",
        by_source={"big": [1.0] * 50, "small": [100.0] * 5},
    )
    weights = {"big": 1.0, "small": 1.0}

    result = p.distribution(scores, weights)
    macro = result["views"][p.MACRO]["quantiles"]["p50"]
    micro = result["views"][p.MICRO]["quantiles"]["p50"]

    assert macro != micro
    assert result["views"][p.MACRO]["note"]
    assert result["views"][p.MICRO]["note"]


def test_both_views_are_labelled() -> None:
    scores = p.SignalScores(name="probe", by_source={"a": [1.0, 2.0]})

    result = p.distribution(scores, {"a": 1.0})

    assert set(result["views"]) == {p.MACRO, p.MICRO}


def test_per_source_quantiles_are_reported_separately() -> None:
    scores = p.SignalScores(name="probe", by_source={"a": [1.0], "b": [9.0]})

    result = p.distribution(scores, {"a": 1.0, "b": 1.0})

    assert set(result["per_source_quantiles"]) == {"a", "b"}


def test_micro_weights_follow_each_documents_source() -> None:
    scores = p.SignalScores(name="probe", by_source={"a": [1.0, 2.0], "b": [3.0]})

    assert scores.micro_weights({"a": 10.0, "b": 2.0}) == [10.0, 10.0, 2.0]


# -- retention ----------------------------------------------------------------


def test_a_retention_curve_falls_as_a_max_threshold_tightens() -> None:
    scores = [0.1, 0.2, 0.5, 0.9]

    curve = p.retention_curve(scores, "max", r.Grid(0.0, 1.0, 11))
    retained = [pt["retained"] for pt in curve]

    assert retained == sorted(retained), "a tighter max gate cannot keep more"
    assert retained[-1] == 1.0


def test_every_curve_point_names_its_threshold() -> None:
    curve = p.retention_curve([0.5], "max", r.Grid(0.0, 1.0, 4))

    assert all("threshold" in pt and "retained" in pt for pt in curve)


def test_an_interval_signal_produces_a_surface_not_a_curve() -> None:
    """Reporting a two-sided gate as one line would fix the other bound unstated."""
    grid = r.IntervalGrid(lo_grid=r.Grid(0, 4, 5, integral=True), hi_grid=r.Grid(5, 9, 5, integral=True))

    surface = p.retention_surface([1.0, 3.0, 6.0, 20.0], grid)

    assert surface["kind"] == "surface"
    assert len(surface["retained"]) == len(surface["min_axis"])
    assert len(surface["retained"][0]) == len(surface["max_axis"])


def test_a_surface_reports_a_marginal_for_each_axis() -> None:
    grid = r.IntervalGrid(lo_grid=r.Grid(0, 2, 3, integral=True), hi_grid=r.Grid(3, 5, 3, integral=True))

    surface = p.retention_surface([1.0, 4.0], grid)

    assert len(surface["marginal_min"]) == 3
    assert len(surface["marginal_max"]) == 3


def test_retention_is_weighted_when_weights_are_supplied() -> None:
    scores = [0.1, 0.9]

    unweighted = p.retention_curve(scores, "max", r.Grid(0.5, 0.5, 2))[0]["retained"]
    weighted = p.retention_curve(scores, "max", r.Grid(0.5, 0.5, 2), weights=[9.0, 1.0])[0]["retained"]

    assert unweighted == pytest.approx(0.5)
    assert weighted == pytest.approx(0.9)


# -- bands --------------------------------------------------------------------


def test_bands_cover_the_configured_retention_window() -> None:
    curve = [{"threshold": t / 10, "retained": t / 10} for t in range(11)]

    bands = p.retention_stable_bands(curve, min_keep=0.4, max_keep=0.7)

    assert bands
    assert bands[0]["retained_at_threshold_low"] >= 0.4
    assert bands[0]["retained_at_threshold_high"] <= 0.7


def test_each_band_figure_pairs_with_its_own_threshold() -> None:
    """Retention falls with the threshold for a min-direction signal.

    The fields used to be an independent min/max, so for a falling curve the
    reported retention belonged to the OTHER end of the band. Measured on
    Vietnamese Wikipedia, script_ratio reported 0.9522 at threshold 0.9365 where
    the curve says 0.9939 — four points wrong, in the direction that understates
    what the gate keeps.
    """
    falling = [{"threshold": t / 10, "retained": 1.0 - t / 10} for t in range(11)]

    bands = p.retention_stable_bands(falling, min_keep=0.4, max_keep=0.7)

    band = bands[0]
    at = {round(pt["threshold"], 6): pt["retained"] for pt in falling}
    assert band["retained_at_threshold_low"] == at[round(band["threshold_low"], 6)]
    assert band["retained_at_threshold_high"] == at[round(band["threshold_high"], 6)]
    assert band["retained_at_threshold_low"] > band["retained_at_threshold_high"], (
        "a falling curve must report the higher retention at the lower threshold"
    )


def test_a_curve_entirely_outside_the_window_yields_no_band() -> None:
    curve = [{"threshold": 0.1, "retained": 0.01}]

    assert p.retention_stable_bands(curve, min_keep=0.8, max_keep=0.99) == []


def test_a_gap_splits_a_band_in_two() -> None:
    curve = [
        {"threshold": 0.0, "retained": 0.85},
        {"threshold": 0.1, "retained": 0.10},
        {"threshold": 0.2, "retained": 0.90},
    ]

    assert len(p.retention_stable_bands(curve, 0.8, 0.99)) == 2


# -- co-occurrence ------------------------------------------------------------


def test_every_cooccurrence_entry_carries_its_operating_point() -> None:
    """A bare overlap percentage with no thresholds cannot be reproduced."""
    points = {
        "a": ((0.25,), np.array([True, True, False, False])),
        "b": ((0.5,), np.array([True, False, True, False])),
    }

    entries = p.cooccurrence(points)

    assert entries
    for entry in entries:
        assert entry["thresholds_a"] and entry["thresholds_b"]
        assert entry["signal_a"] and entry["signal_b"]


def test_cooccurrence_counts_documents_both_gates_reject() -> None:
    points = {
        "a": ((0.25,), np.array([True, True, False])),
        "b": ((0.5,), np.array([True, False, False])),
    }

    entry = p.cooccurrence(points)[0]

    assert entry["rejected_by_a"] == 2
    assert entry["rejected_by_b"] == 1
    assert entry["rejected_by_both"] == 1
    assert entry["share_of_a_also_rejected_by_b"] == pytest.approx(0.5)


def test_a_single_signal_has_no_pairs() -> None:
    assert p.cooccurrence({"a": ((0.1,), np.array([True]))}) == []


# -- policy simulation --------------------------------------------------------


def _named(name: str) -> r.Signal:
    return r.Signal(
        name=name,
        factory=_Agrees,
        direction="max",
        units="ratio",
        grid=r.Grid(0.0, 1.0, 8),
        threshold_params=("max_ratio",),
    )


def _overlapping_pair():
    """A rejects 1,000 docs, B rejects 800, and they share 600.

    The union is 1,200, not 1,800. Summing per-gate figures overcounts by exactly
    the overlap, which is the mistake this section exists to prevent.
    """
    a = [1.0] * 1000 + [0.0] * 1000
    b = [0.0] * 400 + [1.0] * 800 + [0.0] * 800
    return {"a": a, "b": b}, [_named("a"), _named("b")], {"a": (0.5,), "b": (0.5,)}


def test_the_policy_drops_the_union_not_the_sum() -> None:
    scored, signals, thresholds = _overlapping_pair()

    sim = p.policy_simulation(scored, signals, thresholds)

    assert sim["dropped"] == 1200, "1,000 + 800 - 600 shared"
    assert sim["sum_of_marginals"] == 1800
    assert sim["kept"] == 800
    assert sim["retention"] == pytest.approx(0.4)


def test_each_gate_reports_what_only_it_rejects() -> None:
    """`unique` is the number that answers 'can I delete this gate'."""
    scored, signals, thresholds = _overlapping_pair()

    by_name = {g["signal"]: g for g in p.policy_simulation(scored, signals, thresholds)["per_gate"]}

    assert by_name["a"]["fails_alone"] == 1000
    assert by_name["a"]["shared"] == 600
    assert by_name["a"]["unique"] == 400
    assert by_name["b"]["unique"] == 200
    # Remove a and the 1,000 it rejects stop mattering; only b's 800 remain.
    assert by_name["a"]["retention_without"] == pytest.approx(0.6)
    assert by_name["b"]["retention_without"] == pytest.approx(0.5)


def test_a_gate_no_other_gate_needs_reports_zero_unique() -> None:
    """foreign_script_ratio rejected 194 documents on real data and none alone."""
    scored = {"wide": [1.0] * 100 + [0.0] * 100, "narrow": [1.0] * 50 + [0.0] * 150}
    signals = [_named("wide"), _named("narrow")]

    by_name = {
        g["signal"]: g for g in p.policy_simulation(scored, signals, {"wide": (0.5,), "narrow": (0.5,)})["per_gate"]
    }

    assert by_name["narrow"]["fails_alone"] == 50
    assert by_name["narrow"]["unique"] == 0
    assert by_name["narrow"]["retention_without"] == pytest.approx(0.5), "deleting it changes nothing"


def test_incremental_credit_moves_with_the_order_but_the_union_does_not() -> None:
    """A ledger's per-gate counts are an artefact of stage order, not a property."""
    scored, signals, thresholds = _overlapping_pair()

    forward = p.policy_simulation(scored, signals, thresholds, order=["a", "b"])
    reverse = p.policy_simulation(scored, signals, thresholds, order=["b", "a"])

    assert [g["incremental"] for g in forward["per_gate"]] == [1000, 200]
    assert [g["incremental"] for g in reverse["per_gate"]] == [800, 400]
    assert forward["dropped"] == reverse["dropped"] == 1200
    assert {g["signal"]: g["unique"] for g in forward["per_gate"]} == {
        g["signal"]: g["unique"] for g in reverse["per_gate"]
    }, "unique is order-independent; incremental is not"


def test_one_gate_is_not_a_policy() -> None:
    """A union of one set is that set, and reporting it as a combination misleads."""
    assert p.policy_simulation({"a": [1.0, 0.0]}, [_named("a")], {"a": (0.5,)}) is None


def test_masks_mark_rejections_not_keeps() -> None:
    signal = _signal(_Agrees)
    masks = p.operating_point_masks({"probe": [0.1, 0.9]}, [signal], {"probe": (0.5,)})

    thresholds, reject = masks["probe"]
    assert thresholds == (0.5,)
    assert list(reject) == [False, True]


# -- regressions --------------------------------------------------------------


def test_weights_follow_their_own_documents_past_a_non_finite_score() -> None:
    """Filtering values but truncating weights pairs each score with a stranger's weight."""
    heavy_is_first_finite = p.weighted_quantiles([float("nan"), 1.0, 100.0], [0.5], weights=[1.0, 100.0, 1.0])["p50"]

    assert heavy_is_first_finite < 2.0, "the weight of 100 belongs to the document scoring 1.0"


def test_misaligned_weights_are_refused_rather_than_truncated() -> None:
    with pytest.raises(ValueError, match="align with values one to one"):
        p.weighted_quantiles([1.0, 2.0], [0.5], weights=[1.0])


def test_macro_and_micro_agree_on_a_single_source_corpus() -> None:
    """A difference here would be an artefact of interpolation, read as a finding."""
    scores = p.SignalScores(name="probe", by_source={"only": [1.0, 2.0, 3.0, 4.0, 7.0]})

    result = p.distribution(scores, {"only": 1.0})

    assert result["views"][p.MACRO]["quantiles"]["p50"] == pytest.approx(result["views"][p.MICRO]["quantiles"]["p50"])


def test_a_signal_that_scored_nothing_reports_nan_not_zero_retention() -> None:
    """0.0 reads as 'removes the whole corpus', a different and alarming claim."""
    curve = p.retention_curve([float("nan")] * 5, "max", r.Grid(0.0, 1.0, 4))

    assert all(np.isnan(point["retained"]) for point in curve)


def test_a_surface_over_nothing_is_nan_too() -> None:
    grid = r.IntervalGrid(r.Grid(0, 2, 3, integral=True), r.Grid(3, 5, 3, integral=True))

    surface = p.retention_surface([float("nan"), float("nan")], grid)

    assert all(np.isnan(v) for row in surface["retained"] for v in row)


def test_a_surface_says_how_its_marginals_were_taken() -> None:
    """A marginal read as 'retention at this bound' would understate the pair."""
    grid = r.IntervalGrid(r.Grid(0, 2, 3, integral=True), r.Grid(3, 5, 3, integral=True))

    assert "maximises over the other bound" in p.retention_surface([1.0], grid)["marginal_note"]


# -- the readable half ---------------------------------------------------------
#
# profile_report.json is 314 KB for 24 signals — 64 retention points and a 32-bin
# histogram each. Nobody picks a threshold from it by eye, and a step whose whole
# purpose is "measure before you gate" has not delivered until the measurement
# can be read without writing a script first.


def _entry(direction="max", **over):
    # A max gate keeps MORE as the threshold rises; a min gate keeps LESS.
    # Getting this backwards in a fixture makes the code look wrong.
    if direction == "min":
        curve = [{"threshold": t / 20, "retained": 1.0 - t / 21} for t in range(21)]
    else:
        curve = [{"threshold": t / 20, "retained": t / 21} for t in range(21)]
    base = {
        "signal": "demo",
        "direction": direction,
        "units": "ratio",
        "health": {"documents_attempted": 100, "documents_scored": 100, "scoring_failures": 0},
        "views": {"micro": {"quantiles": {"p1": 0.1, "p50": 0.5, "p99": 0.9}}},
        "histogram": {"bin_edges": [0.0, 0.5, 1.0], "counts": [40, 60]},
        "retention": {"kind": "curve", "points": curve},
    }
    base.update(over)
    return base


def _report(*entries, **over):
    doc = {
        "corpus": {"document_count": 100, "file_count": 1, "source_count": 1},
        "sampling": {"sampled": 100, "method": "hash-bottom-k", "seed": 0},
        "signals": list(entries),
        "notes": [],
    }
    doc.update(over)
    return doc


def test_a_max_gate_names_the_bound_the_policy_must_set() -> None:
    """The summary is only useful if it says which key to write."""
    text = p.summarise(_report(_entry("max")))

    assert "set `max:`" in text


def test_a_min_gate_names_the_other_bound() -> None:
    text = p.summarise(_report(_entry("min")))

    assert "set `min:`" in text


def test_the_gate_table_reports_documents_dropped() -> None:
    """'99.18%' is abstract; '164 docs' is the thing being thrown away."""
    text = p.summarise(_report(_entry("max")))

    assert "docs |" in text
    assert "actually keeps" in text


def test_a_concentrated_signal_still_gets_one_gate_row_that_gates() -> None:
    """Three levels resolving to one threshold is a table that answers nothing.

    Retention that sits above 0.90 until the very top of the range makes 99/95/90
    all pick the same grid point, so the reader sees one number written three
    times and no threshold that costs anything. punctuation on Vietnamese C4 did
    exactly this: every level said max 1.0, keeps 100.00%, drops 0 docs. The 0.80
    level is the one that separates.
    """
    concentrated = _entry(
        retention={
            "kind": "curve",
            "points": [
                {"threshold": 0.0, "retained": 0.10},
                {"threshold": 0.5, "retained": 0.82},
                {"threshold": 1.0, "retained": 1.00},
            ],
        }
    )

    rows = p.gate_table(concentrated)

    assert [level for _, _, level in rows] == [0.99, 0.95, 0.90, 0.80]
    assert len({threshold for threshold, _, _ in rows}) == 2, rows
    assert rows[0][0] == rows[1][0] == rows[2][0] == 1.0
    assert rows[3][0] == 0.5, "the 0.80 row is the only one that names a gate with a cost"


def test_the_gate_table_floor_matches_the_band_search_floor() -> None:
    """A row proposing retention the band search will not report is a dead end."""
    assert min(p.GATE_LEVELS) == 0.80


def test_a_signal_that_scored_nothing_says_so_instead_of_showing_quantiles() -> None:
    dead = _entry(
        health={"documents_attempted": 100, "documents_scored": 0, "scoring_failures": 100},
        retention_suppressed="0 of 100 produced a usable score",
    )

    text = p.summarise(_report(dead))

    assert "NOT MEASURED" in text
    assert "0 of 100" in text


def test_a_concentrated_distribution_is_described_not_drawn() -> None:
    """Linear bins over a heavy tail draw a picture that says nothing."""
    skewed = _entry(histogram={"bin_edges": [0.0, 0.5, 1.0], "counts": [95, 5]})

    text = p.summarise(_report(skewed))

    assert "heavily concentrated" in text
    assert "▁" not in text, "a 95%-in-one-bin sparkline is worse than a sentence"


def test_a_two_sided_signal_explains_why_it_has_no_gate_table() -> None:
    surface = _entry("interval", retention={"kind": "surface", "points": []})

    text = p.summarise(_report(surface))

    assert "surface, not a curve" in text


def test_report_level_notes_reach_the_summary() -> None:
    """A caveat only in the JSON is a caveat nobody reads."""
    text = p.summarise(_report(_entry(), notes=["cooccurrence not computed"]))

    assert "cooccurrence not computed" in text


def _report_with_a_curve() -> dict:
    points = [{"threshold": i / 63, "retained": 1.0 - i / 200} for i in range(64)]
    return {
        "signals": [
            {
                "signal": "symbol_to_word",
                "direction": "max",
                "units": "ratio",
                "health": {"documents_scored": 1000, "scoring_failures": 0},
                "retention": {"kind": "curve", "points": points},
            }
        ]
    }


def test_the_approve_block_is_inert_until_a_person_edits_it() -> None:
    """The block removes the transcription, not the decision.

    Thresholds are swept grid points like 0.015873015873015872; copying one by
    hand is where a policy nobody measured comes from. So the block is emitted
    ready to paste — and every line is commented, so nothing is executable until
    a person deletes the hashes, with approver and evidence left empty so the run
    is refused until they are filled.
    """
    block = p.approve_block(_report_with_a_curve())

    body = [ln for ln in block if ln.startswith("#") or ln.startswith("#   ")]
    assert body, "the block must be emitted"
    yaml_lines = block[block.index("```yaml") + 1 : block.index("```")]
    assert all(ln.startswith("#") for ln in yaml_lines), "every YAML line must be commented"
    assert any(ln.strip() == "#   approver:" for ln in yaml_lines), "approver must be present and empty"
    assert any(ln.strip() == "#   evidence: >-" for ln in yaml_lines)


def test_the_approve_block_keeps_full_precision_and_states_the_cost() -> None:
    """A rounded grid point is a threshold nobody measured, and a threshold with
    no retention beside it is a number with no consequence attached."""
    block = p.approve_block(_report_with_a_curve())
    line = next(ln for ln in block if "symbol_to_word" in ln)

    assert "keeps " in line and "%" in line
    value = line.split("max: ")[1].split("}")[0]

    grid = {pt["threshold"] for pt in _report_with_a_curve()["signals"][0]["retention"]["points"]}
    assert float(value) in grid, "the offered value is not one of the measured points"
    # repr round-trips exactly; a rounded value would not reproduce its own text.
    assert value == repr(float(value)), f"grid point was rounded: {value}"


def test_a_two_sided_signal_is_not_given_a_single_threshold() -> None:
    """An interval signal's retention is a surface. Offering one number for it
    would misrepresent the shape, so it is left out rather than guessed."""
    report = _report_with_a_curve()
    report["signals"][0]["direction"] = "interval"

    assert p.approve_block(report) == []


def test_a_suppressed_signal_is_never_offered() -> None:
    """A signal the profiler withheld for scoring too few documents has no
    measured retention, so it must not appear as something to approve."""
    report = _report_with_a_curve()
    report["signals"][0]["retention_suppressed"] = "scored 12% of the sample"

    assert p.approve_block(report) == []
