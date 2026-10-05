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

"""What a candidate threshold would do to a corpus.

Everything here is descriptive. A retention curve answers "how much does this
threshold remove"; it does not answer "is what it removes bad". A corpus can
have a small but valuable tail or a large body of spam, and a distribution
cannot tell them apart. The vocabulary in the report is chosen to keep that
distinction visible: *candidate* threshold, *retention-stable* band, never
*correct* or *recommended*.

Two views of every figure, because one alone misleads. Sampling the same number
of documents from a large source and a small one is unbiased within each source
and skewed at corpus level; weighting each sampled document by how many it
stands for fixes the corpus level and hides small sources. Both are reported and
both are labelled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from nemotron.steps.curate.nemo_curator.runtime.registry import Grid, IntervalGrid, Signal

DEFAULT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

MACRO = "macro"
MICRO = "micro"


class DirectionMismatchError(RuntimeError):
    """The registry's stated direction contradicts the filter's own keep_document."""


# -- threshold semantics ------------------------------------------------------


def keep_mask(scores: np.ndarray, direction: str, *thresholds: float) -> np.ndarray:
    """Which documents a gate keeps, as a boolean mask."""
    if direction == "max":
        return cast(np.ndarray, scores <= thresholds[0])
    if direction == "min":
        return cast(np.ndarray, scores >= thresholds[0])
    if direction == "interval":
        lo, hi = thresholds
        return cast(np.ndarray, (scores >= lo) & (scores <= hi))
    raise ValueError(f"cannot sweep a {direction!r} signal")


def verify_direction(
    signal: Signal,
    scores: Sequence[float],
    probes: int = 5,
    build_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Check the registry against the filter it claims to describe.

    The fast path compares scores to thresholds directly rather than building a
    filter per grid point, which is the difference between a profile that
    finishes and one that reloads a tokenizer a thousand times. That shortcut is
    only sound if the registry's ``direction`` matches what the filter actually
    does, and at least one Curator filter has a parameter named ``max_*`` whose
    ``keep_document`` is ``>=``. So the shortcut is verified against the real
    implementation on real scores before it is used.
    """
    finite = [s for s in scores if s is not None and math.isfinite(s)]
    if not finite:
        return

    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    sample = np.asarray(finite[: max(probes * 4, 20)], dtype=float)

    for i in range(probes):
        t = lo + span * (i + 0.5) / probes
        thresholds = (t, hi + span) if signal.direction == "interval" else (t,)
        try:
            document_filter = signal.build(*thresholds, **(build_kwargs or {}))
            actual = np.array([bool(document_filter.keep_document(float(s))) for s in sample])
        except Exception as exc:  # noqa: BLE001 - a filter we cannot build cannot be verified
            raise DirectionMismatchError(
                f"{signal.name}: could not construct the filter to verify its direction: {exc}"
            ) from exc

        expected = keep_mask(sample, signal.direction, *thresholds)
        if not np.array_equal(actual, expected):
            raise DirectionMismatchError(
                f"{signal.name}: registry declares direction={signal.direction!r} but the "
                f"filter's own keep_document disagrees at threshold {t:g}. Refusing to report "
                "retention figures derived from the wrong comparison."
            )


# -- distribution -------------------------------------------------------------


def as_float_array(values: Sequence[float]) -> np.ndarray:
    """Coerce scores to floats, mapping None to NaN so positions are preserved."""
    return cast(np.ndarray, np.asarray([np.nan if v is None else v for v in values], dtype=float))


def weighted_quantiles(
    values: Sequence[float],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Quantiles, optionally weighting each observation.

    Values and weights are filtered by the same mask. Dropping non-finite scores
    from one array while merely truncating the other would pair each surviving
    score with some other document's weight, which silently biases every
    weighted figure rather than failing.
    """
    arr = as_float_array(values)
    keep = np.isfinite(arr)

    if weights is not None and len(weights) != len(values):
        raise ValueError(
            f"weights must align with values one to one: got {len(weights)} weights for {len(values)} values"
        )

    scores = arr[keep]
    if scores.size == 0:
        return {f"p{q * 100:g}": float("nan") for q in quantiles}

    # One convention for both the weighted and unweighted case. The macro view
    # averages unweighted per-source quantiles and the micro view weights them;
    # if the two used different interpolation rules they would disagree even on
    # a single-source corpus, and the difference would look like a finding about
    # the data rather than an artefact of the arithmetic.
    w = np.ones(scores.size) if weights is None else np.asarray(weights, dtype=float)[keep]
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return {f"p{q * 100:g}": float("nan") for q in quantiles}

    order = np.argsort(scores)
    scores, w = scores[order], w[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / total
    return {f"p{q * 100:g}": float(np.interp(q, cumulative, scores)) for q in quantiles}


def histogram(values: Sequence[float], bins: int = 32) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {"bin_edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


@dataclass
class SignalScores:
    """Scores for one signal across a sample, tagged by source."""

    name: str
    by_source: dict[str, list[float]]

    def flat(self) -> list[float]:
        return [v for values in self.by_source.values() for v in values]

    def micro_weights(self, weights: Mapping[str, float]) -> list[float]:
        return [weights.get(source, 1.0) for source, values in self.by_source.items() for _ in values]


def distribution(scores: SignalScores, source_weights: Mapping[str, float]) -> dict[str, Any]:
    """Both views of one signal's distribution, each labelled.

    Macro gives every source equal weight, which is what you want when asking
    whether a threshold is reasonable for each source. Micro reconstructs the
    corpus, which is what you want when asking how many documents a threshold
    removes overall. They differ whenever sources differ in size, and a figure
    that does not say which one it is cannot be acted on.
    """
    per_source = {source: weighted_quantiles(values) for source, values in sorted(scores.by_source.items())}
    macro = {
        key: float(np.nanmean([q[key] for q in per_source.values()])) if per_source else float("nan")
        for key in (per_source[next(iter(per_source))] if per_source else {})
    }
    return {
        "signal": scores.name,
        "views": {
            MACRO: {"quantiles": macro, "note": "each source weighted equally"},
            MICRO: {
                "quantiles": weighted_quantiles(scores.flat(), weights=scores.micro_weights(source_weights)),
                "note": "each sampled document weighted by the documents it stands for",
            },
        },
        "per_source_quantiles": per_source,
        "histogram": histogram(scores.flat()),
        "n_scored": len(scores.flat()),
    }


# -- retention ----------------------------------------------------------------


def retention_curve(
    scores: Sequence[float],
    direction: str,
    grid: Grid,
    weights: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Fraction retained at each candidate threshold."""
    arr = as_float_array(scores)
    finite = np.isfinite(arr)
    w = np.ones(arr.size) if weights is None else np.asarray(weights, dtype=float)
    total = float(w[finite].sum())

    # Nothing was scored. Reporting 0.0 would read as "this threshold removes the
    # entire corpus", which is a different and much more alarming claim than
    # "this signal produced no usable score".
    if total <= 0:
        return [{"threshold": float(t), "retained": float("nan")} for t in grid.values()]

    out = []
    for t in grid.values():
        mask = keep_mask(arr, direction, t) & finite
        out.append({"threshold": float(t), "retained": float(w[mask].sum() / total)})
    return out


def retention_surface(
    scores: Sequence[float],
    grid: IntervalGrid,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Retention over both bounds of a two-sided gate.

    A matrix, not a curve. Reporting an interval signal as a one-dimensional
    sweep would fix one bound at an unstated value and attribute all of the
    retention to the other.
    """
    arr = as_float_array(scores)
    finite = np.isfinite(arr)
    w = np.ones(arr.size) if weights is None else np.asarray(weights, dtype=float)
    total = float(w[finite].sum())

    lows = grid.lo_grid.values()
    highs = grid.hi_grid.values()

    if total <= 0:
        matrix = [[float("nan") for _ in highs] for _ in lows]
    else:
        matrix = [[float(w[keep_mask(arr, "interval", lo, hi) & finite].sum() / total) for hi in highs] for lo in lows]

    return {
        "kind": "surface",
        "min_axis": [float(v) for v in lows],
        "max_axis": [float(v) for v in highs],
        "retained": matrix,
        # Each marginal takes the best case over the other bound, which is the
        # grid edge. Stated here because a marginal read as "retention at this
        # bound" would understate what the pair removes together.
        "marginal_note": "each marginal maximises over the other bound across the grid",
        "marginal_min": [{"threshold": float(lo), "retained": max(row)} for lo, row in zip(lows, matrix)],
        "marginal_max": [{"threshold": float(hi), "retained": max(col)} for hi, col in zip(highs, zip(*matrix))],
    }


def retention_stable_bands(
    curve: Sequence[Mapping[str, float]],
    min_keep: float = 0.80,
    max_keep: float = 0.995,
) -> list[dict[str, float]]:
    """Threshold ranges whose retention sits inside a configured window.

    ``min_keep`` and ``max_keep`` are analysis constraints the caller chose, not
    properties discovered in the data. They bound which part of the curve gets
    reported as worth considering; they say nothing about quality.
    """
    bands: list[dict[str, float]] = []
    current: list[Mapping[str, float]] = []

    for point in curve:
        if min_keep <= point["retained"] <= max_keep:
            current.append(point)
            continue
        if current:
            bands.append(_band(current))
            current = []
    if current:
        bands.append(_band(current))
    return bands


def _band(points: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """One contiguous threshold range, with the retention AT each end.

    The retention fields are paired with their own threshold by position, not
    stored as an independent min/max. That mattered: retention rises with the
    threshold for a ``max`` signal and falls for a ``min`` one, so taking
    ``min()`` and ``max()`` separately silently swapped which end each figure
    belonged to. Measured on Vietnamese Wikipedia, ``script_ratio`` reported
    0.9522 at threshold 0.9365 where the curve says 0.9939 — a reader choosing
    that threshold from this file got a number four points wrong, in the
    direction that understates what the gate keeps.

    The names say what they are so no reader has to know the direction to pair
    them, and ``direction`` is recorded alongside for the same reason.
    """
    return {
        "threshold_low": float(points[0]["threshold"]),
        "threshold_high": float(points[-1]["threshold"]),
        "retained_at_threshold_low": float(points[0]["retained"]),
        "retained_at_threshold_high": float(points[-1]["retained"]),
    }


# -- co-occurrence ------------------------------------------------------------


def cooccurrence(
    operating_points: Mapping[str, tuple[tuple[float, ...], np.ndarray]],
) -> list[dict[str, Any]]:
    """How often two gates reject the same documents.

    Co-occurrence is only defined at a specific threshold per signal, so every
    entry carries the operating point it was computed at. A bare "82% overlap"
    with no thresholds attached cannot be reproduced or acted on.

    ``operating_points`` maps a signal name to its thresholds and the boolean
    mask of documents that gate *rejects*.
    """
    names = sorted(operating_points)
    out: list[dict[str, Any]] = []

    for i, a in enumerate(names):
        thresholds_a, reject_a = operating_points[a]
        for b in names[i + 1 :]:
            thresholds_b, reject_b = operating_points[b]
            both = int(np.sum(reject_a & reject_b))
            only_a = int(np.sum(reject_a)) or 0
            out.append(
                {
                    "signal_a": a,
                    "thresholds_a": [float(t) for t in thresholds_a],
                    "signal_b": b,
                    "thresholds_b": [float(t) for t in thresholds_b],
                    "rejected_by_a": only_a,
                    "rejected_by_b": int(np.sum(reject_b)),
                    "rejected_by_both": both,
                    "share_of_a_also_rejected_by_b": float(both / only_a) if only_a else 0.0,
                }
            )
    return out


def operating_point_masks(
    scored: Mapping[str, Sequence[float]],
    signals: Iterable[Signal],
    thresholds: Mapping[str, tuple[float, ...]],
) -> dict[str, tuple[tuple[float, ...], np.ndarray]]:
    """Reject masks for each signal at a named operating point."""
    masks: dict[str, tuple[tuple[float, ...], np.ndarray]] = {}
    for signal in signals:
        if signal.name not in thresholds or signal.name not in scored:
            continue
        point = thresholds[signal.name]
        arr = np.asarray([s if s is not None else np.nan for s in scored[signal.name]], dtype=float)
        keep = keep_mask(arr, signal.direction, *point) & np.isfinite(arr)
        masks[signal.name] = (point, ~keep & np.isfinite(arr))
    return masks


def policy_simulation(
    scored: Mapping[str, Sequence[float]],
    signals: Iterable[Signal],
    thresholds: Mapping[str, tuple[float, ...]],
    order: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """What a whole policy removes, as against what each gate removes alone.

    Every per-signal table in this report answers "if this were the only gate".
    A policy is a conjunction, so what it costs is the *union* of the rejects,
    and the union is smaller than the sum by however many documents more than
    one gate rejects. No per-signal table gives that number, and neither does
    ``cooccurrence``: pairwise overlap does not determine the union of three or
    more sets, and it is computed at Curator's defaults rather than at the
    thresholds a person actually chose.

    The figures here are computed from the scores this step already holds, so on
    a sample that is the whole corpus they are exact rather than extrapolated.

    Order-independent per gate:

    ``fails_alone``
        documents this gate rejects, ignoring every other gate.
    ``shared``
        of those, how many some other gate also rejects.
    ``unique``
        of those, how many no other gate rejects. Drop this gate and exactly
        these documents come back.
    ``retention_without``
        what the policy would keep with this gate removed.

    Order-dependent, and reported separately for that reason:

    ``incremental``
        what this gate is the first to reject, under ``order``. Real pipelines
        short-circuit, so this is the shape a ledger records -- and changing the
        order moves the credit between gates without changing the union.
    """
    masks = operating_point_masks(scored, signals, thresholds)
    if len(masks) < 2:  # noqa: PLR2004 - a union of one gate is that gate
        return None

    names = list(order) if order else sorted(masks)
    names = [n for n in names if n in masks]
    length = len(next(iter(masks.values()))[1])
    union = np.zeros(length, dtype=bool)
    for name in names:
        union |= masks[name][1]

    dropped = int(np.sum(union))
    per_gate: list[dict[str, Any]] = []
    seen = np.zeros(length, dtype=bool)
    for name in names:
        point, reject = masks[name]
        others = np.zeros(length, dtype=bool)
        for other in names:
            if other != name:
                others |= masks[other][1]
        incremental = int(np.sum(reject & ~seen))
        seen |= reject
        per_gate.append(
            {
                "signal": name,
                "thresholds": [float(t) for t in point],
                "fails_alone": int(np.sum(reject)),
                "shared": int(np.sum(reject & others)),
                "unique": int(np.sum(reject & ~others)),
                "incremental": incremental,
                "retention_without": 1.0 - (int(np.sum(others)) / length),
            }
        )

    return {
        "documents": length,
        "gates": len(names),
        "dropped": dropped,
        "kept": length - dropped,
        "retention": 1.0 - (dropped / length),
        "sum_of_marginals": sum(g["fails_alone"] for g in per_gate),
        "order": names,
        "order_note": (
            "incremental is computed in the order listed above, not the order the filter "
            "runs its stages — the filter's order is a property of the pipeline, not of "
            "the policy, and this step does not know it. Every other column is "
            "order-independent."
        ),
        "per_gate": per_gate,
    }


# -- the human-readable half --------------------------------------------------
#
# profile_report.json is 314 KB for 24 signals: 64 retention points and a 32-bin
# histogram each. It is the machine's copy. Nobody chooses a threshold from it by
# eye, and a step whose whole purpose is "measure before you gate" fails at that
# purpose if its output cannot be read.

SPARK = "▁▂▃▄▅▆▇█"

#: Retention levels worth naming. A user asking "what does this cost me" is
#: really asking one of these questions.
#:
#: 0.80 is the floor because band_search.min_keep_rate defaults there: a gate
#: keeping less is outside the range this step will propose a band for. It earns
#: its row on concentrated signals, where 99/95/90 all resolve to the same grid
#: point and the table repeats one number three times. On a 20,000-document
#: Vietnamese C4 sample, punctuation and sentence_end_ratio each collapsed to a
#: single threshold across all three levels, and 12 of 21 signals gained a
#: distinct threshold once 0.80 was added.
GATE_LEVELS = (0.99, 0.95, 0.90, 0.80)


def sparkline(counts: Sequence[int]) -> str:
    """A distribution in one line of text."""
    top = max(counts) if counts else 0
    if not top:
        return " " * len(counts)
    return "".join(SPARK[min(len(SPARK) - 1, (c * len(SPARK)) // (top + 1))] for c in counts)


def gate_table(entry: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    """``(threshold, retained, target level)`` at each named retention level.

    Answers the question a threshold is actually chosen to answer — "gate here
    and you keep this much" — rather than leaving the reader to interpolate a
    64-point curve.
    """
    retention = entry.get("retention") or {}
    points = retention.get("points") or []
    if not points or retention.get("kind") != "curve":
        return []

    rising = entry.get("direction") == "max"
    out: list[tuple[float, float, float]] = []
    for level in GATE_LEVELS:
        usable = [p for p in points if p.get("retained", 0.0) >= level]
        if not usable:
            continue
        # The tightest threshold still meeting the level: lowest for a max gate,
        # highest for a min gate. Anything looser keeps more and gates less.
        pick = min if rising else max
        best = pick(usable, key=lambda p: p["threshold"])
        # The level is carried so the achieved retention can overshoot it without
        # looking like an error: the grid is i/63, so the nearest swept point
        # meeting "keep 90%" may well keep 94%.
        out.append((float(best["threshold"]), float(best["retained"]), level))
    return out


def _fmt(value: float, units: str) -> str:
    if units in ("words", "tokens", "characters", "patterns"):
        return f"{value:,.0f}"
    return f"{value:.4f}"


def summarise(report: Mapping[str, Any]) -> str:
    """Render a profile report as something a person reads before choosing gates."""
    corpus = report.get("corpus") or {}
    sampling = report.get("sampling") or {}
    pack = report.get("langpack") or {}
    lines: list[str] = []

    lines.append("# Corpus profile")
    lines.append("")
    lines.append(f"- documents: {corpus.get('document_count', 0):,} from {corpus.get('file_count', 0)} file(s)")
    lines.append(f"- sources: {corpus.get('source_count', 0)}")
    method = sampling.get("method", "?")
    lines.append(f"- sampled: {sampling.get('sampled', 0):,} ({method}, seed {sampling.get('seed')})")
    if pack:
        lines.append(f"- language pack: {pack.get('pack_id')} ({', '.join(pack.get('capabilities') or [])})")
    for note in report.get("notes") or []:
        lines.append(f"- NOTE: {note}")
    lines.append("")
    lines.append("Each signal below shows where the corpus actually sits, then what a gate")
    lines.append("would cost. Pick a threshold from the `gate at` table — those are swept")
    lines.append("grid points, so their retention was measured rather than interpolated.")
    lines.append("")
    lines.append("`max` means the threshold is an upper bound and documents above it are")
    lines.append("removed, `min` a lower bound, `interval` a two-sided pair. Which side of a")
    lines.append("signal is desirable is not stated, because it does not generalise: a ratio")
    lines.append("that marks noise in one language is ordinary in another.")
    lines.append("")

    comp = report.get("language_composition")
    if comp:
        lines.append("## Language composition")
        lines.append("")
        lines.append(f"{comp['scored']:,} sampled documents, scored with the same FastText model")
        lines.append("the filter uses. No signal below can answer this: languages that share a")
        lines.append("script are indistinguishable to a script ratio.")
        lines.append("")
        lines.append("| language | documents | share | median confidence |")
        lines.append("|---|---:|---:|---:|")
        for row in comp["languages"][:8]:
            lines.append(
                f"| `{row['language']}` | {row['documents']:,} | {row['share'] * 100:.2f}% "
                f"| {row['confidence_p50']:.3f} |"
            )
        rest = comp["languages"][8:]
        if rest:
            n = sum(r["documents"] for r in rest)
            lines.append(f"| _{len(rest)} more_ | {n:,} | {n / comp['scored'] * 100:.2f}% | |")
        lines.append("")
        c = comp["confidence"]
        lines.append("Confidence, over every scored document. `min_langid_score` drops documents the")
        lines.append("model was unsure about, whatever language it guessed:")
        lines.append("")
        lines.append("| p1 | p5 | p25 | p50 | p75 |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| {c['p1']:.3f} | {c['p5']:.3f} | {c['p25']:.3f} | {c['p50']:.3f} | {c['p75']:.3f} |")
        lines.append("")

    for entry in report.get("signals") or []:
        name = entry.get("signal", "?")
        units = entry.get("units", "")
        direction = entry.get("direction", "")
        health = entry.get("health") or {}
        scored = health.get("documents_scored", 0)
        failed = health.get("scoring_failures", 0)

        lines.append(f"## {name}")
        lines.append(" · ".join(part for part in (f"`{direction}`", units) if part))
        lines.append("")

        if not scored:
            attempted = health.get("documents_attempted", 0)
            lines.append(f"NOT MEASURED — {failed:,} of {attempted:,} documents failed to score.")
            lines.append(entry.get("retention_suppressed") or "")
            lines.append("")
            continue
        if failed:
            lines.append(f"{scored:,} scored, {failed:,} failed.")

        quantiles = (entry.get("views") or {}).get("micro", {}).get("quantiles") or {}
        if quantiles:
            keys = [k for k in ("p1", "p5", "p25", "p50", "p75", "p95", "p99") if k in quantiles]
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("|" + "---|" * len(keys))
            lines.append("| " + " | ".join(_fmt(quantiles[k], units) for k in keys) + " |")
            lines.append("")

        histogram = entry.get("histogram") or {}
        counts = histogram.get("counts") or []
        edges = histogram.get("bin_edges") or []
        if counts and edges:
            total_counted = sum(counts)
            crowded = total_counted and max(counts) / total_counted > 0.8  # noqa: PLR2004
            if crowded:
                # Linear bins over a heavy tail put everything in one bucket and
                # draw a picture that says nothing. The quantiles above still do.
                busiest = counts.index(max(counts))
                lines.append(
                    f"Distribution is heavily concentrated: {max(counts) / total_counted * 100:.0f}% "
                    f"of documents fall between {_fmt(edges[busiest], units)} and "
                    f"{_fmt(edges[busiest + 1], units)}. Read the quantiles, not a histogram."
                )
            else:
                lines.append(f"```\n{_fmt(edges[0], units)} {sparkline(counts)} {_fmt(edges[-1], units)}\n```")
            lines.append("")

        gates = gate_table(entry)
        if gates:
            bound = "max" if direction == "max" else "min"
            lines.append("This gate applied alone. A policy applies all of them at once —")
            lines.append("see `Policy simulation` for what the combination costs.")
            lines.append("")
            lines.append(f"| to keep | set `{bound}:` | actually keeps | drops |")
            lines.append("|---|---|---|---|")
            total = corpus.get("document_count") or scored
            for threshold, retained, level in gates:
                lines.append(
                    f"| ~{level * 100:.0f}% | {_fmt(threshold, units)} | {retained * 100:.2f}% | "
                    f"{round((1 - retained) * total):,} docs |"
                )
            lines.append("")
        elif (entry.get("retention") or {}).get("kind") == "surface":
            lines.append("Two-sided gate: retention is a surface, not a curve — the cost of a")
            lines.append("lower bound depends on where the upper bound sits. Use the quantiles.")
            lines.append("")

        default = entry.get("curator_default")
        if default:
            lines.append(
                f"Curator's own bound {default['thresholds']} would keep {default['retained'] * 100:.2f}% here."
            )
            lines.append("")

    lines.extend(_simulation_block(report))
    lines.extend(approve_block(report))
    return "\n".join(lines).rstrip() + "\n"


def _simulation_block(report: Mapping[str, Any]) -> list[str]:
    """The union of the proposed gates, which the per-signal tables cannot give."""
    sim = report.get("policy_simulation")
    if not sim:
        return []

    total = sim["documents"]
    out = ["", "## Policy simulation", ""]
    out.append(f"Every table above is one gate applied alone. Applying all {sim['gates']} at once")
    out.append("removes the *union* of what they reject, which is smaller than the sum because")
    out.append("gates overlap. Computed on the profiled sample from the scores already taken,")
    out.append("so these are counts, not estimates.")
    out.append("")
    out.append(f"- keeps **{sim['kept']:,}** of {total:,} (**{sim['retention'] * 100:.2f}%**)")
    out.append(f"- drops **{sim['dropped']:,}** ({(1 - sim['retention']) * 100:.2f}%)")
    out.append(
        f"- the per-gate figures sum to {sim['sum_of_marginals']:,}; the gap to "
        f"{sim['dropped']:,} is documents more than one gate rejects"
    )
    out.append("")
    out.append("`only this` is what a gate is alone responsible for: remove it and exactly")
    out.append("those documents return. A gate whose `only this` is 0 changes nothing about")
    out.append("what this policy delivers.")
    out.append("")
    out.append("| gate | at | fails alone | also caught | only this | keeps without |")
    out.append("|---|---|---:|---:|---:|---:|")
    for g in sorted(sim["per_gate"], key=lambda g: -g["unique"]):
        at = ", ".join(f"{t:g}" for t in g["thresholds"])
        out.append(
            f"| {g['signal']} | {at} | {g['fails_alone']:,} | {g['shared']:,} "
            f"| {g['unique']:,} | {g['retention_without'] * 100:.2f}% |"
        )
    out.append("")
    out.append(sim["order_note"])
    out.append("")
    out.append("| gate | first to reject |")
    out.append("|---|---:|")
    for g in sim["per_gate"]:
        out.append(f"| {g['signal']} | {g['incremental']:,} |")
    out.append("")
    return out


def approve_block(report: Mapping[str, Any]) -> list[str]:
    """A paste-ready ``approve:`` block, commented out, with retention per line.

    Transcribing thresholds by hand is where the typos come from: the numbers are
    swept grid points like 0.015873015873015872, and a mistyped one is a policy
    nobody measured. So the block is written out in full, already valid YAML,
    with the retention each threshold buys on the same line.

    Every line is commented. Nothing here is executable until a person deletes
    the ``#`` characters, and the approver and evidence fields are left empty so
    the run is refused until they are filled. That keeps the decision where it
    belongs while removing the copying.

    The value offered is the most permissive grid point that still meets the
    highest retention level in the gate table -- the least aggressive gate the
    measurement supports. It is a starting point chosen mechanically, not a
    recommendation, and the block says so.
    """
    rows: list[tuple[str, str, float, float]] = []
    for entry in report.get("signals") or []:
        if entry.get("retention_suppressed") or not (entry.get("health") or {}).get("documents_scored"):
            continue
        direction = entry.get("direction")
        if direction not in ("min", "max"):
            # An interval signal's retention is a surface, so there is no single
            # threshold to offer and guessing one would misrepresent the shape.
            continue
        table = gate_table(entry)
        if not table:
            continue
        threshold, retained, _level = table[0]
        rows.append((str(entry.get("signal", "?")), direction, threshold, retained))

    if not rows:
        return []

    width = max(len(name) for name, _, _, _ in rows)
    out = ["", "## Approve block", ""]
    out.append("Copy this into your config for run 2 and remove the `#` from every line you")
    out.append("want. The numbers are measured grid points, so they are correct as written --")
    out.append("but which one is right is your decision, not a measurement. Change any of")
    out.append("them. approver and evidence are empty on purpose: the run is refused until")
    out.append("a person fills them in.")
    out.append("")
    out.append("```yaml")
    out.append("# approve:")
    out.append("#   thresholds:")
    for name, direction, threshold, retained in rows:
        pad = " " * (width - len(name))
        out.append(f"#     - {{signal: {name},{pad} {direction}: {threshold!r}}}   # keeps {retained * 100:.2f}%")
    out.append("#   approver:")
    out.append("#   date:")
    out.append("#   method: manual")
    out.append("#   evidence: >-")
    out.append("#     Why these thresholds, on this corpus. What you looked at.")
    out.append("```")
    out.append("")
    return out
