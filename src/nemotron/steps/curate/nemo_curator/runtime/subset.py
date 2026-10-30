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

"""Stratified token-budget subsets that nest.

A filtering ablation compares two policies. If policy A retains 80% of a corpus
and policy B retains 95%, training on whatever survives measures filter quality
and dataset size at once, and the two cannot be separated afterwards. Fixing the
token budget is what makes the comparison mean anything.

Two properties are wanted from such a subset and **cannot both hold**. With
documents of 4, 3 and 2 tokens, the selection that packs a budget of 4 most
fully is ``{4}`` and the one that packs 5 most fully is ``{3, 2}`` — and
``{4}`` is not a subset of ``{3, 2}``. So a choice is forced, and this module
makes it explicitly:

    **Nesting is guaranteed. Filling the budget is not.**

Three consequences run through everything below:

1. Every tier is produced from **one plan in one run**. Tiers computed
   independently cannot be shown to nest, whatever ordering they used.
2. Selection within a stratum is a **prefix** of a single fixed ordering. A
   prefix of a fixed sequence is nested in any longer prefix by construction,
   which is a stronger statement than a test over the tiers that happened to be
   requested.
3. Per-stratum quotas use a **house-monotone** apportionment, so raising the
   budget can only add documents to a stratum, never swap one out. The obvious
   method — largest remainder — is not house-monotone, and its failure mode is
   silent: see ``largest_remainder`` and the test that pins it.

Shortfall is reported, never redistributed. Moving a stratum's unfilled tokens
to another stratum would change the composition of the subset to hit a number,
which is the opposite of what a stratified subset is for.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

from nemotron.steps.curate.nemo_curator.runtime.determinism import stable_uint64

#: Bumped when a change would alter which documents a given plan selects.
SCHEMA_VERSION = 1

#: Upper edges, in tokens, of the default length bands. A short document and a
#: long one behave differently under nearly every filter, so a subset that did
#: not stratify by length could shift the length distribution while keeping
#: per-source proportions exactly right.
DEFAULT_LENGTH_BANDS: tuple[int, ...] = (128, 512, 2048, 8192)

#: How many examples an error message shows before summarising the rest.
MAX_REPORTED_EXAMPLES = 3


class SubsetError(ValueError):
    """A subset cannot be produced as specified."""


class NestingViolationError(SubsetError):
    """A tier is not contained in a larger tier. Indicates a defect here."""


@dataclass(frozen=True)
class ScanRow:
    """One document, as the planner sees it.

    ``tokens`` is authoritative and comes from the configured tokenizer at a
    pinned revision. Two subsets counted under different revisions are not
    comparable and must not be presented as an ablation pair.
    """

    doc_id: str
    source: str
    tokens: int
    score: float | None = None


@dataclass
class TierResult:
    """What one budget actually produced, including where it fell short."""

    budget: int
    doc_ids: list[str]
    achieved_tokens: int
    token_shortfall: int
    per_stratum_deviation: dict[str, int]
    documents_refilled: int
    strata_exhausted: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "document_count": len(self.doc_ids),
            "achieved_tokens": self.achieved_tokens,
            "token_shortfall": self.token_shortfall,
            "per_stratum_deviation": dict(sorted(self.per_stratum_deviation.items())),
            "documents_refilled": self.documents_refilled,
            "strata_exhausted": sorted(self.strata_exhausted),
        }


# -- stratification -----------------------------------------------------------


def length_band(tokens: int, edges: tuple[int, ...] = DEFAULT_LENGTH_BANDS) -> str:
    """Name the length band a document falls in.

    Bands are labelled by their bounds rather than by index, so a report stays
    readable after someone changes the edges.
    """
    i = bisect.bisect_left(list(edges), tokens)
    if i == 0:
        return f"0-{edges[0]}"
    if i == len(edges):
        return f"{edges[-1]}+"
    return f"{edges[i - 1]}-{edges[i]}"


def score_deciles(scores: list[float]) -> list[float]:
    """Nine interior cut points splitting the observed scores into ten parts.

    Computed from the corpus rather than assumed, because a score column's range
    is a property of the signal and the data, not something a config can state
    in advance.
    """
    ordered = sorted(scores)
    if not ordered:
        return []
    return [ordered[min(len(ordered) - 1, (len(ordered) * d) // 10)] for d in range(1, 10)]


def score_decile(score: float, cut_points: list[float]) -> int:
    return bisect.bisect_right(cut_points, score)


def stratum_key(row: ScanRow, cut_points: list[float] | None, edges: tuple[int, ...]) -> str:
    """``source × length band``, and score decile when a score field is in use.

    A string key rather than a tuple so it survives a JSON round trip into
    ``plan.json`` and back without changing which documents it names.
    """
    parts = [row.source, length_band(row.tokens, edges)]
    if cut_points is not None:
        if row.score is None:
            raise SubsetError(f"document {row.doc_id!r} has no score but score stratification is on")
        parts.append(f"d{score_decile(row.score, cut_points)}")
    return "|".join(parts)


# -- apportionment ------------------------------------------------------------


def largest_remainder(budget: int, weights: dict[str, int]) -> dict[str, int]:
    """Hamilton's method. **Not used** — kept because it is what one reaches for.

    Present so the test suite can demonstrate the Alabama paradox against the
    method actually used, rather than asserting in a comment that it exists.
    """
    total = sum(weights.values())
    if total <= 0:
        return {s: 0 for s in weights}
    exact = {s: budget * w / total for s, w in weights.items()}
    quota = {s: int(v) for s, v in exact.items()}
    remaining = budget - sum(quota.values())
    order = sorted(weights, key=lambda s: (-(exact[s] - quota[s]), s))
    for s in order[:remaining]:
        quota[s] += 1
    return quota


def apportion(budget: int, weights: dict[str, int]) -> dict[str, int]:
    """Split ``budget`` across strata in proportion to ``weights``, house-monotonically.

    Jefferson's divisor method: ``quota_s = floor(weight_s / d)`` for the divisor
    ``d`` that makes the quotas sum to the budget. Equivalently, hand out one
    unit at a time to whichever stratum currently maximises
    ``weight_s / (quota_s + 1)`` — and since units are only ever *added* in that
    process, raising the budget cannot take one away from a stratum.

    That property is the whole point. It is what lets a larger tier be a
    superset of a smaller one instead of merely a similar size.
    """
    if budget <= 0 or not weights:
        return {s: 0 for s in weights}
    total = sum(weights.values())
    if total <= 0:
        return {s: 0 for s in weights}

    # Bracket the divisor, then bisect. Any d in this range gives a quota sum
    # within a few units of the budget; the greedy pass below closes the gap
    # exactly, so float error here costs iterations, never correctness.
    lo, hi = total / (budget + len(weights)), float(total)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid <= 0:
            break
        if sum(int(w / mid) for w in weights.values()) > budget:
            lo = mid
        else:
            hi = mid

    quota = {s: min(w, int(w / hi)) if hi > 0 else 0 for s, w in weights.items()}

    # Close the remaining gap with the exact greedy rule Jefferson describes, so
    # the result is the true apportionment rather than a rounding of it.
    while sum(quota.values()) < budget:
        candidates = [s for s in weights if quota[s] < weights[s]]
        if not candidates:
            break
        best = max(candidates, key=lambda s: (weights[s] / (quota[s] + 1), s))
        quota[best] += 1
    while sum(quota.values()) > budget:
        candidates = [s for s in weights if quota[s] > 0]
        if not candidates:
            break
        worst = min(candidates, key=lambda s: (weights[s] / quota[s], s))
        quota[worst] -= 1

    return quota


# -- planning -----------------------------------------------------------------


def _validate_ids(rows: list[ScanRow]) -> None:
    """Refuse a corpus whose identifiers cannot address a document.

    Nesting is a statement about sets of IDs. If two documents share an ID the
    statement is not false so much as meaningless, so this is a hard error
    rather than a warning.
    """
    blank = [r for r in rows if r.doc_id is None or str(r.doc_id).strip() == ""]
    if blank:
        raise SubsetError(
            f"{len(blank)} document(s) have an empty id_field. A subset is a set of ids; "
            "without them nesting cannot be stated, let alone verified."
        )

    seen: dict[str, int] = {}
    for row in rows:
        seen[row.doc_id] = seen.get(row.doc_id, 0) + 1
    duplicates = sorted(k for k, n in seen.items() if n > 1)
    if duplicates:
        examples = ", ".join(repr(d) for d in duplicates[:MAX_REPORTED_EXAMPLES])
        extra = len(duplicates) - MAX_REPORTED_EXAMPLES
        more = f" (and {extra} more)" if extra > 0 else ""
        raise SubsetError(
            f"id_field is not unique: {len(duplicates)} repeated value(s), e.g. {examples}{more}. "
            "Curator's AddId is positional and does not survive resharding — use the corpus's own id."
        )


def _ordering(rows: list[ScanRow], seed: int) -> list[ScanRow]:
    """The one ordering every tier draws from.

    A float score column with many ties makes ``sort_values`` run-dependent at
    exactly the cut boundary, which breaks nesting without failing anything, so
    the ordering here is total: a hash of the document id, ties broken by the id
    itself. Never Python's ``hash()``, which is salted per process.
    """
    return sorted(rows, key=lambda r: (stable_uint64(r.doc_id, seed), r.doc_id))


@dataclass
class SubsetPlan:
    """Every tier's per-stratum quota, inspectable before anything is written."""

    schema_version: int
    seed: int
    budgets: list[int]
    length_bands: tuple[int, ...]
    score_field: str | None
    score_cut_points: list[float]
    strata: dict[str, list[str]]
    stratum_tokens: dict[str, int]
    quotas: dict[int, dict[str, int]]
    total_tokens: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "budgets": list(self.budgets),
            "length_bands": list(self.length_bands),
            "score_field": self.score_field,
            "score_cut_points": list(self.score_cut_points),
            "total_tokens": self.total_tokens,
            "stratum_tokens": dict(sorted(self.stratum_tokens.items())),
            "strata": {k: list(v) for k, v in sorted(self.strata.items())},
            "quotas": {str(b): dict(sorted(q.items())) for b, q in sorted(self.quotas.items())},
            "warnings": list(self.warnings),
        }


def build_plan(
    rows: list[ScanRow],
    budgets: list[int],
    *,
    seed: int = 0,
    score_field: str | None = None,
    length_bands: tuple[int, ...] = DEFAULT_LENGTH_BANDS,
) -> SubsetPlan:
    """Plan every requested tier at once.

    Producing tiers one run at a time cannot guarantee nesting no matter how
    carefully each run is seeded, because the strata themselves depend on the
    corpus each run happened to see. One plan, one corpus, all tiers.
    """
    if not rows:
        raise SubsetError("no documents to subset")
    if not budgets:
        raise SubsetError("no token budgets requested")
    if any(b <= 0 for b in budgets):
        raise SubsetError(f"token budgets must be positive, got {sorted(budgets)}")
    if not length_bands or any(edge <= 0 for edge in length_bands):
        raise SubsetError(f"length bands must be positive, got {list(length_bands)}")
    if tuple(sorted(set(length_bands))) != length_bands:
        raise SubsetError(f"length bands must be strictly increasing, got {list(length_bands)}")

    _validate_ids(rows)
    invalid_tokens = [row.doc_id for row in rows if row.tokens <= 0]
    if invalid_tokens:
        examples = ", ".join(repr(doc_id) for doc_id in invalid_tokens[:MAX_REPORTED_EXAMPLES])
        raise SubsetError(
            f"{len(invalid_tokens)} document(s) have a non-positive token count, e.g. "
            f"{examples}. A token budget requires every selected document to have a "
            "positive cost."
        )
    budgets = sorted(set(budgets))

    cut_points: list[float] | None = None
    if score_field:
        missing = [r.doc_id for r in rows if r.score is None]
        if missing:
            examples = ", ".join(repr(d) for d in missing[:MAX_REPORTED_EXAMPLES])
            raise SubsetError(
                f"quality_score_field {score_field!r} is set but absent from {len(missing)} "
                f"document(s), e.g. {examples}. Produce it with curate/nemo_curator "
                "mode: annotate or mode: both, or unset quality_score_field."
            )
        non_finite = [row.doc_id for row in rows if row.score is not None and not math.isfinite(float(row.score))]
        if non_finite:
            examples = ", ".join(repr(doc_id) for doc_id in non_finite[:MAX_REPORTED_EXAMPLES])
            raise SubsetError(
                f"quality_score_field {score_field!r} is non-finite for "
                f"{len(non_finite)} document(s), e.g. {examples}. NaN or infinity "
                "cannot define deterministic score deciles."
            )
        scores = [float(row.score) for row in rows if row.score is not None]
        cut_points = score_deciles(scores)

    ordered = _ordering(rows, seed)
    strata: dict[str, list[str]] = {}
    stratum_tokens: dict[str, int] = {}
    for row in ordered:
        key = stratum_key(row, cut_points, length_bands)
        strata.setdefault(key, []).append(row.doc_id)
        stratum_tokens[key] = stratum_tokens.get(key, 0) + row.tokens

    total_tokens = sum(stratum_tokens.values())
    warnings: list[str] = []
    for budget in budgets:
        if budget > total_tokens:
            warnings.append(
                f"budget {budget} exceeds the corpus total of {total_tokens} tokens; "
                "that tier is the whole corpus and its shortfall is the difference."
            )

    quotas = {b: apportion(min(b, total_tokens), stratum_tokens) for b in budgets}
    _assert_quotas_are_monotonic(budgets, quotas)

    # A budget spread across many strata can give each one less than its
    # shortest document, so the tier comes back empty for no visible reason.
    # Stratifying more finely makes this worse, which is the opposite of what a
    # user tuning the stratum key expects.
    smallest: dict[str, int] = {}
    for row in ordered:
        key = stratum_key(row, cut_points, length_bands)
        smallest[key] = min(smallest.get(key, row.tokens), row.tokens)
    for budget in budgets:
        starved = sorted(s for s, q in quotas[budget].items() if q < smallest.get(s, 0))
        if starved:
            warnings.append(
                f"budget {budget}: {len(starved)} of {len(quotas[budget])} strata get a quota "
                f"smaller than their shortest document and will contribute nothing, e.g. "
                f"{', '.join(repr(s) for s in starved[:MAX_REPORTED_EXAMPLES])}. Raise the budget "
                "or stratify more coarsely."
            )

    return SubsetPlan(
        schema_version=SCHEMA_VERSION,
        seed=seed,
        budgets=budgets,
        length_bands=length_bands,
        score_field=score_field,
        score_cut_points=list(cut_points or []),
        strata=strata,
        stratum_tokens=stratum_tokens,
        quotas=quotas,
        total_tokens=total_tokens,
        warnings=warnings,
    )


def _assert_quotas_are_monotonic(budgets: list[int], quotas: dict[int, dict[str, int]]) -> None:
    """Check the property the apportionment is chosen for, on the real numbers.

    Jefferson is house-monotone as a theorem, but the implementation is a
    divisor search with a greedy fixup. Checking the output is cheap and turns a
    subtle nesting failure into a loud one at plan time, before any write.
    """
    for smaller, larger in zip(budgets, budgets[1:], strict=False):
        for stratum, quota in quotas[smaller].items():
            if quotas[larger].get(stratum, 0) < quota:
                raise NestingViolationError(
                    f"stratum {stratum!r} was allocated {quota} tokens at budget {smaller} but "
                    f"{quotas[larger].get(stratum, 0)} at budget {larger}. Raising a budget must "
                    "never take a document away; the apportionment is not house-monotone."
                )


# -- materialization ----------------------------------------------------------


def materialize(plan: SubsetPlan, rows: list[ScanRow]) -> dict[int, TierResult]:
    """Turn a plan into one document-id list per tier.

    Selection is a prefix of each stratum's ordering, stopping before the first
    document that would carry the stratum past its quota. Stopping rather than
    skipping ahead is what makes nesting structural: a prefix of a fixed
    sequence is contained in every longer prefix of it, so the guarantee does
    not depend on which budgets were asked for.
    """
    tokens_by_id = {r.doc_id: r.tokens for r in rows}
    results: dict[int, TierResult] = {}

    for budget in plan.budgets:
        quotas = plan.quotas[budget]
        selected: list[str] = []
        achieved = 0
        deviation: dict[str, int] = {}
        exhausted: list[str] = []
        refilled = 0

        for stratum, doc_ids in sorted(plan.strata.items()):
            quota = quotas.get(stratum, 0)
            taken = 0
            for doc_id in doc_ids:
                cost = tokens_by_id.get(doc_id)
                if cost is None:
                    # Planned but not present in this corpus. Continue down the
                    # same stratum rather than borrowing from another one: the
                    # composition is the point, and the ordering is unchanged,
                    # so the prefix property survives.
                    refilled += 1
                    continue
                if taken + cost > quota:
                    break
                selected.append(doc_id)
                taken += cost
            achieved += taken
            deviation[stratum] = taken - quota
            if taken < quota and all(d in tokens_by_id for d in doc_ids):
                # Every document in the stratum was offered and the quota is
                # still unmet, so the stratum simply has no more to give.
                if sum(tokens_by_id[d] for d in doc_ids) <= quota:
                    exhausted.append(stratum)

        results[budget] = TierResult(
            budget=budget,
            doc_ids=selected,
            achieved_tokens=achieved,
            token_shortfall=budget - achieved,
            per_stratum_deviation=deviation,
            documents_refilled=refilled,
            strata_exhausted=exhausted,
        )

    return results


def verify_nesting(results: dict[int, TierResult]) -> list[str]:
    """Check ``ids(N1) ⊆ ids(N2)`` for every pair, and report what escaped.

    Returned rather than raised so a caller can put the detail in a report. The
    step treats a non-empty result as fatal — a subset family that does not nest
    cannot support the ablation it exists for.
    """
    problems: list[str] = []
    budgets = sorted(results)
    for smaller, larger in zip(budgets, budgets[1:], strict=False):
        lost = sorted(set(results[smaller].doc_ids) - set(results[larger].doc_ids))
        if lost:
            examples = ", ".join(repr(d) for d in lost[:MAX_REPORTED_EXAMPLES])
            problems.append(
                f"{len(lost)} document(s) in tier {smaller} are absent from tier {larger}, e.g. {examples}"
            )
    return problems
