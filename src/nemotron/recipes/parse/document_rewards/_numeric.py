"""One-to-one, unit-aware numeric fact matching."""
from __future__ import annotations

import math
from typing import Any, Callable, Hashable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

Numeric = tuple[float, str]
Fact = tuple[Hashable, Numeric]
MAX_ASSIGNMENT_EDGES = 65536


def compare_numeric(
    predicted: Numeric,
    reference: Numeric,
    *,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Unit-safe symmetric tolerance with a full-credit acceptance plateau.

    The absolute floor is an explicit training policy for values near zero.
    Invalid/non-finite numbers and mismatched units earn no numeric credit.
    """
    if not (math.isfinite(relative_tolerance) and 0 <= relative_tolerance < 1):
        raise ValueError("relative_tolerance must be finite and in [0, 1)")
    if not (math.isfinite(absolute_tolerance) and absolute_tolerance > 0):
        raise ValueError("absolute_tolerance must be finite and positive")
    p, r = float(predicted[0]), float(reference[0])
    compatible = predicted[1] == reference[1]
    finite = math.isfinite(p) and math.isfinite(r)
    result: dict[str, Any] = {
        "accepted": False, "exact": False, "smooth": 0.0,
        "unit_compatible": compatible, "finite": finite,
        "absolute_error": None, "relative_error": None,
        "acceptance_bound": None,
    }
    if not compatible or not finite:
        return result
    error = abs(p - r)
    scale = max(abs(p), abs(r))
    bound = max(absolute_tolerance, relative_tolerance * scale)
    relative = abs(p / scale - r / scale) if scale else 0.0
    accepted = error <= bound
    # The accepted plateau eliminates a preference for exact-but-partly-wrong
    # charts over completely accepted charts. Catastrophic errors get zero aux.
    smooth = 1.0 if accepted else 1.0 / (1.0 + (error - bound) / bound)
    if not accepted and (relative >= 0.5 or error >= 10.0 * bound):
        smooth = 0.0
    result.update(
        accepted=accepted, exact=error == 0.0, smooth=smooth,
        absolute_error=error if math.isfinite(error) else None, relative_error=relative, acceptance_bound=bound,
    )
    return result


def score_facts(
    predicted_facts: Iterable[Fact],
    reference_facts: Iterable[Fact],
    *,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 1e-8,
    prediction_fact_limit: int | None = None,
    key_similarity: Callable[[Hashable, Hashable], float] | None = None,
) -> dict[str, Any]:
    """Maximize accepted matches first, with no prediction occurrence reuse.

    Strict semantic keys must already preserve scope, field role and bindings.
    An optional key_similarity is *auxiliary only* and must be supplied by a
    role/scope-aware caller. It never makes a non-exact key an accepted fact.

    The returned scalar is primary accepted F1, with no auxiliary mixture.
    Thus any strictly better accepted F1 has strictly better scalar regardless
    of prediction counts. ``auxiliary`` is separate for an explicitly controlled
    optimizer path. No data-dependent count cap changes semantic quality.
    """
    predicted, reference = list(predicted_facts), list(reference_facts)
    npred, nref = len(predicted), len(reference)
    if prediction_fact_limit is not None and (
        not isinstance(prediction_fact_limit, int)
        or isinstance(prediction_fact_limit, bool)
        or prediction_fact_limit < nref
    ):
        raise ValueError("prediction_fact_limit must be an integer >= reference fact count")
    result: dict[str, Any] = {
        "score": 0.0, "primary_f1": 0.0, "precision": 0.0, "recall": 0.0,
        "accepted_matches": 0, "matched_occurrences": 0,
        "points_pred": npred, "points_ref": nref,
        "false_positives": npred, "false_negatives": nref,
        "auxiliary": 0.0, "max_auxiliary_budget": 0.0, "auxiliary_is_separate": True,
        "prediction_fact_limit": prediction_fact_limit, "count_contract_valid": True,
        "requested_count_limit_exceeded": prediction_fact_limit is not None and npred > prediction_fact_limit,
        "ordering_guarantee": "scalar_equals_accepted_factual_f1_all_prediction_counts",
        "numeric_policy_version": "v78_unitsafe_symmetric_relative_absolute_v1",
        "auxiliary_label_matching_skipped": False, "auxiliary_assignment_approximate": False,
        "relative_tolerance": relative_tolerance, "absolute_tolerance": absolute_tolerance,
        "matches": [],
    }
    # Validate policy even when there are no matching points.
    compare_numeric((0.0, ""), (0.0, ""), relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance)
    if not nref:
        result.update(score=float(not npred), primary_f1=float(not npred),
                      precision=float(not npred), recall=1.0)
        return result
    if not npred:
        return result
    # Exact-key blocks make the usual path linear in the number of unique
    # associated facts. Only repeated occurrences require a local assignment.
    # Optional guarded near-label matching uses one joint assignment so the
    # same prediction occurrence cannot earn credit twice.
    use_label_aux = key_similarity is not None and npred * nref <= MAX_ASSIGNMENT_EDGES
    result["auxiliary_label_matching_skipped"] = key_similarity is not None and not use_label_aux
    if not use_label_aux:
        from collections import defaultdict
        pb, rb = defaultdict(list), defaultdict(list)
        for pi, (key, value) in enumerate(predicted):
            pb[(key, value[1])].append(pi)
        for ri, (key, value) in enumerate(reference):
            rb[(key, value[1])].append(ri)
        blocks = [(indices, rb[key]) for key, indices in pb.items() if key in rb]
    else:
        blocks = [(list(range(npred)), list(range(nref)))]
    accepted_count, partial_total = 0, 0.0
    for pindices, rindices in blocks:
        if len(pindices) * len(rindices) > MAX_ASSIGNMENT_EDGES:
            # Same semantic key/unit gives ordered acceptance intervals. Match
            # the lowest eligible numeric occurrence first: maximum cardinality
            # for these monotone intervals, without a quadratic dense matrix.
            result["auxiliary_assignment_approximate"] = True
            pp = sorted((float(predicted[i][1][0]), i) for i in pindices
                        if math.isfinite(float(predicted[i][1][0])))
            rr = sorted((float(reference[i][1][0]), i) for i in rindices
                        if math.isfinite(float(reference[i][1][0])))
            ip = ir = 0
            premaining, rremaining = [], []
            while ip < len(pp) and ir < len(rr):
                pv, pi = pp[ip]; rv, ri = rr[ir]
                comp = compare_numeric(predicted[pi][1], reference[ri][1],
                                       relative_tolerance=relative_tolerance,
                                       absolute_tolerance=absolute_tolerance)
                if comp["accepted"]:
                    accepted_count += 1; partial_total += 1.0
                    result["matches"].append({"prediction_index": pi, "reference_index": ri,
                                              "accepted": True, "exact_key": True,
                                              "auxiliary_credit": 1.0})
                    ip += 1; ir += 1
                elif pv < rv:
                    premaining.append(pi); ip += 1
                else:
                    rremaining.append(ri); ir += 1
            premaining.extend(i for _, i in pp[ip:])
            rremaining.extend(i for _, i in rr[ir:])
            for pi, ri in zip(premaining, rremaining):
                comp = compare_numeric(predicted[pi][1], reference[ri][1],
                                       relative_tolerance=relative_tolerance,
                                       absolute_tolerance=absolute_tolerance)
                partial = comp["smooth"]
                if not partial:
                    continue
                partial_total += partial
                result["matches"].append({"prediction_index": pi, "reference_index": ri,
                                          "accepted": False, "exact_key": True,
                                          "auxiliary_credit": partial})
            continue
        utility = np.zeros((len(pindices), len(rindices)), dtype=np.float64)
        edges: dict[tuple[int, int], tuple[bool, float, bool]] = {}
        accepted_priority = float(min(len(pindices), len(rindices)) + 1)
        for local_pi, pi in enumerate(pindices):
            pkey, pvalue = predicted[pi]
            for local_ri, ri in enumerate(rindices):
                rkey, rvalue = reference[ri]
                exact_key = pkey == rkey
                similarity = 1.0 if exact_key else 0.0
                if not exact_key and use_label_aux:
                    raw = float(key_similarity(pkey, rkey))
                    if not math.isfinite(raw) or not 0 <= raw <= 1:
                        raise ValueError("key_similarity must return a finite number in [0, 1]")
                    similarity = 0.25 * raw
                if not similarity:
                    continue
                comp = compare_numeric(pvalue, rvalue, relative_tolerance=relative_tolerance,
                                       absolute_tolerance=absolute_tolerance)
                accepted = exact_key and comp["accepted"]
                partial = similarity * comp["smooth"]
                if not accepted and not partial:
                    continue
                utility[local_pi, local_ri] = accepted_priority * accepted + partial
                edges[local_pi, local_ri] = (accepted, partial, exact_key)
        prow, rcol = linear_sum_assignment(-utility)
        for local_pi, local_ri in zip(prow.tolist(), rcol.tolist()):
            edge = edges.get((local_pi, local_ri))
            if edge is None:
                continue
            accepted, partial, exact_key = edge
            accepted_count += int(accepted)
            partial_total += partial
            result["matches"].append({
                "prediction_index": pindices[local_pi], "reference_index": rindices[local_ri],
                "accepted": accepted, "exact_key": exact_key, "auxiliary_credit": partial,
            })
    result["matches"].sort(key=lambda m: (m["prediction_index"], m["reference_index"]))
    primary = 2.0 * accepted_count / (npred + nref)
    auxiliary = 2.0 * partial_total / (npred + nref)
    score = primary
    result.update(
        score=score, primary_f1=primary, precision=accepted_count / npred,
        recall=accepted_count / nref, accepted_matches=accepted_count,
        matched_occurrences=len(result["matches"]), false_positives=npred - accepted_count,
        false_negatives=nref - accepted_count, auxiliary=auxiliary,
    )
    return result
