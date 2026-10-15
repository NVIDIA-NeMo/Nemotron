# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a run segment by segment (S1 to S10) against the oracle, and end to end.

Each segment is a step the investigation must get right; each check reads only the run's own record.
A segment that does not apply to a world (no screenshot to read, no proposal expected) scores None.

S1 first look      the note before the first tool call repeats values from the attached screenshot
S2 reconcile       a query or code result states both periods' spend; no misbucketed total is reported
S3 decompose       the report names the driver(s) that really moved, in the right direction
S4 attribute       the report cites every cause (the causal change, the price schedule) and blames no decoy
S5 retrieve        the current price schedule and (if published only as images) both eval report pages were opened
S6 transcribe      every needed price (and eval number read from an image) appears exactly in code; no misreads
S7 compute         the filed proposal's savings fit its own routes; without a proposal, cost per ticket is right
S8 decide          the filed routing is the cheapest one within the quality bar, or no proposal when none is due
S9 report          spend and ticket changes stated correctly, evidence labelled, no true cause ruled out
S10 finish         the run submitted a report without an error
"""

import datetime as dt
import json
import re

import world

from .truth import _days, _metrics, kpi_cards, truth, unit_cost

SEGMENTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
NAMES = {"S1": "first look", "S2": "reconcile", "S3": "decompose", "S4": "attribute", "S5": "retrieve",
         "S6": "transcribe", "S7": "compute", "S8": "decide", "S9": "report", "S10": "finish"}
SWITCH_CLAIM = re.compile(r"(switch\w*|migrat\w*|moved|mov\w+|rerout\w*|roll\w*[- ]?out|replac\w*)[^.;]{0,60}\bmodel|\bmodel\b[^.;]{0,40}(switch|change|migrat|rollout|replac)", re.I)
NEGATION = re.compile(r"\b(no|not|without|unchanged|neither|nor|only|ruled out|rather than|instead of)\b|n't", re.I)
PRICE_CLAIM = re.compile(r"price (increase|hike|rise|change)|rais(ed|ing) ([\w.-]+ ){0,4}?prices?|prices? (rose|went up|increased|were raised)|higher (list |per-token )?prices?|new (price )?schedule|price schedule (took effect|came into effect|raised|increased|went into effect)|(new|updated|higher) (list )?rates?", re.I)
VOLUME_CLAIM = re.compile(r"(more tickets|ticket volume (grew|rose|increased|jumped)|volume (growth|increase|surge|jump)|tickets (grew|rose|increased|jumped) (by )?(about |roughly |~)?(1\d|[2-9]\d)%)", re.I)
MIX_CLAIM = re.compile(r"(?<!weekday-)(?<!weekday )\bmix\b|share of (tickets|technical|hard|complex)|more (of the )?(hard|complex|technical|expensive)|shift(ed)? toward|fewer (easy|simple|account|password)", re.I)
USAGE_CLAIM = re.compile(r"retr(y|ies)|requests? per ticket|input tokens? per ticket|timeout|memory window|context window|longer (prompts|context)", re.I)
CLEARED = re.compile(r"\b(no|not|none|neither|nor|unrelated|irrelevant|non-routing|ruled out|cannot)\b|n't", re.I)
HEDGE = re.compile(r"(\b(not|cannot)|n't)\s+(be\s+)?(yet\s+)?(proven|proved|prove|confirmed|confirm|certain|established|definitive|conclusive)\w*", re.I)
NOT_OBSERVED = re.compile(r"project|propos|forecast|expected|saving", re.I)
NO_CHANGE = re.compile(r"no (single |clear |material |significant |major )?(cause|driver|change|anomal)|within normal|normal (variation|growth|fluctuation)|small|modest|slight|in line with", re.I)


def _claims(pattern, text):
    """Sentences that make the claim, skipping negated ones ("no model change", "not a price increase")."""
    return [x for x in re.split(r"(?<=[.;!?])\s+", text) if pattern.search(x) and not NEGATION.search(x)]


def _units(*texts):
    return [u for text in texts for u in re.split(r"(?<=[.;!?])\s+", str(text or "")) if u.strip()]


def _named(ids, units):
    """The ids the report puts forward as part of the explanation: named in a sentence or evidence item that does not
    clear them. Hedging ("a hypothesis, not proven") still puts an id forward."""
    return {i for u in units if not CLEARED.search(HEDGE.sub("", u)) for i in ids if i in u}


def _report_text(rep, fields=("finding", "how_it_happened")):
    return " ".join(str(rep.get(f, "")) for f in fields)


def _causal_text(rep):
    """What the report presents as the cause: finding, mechanism, and evidence that is not a rejected candidate."""
    parts = [_report_text(rep, ("finding", "how_it_happened", "engineer_summary")), json.dumps(rep.get("attribution") or "")]
    parts += [str(e.get("claim", "")) + " " + str(e.get("source", "")) for e in rep.get("evidence") or [] if isinstance(e, dict)]
    return " ".join(parts)


def _attributions(rep):
    """The attribution field's entries: one object, or one per driver."""
    a = rep.get("attribution")
    return [a] if isinstance(a, dict) else [x for x in a if isinstance(x, dict)] if isinstance(a, list) else []


def _causal_units(rep):
    a = rep.get("attribution")
    explanations = [x.get("explanation") for x in _attributions(rep)] if _attributions(rep) else [a]
    evidence = [e.get(k) for e in rep.get("evidence") or [] if isinstance(e, dict) for k in ("claim", "source")]
    return _units(rep.get("finding"), rep.get("how_it_happened"), rep.get("engineer_summary"), *explanations, *evidence)


def _variants(text):
    out = {text, text.lstrip("$+"), text.lstrip("+")}
    if text.endswith("%"):
        out |= {text.lstrip("+").replace(".0%", "%"), text.lstrip("+-").replace(".0%", "%")}
    return {v for v in out if v and v not in ("0%", "%")}


def _numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def _code(run):
    return " ".join(str((c.arguments or {}).get("code", "")) for _, c in run.calls("run_python"))


def _code_numbers(run):
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", _code(run))]


def _spend_values(result):
    if "columns" in result:
        cols = [i for i, c in enumerate(result["columns"]) if re.search(r"spend|usd|cost|total", str(c), re.I)]
        return [row[i] for row in result.get("rows", []) for i in cols if isinstance(row[i], (int, float))]
    return [float(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})+\.\d{2}|\d{4,}\.\d{2}", result.get("stdout", ""))]


def _drivers(rep):
    """(category, before, after) for each report driver whose metric we recognize."""
    out = []
    for d in rep.get("drivers") or []:
        if not isinstance(d, dict) or not isinstance(d.get("before"), (int, float)) or not isinstance(d.get("after"), (int, float)):
            continue
        m = str(d.get("metric", "")).lower()
        if NOT_OBSERVED.search(m):
            continue
        if re.search(r"request", m):
            cat = "requests_per_ticket"
        elif re.search(r"token", m) and re.search(r"per ticket|/ ?ticket|per-ticket", m):
            cat = "tokens_per_ticket"
        elif re.search(r"price|per 1m|per million|per token|rate card", m):
            cat = "price"
        elif re.search(r"cost per|cost/|per ticket", m):
            cat = "cost_per_ticket"
        elif re.search(r"mix|share", m):
            cat = "mix"
        elif re.search(r"token", m):
            cat = "tokens"
        elif re.search(r"ticket|volume", m):
            cat = "tickets"
        elif re.search(r"spend|cost|bill", m):
            cat = "spend"
        else:
            cat = "other"
        out.append((cat, d["before"], d["after"]))
    return out


def _moved(drivers, cat, up=True, threshold=0.05):
    return any(c == cat and b and ((a - b) / abs(b) > threshold if up else (b - a) / abs(b) > threshold) for c, b, a in drivers)


def savings_range(scenario, routes):
    """Accepted savings for a given routing, by the same bases the oracle uses."""
    t = truth(scenario)
    s = world.spec(scenario)
    conn = world.connect(scenario)
    types = t["ticket_types"]
    if any(routes.get(x) not in s["models"] for x in types):
        return None
    proposal = {x: unit_cost(scenario, routes[x], x) for x in types}
    projected = {x: unit_cost(scenario, t["production"][x], x) for x in types}
    last = s["last_day"]
    week = ((dt.date.fromisoformat(last) - dt.timedelta(days=6)).isoformat(), last)
    after = [(t["change_start"], last), week]
    values = []
    for a, b in {tuple(s["compare"]), tuple(s["period"]), *after}:
        daily = {x: world.stats(conn, a, b, x)["tickets"] / _days(a, b) for x in types}
        for current in [projected] + [{x: _metrics(conn, wa, wb, x)["cost_per_ticket"] for x in types} for wa, wb in after]:
            for month in (30, 31):
                values.append(month * sum(daily[x] * (current[x] - proposal[x]) for x in types))
    return 0.85 * min(values), 1.15 * max(values)


def interventions(result):
    return (result.get("nudges", 0) + result.get("truncations", 0) + result.get("rejected_reports", 0)
            + result.get("why_missing", 0) + result.get("image_budget_rejections", 0) + result.get("rejected_proposals", 0))


def score(run):
    """{segment: True/False/None, ..., "e2e": bool, "unassisted": bool, details}."""
    t = truth(run.scenario)
    rep, steps = run.report, run.steps
    out, why = {}, {}

    # S1 first look
    if t["screenshot"]:
        note = steps[0].note if steps else ""
        values = {v for _, value, change, before in kpi_cards(run.scenario) for v in (value, change, before) if v}
        hits = {v for v in values if any(x in note.replace("−", "-") for x in _variants(v))}
        out["S1"] = len(hits) >= 2
        why["S1"] = sorted(hits)
    else:
        out["S1"] = None

    # S2 reconcile
    (before, after), (wrong, _) = t["spend"], t["misbucketed"]
    seen = [v for _, c in run.calls("run_sql", "run_python") for v in _spend_values(c.result or {})]
    got_after = any(abs(v - after) <= 0.005 * after for v in seen)
    got_before = any(abs(v - before) <= 0.005 * before for v in seen)
    stated = _numbers(str(rep.get("finding", ""))) + [x.get(k) for f in ("key_figures", "drivers") for x in rep.get(f) or []
                                                      if isinstance(x, dict) for k in ("value", "before", "after")
                                                      if isinstance(x.get(k), (int, float))]
    misbucketed = any(abs(v - wrong) <= 0.005 * wrong for v in stated)
    out["S2"] = got_after and got_before and not misbucketed
    why["S2"] = {"after_seen": got_after, "before_seen": got_before, "misbucketed_reported": misbucketed}

    # S3 decompose
    drivers = _drivers(rep)
    text = _report_text(rep, ("finding", "how_it_happened"))
    fam = t["family"]
    tickets_ratio = t["tickets"][1] / t["tickets"][0]
    tokens_up = _moved(drivers, "tokens_per_ticket") or any(c == "tokens" and b and a / b > 1.05 * tickets_ratio for c, b, a in drivers)
    checks = {
        "model_switch": bool(tokens_up or re.search(r"(more|higher|longer|increase\w*|rose|jump\w*|grew|\d(\.\d)?\s*[x×])[^.]{0,40}(token|reasoning|text|output|generat|answer|response)|(token|reasoning|text|output)[^.]{0,40}(more|higher|increase|rose|jump|grew|doubl|tripl)", text, re.I))
                        and bool(t["new_model"] and t["new_model"].lower() in text.lower()),
        "price": _moved(drivers, "price") or bool(PRICE_CLAIM.search(text)),
        "volume": _moved(drivers, "tickets", threshold=0.08) or bool(VOLUME_CLAIM.search(text)),
        "mix": bool(MIX_CLAIM.search(text)) or any(re.search(rf"{re.escape(x)}[^.]{{0,60}}(surge|jump|rose|grew|increase|share|fell|drop|declin|fewer|more)|"
                                                            rf"(surge|jump|rise|growth|increase|fall|drop|decline)[^.]{{0,40}}{re.escape(x)}", text, re.I)
                                                  for x in t["mix_types"]),
        "volume_price": (_moved(drivers, "tickets", threshold=0.08) or bool(VOLUME_CLAIM.search(text))) and (_moved(drivers, "price") or bool(PRICE_CLAIM.search(text))),
        "retry_bug": _moved(drivers, "requests_per_ticket") or bool(USAGE_CLAIM.search(text)),
        "none": not any(_moved(drivers, c, threshold=0.10) or _moved(drivers, c, up=False, threshold=0.10)
                        for c in ("tokens_per_ticket", "price", "requests_per_ticket", "cost_per_ticket"))
                and not (_claims(SWITCH_CLAIM, text) or _claims(PRICE_CLAIM, text) or _claims(USAGE_CLAIM, text)
                         or _claims(MIX_CLAIM, text)),
    }
    out["S3"] = bool(rep) and bool(checks[fam])

    # S4 attribute: cite every cause (the causal change and the price schedule, where each exists), blame no decoy,
    # claim no false mechanism. An id counts when the attribution field lists it or a sentence names it without clearing it.
    causal = _causal_text(rep)
    finding = _report_text(rep, ("finding", "how_it_happened"))
    listed = set()
    for entry in _attributions(rep):
        ids = {str(x) for f in ("change_ids", "documents") if isinstance(entry.get(f), list) for x in entry[f]}
        listed |= ids - {i for u in _units(entry.get("explanation")) if CLEARED.search(HEDGE.sub("", u)) for i in ids if i in u}
    put_forward = lambda ids, units: sorted({i for i in ids if i in listed} | _named(ids, units))
    dates = sorted({t["changes"][c][1] for c in t["causal_changes"]})
    units = _causal_units(rep)
    change_cited = put_forward(t["causal_changes"] + dates, units)
    if fam == "model_switch" and t["new_model"] and t["new_model"].lower() in causal.lower():
        change_cited.append("new model named")
    document_cited = put_forward(t["causal_documents"], units)
    if t["causal_documents"] and re.search(r"(new|september|updated|latest|current) (price )?schedule|schedule effective|price schedule", causal, re.I):
        document_cited.append("schedule")
    decoys = put_forward(t["decoys"], _units(rep.get("finding"), rep.get("how_it_happened")))
    false = []
    if fam not in ("model_switch",) and _claims(SWITCH_CLAIM, finding):
        false.append("model switch")
    if fam not in ("price", "volume_price") and _claims(PRICE_CLAIM, finding):
        false.append("price change")
    causes = (not t["causal_changes"] or bool(change_cited)) and (not t["causal_documents"] or bool(document_cited))
    out["S4"] = bool(rep) and causes and not decoys and not false
    why["S4"] = {"cited": change_cited + document_cited, "change_cited": bool(change_cited) if t["causal_changes"] else None,
                 "document_cited": bool(document_cited) if t["causal_documents"] else None,
                 "decoys_blamed": decoys, "false_mechanism": false}

    # S5 retrieve
    views = [((c.arguments or {}).get("doc_id"), int((c.arguments or {}).get("page") or 1)) for _, c in run.calls("view_document") if c.arguments]
    need = {(t["current_schedule"], 1)}
    if t["eval_document"]:
        need |= {(t["eval_document"], 1), (t["eval_document"], 2)}
    opened = set(views)
    reopened = sum(views.count(v) - 1 for v in opened)
    out["S5"] = need <= opened
    why["S5"] = {"missing": sorted(need - opened), "reopened": reopened}

    # S6 transcribe: prices exist only on the scanned schedule, so a filed proposal's cost projection needs them read exactly
    nums = _code_numbers(run)
    has = lambda v: any(abs(n - v) < 1e-9 for n in nums)
    prices = world.current_prices(run.scenario)
    printed = {round(v, 4) for p in prices.values() for v in (p["input"], p["output"])}
    misread = sorted({n for n in nums if n not in printed and any(0 < abs(n - v) <= 0.1 * v for v in printed)
                      and len(str(n).split(".")[-1]) <= 3 and n < 100})
    filed = run.proposals[0]["routes"] if run.proposals else None
    if filed:
        models = {m for m in filed.values() if m in prices} | set(t["production"].values())
        missing = [v for m in sorted(models) for v in (prices[m]["input"], prices[m]["output"]) if not has(v)]
        out["S6"] = not missing
    else:
        missing = []
        out["S6"] = None
    eval_missing = [v for v in t["needed_eval"].values() if not has(v)]
    why["S6"] = {"missing": missing, "possible_misreads": misread, "eval_values_not_in_code": eval_missing}

    # S7 compute and S8 decide
    accepted = run.proposals
    proposal = accepted[0] if accepted else None
    if proposal:
        rng = savings_range(run.scenario, proposal["routes"])
        value = proposal.get("expected_monthly_savings_usd")
        out["S7"] = bool(rng) and isinstance(value, (int, float)) and rng[0] <= value <= rng[1] if rng and rng[1] > 0 else False
        why["S7"] = {"stated": value, "range": rng}
    elif t["expected_proposal"] is None:
        cpt = [(b, a) for c, b, a in drivers if c == "cost_per_ticket"]
        tb, ta = t["before"]["cost_per_ticket"], t["after"]["cost_per_ticket"]
        out["S7"] = any(abs(b - tb) <= 0.01 * tb and abs(a - ta) <= 0.01 * ta for b, a in cpt) or None
        why["S7"] = {"cost_per_ticket_drivers": cpt}
    else:
        out["S7"] = False
        why["S7"] = "no proposal filed"
    routes = proposal["routes"] if proposal else None
    compliant = bool(routes) and all(world.pass_pct(run.scenario, routes[x], x) >= t["floors"][x] - 1e-9 for x in t["ticket_types"] if routes.get(x))
    saving = bool(routes) and sum(unit_cost(run.scenario, routes[x], x) for x in t["ticket_types"] if routes.get(x)) < \
        sum(unit_cost(run.scenario, t["production"][x], x) for x in t["ticket_types"]) - 1e-12
    if t["expected_proposal"]:
        out["S8"] = len(accepted) == 1 and routes == t["optimal"]
    elif t["proposal_optional"]:
        out["S8"] = not accepted or (compliant and saving)
    else:
        out["S8"] = not accepted
    why["S8"] = {"filed": routes, "optimal": t["optimal"], "expected": bool(t["expected_proposal"]),
                 "valid_saving": compliant and saving}

    # S9 report
    pct = lambda pair: 100 * (pair[1] / pair[0] - 1)
    nums_rep = _numbers(json.dumps(rep)) + [x.get("value") for x in rep.get("key_figures") or [] if isinstance(x, dict) and isinstance(x.get("value"), (int, float))]
    near = lambda target, tol, rel=False: any(abs(v - target) <= (abs(target) * tol if rel else tol) for v in nums_rep)
    spend_ok = near(pct(t["spend"]), 1.0) or (near(t["spend"][0], 0.01, True) and near(t["spend"][1], 0.01, True))
    tickets_ok = near(pct(t["tickets"]), 1.0) or (near(t["tickets"][0], 0.01, True) and near(t["tickets"][1], 0.01, True))
    labels = {e.get("label") for e in rep.get("evidence") or [] if isinstance(e, dict)}
    evidence = rep.get("evidence") or []
    label_of = lambda e: str(e.get("label", "")) if isinstance(e, dict) else str(e)
    asked_for_labels = not run.transcript or "Label each claim Observed" in str(run.transcript[0].get("content", ""))
    labels_on = asked_for_labels and any(isinstance(e, dict) and "label" in e for e in evidence)
    labelled = bool(evidence) and (all(re.search(r"Observed|Calculated|Hypothesis", label_of(e)) for e in evidence) if labels_on else True)
    wrongly_ruled = False
    for r in rep.get("ruled_out") or []:
        candidate = str(r.get("candidate", "")) if isinstance(r, dict) else ""
        if NEGATION.search(candidate) or re.search(r"non-production|other models|not in production|other than|except|besides|apart from", candidate, re.I):
            continue
        names = lambda model: re.search(rf"{re.escape(model)}(?!\s*(Pro|Lite|Thinking|-))", candidate, re.I)
        if fam in ("price", "volume_price") and PRICE_CLAIM.search(candidate) and names(t["baseline_model"]):
            wrongly_ruled = True
        if fam == "model_switch" and t["new_model"] and names(t["new_model"]) and SWITCH_CLAIM.search(candidate):
            wrongly_ruled = True
        if fam == "mix" and MIX_CLAIM.search(candidate):
            wrongly_ruled = True
        if fam == "retry_bug" and USAGE_CLAIM.search(candidate):
            wrongly_ruled = True
    out["S9"] = bool(rep) and spend_ok and tickets_ok and labelled and not wrongly_ruled
    why["S9"] = {"spend": spend_ok, "tickets": tickets_ok, "labelled": labelled, "true_cause_ruled_out": wrongly_ruled}

    # S10 finish
    out["S10"] = bool(rep) and not run.result.get("error")

    need7 = out["S7"] if (t["expected_proposal"] or proposal) else True
    out["e2e"] = bool(out["S3"] and out["S4"] and out["S8"] and out["S9"] and out["S10"] and need7)
    out["valid_proposal"] = bool(proposal and compliant and saving) or (not t["expected_proposal"] and not proposal)
    out["interventions"] = interventions(run.result)
    out["unassisted"] = out["e2e"] and out["interventions"] == 0
    out["why"] = why
    return out
