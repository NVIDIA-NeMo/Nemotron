# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The oracle: what really happened in a world, and the best decision, computed from its spec and data.

Works for the hand-built Fernledger months and for generated worlds. The agent never sees any of this.
"""

import datetime as dt
from functools import lru_cache

import world
from investigator.policy import policy_for
from world.dashboard import _k, _pct

LEGACY_TRUTH = {
    "model-switch": {"family": "model_switch", "causal_changes": ["CHG-5120", "CHG-5124", "CHG-5129"], "causal_documents": [],
                     "decoys": ["CHG-5031", "CHG-5107"]},
    "volume-growth": {"family": "volume", "causal_changes": ["CHG-5098"], "causal_documents": [],
                      "decoys": ["CHG-5031", "CHG-5107", "CHG-5116"]},
    "price-increase": {"family": "price", "causal_changes": [], "causal_documents": ["price-schedule-2026-09"],
                       "decoys": ["CHG-5031", "CHG-5107", "CHG-5116"]},
}
LEGACY_TRUTH["model-switch-visual"] = LEGACY_TRUTH["model-switch"]


MATERIAL = 0.03  # a routing change saving less than 3% of monthly spend is optional, not expected


def _date(s):
    return dt.date.fromisoformat(s)


def _days(a, b):
    return (_date(b) - _date(a)).days + 1


def _metrics(conn, start, end, ticket_type=None):
    st = world.stats(conn, start, end, ticket_type)
    where, args = "date BETWEEN ? AND ?", [start, end]
    if ticket_type:
        where += " AND ticket_type = ?"
        args.append(ticket_type)
    requests = conn.execute(f"SELECT SUM(requests) FROM usage_daily WHERE {where}", args).fetchone()[0] or 0
    tokens = st["input"] + st["answer"] + st["reasoning"]
    t = st["tickets"] or 1
    return {"spend": st["spend"], "tickets": st["tickets"], "cost_per_ticket": st["spend"] / t,
            "tokens_per_ticket": tokens / t, "input_per_ticket": st["input"] / t, "reasoning_per_ticket": st["reasoning"] / t,
            "requests_per_ticket": requests / t, "price_per_1m": st["spend"] * 1e6 / tokens if tokens else 0.0,
            "resolution_pct": 100 * st["resolved"] / t}


def unit_cost(scenario, model, ticket_type, prices=None):
    """Projected cost per ticket from the eval report's tokens and the current price schedule."""
    t = world.model_tokens(model, ticket_type, scenario)
    p = (prices or world.current_prices(scenario))[model]
    return (t["input"] * p["input"] + (t["answer"] + t["reasoning"]) * p["output"]) / 1e6


def _change_start(s):
    """When the cause finished taking effect (or the period start if nothing changed)."""
    dates = []
    for e in s["events"]:
        if e["type"] == "model_switch":
            dates.append(e["complete"])
        elif e["type"] in ("volume", "mix_shift", "retry_bug") and e["start"] > s["period"][0]:
            dates.append(e["start"])
    for schedule in s["price_schedules"][1:]:
        if schedule["effective"] > s["period"][0]:
            dates.append(schedule["effective"])
    return max(dates) if dates else s["period"][0]


@lru_cache(maxsize=None)
def truth(scenario):
    s = world.spec(scenario)
    world.ensure(scenario)
    conn = world.connect(scenario)
    meta = s.get("truth") or LEGACY_TRUTH[scenario]
    before, after = _metrics(conn, *s["compare"]), _metrics(conn, *s["period"])
    change = {k: 100 * (after[k] / before[k] - 1) if before[k] else 0.0 for k in before}
    wrong_start = (_date(s["compare"][1]) + dt.timedelta(days=1)).isoformat()
    wrong = world.stats(conn, wrong_start, s["period"][1])
    types, catalog = s["ticket_types"], list(s["models"])
    last = s["last_day"]
    production = {}
    for t in types:
        row = conn.execute("SELECT model FROM usage_daily WHERE date = ? AND ticket_type = ? ORDER BY tickets DESC LIMIT 1",
                           (last, t)).fetchone()
        production[t] = row[0]
    q = policy_for(scenario)["quality_bar"]
    floors = {t: world.pass_pct(scenario, q["baseline_model"], t) - q["max_drop_pct_points"] for t in types}
    prices = world.current_prices(scenario)
    cost = {(m, t): unit_cost(scenario, m, t, prices) for m in catalog for t in types}
    optimal = {}
    for t in types:
        ok = [m for m in catalog if world.pass_pct(scenario, m, t) >= floors[t] - 1e-9]
        optimal[t] = min(ok, key=lambda m: (round(cost[(m, t)], 9), -world.pass_pct(scenario, m, t)))
    expected = None if optimal == production else optimal

    start = _change_start(s)
    windows = {tuple(s["compare"]), tuple(s["period"]), (start, last),
               ((_date(last) - dt.timedelta(days=6)).isoformat(), last)}
    after_windows = [(start, last), ((_date(last) - dt.timedelta(days=6)).isoformat(), last)]
    savings = []
    if expected:
        proposal_cost = {t: cost[(optimal[t], t)] for t in types}
        projected_now = {t: cost[(production[t], t)] for t in types}
        for a, b in windows:
            daily = {t: world.stats(conn, a, b, t)["tickets"] / _days(a, b) for t in types}
            currents = [projected_now] + [{t: (lambda m: m["cost_per_ticket"])(_metrics(conn, wa, wb, t)) for t in types}
                                          for wa, wb in after_windows]
            for current in currents:
                for month in (30, 31):
                    savings.append(month * sum(daily[t] * (current[t] - proposal_cost[t]) for t in types))
    monthly_spend = 30 * after["spend"] / _days(*s["period"])
    optional = bool(expected) and savings and max(savings) < MATERIAL * monthly_spend
    eval_doc = None if s["eval"]["in_db"] else s["eval"]["id"]
    current_schedule = max((x for x in s["price_schedules"]), key=lambda x: x["effective"])
    needed_models = sorted(set(optimal.values()) | set(production.values()))
    return {
        "scenario": scenario, "family": meta["family"], "causal_changes": meta["causal_changes"],
        "causal_documents": meta["causal_documents"], "decoys": meta["decoys"],
        "changes": {c[0]: c for c in s["changes"]},
        "before": before, "after": after, "change_pct": change,
        "spend": (before["spend"], after["spend"]), "tickets": (before["tickets"], after["tickets"]),
        "misbucketed": (wrong["spend"], wrong["tickets"]), "change_start": start,
        "production": production, "optimal": optimal, "expected_proposal": None if optional else expected,
        "proposal_optional": optional, "floors": floors, "monthly_spend": monthly_spend,
        "unit_cost": {f"{m}|{t}": c for (m, t), c in cost.items()},
        "savings_range": (0.85 * min(savings), 1.15 * max(savings)) if savings else None,
        "current_schedule": current_schedule["id"], "eval_document": eval_doc,
        "tokens_page": s["eval"].get("tokens_page", 2) if eval_doc else None,
        "needed_prices": {m: prices[m] for m in needed_models},
        "needed_eval": {(m, t): world.pass_pct(scenario, m, t) for t in types for m in {optimal[t], q["baseline_model"]}} if eval_doc else {},
        "baseline_model": q["baseline_model"], "screenshot": s.get("attach_overview", False),
        "models": catalog, "ticket_types": types,
        "new_model": next((e["to"] for e in s["events"] if e["type"] == "model_switch"), None),
        "mix_types": [x for e in s["events"] if e["type"] == "mix_shift" for x in e["share"]
                      if abs(e["share"][x] - s["share"][x]) > 0.02],
        "provider": s["provider"],
    }


def kpi_cards(scenario):
    """The Overview tab's KPI cards for the default periods, formatted as the dashboard shows them."""
    s = world.spec(scenario)
    world.ensure(scenario)
    conn = world.connect(scenario)
    p, c = world.stats(conn, *s["period"]), world.stats(conn, *s["compare"])
    cp, cc = p["spend"] / p["tickets"], c["spend"] / c["tickets"]
    tp, tc = (p["input"] + p["answer"] + p["reasoning"]) / 1e6, (c["input"] + c["answer"] + c["reasoning"]) / 1e6
    return [
        ("AI spend", _k(p["spend"]), _pct(p["spend"], c["spend"]), _k(c["spend"])),
        ("Tickets handled by AI", f"{p['tickets'] / 1e6:.2f}M", _pct(p["tickets"], c["tickets"]), f"{c['tickets'] / 1e6:.2f}M"),
        ("AI cost per ticket", f"${cp:.4f}", _pct(cp, cc), f"${cc:.4f}"),
        ("Resolution rate", f"{100 * p['resolved'] / p['tickets']:.1f}%", None, f"{100 * c['resolved'] / c['tickets']:.1f}%"),
        ("Average price per 1M tokens", f"${p['spend'] / tp:.2f}", _pct(p["spend"] / tp, c["spend"] / tc), f"${c['spend'] / tc:.2f}"),
    ]
