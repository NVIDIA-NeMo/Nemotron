# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground truth, and checks of a run against it.

exact_checks() scores a run: the filed proposal, its savings, and the key numbers in the report. The other
functions compare what a run read or computed with the truth; the notebook's step cards show their results.
"""

import datetime as dt
import json
import math
import re

import world
from world.dashboard import _k, _pct

BEST = {"Account & password": "Vesk 4 Lite", "Billing": "Vesk 4 Lite",
        "Technical troubleshooting": "Vesk 4", "Integrations & API": "Vesk 4"}
CURRENT = {"model-switch": "Osric 2 Thinking", "volume-growth": "Vesk 4"}
BASES = {"model-switch": [("2026-08-01", "2026-08-21"), ("2026-09-01", "2026-09-21"), ("2026-09-10", "2026-09-21"),
                          ("2026-09-15", "2026-09-21")],
         "volume-growth": [("2026-08-01", "2026-08-21"), ("2026-09-01", "2026-09-21"), ("2026-09-15", "2026-09-21")]}
CURRENT["model-switch-visual"], BASES["model-switch-visual"] = CURRENT["model-switch"], BASES["model-switch"]
CURRENT["price-increase"], BASES["price-increase"] = "Vesk 4", BASES["volume-growth"]
CURRENT["model-switch-visual-p1"], BASES["model-switch-visual-p1"] = CURRENT["model-switch"], BASES["model-switch"]
_SWITCH = ("The default model moved from Vesk 4 to Osric 2 Thinking on Sep 8-10. Osric is cheaper per token but "
           "generates far more reasoning per ticket, so each ticket costs more; it also resolves fewer technical and "
           "integration tickets.")
TRUTH = {  # what really happened in September, in plain words
    "model-switch": _SWITCH, "model-switch-visual": _SWITCH,
    "volume-growth": "In-app help chat launched on Aug 31 and tickets rose about 30%; cost per ticket stayed flat.",
    "price-increase": "Larkfield Cloud raised Vesk 4's prices about 30% from Sep 1 (the September price schedule); "
                      "tickets and tokens per ticket stayed flat.",
}
EXACT_LABELS = {"report_submitted": "Submitted a report", "M4_proposal": "Filed the best routing",
                "M5_savings": "Savings in the expected range", "M6_spend": "Spend change matches the data",
                "M6_tickets": "Ticket change matches the data"}


def _date(s):
    return dt.date.fromisoformat(s)


def unit_cost(model, ticket_type, scenario="model-switch"):
    t, p = world.model_tokens(model, ticket_type), world.current_prices(scenario)[model]
    return (t["input"] * p["input"] + (t["answer"] + t["reasoning"]) * p["output"]) / 1e6


def truth(scenario):
    world.ensure(scenario)
    conn = world.connect(scenario)
    a, b = world.stats(conn, *world.DEFAULT_COMPARE), world.stats(conn, *world.DEFAULT_PERIOD)
    savings = []
    for start, end in BASES[scenario]:
        days = (_date(end) - _date(start)).days + 1
        daily = {t: world.stats(conn, start, end, t)["tickets"] / days for t in world.TICKET_TYPES}
        window = (("2026-09-10", "2026-09-21") if scenario.startswith("model-switch")
                  else ("2026-09-01", "2026-09-21") if scenario == "price-increase" else (start, end))
        measured = {t: (lambda s: s["spend"] / s["tickets"])(world.stats(conn, *window, t)) for t in world.TICKET_TYPES}
        projected = {t: unit_cost(CURRENT[scenario], t, scenario) for t in world.TICKET_TYPES}
        best = sum(daily[t] * unit_cost(BEST[t], t, scenario) for t in world.TICKET_TYPES)
        for current in (measured, projected):
            for month in (30, 31):
                savings.append(month * (sum(daily[t] * current[t] for t in world.TICKET_TYPES) - best))
    wrong = world.stats(conn, "2026-08-22", "2026-09-21")  # "Sep 1-21" when Aug 22-31 falls into an ELSE bucket
    return {"spend": (a["spend"], b["spend"]), "spend_pct": 100 * (b["spend"] / a["spend"] - 1),
            "tickets": (a["tickets"], b["tickets"]), "misbucketed": (wrong["spend"], wrong["tickets"]),
            "tickets_pct": 100 * (b["tickets"] / a["tickets"] - 1),
            "savings_range": (0.85 * min(savings), 1.15 * max(savings)), "base_savings": (min(savings), max(savings))}


def numbers(report):
    text = json.dumps(report)
    found = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]
    for fig in report.get("key_figures") or []:
        if isinstance(fig.get("value"), (int, float)):
            found.append(float(fig["value"]))
    return found


def near(values, target, tol, relative=False):
    return any(abs(v - target) <= (abs(target) * tol if relative else tol) for v in values)


def exact_checks(run):
    """{check: passed} for the proposal, its savings, and the key numbers in the report."""
    k, rep, accepted = truth(run.scenario), run.report, run.proposals
    nums = numbers(rep)
    lo, hi = k["savings_range"]
    savings = accepted[0].get("expected_monthly_savings_usd") if accepted else None
    return {
        "report_submitted": bool(rep),
        "M4_proposal": len(accepted) == 1 and accepted[0]["routes"] == BEST and rep.get("proposal_id") == accepted[0]["proposal_id"],
        "M5_savings": isinstance(savings, (int, float)) and lo <= savings <= hi,
        "M6_spend": near(nums, k["spend_pct"], 1.0) or (near(nums, k["spend"][0], 0.01, True) and near(nums, k["spend"][1], 0.01, True)),
        "M6_tickets": near(nums, k["tickets_pct"], 1.0),
    }


def token_breakdown(scenario):
    """Per-ticket input, answer, reasoning, and total tokens for the two displayed periods."""
    world.ensure(scenario)
    conn = world.connect(scenario)
    rows = []
    for label, period in (("Aug 1–21", world.DEFAULT_COMPARE), ("Sep 1–21", world.DEFAULT_PERIOD)):
        stats = world.stats(conn, *period)
        rows.append({
            "period": label,
            "input": stats["input"] / stats["tickets"],
            "answer": stats["answer"] / stats["tickets"],
            "reasoning": stats["reasoning"] / stats["tickets"],
            "total": (stats["input"] + stats["answer"] + stats["reasoning"]) / stats["tickets"],
        })
    return rows


def exact_bridge(scenario):
    """Exact additive LMDI split of spend = tickets × tokens/ticket × effective price/token."""
    world.ensure(scenario)
    conn = world.connect(scenario)
    before = world.stats(conn, *world.DEFAULT_COMPARE)
    after = world.stats(conn, *world.DEFAULT_PERIOD)

    def factors(stats):
        tokens = stats["input"] + stats["answer"] + stats["reasoning"]
        return stats["tickets"], tokens / stats["tickets"], stats["spend"] * 1e6 / tokens

    t0, u0, p0 = factors(before)
    t1, u1, p1 = factors(after)
    s0, s1 = before["spend"], after["spend"]
    log_mean = (s1 - s0) / math.log(s1 / s0) if s1 != s0 else s0
    parts = [
        ("volume", log_mean * math.log(t1 / t0)),
        ("usage", log_mean * math.log(u1 / u0)),
        ("price", log_mean * math.log(p1 / p0)),
    ]
    return {
        "before": s0, "after": s1, "parts": parts,
        "factors": {
            "tickets": (t0, t1),
            "tokens_per_ticket": (u0, u1),
            "effective_price_per_million": (p0, p1),
        },
    }


def proposal_projection(run):
    """Reproduce the mixed-route projection from source data and the recorded proposal."""
    world.ensure(run.scenario)
    conn = world.connect(run.scenario)
    recent = world.stats(conn, "2026-09-10", "2026-09-21")
    period = world.stats(conn, *world.DEFAULT_PERIOD)
    daily_tickets = recent["tickets"] / 12
    current_cost = recent["spend"] / recent["tickets"]
    mix = {ticket_type: world.stats(conn, *world.DEFAULT_PERIOD, ticket_type)["tickets"] / period["tickets"]
           for ticket_type in world.TICKET_TYPES}
    routes = run.proposals[0]["routes"]
    proposed_cost = sum(mix[ticket_type] * unit_cost(routes[ticket_type], ticket_type, run.scenario)
                        for ticket_type in world.TICKET_TYPES)
    current_month = 30 * daily_tickets * current_cost
    proposed_month = 30 * daily_tickets * proposed_cost

    def weighted_quality(model_by_type):
        return sum(mix[ticket_type] * world.PASS_PCT[model_by_type[ticket_type]][i]
                   for i, ticket_type in enumerate(world.TICKET_TYPES))

    all_vesk = {ticket_type: "Vesk 4" for ticket_type in world.TICKET_TYPES}
    all_osric = {ticket_type: "Osric 2 Thinking" for ticket_type in world.TICKET_TYPES}
    return {
        "daily_tickets": daily_tickets,
        "current_cost_per_ticket": current_cost,
        "proposed_cost_per_ticket": proposed_cost,
        "current_monthly_spend": current_month,
        "proposed_monthly_spend": proposed_month,
        "monthly_savings": current_month - proposed_month,
        "savings_pct": 100 * (current_month - proposed_month) / current_month,
        "ticket_mix": mix,
        "quality": {
            "vesk4_baseline": weighted_quality(all_vesk),
            "osric_current": weighted_quality(all_osric),
            "proposal": weighted_quality(routes),
        },
    }


# Comparisons for the step cards --------------------------------------------------------------------------

def kpi_cards(scenario):
    """The Overview tab's five KPI cards for the default periods, formatted as the dashboard shows them."""
    world.ensure(scenario)
    conn = world.connect(scenario)
    p, c = world.stats(conn, *world.DEFAULT_PERIOD), world.stats(conn, *world.DEFAULT_COMPARE)
    cp, cc = p["spend"] / p["tickets"], c["spend"] / c["tickets"]
    tp, tc = (p["input"] + p["answer"] + p["reasoning"]) / 1e6, (c["input"] + c["answer"] + c["reasoning"]) / 1e6
    return [
        ("AI spend", _k(p["spend"]), _pct(p["spend"], c["spend"]), _k(c["spend"])),
        ("Tickets handled by AI", f"{p['tickets'] / 1e6:.2f}M", _pct(p["tickets"], c["tickets"]), f"{c['tickets'] / 1e6:.2f}M"),
        ("AI cost per ticket", f"${cp:.4f}", _pct(cp, cc), f"${cc:.4f}"),
        ("Resolution rate", f"{100 * p['resolved'] / p['tickets']:.1f}%", None, f"{100 * c['resolved'] / c['tickets']:.1f}%"),
        ("Average price per 1M tokens", f"${p['spend'] / tp:.2f}", _pct(p["spend"] / tp, c["spend"] / tc), f"${c['spend'] / tc:.2f}"),
    ]


def _variants(text):
    """A dashboard value and the ways a note might write it: $122.7k, 122.7k, $0.0542, 0.0542, +30.0%, 30.0%, 30%."""
    out = {text, text.lstrip("$+"), text.lstrip("+")}
    if text.endswith("%"):
        out |= {text.lstrip("+").replace(".0%", "%"), text.lstrip("+-").replace(".0%", "%")}
    return {v for v in out if v and v not in ("0%", "%")}


def screenshot_facts(note, scenario):
    """Which KPI-card values from the screenshot the agent's note repeats: [(card, value shown, found in note)]."""
    text = note.replace("−", "-")
    rows = []
    for label, value, change, before in kpi_cards(scenario):
        for shown in [value, change, before]:
            if shown:
                rows.append((label, shown, any(v in text for v in _variants(shown))))
    return rows


SPEND_COLUMN = re.compile(r"spend|usd|cost|total", re.I)


def _spend_values(result):
    """Dollar amounts in a result: spend-like SQL columns, or amounts with cents in printed output."""
    if "columns" in result:
        spend = [i for i, c in enumerate(result["columns"]) if SPEND_COLUMN.search(str(c))]
        return [row[i] for row in result.get("rows", []) for i in spend if isinstance(row[i], (int, float))]
    return [float(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})+\.\d{2}|\d{4,}\.\d{2}", result.get("stdout", ""))]


def spend_checks(run):
    """Every query or code result that states a period's total spend, compared with the dashboard's KPI cards."""
    k = truth(run.scenario)
    (aug, sep), (wrong, _) = k["spend"], k["misbucketed"]
    events = []
    for step, call in run.calls("run_sql", "run_python"):
        for value in _spend_values(call.result):
            for kind, target, label in (("match", sep, "September 1-21 spend"), ("match", aug, "August 1-21 spend"),
                                        ("mismatch", wrong, "a September total that also counts Aug 22-31")):
                if abs(value - target) <= 0.005 * target:
                    events.append({"step": step, "call": call, "kind": kind, "value": value, "target": target, "label": label})
                    break
    return events


EVAL_NUMBERS = [  # the report numbers the routing recommendation depends on, with where they are printed
    ("Vesk 4 Lite resolution, Account & password", (92.1,), "page 1"),
    ("Vesk 4 Lite resolution, Billing", (88.2,), "page 1"),
    ("Vesk 4 Lite answer tokens, Account & password", (570,), "page 2"),
    ("Vesk 4 Lite answer tokens, Billing", (741,), "page 2"),
    ("Vesk 4 resolution (or its floor), Technical troubleshooting", (76, 74), "page 1"),
    ("Vesk 4 resolution (or its floor), Integrations & API", (71, 69), "page 1"),
]


def _written(run):
    code = " ".join(c.arguments.get("code", "") for _, c in run.calls("run_python") if c.arguments)
    return code, [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", code)], json.dumps(run.report)


def eval_numbers_used(run):
    """[(what, value in the report, where it was used)] for the eval-report numbers the proposal depends on."""
    code, numbers_in_code, report = _written(run)
    rows = []
    for label, values, page in EVAL_NUMBERS:
        where = ("code" if any(abs(n - v) < 1e-9 for n in numbers_in_code for v in values) else
                 "report" if any(re.search(rf"(?<![\d.]){v:g}(?![\d])", report) for v in values) else None)
        rows.append((label, values[0], page, where))
    return rows


def prices_used(run):
    """[(model, schedule input, schedule output, both typed into the agent's code)] for the models it priced."""
    code, numbers_in_code, _ = _written(run)
    rows = []
    for model, price in world.current_prices(run.scenario).items():
        found = [any(abs(n - v) < 1e-9 for n in numbers_in_code) for v in (price["input"], price["output"])]
        if model in code or any(found):
            rows.append((model, price["input"], price["output"], all(found)))
    return rows
