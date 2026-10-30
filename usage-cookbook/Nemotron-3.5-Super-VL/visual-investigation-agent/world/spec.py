# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""World specs: everything that defines one month's world, as plain data.

The four hand-built months are specs built from the Fernledger constants in data.py, so they rebuild exactly
as before. Generated worlds (world/generator.py) are specs with other companies, models, ticket types, dates,
documents, and causes. Every spec is saved next to its data as spec.json, so any process can reload it.
"""

import json
from pathlib import Path

WORLD = Path(__file__).resolve().parents[1] / ".world"  # generated files; safe to delete
_REGISTRY = {}
VARIANTS = {  # a hand-built month with its eval report laid out differently; the data are the base month's
    "model-switch-visual-p1": ("model-switch-visual", {"tokens_page": 1, "pointer_note": False}),
}

LEGACY_COLORS = {"Dunlin-3-8B-Instruct": "#9CA3AF", "Vesk 4 Lite": "#10B981", "Osric 2 Thinking": "#8E6BD1",
                 "Vesk 4": "#4C7BD9", "Vesk 4 Pro": "#E8A33D"}
LEGACY_SHORT = {"Account & password": "Account", "Billing": "Billing", "Technical troubleshooting": "Technical",
                "Integrations & API": "Integrations"}


def register(spec):
    _REGISTRY[spec["name"]] = spec
    return spec


def spec(name):
    """The spec for a world name: registered, hand-built, saved next to its data, or regenerated from its name."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    from . import data

    if name in data.SCENARIOS:
        return register(legacy_spec(name))
    if name in VARIANTS:
        from evals.truth import LEGACY_TRUTH

        base, layout = VARIANTS[name]
        s = legacy_spec(base)
        return register({**s, "name": name, "base": base, "eval": {**s["eval"], **layout}, "truth": LEGACY_TRUTH[base]})
    saved = WORLD / name / "spec.json"
    if saved.exists():
        return register(json.loads(saved.read_text()))
    if name.startswith("g-"):
        from .generator import from_name

        return register(from_name(name))
    raise KeyError(f"unknown world {name!r}")


def save(spec):
    path = WORLD / spec["name"] / "spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))


def legacy_spec(name):
    """The hand-built Fernledger month `name`, as a spec. Rebuilds byte-for-byte as before."""
    from . import data as d

    s = d.SCENARIOS[name]
    schedules = [{"id": "price-schedule-2026-08", "title": "Larkfield Cloud model price schedule, effective Aug 1, 2026 "
                  "(scanned copy received by Finance)", "effective": "2026-08-01", "effective_text": "August 1, 2026",
                  "received": "JUL 24 2026", "supersedes": "May 1, 2026", "prices": d.PRICES,
                  "asset": "price-schedule-2026-08.jpg", "file": "price-schedule-2026-08.jpg"}]
    change = s.get("price_change")
    if change:
        current = {m: (({"input": change["input"], "output": change["output"]}) if m == change["model"] else p)
                   for m, p in d.PRICES.items()}
        schedules.append({"id": "price-schedule-2026-09", "title": "Larkfield Cloud model price schedule, effective Sep 1, "
                          "2026 (scanned copy received by Finance)", "effective": change["effective"],
                          "effective_text": "September 1, 2026", "received": "AUG 25 2026", "supersedes": "August 1, 2026",
                          "prices": current, "file": "price-schedule-2026-09.jpg"})
    events = [{"type": "volume", "start": "2026-09-01", "multiplier": s["sep_volume"]}]
    if s["switch_complete"]:
        events.append({"type": "model_switch", "from": "Vesk 4", "to": "Osric 2 Thinking", "rollout": s["rollout"],
                       "complete": s["switch_complete"]})
    return {
        "name": name, "authored": True, "family": {"model-switch": "model_switch", "model-switch-visual": "model_switch",
                                                  "volume-growth": "volume", "price-increase": "price"}[name],
        "salt": "", "company": "Fernledger", "provider": "Larkfield Cloud", "eval_team": "Fernledger Eval Team",
        "first_day": d.FIRST_DAY.isoformat(), "last_day": d.LAST_DAY.isoformat(), "today": "2026-09-22",
        "period": list(d.DEFAULT_PERIOD), "compare": list(d.DEFAULT_COMPARE),
        "models": {m: {"slug": d.SLUGS[m], "tokens": {t: list(d.legacy_tokens(m, t)) for t in d.TICKET_TYPES},
                       "pass_pct": dict(zip(d.TICKET_TYPES, d.PASS_PCT[m])), "color": LEGACY_COLORS[m]} for m in d.PRICES},
        "ticket_types": list(d.TICKET_TYPES), "short": LEGACY_SHORT, "share": dict(d.SHARE),
        "requests_per_ticket": dict(d.REQUESTS_PER_TICKET), "eval_tickets": dict(d.EVAL_TICKETS),
        "weekday_tickets": d.WEEKDAY_TICKETS, "weekend_factor": d.WEEKEND_FACTOR,
        "production": {t: "Vesk 4" for t in d.TICKET_TYPES}, "baseline_model": "Vesk 4",
        "baseline_reason": "Vesk 4 is the baseline approved by the June 2026 quality review.",
        "events": events, "changes": [list(c) for c in s["changes"]], "router": s["router"],
        "price_schedules": schedules,
        "eval": {"in_db": s.get("eval_in_db", True), "id": "eval-report-2026-08", "published": "Aug 28, 2026",
                 "title": "Help-desk offline eval v4: results by model and ticket type (Eval Team report export, Aug 28, 2026)",
                 "tokens_page": 2, "pointer_note": True},
        "attach_overview": s.get("attach_overview", False), "request": s.get("request"),
        "scan": {"rotate": 0.7, "blur": 0.25, "contrast": 0.92, "quality": 70},
    }
