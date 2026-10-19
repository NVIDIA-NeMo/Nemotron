# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The help desk's data: models and prices, what happened in each scenario month, and the SQLite tables.

Everything is fictional and illustrative. Every scenario has the same tables; scenarios differ only in what
happened in September. The data is generated deterministically, so every build is identical.
"""

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path

from .spec import WORLD

FIRST_DAY, LAST_DAY = dt.date(2026, 8, 1), dt.date(2026, 9, 21)
DEFAULT_PERIOD = ("2026-09-01", "2026-09-21")
DEFAULT_COMPARE = ("2026-08-01", "2026-08-21")

PRICES = {  # USD per 1M tokens; reasoning tokens bill as output
    "Dunlin-3-8B-Instruct": {"input": 0.065, "output": 0.325},
    "Vesk 4 Lite": {"input": 0.17, "output": 0.68},
    "Osric 2 Thinking": {"input": 0.475, "output": 1.90},
    "Vesk 4": {"input": 1.90, "output": 7.60},
    "Vesk 4 Pro": {"input": 5.70, "output": 22.80},
}
SLUGS = {m: m.lower().replace(" ", "-") for m in PRICES}
TICKET_TYPES = ["Account & password", "Billing", "Technical troubleshooting", "Integrations & API"]
SHARE = {"Account & password": 0.34, "Billing": 0.26, "Technical troubleshooting": 0.28, "Integrations & API": 0.12}
STANDARD_TOKENS = {  # per ticket: input, visible answer, reasoning
    "Account & password": (9_000, 600, 100),
    "Billing": (12_000, 780, 120),
    "Technical troubleshooting": (26_000, 1_900, 300),
    "Integrations & API": (30_000, 2_250, 350),
}
REQUESTS_PER_TICKET = {"Account & password": 7, "Billing": 7, "Technical troubleshooting": 10, "Integrations & API": 10}
PASS_PCT = {  # offline eval resolution rate by model and ticket type
    "Dunlin-3-8B-Instruct": [81.0, 70.0, 38.0, 30.0],
    "Vesk 4 Lite": [92.1, 88.2, 58.0, 49.0],
    "Osric 2 Thinking": [92.0, 88.0, 72.0, 64.0],
    "Vesk 4": [93.0, 89.0, 76.0, 71.0],
    "Vesk 4 Pro": [94.0, 91.0, 82.0, 79.0],
}
EVAL_TICKETS = {"Account & password": 3_000, "Billing": 2_400, "Technical troubleshooting": 2_600, "Integrations & API": 1_500}
WEEKDAY_TICKETS = 120_000
WEEKEND_FACTOR = 0.55

ROUTER = """# gateway router config for the help-desk agent (router/helpdesk.yaml)
router: helpdesk-agent
default_model: {default}
routes: []          # no per-ticket-type routes; every ticket uses default_model
fallbacks:
  - {fallback}
"""

SCENARIOS = {
    "model-switch": {
        "request": ("Our help-desk AI bill is up about 30% this month, even though we moved to a cheaper model. What "
                    "happened? If we can route tickets differently to cut cost without hurting quality, prepare that "
                    "change for review; don't change anything yet."),
        "rollout": {"2026-09-08": 0.25, "2026-09-09": 0.60}, "switch_complete": "2026-09-10", "sep_volume": 1.03,
        "router": ROUTER.format(default="osric-2-thinking", fallback="vesk-4"),
        "changes": [
            ("CHG-5031", "2026-08-19", "17:20", "Help-center search index rebuilt (no content change)"),
            ("CHG-5107", "2026-09-03", "16:05", "Help-center articles refreshed for Q3 policy updates"),
            ("CHG-5120", "2026-09-08", "16:00", "Default help-desk model: Vesk 4 -> Osric 2 Thinking, rollout to 25% of tickets"),
            ("CHG-5124", "2026-09-09", "16:00", "Osric 2 Thinking rollout increased to 60% of tickets"),
            ("CHG-5129", "2026-09-10", "16:00", "Osric 2 Thinking rollout completed (100% of tickets)"),
        ],
    },
    "volume-growth": {
        "rollout": {}, "switch_complete": None, "sep_volume": 1.30,
        "router": ROUTER.format(default="vesk-4", fallback="vesk-4-pro"),
        "changes": [
            ("CHG-5031", "2026-08-19", "17:20", "Help-center search index rebuilt (no content change)"),
            ("CHG-5098", "2026-08-31", "22:00", "In-app help chat enabled for all mobile app users"),
            ("CHG-5107", "2026-09-03", "16:05", "Help-center articles refreshed for Q3 policy updates"),
            ("CHG-5116", "2026-09-08", "16:00", "Help-desk reply tone guidelines updated (prompt wording only)"),
        ],
    },
}

SCENARIOS["model-switch-visual"] = {**SCENARIOS["model-switch"], "eval_in_db": False, "attach_overview": True}
SCENARIOS["price-increase"] = {  # held out: built after the playbook edits were written
    "rollout": {}, "switch_complete": None, "sep_volume": 1.0,
    "router": ROUTER.format(default="vesk-4", fallback="vesk-4-pro"),
    "price_change": {"model": "Vesk 4", "effective": "2026-09-01", "input": 2.47, "output": 9.88},
    "changes": [
        ("CHG-5031", "2026-08-19", "17:20", "Help-center search index rebuilt (no content change)"),
        ("CHG-5107", "2026-09-03", "16:05", "Help-center articles refreshed for Q3 policy updates"),
        ("CHG-5116", "2026-09-08", "16:00", "Help-desk reply tone guidelines updated (prompt wording only)"),
    ],
}


def legacy_tokens(model, ticket_type):
    """Per-ticket input, answer, and reasoning tokens for a Fernledger catalog model."""
    inp, ans, rsn = STANDARD_TOKENS[ticket_type]
    if model == "Osric 2 Thinking":
        inp, ans, rsn = inp * 2.5, ans * 1.05, ans * 17
    elif model in ("Vesk 4 Lite", "Dunlin-3-8B-Instruct"):
        inp, ans, rsn = inp, ans * 0.95, 0
    elif model == "Vesk 4 Pro":
        inp, ans, rsn = inp, ans * 1.1, rsn * 3
    return round(inp), round(ans), round(rsn)


def _spec(scenario):
    from .spec import spec

    return spec(scenario)


def schedules(scenario):
    return _spec(scenario)["price_schedules"]


def price(scenario, model, day):
    """The list price of `model` on `day` (YYYY-MM-DD): the latest schedule in effect that lists it."""
    found = None
    for schedule in schedules(scenario):
        if schedule["effective"] <= day and model in schedule["prices"]:
            found = schedule["prices"][model]
    return found


def catalog(scenario):
    """The world's catalog models, in display order."""
    return list(_spec(scenario)["models"])


def ticket_types(scenario):
    return _spec(scenario)["ticket_types"]


def pass_pct(scenario, model, ticket_type):
    return _spec(scenario)["models"][model]["pass_pct"][ticket_type]


def current_prices(scenario):
    s = _spec(scenario)
    return {m: price(scenario, m, s["last_day"]) for m in s["models"]}


def eval_in_db(scenario):
    return _spec(scenario)["eval"]["in_db"]


def model_tokens(model, ticket_type, scenario=None):
    """Per-ticket tokens as {"input", "answer", "reasoning"}: the world's catalog, or Fernledger's by default."""
    if scenario is not None and not _spec(scenario).get("authored"):
        inp, ans, rsn = _spec(scenario)["models"][model]["tokens"][ticket_type]
    else:
        inp, ans, rsn = legacy_tokens(model, ticket_type)
    return {"input": inp, "answer": ans, "reasoning": rsn}


def _noise(*key, spread=0.04):
    digest = hashlib.sha256(":".join(map(str, key)).encode()).digest()
    return 1 + ((digest[0] / 255) - 0.5) * spread


def _days(start=FIRST_DAY, end=LAST_DAY):
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def dates(scenario):
    """(first day, last day) of the world's data as dates."""
    s = _spec(scenario)
    return dt.date.fromisoformat(s["first_day"]), dt.date.fromisoformat(s["last_day"])


def _events(s, kind):
    return [e for e in s["events"] if e["type"] == kind]


def _switch_share(event, day):
    if event["complete"] and day >= event["complete"]:
        return 1.0
    return event["rollout"].get(day, 0.0)


def _volume(s, day):
    factor = 1.0
    for event in _events(s, "volume"):
        if day >= event["start"]:
            factor *= event["multiplier"]
    return factor


def _shares(s, day):
    shares = dict(s["share"])
    for event in _events(s, "mix_shift"):
        if day >= event["start"]:
            shares = dict(event["share"])
    return shares


def _usage(s, ticket_type, day):
    """(input-token factor, requests factor) from retry bugs in effect on `day` for this ticket type."""
    tokens, requests = 1.0, 1.0
    for event in _events(s, "retry_bug"):
        if event["start"] <= day < (event.get("end") or "9999") and ticket_type in event["ticket_types"]:
            tokens *= event["input_factor"]
            requests *= event["requests_factor"]
    return tokens, requests


def day_rows(s, day):
    """One row per ticket type and model for `day`: tickets, resolutions, requests, tokens, and billed USD."""
    iso = day.isoformat()
    key = (s["salt"],) if s["salt"] else ()
    volume = s["weekday_tickets"] * (s["weekend_factor"] if day.weekday() >= 5 else 1.0) * _volume(s, iso)
    shares = _shares(s, iso)
    rows = []
    for ticket_type in s["ticket_types"]:
        tickets = round(volume * shares[ticket_type] * _noise(*key, "tickets", ticket_type, day))
        old = s["production"][ticket_type]
        split = {}
        for event in _events(s, "model_switch"):
            if event["from"] == old and ticket_type in event.get("ticket_types", s["ticket_types"]):
                split[event["to"]] = round(tickets * _switch_share(event, iso))
        split[old] = tickets - sum(split.values())
        token_factor, request_factor = _usage(s, ticket_type, iso)
        for model, n in split.items():
            if n == 0:
                continue
            drift = _noise(*key, "tokens", model, ticket_type, day, spread=0.04)
            inp, ans, rsn = s["models"][model]["tokens"][ticket_type]
            tok = {k: round(n * v * drift) for k, v in (("input", inp * token_factor), ("answer", ans), ("reasoning", rsn))}
            p = price(s["name"], model, iso)
            resolved_pct = s["models"][model]["pass_pct"][ticket_type] + (_noise(*key, "res", model, ticket_type, day, spread=0.02) - 1) * 50
            requests = round(n * s["requests_per_ticket"][ticket_type] * request_factor
                             * _noise(*key, "req", model, ticket_type, day, spread=0.02))
            rows.append({
                "date": iso, "ticket_type": ticket_type, "model": model, "tickets": n,
                "resolved_tickets": round(n * resolved_pct / 100), "requests": requests,
                "input_tokens": tok["input"], "output_answer_tokens": tok["answer"], "output_reasoning_tokens": tok["reasoning"],
                "input_usd": tok["input"] * p["input"] / 1e6,
                "output_usd": (tok["answer"] + tok["reasoning"]) * p["output"] / 1e6,
            })
    return rows


SCHEMA = """
CREATE TABLE usage_daily (date TEXT, ticket_type TEXT, model TEXT, tickets INTEGER, resolved_tickets INTEGER,
  requests INTEGER, input_tokens INTEGER, output_answer_tokens INTEGER, output_reasoning_tokens INTEGER);
CREATE TABLE billing_daily (date TEXT, ticket_type TEXT, model TEXT, input_usd REAL, output_usd REAL, total_usd REAL);
CREATE TABLE change_log (id TEXT, date TEXT, time_utc TEXT, title TEXT);
CREATE TABLE eval_results (model TEXT, ticket_type TEXT, resolution_rate_pct REAL, eval_tickets INTEGER,
  avg_input_tokens INTEGER, avg_output_answer_tokens INTEGER, avg_output_reasoning_tokens INTEGER, avg_requests INTEGER);
"""


def db_path(scenario):
    return WORLD / scenario / "helpdesk.sqlite"



def build_database(scenario):
    """Write the scenario's SQLite database from scratch."""
    s = _spec(scenario)
    path = db_path(scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    if not s.get("authored"):
        from .spec import save

        save(s)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for day in _days(*dates(scenario)):
        for r in day_rows(s, day):
            conn.execute("INSERT INTO usage_daily VALUES (?,?,?,?,?,?,?,?,?)",
                         (r["date"], r["ticket_type"], r["model"], r["tickets"], r["resolved_tickets"], r["requests"],
                          r["input_tokens"], r["output_answer_tokens"], r["output_reasoning_tokens"]))
            conn.execute("INSERT INTO billing_daily VALUES (?,?,?,?,?,?)",
                         (r["date"], r["ticket_type"], r["model"], round(r["input_usd"], 2), round(r["output_usd"], 2),
                          round(r["input_usd"] + r["output_usd"], 2)))
    conn.executemany("INSERT INTO change_log VALUES (?,?,?,?)", [tuple(c) for c in s["changes"]])
    if s["eval"]["in_db"]:
        for model, m in s["models"].items():
            for t in s["ticket_types"]:
                inp, ans, rsn = m["tokens"][t]
                conn.execute("INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?)",
                             (model, t, m["pass_pct"][t], s["eval_tickets"][t], inp, ans, rsn, s["requests_per_ticket"][t]))
    else:
        conn.execute("DROP TABLE eval_results")
    conn.commit()
    conn.close()
    return path


def content_sha256(scenario):
    """Hash of every table's rows. SQLite file bytes vary with the library version; the contents do not."""
    conn = connect(scenario)
    digest = hashlib.sha256()
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall():
        columns = len(conn.execute(f"SELECT * FROM {table} LIMIT 1").description)
        order = ", ".join(str(i + 1) for i in range(columns))
        digest.update(table.encode())
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            digest.update(json.dumps(row).encode())
    conn.close()
    return digest.hexdigest()


def connect(scenario):
    return sqlite3.connect(f"file:{db_path(scenario)}?mode=ro", uri=True)


def stats(conn, start, end, ticket_type=None):
    where, args = "date BETWEEN ? AND ?", [start, end]
    if ticket_type:
        where += " AND ticket_type = ?"
        args.append(ticket_type)
    t, r, i, a, z = conn.execute(f"SELECT SUM(tickets), SUM(resolved_tickets), SUM(input_tokens), SUM(output_answer_tokens), "
                                 f"SUM(output_reasoning_tokens) FROM usage_daily WHERE {where}", args).fetchone()
    spend = conn.execute(f"SELECT SUM(total_usd) FROM billing_daily WHERE {where}", args).fetchone()[0]
    return {"tickets": t or 0, "resolved": r or 0, "input": i or 0, "answer": a or 0, "reasoning": z or 0, "spend": spend or 0.0}
