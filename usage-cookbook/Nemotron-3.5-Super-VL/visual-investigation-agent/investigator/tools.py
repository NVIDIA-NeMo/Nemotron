# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's tools: the only way it can see the help desk or act on it.

Eight general tools. Dashboards and scanned documents come back as images; data comes back from read-only SQL;
calculations run in a Python sandbox; the one action, propose_change, only files a proposal for a person.
None of them returns a month's answer. Three optional tools (notes, zoom_image, verify_figures) are off in the
agent and can be switched on in its configuration.
"""

import json
import sqlite3
from pathlib import Path

import world

from . import sandbox
from .config import DEFAULT
from .contract import report_schema
from .policy import check_proposal

MAX_IMAGES = 4  # images per request to the model


def _fn(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


DATE = {"type": "string", "description": "YYYY-MM-DD"}
WHY = {"type": "string", "description": "One sentence: what you have seen so far that leads to this call, and what you expect to learn."}
ACTION_TOOLS = ("render_dashboard", "view_document", "run_sql", "run_python", "propose_change", "zoom_image")
FIGURE_METRICS = ["spend", "tickets", "resolved_tickets", "resolution_rate_pct", "cost_per_ticket", "tokens_per_ticket",
                  "input_tokens_per_ticket", "reasoning_tokens_per_ticket", "requests_per_ticket", "price_per_1m_tokens"]


def tools_for(scenario="model-switch", config=None):
    """The tool schemas the model sees in this world, under this configuration."""
    config = config or DEFAULT
    s = world.spec(scenario)
    images = config["image_budget"]
    image_note = f"Each request to the model holds at most {images} images, including images the user attached."
    why = {"why": WHY} if config["why"] else {}
    need = ["why"] if config["why"] else []
    routes = {"type": "object", "description": "Model for each ticket type",
              "properties": {t: {"type": "string", "enum": list(s["models"])} for t in s["ticket_types"]},
              "required": list(s["ticket_types"])}
    tools = [
        _fn("list_dashboards", "List the team's dashboards, their tabs, and each tab's parameters.", {}, []),
        _fn("render_dashboard", "Render one dashboard tab as an image, like a BI tool's render or export API. "
            f"The image is attached in the next message. {image_note}", {
                "dashboard_id": {"type": "string"}, "tab": {"type": "string", "enum": [t["tab"] for t in world.DASHBOARDS[0]["tabs"]]},
                "start_date": DATE, "end_date": DATE, "compare_start_date": DATE, "compare_end_date": DATE, **why},
            ["dashboard_id", "tab", *need]),
        _fn("list_documents", "List the team's documents.", {}, []),
        _fn("view_document", "Open one page of a document. Scanned documents come back as an image attached in the next "
            f"message; text documents come back as text. {image_note}",
            {"doc_id": {"type": "string"}, "page": {"type": "integer"}, **why}, ["doc_id", *need]),
        _fn("run_sql", "Run one read-only SQLite query against the help-desk database. Returns up to 200 rows.",
            {"query": {"type": "string"}, **why}, ["query", *need]),
        _fn("run_python", "Run Python 3 (standard library only) in a sandbox with no network. A read-only sqlite3 connection "
            "to the help-desk database is available as `conn`. Print what you need; stdout is returned.",
            {"code": {"type": "string"}, **why}, ["code", *need]),
        _fn("propose_change", "File one routing change for human review. Policy checks run first; a rejected proposal is "
            "not recorded. Nothing changes until a person approves.", {
                "title": {"type": "string"}, "routes": routes,
                "expected_monthly_savings_usd": {"type": "number", "description": "30 x recent average tickets per day x (cost per "
                                                 "ticket of the setup running today, measured after any recent change finished "
                                                 "rolling out, minus the projected cost per ticket of this proposal)"},
                "rationale": {"type": "string"}, **why},
            ["title", "routes", "expected_monthly_savings_usd", "rationale", *need]),
    ]
    flags = config["tools"]
    if flags["savings_inputs"]:  # the proposal states the inputs of its savings, and the tool checks them against the data
        params = tools[-1]["function"]["parameters"]
        params["properties"].update({
            "basis_start_date": {**DATE, "description": "First day of the window the daily tickets and current cost come from"},
            "basis_end_date": {**DATE, "description": "Last day of that window"},
            "daily_tickets": {"type": "number", "description": "Average tickets per day over the window"},
            "current_cost_per_ticket": {"type": "number", "description": "Measured cost per ticket over the window (USD)"},
            "proposed_cost_per_ticket": {"type": "number", "description": "Projected cost per ticket of this proposal (USD)"}})
        params["required"] = params["required"][:-len(need) or None] + ["basis_start_date", "basis_end_date", "daily_tickets",
                                                                           "current_cost_per_ticket", "proposed_cost_per_ticket", *need]
    if flags["notes"]:
        tools.append(_fn("notes", "Keep working notes: a plan, findings so far, open questions. The notes are returned to you "
                         "after each change. They are not shown to the user.",
                         {"text": {"type": "string"}, "mode": {"type": "string", "enum": ["replace", "append"]}}, ["text"]))
    if flags["zoom"]:
        tools.append(_fn("zoom_image", "Crop part of an image you have already received and return it enlarged, to read small "
                         f"text or numbers. The region is given as fractions of the width and height (0 to 1). {image_note}", {
                             "image_id": {"type": "string", "description": "For example img-2"},
                             "left": {"type": "number"}, "top": {"type": "number"},
                             "width": {"type": "number"}, "height": {"type": "number"}, **why},
                         ["image_id", "left", "top", "width", "height", *need]))
    if flags["check_report"]:
        tools.append(_fn("verify_figures", "Recompute figures you plan to report from the help-desk database and compare them "
                         "with your values. Each figure names a metric, a date range, and optionally a ticket type or model.", {
                             "figures": {"type": "array", "items": {"type": "object", "properties": {
                                 "metric": {"type": "string", "enum": FIGURE_METRICS}, "value": {"type": "number"},
                                 "start_date": DATE, "end_date": DATE, "ticket_type": {"type": "string"},
                                 "model": {"type": "string"}}, "required": ["metric", "value", "start_date", "end_date"]}}},
                         ["figures"]))
    report = report_schema(config)
    tools.append(_fn("submit_report", "Submit the final report. Call once, at the end.", report, list(report)))
    return tools


TOOLS = tools_for()


def _authorizer(action, *_):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _round(v):
    return round(v, 6) if isinstance(v, float) else v


def _clip(result, limit=12_000):
    text = json.dumps(result)
    return result if len(text) <= limit else {"truncated_output": text[:limit], "note": f"output clipped to {limit} characters"}


def run_sql(conn, query):
    conn.set_authorizer(_authorizer)
    try:
        cur = conn.execute(query)
        rows = cur.fetchmany(201)
    except sqlite3.Error as error:
        return {"error": f"SQL error: {error}"}
    columns = [c[0] for c in cur.description or []]
    return _clip({"columns": columns, "rows": [[_round(v) for v in r] for r in rows[:200]], "truncated": len(rows) > 200})


def _model_name(name, scenario="model-switch"):
    models = world.spec(scenario)["models"]
    by_slug = {m["slug"]: k for k, m in models.items()}
    return name if name in models else by_slug.get(str(name).strip().lower())


def check_savings_inputs(arguments, conn):
    """Compare the proposal's stated savings inputs with the data: daily tickets, measured cost, and the arithmetic."""
    try:
        start, end = str(arguments["basis_start_date"]), str(arguments["basis_end_date"])
        daily, current, proposed = (float(arguments[k]) for k in ("daily_tickets", "current_cost_per_ticket", "proposed_cost_per_ticket"))
        savings = float(arguments["expected_monthly_savings_usd"])
    except (KeyError, TypeError, ValueError):
        return "state basis_start_date, basis_end_date, daily_tickets, current_cost_per_ticket, and proposed_cost_per_ticket as numbers and dates"
    import datetime as dt

    try:
        days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    except ValueError:
        return "basis dates must be YYYY-MM-DD"
    tickets = _figure(conn, "tickets", start, end)
    measured = _figure(conn, "cost_per_ticket", start, end)
    if not tickets or days <= 0:
        return f"no tickets between {start} and {end}"
    problems = []
    if abs(daily - tickets / days) > 0.02 * tickets / days:
        problems.append(f"daily_tickets {daily:,.0f} does not match the data: {tickets / days:,.0f} tickets per day from {start} to {end}")
    if abs(current - measured) > 0.02 * measured:
        problems.append(f"current_cost_per_ticket {current:.6f} does not match the data: {measured:.6f} measured from {start} to {end}")
    if abs(savings - 30 * daily * (current - proposed)) > 0.01 * max(abs(savings), 1):
        problems.append(f"expected_monthly_savings_usd {savings:,.2f} is not 30 x daily_tickets x (current - proposed) = "
                        f"{30 * daily * (current - proposed):,.2f}")
    return "; ".join(problems) or None


def propose_change(arguments, state, enforce, scenario="model-switch", conn=None, savings_inputs=False):
    """File a routing proposal for review, after the policy checks. Nothing in the help desk changes."""
    routes = {t: _model_name(m, scenario) for t, m in (arguments.get("routes") or {}).items()}
    rejection = check_proposal(routes, state, scenario) if enforce else None
    if not rejection and savings_inputs and conn is not None:
        problem = check_savings_inputs(arguments, conn)
        rejection = f"Rejected: the savings inputs do not check out: {problem}. Nothing was recorded." if problem else None
    if rejection:
        return {"error": rejection}
    proposal_id = f"PRP-{len(state['proposals']) + 1:04d}"
    state["proposals"].append({"proposal_id": proposal_id, **arguments, "routes": routes})
    return {"status": "pending_review", "proposal_id": proposal_id, "note": "Recorded for human review. Nothing was changed."}


def notes(arguments, state):
    text = str(arguments.get("text", ""))
    state["notes"] = (state.get("notes", "") + "\n" + text).strip() if arguments.get("mode") == "append" else text
    return {"status": "saved", "notes": state["notes"]}


def zoom_image(arguments, state, out_dir):
    from PIL import Image

    source = state.get("image_paths", {}).get(str(arguments.get("image_id")))
    if not source:
        return {"error": f"unknown image_id {arguments.get('image_id')!r}; use an id from an image you received"}, None
    try:
        left, top, width, height = (float(arguments[k]) for k in ("left", "top", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return {"error": "left, top, width, and height must be numbers between 0 and 1"}, None
    if not (0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 - left + 1e-9 and 0 < height <= 1 - top + 1e-9):
        return {"error": "the region must lie inside the image: 0 <= left, top < 1 and left + width, top + height <= 1"}, None
    image = Image.open(source)
    w, h = image.size
    box = (round(left * w), round(top * h), round((left + width) * w), round((top + height) * h))
    crop = image.crop(box)
    scale = max(1.0, min(3.0, 1400 / max(crop.size)))
    crop = crop.resize((round(crop.size[0] * scale), round(crop.size[1] * scale)), Image.Resampling.LANCZOS)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"zoom-{len(list(out_dir.glob('zoom-*.png'))) + 1}.png"
    crop.convert("RGB").save(path)
    desc = f"zoom of {arguments['image_id']} ({left:.2f}, {top:.2f}, {width:.2f}, {height:.2f})"
    return {"status": "zoomed", "description": desc}, (str(path), desc)


def _figure(conn, metric, start, end, ticket_type=None, model=None):
    where, args = ["date BETWEEN ? AND ?"], [start, end]
    for column, value in (("ticket_type", ticket_type), ("model", model)):
        if value:
            where.append(f"{column} = ?")
            args.append(value)
    clause = " AND ".join(where)
    t, r, i, a, z, q = conn.execute(f"SELECT SUM(tickets), SUM(resolved_tickets), SUM(input_tokens), SUM(output_answer_tokens), "
                                    f"SUM(output_reasoning_tokens), SUM(requests) FROM usage_daily WHERE {clause}", args).fetchone()
    spend = conn.execute(f"SELECT SUM(total_usd) FROM billing_daily WHERE {clause}", args).fetchone()[0]
    if not t:
        return None
    tokens = i + a + z
    return {"spend": spend, "tickets": t, "resolved_tickets": r, "resolution_rate_pct": 100 * r / t, "cost_per_ticket": spend / t,
            "tokens_per_ticket": tokens / t, "input_tokens_per_ticket": i / t, "reasoning_tokens_per_ticket": z / t,
            "requests_per_ticket": q / t, "price_per_1m_tokens": spend * 1e6 / tokens}[metric]


def verify_figures(conn, arguments):
    rows = []
    for figure in (arguments.get("figures") or [])[:30]:
        try:
            actual = _figure(conn, figure["metric"], figure["start_date"], figure["end_date"], figure.get("ticket_type"), figure.get("model"))
        except (KeyError, sqlite3.Error) as exc:
            rows.append({**figure, "error": f"could not compute: {exc}"})
            continue
        if actual is None:
            rows.append({**figure, "error": "no rows match this date range, ticket type, and model"})
            continue
        given = figure.get("value")
        ok = isinstance(given, (int, float)) and abs(given - actual) <= 0.005 * abs(actual)
        rows.append({**figure, "database_value": _round(actual), "matches": ok})
    return {"figures": rows}


def execute(scenario, conn, name, arguments, state, enforce, out_dir=None, config=None):
    """Return (result, image) where image is (path, description) or None."""
    config = config or DEFAULT
    if name == "list_dashboards":
        return {"dashboards": world.dashboards(scenario)}, None
    if name == "render_dashboard":
        if arguments.get("dashboard_id") != world.DASHBOARDS[0]["id"]:
            return {"error": f"unknown dashboard_id; call list_dashboards"}, None
        out = world.render(scenario, arguments.get("tab"), **{k: arguments.get(k) for k in
                           ("start_date", "end_date", "compare_start_date", "compare_end_date")})
        return (out, None) if "error" in out else ({"status": "rendered", "description": out["description"]}, (out["path"], out["description"]))
    if name == "list_documents":
        return {"documents": world.list_documents(scenario)}, None
    if name == "view_document":
        out = world.document(scenario, arguments.get("doc_id"), int(arguments.get("page") or 1))
        if out.get("format") == "image":
            desc = f"{out['doc_id']}, page {out.get('page', 1)}"
            return {"status": "opened", "description": desc}, (out["path"], desc)
        return out, None
    if name == "run_sql":
        return run_sql(conn, str(arguments.get("query", ""))), None
    if name == "run_python":
        return _clip(sandbox.run_python(scenario, str(arguments.get("code", "")))), None
    if name == "propose_change":
        return propose_change(arguments, state, enforce, scenario, conn, config["tools"]["savings_inputs"]), None
    if name == "notes":
        return notes(arguments, state), None
    if name == "zoom_image":
        return zoom_image(arguments, state, Path(out_dir or world.WORLD / scenario / "zooms"))
    if name == "verify_figures":
        return verify_figures(conn, arguments), None
    return {"error": f"unknown tool {name!r}"}, None
