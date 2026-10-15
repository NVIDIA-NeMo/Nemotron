# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The team's dashboard: three tabs (Overview, Models, Changes) built as HTML from the database.

The agent never clicks anything. It asks for a tab through a render call and gets a screenshot back, the way
BI tools export a dashboard view as an image.
"""

import datetime as dt
import hashlib
import math

from .data import (DEFAULT_COMPARE, DEFAULT_PERIOD, FIRST_DAY, LAST_DAY, PASS_PCT, SHARE, TICKET_TYPES, WORLD,
                   _days, connect, current_prices, model_tokens, stats)
from .browser import render_lock, screenshot
from .spec import spec

DASHBOARDS = [{
    "id": "helpdesk-ai", "title": "Fernledger · AI Help Desk",
    "tabs": [
        {"tab": "Overview", "shows": "KPI cards (spend, tickets, cost per ticket, resolution rate, average price per 1M tokens); "
         "spend change by ticket type; daily tokens by type and daily spend; table by ticket type",
         "parameters": "start_date, end_date, compare_start_date, compare_end_date (YYYY-MM-DD, all optional; "
         "default Sep 1-21 compared with Aug 1-21, 2026)"},
        {"tab": "Models", "shows": "Model catalog: offline-eval resolution rate versus list price", "parameters": "none"},
        {"tab": "Changes", "shows": "Production change log", "parameters": "start_date, end_date (optional)"},
    ],
    "data_range": f"{FIRST_DAY.isoformat()} to {LAST_DAY.isoformat()}",
}]
TOKEN_COLORS = {"input": "#4C7BD9", "answer": "#9CC3F0", "reasoning": "#8E6BD1"}
MODEL_COLORS = {"Dunlin-3-8B-Instruct": "#9CA3AF", "Vesk 4 Lite": "#10B981", "Osric 2 Thinking": "#8E6BD1", "Vesk 4": "#4C7BD9", "Vesk 4 Pro": "#E8A33D"}
FONT = 'font-family="Inter,Helvetica,Arial"'
CSS = """
*{box-sizing:border-box}body{margin:0;background:#F3F4F6;font-family:Inter,Helvetica,Arial,sans-serif;color:#111827}
.top{height:56px;background:#fff;border-bottom:1px solid #E5E7EB;display:flex;align-items:center;padding:0 28px;gap:28px}
.brand{font-weight:700;font-size:17px}.tab{font-size:14px;color:#6B7280}.tab.on{color:#111827;font-weight:600}
.head{display:flex;justify-content:space-between;align-items:center;padding:18px 28px 6px}h1{font-size:22px;margin:0}
.pill{display:inline-block;border:1px solid #D1D5DB;background:#fff;border-radius:8px;padding:6px 12px;font-size:13px;margin-left:8px}
.badge{display:inline-block;background:#FEF3C7;color:#92400E;border-radius:6px;padding:3px 8px;font-size:12px;margin-left:10px;vertical-align:3px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;padding:10px 28px}
.kpi{background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:12px 14px}.kl{font-size:12.5px;color:#4B5563}
.kv{font-size:26px;font-weight:700;margin:4px 0}.ks{font-size:12.5px;color:#6B7280}
.row{display:grid;grid-template-columns:640px 1fr;gap:14px;padding:6px 28px}
.card{background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:14px 16px}.ct{font-weight:600;font-size:15px}.cs{font-size:12.5px;color:#6B7280;margin:2px 0 6px}
.lg{font-size:12px;color:#374151;margin-right:14px}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:#6B7280;font-weight:500;border-bottom:1px solid #E5E7EB;padding:7px 8px}
td{border-bottom:1px solid #F0F1F3;padding:7px 8px}.r{text-align:right}
"""


def _k(v):
    return f"${v / 1000:,.1f}k"


def _pct(new, old):
    if not old:
        return "n/a"
    change = round(100 * (new / old - 1), 1)
    return f"{change + 0.0:+.1f}%"


def _date(s):
    return dt.date.fromisoformat(s)


def fmt_range(start, end):
    a, b = _date(start), _date(end)
    if a.month == b.month:
        return f"{a.strftime('%b')} {a.day}–{b.day}, {b.year}"
    return f"{a.strftime('%b')} {a.day} – {b.strftime('%b')} {b.day}, {b.year}"


def _nice(v, steps=4):
    if v <= 0:
        return 1.0
    raw = v / steps
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * mag >= raw)
    return step * steps


def dashboards(scenario):
    """The dashboard list for a world: Fernledger's for the hand-built months, otherwise the world's company and dates."""
    s = spec(scenario)
    if s.get("authored"):
        return DASHBOARDS
    tabs = [dict(t) for t in DASHBOARDS[0]["tabs"]]
    period, compare = (tuple(_date(d) for d in s[k]) for k in ("period", "compare"))
    short = lambda a, b: f"{a.strftime('%b')} {a.day}-{b.day}" if a.month == b.month else f"{a.strftime('%b')} {a.day}-{b.strftime('%b')} {b.day}"
    tabs[0]["parameters"] = (f"start_date, end_date, compare_start_date, compare_end_date (YYYY-MM-DD, all optional; "
                             f"default {short(*period)} compared with {short(*compare)}, {period[1].year})")
    return [{"id": DASHBOARDS[0]["id"], "title": f"{s['company']} · AI Help Desk", "tabs": tabs,
             "data_range": f"{s['first_day']} to {s['last_day']}"}]


def _top(active, s=None):
    brand = f"{s['company']} · AI Help Desk" if s else "Fernledger · AI Help Desk"
    return (f'<div class="top"><span class="brand">{brand}</span>'
            + "".join(f'<span class="tab{" on" if t["tab"] == active else ""}">{t["tab"]}</span>' for t in DASHBOARDS[0]["tabs"])
            + "</div>")


def _kpis(p, c):
    cp, cc = p["spend"] / p["tickets"], c["spend"] / c["tickets"]
    tp = (p["input"] + p["answer"] + p["reasoning"]) / 1e6
    tc = (c["input"] + c["answer"] + c["reasoning"]) / 1e6
    cards = [
        ("AI spend", _k(p["spend"]), f"{_pct(p['spend'], c['spend'])} vs {_k(c['spend'])}"),
        ("Tickets handled by AI", f"{p['tickets'] / 1e6:.2f}M", f"{_pct(p['tickets'], c['tickets'])} vs {c['tickets'] / 1e6:.2f}M"),
        ("AI cost per ticket", f"${cp:.4f}", f"{_pct(cp, cc)} vs ${cc:.4f}"),
        ("Resolution rate", f"{100 * p['resolved'] / p['tickets']:.1f}%", f"was {100 * c['resolved'] / c['tickets']:.1f}%"),
        ("Average price per 1M tokens", f"${p['spend'] / tp:.2f}", f"{_pct(p['spend'] / tp, c['spend'] / tc)} vs ${c['spend'] / tc:.2f}"),
    ]
    return "".join(f'<div class="kpi"><div class="kl">{x}</div><div class="kv">{y}</div><div class="ks">{z}</div></div>' for x, y, z in cards)


def _change_by_type(conn, period, compare, s):
    rows = [(t, stats(conn, *period, t)["spend"] - stats(conn, *compare, t)["spend"]) for t in s["ticket_types"]]
    rows.sort(key=lambda kv: -kv[1])
    w, h, left, right, top_, bottom = 600, 300, 60, 50, 36, 48
    pw, ph = w - left - right, h - top_ - bottom
    hi, lo = max(0.0, max(v for _, v in rows)), min(0.0, min(v for _, v in rows))
    span = _nice(hi - lo)
    ymax = hi + (span - (hi - lo)) if lo == 0 else _nice(hi)
    ymin = ymax - span if lo < 0 else 0.0
    Y = lambda v: top_ + ph - ph * (v - ymin) / (ymax - ymin)
    parts = [f'<svg width="{w}" height="{h}" {FONT} font-size="12">']
    for i in range(5):
        v = ymin + (ymax - ymin) * i / 4
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#E6E8EC"/>'
                     f'<text x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#6B7280">{"-" if v < 0 else ""}${abs(v) / 1000:,.1f}k</text>')
    total = sum(v for _, v in rows)
    cumulative = total > 0 and all(v >= 0 for _, v in rows)
    if cumulative:
        for pct in (0, 20, 40, 60, 80, 100):
            y = top_ + ph - ph * pct / 100
            parts.append(f'<text x="{w - right + 8}" y="{y + 4}" fill="#6B7280">{pct}%</text>')
    bw = pw / len(rows) * 0.5
    cum, points = 0.0, []
    for i, (name, v) in enumerate(rows):
        cx = left + pw / len(rows) * (i + 0.5)
        y0, y1 = sorted((Y(0), Y(v)))
        short = s["short"][name]
        parts.append(f'<rect x="{cx - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(y1 - y0, 0.5):.1f}" fill="#4C7BD9" rx="2"/>'
                     f'<text x="{cx:.1f}" y="{(y0 - 7) if v >= 0 else (y1 + 16):.1f}" text-anchor="middle" font-weight="600">{"+" if v >= 0 else "-"}{_k(abs(v))}</text>'
                     f'<text x="{cx:.1f}" y="{h - bottom + 18}" text-anchor="middle" fill="#374151">{short}</text>')
        cum += v
        points.append((cx, top_ + ph - ph * cum / total if cumulative else 0, 100 * cum / total if total else 0))
    if cumulative:
        y80 = top_ + ph * 0.2
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y80}" y2="{y80}" stroke="#9CA3AF" stroke-dasharray="5,4"/>')
        parts.append('<polyline fill="none" stroke="#1F2937" stroke-width="2" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points) + '"/>')
        for x, y, pct in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1F2937"/>'
                         f'<text x="{x + 12:.1f}" y="{y + 16:.1f}" stroke="#fff" stroke-width="3" paint-order="stroke">{pct:.0f}%</text>')
    parts.append(f'<text x="14" y="{top_ + ph / 2}" transform="rotate(-90 14 {top_ + ph / 2})" text-anchor="middle" fill="#6B7280">Change (USD)</text></svg>')
    note = "sorted, with cumulative share (dashed line: 80%)" if cumulative else "sorted"
    return note, "".join(parts)


def _daily(conn, period, compare):
    first = min(_date(period[0]), _date(compare[0]))
    last = max(_date(period[1]), _date(compare[1]))
    days = _days(first, last)
    data = {}
    for d, i, a, z in conn.execute("SELECT date, SUM(input_tokens), SUM(output_answer_tokens), SUM(output_reasoning_tokens) "
                                   "FROM usage_daily WHERE date BETWEEN ? AND ? GROUP BY date", (first.isoformat(), last.isoformat())):
        data[d] = {"input": i / 1e9, "answer": a / 1e9, "reasoning": z / 1e9}
    for d, usd in conn.execute("SELECT date, SUM(total_usd) FROM billing_daily WHERE date BETWEEN ? AND ? GROUP BY date",
                               (first.isoformat(), last.isoformat())):
        data[d]["usd"] = usd
    w, h, left, right, top_, bottom = 690, 300, 52, 56, 40, 50
    pw, ph = w - left - right, h - top_ - bottom
    ymax_tok = _nice(max(v["input"] + v["answer"] + v["reasoning"] for v in data.values()) * 1.05, 3)
    ymax_usd = _nice(max(v["usd"] for v in data.values()) * 1.05, 3)
    slot = pw / len(days)
    bw = slot * 0.72
    parts = [f'<svg width="{w}" height="{h}" {FONT} font-size="12">']
    for name, (a, b), fill in (("Compared", compare, "#F3F4F6"), ("Selected", period, "#EEF4FF")):
        x0 = left + slot * (_date(a) - first).days
        x1 = left + slot * ((_date(b) - first).days + 1)
        parts.append(f'<rect x="{x0:.1f}" y="{top_}" width="{x1 - x0:.1f}" height="{ph}" fill="{fill}"/>'
                     f'<text x="{(x0 + x1) / 2:.1f}" y="{h - 10}" text-anchor="middle" font-weight="600">{name}: {fmt_range(a, b)}</text>')
    for i in range(4):
        y = top_ + ph - ph * i / 3
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E6E8EC"/>'
                     f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#6B7280">{ymax_tok * i / 3:g}B</text>'
                     f'<text x="{w - right + 8}" y="{y + 4:.1f}" fill="#B45309">${ymax_usd * i / 3 / 1000:,.1f}k</text>')
    line = []
    for i, day in enumerate(days):
        v = data.get(day.isoformat())
        if not v:
            continue
        x, y = left + slot * i + (slot - bw) / 2, top_ + ph
        for t in ("input", "answer", "reasoning"):
            seg = ph * v[t] / ymax_tok
            y -= seg
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{seg:.1f}" fill="{TOKEN_COLORS[t]}"/>')
        line.append(f"{x + bw / 2:.1f},{top_ + ph - ph * v['usd'] / ymax_usd:.1f}")
        if day.day in (1, 8, 15, 22):
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{top_ + ph + 16}" text-anchor="middle" fill="#374151">{day.strftime("%b")} {day.day}</text>')
    parts.append('<polyline fill="none" stroke="#B45309" stroke-width="2" points="' + " ".join(line) + '"/></svg>')
    legend = ("".join(f'<span class="lg"><i style="background:{TOKEN_COLORS[t]}"></i>{n}</span>'
                      for t, n in (("input", "Input tokens"), ("answer", "Output: answer"), ("reasoning", "Output: reasoning")))
              + '<span class="lg"><i style="background:#B45309;height:3px;vertical-align:3px"></i>AI spend per day (right axis)</span>')
    return legend + "".join(parts)


def _type_table(conn, period, compare, s):
    total = stats(conn, *period)["tickets"]
    head = ("<tr><th>Ticket type</th><th class=r>Share of tickets</th><th class=r>Tickets, compared → selected</th>"
            "<th class=r>AI cost per ticket, compared → selected</th><th class=r>Resolution rate, compared → selected</th></tr>")
    body = []
    for t in s["ticket_types"]:
        a, b = stats(conn, *compare, t), stats(conn, *period, t)
        body.append(f"<tr><td>{t}</td><td class=r>{100 * b['tickets'] / total:.0f}%</td>"
                    f"<td class=r>{a['tickets'] / 1e3:,.0f}k → {b['tickets'] / 1e3:,.0f}k</td>"
                    f"<td class=r>${a['spend'] / a['tickets']:.4f} → ${b['spend'] / b['tickets']:.4f}</td>"
                    f"<td class=r>{100 * a['resolved'] / a['tickets']:.1f}% → {100 * b['resolved'] / b['tickets']:.1f}%</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _overview(conn, period, compare, s):
    p, c = stats(conn, *period), stats(conn, *compare)
    note, chart = _change_by_type(conn, period, compare, s)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{_top("Overview", s)}
<div class="head"><h1>Cost and quality overview</h1><div><span class="pill">Period {fmt_range(*period)}</span>
<span class="pill">Compared with {fmt_range(*compare)}</span><span class="pill">Channel All</span></div></div>
<div class="kpis">{_kpis(p, c)}</div>
<div class="row"><div class="card"><div class="ct">Where the AI spend change came from</div><div class="cs">Change by ticket type, {note}</div>{chart}</div>
<div class="card"><div class="ct">Daily tokens by type and daily AI spend</div><div class="cs">All tickets; tokens in billions per day</div>{_daily(conn, period, compare)}</div></div>
<div style="padding:6px 28px"><div class="card"><div class="ct" style="margin-bottom:6px">By ticket type</div>{_type_table(conn, period, compare, s)}</div></div></body></html>"""


def _models(conn, scenario):
    s, pts = spec(scenario), {}
    for m, p in current_prices(scenario).items():
        rows = [(t, s["models"][m]["pass_pct"][t], model_tokens(m, t, scenario)["answer"] + model_tokens(m, t, scenario)["reasoning"])
                for t in s["ticket_types"]]
        pts[m] = {"price": (3 * p["input"] + p["output"]) / 4,
                  "quality": sum(s["share"][t] * q for t, q, _ in rows), "output_tokens": sum(s["share"][t] * o for t, _, o in rows)}
    w, h, left, right, top_, bottom = 1000, 520, 80, 40, 30, 70
    pw, ph = w - left - right, h - top_ - bottom
    xmin, xmax, ymin, ymax = math.log10(0.05), math.log10(20), 50, 95
    X = lambda v: left + pw * (math.log10(v) - xmin) / (xmax - xmin)
    Y = lambda v: top_ + ph - ph * (v - ymin) / (ymax - ymin)
    parts = [f'<svg width="{w}" height="{h}" {FONT} font-size="13">']
    for v in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20):
        parts.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top_}" y2="{top_ + ph}" stroke="#EEF0F3"/>'
                     f'<text x="{X(v):.1f}" y="{top_ + ph + 20}" text-anchor="middle" fill="#6B7280">${v:g}</text>')
    for v in range(50, 96, 5):
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#EEF0F3"/>'
                     f'<text x="{left - 10}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#6B7280">{v}%</text>')
    frontier = sorted(pts.items(), key=lambda kv: kv[1]["price"])
    parts.append('<polyline fill="none" stroke="#9CA3AF" stroke-dasharray="6,4" stroke-width="1.5" points="'
                 + " ".join(f"{X(v['price']):.1f},{Y(v['quality']):.1f}" for _, v in frontier) + '"/>')
    for m, v in pts.items():
        r = 6 + 0.9 * math.sqrt(v["output_tokens"] / 10)
        color = s["models"][m]["color"]
        parts.append(f'<circle cx="{X(v["price"]):.1f}" cy="{Y(v["quality"]):.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.55" stroke="{color}"/>'
                     f'<text x="{X(v["price"]) + r + 6:.1f}" y="{Y(v["quality"]) + 5:.1f}" font-weight="600">{m}</text>')
    parts.append(f'<text x="{left + pw / 2}" y="{h - 18}" text-anchor="middle" fill="#374151">List price, blended per 1M tokens (log scale)</text>'
                 f'<text x="22" y="{top_ + ph / 2}" transform="rotate(-90 22 {top_ + ph / 2})" text-anchor="middle" fill="#374151">Quality: offline eval resolution rate</text></svg>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{_top("Models", s)}
<div class="head"><h1>Model catalog</h1><div><span class="pill">Eval suite: help-desk v4</span></div></div>
<div style="padding:6px 28px"><div class="card"><div class="ct">Quality versus price</div>
<div class="cs">Quality = offline-eval resolution rate weighted by ticket mix. Blended price = (3 × input price + output price) divided by 4. Bubble area = average output tokens per ticket, including reasoning. Dashed line: price–quality frontier.</div>
{"".join(parts)}</div></div></body></html>"""


def _changes(conn, start, end, s):
    rows = conn.execute("SELECT id, date, time_utc, title FROM change_log WHERE date BETWEEN ? AND ? ORDER BY date, time_utc",
                        (start, end)).fetchall()
    body = "".join(f"<tr><td>{i}</td><td>{d}</td><td>{t} UTC</td><td>{title}</td></tr>" for i, d, t, title in rows) or \
        "<tr><td colspan=4>No changes in this period</td></tr>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{_top("Changes", s)}
<div class="head"><h1>Production changes</h1><div><span class="pill">Period {fmt_range(start, end)}</span></div></div>
<div style="padding:6px 28px"><div class="card"><table><tr><th>Change</th><th>Date</th><th>Time</th><th>Title</th></tr>{body}</table></div></div></body></html>"""


def _valid(d, s):
    try:
        return _date(s["first_day"]) <= _date(d) <= _date(s["last_day"])
    except (TypeError, ValueError):
        return False



def page(scenario, tab, start_date=None, end_date=None, compare_start_date=None, compare_end_date=None):
    """Return {"html": ..., "size": (width, height), "description": ...} for one tab, or {"error": ...}."""
    s = spec(scenario)
    given = [d for d in (start_date, end_date, compare_start_date, compare_end_date) if d]
    if any(not _valid(d, s) for d in given):
        return {"error": f"dates must be YYYY-MM-DD within {s['first_day']} to {s['last_day']}"}
    conn = connect(scenario)
    if tab == "Overview":
        period = (start_date or s["period"][0], end_date or s["period"][1])
        if compare_start_date or compare_end_date:
            compare = (compare_start_date or compare_end_date, compare_end_date or compare_start_date)
        elif start_date or end_date:
            n = (_date(period[1]) - _date(period[0])).days + 1
            compare = ((_date(period[0]) - dt.timedelta(days=n)).isoformat(), (_date(period[0]) - dt.timedelta(days=1)).isoformat())
            if not _valid(compare[0], s):
                compare = (s["first_day"], compare[1])
        else:
            compare = tuple(s["compare"])
        if period[0] > period[1] or compare[0] > compare[1] or not _valid(compare[1], s):
            return {"error": "each period needs start_date <= end_date within the data range"}
        html, size = _overview(conn, period, compare, s), (1440, 860)
        desc = f"Overview, period {period[0]}..{period[1]} compared with {compare[0]}..{compare[1]}"
    elif tab == "Models":
        html, size, desc = _models(conn, scenario), (1100, 700), "Models"
    elif tab == "Changes":
        start, end = start_date or s["first_day"], end_date or s["last_day"]
        html, size, desc = _changes(conn, start, end, s), (1100, 520), f"Changes, {start}..{end}"
    else:
        return {"error": f"unknown tab {tab!r}; tabs are Overview, Models, Changes"}
    conn.close()
    return {"html": html, "size": size, "description": desc}


def render_path(scenario, description):
    """Where the render of one tab and date range is cached."""
    return WORLD / scenario / "renders" / f"{hashlib.sha256(f'{scenario}|{description}'.encode()).hexdigest()[:16]}.png"


def render(scenario, tab, **dates):
    """Render one tab to a PNG, like a BI tool's render API. Return {"path", "description"} or {"error"}."""
    out = page(scenario, tab, **dates)
    if "error" in out:
        return out
    png = render_path(scenario, out["description"])
    with render_lock:
        if not png.exists():
            png.parent.mkdir(parents=True, exist_ok=True)
            html = png.with_suffix(".html")
            html.write_text(out["html"])
            screenshot(html, png, out["size"])
    return {"path": str(png), "description": out["description"]}
