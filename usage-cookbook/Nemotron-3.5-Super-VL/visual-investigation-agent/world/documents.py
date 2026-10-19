# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The team's documents: scanned price schedules, the two-page offline-eval report, and the router config.

Scanned pages and report pages are images, generated from HTML with a headless browser, like the scans and
report exports a team keeps in its document store.
"""

import shutil
from pathlib import Path

from .browser import screenshot, to_jpeg
from .data import WORLD
from .spec import spec

ASSETS = Path(__file__).resolve().parent / "assets"
AUGUST_SCAN = ASSETS / "price-schedule-2026-08.jpg"
ROUTER_DOC = {"id": "router-config", "title": "router/helpdesk.yaml: gateway router config for the help-desk agent",
              "format": "text", "pages": 1}
EVAL_REPORT = {"id": "eval-report-2026-08", "title": "Help-desk offline eval v4: results by model and ticket type "
               "(Eval Team report export, Aug 28, 2026)", "format": "report export (image)", "pages": 2}
NEW_SCHEDULE = {"id": "price-schedule-2026-09", "title": "Larkfield Cloud model price schedule, effective Sep 1, 2026 "
                "(scanned copy received by Finance)", "format": "scanned image", "pages": 1}


def _schedule_doc(schedule):
    return {"id": schedule["id"], "title": schedule["title"], "format": "scanned image", "pages": 1}


def list_documents(scenario):
    s = spec(scenario)
    first, *later = s["price_schedules"]
    docs = [_schedule_doc(first), ROUTER_DOC] + [_schedule_doc(x) for x in later]
    if not s["eval"]["in_db"]:
        docs.append({"id": s["eval"]["id"], "title": s["eval"]["title"], "format": "report export (image)", "pages": 2})
    return docs


def _price_scan(prices, effective, received, supersedes, png, provider="Larkfield Cloud", scan=None):
    page = png.with_suffix(".html")
    page.write_text(price_scan_html(prices, effective, received, supersedes, provider, scan))
    screenshot(page, png, (1060, 560))
    jpg = to_jpeg(png, quality=(scan or {}).get("quality", 70))
    png.unlink()
    return jpg


def price_scan_html(prices, effective, received, supersedes, provider="Larkfield Cloud", scan=None):
    scan = scan or {"rotate": 0.7, "blur": 0.25, "contrast": 0.92}
    money = lambda v: f"${v:.3f}" if round(v * 1000) % 10 else f"${v:.2f}"
    rows = "".join(f"<tr><td>{m}</td><td class=r>{money(p['input'])}</td><td class=r>{money(p['output'])}</td></tr>" for m, p in prices.items())
    notes = ["Reasoning (thinking) tokens are billed as output tokens, whether or not they are returned in the response.",
             "Prices are per request and per token; there is no minimum charge and no monthly commitment.",
             f"This schedule supersedes the schedule effective {supersedes}."]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#D8D4CC;font-family:Georgia,'Times New Roman',serif;color:#1F2328}}
.doc{{width:900px;margin:24px auto;background:#FBF9F4;padding:48px 64px;transform:rotate({scan['rotate']}deg);box-shadow:0 2px 10px rgba(0,0,0,.25);filter:contrast({scan['contrast']}) blur({scan['blur']}px)}}
h1{{font-family:Helvetica,Arial,sans-serif;font-size:22px;margin:8px 0}}.muted{{color:#4B5563;font-size:13px}}
table{{width:100%;border-collapse:collapse;margin:18px 0;font-family:Helvetica,Arial,sans-serif;font-size:14.5px}}
th,td{{border-bottom:1px solid #C9C5BC;padding:8px}}th{{text-align:left;background:#EFEBE3}}.r{{text-align:right}}
ol{{font-size:13px;line-height:1.5;padding-left:20px}}.stamp{{float:right;border:2px solid #7F1D1D;color:#7F1D1D;padding:4px 10px;font-family:Helvetica,Arial;font-size:12px;transform:rotate(4deg)}}</style></head>
<body><div class="doc"><span class="stamp">RECEIVED · FINANCE · {received}</span>
<div class="muted">{provider} · Model price schedule · Effective {effective} · Illustrative</div>
<h1>Hosted model token prices (USD per 1 million tokens)</h1>
<table><tr><th>Model</th><th class=r>Input</th><th class=r>Output</th></tr>{rows}</table>
<b style="font-family:Helvetica,Arial,sans-serif;font-size:14px">Notes</b><ol>{"".join(f"<li>{n}</li>" for n in notes)}</ol></div></body></html>"""
    return html


EVAL_CSS = """
body{margin:0;background:#E5E7EB;font-family:Helvetica,Arial,sans-serif;color:#111827}
.page{width:1000px;margin:20px auto;background:#fff;padding:40px 56px;box-shadow:0 1px 6px rgba(0,0,0,.2)}
.muted{color:#4B5563;font-size:13px}h1{font-size:21px;margin:6px 0 4px}h2{font-size:16px;margin:22px 0 8px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid #D1D5DB;padding:7px 8px}
th{text-align:left;background:#F3F4F6;font-weight:600}.r{text-align:right}ol{font-size:13px;line-height:1.5;padding-left:20px}
.foot{margin-top:18px;font-size:12px;color:#6B7280;display:flex;justify-content:space-between}
"""


def _header(s):
    return (f'<div class="muted">{s["eval_team"]} · Help-desk offline eval v4 · Published {s["eval"]["published"]} · Illustrative</div>'
            "<h1>Offline eval results by model and ticket type</h1>")


def _resolution_section(s, number, pointer):
    head = "<tr><th>Model</th>" + "".join(f"<th class=r>{t}</th>" for t in s["ticket_types"]) + "</tr>"
    rows = "".join(f"<tr><td>{m}</td>" + "".join(f"<td class=r>{s['models'][m]['pass_pct'][t]:.1f}%</td>" for t in s["ticket_types"]) + "</tr>"
                   for m in s["models"])
    n = "".join(f"<td class=r>{s['eval_tickets'][t]:,}</td>" for t in s["ticket_types"])
    last = (f"Token counts per ticket are on page {pointer}. " if pointer else "") + "Prices are not included; use the current price schedule."
    return f"""<h2>{number}. Resolution rate</h2><div class="muted" style="margin-bottom:8px">Share of replayed tickets resolved without human follow-up, graded by the resolution rubric.</div>
<table>{head}{rows}<tr><td class="muted">Eval tickets per type</td>{n}</tr></table>
<h2>Method</h2><ol><li>Historical tickets from {s['eval'].get('replayed', 'May–July 2026')} replayed against each model with a simulated customer.</li>
<li>Every model used the same help-desk prompt and tools; reasoning models used their default reasoning setting.</li>
<li>{last}</li></ol>"""


def _token_section(s, number):
    head = ("<tr><th>Model</th><th>Ticket type</th><th class=r>Input tokens</th><th class=r>Output tokens: answer</th>"
            "<th class=r>Output tokens: reasoning</th><th class=r>Requests</th></tr>")
    rows = []
    for m in s["models"]:
        for t in s["ticket_types"]:
            inp, ans, rsn = s["models"][m]["tokens"][t]
            rows.append(f"<tr><td>{m}</td><td>{t}</td><td class=r>{inp:,}</td><td class=r>{ans:,}</td>"
                        f"<td class=r>{rsn:,}</td><td class=r>{s['requests_per_ticket'][t]}</td></tr>")
    return f"""<h2>{number}. Average tokens and requests per ticket</h2><div class="muted" style="margin-bottom:8px">Averages over all replayed tickets of each type. Reasoning tokens are generated but not shown to the customer.</div>
<table>{head}{''.join(rows)}</table>"""


def _page(s, body, number):
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{EVAL_CSS}</style></head><body><div class="page">{_header(s)}
{body}
<div class="foot"><span>Help-desk offline eval v4</span><span>Page {number} of 2</span></div></div></body></html>"""


def eval_pages(scenario="model-switch-visual"):
    """The two report pages as HTML, in page order."""
    s = spec(scenario)
    tokens_page = s["eval"].get("tokens_page", 2)
    pointer = tokens_page if s["eval"].get("pointer_note", True) else None
    if tokens_page == 2:
        return [_page(s, _resolution_section(s, 1, pointer), 1), _page(s, _token_section(s, 2), 2)]
    return [_page(s, _token_section(s, 1), 1), _page(s, _resolution_section(s, 2, pointer), 2)]


def page1(scenario="model-switch-visual"):
    return eval_pages(scenario)[0]


def page2(scenario="model-switch-visual"):
    return eval_pages(scenario)[1]


def build_documents(scenario, images=True):
    """Write the scenario's documents. Without images, only the files that need no browser are written."""
    s = spec(scenario)
    out = WORLD / scenario
    out.mkdir(parents=True, exist_ok=True)
    (out / "router-helpdesk.yaml").write_text(s["router"])
    for schedule in s["price_schedules"]:
        if schedule.get("asset"):
            shutil.copy(ASSETS / schedule["asset"], out / schedule["file"])
    if not images:
        return
    if not s["eval"]["in_db"]:
        for i, (html, height) in enumerate(zip(eval_pages(scenario), (720, 900) if s["eval"].get("tokens_page", 2) == 2 else (900, 720)), start=1):
            page = out / f"{s['eval']['id']}-p{i}.html"
            page.write_text(html)
            screenshot(page, page.with_suffix(".png"), (1060, height))
    for schedule in s["price_schedules"]:
        if not schedule.get("asset"):
            _price_scan(schedule["prices"], schedule["effective_text"], schedule["received"], schedule["supersedes"],
                        out / Path(schedule["file"]).with_suffix(".png"), s["provider"], s.get("scan"))


def images_ready(scenario):
    s = spec(scenario)
    out = WORLD / scenario
    files = [x["file"] for x in s["price_schedules"]]
    if not s["eval"]["in_db"]:
        files += [f"{s['eval']['id']}-p{i}.png" for i in (1, 2)]
    return all((out / f).exists() for f in files)


def document(scenario, doc_id, page=1):
    s = spec(scenario)
    docs = {d["id"]: d for d in list_documents(scenario)}
    if doc_id not in docs:
        return {"error": f"unknown doc_id {doc_id!r}; call list_documents"}
    if not 1 <= page <= docs[doc_id]["pages"]:
        return {"error": f"{doc_id} has {docs[doc_id]['pages']} page(s)"}
    if doc_id == s["eval"]["id"]:
        return {"doc_id": doc_id, "format": "image", "path": str(WORLD / scenario / f"{doc_id}-p{page}.png"), "page": page}
    if doc_id == ROUTER_DOC["id"]:
        return {"doc_id": doc_id, "format": "text", "content": (WORLD / scenario / "router-helpdesk.yaml").read_text()}
    schedule = next(x for x in s["price_schedules"] if x["id"] == doc_id)
    return {"doc_id": doc_id, "format": "image", "path": str(WORLD / scenario / schedule["file"])}
