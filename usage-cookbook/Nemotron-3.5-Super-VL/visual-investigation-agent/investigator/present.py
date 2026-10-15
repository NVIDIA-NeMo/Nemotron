# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a run, recorded or live, into the notebook's outputs: timeline, step cards, evidence, chart, proposal.

Everything shown comes from the run's own record: the images the agent requested, its visible notes and stated
reasons, its tool calls and their results. Checks against the ground truth come from evals/, which the agent
never sees. Nothing here changes what the agent saw or said.
"""

import base64
import html
import inspect
import io
import json
import math
import re

from pathlib import Path

import world
from evals import checks

from . import contract, policy, tools
from . import highlights
from .config import DEFAULT, VERSIONS, config_hash
from .runs import load_recording

FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
CARD = (f"font-family:{FONT};font-size:14px;line-height:1.5;color:#1F2328;background:#FFFFFF;border:1px solid #D0D7DE;"
        "border-radius:10px;padding:14px 18px;margin:8px 0;max-width:1080px")
MUTED = "color:#57606A;font-size:12.5px"
PRE = ("background:#F6F8FA;border:1px solid #EAEEF2;border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.4;"
       "white-space:pre-wrap;overflow-x:auto;margin:6px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace")
TABLE = "border-collapse:collapse;width:100%;font-size:13px;margin:6px 0"
TH = "text-align:left;border-bottom:1px solid #D0D7DE;padding:5px 8px;color:#57606A;font-weight:600"
TD = "border-bottom:1px solid #EAEEF2;padding:5px 8px;vertical-align:top;text-align:left"
STAGE_COLORS = {"Ask": ("#F6F8FA", "#24292F"), "Look": ("#DDF4FF", "#0550AE"), "Drill down": ("#FBEFFF", "#6639BA"),
                "Verify": ("#DAFBE1", "#116329"), "Report": ("#FFF8C5", "#7D4E00"), "Propose": ("#FFEBE9", "#A40E26"),
                "Harness": ("#FFF1E5", "#953800")}
LABEL_COLORS = {"Observed": ("#DDF4FF", "#0550AE"), "Calculated": ("#DAFBE1", "#116329"), "Hypothesis": ("#FFF8C5", "#7D4E00")}
MONTHS = {"model-switch-visual": "September: the model switch (the demo run)", "model-switch": "September: the model switch",
          "volume-growth": "A different September: ticket volume grew",
          "price-increase": "A different September, held out: a price increase"}
MODEL_ROLES = {
    "Dunlin-3-8B-Instruct": "smallest model",
    "Vesk 4 Lite": "small model",
    "Osric 2 Thinking": "new model, in use now",
    "Vesk 4": "previous model",
    "Vesk 4 Pro": "largest model",
}
OK, NO = "<span style='color:#1A7F37;font-weight:600'>✓</span>", "<span style='color:#CF222E;font-weight:600'>✗</span>"


class Card:
    """HTML that renders itself in a notebook, with its title as the plain-text fallback."""

    def __init__(self, body, title=""):
        self.body, self.title = body, title

    def _repr_html_(self):
        return self.body

    def __repr__(self):
        return f"[{self.title}]" if self.title else "[card]"


# Building blocks --------------------------------------------------------------------------------------

def _e(text):
    return html.escape(str(text if text is not None else ""))


def _note(text):
    """An agent note as HTML: escaped, with its bold markup kept and trailing rules dropped."""
    text = re.sub(r"\s*-{3,}\s*$", "", str(text or "").strip())
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _e(text)).replace("\n", "<br>")


def _chip(text, colors):
    bg, fg = colors
    return (f"<span style='background:{bg};color:{fg};border-radius:999px;padding:1px 9px;font-size:12px;"
            f"font-weight:600;white-space:nowrap'>{_e(text)}</span>")


def _mode_badge(run):
    """Every card that shows a run says whether it is the recording or your live run."""
    if run is None or not hasattr(run, "recorded"):
        return ""
    live = not run.recorded
    return (f"<span style='font-size:12px;font-weight:600;padding:2px 9px;border-radius:10px;white-space:nowrap;"
            f"background:{'#DAFBE1' if live else '#EAEEF2'};color:{'#1A7F37' if live else '#57606A'}'>"
            f"{'Live run' if live else 'Replay · recorded run'}</span>")


def _card(title, body, run=None, subtitle=""):
    sub = f"<div style='{MUTED};margin-top:2px'>{subtitle}</div>" if subtitle else ""
    # Dollar amounts are text, not math: the ignore classes stop notebook viewers from typesetting "$...$".
    return Card(f"<div class='tex2jax_ignore mathjax_ignore' style=\"{CARD}\">"
                "<div style='display:flex;justify-content:space-between;gap:12px;align-items:baseline'>"
                f"<div style='font-weight:700;font-size:15.5px'>{_e(title)}</div>{_mode_badge(run)}</div>{sub}"
                f"<div style='margin-top:8px'>{body}</div></div>", title)


_IMAGES = {}


def _img(path, width, alt):
    """An image, embedded so the saved notebook shows it anywhere. It is stored at twice its display width (or its
    full size, if smaller), so it stays sharp on high-resolution screens."""
    from PIL import Image

    key = (str(path), width)
    if key not in _IMAGES:
        image = Image.open(path)
        stored = min(image.width, 2 * width)
        if image.width > stored:
            image = image.resize((stored, round(image.height * stored / image.width)), Image.LANCZOS)
        buffer, fmt = io.BytesIO(), "JPEG" if str(path).lower().endswith(".jpg") else "PNG"
        image.convert("RGB").save(buffer, fmt, **({"quality": 88} if fmt == "JPEG" else {"optimize": True}))
        _IMAGES[key] = f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"
    return (f"<img src='{_IMAGES[key]}' alt='{_e(alt)}' style='max-width:100%;width:{width}px;border:1px solid #D0D7DE;"
            "border-radius:6px'>")


def _img_bytes(data, width, alt):
    key = (hash(data), width)
    if key not in _IMAGES:
        _IMAGES[key] = f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return (f"<img src='{_IMAGES[key]}' alt='{_e(alt)}' style='max-width:100%;width:{width}px;border:1px solid #D0D7DE;"
            "border-radius:6px'>")


def _highlighted_figure(run, image_id, width, caption, ids=None, crop_to_boxes=False, crop_margin=28,
                        show_annotation_label=False):
    boxes = highlights.load(run).get(image_id, [])
    if ids:
        boxes = [box for box in boxes if box["id"] in ids]
    image = (_img_bytes(highlights.png_bytes(run, image_id, ids, crop_to_boxes, crop_margin), width, caption) if boxes
             else _img(run.image(image_id), width, caption))
    legend = (f"<div style='{MUTED};color:#953800'>{highlights.LABEL}</div>"
              if boxes and show_annotation_label else "")
    return (f"<div style='display:inline-block;vertical-align:top;margin:0 12px 8px 0;max-width:{width}px'>"
            f"{image}<div style='{MUTED}'>{_e(caption)} · illustrative, synthetic data</div>{legend}</div>")


def _table(head, rows, widths=None):
    widths = widths or [""] * len(head)
    return (f"<table style='{TABLE}'><tr>" + "".join(f"<th style='{TH};{f'width:{w}' if w else ''}'>{h}</th>"
                                                     for h, w in zip(head, widths)) + "</tr>"
            + "".join("<tr>" + "".join(f"<td style='{TD}'>{c}</td>" for c in row) + "</tr>" for row in rows) + "</table>")


def _details(summary, body, open_=False):
    return f"<details{' open' if open_ else ''}><summary style='cursor:pointer;{MUTED}'>{_e(summary)}</summary>{body}</details>"


def _code(text):
    return f"<pre style=\"{PRE}\">{_e(text)}</pre>"


def _clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _money(v):
    return f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}"


def _price(v):
    return f"${v:.2f}" if round(v, 2) == v else f"${v:g}"


def _config(run):
    """The configuration a run was made with: its frozen version, by name or configuration hash, else the default."""
    manifest = run.manifest or {}
    if manifest.get("agent_version") in VERSIONS:
        return VERSIONS[manifest["agent_version"]]
    recorded = manifest.get("config_hash") or run.result.get("config_hash")
    return next((c for c in VERSIONS.values() if config_hash(c) == recorded), DEFAULT)


def _sentences(text, n=2):
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    return " ".join(re.split(r"(?<=[.!?])\s+", clean)[:n])


def _excerpt(text, cues=(), n=2):
    """Sentences containing any cue, in order; otherwise the first n sentences. Verbatim."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    chosen = [s for s in sentences if any(cue.lower() in s.lower() for cue in cues)][:n] or sentences[:n]
    joined = " ".join(chosen)
    return joined[:520] + ("…" if len(joined) > 520 else "")


def _kpi_change(scenario, label):
    return next((change for name, _, change, _ in checks.kpi_cards(scenario) if name == label), "")


def _eval_layout(scenario):
    """(page with resolution rates, page with token profiles, whether page 1 points to the token page)."""
    e = world.spec(scenario)["eval"]
    tokens = e.get("tokens_page", 2)
    return 3 - tokens, tokens, e.get("pointer_note", True)


def _change_entries(scenario):
    return {c[0]: (c[1], c[3]) for c in world.spec(scenario)["changes"]}


def _attribution_block(run):
    """The report's attribution field: each cited change-log entry with its date and title, the documents, and why."""
    entries = run.report.get("attribution")
    entries = [entries] if isinstance(entries, dict) else [e for e in entries or [] if isinstance(e, dict)]
    if not entries:
        return ""
    changes = _change_entries(run.scenario)
    parts = []
    for e in entries:
        head = f"<div><b>{_e(e['driver'])}</b> (from {_e(e.get('started'))})</div>" if e.get("driver") else ""
        cited = [f"<li><code>{_e(c)}</code> · {_e(changes[c][0])} · {_e(changes[c][1])}</li>" if c in changes else f"<li><code>{_e(c)}</code></li>"
                 for c in e.get("change_ids") or []]
        docs = ", ".join(f"<code>{_e(d)}</code>" for d in e.get("documents") or [])
        parts.append(head + (f"<ul style='margin:4px 0 4px 18px;padding:0'>{''.join(cited)}</ul>" if cited else
                             "<div>No change-log entry cited.</div>")
                     + (f"<div style='{MUTED}'>Documents it relied on: {docs}</div>" if docs else "")
                     + (f"<div style='margin-top:4px'>Its explanation, in Super VL's exact words:</div>"
                        f"{_said(e.get('explanation'))}" if e.get("explanation") else ""))
    return ("<div style='margin-top:10px;padding:9px 11px;border:1px solid #D0D7DE;border-radius:6px'>"
            "<b>What it says caused the change</b>" + "".join(parts) + "</div>")


def show_setup(live, scenario, run=None):
    """Which mode the notebook is in, and what every card below shows."""
    from . import endpoint

    if run is not None:
        summary = (f"{len(run.steps)} steps and {len(run.calls())} tool calls over "
                   f"{run.result.get('seconds', 0) / 60:.1f} minutes")
        if not live:
            return _card("Replay mode",
                         f"Loaded a recording captured during a live Super VL run against the synthetic scenario: {summary}. "
                         "Replay uses its saved model replies and tool results, so no endpoint or API key is required. Every "
                         "card below shows this recording. To run Super VL yourself, see <i>Run this notebook</i> above.", run)
        return _card("Live mode",
                     f"Super VL finished on your endpoint: {summary}, saved to <code>runs/{_e(run.folder.name)}/</code>. "
                     "Every card below shows this run.", run)
    if not live:
        return _card("Replay ready",
                     "Loaded a recording captured during a live Super VL run against the synthetic scenario. "
                     "Replay uses its saved model replies and tool results, so no endpoint or API key is required.")
    failed = [(name, detail) for name, passed, detail in endpoint.check() if not passed]
    if failed:
        body = "<b>Live endpoint needs attention.</b><ul>" + "".join(
            f"<li>{_e(name)}: {_e(detail)}</li>" for name, detail in failed) + "</ul>"
    else:
        body = "<b>Live endpoint validated.</b> Text, thinking, image, and tool-call checks passed."
    return _card("Live endpoint validation", body)


# Section 5: the dashboard -------------------------------------------------------------------------------

def _overview_image(scenario, live):
    if live:
        return Path(world.render(scenario, "Overview")["path"])
    run = load_recording(scenario)
    return run.image(run.request[1][0])


def _sequence(steps):
    """Numbered steps in the order they happened."""
    rows = []
    for number, (label, body) in enumerate(steps, start=1):
        rows.append(
            "<div style='display:flex;gap:12px;margin-top:14px'>"
            "<div style='flex:0 0 24px;height:24px;border-radius:12px;background:#76B900;color:#FFFFFF;font-weight:700;"
            f"text-align:center;line-height:24px;font-size:13px'>{number}</div>"
            f"<div style='flex:1;min-width:0'><div style='font-weight:700;margin-bottom:5px'>{label}</div>{body}</div></div>")
    return "".join(rows)


def _said(text):
    """Super VL's own words, verbatim from the run."""
    return (f"<div style='padding:9px 11px;background:#F6F8FA;border-left:3px solid #76B900;border-radius:4px'>"
            f"“{_note(text)}”</div>")


def _calls_list(calls):
    """Each tool call, in the order it was made, with the reason Super VL gave. Listing tools take no reason."""
    items = []
    for c in calls:
        reason = (f", because “{_e(c.why)}”" if c.why else
                  f": lists the {c.name.removeprefix('list_')} it can open" if c.name.startswith("list_") else "")
        items.append(f"<li><code>{_e(c.name)}</code>{reason}</li>")
    return "<ol style='margin:4px 0 0 18px;padding:0'>" + "".join(items) + "</ol>"


def _typed_in_python(run, values):
    """True if every value appears as a number in the Python Super VL wrote."""
    code = " ".join(c.arguments.get("code", "") for _, c in run.calls("run_python") if c.arguments)
    numbers = [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", code.replace("_", ""))]
    return bool(numbers) and all(any(abs(n - v) < 1e-9 for n in numbers) for v in values)


def show_visual_triage(run):
    """Moment 1, in order: what Super VL has, what it sees on the screenshot, what it writes, and what it calls next."""
    if not run.request[1] or not run.steps:
        return _card("1 · Read the dashboard", "<i>No screenshot was attached in this run.</i>", run)
    image_id, first = run.request[1][0], run.steps[0]
    dashboard = _highlighted_figure(run, image_id, 900, "The screenshot pasted with the request",
                                    ids=["spend", "tickets", "cost-per-ticket", "average-price"],
                                    show_annotation_label=True)
    kpis = (f"The KPI cards read: AI spend {_kpi_change(run.scenario, 'AI spend')}, tickets "
            f"{_kpi_change(run.scenario, 'Tickets handled by AI')}, cost per ticket "
            f"{_kpi_change(run.scenario, 'AI cost per ticket')}, and average price per 1M tokens "
            f"{_kpi_change(run.scenario, 'Average price per 1M tokens')}.")
    body = _sequence([
        ("What it has",
         "The request, this screenshot, and the team's playbook, which asks it to compare like-for-like periods, break the "
         "change into its drivers, check the change log, and check quality alongside cost. Its instructions also ask it to "
         "say what the screenshot tells it before its first tool call, so you can check how it read the image."),
        ("What it sees", dashboard + f"<div style='margin-top:4px'>{_e(kpis)}</div>"),
        ("What it says, before it touches any data", _said(_excerpt(first.note, n=4))),
        ("What it does next", f"In the same step, it makes {len(first.calls)} tool calls:" + _calls_list(first.calls)),
    ])
    return _card("1 · Read the dashboard", body, run)


def show_missing_evidence(run):
    """Moment 2, in order: where the playbook points, what Super VL opens, what it says, and the numbers it takes."""
    prices_seen = _document_views(run, "price-schedule")
    reports = _document_views(run, "eval-report")
    quality_page, token_page, pointer = _eval_layout(run.scenario)
    by_page = {}
    for step, call in reports:
        by_page.setdefault(int(call.arguments.get("page") or 1), (step, call))
    if not prices_seen or quality_page not in by_page or token_page not in by_page or not run.proposals:
        return _card("2 · Find the missing evidence",
                     "<i>This run did not open the price list and both report pages, or filed no proposal.</i>", run)
    price = prices_seen[0][1]
    quality_call, token_call = by_page[quality_page][1], by_page[token_page][1]
    token_step = by_page[token_page][0]
    reason = (token_step.note if re.search(r"page|token", token_step.note, re.I) else "") or token_call.why
    prices = world.current_prices(run.scenario)
    selected = run.proposals[0]["routes"]
    shown_models = ("Vesk 4 Lite", "Osric 2 Thinking", "Vesk 4")
    price_rows = [(f"{_e(m)}<br><span style='{MUTED}'>{_e(MODEL_ROLES[m])}</span>", _price(prices[m]['input']),
                   _price(prices[m]['output'])) for m in shown_models]
    quality_rows, token_rows, values = [], [], [prices[m][k] for m in shown_models for k in ("input", "output")]
    for i, ticket_type in enumerate(world.TICKET_TYPES):
        model, profile = selected[ticket_type], world.model_tokens(selected[ticket_type], ticket_type)
        named = f"{_e(model)}<br><span style='{MUTED}'>{_e(MODEL_ROLES[model])}</span>"
        resolved, floor = world.PASS_PCT[model][i], policy.quality_floor(ticket_type)
        quality_rows.append((_e(ticket_type), named, f"{resolved:.1f}%", f"{floor:.1f}%",
                             "Pass" if resolved >= floor else "Fail"))
        token_rows.append((_e(ticket_type), named, f"{profile['input']:,}", f"{profile['answer']:,}",
                           f"{profile['reasoning']:,}"))
        values += [world.PASS_PCT[model][i], profile["input"], profile["answer"], profile["reasoning"]]
    current, hardest = "Osric 2 Thinking", "Technical troubleshooting"
    profile = world.model_tokens(current, hardest)
    token_rows.append((_e(hardest), f"{_e(current)}<br><span style='{MUTED}'>{_e(MODEL_ROLES[current])}</span>",
                       f"{profile['input']:,}", f"{profile['answer']:,}", f"{profile['reasoning']:,}"))
    values += [profile["input"], profile["answer"], profile["reasoning"]]
    page = lambda call, caption, ids: _highlighted_figure(run, call.image_id, 820, caption, ids=ids)
    why = lambda call: f"<div style='margin:2px 0 10px'><b>Why it opened this:</b> “{_e(call.why)}”</div>"
    opened = (
        page(price, "The scanned price list", ["candidate-price-rows"]) + why(price)
        + page(quality_call, f"Eval report, page {quality_page}", ["vesk-lite-quality", "vesk-4-quality"]) + why(quality_call)
        + (f"<div>At the bottom of page {quality_page}, a note says token counts per ticket are on page "
           f"{token_page}.</div>" if pointer else "")
        + f"<div style='{MUTED};color:#953800;margin-top:4px'>{highlights.LABEL}</div>")
    column = lambda title, table: f"<div style='flex:1 1 470px'><b>{title}</b>{table}</div>"
    taken = (
        ("It opens page {page}, then writes Python code using each highlighted value to calculate cost per ticket and "
         "apply the quality bar:" if _typed_in_python(run, values)
         else "It opens page {page}. These highlighted values are inputs to its cost and quality calculation:").format(
             page=token_page)
        + page(token_call, f"Eval report, page {token_page}",
               ["vesk-lite-account-tokens", "vesk-lite-billing-tokens", "osric-tech-tokens", "vesk-4-tech-tokens",
                "vesk-4-integrations-tokens"])
        + f"<div style='{MUTED};margin:6px 0'>The three tables below pull out the exact price, quality, and token values "
          "used in the calculation. The middle table compares each proposed model's offline resolution with the required "
          "quality bar and marks the result. The tables show Vesk 4 Lite and Vesk 4, the models in the final proposal; the "
          "price table and the last token row add Osric 2 Thinking, the model in use now, for comparison. The page 1 and "
          "page 2 images above show the full report tables.</div>"
        + "<div style='display:flex;gap:12px;flex-wrap:wrap'>"
        + column("Prices per 1M tokens, from the scan", _table(["Model", "Input", "Output"], price_rows))
        + column(f"Offline quality check, page {quality_page}",
                 _table(["Ticket type", "Model", "Resolves offline", "Required", "Result"], quality_rows))
        + column(f"Tokens per ticket, page {token_page}",
                 _table(["Ticket type", "Model", "Input", "Answer", "Reasoning"], token_rows))
        + "</div>")
    has = ("What it has",
           "The playbook says prices and offline-eval results aren't in the database: the current price schedule and the "
           "offline eval report are documents. It also says to project each model's cost from the report's tokens per ticket "
           "and the price schedule. The playbook names the documents; which pages to open and which numbers to pull is up "
           "to Super VL.")
    first_doc = min(prices_seen[0][0].number, min(s.number for s, _ in reports))
    glance = next(((s, c) for s, c in run.calls("render_dashboard")
                   if c.image_id and s.number < first_doc and c.arguments.get("tab") == "Models"), None)
    doc_step = next(s for s in run.steps if s.number == first_doc)
    says_next = ("What it says next", _said(_excerpt(reason, n=4)))
    if glance and doc_step.note:
        chart = _img(run.image(glance[1].image_id), 620, "The dashboard's Models tab")
        steps = [has,
                 ("What it looks at", "The dashboard's Models tab, a chart of price against quality for every model."
                  f"{chart}<div><b>Why it opened this:</b> “{_e(glance[1].why)}”</div>"),
                 ("What it says", _said(_excerpt(doc_step.note, n=2))),
                 ("What it opens", opened), says_next, ("What it does next", taken)]
    else:
        steps = [has, ("What it opens", opened), says_next, ("What it does next", taken)]
    return _card("2 · Find the missing evidence", _sequence(steps), run)


def _disclosures(run):
    """Hand-written notes, added after the run, on statements in the recorded report that the data contradicts."""
    path = run.folder / "disclosures.json" if run.folder else None
    return json.loads(path.read_text()) if path and path.exists() else []


def _slip_notes(run):
    """The recording's two wording slips, each with its correct value; "" for any other run."""
    if len(_disclosures(run)) != 2:
        return ""
    return (
        "<b>Two slips in the report's wording; the correctness of the proposal is unaffected.</b>"
        "<ul style='margin:5px 0 5px 20px;padding:0'>"
        "<li style='margin-bottom:5px'><b>Output price.</b> The report says Osric “generates far more output units per "
        "ticket … at a higher output price” (“units” means tokens). The scanned price list shows Osric's output price is "
        "lower: $1.90 per 1M tokens, against $7.60 for Vesk 4. The bill rose because Osric uses far more tokens per "
        "ticket (32,300 reasoning tokens on a technical ticket, against Vesk 4's 300), not because of its price.</li>"
        "<li><b>Starting cost.</b> The recommendation says the routing cuts cost per ticket “from $0.0542 to $0.0290, "
        "saving about $102.6k per month.” $0.0542 is the Sep 1–21 average, which includes the week before the switch. "
        "The saving starts from $0.0616, the cost per ticket after the rollout finished (Sep 10–21): ($0.0616 − $0.0290) "
        "× 104,969 tickets a day × 30 days ≈ $103k a month. Starting from $0.0542 would give about $79k.</li></ul>"
        "Super VL computed the proposal in code from the correct prices and the $0.0616 baseline before writing the "
        "report, and our recheck against the source data confirms its routes and savings. We show the report unedited, "
        "slips included."
    )


def _disclosure_banner(run):
    """A note beside a recorded report whose raw wording is corrected after the run."""
    disclosed = _disclosures(run)
    if not disclosed:
        return ""
    body = _slip_notes(run) or (f"<b>The report needs corrections.</b> We found {_count_word(len(disclosed))} statements "
                                "that disagree with the data; the corrections are below.")
    return ("<div style='margin:8px 0;padding:9px 11px;background:#F6F8FA;border:1px solid #D0D7DE;border-radius:6px'>"
            f"<div style='{MUTED};margin-bottom:4px'>Our check after the run; Super VL did not see it.</div>{body}</div>")


MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}


def _stated_windows(period, year):
    """The two date ranges a driver's period names ("Aug 1-21 vs Sep 10-21", or ISO dates), or None."""
    text = str(period or "")
    iso = re.findall(r"(\d{4}-\d{2}-\d{2})\s*(?:to|–|-|\.\.)\s*(\d{4}-\d{2}-\d{2})", text)
    if len(iso) >= 2:
        return iso[0], iso[1]
    named = [(m.lower()[:3], int(lo), int(hi)) for m, lo, hi in
             re.findall(r"\b([A-Z][a-z]{2,8})\.?\s+(\d{1,2})\s*[-–]\s*(\d{1,2})\b", text)]
    named = [n for n in named if n[0] in MONTHS]
    if len(named) >= 2:
        return tuple((f"{year}-{MONTHS[m]:02d}-{lo:02d}", f"{year}-{MONTHS[m]:02d}-{hi:02d}") for m, lo, hi in named[:2])
    return None


def _measure(stats, role):
    tokens = stats["input"] + stats["answer"] + stats["reasoning"]
    return {"volume": stats["tickets"], "usage": tokens / stats["tickets"], "price": stats["spend"] * 1e6 / tokens,
            "cost": stats["spend"] / stats["tickets"], "spend": stats["spend"],
            "resolution": 100 * stats["resolved"] / stats["tickets"]}[role]


def _figure_checks(run):
    """The report's drivers, recomputed from the data for the periods each one states, and the proposal's savings."""
    names = {"volume": "Tickets", "usage": "Tokens per ticket", "price": "Price per 1M tokens", "cost": "Cost per ticket",
             "spend": "AI spend", "resolution": "Resolution rate"}
    conn = world.connect(run.scenario)
    year = world.spec(run.scenario)["first_day"][:4]
    drivers = list(run.report.get("drivers") or [])
    classified = {id(d): role for role, d in _classify(drivers).items()}
    fmt = {"spend": _money, "resolution": lambda v: f"{v:.1f}%", "volume": lambda v: f"{v:,.0f}",
           "usage": lambda v: f"{v:,.0f}", "price": lambda v: f"${v:.2f}", "cost": lambda v: f"${v:.4f}"}
    rows, seen = [], set()
    for d in drivers:
        metric = str(d.get("metric", "")).lower()
        role = classified.get(id(d)) or ("spend" if "spend" in metric and "per" not in metric else
                                         "resolution" if "resolution" in metric else None)
        if not role or role in seen or not all(isinstance(d.get(k), (int, float)) for k in ("before", "after")):
            continue
        windows = _stated_windows(d.get("period"), year) or (world.DEFAULT_COMPARE, world.DEFAULT_PERIOD)
        measured = [world.stats(conn, *w) for w in windows]
        if not all(s["tickets"] for s in measured):
            continue
        seen.add(role)
        b0, b1 = (_measure(s, role) for s in measured)
        show = fmt[role]
        same = all(abs(a - b) <= 0.02 * abs(b) for a, b in ((d["before"], b0), (d["after"], b1)))
        rows.append((f"{names[role]}<br><span style='{MUTED}'>{_e(d.get('period'))}</span>",
                     f"{show(d['before'])} → {show(d['after'])}", f"{show(b0)} → {show(b1)}",
                     "matches" if same else "differs from the data"))
    stated = run.proposals[0].get("expected_monthly_savings_usd")
    if isinstance(stated, (int, float)):
        truth = checks.truth(run.scenario)
        lo, hi = truth["savings_range"]
        low, high = truth["base_savings"]
        again = checks.proposal_projection(run)["monthly_savings"]
        rows.append((f"Monthly savings<br><span style='{MUTED}'>playbook's rule, Sep 10–21 rate</span>", _money(stated),
                     _money(again) + f"<br><span style='{MUTED}'>other reasonable base periods give "
                     f"{_money(low)} to {_money(high)}</span>",
                     "matches" if abs(stated - again) <= 0.02 * abs(again) else
                     "differs, but within the reasonable range" if lo <= stated <= hi else "differs"))
    return rows


def _routes_sentence(routes):
    by_model = {}
    for ticket_type, model in routes.items():
        by_model.setdefault(model, []).append(ticket_type)
    return "; ".join(f"{' and '.join(types)} → {model}" for model, types in by_model.items()) + "."


def _thin_margins(run, proposal):
    """A sentence on routes that clear the quality bar by less than two points, if any."""
    thin = [(t, world.PASS_PCT[m][i] - policy.quality_floor(t)) for i, t in enumerate(world.TICKET_TYPES)
            if (m := proposal["routes"].get(t)) in world.PASS_PCT and world.PASS_PCT[m][i] - policy.quality_floor(t) < 2]
    if not thin:
        return ""
    names = " and ".join(t.lower() for t, _ in thin)
    return (f" On {names} tickets, the chosen model clears the bar by {min(g for _, g in thin):.1f} to "
            f"{max(g for _, g in thin):.1f} points: a thin margin, and one more reason a person reviews the proposal.")


def _count_word(n):
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n] if 0 <= n < 10 else str(n)


def _k(v):
    return f"${abs(v) / 1000:,.1f}k"


def _bill_moved(run):
    """Why the bill went up: the key idea in plain words over the chart, with the exact math folded underneath."""
    bridge = checks.exact_bridge(run.scenario)
    f, parts = bridge["factors"], dict(bridge["parts"])
    tickets, usage, price = (f[k][1] / f[k][0] for k in ("tickets", "tokens_per_ticket", "effective_price_per_million"))
    per_ticket, total = usage * price, tickets * usage * price
    pct = lambda x: f"{'rose' if x >= 1 else 'fell'} {abs(100 * (x - 1)):.0f}%"
    if price < 1 < usage and per_ticket > 1:
        cheaper = "about half" if 0.45 <= price <= 0.55 else f"{100 * (1 - price):.0f}%"
        more = "about a quarter more" if 1.2 <= per_ticket <= 1.3 else f"about {100 * (per_ticket - 1):.0f}% more"
        plain = (f"Across Sep 1–21, which includes a week before the switch, the average price per token fell by {cheaper}, "
                 f"but tokens per ticket rose about {usage:.1f} times, so each ticket cost {more}. Tickets rose only "
                 f"{100 * (tickets - 1):.0f}%, so the bill rose about "
                 f"{100 * (total - 1):.0f}%. In the chart, orange bars add to the bill and blue bars cut it.")
    else:
        plain = f"Tickets {pct(tickets)}, tokens per ticket {pct(usage)}, and the price per token {pct(price)}."
    sign = lambda v: f"{'+' if v >= 0 else '−'} {_k(v)}"
    math = (f"Spend = tickets × tokens per ticket × average price per token. From Aug 1–21 to Sep 1–21, those changed by "
            f"×{tickets:.3f}, ×{usage:.3f}, and ×{price:.3f}: {tickets:.3f} × {usage:.3f} × {price:.3f} = {total:.3f}, "
            f"so spend went from {_k(bridge['before'])} to {_k(bridge['after'])}. The chart splits the "
            f"{_k(bridge['after'] - bridge['before'])} rise: {_k(bridge['before'])} {sign(parts['volume'])} (more tickets) "
            f"{sign(parts['usage'])} (more tokens per ticket) {sign(parts['price'])} (lower price) = {_k(bridge['after'])}.")
    image = (f"<img src='data:image/png;base64,{base64.b64encode(bridge_png(run)).decode()}' alt='Spend bridge' "
             "style='max-width:100%;width:700px'>")
    return (f"<div style='margin:4px 0 6px'><b>Why the bill went up.</b> We drew this chart from the same help-desk data "
            f"Super VL queried; Super VL did not see this chart. {plain}</div>{image}"
            + _details("Check the math", f"<p style='margin:4px 0'>{math}</p>"))


def _savings_math(run, projection, proposal, baseline):
    """Return a short savings takeaway and the exact arithmetic for an expandable check."""
    daily, saved = projection["daily_tickets"], projection["monthly_savings"]
    now_month, new_month = projection["current_monthly_spend"], projection["proposed_monthly_spend"]
    mix, routes = projection["ticket_mix"], proposal["routes"]
    cost = lambda model, t: checks.unit_cost(model, t, run.scenario)
    # the exact math as a reader would check it: costs per ticket to 4 decimals, months in thousands, so the sums add up
    now, new = round(projection["current_cost_per_ticket"], 4), round(projection["proposed_cost_per_ticket"], 4)
    back_exact = sum(mix[t] * cost(baseline, t) for t in world.TICKET_TYPES)
    back = round(back_exact, 4)
    month = lambda per_ticket: round(30 * daily * per_ticket / 1000)
    weighted = " + ".join(f"{mix[t]:.2f} × ${cost(routes[t], t):.4f}" for t in world.TICKET_TYPES)
    bridge = checks.exact_bridge(run.scenario)
    sep_avg = bridge["after"] / bridge["factors"]["tickets"][1]
    cents = lambda v: f"{round(100 * v)} cents"
    summary = (f"Using the Sep 10–21 average ticket volume for a 30-day month, the proposed routing cuts average cost "
               f"per ticket from about {cents(now)} to about {cents(new)}, saving about "
               f"{_money(round(saved, -3))} a month.")
    math = [f"Today: since the switch (Sep 10–21), {daily:,.0f} tickets a day at ${now:.4f} a ticket (the Sep 1–21 "
            f"average, ${sep_avg:.4f}, is lower because it includes the week before the switch). For 30 days: "
            f"{daily:,.0f} × ${now:.4f} × 30 ≈ ${month(now)}k.",
            f"With this routing: each type's cost times its share of tickets, {weighted} ≈ ${new:.4f} a ticket. For 30 "
            f"days: {daily:,.0f} × ${new:.4f} × 30 ≈ ${month(new)}k.",
            f"Saving: ${month(now)}k − ${month(new)}k ≈ ${month(now) - month(new)}k a month (unrounded, {_money(saved)})."]
    cheaper = [t for t in world.TICKET_TYPES if cost(routes[t], t) < cost(baseline, t)]
    if now - back > 0.01 * now and back > new and cheaper:
        models = " and ".join(sorted({routes[t] for t in cheaper}))
        share = round(100 * sum(mix[t] for t in cheaper))
        ratio = sum(mix[t] * cost(routes[t], t) for t in cheaper) / sum(mix[t] * cost(baseline, t) for t in cheaper)
        fraction = ("less than a tenth as much" if 0.05 <= ratio < 0.1 else "about a tenth as much" if 0.1 <= ratio <= 0.13
                    else f"{100 * ratio:.0f}% as much")
        undo_exact, extra_exact = now_month - 30 * daily * back_exact, 30 * daily * back_exact - new_month
        undo, extra = round(undo_exact, -3), round(extra_exact, -3)
        math.append(
            f"<b>Why the saving is larger than the bill's rise.</b> Going back to {_e(baseline)} for every ticket would only "
            f"undo the switch, saving about {_money(undo)} a month (the dashboard's rise is smaller because it covers three "
            f"weeks, one of them before the switch). Sending {' and '.join(t.lower() for t in cheaper)} tickets "
            f"({share}% of all tickets) to {_e(models)}, which costs {fraction} as {_e(baseline)} and still meets the "
            f"bar, saves about {_money(extra)} a month more.")
        now_exact, new_exact = projection["current_cost_per_ticket"], projection["proposed_cost_per_ticket"]
        math.append(f"The two parts, from unrounded costs: back to {_e(baseline)}, ${now:.4f} − ${back:.4f} ≈ "
                    f"${now_exact - back_exact:.4f} a ticket, about ${undo_exact / 1000:.1f}k a month; then {_e(models)}, "
                    f"${back:.4f} − ${new:.4f} ≈ ${back_exact - new_exact:.4f} a ticket, about ${extra_exact / 1000:.1f}k "
                    f"a month.")
    return summary, "".join(f"<p style='margin:4px 0'>{line}</p>" for line in math)


def show_reasoned_decision(run):
    """Moment 3, in order: what Super VL has, says, does next, and leaves for a person."""
    if not run.proposals or not run.report:
        return _card("3 · Decide", "<i>This run filed no proposal.</i>", run)
    proposal = run.proposals[0]
    projection = checks.proposal_projection(run)
    bar = policy.policy_for(run.scenario)["quality_bar"]
    baseline = bar["baseline_model"]
    chart = _bill_moved(run)
    route_rows = []
    for i, ticket_type in enumerate(world.TICKET_TYPES):
        model = proposal["routes"][ticket_type]
        route_rows.append((_e(ticket_type), f"{_e(model)}<br><span style='{MUTED}'>{_e(MODEL_ROLES[model])}</span>",
                           f"${checks.unit_cost(model, ticket_type, run.scenario):.4f}",
                           f"{world.PASS_PCT[model][i]:.1f}%", f"{policy.quality_floor(ticket_type):.1f}%"))
    disclosed = _disclosures(run)
    corrections = (_table(["The report says", "The data shows", "Effect"],
                          [(_e(d["recorded"]), _e(d["verified"]), _e(d["disposition"])) for d in disclosed],
                          ["30%", "35%", "35%"]) if disclosed else "")
    reviewed = (f"<div style='margin-top:10px'>{_slip_notes(run)}</div>" if _slip_notes(run) else
                ("<p style='margin-top:10px'><b>One statement in the report is wrong.</b></p>" if len(disclosed) == 1 else
                 f"<p style='margin-top:10px'><b>{_count_word(len(disclosed)).capitalize()} statements in the report are "
                 "wrong.</b></p>") + corrections
                if disclosed else
                f"<p style='{MUTED};margin-top:10px'>The report's wording is reviewed by hand only for the recording. For "
                "your own runs, read the report against the figures above.</p>" if not run.recorded else "")
    numbers = ("<p>After the run, we recomputed the report's figures from the data. Super VL did not see this "
               "check, and the recorded report is shown unedited.</p>"
               + _table(["Figure", "In the report", "From the data", "Result"], _figure_checks(run), ["30%", "22%", "28%", "20%"])
               + reviewed)
    validator_rows = [
        ("Proposal rule", "At most one proposal; every ticket type names a catalog model; each route meets its quality bar.",
         "Whether the routing is the cheapest, the savings arithmetic, or the numbers in the rationale."),
        ("Report rule", "Required fields are present; the proposal ID matches; the manager summary is short and avoids "
         "technical terms.", "Whether the report's figures and explanations agree with the data."),
    ]
    policy_scope = (
        _table(["Rule", "Checked by the tools", "Not checked"], validator_rows, ["18%", "42%", "40%"])
        + f"<p style='{MUTED}'>The instruction to pick the cheapest model that meets the bar comes from the system prompt; the "
          "tools don't enforce it.</p>"
        + "<div style='margin-top:8px;padding:9px 11px;background:#FFF8C5;border:1px solid #D4A72C;border-radius:6px'>"
          "<b>The model in use today would fail the quality bar.</b> The current model resolves 72.0% of technical "
          "tickets against a 74.0% bar, and 64.0% of integration tickets against 69.0%. The rule applies only to proposals "
          "filed through this agent, so nobody checked the rollout against it.</div>")
    units = (f"<div style='{MUTED};margin-top:4px'>Super VL writes “units” here for tokens.</div>"
             if re.search(r"\bunits?\b", str(run.report.get("finding")) + str(run.report.get("attribution")), re.I) else "")
    route_details = (
        _table(["Ticket type", "Model", "Cost per ticket", "Resolves in offline eval", "Quality bar"], route_rows,
               ["25%", "27%", "16%", "16%", "16%"])
        + f"<div style='{MUTED};margin-top:5px'>Assumes the Sep 10–21 volume, the Sep 1–21 ticket mix, the scanned prices, and "
          f"the report's token counts hold for 30 days.{_thin_margins(run, proposal)}</div>")
    _, savings_details = _savings_math(run, projection, proposal, baseline)
    full_recommendation = re.sub(r"\s+", " ", str(run.report.get("recommendation") or "")).strip()
    first = _sentences(full_recommendation, 1)
    recommendation = (
        ("<div>The report's recommendation begins, in Super VL's exact words:</div>" + _said(first + " …")
         + f"<div style='{MUTED};margin-top:4px'>The full recommendation is under <i>Read Super VL's full conclusion "
           "and attribution</i> below.</div>"
         if first != full_recommendation else
         "<div>The report recommends, in Super VL's exact words:</div>" + _said(first))
        if first else _routes_sentence(proposal["routes"])
    )
    full_conclusion = ("<div>Its finding, in Super VL's exact words:</div>" + _said(run.report.get("finding"))
                       + ("<div style='margin-top:8px'>Its recommendation:</div>" + _said(full_recommendation)
                          if full_recommendation else "")
                       + units + _attribution_block(run))
    body = _sequence([
        ("What it has",
         f"The team uses its former model, {_e(baseline)}, as the quality baseline. For each ticket type, Super VL must "
         f"choose the cheapest model whose offline test result is no more than {bar['max_drop_pct_points']:g} percentage "
         "points below that baseline. It may submit only one proposal in this run."),
        ("What it does next",
         "It applies that rule to the candidate models and submits one proposal for review. It cannot approve or apply it."),
        ("What it says", recommendation),
        ("What happens next",
         "A person decides whether to approve and apply the change. Nothing has changed."),
    ]) + (
        _details("Check the proposed routes and savings math", route_details + savings_details)
        + _details("Read Super VL's full conclusion and attribution", full_conclusion)
        + _details("Check the 30% bill increase", chart + _bridge_explanation(run))
        + _details("Check the report against the data", numbers)
        + _details("Inspect the policy and every candidate model", policy_scope + _options_matrix(run, proposal))
        + _details("Reproduce the savings projection",
                   _code(inspect.getsource(checks.proposal_projection)) + _code(json.dumps(projection, indent=2)))
    )
    return _card("3 · Decide", body, run)


def _bridge_explanation(run):
    bridge = checks.exact_bridge(run.scenario)
    f = bridge["factors"]
    ticket_factor, token_factor, price_factor = (f[k][1] / f[k][0] for k in
                                                 ("tickets", "tokens_per_ticket", "effective_price_per_million"))
    cost_factor = token_factor * price_factor
    token_table = _table(
        ["Period", "Input", "Answer", "Reasoning", "Total"],
        [(row["period"], f"{row['input']:,.0f}", f"{row['answer']:,.0f}", f"{row['reasoning']:,.0f}",
          f"{row['total']:,.0f}") for row in checks.token_breakdown(run.scenario)],
    )
    return (
        f"<p style='margin:4px 0 8px'>From Aug 1–21 to Sep 1–21, tokens per ticket rose {token_factor:.3f}× and the effective price per token fell {100 * (1 - price_factor):.1f}%, "
        f"so cost per ticket rose {100 * (cost_factor - 1):.1f}% ({token_factor:.3f} × {price_factor:.3f} = {cost_factor:.3f}). "
        f"With {100 * (ticket_factor - 1):.1f}% more tickets, spend rose {100 * (ticket_factor * cost_factor - 1):.1f}%.</p>"
        f"<p style='margin:0 0 8px'>The effective price fell less than the 75% cut in list prices because the new model's "
        f"tokens are far more often output, which is billed at four times the input rate, and because the first week of "
        f"September still ran on the previous model.</p>"
        f"<b>Tokens per ticket</b>{token_table}"
        f"<div style='{MUTED};margin-top:2px'>Values are rounded, so a row may not add up exactly.</div>"
        f"<p style='{MUTED};margin:8px 0'>The chart splits the change with a standard accounting method, the log-mean "
        f"Divisia index (LMDI), so the three bars add up exactly to the change in spend. It shows how much each factor "
        f"contributed, not why. The data shows that input and reasoning tokens rose after the model change, and the offline report "
        f"shows the same profile; it does not say whether the extra input comes from longer context, repeated history, "
        f"prompting, or serving.</p>"
        + _details("The code behind the split", _code(inspect.getsource(checks.exact_bridge))
                   + _code(inspect.getsource(checks.token_breakdown)))
    )


# Section 6: the request and the timeline ------------------------------------------------------------------

def show_request(scenario, live=False, run=None):
    """What the agent is given: the user's request, the pasted screenshot, and its tools."""
    config = _config(run) if run else DEFAULT
    text = run.request[0] if run else contract.request_for(scenario, config["request"])
    attachment_lead = "Here's what our help-desk dashboard shows."
    quoted = text
    lead = ""
    if text.endswith(" " + attachment_lead):
        quoted = text[: -len(attachment_lead) - 1]
        lead = f"<div style='margin-top:8px'>{_e(attachment_lead)}</div>"
    shot_path = run.image(run.request[1][0]) if run and run.request[1] else (
        _overview_image(scenario, live) if world.spec(scenario).get("attach_overview") else None)
    shot = (f"<div style='flex:0 0 300px'>{_img(shot_path, 300, 'The pasted dashboard screenshot')}"
            f"<div style='{MUTED}'>The screenshot pasted with the request</div></div>") if shot_path else ""
    names = [t["function"]["name"] for t in tools.tools_for(scenario, config)]
    ask = (f"<div style='flex:1;min-width:300px'><div style='font-size:16px'>“{_e(quoted)}”</div>{lead}"
           f"<div style='margin-top:12px'><b>What else it gets:</b> {len(names)} tools ({', '.join(f'<code>{n}</code>' for n in names)}), "
           "the team's playbook, and its policy. Near the bottom of the notebook, the <i>Look inside the implementation</i> "
           "section shows the exact request, playbook, policy, tool schemas, and model settings used for this run.</div></div>")
    return _card("The request", f"<div style='display:flex;gap:16px;flex-wrap:wrap'>{shot}{ask}</div>", run)


DOC_NAMES = {"price-schedule": "the scanned price list", "eval-report": "the eval report", "router-config": "the router config"}


def _doc_name(doc_id):
    return next((name for prefix, name in DOC_NAMES.items() if str(doc_id).startswith(prefix)), str(doc_id))


def _call_label(call):
    """What a tool call did, in plain words, and what came back."""
    a, r = call.arguments or {}, call.result or {}
    if call.name == "list_dashboards":
        return "Lists the dashboards it can open", "returned the list"
    if call.name == "list_documents":
        return "Lists the documents it can open", "returned the list"
    if call.name == "render_dashboard":
        return f"Opens the dashboard's {a.get('tab', '')} tab", "returned an image"
    if call.name == "view_document":
        page = int(a.get("page") or 1)
        return (f"Opens {_doc_name(a.get('doc_id'))}" + (f", page {page}" if str(a.get("doc_id", "")).startswith("eval-report") else ""),
                "returned the page as an image" if call.image_id else "returned the text")
    if call.name == "run_sql":
        return "Queries the database", f"returned {len(r['rows'])} rows" if "rows" in r else ""
    if call.name == "run_python":
        return "Runs Python in the sandbox", "finished" if not call.error and not r.get("exit_code") else ""
    if call.name == "propose_change":
        return "Files a proposal", f"{r['proposal_id']} recorded, waiting for a person to approve it" if r.get("proposal_id") else ""
    if call.name == "submit_report":
        return "Submits its report", "accepted" if not call.error else ""
    return call.name, ""


CALL_KINDS = {"list_dashboards": ("Catalog", "Ask"), "list_documents": ("Catalog", "Ask"),
              "render_dashboard": ("Dashboard", "Look"), "view_document": ("Document", "Look"),
              "run_sql": ("SQL", "Drill down"), "run_python": ("Python", "Verify"),
              "propose_change": ("Proposal", "Propose"), "submit_report": ("Report", "Report")}


def _returned(run, call, max_rows=25):
    """What a tool call returned, readable: the full image, the rows of a query, or a text document."""
    r = call.result or {}
    if call.image_id and run.image(call.image_id):
        if highlights.load(run).get(call.image_id):
            return (_highlighted_figure(run, call.image_id, 860, run.image_description(call.image_id),
                                        ids=[b["id"] for b in highlights.load(run)[call.image_id]])
                    + f"<div style='{MUTED};color:#953800'>{highlights.LABEL}</div>")
        return _img(run.image(call.image_id), 860, run.image_description(call.image_id))
    if r.get("columns") and r.get("rows") is not None:
        rows = r["rows"][:max_rows]
        more = len(r["rows"]) - len(rows)
        return (_table([_e(c) for c in r["columns"]], [tuple(_e(v) for v in row) for row in rows])
                + (f"<div style='{MUTED}'>…and {more} more rows</div>" if more > 0 else ""))
    if r.get("format") == "text" and r.get("content"):
        return _code(r["content"])
    return ""


def _call_html(run, call, index):
    """One tool call as a compact row: kind, what it did, what came back, why; click for arguments and result."""
    label, outcome = _call_label(call)
    kind, colors = CALL_KINDS.get(call.name, (call.name, "Ask"))
    failed = call.error or (call.name == "run_python" and (call.result or {}).get("exit_code"))
    status = (f"<span style='color:#CF222E'> · {_e(str(call.error or 'Python error')[:240])}</span>" if failed else
              f"<span style='color:#1A7F37'> · {_e(outcome)}</span>" if outcome else "")
    reason = (f"<div style='{MUTED};font-style:italic;margin-top:2px'>“{_e(call.why)}”</div>" if call.why else "")
    thumb = (f"<div style='flex:0 0 132px'>{_img(run.image(call.image_id), 132, run.image_description(call.image_id))}</div>"
             if call.image_id and run.image(call.image_id) else "")
    correction = _disclosure_banner(run) if call.name == "submit_report" else ""
    return (f"<details data-call='{index}' style='margin:6px 0 0;border:1px solid #E4E7EB;border-radius:6px;background:#FFFFFF'>"
            f"<summary style='display:flex;gap:10px;align-items:flex-start;padding:7px 10px;cursor:pointer'>"
            f"<div style='flex:0 0 78px'>{_chip(kind, STAGE_COLORS[colors])}</div>"
            f"<div style='flex:1;min-width:0'><b>{_e(label)}</b>{status}{reason}</div>{thumb}"
            f"<div style='flex:0 0 auto;{MUTED};font-size:12px;white-space:nowrap'>Call {index} ▸</div></summary>"
            f"<div style='padding:0 10px 8px'>{correction}{_returned(run, call)}"
            f"{_details('Raw arguments and result', _code(json.dumps({'arguments': call.arguments, 'result': call.result}, indent=2)))}"
            "</div></details>")


def _step_html(run, step, first_index=1):
    """One reply from Super VL: its note in its own words, then the tool calls it made, then any message from the loop."""
    note = (f"<div style='margin-top:5px;padding:8px 11px;background:#F6F8FA;border-left:3px solid #76B900;border-radius:4px'>"
            f"{_note(step.note)}</div>" if step.note else "")
    calls = "".join(_call_html(run, c, first_index + i) for i, c in enumerate(step.calls))
    loop = "".join(f"<div style='margin-top:6px;padding:6px 10px;background:#FFF8C5;border-radius:6px'><b>The loop:</b> "
                   f"{_e(m)}</div>" for m in step.harness)
    return (f"<div data-step='{step.number}' style='margin:12px 0 0'>"
            f"<div style='display:flex;align-items:center;gap:7px;font-weight:600'>"
            f"<span style='display:inline-block;width:9px;height:9px;border-radius:5px;background:#76B900'></span>"
            f"Super VL <span style='{MUTED};font-weight:400'>· step {step.number} · {_clock(step.elapsed)}</span></div>"
            f"{note}<div style='margin-left:4px;padding-left:12px;border-left:2px solid #E4E7EB'>{calls}{loop}</div></div>")


def _moment_spans(run):
    """{first step of each moment: (title, last step)} for this run."""
    doc_steps = [s.number for s, _ in run.calls("view_document", "render_dashboard")]
    last = run.steps[-1].number if run.steps else 1
    evidence_end = max(doc_steps) if doc_steps else 1
    spans = {1: ("Moment 1 · Read the dashboard", 1)}
    if evidence_end >= 2:
        spans[2] = ("Moment 2 · Find the missing evidence", evidence_end)
    if evidence_end < last:
        spans[evidence_end + 1] = ("Moment 3 · Decide", last)
    return spans


PLAYBACK_SECONDS = 30  # a played-back run lasts about this long, however long the run took


def _playable(conversation, seconds):
    """The conversation, with a Play button that replays its steps in order, condensed. Without JavaScript the button stays
    hidden and the full conversation shows; the recorder seeks with window.agentViewSeek(t)."""
    speed = max(1.0, seconds / PLAYBACK_SECONDS)
    return (f"<div class='av-root' data-end='{seconds:.2f}' data-speed='{speed:.3f}' style='max-width:960px'>"
            "<div class='av-controls' style='display:none;align-items:center;gap:10px;margin-bottom:8px'>"
            "<button type='button' style='font:inherit;font-weight:600;padding:4px 12px;border:1px solid #76B900;"
            "border-radius:6px;background:#FFFFFF;color:#1F2328;cursor:pointer'>▶ Play the run</button>"
            f"<span style='{MUTED}'>replays the steps in order, {speed:.0f}× faster than the run</span></div>"
            f"{conversation}<div class='av-working' style='display:none;margin:12px 0 0;{MUTED};font-style:italic'>"
            "Super VL is working…</div></div>" + _PLAYBACK_SCRIPT)


_PLAYBACK_SCRIPT = """<script>
(function () {
  var roots = document.querySelectorAll('.av-root:not([data-ready])');
  roots.forEach(function (root) {
    root.setAttribute('data-ready', '1');
    var items = Array.prototype.slice.call(root.querySelectorAll('.av-item'));
    var bar = root.querySelector('.av-controls'), button = bar.querySelector('button');
    var end = parseFloat(root.getAttribute('data-end')), speed = parseFloat(root.getAttribute('data-speed'));
    var timer = null, shown = null, full = 0;
    var working = root.querySelector('.av-working');
    function seek(t, follow) {
      if (!full) { full = root.offsetHeight; }
      var last = null, playing = t < end;
      items.forEach(function (el) {
        var on = parseFloat(el.getAttribute('data-t')) <= t;
        if (el.shownAs === undefined) { el.shownAs = el.style.display; }
        el.style.display = on ? el.shownAs : 'none';
        if (on) { last = el; }
      });
      working.style.display = playing && t >= 0 ? '' : 'none';
      root.style.minHeight = playing ? full + 'px' : '';
      if (follow && last && last !== shown) { last.scrollIntoView({behavior: 'smooth', block: 'nearest'}); }
      shown = last;
    }
    function stop() { clearInterval(timer); timer = null; button.textContent = '\u25B6 Play the run'; seek(Infinity); }
    button.onclick = function () {
      if (timer) { stop(); return; }
      var start = performance.now();
      button.textContent = '\u25A0 Stop';
      seek(-1, true);
      timer = setInterval(function () {
        var t = (performance.now() - start) / 1000 * speed;
        seek(t, true);
        if (t > end) { clearInterval(timer); timer = null; button.textContent = '\u25B6 Play the run'; }
      }, 200);
    };
    bar.style.display = 'flex';
    window.agentViewSeek = function (t) { seek(t, false); };
  });
})();
</script>"""


def show_agent_view(run):
    """The whole run as a conversation: the request, each step's note and tool calls, what came back, and the outcome."""
    text, attached = run.request
    r_end = f"{run.result.get('seconds', 0):.2f}"
    shot = "".join(f"<details data-call='request' style='margin-top:6px'><summary style='cursor:pointer;list-style:none'>"
                   f"{_img(run.image(i), 240, 'The screenshot pasted with the request')}</summary>"
                   + _highlighted_figure(run, i, 860, "The screenshot pasted with the request",
                                         ids=[b["id"] for b in highlights.load(run).get(i, [])])
                   + f"<div style='{MUTED};color:#953800'>{highlights.LABEL}</div></details>"
                   for i in attached if run.image(i))
    parts = [f"<div class='av-item' data-t='0' style='display:flex;justify-content:flex-end'><div style='max-width:720px;padding:10px 13px;"
             f"background:#EEF2F6;border-radius:10px'><div style='font-weight:600'>You</div><div>{_e(text)}</div>{shot}"
             "</div></div>"]
    spans, index = _moment_spans(run), 1
    for step in run.steps:
        if step.number in spans:
            title, end = spans[step.number]
            steps = f"step {step.number}" if end == step.number else f"steps {step.number}–{end}"
            parts.append(f"<div class='av-item' data-t='{step.elapsed:.2f}' style='display:flex;align-items:baseline;gap:8px;margin:20px 0 2px'>"
                         f"<span style='font-weight:700'>{_e(title)}</span><span style='{MUTED}'>{steps}</span>"
                         "<span style='flex:1;border-bottom:1px solid #D0D7DE'></span></div>")
        parts.append(f"<div class='av-item' data-t='{step.elapsed:.2f}'>{_step_html(run, step, index)}</div>")
        index += len(step.calls)
    if run.proposals and "proposal_id" in run.proposals[0]:
        p = run.proposals[0]
        savings = p.get("expected_monthly_savings_usd")
        parts.append(f"<div class='av-item' data-t='{r_end}' style='margin:16px 0 0;padding:10px 13px;background:#FFF8C5;border-radius:8px'>"
                     f"<b>Outcome:</b> proposal {_e(p['proposal_id'])}. {_e(_routes_sentence(p['routes']))}"
                     + (f" Projected savings: {_money(savings)} a month." if isinstance(savings, (int, float)) else "")
                     + " It is waiting for a person to approve it; nothing has changed. Moment 3, above, checks these figures.</div>")
    r = run.result
    tokens = sum((t.get("prompt_tokens") or 0) + (t.get("completion_tokens") or 0) for t in r.get("turn_log", []))
    problems = sum(1 for _, c in run.calls() if c.error)
    parts.append(f"<div class='av-item' data-t='{r_end}' style='{MUTED};margin-top:10px'>{len(run.steps)} steps · {len(run.calls())} tool calls · "
                 f"{len(r.get('images', []))} images · {r.get('seconds', 0) / 60:.1f} minutes · reminders from the loop: "
                 f"{r.get('nudges', 0) + r.get('truncations', 0)} · errors and rejections: {problems} · "
                 f"{tokens:,} model tokens</div>")
    which = "this recording" if run.recorded else "your live run"
    return _card("The full run", _playable("".join(parts), run.result.get("seconds", 0)), run,
                 f"Every step of {which} as a conversation. Super VL's own words are in the gray boxes, times are from "
                 "the start of the run, and clicking a tool call shows its arguments and result.")


def show_live_step(run):
    """The latest step of a run in progress, in the same style as the full run."""
    if not run.steps:
        return _card("Live investigation", "<i>Waiting for Super VL's first reply…</i>", run)
    step = run.steps[-1]
    first = 1 + sum(len(s.calls) for s in run.steps[:-1])
    return _card(f"Live investigation · step {step.number}",
                 _step_html(run, step, first) + f"<div style='{MUTED}'>The cards below fill in when the run finishes.</div>", run)



# Sections 7 to 10: step cards -----------------------------------------------------------------------------

def _document_views(run, prefix):
    return [(s, c) for s, c in run.calls("view_document") if str((c.arguments or {}).get("doc_id", "")).startswith(prefix) and c.image_id]


DISAGREES = re.compile(r"not match|n't match|discrepan|mismatch|double[- ]count|\bbug\b|bucket|inflat|disagree", re.I)


# Section 11: the answer ------------------------------------------------------------------------------------

def _classify(drivers):
    roles = {}
    for d in drivers:
        metric = str(d.get("metric", "")).lower()
        if not isinstance(d.get("before"), (int, float)) or not isinstance(d.get("after"), (int, float)) or d["before"] <= 0 or d["after"] <= 0:
            continue
        role = ("cost" if re.search(r"cost per (ticket|resolved)", metric) and "resolved" not in metric else
                "usage" if "token" in metric and "ticket" in metric and "price" not in metric else
                "price" if re.search(r"price|per 1m|per million", metric) else
                "volume" if "ticket" in metric and not re.search(r"per|cost|token", metric) else None)
        if role and role not in roles:
            roles[role] = d
    return roles


def bridge(drivers):
    """Split the spend change into volume, usage, and price effects (log-mean Divisia index, LMDI).

    Spend = tickets x cost per ticket, and cost per ticket = tokens per ticket x price per token. LMDI splits an
    exact change in a product into additive parts that do not depend on the order of the factors. The usage and
    price parts share the cost-per-ticket part in proportion to their log changes.
    """
    roles = _classify(drivers)
    if "volume" not in roles or not ({"cost"} <= set(roles) or {"usage", "price"} <= set(roles)):
        return None
    t0, t1 = roles["volume"]["before"], roles["volume"]["after"]
    if "cost" in roles:
        c0, c1 = roles["cost"]["before"], roles["cost"]["after"]
    else:
        c0, c1 = (roles["usage"]["before"] * roles["price"]["before"] / 1e6, roles["usage"]["after"] * roles["price"]["after"] / 1e6)
    s0, s1 = t0 * c0, t1 * c1
    weight = (s1 - s0) / math.log(s1 / s0) if abs(s1 - s0) > 1e-9 else s0
    parts = [("volume", weight * math.log(t1 / t0))]
    per_ticket = weight * math.log(c1 / c0)
    if {"usage", "price"} <= set(roles):
        lu = math.log(roles["usage"]["after"] / roles["usage"]["before"])
        lp = math.log(roles["price"]["after"] / roles["price"]["before"])
        total = lu + lp
        if abs(total) > 1e-9:
            parts += [("usage", per_ticket * lu / total), ("price", per_ticket * lp / total)]
        else:
            parts += [("usage", per_ticket), ("price", 0.0)]
    else:
        parts += [("cost per ticket", per_ticket)]
    return {"before": s0, "after": s1, "parts": parts, "drivers": roles, "consistent": _consistent(drivers, roles, s0, s1)}


def _consistent(drivers, roles, s0, s1):
    """The drivers describe one comparison: they reproduce any spend the report states, and tokens per ticket times
    price per token reproduces cost per ticket. Drivers taken from different periods fail one of these."""
    close = lambda a, b, tol: abs(a - b) <= tol * abs(b)
    spend = next((d for d in drivers if re.search(r"\bspend\b|\bbill\b", str(d.get("metric", "")), re.I)
                  and isinstance(d.get("before"), (int, float)) and isinstance(d.get("after"), (int, float))), None)
    if spend and not (close(s0, spend["before"], 0.02) and close(s1, spend["after"], 0.02)):
        return False
    if {"cost", "usage", "price"} <= set(roles):
        u, p, c = roles["usage"], roles["price"], roles["cost"]
        return all(close(u[k] * p[k] / 1e6, c[k], 0.03) for k in ("before", "after"))
    return True


def _change(d):
    ratio = d["after"] / d["before"]
    return f"{ratio:.1f}×" if ratio >= 1.5 else f"{100 * (ratio - 1):+.1f}%".replace("-", "−")


def bridge_png(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = checks.exact_bridge(run.scenario)
    factors = split["factors"]
    names = {"volume": ("More tickets", "Fewer tickets"), "usage": ("More tokens per ticket", "Fewer tokens per ticket"),
             "price": ("Higher price per token", "Lower price per token")}
    labels, bottoms, heights, colors = ["Bill, Aug 1–21"], [0], [split["before"]], ["#8C959F"]
    level = split["before"]
    for key, value in split["parts"]:
        up, down = names[key]
        before, after = factors[{"volume": "tickets", "usage": "tokens_per_ticket",
                                 "price": "effective_price_per_million"}[key]]
        change = f"\n({_change({'before': before, 'after': after})})"
        labels.append((up if value >= 0 else down) + change)
        bottoms.append(level if value >= 0 else level + value)
        heights.append(abs(value))
        colors.append("#D55E00" if value >= 0 else "#0072B2")
        level += value
    labels.append("Bill, Sep 1–21")
    bottoms.append(0)
    heights.append(split["after"])
    colors.append("#57606A")
    fig, ax = plt.subplots(figsize=(9.6, 4.3), dpi=220)  # drawn at 2x its display size, for sharp text
    ax.bar(range(len(labels)), heights, bottom=bottoms, color=colors, width=0.62)
    signed = [split["before"]] + [v for _, v in split["parts"]] + [split["after"]]
    for i, (b, h, v) in enumerate(zip(bottoms, heights, signed)):
        text = f"${v / 1000:,.1f}k" if i in (0, len(labels) - 1) else f"{'+' if v >= 0 else '−'}${abs(v) / 1000:,.1f}k"
        ax.text(i, b + h + max(heights) * 0.015, text, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylim(0, max(b + h for b, h in zip(bottoms, heights)) * 1.14)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("How volume, usage, and price moved the bill", fontsize=12, loc="left", fontweight="bold")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


# Section 13: the proposal ------------------------------------------------------------------------------

def _model_today(scenario):
    router = world.spec(scenario)["router"]
    slug = re.search(r"default_model:\s*(\S+)", router).group(1)
    return next((m for m, s in world.SLUGS.items() if s == slug), slug)


def _options_matrix(run, proposal):
    """Ticket type x model: projected cost and whether each option clears policy."""
    today = _model_today(run.scenario)
    head = ["Ticket type"] + [
        m + f"<br><span style='font-weight:400'>{_e(MODEL_ROLES[m])}"
        + ("; today" if m == today else "") + "</span>" for m in world.PRICES
    ]
    rows = []
    for i, ticket_type in enumerate(world.TICKET_TYPES):
        floor = policy.quality_floor(ticket_type)
        cells = [f"<b>{_e(ticket_type)}</b><br><span style='{MUTED}'>quality bar {floor:.1f}%</span>"]
        chosen = proposal["routes"].get(ticket_type)
        for model in world.PRICES:
            cost = checks.unit_cost(model, ticket_type, run.scenario)
            rate = world.PASS_PCT[model][i]
            passes = rate >= floor
            style = ("outline:3px solid #1A7F37;outline-offset:-3px;background:#EAF7EE;"
                     if model == chosen else ("background:#F6F8FA;" if passes else "background:#FFF1F0;"))
            status = "Chosen" if model == chosen else ("Meets bar" if passes else "Below bar")
            cells.append(f"<div style='{style}padding:7px;border-radius:5px'><b>{status}</b><br>"
                         f"${cost:.4f}/ticket<br>{rate:.1f}% resolution</div>")
        rows.append(cells)
    cheapest = {t: min((m for m in world.PRICES if world.PASS_PCT[m][i] >= policy.quality_floor(t)),
                       key=lambda m: checks.unit_cost(m, t, run.scenario), default=None)
                for i, t in enumerate(world.TICKET_TYPES)}
    misses = [t for t in world.TICKET_TYPES if proposal["routes"].get(t) != cheapest[t]]
    verdict = ("For every ticket type, the chosen model is the cheapest one that meets the bar." if not misses else
               f"For {_e(', '.join(misses))}, the chosen model is not the cheapest one that meets the bar.")
    return (f"<p style='margin:4px 0 8px'>Cost per ticket and offline quality for every model, which we computed "
            f"from the price list and the report. {verdict} The policy does not check this.</p>"
            + _table(head, rows, ["20%"] + ["16%"] * len(world.PRICES)))


# Section 14: other months -------------------------------------------------------------------------------

def show_build_guide(run=None):
    """The implementation behind a run: the loop, the tools, the exact prompt and request, and the settings."""
    from .harness import investigate

    run = run or load_recording("model-switch-visual")
    config = _config(run)
    run_tools = tools.tools_for(run.scenario, config)
    tool_rows = []
    for tool in run_tools:
        fn = tool["function"]
        tool_rows.append((f"<code>{_e(fn['name'])}</code>", _e(fn["description"].split(". ")[0].rstrip(".") + ".")))
    agent_guide = contract.playbook(run.scenario, config["playbook"])
    body = (
        "<p><b>The loop</b> (<code>investigator/harness.py</code>)</p>"
        + f"<p style='{MUTED}'>Each pass sends the conversation to Super VL and runs the tools it calls. If a reply calls no "
          "tool, the loop sends a fixed reminder (a <i>nudge</i>), up to two per run. The lines about a draft belong to an optional review "
          "mode that this agent doesn't use.</p>"
        + _details("Show the source", _code(inspect.getsource(investigate)))
        + "<p><b>The eight tools</b></p>" + _table(["Tool", "What it does"], tool_rows)
        + _details("Complete tool schemas", _code(json.dumps(run_tools, indent=2)))
        + _details("The playbook this run saw", _code(agent_guide))
        + f"<p style='{MUTED}'>The base file is <code>playbook/helpdesk.md</code>. In this scenario, the offline-eval "
          "results live in the two-page report instead of a database table, so Super VL saw the version above, which points "
          "to the report.</p>"
        + _details("The policy (policy/helpdesk.json)", _code(policy.policy_text())
                   + f"<p style='{MUTED}'><code>contract.py</code> also adds a routing objective to the system prompt: for "
                     "each ticket type, the lowest-cost catalog model that meets the quality bar. The tools don't enforce "
                     "it.</p>")
        + _details("Enforced validation implementation",
                   _code(inspect.getsource(contract.validate_report) + "\n\n"
                         + inspect.getsource(policy.check_proposal) + "\n\n"
                         + inspect.getsource(policy.check_report)))
        + _details("Exact initial system/user messages",
                   _code(json.dumps({
                       "system": contract.system_prompt(run.scenario, run.result.get("ablate"), config),
                       "user": run.request[0],
                       "attached_image_ids": run.request[1],
                   }, indent=2)))
        + _details("Recorded model settings and capture metadata", _code(json.dumps(run.manifest, indent=2)))
    )
    return _card("Look inside the implementation", body)
