# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the model is told: the system contract, the playbook, the policy rules, the request, and the report contract.

None of these name a month's answer. The playbook (playbook/helpdesk.md) and the policy (policy/helpdesk.json)
live in their own files so the people who own them can edit them. Company names, dates, and document IDs
come from the world; the agent configuration (investigator/config.py) selects the prompt variants.
"""

import copy
from pathlib import Path

import world

from .config import DEFAULT
from .policy import policy_text

PLAYBOOK_FILE = Path(__file__).resolve().parents[1] / "playbook" / "helpdesk.md"
PLAYBOOK = PLAYBOOK_FILE.read_text()

REQUEST = ("Our help-desk AI bill is up about 30% this month. What happened? If we can route tickets differently to cut "
           "cost without hurting quality, prepare that change for review; don't change anything yet.")
NEUTRAL_REQUEST = ("Our help-desk AI bill is up about {pct}% this month. What happened? If we can route tickets differently "
                   "to cut cost without hurting quality, prepare that change for review; don't change anything yet.")
TRUE_HINTS = {  # what a user who already half-knows the cause might say
    "model_switch": "even though we moved to a cheaper model",
    "volume": "and we've been getting a lot more tickets",
    "price": "and I heard our provider changed its prices",
    "mix": "and we're seeing more of the harder tickets",
    "volume_price": "and we've been getting more tickets",
    "retry_bug": "after a gateway change",
    "none": "",
}
FLIP_HINTS = {  # a plausible but wrong explanation, by what really happened
    "model_switch": "I think it's because we're handling a lot more tickets.",
    "volume": "even though we moved to a cheaper model",
    "price": "I think it's because we're handling a lot more tickets.",
    "mix": "even though we moved to a cheaper model",
    "retry_bug": "I think the provider raised its prices.",
    "volume_price": "even though we moved to a cheaper model",
    "none": "even though we moved to a cheaper model",
}


def _spend_change(scenario):
    s = world.spec(scenario)
    world.ensure(scenario)
    conn = world.connect(scenario)
    a, b = world.stats(conn, *s["compare"]), world.stats(conn, *s["period"])
    return round(100 * (b["spend"] / a["spend"] - 1))


def request_for(scenario, mode="scenario"):
    """The user's question. scenario: what the world's user says (it may include what they believe happened);
    neutral: only the size of the change; hint: the size and a true clue; flip: the size and a wrong explanation."""
    s = world.spec(scenario)
    if mode == "scenario":
        if s.get("request"):
            return s["request"]
        return REQUEST if s.get("authored") else NEUTRAL_REQUEST.format(pct=_spend_change(scenario))
    if mode == "neutral":
        return NEUTRAL_REQUEST.format(pct=_spend_change(scenario))
    if mode in ("flip", "hint"):
        hint = (FLIP_HINTS if mode == "flip" else TRUE_HINTS)[s["family"]]
        if not hint:
            return NEUTRAL_REQUEST.format(pct=_spend_change(scenario))
        pct = _spend_change(scenario)
        if hint.startswith(("even though", "and ", "after ")):
            return (f"Our help-desk AI bill is up about {pct}% this month, {hint}. What happened? If we can route tickets "
                    "differently to cut cost without hurting quality, prepare that change for review; don't change anything yet.")
        return (f"Our help-desk AI bill is up about {pct}% this month. {hint} What happened? If we can route tickets "
                "differently to cut cost without hurting quality, prepare that change for review; don't change anything yet.")
    raise ValueError(f"unknown request mode {mode!r}")


CONTRACT = """You are an investigation agent for the Fernledger AI help-desk team. You answer questions about the help desk's AI cost and quality. You can look at the team's dashboards and documents, query its data with SQL, and run Python.

How to work:
1. Decide what to look at and what to check. Dashboard tabs and scanned documents come back as images. Before your first tool call, say in one or two sentences what the request and any attached images tell you, and what you will check first.
2. Every exact number you report must come from a SQL result, a Python result, or a document. Do not estimate exact numbers from chart pixels.
3. Label each claim Observed (seen in a dashboard, document, or query result), Calculated (computed with SQL or Python), or Hypothesis (a likely explanation the data does not prove). Two things happening at the same time is an association, not proof of cause.
4. You may file at most one change proposal, only with propose_change. Filing is optional; if nothing should change, say so.
5. If the evidence does not support a conclusion, say so and set confidence to "inconclusive".
6. Finish by calling submit_report once. Do not write the final answer as plain text."""

FIRST_LOOK = (" Before your first tool call, say in one or two sentences what the request and any attached images tell you, "
              "and what you will check first.")
LABELS = ("3. Label each claim Observed (seen in a dashboard, document, or query result), Calculated (computed with SQL or "
          "Python), or Hypothesis (a likely explanation the data does not prove). Two things happening at the same time is an "
          "association, not proof of cause.")
NO_LABELS = "3. Two things happening at the same time is an association, not proof of cause."
TEST_REQUEST = ("If the request suggests an explanation, treat it as a hypothesis to test against the data; it may be wrong "
                "or incomplete.")
OBJECTIVE = ("- Routing objective: a routing proposal assigns each ticket type the lowest-cost catalog model that meets the "
             "quality bar for that ticket type.")
PLAN = ("Before you investigate, write a short plan of the checks you will run with the notes tool, and update it as you "
        "learn.")


def contract_text(scenario, config=None):
    config = config or DEFAULT
    s = world.spec(scenario)
    text = CONTRACT.replace("Fernledger", s["company"])
    if not config["first_look"]:
        text = text.replace(FIRST_LOOK, "")
    if not config["labels"]:
        text = text.replace(LABELS, NO_LABELS)
    extra = ([TEST_REQUEST] if config["contract"] == "v2" else []) + ([PLAN] if config["plan_first"] else [])
    if extra:
        rules = text.split("\n")
        last = rules.pop()  # "6. Finish by calling submit_report once..."
        number = int(last.split(".", 1)[0])
        rules += [f"{number + i}. {rule}" for i, rule in enumerate(extra)] + [f"{number + len(extra)}." + last.split(".", 1)[1]]
        text = "\n".join(rules)
    return text


EVAL_AS_REPORT = [  # where the playbook points when offline-eval results are published only as a report
    ("- **Offline-eval resolution rate:** from `eval_results`, where historical tickets",
     "- **Offline-eval resolution rate:** from the Eval Team's offline eval report (document `eval-report-2026-08`), where historical tickets"),
    ("- `eval_results(model, ticket_type, resolution_rate_pct, eval_tickets, avg_input_tokens, avg_output_answer_tokens, "
     "avg_output_reasoning_tokens, avg_requests)`: offline eval for every catalog model, including models not in production.\n", ""),
    ("Prices are not in the database; the current price schedule is a document.",
     "Prices and offline eval results are not in the database; the current price schedule and the offline eval report are documents."),
    ("from `eval_results` tokens and the price schedule", "from the eval report's tokens per ticket and the price schedule"),
]


def playbook(scenario, variant="generic"):
    """The playbook file with the world's names and dates."""
    s = world.spec(scenario)
    text = PLAYBOOK
    if not s["eval"]["in_db"]:
        for old, new in EVAL_AS_REPORT:
            assert old in text, old
            text = text.replace(old, new)
    if not s.get("authored"):
        for old, new in (("Fernledger", s["company"]), ("Larkfield Cloud", s["provider"]), ("eval-report-2026-08", s["eval"]["id"]),
                         ("2026-08-01", s["first_day"]), ("2026-09-21", s["last_day"]), ("2026-09-22", s["today"])):
            text = text.replace(old, new)
    return text


def system_prompt(scenario, ablate=None, config=None):
    """The contract, then the playbook, then the policy. ablate="playbook" leaves the playbook out."""
    config = config or DEFAULT
    variant = "none" if ablate == "playbook" else config["playbook"]
    parts = [contract_text(scenario, config)]
    if variant != "none":
        parts.append(f"The team's playbook:\n<playbook>\n{playbook(scenario, variant)}\n</playbook>")
    policy = policy_text(scenario) + (f"\n{OBJECTIVE}" if config["objective"] == "explicit" else "")
    parts.append(f"The team's policy, enforced by the tools:\n<policy>\n{policy}\n</policy>")
    return "\n\n".join(parts)


# The report contract: the same fields for every question -----------------------------------------------

REPORT = {
    "finding": {"type": "string", "description": "What happened, in 2 to 4 sentences"},
    "how_it_happened": {"type": "string", "description": "How the cause changed the metric, with the numbers that show it"},
    "drivers": {"type": "array", "description": "The metric that changed and each driver, before and after (for cost: tickets, "
                "tokens per ticket, price per 1M tokens, cost per ticket)", "items": {
        "type": "object", "properties": {"metric": {"type": "string"}, "before": {"type": "number"}, "after": {"type": "number"},
                                         "unit": {"type": "string"}, "period": {"type": "string"}, "source": {"type": "string"}},
        "required": ["metric", "before", "after", "unit", "period", "source"]}},
    "quality_impact": {"type": "string", "description": "What happened to quality, and whether the setup running today meets the quality bar"},
    "key_figures": {"type": "array", "description": "The numbers that support the finding", "items": {
        "type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "number"}, "unit": {"type": "string"},
                                         "period": {"type": "string"}, "source": {"type": "string"}},
        "required": ["name", "value", "unit", "period", "source"]}},
    "evidence": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "label": {"type": "string", "enum": ["Observed", "Calculated", "Hypothesis"]},
        "source": {"type": "string"}}, "required": ["claim", "label", "source"]}},
    "ruled_out": {"type": "array", "description": "Explanations you checked and rejected", "items": {
        "type": "object", "properties": {"candidate": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["candidate", "reason"]}},
    "recommendation": {"type": "string"},
    "proposal_id": {"type": "string", "description": "The ID returned by propose_change, or \"none\""},
    "confidence": {"type": "string", "enum": ["high", "medium", "low", "inconclusive"]},
    "manager_summary": {"type": "string"},
    "engineer_summary": {"type": "string"},
}


ATTRIBUTION = {"type": "object", "description": "What set the change off: the change-log entries and documents (by ID) that "
               "explain the driver, and how they caused it. Use empty lists if no recorded change explains it.", "properties": {
                   "change_ids": {"type": "array", "items": {"type": "string"}},
                   "documents": {"type": "array", "items": {"type": "string"}},
                   "explanation": {"type": "string"}}, "required": ["change_ids", "documents", "explanation"]}
ATTRIBUTION_BY_DRIVER = {"type": "array", "description": "One entry per driver that moved: the day it started to move, the "
                         "change-log entries and documents (by ID) that explain it, and how they caused it. Use empty lists for "
                         "a driver that no recorded change explains.", "items": {"type": "object", "properties": {
                             "driver": {"type": "string"}, "started": {"type": "string", "description": "YYYY-MM-DD"},
                             "change_ids": {"type": "array", "items": {"type": "string"}},
                             "documents": {"type": "array", "items": {"type": "string"}},
                             "explanation": {"type": "string"}},
                             "required": ["driver", "started", "change_ids", "documents", "explanation"]}}


def report_schema(config=None):
    config = config or DEFAULT
    if config["schema"] == "v1" and config["labels"] and not config["attribution"]:
        return REPORT
    report = copy.deepcopy(REPORT)
    if config["attribution"]:
        fields = list(report.items())
        at = next(i for i, (k, _) in enumerate(fields) if k == "how_it_happened") + 1
        schema = ATTRIBUTION_BY_DRIVER if config["attribution"] == "per_driver" else ATTRIBUTION
        report = dict(fields[:at] + [("attribution", schema)] + fields[at:])
    if config["schema"] == "neutral":
        report["drivers"]["description"] = "The metric that changed and each driver, before and after"
    if not config["labels"]:
        item = report["evidence"]["items"]
        item["properties"].pop("label")
        item["required"] = ["claim", "source"]
    return report


def validate_report(report, config=None):
    """Contract checks that apply to every report, before any policy check."""
    missing = [f for f in report_schema(config) if f not in report]
    if missing:
        return [f"missing required fields {missing}"]
    if not isinstance(report.get("drivers"), list) or not report["drivers"]:
        return ["drivers is empty; list the metric that changed and each driver, before and after"]
    return []
