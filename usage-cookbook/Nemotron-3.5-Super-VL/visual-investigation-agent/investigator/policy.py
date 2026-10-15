# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The team's policy (policy/helpdesk.json): stated to the model, and enforced here in code.

The model is told every rule in its system prompt. When an action breaks a rule, the tool returns an error
that names the rule, and nothing is recorded. Nothing here depends on which month is being investigated;
the baseline model, catalog, and ticket types come from the world.
"""

import json
from pathlib import Path

import world

POLICY_FILE = Path(__file__).resolve().parents[1] / "policy" / "helpdesk.json"
POLICY = json.loads(POLICY_FILE.read_text())
DEFAULT_WORLD = "model-switch"


def policy_for(scenario=DEFAULT_WORLD):
    """The policy as it applies in this world: Fernledger's file, with the world's company and baseline model."""
    s = world.spec(scenario)
    if s.get("authored"):
        return POLICY
    policy = json.loads(POLICY_FILE.read_text().replace("Fernledger", s["company"]))
    policy["quality_bar"]["baseline_model"] = s["baseline_model"]
    policy["quality_bar"]["baseline_reason"] = s["baseline_reason"]
    return policy


def policy_text(scenario=DEFAULT_WORLD):
    p = policy_for(scenario)
    q, pr, r = p["quality_bar"], p["proposals"], p["report"]
    return "\n".join([
        f"- Quality bar: {q['rule']} Baseline model: {q['baseline_model']}. {q['baseline_reason']}",
        f"- Proposals: {pr['rule']}",
        f"- Report: {r['rule']}",
        f"- Manager summary: {r['manager_summary_rule']} Terms: {', '.join(r['manager_summary_banned_terms'])}.",
    ])


def quality_floor(ticket_type, scenario=DEFAULT_WORLD):
    """The lowest offline-eval resolution rate a proposed model may have for this ticket type."""
    q = policy_for(scenario)["quality_bar"]
    return world.pass_pct(scenario, q["baseline_model"], ticket_type) - q["max_drop_pct_points"]


def check_proposal(routes, state, scenario=DEFAULT_WORLD):
    """Return why a routing proposal breaks the policy, or None if it may be filed."""
    policy = policy_for(scenario)
    if len(state["proposals"]) >= policy["proposals"]["max_per_run"]:
        return (f"Rejected by policy: proposal {state['proposals'][-1]['proposal_id']} is already pending review; "
                "only one proposal per investigation is allowed. Nothing was recorded.")
    types = world.ticket_types(scenario)
    missing = [t for t in types if not routes.get(t)]
    if missing:
        return (f"Rejected: routes must name a catalog model for every ticket type (missing or unknown: {missing}). "
                "Nothing was recorded.")
    q = policy["quality_bar"]
    rate = {(m, t): world.pass_pct(scenario, m, t) for m in world.catalog(scenario) for t in types}
    below = [f"{t} on {m} resolves {rate[(m, t)]:.1f}% offline, below the bar of "
             f"{rate[(q['baseline_model'], t)] - q['max_drop_pct_points']:.1f}%"
             for t, m in routes.items() if (m, t) in rate and rate[(m, t)] < rate[(q["baseline_model"], t)] - q["max_drop_pct_points"]]
    if below:
        return "Rejected by the quality-bar policy: " + "; ".join(below) + ". Nothing was recorded."
    return None


def check_report(report, state, scenario=DEFAULT_WORLD):
    """Policy checks on a submitted report: the proposal ID, and a plain-language manager summary."""
    r, problems = policy_for(scenario)["report"], []
    filed = [p["proposal_id"] for p in state["proposals"]]
    given = str(report.get("proposal_id", "")).strip()
    if (given.lower() != "none" or filed) and given not in filed:
        problems.append(f"proposal_id {given!r} does not match a filed proposal ({filed or 'none filed'}); "
                        "use the ID returned by propose_change, or \"none\" if you filed nothing")
    summary = str(report.get("manager_summary", ""))
    hits = [term for term in r["manager_summary_banned_terms"] if term in summary.lower()]
    if hits:
        problems.append(f"manager_summary uses terms a product manager may not know ({', '.join(hits)}); "
                        "rewrite only the manager_summary in plain language")
    if len(summary.split()) > r["manager_summary_max_words"]:
        problems.append(f"manager_summary is {len(summary.split())} words; keep it to {r['manager_summary_max_words']} or fewer")
    return problems
