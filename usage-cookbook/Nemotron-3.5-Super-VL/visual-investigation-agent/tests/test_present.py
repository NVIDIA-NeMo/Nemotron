# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reader-facing claims name actors and sources, and presentation aids stay honest."""

import world

from investigator import load_recording, present
from investigator import highlights
from evals import checks

VIEWS = [present.show_visual_triage, present.show_missing_evidence,
         present.show_reasoned_decision, present.show_agent_view]


def test_reader_views_render_without_ambiguous_checks():
    run = load_recording("model-switch-visual")
    for view in VIEWS:
        html = view(run)._repr_html_()
        assert "Not seen in this run" not in html, view.__name__
        assert "✓" not in html, view.__name__
        assert "agent_version" not in html and "config_hash" not in html, view.__name__


def test_bridge_parts_add_up():
    split = checks.exact_bridge("model-switch-visual")
    f = split["factors"]
    drivers = [{"metric": "Tickets", "before": f["tickets"][0], "after": f["tickets"][1]},
               {"metric": "Tokens per ticket", "before": f["tokens_per_ticket"][0], "after": f["tokens_per_ticket"][1]},
               {"metric": "Price per 1M tokens", "before": f["effective_price_per_million"][0], "after": f["effective_price_per_million"][1]}]
    bridge = present.bridge(drivers)
    assert abs(bridge["before"] + sum(v for _, v in bridge["parts"]) - bridge["after"]) < 1e-6
    assert [k for k, _ in bridge["parts"]] == ["volume", "usage", "price"]
    decision = present.show_reasoned_decision(load_recording("model-switch-visual"))._repr_html_()
    assert "Super VL did not see this chart" in decision


def test_recorded_annotations_are_fixed_reader_aids():
    from PIL import Image

    run = load_recording("model-switch-visual")
    metadata = highlights.load(run)
    by_description = {run.image_description(i): boxes for i, boxes in metadata.items()}
    assert any(box["id"] == "spend" for box in by_description["screenshot attached by the user"])
    assert any(box["id"] == "vesk-lite-quality" for box in by_description["eval-report-2026-08, page 1"])
    assert all(box["width"] > 0 and box["height"] > 0 and box["x"] >= 0 and box["y"] >= 0
               for boxes in metadata.values() for box in boxes)
    assert highlights.LABEL in present.show_visual_triage(run)._repr_html_()
    assert "drawn after the run" in highlights.LABEL and "unmarked" in highlights.LABEL
    assert highlights.annotated(run, "img-0").size == Image.open(run.image("img-0")).size
    assert "The screenshot pasted with the request" in present.show_visual_triage(run)._repr_html_()


def test_capability_claims_include_verbatim_recorded_evidence():
    run = load_recording("model-switch-visual")
    triage = present.show_visual_triage(run)._repr_html_()
    navigation = present.show_missing_evidence(run)._repr_html_()
    decision = present.show_reasoned_decision(run)._repr_html_()
    order = {triage: ("What it has", "What it sees", "What it says", "What it does next"),
             navigation: ("What it has", "What it looks at", "What it says", "What it opens", "What it says next",
                          "What it does next"),
             decision: ("What it has", "What it does next", "What it says", "What happens next")}
    for html, labels in order.items():
        steps = [html.index(label) for label in labels]
        assert steps == sorted(steps), labels
        assert "Replay · recorded run" in html
    assert "The Models tab gives a visual but I need exact" in navigation
    assert "Why it opened this:" in navigation
    assert "What it says, before it touches any data" in triage and "What it does next" in triage
    assert "so the driver is a sharp rise in tokens per ticket" in triage
    assert triage.index("<code>list_dashboards</code>") < triage.index("<code>run_sql</code>")
    assert "say what the screenshot tells it before its first tool call" in triage
    assert "I need page 2 for tokens per ticket" in navigation
    assert highlights.LABEL in navigation
    assert "token counts per ticket are on page 2" in navigation
    assert "writes Python code using each highlighted value" in navigation
    assert "The three tables below pull out the exact price, quality, and token values" in navigation
    assert "Offline quality check, page 1" in navigation
    assert "Resolves offline" in navigation and "Required" in navigation and "Result" in navigation
    assert "92.1%" in navigation and "91.0%" in navigation and "Pass" in navigation
    assert "the models in the final proposal" in navigation and "add Osric 2 Thinking, the model in use now, for comparison" in navigation
    assert "types every one of these numbers into its Python" not in navigation
    assert "models it goes on to choose in Moment 3" not in navigation
    assert "The driver is a step-up in units used per ticket" in decision and "writes “units” here for tokens" in decision
    assert "It applies that rule to the candidate models and submits one proposal for review." in decision
    assert "It cannot approve or apply it." in decision
    assert "A person decides whether to approve and apply the change. Nothing has changed." in decision
    assert "Our check after the run." not in decision
    assert "Approve proposal PRP-0001" in decision and "$102,583" in decision
    assert "What it says caused the change" in decision and "rollout completed (100% of tickets)" in decision
    assert "former model, Vesk 4, as the quality baseline" in decision
    assert "offline test result is no more than 2 percentage points below that baseline" in decision
    assert "choose the cheapest model" in decision and "only one proposal in this run" in decision
    assert "The rule it must follow" not in decision and "Super VL's recommendation" not in decision


def test_recorded_errors_are_disclosed_with_grounded_corrections():
    run = load_recording("model-switch-visual")
    assert "at a higher output price" in run.report["how_it_happened"]
    prices = world.current_prices(run.scenario)
    assert prices["Osric 2 Thinking"]["output"] < prices["Vesk 4"]["output"]
    html = present.show_reasoned_decision(run)._repr_html_()
    full_run = present.show_agent_view(run)._repr_html_()
    for view in (html, full_run):
        assert "Two slips in the report's wording; the correctness of the proposal is unaffected." in view
        assert "at a higher output price” (“units” means tokens)" in view
        assert "Osric's output price is lower: $1.90 per 1M tokens, against $7.60 for Vesk 4" in view
        assert "“from $0.0542 to $0.0290, saving about $102.6k per month.”" in view
        assert "($0.0616 − $0.0290) × 104,969 tickets a day × 30 days ≈ $103k a month" in view
        assert "Starting from $0.0542 would give about $79k." in view
        assert "Super VL computed the proposal in code from the correct prices and the $0.0616 baseline" in view
        assert "our recheck against the source data confirms its routes and savings" in view
        assert "We show the report unedited, slips included." in view
        assert "How to read this notebook replay" not in view
    assert html.index("What happens next") < html.index("Check the report against the data") < html.index("Two slips in")
    assert html.count("Two slips in the report's wording") == 1
    assert "the recorded report is shown unedited" in html and "Super VL did not see this check" in html
    assert "output price is lower" in html and "$1.90" in html and "$7.60" in html
    assert "matches" in html and "other reasonable base periods give" in html and "$102,583" in html
    rows = present._figure_checks(run)
    assert [r[0].split("<br>")[0] for r in rows] == ["AI spend", "Cost per ticket", "Resolution rate", "Monthly savings"]
    assert all(r[-1] in ("matches", "within range") for r in rows)
    assert "The model in use today would fail the quality bar" in html
    assert "nobody checked the rollout against it" in html


def test_decision_exposes_routes_assumptions_and_validator_boundaries():
    run = load_recording("model-switch-visual")
    html = present.show_reasoned_decision(run)._repr_html_()
    assert all(ticket_type in html for ticket_type in (
        "Account &amp; password", "Billing", "Technical troubleshooting", "Integrations &amp; API"))
    assert "Assumes the Sep 10–21 volume" in html and "token counts hold for 30 days" in html
    assert "clears the bar by 1.1 to 1.2 points" in html
    assert "Whether the routing is the cheapest" in html
    assert "figures and explanations agree with the data" in html
    assert "the chosen model is the cheapest one that meets the bar. The policy does not check this" in html
    assert "log-mean Divisia index (LMDI)" in html
    assert "Check the proposed routes and savings math" in html
    assert "Check the report against the data" in html
    assert "Inspect the policy and every candidate model" in html
    assert "Route fields from the same response" not in html


def test_agent_view_shows_the_run_as_a_conversation():
    run = load_recording("model-switch-visual")
    html = present.show_agent_view(run)._repr_html_()
    for header in ("Moment 1 · Read the dashboard", "Moment 2 · Find the missing evidence", "Moment 3 · Decide"):
        assert header in html
    assert html.index("Moment 1") < html.index("Moment 2") < html.index("Moment 3")
    assert "Our help-desk AI bill is up about 30%" in html and "· step 1 · 0:11" in html
    assert "The Models tab gives a visual but I need exact" in html
    assert "Opens the scanned price list" in html and "Runs Python in the sandbox" in html
    assert "Call 1 ▸" in html and "Call 18 ▸" in html and ">Catalog<" in html and ">Python<" in html
    assert "I need page 2 for tokens per ticket" in html
    assert "PRP-0001" in html and "waiting for a person to approve it; nothing has changed" in html
    assert "8 steps · 18 tool calls" in html and "errors and rejections: 0" in html
    assert "▶ Play the run" in html and "agentViewSeek" in html and html.count("class='av-item'") == 14
    live = present.show_live_step(run)._repr_html_()
    assert "Live investigation · step 8" in live and "Submits its report" in live

def test_run_provenance_distinguishes_live_trace_from_synthetic_data():
    run = load_recording("model-switch-visual")
    replay = present.show_setup(False, run.scenario)._repr_html_()
    assert "captured during a live Super VL run against the synthetic scenario" in replay
    assert "saved model replies and tool results" in replay


def test_exact_bridge_uses_unrounded_factors():
    split = checks.exact_bridge("model-switch-visual")
    before, after = split["factors"]["effective_price_per_million"]
    assert round(100 * (after / before - 1), 1) == -49.6
    assert abs(split["before"] + sum(value for _, value in split["parts"]) - split["after"]) < 1e-6


def test_projection_shows_baseline_and_reproduces_recorded_savings():
    run = load_recording("model-switch-visual")
    projection = checks.proposal_projection(run)
    assert round(projection["current_monthly_spend"]) == 193964
    assert round(projection["proposed_monthly_spend"]) == 91380
    assert round(projection["monthly_savings"]) == 102583
    assert round(projection["savings_pct"], 1) == 52.9


def test_standalone_build_guide_links_and_implementation():
    html = present.show_build_guide()._repr_html_()
    assert "The eight tools" in html and "Run it" not in html
    assert "def investigate" in html
    assert "Complete tool schemas" in html
    assert "Exact initial system/user messages" in html
    assert "Recorded model settings and capture metadata" in html
    assert "The playbook this run saw" in html
    assert "eval-report-2026-08" in html
    assert "eval_results(model, ticket_type" not in html
    assert "def validate_report" in html and "drivers is empty" in html
    assert "def check_proposal" in html and "def check_report" in html
    assert "README.html" not in html and "evaluation-summary" not in html


def test_request_points_to_the_exact_implementation_section():
    run = load_recording("model-switch-visual")
    html = present.show_request(run.scenario, False, run)._repr_html_()
    assert "change anything yet.”" in html
    assert "dashboard shows.”" not in html
    assert html.index("change anything yet.”") < html.index("help-desk dashboard shows.")
    assert "Near the bottom of the notebook" in html
    assert "exact request, playbook, policy, tool schemas, and model settings used for this run" in html
    assert "shows every word" not in html


def _reworded(run):
    """The recording with plain evidence, an attribution field, and no note about page 2: another run's report shape."""
    import copy

    from investigator.config import DEFAULT, config_hash
    from investigator.runs import Run

    result = copy.deepcopy(run.result)
    report = result["report"]
    report["evidence"] = [{"claim": e["claim"], "source": e["source"]} for e in report["evidence"]]
    report["attribution"] = {"change_ids": ["CHG-5120", "CHG-5124", "CHG-5129"], "documents": [],
                             "explanation": "The rollout moved every ticket type to the new model."}
    transcript = copy.deepcopy(run.transcript)
    for message in transcript:
        if message["role"] == "assistant" and isinstance(message.get("content"), str):
            message["content"] = message["content"].replace("page 2", "the next page")
    manifest = {**run.manifest, "agent_version": DEFAULT["name"], "config_hash": config_hash(DEFAULT)}
    return Run(transcript, run.tool_log, result, manifest, run.folder)


def test_a_differently_worded_report_renders_with_the_agent_configuration():
    run = _reworded(load_recording("model-switch-visual"))
    assert present._config(run)["name"] == "agent"
    for view in VIEWS + [present.show_build_guide]:
        html = view(run)._repr_html_()
        assert "This run did not open" not in html and "filed no proposal" not in html, view.__name__
    decision = present.show_reasoned_decision(run)._repr_html_()
    assert "What it says caused the change" in decision and "Osric 2 Thinking rollout completed" in decision
    assert "three disclosed numerical errors" not in decision and "Check the report against the data" in decision
    guide = present.show_build_guide(run)._repr_html_()
    assert "Label each claim Observed" not in guide and "Break the change into its drivers" in guide


def test_figure_check_uses_each_drivers_stated_periods():
    import copy

    from investigator.runs import Run

    run = load_recording("model-switch-visual")
    result = copy.deepcopy(run.result)
    result["report"]["drivers"] = [
        {"metric": "cost per ticket", "before": 0.043, "after": 0.0616, "period": "Aug 1-21 (Vesk 4) vs Sep 10-21 (all Osric)"},
        {"metric": "AI spend", "before": 94384.22, "after": 165053.22, "period": "Aug 1-21 vs Sep 1-21"},
    ]
    rows = present._figure_checks(Run(run.transcript, run.tool_log, result, run.manifest, run.folder))
    assert [r[-1] for r in rows[:2]] == ["matches", "differs from the data"]
