# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated worlds, the oracle, and the segment scorer: deterministic, consistent, and strict where it matters."""

import copy
import json

import pytest

import world
from evals import segments
from evals.truth import kpi_cards, truth
from investigator import check_config
from investigator import config as agent_config
from investigator.contract import request_for, system_prompt
from investigator.policy import check_proposal
from investigator.runs import Run
from investigator.tools import TOOLS, tools_for, verify_figures
from world import generator

WORLDS = generator.pick("dev", ["model_switch", "price", "volume", "mix", "volume_price", "none"], 6)


def test_generation_is_deterministic_and_named_by_family_and_seed():
    for name in WORLDS:
        family, seed = name[2:name.rindex("-")], int(name[name.rindex("-") + 1:])
        a, b = generator.generate(family, seed), generator.generate(family, seed)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert a["name"] == name and a["split"] == "dev"


def test_generated_worlds_share_no_names_with_the_tuned_fernledger_month():
    authored = {"Fernledger", "Larkfield Cloud", "Vesk", "Osric", "Dunlin"}
    for name in generator.pick("dev", generator.FAMILIES[:-1], 30):
        text = json.dumps(world.spec(name))
        assert not any(word in text for word in authored), name


def test_sealed_family_is_not_in_development_picks():
    families = ["model_switch", "price", "volume", "mix", "volume_price", "none"]
    assert not any("retry_bug" in n for n in generator.pick("dev", families, 60) + generator.pick("validation", families, 60))


@pytest.mark.parametrize("name", WORLDS)
def test_oracle_routing_meets_the_policy_and_is_cheapest(name):
    t = truth(name)
    assert check_proposal(t["optimal"], {"proposals": []}, name) is None
    for ticket_type, model in t["optimal"].items():
        cost = t["unit_cost"][f"{model}|{ticket_type}"]
        for other in t["models"]:
            if world.pass_pct(name, other, ticket_type) >= t["floors"][ticket_type]:
                assert cost <= t["unit_cost"][f"{other}|{ticket_type}"] + 1e-12


@pytest.mark.parametrize("name", WORLDS)
def test_family_signature_is_visible_in_the_data(name):
    t = truth(name)
    c, fam = t["change_pct"], t["family"]
    if fam == "none":
        assert abs(c["spend"]) < 6
    else:
        assert c["spend"] > 10
    if fam == "model_switch":
        assert c["tokens_per_ticket"] > 30 and c["price_per_1m"] < -10
    if fam in ("price", "volume_price"):
        assert c["price_per_1m"] > 10
    if fam in ("volume", "volume_price"):
        assert c["tickets"] > 10


def test_the_default_is_the_agent():
    assert agent_config.DEFAULT is agent_config.AGENT and tools_for("model-switch") == TOOLS
    for s in world.SCENARIOS:
        assert system_prompt(s) == system_prompt(s, None, agent_config.AGENT)
    assert all(check_config().values())


def test_request_modes_differ_only_in_the_clue():
    name = WORLDS[0]
    neutral, hint, flip = (request_for(name, m) for m in ("neutral", "hint", "flip"))
    assert "cheaper model" not in neutral and neutral != flip and neutral != hint


def _run(name, report, proposals=None, code="", views=(), sql_results=(), note=""):
    """A minimal saved run: one assistant step with the given calls."""
    calls, tool_msgs = [], []
    for i, (doc, page) in enumerate(views):
        calls.append({"id": f"v{i}", "type": "function", "function": {"name": "view_document", "arguments": json.dumps({"doc_id": doc, "page": page, "why": "x"})}})
        tool_msgs.append({"role": "tool", "tool_call_id": f"v{i}", "content": json.dumps({"status": "opened"})})
    for i, result in enumerate(sql_results):
        calls.append({"id": f"s{i}", "type": "function", "function": {"name": "run_sql", "arguments": json.dumps({"query": "q", "why": "x"})}})
        tool_msgs.append({"role": "tool", "tool_call_id": f"s{i}", "content": json.dumps(result)})
    calls.append({"id": "p0", "type": "function", "function": {"name": "run_python", "arguments": json.dumps({"code": code, "why": "x"})}})
    tool_msgs.append({"role": "tool", "tool_call_id": "p0", "content": json.dumps({"stdout": ""})})
    transcript = [{"role": "system", "content": ""}, {"role": "user", "content": "q"},
                  {"role": "assistant", "content": note, "tool_calls": calls}, *tool_msgs]
    result = {"scenario": name, "report": report, "proposals": proposals or [], "error": None, "turn_log": [{"seconds": 1, "finish_reason": "tool_calls"}],
              "nudges": 0, "truncations": 0, "rejected_reports": 0, "why_missing": 0, "image_budget_rejections": 0, "rejected_proposals": 0}
    return Run(transcript, [], result, {}, None)


def _oracle_run(name, **overrides):
    t = truth(name)
    s = world.spec(name)
    (b0, b1), (k0, k1) = t["spend"], t["tickets"]
    cause = {"model_switch": f"Production moved to {t['new_model']}, which uses far more reasoning tokens per ticket.",
             "price": f"{t['provider']} raised the price of {t['baseline_model']} in the new price schedule.",
             "volume": "Ticket volume grew by more than 10% after a launch.", "mix": "The ticket mix shifted toward harder tickets.",
             "volume_price": "More tickets grew volume and a price increase in the new price schedule raised the price.",
             "none": "No material change: spend moved in line with normal variation."}[t["family"]]
    cites = " ".join(t["causal_changes"] + t["causal_documents"])
    pct = lambda pair: 100 * (pair[1] / pair[0] - 1)
    report = {"finding": f"{cause} Spend changed {pct(t['spend']):.1f}% and tickets {pct(t['tickets']):.1f}%. See {cites}.",
              "how_it_happened": cause, "drivers": [{"metric": "AI spend", "before": b0, "after": b1, "unit": "USD", "period": "p", "source": "sql"}],
              "evidence": [{"claim": cause, "label": "Observed", "source": cites or "sql"}], "ruled_out": [], "confidence": "high",
              "key_figures": [], "quality_impact": "", "recommendation": "", "proposal_id": "none", "manager_summary": "m", "engineer_summary": "e"}
    proposals = []
    if t["expected_proposal"]:
        rng = segments.savings_range(name, t["optimal"])
        proposals = [{"proposal_id": "PRP-0001", "routes": t["optimal"], "expected_monthly_savings_usd": (rng[0] + rng[1]) / 2}]
        report["proposal_id"] = "PRP-0001"
    prices = " ".join(f"{p['input']} {p['output']}" for p in t["needed_prices"].values())
    evals = " ".join(str(v) for v in t["needed_eval"].values())
    views = [(t["current_schedule"], 1)] + ([(t["eval_document"], 1), (t["eval_document"], 2)] if t["eval_document"] else [])
    sql = [{"columns": ["period", "total_usd"], "rows": [["a", b0], ["b", b1]]}]
    cards = kpi_cards(name)
    note = f"The screenshot shows AI spend {cards[0][1]} ({cards[0][2]}) and tickets {cards[1][1]}." if t["screenshot"] else ""
    kwargs = dict(report=report, proposals=proposals, code=f"x = [{prices} {evals}]", views=views, sql_results=sql, note=note)
    kwargs.update(overrides)
    return _run(name, **kwargs)


@pytest.mark.parametrize("name", WORLDS)
def test_an_oracle_run_passes_every_segment_that_applies(name):
    s = segments.score(_oracle_run(name))
    failed = [k for k in segments.SEGMENTS if s[k] is False]
    assert not failed, (failed, {k: s["why"].get(k) for k in failed})
    assert s["e2e"] and s["unassisted"]


def test_blaming_a_decoy_fails_attribution():
    name = next(n for n in WORLDS if truth(n)["family"] == "price")
    run = _oracle_run(name)
    run.result["report"]["finding"] += f" The cost rose because of {truth(name)['decoys'][0]}."
    assert segments.score(run)["S4"] is False


def _family(family):
    return next(n for n in WORLDS if truth(n)["family"] == family)


def test_naming_a_decoy_to_clear_it_is_not_blaming_it():
    name = _family("price")
    decoy = truth(name)["decoys"][0]
    run = _oracle_run(name)
    run.result["report"]["how_it_happened"] += f" The change-log entries around that date ({decoy}) are unrelated wording changes."
    run.result["report"]["ruled_out"] = [{"candidate": f"Change-log entry {decoy}", "reason": "content only"}]
    assert segments.score(run)["S4"] is True


def test_a_world_with_two_causes_needs_both_cited():
    name = _family("volume_price")
    t = truth(name)
    run = _oracle_run(name)
    report = run.result["report"]
    for field in ("finding", "how_it_happened"):
        report[field] = report[field].replace(" ".join(t["causal_changes"] + t["causal_documents"]), t["causal_documents"][0])
    report["evidence"][0]["source"] = t["causal_documents"][0]
    s = segments.score(run)
    assert s["S4"] is False and s["why"]["S4"]["change_cited"] is False and s["why"]["S4"]["document_cited"] is True


def test_a_hedged_attribution_cites_the_change_but_clearing_it_does_not():
    name = _family("volume")
    change = truth(name)["causal_changes"][0]
    hedged = _oracle_run(name)
    hedged.result["report"]["finding"] = f"Tickets grew 30%, most plausibly tied to {change}, a hypothesis, not proven."
    hedged.result["report"]["evidence"][0]["source"] = "sql"
    assert segments.score(hedged)["S4"] is True
    cleared = _oracle_run(name)
    cleared.result["report"]["finding"] = f"Tickets grew 30%; {change} is unrelated."
    cleared.result["report"]["evidence"][0]["source"] = "sql"
    assert segments.score(cleared)["S4"] is False


def test_projected_figures_are_not_observed_drivers():
    name = _family("none")
    run = _oracle_run(name)
    run.result["report"]["drivers"].append({"metric": "Projected cost per ticket (proposal)", "before": 0.04, "after": 0.017,
                                            "unit": "USD", "period": "p", "source": "python"})
    assert segments.score(run)["S3"] is True


def test_a_price_raise_naming_the_model_is_a_price_claim():
    assert segments.PRICE_CLAIM.search("Northgate Models raised Merrow 2 prices effective Aug 1")


def test_a_per_driver_attribution_cites_each_listed_change():
    name = _family("volume_price")
    t = truth(name)
    run = _oracle_run(name)
    report = run.result["report"]
    for field in ("finding", "how_it_happened"):
        report[field] = "Spend rose."
    report["evidence"][0]["source"] = "sql"
    report["attribution"] = [
        {"driver": "price", "started": "x", "change_ids": [], "documents": t["causal_documents"], "explanation": "The new schedule."},
        {"driver": "tickets", "started": "x", "change_ids": t["causal_changes"], "documents": [], "explanation": "The launch."}]
    s = segments.score(run)
    assert s["why"]["S4"]["change_cited"] and s["why"]["S4"]["document_cited"]


def test_a_price_per_million_tokens_driver_is_a_price():
    rep = {"drivers": [{"metric": "price per 1M tokens", "before": 2.33, "after": 3.03},
                       {"metric": "Reasoning tokens per day", "before": 1, "after": 2}]}
    assert [c for c, _, _ in segments._drivers(rep)] == ["price", "tokens"]


def test_ruling_out_other_models_prices_does_not_rule_out_the_cause():
    name = _family("price")
    run = _oracle_run(name)
    run.result["report"]["ruled_out"] = [{"candidate": f"Price changes for models other than {truth(name)['baseline_model']}",
                                          "reason": "unchanged"}]
    assert segments.score(run)["S9"] is True


def test_a_token_total_growing_faster_than_tickets_shows_the_switch():
    name = _family("model_switch")
    t = truth(name)
    run = _oracle_run(name)
    run.result["report"]["finding"] = run.result["report"]["how_it_happened"] = f"Production moved to {t['new_model']}."
    run.result["report"]["drivers"].append({"metric": "Reasoning tokens per day", "before": 3.7e7, "after": 2.8e9,
                                            "unit": "tokens", "period": "p", "source": "sql"})
    assert segments.score(run)["S3"] is True


def test_a_compliant_but_not_cheapest_routing_fails_decide_but_counts_as_valid():
    name = next(n for n in WORLDS if truth(n)["expected_proposal"])
    t = truth(name)
    routes = {k: t["baseline_model"] for k in t["optimal"]}
    if routes == t["optimal"]:
        pytest.skip("baseline is optimal here")
    run = _oracle_run(name, proposals=[{"proposal_id": "PRP-0001", "routes": routes, "expected_monthly_savings_usd": 1.0}])
    s = segments.score(run)
    assert s["S8"] is False and not s["e2e"]


def test_reporting_a_misbucketed_total_fails_reconcile():
    name = WORLDS[1]
    run = _oracle_run(name)
    run.result["report"]["key_figures"] = [{"name": "spend", "value": truth(name)["misbucketed"][0]}]
    assert segments.score(run)["S2"] is False


def test_verify_figures_recomputes_from_the_database():
    name = WORLDS[2]
    s = world.spec(name)
    conn = world.connect(name)
    t = truth(name)
    out = verify_figures(conn, {"figures": [{"metric": "spend", "value": t["spend"][1], "start_date": s["period"][0], "end_date": s["period"][1]},
                                            {"metric": "tickets", "value": 1, "start_date": s["period"][0], "end_date": s["period"][1]}]})
    assert [f["matches"] for f in out["figures"]] == [True, False]


def test_optional_tools_appear_only_when_switched_on(tmp_path):
    from investigator.tools import notes, zoom_image

    plain = {t["function"]["name"] for t in tools_for(WORLDS[0])}
    extended = {t["function"]["name"] for t in tools_for(WORLDS[0], agent_config.make({"tools": {"notes": True, "zoom": True, "check_report": True}}))}
    assert extended - plain == {"notes", "zoom_image", "verify_figures"}
    state = {}
    assert notes({"text": "plan"}, state)["notes"] == "plan"
    assert notes({"text": "more", "mode": "append"}, state)["notes"] == "plan\nmore"
    from PIL import Image

    source = tmp_path / "a.png"
    Image.new("RGB", (400, 200), "white").save(source)
    result, image = zoom_image({"image_id": "img-1", "left": 0.1, "top": 0.1, "width": 0.5, "height": 0.5},
                               {"image_paths": {"img-1": str(source)}}, tmp_path / "zooms")
    assert result["status"] == "zoomed" and Image.open(image[0]).size[0] > 200
    bad, _ = zoom_image({"image_id": "img-1", "left": 0.8, "top": 0, "width": 0.5, "height": 0.5}, {"image_paths": {"img-1": str(source)}}, tmp_path)
    assert "error" in bad
