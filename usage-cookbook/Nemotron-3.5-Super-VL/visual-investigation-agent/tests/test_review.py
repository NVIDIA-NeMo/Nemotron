# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional review step: the first report is a draft, one fixed review message follows, the second report is final."""

import json

from investigator import config as agent_config
from investigator.contract import report_schema
from investigator.harness import REVIEW, investigate
from investigator.runs import load_run

WORLD = "volume-growth"


def _report(finding):
    report = {k: "x" for k in report_schema(agent_config.AGENT)}
    report.update(finding=finding, proposal_id="none", manager_summary="Spend rose with volume.", ruled_out=[], evidence=[],
                  key_figures=[], drivers=[{"metric": "AI spend", "before": 1, "after": 2, "unit": "USD", "period": "p", "source": "sql"}],
                  attribution={"change_ids": [], "documents": [], "explanation": "x"})
    return report


def _scripted(replies):
    """A model that sends the given replies in order; each is a report to submit, or None for a reply with no tool call."""
    seen = []

    def chat(messages, tools=None, **kwargs):
        seen.append(messages[-1])
        reply = replies[len(seen) - 1]
        if reply is None:
            message = {"role": "assistant", "content": "done"}
        else:
            message = {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{len(seen)}", "type": "function", "function": {
                "name": "submit_report", "arguments": json.dumps(reply)}}]}
        return {"choices": [{"message": message, "finish_reason": "tool_calls" if reply else "stop"}], "usage": {}}, 0.1
    return chat, seen


def test_review_makes_the_first_report_a_draft_and_the_second_final(tmp_path):
    chat, seen = _scripted([_report("draft"), _report("final")])
    run = load_run(investigate(WORLD, run_dir=tmp_path / "r", chat=chat, config=agent_config.make({"review": True}, base=agent_config.AGENT)))
    assert run.report["finding"] == "final" and run.result["draft"]["finding"] == "draft"
    assert seen[1]["content"] == REVIEW
    assert run.result["nudges"] == 0 and run.result["error"] is None


def test_without_a_resubmission_the_draft_stands_and_the_nudges_count(tmp_path):
    chat, _ = _scripted([_report("draft"), None, None, None])
    run = load_run(investigate(WORLD, run_dir=tmp_path / "r", chat=chat, config=agent_config.make({"review": True}, base=agent_config.AGENT)))
    assert run.report["finding"] == "draft" and run.result["error"] is None and run.result["nudges"] == 2


def test_without_review_the_first_report_is_final(tmp_path):
    chat, _ = _scripted([_report("only")])
    run = load_run(investigate(WORLD, run_dir=tmp_path / "r", chat=chat, config=agent_config.AGENT))
    assert run.report["finding"] == "only" and "draft" not in run.result


def test_the_review_flag_does_not_change_earlier_configuration_hashes():
    assert agent_config.config_hash(agent_config.AGENT) == "06adac9ebd66d189"
    assert agent_config.config_hash(agent_config.make({"review": True}, base=agent_config.AGENT)) != "06adac9ebd66d189"


def test_blind_runs_receive_no_images(tmp_path):
    from investigator.harness import BLIND

    view = {"id": "v1", "type": "function", "function": {"name": "view_document", "arguments": json.dumps(
        {"doc_id": "price-schedule-2026-08", "why": "prices"})}}
    replies, seen = [view, None], []

    def chat(messages, tools=None, **kwargs):
        seen.append([m for m in messages])
        reply = replies[min(len(seen) - 1, 1)]
        calls = [reply] if reply else [{"id": f"s{len(seen)}", "type": "function", "function": {
            "name": "submit_report", "arguments": json.dumps(_report("blind"))}}]
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": calls}, "finish_reason": "tool_calls"}],
                "usage": {}}, 0.1

    blind = agent_config.make({"blind": True}, base=agent_config.AGENT)
    run = load_run(investigate("model-switch-visual", run_dir=tmp_path / "r", chat=chat, config=blind))
    images = [p for m in seen[-1] if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
    assert not images and not run.result["images"] and BLIND in json.dumps(seen[-1])
    assert agent_config.config_hash(agent_config.AGENT) == "06adac9ebd66d189"
