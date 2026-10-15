# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in model that replays a recording's replies, to test the harness and tools without an endpoint.

This is plumbing, not model evidence: it proves the harness and tools reproduce the recorded run exactly.
"""

import json
import shutil

import world
from investigator.harness import _for_log
from world.dashboard import render_path


def seed_images(run):
    """Put the recording's images where the world would render them, so no browser is needed."""
    scenario = run.scenario
    world.build(scenario, images=False)
    folder = world.WORLD / scenario
    for item in run.result["images"]:
        desc, source = item["description"], run.image(item["image_id"])
        if desc.startswith("screenshot"):
            target = render_path(scenario, world.page(scenario, "Overview")["description"])
        elif desc.startswith(("Overview", "Models", "Changes")):
            target = render_path(scenario, desc)
        else:
            doc_id, page = desc.split(", page ")
            name = f"{doc_id}-p{page}.png" if doc_id == world.EVAL_REPORT["id"] else f"{doc_id}.jpg"
            target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


CLEARED = "[older image cleared to stay within the image limit; request it again if needed]"


def _slots(messages):
    """Messages with every image, sent or later cleared to fit the image limit, as the same placeholder.

    The saved transcript is the conversation's final state, after the harness cleared the oldest images; an earlier
    request still carried them. Image bytes are never logged, so only the image's position is compared."""
    slot = {"type": "image slot"}
    return [{**m, "content": [slot if p.get("type") == "image" or p.get("text") == CLEARED else p for p in m["content"]]}
            if isinstance(m.get("content"), list) else m for m in messages]


class RecordedModel:
    """Returns the recording's replies in order and checks each request against the recorded conversation."""

    def __init__(self, run):
        self.transcript = run.transcript
        self.replies = [m for m in run.transcript if m["role"] == "assistant"]
        self.timing = [t for t in run.result["turn_log"] if t["finish_reason"] != "length"]
        self.calls = 0
        self.mismatches = []

    def __call__(self, messages, **_):
        sent = _for_log(messages)
        if json.dumps(_slots(sent)) != json.dumps(_slots(self.transcript[:len(sent)])):
            self.mismatches.append(self.calls)
        reply = self.replies[self.calls]
        seconds = self.timing[self.calls]["seconds"] if self.calls < len(self.timing) else 0.0
        self.calls += 1
        message = {"content": reply["content"], "tool_calls": reply.get("tool_calls")}
        return {"choices": [{"message": message, "finish_reason": "tool_calls"}], "usage": {}}, seconds
