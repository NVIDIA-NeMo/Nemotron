# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness: the loop that alternates model calls and tool calls, with limits, and saves every step.

The model decides what to look at, what to run, and when it is done. The harness runs the tools it asks for,
enforces the limits, and records the run so it can be replayed. Its only messages to the model are the fixed
reminders below; it never suggests a next step.
"""

import base64
import copy
import datetime as dt
import json
import re
import shutil
import time
from pathlib import Path

import world

from . import endpoint
from .config import config_hash, from_ablate
from .contract import request_for, system_prompt, validate_report
from .policy import check_report
from .tools import ACTION_TOOLS, execute, tools_for
from .version import run_manifest

MAX_TURNS = 40
MAX_NUDGES = 2
MAX_TOKENS = 16384
NUDGE = "Continue with tools. When you are finished, call submit_report."
TOOL_AS_TEXT = "Your last reply wrote a tool call as text, so it was not run. Send your next step as a tool call."
TOOL_MARKUP = re.compile(r"<invoke|<function[=\s]|<tool_call|<parameter")
TRUNCATED = ("Your last response hit the output token limit and was discarded before any tool ran. "
             "Reason more briefly, then resend the tool call.")
BLIND = "Image unavailable: this run receives no images."
REVIEW = ("Before this report is final, check it against the data. For each driver that moved, find the day it started to move "
          "and the change-log entries in the days just before it, and make sure the report's explanation of what caused each "
          "driver agrees with what you find. Then call submit_report again with the final report, changed or not.")
RUNS = Path(__file__).resolve().parents[1] / "runs"


def investigate(scenario, run_dir=None, ablate=None, chat=None, on_step=None, config=None):
    """Investigate one month, starting from the user's request. Saves the run and returns its folder."""
    session = Session(scenario, ablate, chat or endpoint.chat, config or from_ablate(ablate))
    for turn in range(MAX_TURNS):
        reply = session.ask_model(turn)
        if reply is None:  # the endpoint failed
            break
        if reply["finish_reason"] == "length":  # cut off before any tool ran
            if not session.retry_truncated():
                break
        elif not reply["tool_calls"]:  # no tool call: a fixed reminder, and the next reply must call a tool
            if not session.remind(reply["content"]):
                break
        else:
            session.run_tools(reply["tool_calls"], turn)
        if on_step:
            on_step(session.snapshot())
        if session.report is not None:
            break
    else:
        session.error = "turn limit reached"
    if session.report is None and session.draft is not None and session.error in (None, "turn limit reached",
                                                                                   "no submit_report after nudges"):
        session.report, session.error = session.draft, None  # the reviewed draft stands; any nudges are still counted
    return session.save(run_dir)


class Session:
    """Everything one investigation accumulates: messages, tool results, images, and counts."""

    def __init__(self, scenario, ablate, chat, config):
        world.ensure(scenario, images=True)
        self.scenario, self.ablate, self.chat, self.config = scenario, ablate, chat, config
        self.enforce = ablate != "policy"
        self.conn = world.connect(scenario)
        self.tools = tools_for(scenario, config)
        self.tool_log, self.turns, self.images = [], [], []
        request = request_for(scenario, config["request"])
        self.state = {"proposals": [], "image_paths": {}}
        if world.spec(scenario).get("attach_overview") and config["blind"]:
            request = [{"type": "text", "text": request + " Here's what our help-desk dashboard shows."},
                       {"type": "text", "text": BLIND}]
        elif world.spec(scenario).get("attach_overview"):  # the user pastes a dashboard screenshot
            shot = world.render(scenario, "Overview")
            self.images.append({"image_id": "img-0", "path": shot["path"], "description": "screenshot attached by the user"})
            self.state["image_paths"]["img-0"] = shot["path"]
            request = [{"type": "text", "text": request + " Here's what our help-desk dashboard shows."}, _image_part(shot["path"])]
        self.messages = [{"role": "system", "content": system_prompt(scenario, ablate, config)}, {"role": "user", "content": request}]
        self.image_budget = config["image_budget"] - len(_image_parts(self.messages[1]))
        self.zoom_dir = None
        self.report, self.error = None, None
        self.draft, self.review_due = None, False
        self.nudges = self.truncations = self.rejected_reports = 0
        self.why_missing = self.budget_rejections = self.forced_retries = 0
        self.force_tool = False
        self.started = time.monotonic()

    def ask_model(self, turn):
        """One model call. Returns the reply, or None if the endpoint failed."""
        _prune_images(self.messages, self.image_budget)
        sampling = self.config["sampling"]
        extra = {"tool_choice": "required"} if self.force_tool else {}
        if sampling["seed"] is not None:
            extra["seed"] = sampling["seed"] + turn
        try:
            payload, seconds = self.chat(self.messages, tools=self.tools, max_tokens=sampling["max_tokens"],
                                         temperature=sampling["temperature"], top_p=sampling["top_p"],
                                         thinking=sampling["thinking"], **extra)
        except Exception as exc:
            self.error = f"endpoint: {exc}"
            return None
        choice = payload["choices"][0]
        message, usage = choice["message"], payload.get("usage", {})
        calls = message.get("tool_calls") or []
        self.turns.append({"turn": turn, "seconds": round(seconds, 1), "finish_reason": choice.get("finish_reason"),
                           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                           "tool_calls": [c["function"]["name"] for c in calls]})
        reply = {"content": message.get("content") or "", "tool_calls": calls, "finish_reason": choice.get("finish_reason")}
        if reply["finish_reason"] != "length":
            assistant = {"role": "assistant", "content": reply["content"]}
            if calls:
                assistant["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in calls]
            self.messages.append(assistant)
            self.force_tool = False
        return reply

    def retry_truncated(self):
        self.truncations += 1
        if self.truncations > self.config["max_nudges"]:
            self.error = "output token limit hit repeatedly"
            return False
        self.messages.append({"role": "user", "content": TRUNCATED})
        return True

    def remind(self, content):
        if self.nudges >= self.config["max_nudges"]:
            self.error = "no submit_report after nudges"
            return False
        self.nudges += 1
        self.forced_retries += 1
        self.force_tool = True
        self.messages.append({"role": "user", "content": TOOL_AS_TEXT if TOOL_MARKUP.search(content) else NUDGE})
        return True

    def run_tools(self, calls, turn):
        """Run each tool call in order, then attach any images they returned in one user message."""
        attachments = []
        for call in calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                arguments, result, image = None, {"error": f"arguments are not valid JSON: {exc}"}, None
            else:
                result, image = self._run_one(name, arguments)
            if image and self.config["blind"]:
                result, image = {**result, "image": BLIND}, None
            if image and len(attachments) >= self.image_budget:
                self.budget_rejections += 1
                result, image = {"error": f"Image budget: a request to the model holds at most {self.config['image_budget']} images and "
                                          f"{self.config['image_budget'] - self.image_budget} came with the user's request, so at most "
                                          f"{self.image_budget} new images fit per turn. This image was not attached; "
                                          "request it in your next turn."}, None
            if image:
                image_id = f"img-{len(self.images) + 1}"
                self.images.append({"image_id": image_id, "path": image[0], "description": image[1]})
                self.state["image_paths"][image_id] = image[0]
                result = {**result, "image_id": image_id, "note": "The image is attached in the next message."}
                attachments.append((image_id, *image))
            self.tool_log.append({"turn": turn, "name": name, "arguments": arguments, "result": result})
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
        if attachments:
            content = []
            for image_id, path, desc in attachments:
                content += [{"type": "text", "text": f"{image_id}: {desc}"}, _image_part(path)]
            self.messages.append({"role": "user", "content": content})
        if self.review_due:
            self.review_due = False
            self.messages.append({"role": "user", "content": REVIEW})

    def _run_one(self, name, arguments):
        if self.config["why"] and name in ACTION_TOOLS and not str(arguments.get("why", "")).strip():
            self.why_missing += 1
            return {"error": f"{name} needs `why`: one sentence on what you have seen so far that leads to this "
                             "call, and what you expect to learn. Resend the call with it."}, None
        if name == "submit_report":
            problems = validate_report(arguments, self.config) or (check_report(arguments, self.state, self.scenario) if self.enforce else [])
            if problems:
                self.rejected_reports += 1
                return {"error": "submit_report rejected: " + "; ".join(problems) + ". Resubmit the complete report."}, None
            if self.config["review"] and self.draft is None:
                self.draft, self.review_due = arguments, True
                return {"status": "draft recorded; it becomes final when you call submit_report again"}, None
            self.report = self.report or arguments
            return {"status": "recorded"}, None
        if name == "zoom_image" and self.zoom_dir is None:
            import tempfile

            self.zoom_dir = Path(tempfile.mkdtemp(prefix="zoom-", dir=world.WORLD))
        return execute(self.scenario, self.conn, name, arguments, self.state, self.enforce, self.zoom_dir, self.config)

    def record(self):
        return {"scenario": self.scenario, "ablate": self.ablate, "config": self.config["name"],
                "config_hash": config_hash(self.config), "report": self.report, "proposals": self.state["proposals"],
                "error": self.error, "seconds": round(time.monotonic() - self.started, 1), "turns": len(self.turns),
                "nudges": self.nudges, "truncations": self.truncations, "rejected_reports": self.rejected_reports,
                "why_missing": self.why_missing, "image_budget_rejections": self.budget_rejections,
                "forced_tool_retries": self.forced_retries,
                "rejected_proposals": sum(1 for c in self.tool_log if c["name"] == "propose_change" and "error" in c["result"]),
                "tool_counts": {n: sum(1 for c in self.tool_log if c["name"] == n) for n in [t["function"]["name"] for t in self.tools]},
                "notes": self.state.get("notes"), "images": self.images, "turn_log": self.turns,
                **({"draft": self.draft} if self.config["review"] else {})}

    def snapshot(self):
        """The run so far, for a live timeline."""
        from .runs import Run
        return Run(_for_log(self.messages), self.tool_log, self.record(), None, None)

    def save(self, run_dir=None):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(run_dir or RUNS / f"{self.scenario}-{stamp}")
        (run_dir / "images").mkdir(parents=True, exist_ok=True)
        record = self.record()
        for image in record["images"]:
            name = f"{image['image_id']}{Path(image['path']).suffix}"
            shutil.copy(image["path"], run_dir / "images" / name)
            image["path"] = f"images/{name}"
        (run_dir / "transcript.json").write_text(json.dumps(_for_log(self.messages), indent=1))
        (run_dir / "tool_log.json").write_text(json.dumps(self.tool_log, indent=1))
        (run_dir / "result.json").write_text(json.dumps(record, indent=1))
        (run_dir / "manifest.json").write_text(json.dumps(run_manifest(self.scenario, self.ablate, stamp, self.config), indent=1))
        return run_dir


# Images in messages -------------------------------------------------------------------------------------

def _image_part(path):
    mime = "image/png" if str(path).endswith(".png") else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"}}


def _image_parts(message):
    content = message.get("content")
    return [j for j, p in enumerate(content) if p.get("type") == "image_url"] if isinstance(content, list) else []


def _prune_images(messages, budget):
    """Clear the oldest tool-returned images so each request fits MAX_IMAGES. The user's attached images
    (in messages[1]) always stay; `budget` is how many other images fit."""
    parts = [(i, j) for i, m in enumerate(messages) if i >= 2 for j in _image_parts(m)]
    for i, j in parts[:max(0, len(parts) - budget)]:
        messages[i]["content"][j] = {"type": "text", "text": "[older image cleared to stay within the image limit; request it again if needed]"}


def _for_log(messages):
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), list):
            m["content"] = [{"type": "image", "note": "image omitted from log"} if p.get("type") == "image_url" else p for p in m["content"]]
    return out
