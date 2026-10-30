# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A saved run, recorded or live, read back as steps: what the agent said, what it called, and what came back.

A run folder holds transcript.json (every message except image bytes), tool_log.json, result.json (the report,
proposals, counts, and timing), manifest.json (the configuration hashes), and images/.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "replay"
STAGES = {"list_dashboards": "Evidence request", "render_dashboard": "Evidence request",
          "list_documents": "Evidence request", "view_document": "Evidence request",
          "run_sql": "Data tool", "run_python": "Code tool", "submit_report": "Report", "propose_change": "Proposal"}


@dataclass
class Call:
    name: str
    arguments: dict
    result: dict = field(default_factory=dict)

    @property
    def why(self):
        return str((self.arguments or {}).get("why", "")).strip()

    @property
    def stage(self):
        return STAGES.get(self.name, "")

    @property
    def image_id(self):
        return self.result.get("image_id")

    @property
    def error(self):
        return self.result.get("error")

    @property
    def action(self):
        """The call in plain words."""
        a, r = self.arguments or {}, self.result
        if self.name == "render_dashboard":
            dates = f", {a['start_date']} to {a['end_date']}" if a.get("start_date") and a.get("end_date") else ""
            return f"Model requested {a.get('tab')}{dates} → render tool returned an image"
        if self.name == "view_document":
            return f"Model requested {a.get('doc_id')}, page {a.get('page') or 1} → document tool returned the page"
        if self.name == "run_sql":
            return (f"Model wrote SQL → read-only data tool returned {len(r.get('rows', []))} rows"
                    if "rows" in r else "Model wrote SQL → read-only data tool returned an error")
        if self.name == "run_python":
            return "Model wrote Python → isolated code tool executed it"
        if self.name == "propose_change":
            return (f"Model submitted proposal {r.get('proposal_id')} → policy validator recorded it for review"
                    if r.get("proposal_id") else "Model submitted a proposal → policy validator rejected it")
        return {"list_dashboards": "Model requested dashboard catalog → tool returned it",
                "list_documents": "Model requested document catalog → tool returned it",
                "submit_report": "Model submitted the structured report"}.get(self.name, self.name)


@dataclass
class Step:
    number: int  # 1-based, one per model reply
    note: str  # the agent's visible text, never its private reasoning
    calls: list
    seconds: float = 0.0  # time the model took for this reply
    elapsed: float = 0.0  # time into the run when the reply arrived
    images: list = field(default_factory=list)  # image IDs attached after this step's calls
    harness: list = field(default_factory=list)  # messages the harness sent after this step


class Run:
    def __init__(self, transcript, tool_log, result, manifest, folder):
        self.transcript, self.tool_log, self.result = transcript, tool_log, result
        self.manifest, self.folder = manifest or {}, Path(folder) if folder else None
        self.steps = self._steps()

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        read = lambda name: json.loads((folder / name).read_text()) if (folder / name).exists() else None
        return cls(read("transcript.json"), read("tool_log.json"), read("result.json"), read("manifest.json"), folder)

    # What happened ------------------------------------------------------------------------------------

    @property
    def scenario(self):
        return self.result["scenario"]

    @property
    def report(self):
        return self.result.get("report") or {}

    @property
    def proposals(self):
        return self.result.get("proposals") or []

    @property
    def recorded(self):
        return self.folder is not None and REPLAY in self.folder.resolve().parents

    @property
    def label(self):
        minutes = self.result.get("seconds", 0) / 60
        if self.recorded:
            return f"Recorded run · {self.manifest.get('agent_version', '')} · {self.manifest.get('recorded', '')} · {minutes:.1f} min"
        return f"Live run · {self.manifest.get('agent_version', 'in progress')} · {minutes:.1f} min"

    @property
    def request(self):
        """The user's request text and the IDs of any images they attached."""
        content = self.transcript[1]["content"]
        if isinstance(content, str):
            return content, []
        text = " ".join(p["text"] for p in content if p.get("type") == "text")
        return text, [i["image_id"] for i in self.result.get("images", []) if i["description"].startswith("screenshot")]

    def image(self, image_id):
        """Path of an image the agent saw, by ID."""
        for item in self.result.get("images", []):
            if item["image_id"] == image_id:
                path = Path(item["path"])
                return path if path.is_absolute() or self.folder is None else self.folder / path
        return None

    def image_description(self, image_id):
        return next((i["description"] for i in self.result.get("images", []) if i["image_id"] == image_id), "")

    def calls(self, *names):
        """(step, call) pairs for the named tools, in order."""
        return [(s, c) for s in self.steps for c in s.calls if not names or c.name in names]

    def harness_messages(self):
        return [(s, m) for s in self.steps for m in s.harness]

    # Parsing ------------------------------------------------------------------------------------------

    def _steps(self):
        replies = [t for t in self.result.get("turn_log", []) if t.get("finish_reason") != "length"]
        steps, by_id, elapsed = [], {}, 0.0
        timing = iter(self.result.get("turn_log", []))
        for message in self.transcript[2:]:
            role, content = message["role"], message.get("content")
            if role == "assistant":
                for t in timing:  # truncated replies take time but produce no step
                    elapsed += t.get("seconds", 0)
                    if t.get("finish_reason") != "length":
                        break
                calls = []
                for c in message.get("tool_calls", []):
                    try:
                        arguments = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = None
                    calls.append(Call(c["function"]["name"], arguments))
                    by_id[c["id"]] = calls[-1]
                seconds = replies[len(steps)]["seconds"] if len(steps) < len(replies) else 0.0
                steps.append(Step(len(steps) + 1, (content or "").strip(), calls, seconds, round(elapsed, 1)))
            elif role == "tool" and message.get("tool_call_id") in by_id:
                by_id[message["tool_call_id"]].result = json.loads(message["content"])
            elif role == "user" and steps:
                if isinstance(content, list):
                    steps[-1].images += [p["text"].split(":")[0] for p in content if p.get("type") == "text"]
                else:
                    steps[-1].harness.append(content)
        return steps


def load_run(folder):
    return Run.load(folder)


def load_recording(name):
    """A run saved with the cookbook, by scenario name, for example "model-switch-visual"."""
    return Run.load(REPLAY / name)
