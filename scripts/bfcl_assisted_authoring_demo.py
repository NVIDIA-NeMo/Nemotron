#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One runnable walk of the BFCL assisted authoring flow, end to end.

This drives the shipped guided CLI rather than reimplementing it: a reviewed source
package is certified by live probes, an authoring model drafts the plans that source can
support, the drafts plus reviewed semantics are bound into a candidate pack, and the pack
is validated, reviewed, frozen, and published into a benchmark. The benchmark is then
scored by a real evaluation run against a candidate served on loopback.

Two things here stand in for people, and both say so when they run. The authoring model is
scripted by default, because a demo that needs credentials is not one most readers can run;
pass --author-model live to send the same prompts to a real endpoint. The human review
steps — exposure authorization, evidence approval, the reviewed semantics supplement, and
the release checklist — are answered from the constants below, and each prints what a
reviewer would have been deciding. Nothing else is faked: intake probes a real package,
validation runs unmocked and derives its own tier, and evaluation replays the published
benchmark through the real scorer.

    uv run python scripts/bfcl_assisted_authoring_demo.py --workdir /tmp/bfcl-demo

The candidate answers from the benchmark's own recorded turns, so a clean run should score
1.0 and prove the flow produces a benchmark a model can pass. To watch the scorer fail a
model instead, re-score the published benchmark with one task sabotaged:

    uv run python scripts/bfcl_assisted_authoring_demo.py --workdir /tmp/bfcl-demo \
        --stage eval --wrong-answer-task <task_id>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import threading
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BYOB_ROOT
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.source_adapters.certification import (
    load_trusted_certification_key,
)
from nemotron.steps.byob.scripts import bfcl_author

TINY = BYOB_ROOT / "data" / "tiny_oracle_pack"
PACK_ID = "tiny_library"
PACK_VERSION = "0.1.0"
KEY_ID = "bfcl-demo"
REVIEWER = "reviewer@example.test"
OWNER = "owner@example.test"
CLOCK = "2026-03-02T09:00:00+07:00"

# A full reindex is the one operation this domain cannot bound, which is what lets the
# reviewed probe plan observe timeout cleanup and therefore reach A2.
_INDEX_TOOL = """

def _rebuild_catalog_index(arguments: dict) -> dict:
    full = arguments.get("full", False)
    if not isinstance(full, bool):
        return {
            "error": {
                "code": "invalid_argument",
                "entity": None,
                "id": None,
                "field": "full",
                "message": "full must be a boolean",
            }
        }
    if full:
        time.sleep(3600)
    return {"status": "succeeded", "indexed": len(_STATE.get("books", []))}
"""


def _banner(step: str, title: str) -> None:
    print(f"\n\033[1m== {step}  {title}\033[0m", flush=True)


def _simulated(decision: str, detail: str) -> None:
    print(f"  \033[33m[simulated human review]\033[0m {decision}: {detail}", flush=True)


def _note(text: str) -> None:
    print(f"  {text}", flush=True)


# --------------------------------------------------------------------------------------
# The reviewed source package a customer would bring, plus the probe plan that certifies it
# --------------------------------------------------------------------------------------


def write_source_package(root: Path) -> Path:
    """Materialize a reviewed local Python source package from the tiny sample pack."""
    package = root / "library-source"
    package.mkdir(parents=True)
    shutil.copyfile(TINY / "fixtures.json", package / "fixtures.json")

    backend = (TINY / "backend.py").read_text(encoding="utf-8")
    backend = backend.replace("import copy\n", "import copy\nimport time\n", 1)
    backend = backend.replace(
        'return ["get_book_status", "checkout_book"]',
        'return ["get_book_status", "checkout_book", "rebuild_catalog_index"]',
    )
    backend = backend.replace(
        '    if name == "checkout_book":\n        return _checkout_book(arguments)\n',
        '    if name == "checkout_book":\n        return _checkout_book(arguments)\n'
        '    if name == "rebuild_catalog_index":\n'
        "        return _rebuild_catalog_index(arguments)\n",
    )
    (package / "backend.py").write_text(backend + _INDEX_TOOL, encoding="utf-8")

    tools = json.loads((TINY / "tools.json").read_text(encoding="utf-8"))
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "rebuild_catalog_index",
                "description": "Rebuild the search index; a full rebuild walks every shelf.",
                "parameters": {
                    "type": "object",
                    "properties": {"full": {"type": "boolean", "default": False}},
                    "required": ["full"],
                    "additionalProperties": False,
                },
            },
        }
    )
    (package / "tools.json").write_text(json.dumps(tools, indent=2), encoding="utf-8")
    (package / "dependency-lock.json").write_text(
        json.dumps({"schema_version": "bfcl-python-dependency-lock-v1", "dependencies": []}),
        encoding="utf-8",
    )
    return package


def write_probe_plan(path: Path) -> Path:
    """The reviewed probe plan: what intake is allowed to call, and what it must observe."""
    path.write_text(
        json.dumps(
            {
                "schema_version": "bfcl-local-probe-plan-v1",
                "clock": CLOCK,
                "seed": 7,
                "fixtures": json.loads((TINY / "fixtures.json").read_text(encoding="utf-8")),
                "cases": [
                    {
                        "case_id": "a_status_available",
                        "tool": "get_book_status",
                        "arguments": {"book_id": "BK-100"},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "b_checkout_committed",
                        "tool": "checkout_book",
                        "arguments": {
                            "book_id": "BK-200",
                            "patron_id": "P-1",
                            "confirm": True,
                        },
                        "expectation": "success",
                        "expected_state_change": True,
                    },
                    {
                        "case_id": "c_index_incremental",
                        "tool": "rebuild_catalog_index",
                        "arguments": {"full": False},
                        "expectation": "success",
                        "expected_state_change": False,
                    },
                    {
                        "case_id": "d_status_absent",
                        "tool": "get_book_status",
                        "arguments": {"book_id": "BK-ABSENT-1"},
                        "expectation": "structured_error",
                        "expected_error_code": "not_found",
                    },
                    {
                        "case_id": "e_full_reindex_hangs",
                        "tool": "rebuild_catalog_index",
                        "arguments": {"full": True},
                        "expectation": "timeout",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# The authoring model
# --------------------------------------------------------------------------------------

# What a scripted author answers for each drafting stage. Grounding, blocker detection, and
# assertion compilation still run for real over these answers, so a plan that named a tool
# the source never published would be refused here exactly as a live model's would.
DRAFT_RESPONSES: dict[str, dict[str, Any]] = {
    "mcp_coverage_plan": {
        "tools": [
            {
                "tool": "get_book_status",
                "purpose": "Report whether one book is on the shelf.",
                "policies": ["Never guess a book id."],
                "positive_intents": ["Report the status of a book the library holds."],
                "negative_intents": ["Report that an unknown book id is not held."],
                "depends_on": [],
                "confirmation_relevant": False,
            },
            {
                "tool": "checkout_book",
                "purpose": "Lend one book to one patron.",
                "policies": ["Confirm with the patron before committing a loan."],
                "positive_intents": ["Lend an available book after confirmation."],
                "negative_intents": ["Hold the loan until the patron confirms."],
                "depends_on": ["get_book_status"],
                "confirmation_relevant": True,
            },
            {
                "tool": "rebuild_catalog_index",
                "purpose": "Refresh the catalogue search index.",
                "policies": ["Prefer an incremental rebuild during opening hours."],
                "positive_intents": ["Refresh the index incrementally."],
                "negative_intents": ["Reject a scope that is not a boolean."],
                "depends_on": [],
                "confirmation_relevant": False,
            },
        ],
        "cross_tool_notes": ["Check a book's status before lending it."],
    },
    "mcp_validation_cases": {
        "cases": [
            {
                "case_id": "status_of_a_shelved_book",
                "tool": "get_book_status",
                "kind": "success",
                "intent": "Report the status of a book the catalogue holds.",
                "arguments": [{"name": "book_id", "source": "fixture"}],
                "expectation": "The catalogue reports that book's loan status.",
                "blocked_on": [],
            },
            {
                "case_id": "status_of_an_unknown_book",
                "tool": "get_book_status",
                "kind": "error",
                "intent": "Report that an unknown book id is not held.",
                "arguments": [{"name": "book_id", "source": "absent_id"}],
                "expectation": "The catalogue reports the book was not found.",
                "blocked_on": [],
            },
            {
                "case_id": "lend_after_confirmation",
                "tool": "checkout_book",
                "kind": "success",
                "intent": "Lend an available book once the patron has confirmed.",
                "arguments": [
                    {"name": "book_id", "source": "fixture"},
                    {"name": "patron_id", "source": "fixture"},
                    {"name": "confirm", "source": "confirmation_flag"},
                ],
                "expectation": "The loan is committed and the book leaves the shelf.",
                "blocked_on": [],
            },
            {
                "case_id": "lend_waits_for_confirmation",
                "tool": "checkout_book",
                "kind": "confirmation_pending",
                "intent": "Hold a loan until the patron confirms it.",
                "arguments": [
                    {"name": "book_id", "source": "fixture"},
                    {"name": "patron_id", "source": "fixture"},
                ],
                "expectation": "The loan waits rather than committing.",
                "blocked_on": [],
            },
            {
                "case_id": "incremental_reindex",
                "tool": "rebuild_catalog_index",
                "kind": "success",
                "intent": "Refresh the index without walking every shelf.",
                "arguments": [{"name": "full", "source": "literal", "literal": "false"}],
                "expectation": "The index is refreshed and reports how many books it saw.",
                "blocked_on": [],
            },
        ]
    },
    "mcp_task_templates": {
        "templates": [
            {
                "template_id": "status_of_one_book",
                "user_goal": "I want to know whether a book is on the shelf.",
                "required_tools": ["get_book_status"],
                "milestones": [
                    {
                        "description": "Look the book up in the catalogue.",
                        "tool": "get_book_status",
                        "requires_confirmation": False,
                    },
                    {
                        "description": "Tell the patron what the catalogue says.",
                        "tool": None,
                        "requires_confirmation": False,
                    },
                ],
                "policies": ["Never guess a book id."],
                "blocked_on": [],
            },
            {
                "template_id": "borrow_one_book",
                "user_goal": "I would like to borrow a book.",
                "required_tools": ["checkout_book"],
                "milestones": [
                    {
                        "description": "Ask the patron to confirm the loan.",
                        "tool": None,
                        "requires_confirmation": False,
                    },
                    {
                        "description": "Commit the loan once the patron confirms.",
                        "tool": "checkout_book",
                        "requires_confirmation": True,
                    },
                ],
                "policies": ["Confirm with the patron before committing a loan."],
                "blocked_on": [],
            },
        ]
    },
    "mcp_assertion_specs": {
        "assertions": [
            {
                "assertion_id": "status_checked",
                "subject": "trace",
                "predicate": "tool_called",
                "target": "get_book_status",
                "argument": None,
                "tool": None,
                "rationale": "A status question is only answered by consulting the catalogue.",
                "blocked_on": [],
            },
            {
                "assertion_id": "checkout_committed",
                "subject": "trace",
                "predicate": "tool_called",
                "target": "checkout_book",
                "argument": None,
                "tool": None,
                "rationale": "A borrow request has to reach the checkout tool.",
                "blocked_on": [],
            },
            {
                "assertion_id": "no_status_checked",
                "subject": "trace",
                "predicate": "tool_not_called",
                "target": "get_book_status",
                "argument": None,
                "tool": None,
                "rationale": "A request the library cannot serve must not query the catalogue.",
                "blocked_on": [],
            },
            {
                "assertion_id": "no_checkout_attempted",
                "subject": "trace",
                "predicate": "tool_not_called",
                "target": "checkout_book",
                "argument": None,
                "tool": None,
                "rationale": "A request the library cannot serve must not lend anything.",
                "blocked_on": [],
            },
        ]
    },
}


class ScriptedAuthoringModel:
    """Answers each drafting stage with one canned plan; grounding still runs for real."""

    def __init__(self) -> None:
        self.stages: list[str] = []

    def __call__(
        self,
        _run_dir: Path,
        *,
        stage_name: str,
        requests: list[dict[str, str]],
        **_kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        del requests
        self.stages.append(stage_name)
        return {stage_name: DRAFT_RESPONSES[stage_name]}


# --------------------------------------------------------------------------------------
# The reviewed semantics a model is not allowed to invent
# --------------------------------------------------------------------------------------

# Which compiled assertion each reviewed template is expected to hold.
TEMPLATE_ASSERTIONS = {
    "lib_status_single": ["assert_status_checked"],
    "lib_checkout_confirm": ["assert_checkout_committed"],
    "lib_status_parallel": ["assert_status_checked"],
    "lib_irrelevant_renew": [
        "assert_no_status_checked",
        "assert_no_checkout_attempted",
    ],
}


def write_supplement(path: Path) -> Path:
    """The reviewed semantics no draft schema can express, authored once by a human."""
    templates = yaml.safe_load((TINY / "task_templates.yaml").read_text(encoding="utf-8"))
    for template in templates:
        template["success_assertions"] = TEMPLATE_ASSERTIONS[template["template_id"]]
        template["tools_present"] = [
            "get_book_status",
            "checkout_book",
            "rebuild_catalog_index",
        ]
    cases = yaml.safe_load((TINY / "validation_cases.yaml").read_text(encoding="utf-8"))
    cases.extend(
        [
            {
                "id": "success_rebuild_catalog_index",
                "tool": "rebuild_catalog_index",
                "arguments": {"full": False},
                "expect": {"result_class": "success", "error_code": None},
                "reset_before": True,
            },
            {
                "id": "wrong_type_rebuild_catalog_index",
                "tool": "rebuild_catalog_index",
                "arguments": {"full": "yes"},
                "expect": {
                    "result_class": "structured_error",
                    "error_code": "invalid_argument",
                },
                "reset_before": True,
            },
        ]
    )
    manifest = yaml.safe_load((TINY / "manifest.yaml").read_text(encoding="utf-8"))
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "bfcl-candidate-pack-supplement-v1",
                "languages": manifest["languages"],
                "clock": manifest["clock"],
                "absent_ids": manifest["absent_ids"],
                "assistant_turn_templates": manifest["assistant_turn_templates"],
                "task_templates": templates,
                "validation_cases": cases,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Guided CLI plumbing
# --------------------------------------------------------------------------------------


def run_guided(arguments: Sequence[str]) -> dict[str, Any] | None:
    """Invoke the shipped guided CLI in-process and return whatever JSON it printed."""
    previous = sys.argv
    sys.argv = ["bfcl_author.py", *arguments]
    try:
        bfcl_author.main()
    finally:
        sys.argv = previous
    return None


def run_script(module_name: str, arguments: Sequence[str]) -> None:
    import importlib

    module = importlib.import_module(module_name)
    previous = sys.argv
    sys.argv = [module_name, *arguments]
    try:
        module.main()
    finally:
        sys.argv = previous


def write_certification_keys(root: Path) -> tuple[Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_path = root / "certification-private.pem"
    public_path = root / "certification-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def write_generation_config(
    path: Path,
    *,
    pack_manifest: Path,
    output_dir: Path,
    expt_name: str,
    allowed_roots: Sequence[Path],
    lineage_policy: str,
) -> Path:
    document = yaml.safe_load((BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    document["expt_name"] = expt_name
    document["output_dir"] = str(output_dir)
    document["oracle_pack"] = {"manifest_path": str(pack_manifest)}
    document["oracle_runtime"]["allowed_roots"] = [str(root) for root in allowed_roots]
    document["lineage"]["policy"] = lineage_policy
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# A candidate model served on loopback, answering from the benchmark's own expectations
# --------------------------------------------------------------------------------------


def _conversation_key(messages: Sequence[Any]) -> tuple[str, int]:
    """Recognize where a conversation stands from the conversation alone.

    The wire request carries no task id, so a candidate has to place itself the way a
    real model would: from the last thing the user said, plus how many tool results have
    come back since the episode opened, which is what separates two assistant turns that
    answer the same request.
    """
    last_user = ""
    tool_results = 0
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else message["role"]
        content = message.get("content") if isinstance(message, dict) else message["content"]
        if role == "user":
            last_user = str(content or "")
        elif role == "tool":
            tool_results += 1
    return last_user, tool_results


def expected_replies(
    benchmark: Path,
    *,
    wrong_task_ids: frozenset[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Replay the assistant turns the benchmark recorded, keyed by where they belong."""
    import pyarrow.parquet as pq

    replies: dict[tuple[str, int], dict[str, Any]] = {}
    for row in pq.read_table(benchmark).to_pylist():
        answer_wrongly = row["task_id"] in wrong_task_ids
        prefix: list[dict[str, Any]] = []
        for message in row["messages"]:
            if message["role"] == "assistant":
                calls = [] if answer_wrongly else (message["tool_calls"] or [])
                replies[_conversation_key(prefix)] = {
                    "content": message["content"],
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        for call in calls
                    ],
                }
            prefix.append(message)
    return replies


class _CandidateHandler(BaseHTTPRequestHandler):
    replies: dict[tuple[str, int], dict[str, Any]] = {}

    def log_message(self, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler names it this way
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        reply = self.replies.get(
            _conversation_key(request.get("messages", [])),
            {"content": "I cannot help with that.", "tool_calls": []},
        )
        message: dict[str, Any] = {"role": "assistant", "content": reply["content"]}
        if reply["tool_calls"]:
            message["content"] = None
            message["tool_calls"] = reply["tool_calls"]
        payload = json.dumps(
            {
                "id": "chatcmpl-demo",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": message,
                        "finish_reason": ("tool_calls" if reply["tool_calls"] else "stop"),
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextlib.contextmanager
def serve_candidate(replies: dict[tuple[str, int], dict[str, Any]]) -> Iterator[str]:
    handler = type("_Handler", (_CandidateHandler,), {"replies": replies})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=10.0)
        server.server_close()


def write_eval_config(
    path: Path,
    *,
    run_manifest: Path,
    output_dir: Path,
    base_url: str,
) -> Path:
    document = {
        # 1.2, because the candidate below pins no weights.
        "schema_version": "1.2",
        "config_status": "resolved",
        "source_run_manifest": str(run_manifest),
        "source_oracle": None,
        "translation_manifest": None,
        "eval": {"mode": ["trace"]},
        "scoring": {
            "contract": str(BYOB_ROOT / "references" / "bfcl-eval-scoring-contract.md"),
            "argument_matching": "schema_then_canonical",
            "insert_declared_defaults": True,
            "respect_call_order": True,
            "respect_call_group": True,
            "allow_llm_repair": False,
            "task_success": "all_applicable_gates",
        },
        "limits": {
            "max_turns": 5,
            "tool_timeout_s": 10.0,
            "candidate_timeout_s": 60.0,
            "episode_timeout_s": 300.0,
            "max_parallel_tasks": 1,
            "max_retries": 2,
        },
        "candidates": [
            {
                "alias": "loopback_candidate",
                "model": "bfcl-demo-candidate",
                "provider": "nvidia",
                "provider_api_version": "v1",
                "api": {
                    "base_url": base_url,
                    "api_key_env": "BFCL_DEMO_CANDIDATE_KEY",
                },
                # The candidate is a scripted loopback server, so no registry
                # commit and no weight bytes exist to point at. It is recorded
                # unpinned rather than given a plausible-looking commit: the run
                # then reports itself as non-publishable, which is exactly what a
                # score against an unpinnable endpoint is worth. With no pin, the
                # route is the identity, so these two restate provider and model.
                "model_identity": {
                    "source": "nvidia",
                    "model": "bfcl-demo-candidate",
                    "revision": None,
                    "weights_digest": None,
                },
                "inference": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "seed": 42,
                    "tool_choice": "auto",
                    "provider_extensions": {},
                },
            }
        ],
        "contamination": {
            "enforce": True,
            "on_violation": "fail_run",
            "comparison_set": "common_intersection",
        },
        "publication": {"requested": False, "require_same_task_ids": True},
        "outputs": {
            "output_dir": str(output_dir),
            "write_task_results": True,
            "write_eval_manifest": True,
            "cache_candidate_responses": True,
            "cache_tool_results": True,
        },
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# The walk itself
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "eval"),
        default="all",
        help="eval reuses the benchmark an earlier --stage all run published here",
    )
    parser.add_argument(
        "--author-model",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted answers drafting locally; live sends the same prompts out",
    )
    parser.add_argument("--model-provider", default="test")
    parser.add_argument("--model", default="stub")
    parser.add_argument("--model-canonical-id", default="test/stub@1")
    parser.add_argument(
        "--wrong-answer-task",
        action="append",
        default=[],
        help="task id the demo candidate should answer with text instead of a call",
    )
    args = parser.parse_args()

    work = args.workdir.resolve()
    os.environ["BFCL_ENABLE_LOCAL_PYTHON"] = "1"
    os.environ.setdefault("BFCL_DEMO_CANDIDATE_KEY", "demo-key")
    workspace = work / "workspace"
    if args.stage == "eval":
        published = workspace / "generated" / "bfcl-demo"
        if not (published / "run_manifest.json").is_file():
            raise SystemExit(f"no published benchmark under {published}")
        evaluate(work, published, wrong_answer_tasks=args.wrong_answer_task)
        return
    if work.exists():
        raise SystemExit(f"workdir already exists, pick a fresh one: {work}")
    work.mkdir(parents=True)

    caller: ScriptedAuthoringModel | None = None
    if args.author_model == "scripted":
        from nemotron.steps.byob.runtime.pack_authoring import model_client

        caller = ScriptedAuthoringModel()
        model_client._default_caller = caller  # type: ignore[assignment]

    package = write_source_package(work)
    brief = work / "domain-brief.txt"
    brief.write_text(
        "Benchmark deterministic library circulation: look a book up, and lend it only after the patron confirms.",
        encoding="utf-8",
    )
    private_key, public_key = write_certification_keys(work)

    _banner("1/9", "Intake: certify the reviewed source by probing it")
    run_guided(
        [
            "--ci",
            "author",
            "--workspace",
            str(workspace),
            "--source",
            str(package),
            "--brief",
            str(brief),
            "--pack-id",
            PACK_ID,
            "--pack-version",
            PACK_VERSION,
            "--required-tier",
            "A2",
            "--held-out-not-applicable-reason",
            "The catalogue is public reference data.",
            "--held-out-reviewed-by",
            REVIEWER,
            "--certification-private-key",
            str(private_key),
            "--certification-key-id",
            KEY_ID,
            "--probe-plan",
            str(write_probe_plan(work / "probe-plan.json")),
        ]
    )
    intake = workspace / "intake"
    certification = json.loads((intake / "adapter_certification.json").read_text(encoding="utf-8"))
    _note(f"attained certification tier: {certification['attained_tier']}")

    _banner("2/9", "Authorize model exposure")
    _simulated(
        "model exposure",
        "an owner accepts that this redacted brief and tool surface may reach a model",
    )
    run_guided(
        [
            "--ci",
            "authorize",
            "--workspace",
            str(workspace),
            "--subject",
            str(intake / "model_exposure_subject.json"),
            "--authorized-by",
            OWNER,
        ]
    )

    _banner("3/9", "Approve the evidence bundle")
    evidence = load_evidence_bundle(
        intake / "evidence_bundle.json",
        certification_report_path=intake / "adapter_certification.json",
        trusted_certification_keys=load_trusted_certification_key(
            public_key,
            key_id=KEY_ID,
        ),
        domain_brief_source_path=intake / "domain_brief.source.txt",
        domain_brief_report_path=intake / "domain_brief_redaction.json",
        held_out_redaction_report_path=intake / "held_out_redaction.json",
        source_observations_path=intake / "source_observations.json",
    )
    _simulated(
        "evidence approval",
        "a reviewer confirms the normalized bundle describes the source they reviewed",
    )
    approval = workspace / "evidence_approval.json"
    run_guided(
        [
            "--ci",
            "approve",
            "--workspace",
            str(workspace),
            "--boundary",
            "evidence",
            "--approved-by",
            REVIEWER,
            "--source-bundle-digest",
            str(evidence.source_digest),
            "--normalized-bundle-digest",
            evidence.digest,
            "--output",
            str(approval),
        ]
    )

    _banner("4/9", "Draft: the model proposes what this source can support")
    drafting = workspace / "drafting"
    run_guided(
        [
            "--ci",
            "draft",
            "--workspace",
            str(workspace),
            "--bundle",
            str(intake / "evidence_bundle.json"),
            "--certification-report",
            str(intake / "adapter_certification.json"),
            "--certification-public-key",
            str(public_key),
            "--certification-key-id",
            KEY_ID,
            "--domain-brief-source",
            str(intake / "domain_brief.source.txt"),
            "--domain-brief-report",
            str(intake / "domain_brief_redaction.json"),
            "--held-out-redaction-report",
            str(intake / "held_out_redaction.json"),
            "--source-observations",
            str(intake / "source_observations.json"),
            "--exposure-authorization",
            str(workspace / "exposure_authorization.json"),
            "--approval",
            str(approval),
            "--output",
            str(drafting),
            "--model-alias",
            "author",
            "--model-provider",
            args.model_provider,
            "--model",
            args.model,
            "--model-canonical-id",
            args.model_canonical_id,
        ]
    )
    if caller is not None:
        _note(f"drafting stages answered: {', '.join(caller.stages)}")

    _banner("5/9", "Assemble: bind drafts and reviewed semantics into a candidate pack")
    _simulated(
        "reviewed semantics",
        "a human supplies slot bindings, turn policies, and per-language user turns, "
        "which no draft schema is allowed to express",
    )
    # Every guided command binds its output into the session, so its output has to stay
    # inside the workspace the session covers.
    candidate_root = workspace / "candidate"
    run_guided(
        [
            "--ci",
            "assemble",
            "--workspace",
            str(workspace),
            "--supplement",
            str(write_supplement(work / "supplement.yaml")),
            "--output",
            str(candidate_root),
        ]
    )
    pack_root = candidate_root / "pack"
    _note(f"candidate pack: {pack_root}")

    _banner("6/9", "Validate the candidate pack and derive its tier")
    validation_config = write_generation_config(
        work / "candidate-validation.yaml",
        pack_manifest=pack_root / "manifest.yaml",
        output_dir=work / "validation-out",
        expt_name="bfcl-demo-validation",
        allowed_roots=[work],
        lineage_policy="strict_separation",
    )
    report_path = prepare_bfcl(validation_config)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _note(f"tier={report['tier']} gold_eligible={report['gold_eligible']}")
    if not report["gold_eligible"]:
        raise SystemExit("candidate pack is not gold-eligible; nothing to publish")

    _banner("7/9", "Review, approve, and freeze the release")
    packet = workspace / "review_packet.json"
    freeze_inputs = work / "freeze-inputs.json"
    run_guided(
        [
            "--ci",
            "review",
            "--workspace",
            str(workspace),
            "--adapter-kind",
            "local_python",
            "--pack",
            str(pack_root),
            "--evidence",
            str(intake / "evidence_bundle.json"),
            "--certification-report",
            str(intake / "adapter_certification.json"),
            "--certification-public-key",
            str(public_key),
            "--certification-key-id",
            KEY_ID,
            "--domain-brief-source",
            str(intake / "domain_brief.source.txt"),
            "--domain-brief-report",
            str(intake / "domain_brief_redaction.json"),
            "--held-out-redaction-report",
            str(intake / "held_out_redaction.json"),
            "--source-observations",
            str(intake / "source_observations.json"),
            "--intake-provenance",
            str(intake / "intake_provenance.json"),
            "--draft-provenance",
            str(drafting / "draft_provenance.json"),
            "--validation-report",
            str(report_path),
            "--validation-config",
            str(validation_config),
            "--resolved-authoring-config",
            str(workspace / "resolved_authoring_config.json"),
            "--exposure-authorization",
            str(workspace / "exposure_authorization.json"),
            "--evidence-approval",
            str(approval),
            "--output",
            str(packet),
            "--freeze-inputs-output",
            str(freeze_inputs),
        ]
    )
    _simulated(
        "release checklist",
        "a reviewer accepts semantics, descriptions, held-out policy, assumptions, "
        "validation evidence, certification, pre-model authorization, and questions",
    )
    release_approval = workspace / "release_approval.json"
    run_guided(
        [
            "--ci",
            "approve",
            "--workspace",
            str(workspace),
            "--boundary",
            "release",
            "--packet",
            str(packet),
            "--approved-by",
            REVIEWER,
            "--reviewed-at",
            CLOCK,
            "--output",
            str(release_approval),
            "--accept-semantics",
            "--accept-descriptions-and-snapshots",
            "--accept-held-out-policy",
            "--accept-assumptions",
            "--accept-validation-evidence",
            "--accept-independently-verified-certification",
            "--accept-pre-model-authorization",
            "--accept-answered-questions",
        ]
    )
    release_root = workspace / "release"
    run_guided(
        [
            "--ci",
            "freeze",
            "--workspace",
            str(workspace),
            "--freeze-inputs",
            str(freeze_inputs),
            "--approval",
            str(release_approval),
            "--output",
            str(release_root),
        ]
    )

    _banner("8/9", "Publish: fresh validation, then generate the benchmark")
    publication_config = write_generation_config(
        work / "publication.yaml",
        pack_manifest=release_root / "pack" / "manifest.yaml",
        # Guided publication binds run_manifest.json into the session, so the generation
        # output has to land inside the workspace that session covers.
        output_dir=workspace / "generated",
        expt_name="bfcl-demo",
        allowed_roots=[release_root / "pack"],
        lineage_policy="strict_separation",
    )
    run_guided(
        [
            "--ci",
            "publish",
            "--workspace",
            str(workspace),
            "--release",
            str(release_root),
            "--config",
            str(publication_config),
        ]
    )
    published = workspace / "generated" / "bfcl-demo"
    manifest = json.loads((published / "run_manifest.json").read_text(encoding="utf-8"))
    _note(f"benchmark: {published / 'benchmark.parquet'}")
    _note(f"tier={manifest['tier']} gold_eligible={manifest['gold_eligible']}")

    evaluate(work, published, wrong_answer_tasks=args.wrong_answer_task)


def evaluate(
    work: Path,
    published: Path,
    *,
    wrong_answer_tasks: Sequence[str],
) -> None:
    """Score the published benchmark with a candidate served on loopback."""
    _banner("9/9", "Evaluate the published benchmark")
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_runner import (
        run_declared_eval_sync,
    )

    replies = expected_replies(
        published / "benchmark.parquet",
        wrong_task_ids=frozenset(wrong_answer_tasks),
    )
    _note(f"candidate primed with {len(replies)} recorded assistant turns")
    for task_id in wrong_answer_tasks:
        _note(f"candidate deliberately answers {task_id} with text instead of a call")
    # A committed eval directory belongs to the run that wrote it, so a second demo
    # eval over the same benchmark gets its own directory instead of overwriting one.
    output_dir = _fresh_output_dir(work)
    with serve_candidate(replies) as base_url:
        eval_config = write_eval_config(
            output_dir.with_suffix(".yaml"),
            run_manifest=published / "run_manifest.json",
            output_dir=output_dir,
            base_url=base_url,
        )
        run_declared_eval_sync(eval_config)
    _note(f"eval output: {output_dir}")
    _report_scores(output_dir)


def _fresh_output_dir(work: Path) -> Path:
    index = 1
    while (candidate := work / f"eval-{index}").exists():
        index += 1
    return candidate


def _report_scores(output_dir: Path) -> None:
    """Print the scores a reader of the demo actually came for."""
    import pyarrow.parquet as pq

    report = json.loads((output_dir / "eval_report.json").read_text(encoding="utf-8"))
    for candidate in report["candidates"]:
        _note(f"candidate {candidate['alias']} scored on {candidate['scope']}")
        for metric, score in sorted(candidate["metrics"].items()):
            value = score["value"]
            reading = (
                score["not_applicable_reason"]
                if value is None
                else f"{value:.2f}  ({score['numerator']}/{score['denominator']})"
            )
            print(f"    {metric:<28} {reading}", flush=True)
    results = pq.read_table(output_dir / "eval_task_results.parquet").to_pylist()
    for row in sorted(results, key=lambda row: str(row["task_id"])):
        verdict = "pass" if row["task_success"] else "FAIL"
        print(f"    [{verdict}] {row['task_id']}", flush=True)
        if row["task_success"]:
            continue
        for record in row["failure_records"] or []:
            print(
                f"             {record['code']} ({record['attribution']})",
                flush=True,
            )


if __name__ == "__main__":
    main()
