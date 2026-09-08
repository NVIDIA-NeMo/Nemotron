"""One authoring run from a source declaration and a domain brief to a Gold-eligible pack.

Everything here is the real thing except the authoring model, which cannot be real in a
test: intake executes bounded probes against the source and earns A2 from them, drafting
runs the real grounding and compiler over stubbed model answers, assembly binds the pack to
that exact evidence, and oracle validation runs unmocked and decides the tier itself.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BYOB_ROOT
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
    derive_pack_tier,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.source_adapters.certification import (
    load_trusted_certification_key,
)
from nemotron.steps.byob.scripts import assemble_candidate_pack, bfcl_author

TINY = BYOB_ROOT / "data" / "tiny_oracle_pack"
KEY_ID = "bfcl-e2e"
PACK_ID = "tiny_library"
PACK_VERSION = "0.1.0"

# A full reindex is the one operation this domain cannot bound, which is what lets the
# probe plan observe timeout cleanup and therefore reach A2.
_INDEX_TOOL = '''

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
'''


def _source_package(root: Path) -> Path:
    """Write the reviewed local Python source package the declaration points at."""
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
        json.dumps(
            {"schema_version": "bfcl-python-dependency-lock-v1", "dependencies": []}
        ),
        encoding="utf-8",
    )
    return package


def _probe_plan(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "bfcl-local-probe-plan-v1",
                "clock": "2026-03-02T09:00:00+07:00",
                "seed": 7,
                "fixtures": json.loads(
                    (TINY / "fixtures.json").read_text(encoding="utf-8")
                ),
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


# What the model is asked for, and all it is allowed to say. Every tool the source
# published has to be covered, confirmation claims have to match the reviewed profile, and
# nothing may declare itself blocked on an unknown A2 certification already resolved.
_DRAFT_RESPONSES: dict[str, dict[str, Any]] = {
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
                "arguments": [
                    {"name": "full", "source": "literal", "literal": "false"}
                ],
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

# Which compiled assertion each reviewed template expects to hold.
_TEMPLATE_ASSERTIONS = {
    "lib_status_single": ["assert_status_checked"],
    "lib_checkout_confirm": ["assert_checkout_committed"],
    "lib_status_parallel": ["assert_status_checked"],
    "lib_irrelevant_renew": [
        "assert_no_status_checked",
        "assert_no_checkout_attempted",
    ],
}


def _supplement(path: Path) -> Path:
    """The reviewed semantics no draft schema can express, authored once by a human."""
    templates = yaml.safe_load((TINY / "task_templates.yaml").read_text(encoding="utf-8"))
    for template in templates:
        template["success_assertions"] = _TEMPLATE_ASSERTIONS[template["template_id"]]
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


class _StubbedAuthoringModel:
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
        return {stage_name: _DRAFT_RESPONSES[stage_name]}


def _keys(root: Path) -> tuple[Path, Path]:
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


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["bfcl_author.py", *argv])
    bfcl_author.main()


def test_a_source_declaration_and_a_domain_brief_reach_a_gold_eligible_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = _source_package(tmp_path)
    brief = tmp_path / "domain-brief.txt"
    brief.write_text(
        "Benchmark deterministic library circulation: look a book up, and lend it only "
        "after the patron confirms.",
        encoding="utf-8",
    )
    private_key, public_key = _keys(tmp_path)
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("BFCL_ENABLE_LOCAL_PYTHON", "1")
    caller = _StubbedAuthoringModel()
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.pack_authoring.model_client._default_caller",
        caller,
    )

    _run(
        monkeypatch,
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
            "reviewer@example.test",
            "--certification-private-key",
            str(private_key),
            "--certification-key-id",
            KEY_ID,
            "--probe-plan",
            str(_probe_plan(tmp_path / "probe-plan.json")),
        ],
    )
    intake = workspace / "intake"
    certification = json.loads(
        (intake / "adapter_certification.json").read_text(encoding="utf-8")
    )
    assert certification["attained_tier"] == "A2"

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

    _run(
        monkeypatch,
        [
            "--ci",
            "authorize",
            "--workspace",
            str(workspace),
            "--subject",
            str(intake / "model_exposure_subject.json"),
            "--authorized-by",
            "owner@example.test",
        ],
    )
    capsys.readouterr()

    approval = workspace / "evidence_approval.json"
    _run(
        monkeypatch,
        [
            "--ci",
            "approve",
            "--workspace",
            str(workspace),
            "--boundary",
            "evidence",
            "--approved-by",
            "reviewer@example.test",
            "--source-bundle-digest",
            str(evidence.source_digest),
            "--normalized-bundle-digest",
            evidence.digest,
            "--output",
            str(approval),
        ],
    )
    capsys.readouterr()

    drafting = workspace / "drafting"
    _run(
        monkeypatch,
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
            "test",
            "--model",
            "stub",
            "--model-canonical-id",
            "test/stub@1",
        ],
    )
    capsys.readouterr()
    assert caller.stages == [
        "mcp_coverage_plan",
        "mcp_validation_cases",
        "mcp_task_templates",
        "mcp_assertion_specs",
    ]

    candidate = tmp_path / "candidate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble_candidate_pack.py",
            "--evidence",
            str(intake / "evidence_bundle.json"),
            "--source",
            str(package),
            "--drafts",
            str(drafting / "drafts"),
            "--supplement",
            str(_supplement(tmp_path / "supplement.yaml")),
            "--output",
            str(candidate),
        ],
    )
    assemble_candidate_pack.main()
    assembled = json.loads(capsys.readouterr().out)
    assert assembled["status"] == "assembled"
    pack_root = Path(assembled["pack"])

    manifest = yaml.safe_load(
        (pack_root / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["pack_id"] == PACK_ID
    assert manifest["version"] == PACK_VERSION
    # The oracle files are the certified source's own, not a copy a human retyped.
    for name in ("backend.py", "tools.json", "fixtures.json"):
        assert (pack_root / name).read_bytes() == (package / name).read_bytes()

    config_document = yaml.safe_load(
        (BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8")
    )
    config_document["expt_name"] = "authoring-gold-e2e"
    config_document["output_dir"] = str(tmp_path / "generated")
    config_document["oracle_pack"] = {"manifest_path": str(pack_root / "manifest.yaml")}
    config_document["oracle_runtime"]["allowed_roots"] = [str(tmp_path)]
    config = tmp_path / "candidate.yaml"
    config.write_text(yaml.safe_dump(config_document), encoding="utf-8")

    report = json.loads(prepare_bfcl(config).read_text(encoding="utf-8"))
    assert report["pack_id"] == PACK_ID
    assert report["tier"] == "gold"
    assert report["gold_eligible"] is True
    assert derive_pack_tier(report) == (True, "gold")
