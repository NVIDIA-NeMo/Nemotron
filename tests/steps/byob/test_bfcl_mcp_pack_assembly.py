"""What candidate pack assembly binds for an oracle that lives behind a session.

An MCP Mode A pack has no `backend.py` to fingerprint, so the bindings that make a local
pack trustworthy have to be replaced rather than skipped. `endpoint_config.yaml` stands in
for the source tree and is accepted only when it pins the identity certification observed,
`tools.json` is compared tool by tool against the certified surface instead of being
trusted for its location, and `fixtures.json` is written from the reviewed probe plan the
evidence already carries a digest of, because a gateway is handed its world at session open
rather than reading it from the pack.

Intake here is the real one, run at A2 against a gateway that is genuinely listening, so
the refusals below are checked against evidence a live server produced.
"""

from __future__ import annotations

import asyncio
import copy
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    EndpointIdentity,
)
from nemotron.steps.byob.runtime.mcp.authoring.intake import LoadedMcpIntake
from nemotron.steps.byob.runtime.mcp.authoring.runner import run_intake
from nemotron.steps.byob.runtime.mcp.gateway.identity import GatewayIdentity
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.compile_assertions import (
    compile_assertions,
)
from nemotron.steps.byob.runtime.pack_authoring.drafts import AssertionSpecPlan
from nemotron.steps.byob.runtime.pack_authoring.pack_assembly import (
    PackAssemblyError,
    assemble_candidate_pack,
)
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    CertificationAuthority,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import AdapterProbePlan
from tests.steps.byob.mcp_gateway_fixture import (
    COLD_STORAGE_BOOK,
    RunningGateway,
    pin_catalog_digest,
    raw_oracle_config,
    serve_mcp_gateway,
)

GATEWAY_DIGEST = "sha256:" + "b" * 64
BASE_URL = "https://library-gateway.example.test"
CLOCK = "2026-03-02T09:00:00+00:00"
PACK_ID = "library-mcp"
PACK_VERSION = "1.0.0"
FIXTURES: dict[str, Any] = {
    "books": [
        {"book_id": "BK-1", "status": "available", "copies": 2},
        {"book_id": COLD_STORAGE_BOOK, "status": "available", "copies": 1},
    ],
    "patrons": [{"patron_id": "P-1"}],
}

_ASSERTION_SPECS: dict[str, Any] = {
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
    ]
}


def _probe_plan_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "bfcl-adapter-probe-plan-v1",
        "clock": CLOCK,
        "seed": 7,
        "fixtures": copy.deepcopy(FIXTURES),
        "cases": [
            {
                "case_id": "checkout.success",
                "tool": "checkout_book",
                "arguments": {
                    "book_id": "BK-1",
                    "patron_id": "P-1",
                    "confirm": True,
                },
                "expectation": "success",
                "expected_state_change": True,
            },
            {
                "case_id": "status.cold",
                "tool": "get_book_status",
                "arguments": {"book_id": COLD_STORAGE_BOOK},
                "expectation": "timeout",
            },
            {
                "case_id": "status.missing",
                "tool": "get_book_status",
                "arguments": {"book_id": "BK-404"},
                "expectation": "structured_error",
                "expected_error_code": "not_found",
            },
            {
                "case_id": "status.success",
                "tool": "get_book_status",
                "arguments": {"book_id": "BK-1"},
                "expectation": "success",
                "expected_state_change": False,
            },
        ],
        "confirmation_parameter": "confirm",
        "status_field": "status",
        "pending_status": "awaiting_confirmation",
        "error_path": ["error", "code"],
    }
    document.update(overrides)
    return document


def _write_probe_plan(path: Path, **overrides: Any) -> Path:
    path.write_text(json.dumps(_probe_plan_document(**overrides)), encoding="utf-8")
    return path


def _write_oracle_profile(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "mcp_oracle.yaml"
    path.write_text(
        yaml.safe_dump(pin_catalog_digest(raw_oracle_config())),
        encoding="utf-8",
    )
    return path


def _write_declaration(root: Path, *, ca_bundle: Path) -> Path:
    path = root / "mcp_intake.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "intake_version": "bfcl-mcp-intake-v1",
                "kind": "mcp",
                "mcp_oracle_config": "mcp_oracle.yaml",
                "pack": {"pack_id": PACK_ID, "version": PACK_VERSION},
                "gateway": {
                    "base_url": BASE_URL,
                    "gateway_artifact_digest": GATEWAY_DIGEST,
                    # Declared so the emitted pack carries the bundle its own endpoint
                    # config pins, which is the shape assembly has to copy through.
                    "ca_bundle_path": str(ca_bundle),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _endpoint_factory(running: RunningGateway) -> Any:
    """Send attestation and probes to the gateway that is listening.

    The reviewed declaration names a routable host, because a published pack has to, while
    the fixture can only bind loopback. One factory serves both uses, which is what stops a
    test from attesting to one endpoint and probing another.
    """

    def factory(intake: LoadedMcpIntake, identity: GatewayIdentity) -> EndpointConfig:
        return EndpointConfig(
            path=Path("<mcp-assembly-test>"),
            base_url=running.base_url,
            expected=EndpointIdentity.from_mapping(
                identity.as_dict(),
                source="gateway identity",
            ),
            ca_bundle_path=running.certificate_path,
        )

    return factory


def _connection_factory(running: RunningGateway) -> Any:
    @asynccontextmanager
    async def factory(config: Any) -> Any:
        async with running.factory(config) as client:
            yield client

    return factory


class Session:
    """One A2-certified MCP source plus one drafting output, shared by every case."""

    def __init__(self, root: Path, evidence_digest: str) -> None:
        self.root = root
        self.evidence_digest = evidence_digest

    @property
    def evidence(self) -> Path:
        return self.root / "intake" / "evidence_bundle.json"

    @property
    def pack(self) -> Path:
        return self.root / "intake" / "pack"

    @property
    def drafts(self) -> Path:
        return self.root / "drafting" / "drafts"

    @property
    def probe_plan(self) -> Path:
        return self.root / "probe-plan.json"


def _write_drafting_output(root: Path, *, evidence_digest: str) -> Path:
    """Stand in for a drafting run: the assembler reads only these two artifacts."""
    drafts = root / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "assertions.py").write_text(
        compile_assertions(AssertionSpecPlan.model_validate(_ASSERTION_SPECS)),
        encoding="utf-8",
    )
    (root / "draft_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "bfcl-authoring-draft-provenance-v1",
                "evidence": {"bundle_digest": evidence_digest},
                "blocked_on": [],
                "assertions_compiled": True,
            }
        ),
        encoding="utf-8",
    )
    return drafts


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Session:
    root = tmp_path_factory.mktemp("mcp-assembly")
    declaration_root = root / "declaration"
    profile = _write_oracle_profile(declaration_root)
    brief = root / "domain-brief.txt"
    brief.write_text(
        "Evaluate deterministic library circulation over MCP.",
        encoding="utf-8",
    )
    with serve_mcp_gateway(
        profile,
        gateway_artifact_digest=GATEWAY_DIGEST,
        root=root,
        # Review covers the intake directory, so the bundle the declaration pins has to
        # live there; the private key stays outside it, where no pack can reach it.
        certificate_root=declaration_root,
        slow_call_s=5.0,
    ) as running:
        intake_path = _write_declaration(
            declaration_root,
            ca_bundle=running.certificate_path,
        )
        result = asyncio.run(
            run_intake(
                intake_path,
                root / "intake",
                connection_factory=_connection_factory(running),
                endpoint_config_factory=_endpoint_factory(running),
                domain_brief_path=brief,
                certification_authority=CertificationAuthority(
                    key_id="mcp-assembly",
                    private_key=Ed25519PrivateKey.from_private_bytes(b"\x07" * 32),
                ),
                held_out_decision=build_not_applicable_decision(
                    "Synthetic MCP fixture has no held-out evaluation.",
                    reviewed_by="reviewer@example.test",
                ),
                probe_plan=AdapterProbePlan.model_validate(_probe_plan_document()),
                required_tier=AdapterTier.A2,
            )
        )
    assert result.source_evidence is not None
    assert result.source_evidence.certification.attained_tier == "A2"
    digest = result.source_evidence.bundle_digest
    _write_drafting_output(root / "drafting", evidence_digest=digest)
    _write_probe_plan(root / "probe-plan.json")
    return Session(root, digest)


def _supplement_document() -> dict[str, Any]:
    return {
        "schema_version": "bfcl-candidate-pack-supplement-v1",
        "languages": ["en"],
        "clock": CLOCK,
        "absent_ids": {"books": ["BK-404"]},
        "assistant_turn_templates": {"en": {"ack": "Let me check the catalogue."}},
        "task_templates": [
            {
                "template_id": "lib_status_single",
                "required_tools": ["get_book_status"],
                "tools_present": ["get_book_status", "checkout_book"],
                "success_assertions": ["assert_status_checked"],
            },
            {
                "template_id": "lib_checkout_confirm",
                "required_tools": ["checkout_book"],
                "tools_present": ["get_book_status", "checkout_book"],
                "success_assertions": ["assert_checkout_committed"],
            },
        ],
        "validation_cases": [
            {
                "id": "success_get_book_status",
                "tool": "get_book_status",
                "arguments": {"book_id": "BK-1"},
                "expect": {"result_class": "success", "error_code": None},
                "reset_before": True,
            },
        ],
    }


def _supplement(path: Path, **overrides: Any) -> Path:
    document = {**_supplement_document(), **overrides}
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _assemble(
    session: Session,
    tmp_path: Path,
    *,
    source_root: Path | None = None,
    probe_plan: Path | None = None,
    with_probe_plan: bool = True,
    supplement: Path | None = None,
    output: Path | None = None,
) -> Any:
    return assemble_candidate_pack(
        evidence_path=session.evidence,
        source_root=source_root or session.pack,
        draft_root=session.drafts,
        supplement_path=supplement or _supplement(tmp_path / "supplement.yaml"),
        output_root=output or tmp_path / "candidate",
        probe_plan_path=(
            (probe_plan or session.probe_plan) if with_probe_plan else None
        ),
    )


def _copied_pack(session: Session, tmp_path: Path, name: str) -> Path:
    """A writable copy of the certified intake pack, for the tamper cases."""
    target = tmp_path / name
    shutil.copytree(session.pack, target)
    return target


def test_assembly_binds_an_mcp_pack_to_its_endpoint_catalog_and_probe_plan(
    session: Session,
    tmp_path: Path,
) -> None:
    assembled = _assemble(session, tmp_path)

    manifest = yaml.safe_load(assembled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pack_id"] == PACK_ID
    assert manifest["version"] == PACK_VERSION
    # The loader accepts exactly one oracle, and for a session source that is the endpoint.
    assert manifest["paths"]["endpoint"] == "endpoint_config.yaml"
    assert "backend" not in manifest["paths"]
    assert manifest["paths"]["fixtures"] == "fixtures.json"
    # A pack whose confirmation names differ from the gateway's would gate nothing, so the
    # vocabulary comes from evidence rather than from whatever a supplement remembered.
    assert manifest["confirmation"] == {
        "parameter": "confirm",
        "pending_status": "awaiting_confirmation",
        "status_field": "status",
    }

    for name in ("endpoint_config.yaml", "tools.json", "oracle_ca.pem"):
        assert (assembled.pack_root / name).read_bytes() == (
            session.pack / name
        ).read_bytes()
    assert (
        json.loads((assembled.pack_root / "fixtures.json").read_text(encoding="utf-8"))
        == FIXTURES
    )

    record = assembled.record
    assert record["adapter_kind"] == "mcp_mode_a"
    assert record["evidence_digest"] == session.evidence_digest
    assert record["probe_plan_digest"] is not None
    assert set(record["pack_files"]) == {
        "assertions.py",
        "endpoint_config.yaml",
        "fixtures.json",
        "manifest.yaml",
        "oracle_ca.pem",
        "task_templates.yaml",
        "tools.json",
        "validation_cases.yaml",
    }
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    assert record["record_digest"] == sha256_json(unsigned)


def test_assembly_refuses_an_endpoint_that_pins_another_oracle(
    session: Session,
    tmp_path: Path,
) -> None:
    tampered = _copied_pack(session, tmp_path, "relabelled-pack")
    endpoint = tampered / "endpoint_config.yaml"
    document = yaml.safe_load(endpoint.read_text(encoding="utf-8"))
    document["expected"]["content_digest"] = "sha256:" + "f" * 64
    endpoint.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, source_root=tampered)

    assert refusal.value.code == "source_identity_mismatch"
    assert not (tmp_path / "candidate").exists()


def test_assembly_refuses_a_catalog_that_drifted_from_the_certified_surface(
    session: Session,
    tmp_path: Path,
) -> None:
    tampered = _copied_pack(session, tmp_path, "redescribed-pack")
    tools_path = tampered / "tools.json"
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tools[0]["function"]["description"] = "Lend a book without asking anybody."
    tools_path.write_text(json.dumps(tools), encoding="utf-8")

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, source_root=tampered)

    assert refusal.value.code == "source_catalog_mismatch"
    assert tools[0]["function"]["name"] in refusal.value.detail


def test_assembly_refuses_fixtures_the_certification_never_saw(
    session: Session,
    tmp_path: Path,
) -> None:
    other = _write_probe_plan(
        tmp_path / "other-plan.json",
        fixtures={
            "books": [{"book_id": "BK-9", "status": "available", "copies": 1}],
            "patrons": [{"patron_id": "P-9"}],
        },
    )

    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, probe_plan=other)

    assert refusal.value.code == "fixtures_mismatch"


def test_assembly_refuses_a_session_pack_with_no_reviewed_probe_plan(
    session: Session,
    tmp_path: Path,
) -> None:
    with pytest.raises(PackAssemblyError) as refusal:
        _assemble(session, tmp_path, with_probe_plan=False)

    assert refusal.value.code == "fixtures_missing"
