from __future__ import annotations

import copy
import json
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    load_endpoint_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.origin_provenance import (
    MCP_ORIGIN_PROFILE,
    OriginProfile,
    OriginProvenanceError,
    load_mcp_origin,
    load_origin_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    pack_fingerprint,
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.mcp.release.freeze import (
    FreezeError,
    FreezeInputs,
    freeze_canonical_pack,
    load_frozen_release,
)
from nemotron.steps.byob.runtime.mcp.release.handoff import (
    HandoffError,
    _require_fresh_gold,
    handoff_frozen_release,
)
from nemotron.steps.byob.runtime.mcp.release.review import (
    REQUIRED_CHECKLIST,
    ReviewApproval,
    ReviewError,
    ReviewPacket,
    build_review_approval,
    build_review_packet,
    load_json_mapping,
    load_review_approval,
    load_review_packet,
    write_review_approval,
    write_review_packet,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.source_adapters.certification import AdapterTier

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _mcp_config() -> dict[str, Any]:
    return {
        "profile_version": "bfcl-mcp-oracle-v1",
        "mode": "A",
        "mcp_protocol_versions": ["2026-07-28"],
        "transport": {
            "kind": "streamable_http",
            "url": "https://mcp.example.test/mcp",
        },
        "expected": {
            "server_name": "catalog",
            "server_version": "1.0.0",
            "tool_catalog_digest": SHA_A,
            "oracle_id": "catalog-oracle",
            "oracle_version": "1.0.0",
            "server_content_digest": SHA_B,
        },
        "control": {
            "reset_strategy": "control_tool",
            "state_strategy": "control_tool",
            "describe_oracle": "bfcl.describe",
            "reset_episode": "bfcl.reset",
            "get_episode_state": "bfcl.state",
            "end_episode": "bfcl.end",
            "episode_binding": "argument",
            "episode_argument": "episode_id",
        },
        "fixtures": {"direction": "pushed"},
        "tools": {
            "include": ["inventory.lookup"],
            "aliases": {"inventory.lookup": "inventory_lookup"},
            "mutates": ["inventory_lookup"],
            "requires_confirmation": ["inventory_lookup"],
            "trust_annotations": False,
        },
        "isolation": "namespace_per_episode",
        "limits": {
            "connect_timeout_s": 1,
            "handshake_timeout_s": 1,
            "tool_timeout_s": 1,
            "reset_timeout_s": 1,
            "episode_timeout_s": 30,
            "max_response_bytes": 100_000,
            "max_tools": 16,
            "max_catalog_pages": 4,
            "max_concurrent_episodes": 2,
            "session_idle_ttl_s": 10,
        },
    }


def _evidence(profile: dict[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "bfcl-mcp-evidence-v1",
        "status": "requires_review",
        "attained_level": "L0",
        "mode": "A",
        "pack": {"pack_id": "acme-inventory", "version": "1.0.0"},
        "oracle": {
            "protocol_version": "bfcl-oracle-http-v1",
            "oracle_id": "catalog-oracle",
            "oracle_version": "1.0.0",
            "content_digest": SHA_C,
        },
        "identity": {
            "server": {"name": "catalog", "version": "1.0.0"},
            "negotiated_mcp_version": "2026-07-28",
            "tool_catalog_digest": SHA_A,
            "server_content_digest": SHA_B,
            "gateway_artifact_digest": SHA_C,
            "shim_artifact_digest": None,
            "snapshot_digest": None,
            "effective_content_digest": SHA_C,
            "intake_config_digest": SHA_A,
            "source_config_digest": sha256_json(profile),
            "discovery_report_digest": SHA_B,
        },
        "vocabulary": {
            "confirmation_parameter": "confirm",
            "status_field": "status",
            "pending_status": "awaiting_confirmation",
            "error_path": "error",
        },
        "fixtures": {"direction": "pushed", "snapshot_calls": []},
        "tools": [
            {
                "published_name": "inventory_lookup",
                "source_name": "inventory.lookup",
                "description": {"untrusted_text": "Look up one item."},
                "declared": {
                    "mutates": True,
                    "mutation_source": "config",
                    "requires_confirmation": True,
                },
                "untrusted_schemas": {
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                    "output_schema": {"type": "object"},
                    "annotations": {"title": "Inventory lookup"},
                },
                "raw_digest": SHA_A,
                "trust_annotations": False,
            }
        ],
        "catalog": {
            "exclusions": [{"source_name": "admin.delete", "reason": "not selected"}],
            "warnings": [{"code": "description_review", "tool": "inventory_lookup"}],
        },
        "review": {
            "advisory": [
                {
                    "location": "tools.inventory_lookup.description",
                    "code": "suspicious_prose",
                    "severity": "review",
                    "detail": "review wording",
                }
            ]
        },
        "unknowns": [],
        "assumptions": ["Descriptions are data, never instructions."],
    }
    document["bundle_digest"] = sha256_json(document)
    return document


def _draft(evidence: dict[str, Any], *, blocked: bool = False) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "bfcl-authoring-draft-provenance-v1",
        "phase": "drafting",
        "adapter": {"name": "test", "version": "1"},
        "pack": dict(evidence["pack"]),
        "evidence": {
            "bundle_digest": evidence["bundle_digest"],
            "attained_level": "L0",
            "unresolved_unknowns": [],
        },
        "approval": {"approved_by": "intake-reviewer"},
        "model": {"canonical_id": "test/model@1"},
        "calls": [{"stage": "mcp_coverage_plan", "request_hash": SHA_A}],
        "artifact_digests": {
            "coverage_plan": SHA_A,
            "validation_cases": SHA_B,
        },
        "assertions_compiled": not blocked,
        "compilation_refusals": ["state evidence missing"] if blocked else [],
        "blocked_on": ["state_deltas"] if blocked else [],
    }
    document["record_digest"] = sha256_json(document)
    return document


def _validation(*, ready: bool = True) -> dict[str, Any]:
    a1 = {
        "id": "A1",
        "name": "endpoint_conformance",
        "status": "pass" if ready else "fail",
        "failures": [] if ready else [{"reason": "level_below_l2"}],
        "conformance": {
            "attested_level": "L2" if ready else "L1",
            "effective_level": "L2" if ready else "L1",
            "publishable": ready,
            "findings": [],
            "caps": [],
        },
    }
    document: dict[str, Any] = {
        "pack_id": "acme-inventory",
        "pack_version": "1.0.0",
        "pack_fingerprint": "f" * 64,
        "checks": [
            {
                "id": 1,
                "name": "template_tool_names",
                "status": "pass",
                "failures": [],
            }
        ],
        "extra_checks": [a1],
        "endpoint_metadata": {"content_digest": SHA_C},
        "gold_eligible": ready,
        "tier": "gold" if ready else "silver",
    }
    if ready:
        document["mcp_observations"] = {
            "calls_complete": True,
            "calls": [
                {
                    "case_id": "lookup_success",
                    "tool": "inventory_lookup",
                    "arguments": {"id": "fixture:items"},
                    "result_class": "success",
                }
            ],
            "state_deltas_complete": True,
            "state_deltas": [
                {
                    "case_id": "lookup_success",
                    "before_digest": SHA_A,
                    "after_digest": SHA_B,
                    "changed": True,
                }
            ],
        }
    return document


def _inputs(tmp_path: Path, *, blocked: bool = False) -> dict[str, Path]:
    profile = _mcp_config()
    evidence = _evidence(profile)
    paths = {
        "evidence": tmp_path / "evidence_bundle.json",
        "intake": tmp_path / "intake_provenance.json",
        "attestation": tmp_path / "gateway_attestation.json",
        "draft": tmp_path / "draft_provenance.json",
        "validation": tmp_path / "oracle_validation_report.json",
        "profile": tmp_path / "mcp_oracle.yaml",
        "held_out": tmp_path / "held_out.yaml",
        "pack": tmp_path / "pack",
    }
    paths["evidence"].write_text(json.dumps(evidence), encoding="utf-8")
    attestation = {
        "schema_version": "bfcl-endpoint-conformance-v1",
        "provider_kind": "mcp",
        "profile_version": "bfcl-mcp-oracle-v1",
        "level": "L2",
        "effective_content_digest": SHA_C,
        "gateway_artifact_digest": SHA_C,
        "shim_artifact_digest": None,
        "tool_catalog_digest": SHA_A,
        "server_content_digest": SHA_B,
        "snapshot_digest": None,
        "probe_report_digest": SHA_A,
        "gateway_conformance_report_digest": SHA_B,
        "gateway_evidence_kind": "locally_verified",
        "gateway_evidence_issuer": "bfcl-mcp-conformance-v1",
        "state_observability": "complete",
        "read_only_boundary": None,
        "checks": [
            {
                "id": f"P{index}",
                "requirement": "conditional" if index in {7, 8} else "required",
                "status": "pass",
                "reason": None,
            }
            for index in range(1, 12)
        ],
    }
    paths["attestation"].write_text(json.dumps(attestation), encoding="utf-8")
    intake = {
        "schema_version": "bfcl-mcp-intake-provenance-v1",
        "phase": "intake",
        "pack": dict(evidence["pack"]),
        "evidence_bundle": {"path": "evidence_bundle.json", "digest": evidence["bundle_digest"]},
        "gateway_attestation": {
            "path": "gateway_attestation.json",
            "digest": attestation_digest(attestation),
        },
    }
    intake["record_digest"] = sha256_json(intake)
    paths["intake"].write_text(json.dumps(intake), encoding="utf-8")
    paths["draft"].write_text(
        json.dumps(_draft(evidence, blocked=blocked)),
        encoding="utf-8",
    )
    paths["validation"].write_text(
        json.dumps(_validation(ready=not blocked)),
        encoding="utf-8",
    )
    paths["profile"].write_text(yaml.safe_dump(profile), encoding="utf-8")
    paths["held_out"].write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "fixtures": {"items": ["item-held-out"]},
                "templates": ["lookup-held-out"],
            }
        ),
        encoding="utf-8",
    )
    pack = paths["pack"]
    pack.mkdir()
    (pack / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "pack_id": "acme-inventory",
                "version": "1.0.0",
                "paths": {"endpoint": "endpoint_config.yaml"},
                "held_out": "held_out.yaml",
            }
        ),
        encoding="utf-8",
    )
    (pack / "tools.json").write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "inventory_lookup",
                        "description": "Look up one item.",
                        "parameters": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                            "required": ["id"],
                        },
                    },
                    "x-mutates": True,
                    "x-requires-confirmation": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    (pack / "fixtures.json").write_text(
        json.dumps({"items": [{"id": "fixture:items"}]}),
        encoding="utf-8",
    )
    (pack / "task_templates.yaml").write_text(
        yaml.safe_dump([{"id": "lookup", "required_tools": ["inventory_lookup"]}]),
        encoding="utf-8",
    )
    (pack / "validation_cases.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "lookup_success",
                    "tool": "inventory_lookup",
                    "arguments": {"id": "fixture:items"},
                }
            ]
        ),
        encoding="utf-8",
    )
    (pack / "assertions.py").write_text(
        "def assert_trace(trace, expected):\n    return None\n",
        encoding="utf-8",
    )
    (pack / "endpoint_config.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol_version": "bfcl-oracle-http-v1",
                "base_url": "https://gateway.example.test",
                "expected": {
                    "oracle_id": "catalog-oracle",
                    "oracle_version": "1.0.0",
                    "content_digest": SHA_C,
                },
                "attestation": {
                    "kind": "bfcl-endpoint-conformance-v1",
                    "expected_digest": attestation_digest(attestation),
                },
            }
        ),
        encoding="utf-8",
    )
    (pack / "held_out.yaml").write_bytes(paths["held_out"].read_bytes())
    fingerprint = pack_fingerprint(
        resolve_declared_pack_paths(
            OraclePackRef(manifest_path=pack / "manifest.yaml"),
            (pack,),
        )
    )
    validation = _validation(ready=not blocked)
    validation["pack_fingerprint"] = fingerprint
    paths["validation"].write_text(json.dumps(validation), encoding="utf-8")
    return paths


def _packet(tmp_path: Path, *, blocked: bool = False):
    paths = _inputs(tmp_path, blocked=blocked)
    return build_review_packet(
        paths["evidence"],
        paths["intake"],
        paths["attestation"],
        paths["draft"],
        paths["validation"],
        paths["profile"],
        paths["pack"],
        held_out_path=paths["held_out"],
    )


def _checklist() -> dict[str, bool]:
    return {name: True for name in REQUIRED_CHECKLIST}


def test_review_packet_contains_every_review_surface_and_source_digest(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    packet.verify_digest()
    document = packet.document

    assert document["status"] == "ready_for_approval"
    assert document["blockers"] == []
    assert document["tools"][0]["mutates"] is True
    assert document["tools"][0]["requires_confirmation"] is True
    assert document["control_mapping"]["episode_binding"] == "argument"
    assert document["exclusions"][0]["source_name"] == "admin.delete"
    assert document["observations"]["calls_complete"] is True
    assert document["observations"]["state_deltas_complete"] is True
    assert document["fixture_policy"]["held_out"]["status"] == "declared"
    assert document["validation"]["conformance"]["conformance"]["effective_level"] == "L2"
    assert set(document["source_digests"]) == {
        "evidence_bundle",
        "intake_provenance",
        "gateway_attestation",
        "draft_provenance",
        "validation_report",
        "mcp_config",
        "held_out_policy",
        "canonical_pack",
    }
    assert document["canonical_pack"]["task_templates"][0]["id"] == "lookup"
    assert document["canonical_pack"]["validation_cases"][0]["id"] == "lookup_success"
    assert "def assert_trace" in document["canonical_pack"]["assertions"]["source"]
    assert {risk["source"] for risk in document["metadata_risks"]} == {
        "catalog_warning",
        "evidence_advisory",
    }


def test_blocked_packet_names_missing_evidence_and_cannot_be_approved(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path, blocked=True)
    assert packet.document["status"] == "blocked"
    assert {item["id"] for item in packet.document["blockers"]} >= {
        "assertions_not_compiled",
        "draft_unknowns",
        "endpoint_not_l2",
        "observed_calls_missing",
        "pack_not_gold",
        "state_deltas_missing",
        "validation_failures",
    }
    with pytest.raises(ReviewError, match="cannot approve a blocked review packet"):
        build_review_approval(
            packet,
            approved_by="domain-reviewer",
            reviewed_at="2026-08-26T17:00:00+07:00",
            checklist=_checklist(),
            acknowledged_risks=[risk["id"] for risk in packet.document["metadata_risks"]],
        )


def test_review_packet_refuses_a_profile_different_from_discovery(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    profile = yaml.safe_load(paths["profile"].read_text(encoding="utf-8"))
    profile["control"]["episode_argument"] = "different_episode"
    paths["profile"].write_text(yaml.safe_dump(profile), encoding="utf-8")

    with pytest.raises(ReviewError, match="MCP profile differs from discovery"):
        build_review_packet(
            paths["evidence"],
            paths["intake"],
            paths["attestation"],
            paths["draft"],
            paths["validation"],
            paths["profile"],
            paths["pack"],
        )


def test_packet_round_trip_rejects_tampering(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    path = write_review_packet(packet, tmp_path / "review_packet.json")
    assert load_review_packet(path).document == packet.document

    changed = copy.deepcopy(packet.document)
    changed["tools"][0]["mutates"] = False
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ReviewError, match="changed after packet_digest"):
        load_review_packet(path)


def test_json_loader_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"ready","status":"blocked"}', encoding="utf-8")
    with pytest.raises(ReviewError, match="duplicate JSON key"):
        load_json_mapping(duplicate, "test")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ReviewError, match="non-finite JSON constant"):
        load_json_mapping(nonfinite, "test")


def test_named_approval_pins_packet_checklist_and_every_risk(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    risk_ids = [risk["id"] for risk in packet.document["metadata_risks"]]
    approval = build_review_approval(
        packet,
        approved_by="domain-reviewer@example.test",
        reviewed_at="2026-08-26T17:00:00+07:00",
        checklist=_checklist(),
        acknowledged_risks=risk_ids,
        note="Reviewed against the domain contract.",
    )
    approval.verify_digest()
    assert approval.document["review_packet_digest"] == packet.digest
    assert approval.document["approved_by"] == "domain-reviewer@example.test"
    assert set(approval.document["checklist"]) == REQUIRED_CHECKLIST
    assert approval.document["acknowledged_risks"] == sorted(risk_ids)

    path = write_review_approval(approval, tmp_path / "approval.json")
    assert load_review_approval(path, packet).document == approval.document


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"checklist": {"semantics": False}}, "checklist mismatch"),
        ({"acknowledged_risks": []}, "risk acknowledgements mismatch"),
        ({"approved_by": "  "}, "must name the reviewer"),
        ({"reviewed_at": "2026-08-26"}, "must include a timezone"),
    ],
)
def test_approval_refuses_partial_or_unnamed_review(
    tmp_path: Path,
    change: dict[str, Any],
    match: str,
) -> None:
    packet = _packet(tmp_path)
    arguments: dict[str, Any] = {
        "approved_by": "domain-reviewer",
        "reviewed_at": "2026-08-26T17:00:00+07:00",
        "checklist": _checklist(),
        "acknowledged_risks": [risk["id"] for risk in packet.document["metadata_risks"]],
    }
    arguments.update(change)
    with pytest.raises(ReviewError, match=match):
        build_review_approval(packet, **arguments)


def test_approval_of_one_packet_cannot_be_reused_after_packet_changes(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    approval = build_review_approval(
        packet,
        approved_by="domain-reviewer",
        reviewed_at="2026-08-26T17:00:00+07:00",
        checklist=_checklist(),
        acknowledged_risks=[risk["id"] for risk in packet.document["metadata_risks"]],
    )
    approval_path = write_review_approval(approval, tmp_path / "approval.json")

    changed = copy.deepcopy(packet.document)
    changed["assumptions"].append("A newly introduced assumption.")
    changed["packet_digest"] = sha256_json({key: value for key, value in changed.items() if key != "packet_digest"})
    changed_packet = ReviewPacket(document=changed)
    with pytest.raises(ReviewError, match="different review packet"):
        load_review_approval(approval_path, changed_packet)


def test_approval_digest_detects_a_reviewer_name_edit(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    approval = build_review_approval(
        packet,
        approved_by="domain-reviewer",
        reviewed_at="2026-08-26T17:00:00+07:00",
        checklist=_checklist(),
        acknowledged_risks=[risk["id"] for risk in packet.document["metadata_risks"]],
    )
    changed = copy.deepcopy(approval.document)
    changed["approved_by"] = "different-reviewer"
    with pytest.raises(ReviewError, match="changed after approval_digest"):
        ReviewApproval(document=changed).verify_digest()


def _approved_freeze_inputs(tmp_path: Path) -> FreezeInputs:
    paths = _inputs(tmp_path)
    packet = build_review_packet(
        paths["evidence"],
        paths["intake"],
        paths["attestation"],
        paths["draft"],
        paths["validation"],
        paths["profile"],
        paths["pack"],
        held_out_path=paths["held_out"],
    )
    packet_path = write_review_packet(packet, tmp_path / "review_packet.json")
    approval = build_review_approval(
        packet,
        approved_by="domain-reviewer",
        reviewed_at="2026-08-26T17:00:00+07:00",
        checklist=_checklist(),
        acknowledged_risks=[risk["id"] for risk in packet.document["metadata_risks"]],
    )
    approval_path = write_review_approval(approval, tmp_path / "review_approval.json")
    return FreezeInputs(
        pack_root=paths["pack"],
        mcp_config_path=paths["profile"],
        evidence_path=paths["evidence"],
        intake_provenance_path=paths["intake"],
        gateway_attestation_path=paths["attestation"],
        draft_provenance_path=paths["draft"],
        validation_report_path=paths["validation"],
        review_packet_path=packet_path,
        review_approval_path=approval_path,
    )


def test_freeze_atomically_seals_the_canonical_pack_and_lineage(
    tmp_path: Path,
) -> None:
    inputs = _approved_freeze_inputs(tmp_path)
    release = freeze_canonical_pack(inputs, tmp_path / "release")
    verified = load_frozen_release(release.root)

    assert verified.manifest == release.manifest
    assert release.pack_fingerprint.startswith("sha256:")
    assert release.manifest["source_pack_fingerprint"] != release.manifest["frozen_pack_fingerprint"]
    assert (release.pack_root / "mcp_oracle.yaml").is_file()
    assert (release.pack_root / "provenance" / "mcp_lineage.json").is_file()
    assert (release.pack_root / "provenance" / "review_packet.json").is_file()
    assert stat.S_IMODE(release.pack_root.stat().st_mode) == 0o555
    assert stat.S_IMODE((release.pack_root / "manifest.yaml").stat().st_mode) == 0o444
    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=release.pack_root / "manifest.yaml"),
        (release.pack_root,),
    )
    origin = load_mcp_origin(
        paths,
        load_endpoint_config(
            paths.endpoint_config_path,  # type: ignore[arg-type]
            allowed_roots=(release.pack_root,),
        ),
        pack_fingerprint=release.pack_fingerprint,
        pack_id="acme-inventory",
        pack_version="1.0.0",
    )
    assert origin is not None
    assert origin["frozen_pack_fingerprint"] == release.pack_fingerprint
    assert origin["source_pack_fingerprint"] == release.manifest["source_pack_fingerprint"]
    assert origin["effective_content_digest"] == SHA_C
    assert "approved_by" not in origin
    assert "base_url" not in json.dumps(origin)
    assert origin == load_origin_provenance(
        paths,
        load_endpoint_config(
            paths.endpoint_config_path,  # type: ignore[arg-type]
            allowed_roots=(release.pack_root,),
        ),
        pack_fingerprint=release.pack_fingerprint,
        pack_id="acme-inventory",
        pack_version="1.0.0",
    )


def test_gold_freeze_requires_a2_source_certification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _approved_freeze_inputs(tmp_path)

    def checked_loader(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["required_certification_tier"] is AdapterTier.A2
        return load_evidence_bundle(*args, **kwargs)

    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.mcp.release.freeze.load_evidence_bundle",
        checked_loader,
    )
    freeze_canonical_pack(inputs, tmp_path / "release")


def test_generic_origin_resolution_rejects_ambiguous_provider_profiles(
    tmp_path: Path,
) -> None:
    release = freeze_canonical_pack(
        _approved_freeze_inputs(tmp_path),
        tmp_path / "release",
    )
    for path in release.pack_root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    release.pack_root.chmod(0o755)
    second_path = Path("provenance") / "http_lineage.json"
    (release.pack_root / second_path).write_text("{}\n", encoding="utf-8")
    http_profile = OriginProfile(
        schema_version="bfcl-http-publication-lineage-v1",
        provider_kind="http",
        origin="http_backed_endpoint",
        lineage_relative_path=second_path,
        label="HTTP lineage",
        allowed_profile_versions=("bfcl-http-oracle-v1",),
        identity_digest_fields=MCP_ORIGIN_PROFILE.identity_digest_fields,
        endpoint_content_field="effective_content_digest",
        endpoint_attestation_field="conformance_digest",
    )
    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=release.pack_root / "manifest.yaml"),
        (release.pack_root,),
    )

    with pytest.raises(OriginProvenanceError, match="origin_profile_ambiguous"):
        load_origin_provenance(
            paths,
            load_endpoint_config(
                paths.endpoint_config_path,  # type: ignore[arg-type]
                allowed_roots=(release.pack_root,),
            ),
            pack_fingerprint=release.pack_fingerprint,
            pack_id="acme-inventory",
            pack_version="1.0.0",
            profiles={
                MCP_ORIGIN_PROFILE.key: MCP_ORIGIN_PROFILE,
                http_profile.key: http_profile,
            },
        )


def test_generic_origin_resolution_rejects_unknown_provider_identity(
    tmp_path: Path,
) -> None:
    release = freeze_canonical_pack(
        _approved_freeze_inputs(tmp_path),
        tmp_path / "release",
    )
    for path in release.pack_root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    release.pack_root.chmod(0o755)
    lineage_path = release.pack_root / MCP_ORIGIN_PROFILE.lineage_relative_path
    document = json.loads(lineage_path.read_text(encoding="utf-8"))
    document["provider_kind"] = "unknown"
    document.pop("record_digest")
    document["record_digest"] = sha256_json(document)
    lineage_path.write_text(json.dumps(document), encoding="utf-8")
    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=release.pack_root / "manifest.yaml"),
        (release.pack_root,),
    )

    with pytest.raises(OriginProvenanceError, match="origin_profile_mismatch"):
        load_origin_provenance(
            paths,
            load_endpoint_config(
                paths.endpoint_config_path,  # type: ignore[arg-type]
                allowed_roots=(release.pack_root,),
            ),
            pack_fingerprint=release.pack_fingerprint,
            pack_id="acme-inventory",
            pack_version="1.0.0",
        )


def test_publication_origin_refuses_lineage_lifted_from_another_pack(
    tmp_path: Path,
) -> None:
    """A lineage file is copyable, so it must not be believable on its own."""
    release = freeze_canonical_pack(
        _approved_freeze_inputs(tmp_path),
        tmp_path / "release",
    )
    stolen = tmp_path / "stolen"
    shutil.copytree(release.pack_root, stolen)
    for path in stolen.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    manifest_path = stolen / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["pack_id"] = "impostor-inventory"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=manifest_path),
        (stolen,),
    )
    endpoint = load_endpoint_config(
        paths.endpoint_config_path,  # type: ignore[arg-type]
        allowed_roots=(stolen,),
    )

    with pytest.raises(OriginProvenanceError, match="names a different pack"):
        load_mcp_origin(
            paths,
            endpoint,
            pack_fingerprint=f"sha256:{pack_fingerprint(paths)}",
            pack_id="impostor-inventory",
            pack_version="1.0.0",
        )

    (stolen / "provenance" / "review_approval.json").unlink()
    manifest["pack_id"] = "acme-inventory"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(OriginProvenanceError, match="review approval must be a regular file"):
        load_mcp_origin(
            paths,
            endpoint,
            pack_fingerprint=f"sha256:{pack_fingerprint(paths)}",
            pack_id="acme-inventory",
            pack_version="1.0.0",
        )


def test_freeze_refuses_pack_drift_after_review(tmp_path: Path) -> None:
    inputs = _approved_freeze_inputs(tmp_path)
    (inputs.pack_root / "fixtures.json").write_text(
        '{"items":[{"id":"swapped"}]}',
        encoding="utf-8",
    )
    with pytest.raises(FreezeError, match="canonical_pack differs"):
        freeze_canonical_pack(inputs, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_freeze_refuses_symlinks_and_existing_destinations(tmp_path: Path) -> None:
    inputs = _approved_freeze_inputs(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("mutable", encoding="utf-8")
    (inputs.pack_root / "linked.txt").symlink_to(outside)
    with pytest.raises(
        (FreezeError, PermissionError),
        match="symbolic links",
    ):
        freeze_canonical_pack(inputs, tmp_path / "release")

    (inputs.pack_root / "linked.txt").unlink()
    destination = tmp_path / "release"
    destination.mkdir()
    with pytest.raises(FreezeError, match="already exists"):
        freeze_canonical_pack(inputs, destination)


def test_mcp_freeze_refuses_ignored_python_bytecode(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.mcp.release import freeze

    source = tmp_path / "pack"
    cache = source / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "backend.cpython-312.pyc").write_bytes(b"unchecked")

    with pytest.raises(FreezeError, match="executable Python bytecode"):
        freeze._copy_pack_tree(source, tmp_path / "destination")


def test_publication_handoff_requires_a_fresh_l2_gold_report(tmp_path: Path) -> None:
    release = freeze_canonical_pack(
        _approved_freeze_inputs(tmp_path),
        tmp_path / "release",
    )
    report = _validation()
    report["pack_fingerprint"] = release.pack_fingerprint.removeprefix("sha256:")
    report["stats"] = {
        "has_oracle": True,
        "n_templates": 1,
        "n_assertions": 1,
        "n_tools": 1,
    }
    _require_fresh_gold(report, release)

    report["extra_checks"][0]["conformance"]["effective_level"] = "L1"
    report["extra_checks"][0]["conformance"]["publishable"] = False
    with pytest.raises(HandoffError, match="independently verify endpoint L2"):
        _require_fresh_gold(report, release)


def test_publication_handoff_forces_prepare_and_uses_existing_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.mcp.release import handoff

    release = freeze_canonical_pack(
        _approved_freeze_inputs(tmp_path),
        tmp_path / "release",
    )
    output_root = tmp_path / "output"
    experiment = "mcp-publication"
    config = SimpleNamespace(output_dir=output_root, expt_name=experiment)
    observed_force: list[bool] = []
    report_path = tmp_path / "fresh-validation.json"

    def prepare(path: Path, *, force_validation: bool = False) -> Path:
        observed_force.append(force_validation)
        report = _validation()
        report["pack_fingerprint"] = release.pack_fingerprint.removeprefix("sha256:")
        report["stats"] = {
            "has_oracle": True,
            "n_templates": 1,
            "n_assertions": 1,
            "n_tools": 1,
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path

    def generate(path: Path) -> Path:
        publication = output_root / experiment
        publication.mkdir(parents=True)
        benchmark = publication / "benchmark.parquet"
        raw = publication / "benchmark_raw.parquet"
        manifest = publication / "run_manifest.json"
        benchmark.write_bytes(b"benchmark")
        raw.write_bytes(b"raw")
        manifest.write_text(
            json.dumps(
                {
                    "pack": {"content_hash": release.pack_fingerprint},
                    "oracle": {
                        "mcp": {
                            "frozen_pack_fingerprint": release.pack_fingerprint,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return benchmark

    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=release.pack_root / "manifest.yaml"),
        (release.pack_root,),
    )
    monkeypatch.setattr(
        handoff.BfclConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(handoff, "resolve_pack_paths", lambda config: paths)
    monkeypatch.setattr(handoff, "prepare_bfcl", prepare)
    monkeypatch.setattr(handoff, "generate_bfcl", generate)

    result = handoff_frozen_release(release.root, tmp_path / "config.yaml")

    assert observed_force == [True]
    assert result.benchmark_path.read_bytes() == b"benchmark"
    assert result.raw_benchmark_path.read_bytes() == b"raw"
    assert result.run_manifest["pack"]["content_hash"] == release.pack_fingerprint


def test_fresh_prepare_forces_validation_instead_of_reusing_process_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import (
        checkpoint,
        pipeline,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import (
        config as config_module,
    )

    fake_config = SimpleNamespace(output_dir=tmp_path, expt_name="fresh")
    report_path = tmp_path / "oracle_validation_report.json"
    report = {
        "checks": [{"status": "pass", "failures": []}],
        "extra_checks": [],
        "stats": {
            "has_oracle": True,
            "n_templates": 1,
            "n_assertions": 1,
            "n_tools": 1,
        },
    }
    observed: list[bool] = []

    monkeypatch.setattr(
        config_module.BfclConfig,
        "from_yaml",
        classmethod(lambda cls, path: fake_config),
    )
    monkeypatch.setattr(pipeline, "_invalidate_final_outputs", lambda config: None)
    monkeypatch.setattr(checkpoint, "clear_checkpoints", lambda config: None)

    def validate(config, *, force=False):  # type: ignore[no-untyped-def]
        observed.append(force)
        return report, report_path

    monkeypatch.setattr(pipeline, "_validate_pack", validate)
    assert pipeline._prepare_bfcl_unlocked("config.yaml", force_validation=True) == report_path
    assert observed == [True]
