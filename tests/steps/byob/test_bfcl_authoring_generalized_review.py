from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from nemotron.steps.byob.runtime.authoring_release import assembly
from nemotron.steps.byob.runtime.authoring_release.assembly import (
    ReviewAssemblyError,
    ReviewContext,
    assemble_review,
)
from nemotron.steps.byob.runtime.authoring_release.review import (
    REQUIRED_CHECKLIST_V2,
    AuthoringReviewError,
    build_review_approval,
    build_review_packet,
)
from nemotron.steps.byob.runtime.authoring_workflow.resolved_config import (
    resolve_authoring_config,
    write_resolved_authoring_config,
)
from nemotron.steps.byob.runtime.mcp.release.adapter import McpReleaseAdapter
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    authorize_model_exposure_by_human,
    build_exposure_subject,
    write_exposure_authorization,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.source_adapters.evidence import load_source_evidence
from tests.steps.byob.test_bfcl_mcp_release_review import SHA_A, _inputs
from tests.steps.byob.test_bfcl_source_intake import _contract_case


def _candidate_pack(tmp_path: Path, adapter: str, pack_id: str, version: str) -> Path:
    fixture_root = tmp_path / "candidate-fixture"
    fixture_root.mkdir()
    pack = _inputs(fixture_root)["pack"]
    manifest_path = pack / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["pack_id"] = pack_id
    manifest["version"] = version
    if adapter == "local_python":
        (pack / "endpoint_config.yaml").unlink()
        (pack / "backend.py").write_text(
            "def inventory_lookup(id: str):\n    return {'id': id}\n",
            encoding="utf-8",
        )
        manifest["paths"] = {"backend": "backend.py"}
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return pack


class _ReviewCase(NamedTuple):
    context: ReviewContext
    pack: Path
    authorization_digest: str


def _review_case(tmp_path: Path, adapter_kind: str) -> _ReviewCase:
    intake = _contract_case(tmp_path, adapter_kind)
    evidence = load_source_evidence(intake.evidence_path)
    source = tmp_path / f"{adapter_kind}-source"
    source.write_text("reviewed source descriptor\n", encoding="utf-8")
    brief = intake.domain_brief_source_path
    resolved = resolve_authoring_config(
        adapter_kind=adapter_kind,
        source=source,
        domain_brief=brief,
        workspace=tmp_path / "workspace",
        tenant_id="tenant",
        run_id="run",
        pack_id=evidence.pack.pack_id,
        pack_version=evidence.pack.version,
        ci=True,
    )
    resolved_path = write_resolved_authoring_config(
        resolved,
        tmp_path / "resolved_authoring_config.json",
    )
    source_bundle = intake.output_root / "evidence_bundle.v1.json"
    migration = intake.output_root / "evidence_migration.json"
    view = load_evidence_bundle(
        intake.evidence_path,
        certification_report_path=intake.certification_path,
        trusted_certification_keys={
            intake.authority.key_id: intake.authority.public_key
        },
        domain_brief_source_path=intake.domain_brief_source_path,
        domain_brief_report_path=intake.domain_brief_report_path,
        held_out_redaction_report_path=intake.held_out_redaction_path,
        source_bundle_path=source_bundle if source_bundle.exists() else None,
        migration_record_path=migration if migration.exists() else None,
        source_observations_path=intake.observations_path,
    )
    assert view.source_evidence is not None
    assert view.domain_brief_report is not None
    assert view.held_out_redaction_report is not None
    subject = build_exposure_subject(
        view.source_evidence,
        domain_brief_report=view.domain_brief_report,
        held_out_redaction_report=view.held_out_redaction_report,
        resolved_authoring_config_digest=resolved.resolved_authoring_config_digest,
    )
    authorization = authorize_model_exposure_by_human(
        subject,
        authorized_by="reviewer@example.test",
    )
    authorization_path = write_exposure_authorization(
        authorization,
        tmp_path / "exposure_authorization.json",
    )
    draft_document = {
        "schema_version": "bfcl-authoring-draft-provenance-v1",
        "evidence": {"bundle_digest": evidence.bundle_digest},
        "model_exposure_authorization": authorization.model_dump(mode="json"),
        "resolved_authoring_config_digest": (
            resolved.resolved_authoring_config_digest
        ),
        "blocked_on": [],
        "assertions_compiled": True,
    }
    draft_document["record_digest"] = sha256_json(draft_document)
    draft_path = write_canonical_json(
        draft_document,
        tmp_path / "draft_provenance.json",
    )
    pack = _candidate_pack(
        tmp_path,
        adapter_kind,
        evidence.pack.pack_id,
        evidence.pack.version,
    )
    validation_path = write_canonical_json(
        {
            "schema_version": "test-validation-v1",
            "pack_fingerprint": "intentionally-stale",
            "checks": [],
            "extra_checks": [],
        },
        tmp_path / "validation.json",
    )
    context = ReviewContext(
        evidence_path=intake.evidence_path,
        certification_report_path=intake.certification_path,
        trusted_certification_keys={
            intake.authority.key_id: intake.authority.public_key
        },
        domain_brief_source_path=intake.domain_brief_source_path,
        domain_brief_report_path=intake.domain_brief_report_path,
        held_out_redaction_report_path=intake.held_out_redaction_path,
        source_observations_path=intake.observations_path,
        intake_provenance_path=intake.output_root / "intake_provenance.json",
        draft_provenance_path=draft_path,
        validation_report_path=validation_path,
        resolved_authoring_config_path=resolved_path,
        exposure_authorization_path=authorization_path,
        source_bundle_path=source_bundle if source_bundle.exists() else None,
        migration_record_path=migration if migration.exists() else None,
    )
    return _ReviewCase(
        context=context,
        pack=pack,
        authorization_digest=authorization.authorization_digest,
    )


@pytest.mark.parametrize(
    "adapter_kind",
    ["local_python", "http_package", "mcp_mode_a"],
)
def test_generalized_review_verifies_common_trust_records_for_every_adapter(
    tmp_path: Path,
    adapter_kind: str,
) -> None:
    case = _review_case(tmp_path, adapter_kind)

    assembled = assemble_review(
        adapter_kind=adapter_kind,
        pack_root=case.pack,
        context=case.context,
    )

    packet = assembled.packet.document
    assert packet["adapter_kind"] == adapter_kind
    assert packet["adapter_review"]["authoring"][
        "model_exposure_authorization_digest"
    ] == case.authorization_digest
    assert packet["adapter_review"]["authoring"]["questions_status"] == "not_required"
    assert "certification_report" in packet["source_digests"]
    assert "model_exposure_authorization" in packet["source_digests"]
    assert {blocker["code"] for blocker in packet["blockers"]} >= {
        "adapter_under_certified",
        "validation_authority_missing",
        "validation_pack_mismatch",
    }


def test_review_refuses_a_validation_report_the_bound_config_did_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _review_case(tmp_path, "local_python")
    validation_config = tmp_path / "bfcl-validation.yaml"
    validation_config.write_text("reviewed: true\n", encoding="utf-8")
    elsewhere = tmp_path / "unreviewed" / "validation.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text("{}\n", encoding="utf-8")
    observed: list[bool] = []

    def fresh_prepare(_config: Path, *, force_validation: bool = False) -> Path:
        observed.append(force_validation)
        return elsewhere

    monkeypatch.setattr(assembly, "prepare_bfcl", fresh_prepare)

    with pytest.raises(ReviewAssemblyError) as refused:
        assemble_review(
            adapter_kind="local_python",
            pack_root=case.pack,
            context=replace(
                case.context,
                validation_config_path=validation_config,
            ),
        )

    assert refused.value.code == "validation_report_path_mismatch"
    assert refused.value.recovery
    assert observed == [True]


def test_final_approval_cannot_replace_pre_model_authorization(
    tmp_path: Path,
) -> None:
    pack = _candidate_pack(tmp_path, "http_package", "acme-inventory", "1.0.0")
    packet = build_review_packet(
        adapter=McpReleaseAdapter(
            identity_digest=SHA_A,
            review_data={
                "certification": {"report_digest": SHA_A},
                "authoring": {
                    "model_exposure_authorization_digest": None,
                    "questions_status": "not_required",
                },
                "freeze_sidecars": {},
            },
        ),
        pack_root=pack,
        source_digests={"certification_report": SHA_A},
    )

    with pytest.raises(AuthoringReviewError) as raised:
        build_review_approval(
            packet,
            approved_by="release-reviewer",
            reviewed_at="2026-08-29T15:00:00+07:00",
            checklist={name: True for name in REQUIRED_CHECKLIST_V2},
        )
    assert raised.value.code == "pre_model_authorization_missing"


def test_answered_revision_artifacts_must_all_be_bound(tmp_path: Path) -> None:
    pack = _candidate_pack(tmp_path, "http_package", "acme-inventory", "1.0.0")
    packet = build_review_packet(
        adapter=McpReleaseAdapter(
            identity_digest=SHA_A,
            review_data={
                "certification": {"report_digest": SHA_A},
                "authoring": {
                    "model_exposure_authorization_digest": SHA_A,
                    "questions_status": "answered",
                },
                "freeze_sidecars": {},
            },
        ),
        pack_root=pack,
        source_digests={
            "certification_report": SHA_A,
            "model_exposure_authorization": SHA_A,
        },
    )

    with pytest.raises(AuthoringReviewError) as raised:
        build_review_approval(
            packet,
            approved_by="release-reviewer",
            reviewed_at="2026-08-29T15:00:00+07:00",
            checklist={name: True for name in REQUIRED_CHECKLIST_V2},
        )
    assert raised.value.code == "answered_questions_missing"
