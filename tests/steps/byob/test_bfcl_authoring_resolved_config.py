from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.authoring_workflow.resolved_config import (
    ResolvedConfigError,
    load_resolved_authoring_config,
    resolve_authoring_config,
    slug_pack_id_candidate,
    verify_resolved_authoring_inputs,
    write_resolved_authoring_config,
)
from nemotron.steps.byob.runtime.pack_authoring.authorization import (
    ExposureSubject,
    authorize_model_exposure_by_human,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "Customer Inventory"
    source.mkdir()
    (source / "backend.py").write_text("def list_items(): return []\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate reviewed inventory lookup.", encoding="utf-8")
    workspace = tmp_path / "workspace"
    return source, brief, workspace


def _assert_origins(value: object) -> None:
    if hasattr(value, "origin"):
        assert getattr(value, "origin") in {"user", "policy", "adapter", "derived"}
        return
    for field in type(value).model_fields:
        _assert_origins(getattr(value, field))


def test_resolution_is_canonical_deterministic_and_records_every_origin(
    tmp_path: Path,
) -> None:
    source, brief, workspace = _inputs(tmp_path)
    first = resolve_authoring_config(
        adapter_kind="local_python",
        source=source,
        domain_brief=brief,
        workspace=workspace,
        tenant_id="tenant",
        run_id="run",
        pack_id="inventory",
        pack_version="1.0.0",
        ci=True,
    )
    second = resolve_authoring_config(
        adapter_kind="local_python",
        source=source,
        domain_brief=brief,
        workspace=workspace,
        tenant_id="tenant",
        run_id="run",
        pack_id="inventory",
        pack_version="1.0.0",
        ci=True,
    )

    assert first == second
    assert first.resolved_authoring_config_digest == (
        second.resolved_authoring_config_digest
    )
    assert first.semantic_payload.pack_id_candidates.value == ("customer-inventory",)
    for section in (
        first.inputs,
        first.semantic_payload,
        first.resolved_paths,
        first.confirmations,
    ):
        _assert_origins(section)


def test_resolved_source_and_domain_brief_cannot_be_substituted(
    tmp_path: Path,
) -> None:
    source, brief, workspace = _inputs(tmp_path)
    resolved = resolve_authoring_config(
        adapter_kind="local_python",
        source=source,
        domain_brief=brief,
        workspace=workspace,
        tenant_id="tenant",
        run_id="run",
        pack_id="inventory",
        pack_version="1.0.0",
        ci=True,
    )
    verify_resolved_authoring_inputs(
        resolved,
        adapter_kind="local_python",
        source=source,
        domain_brief=brief,
    )
    brief.write_text("Changed after resolution.", encoding="utf-8")
    with pytest.raises(ResolvedConfigError) as raised:
        verify_resolved_authoring_inputs(
            resolved,
            adapter_kind="local_python",
            source=source,
            domain_brief=brief,
        )
    assert raised.value.code == "resolved_input_digest_mismatch"


def test_derived_pack_id_requires_confirmation_in_ci(tmp_path: Path) -> None:
    source, brief, workspace = _inputs(tmp_path)

    with pytest.raises(ResolvedConfigError) as raised:
        resolve_authoring_config(
            adapter_kind="local_python",
            source=source,
            domain_brief=brief,
            workspace=workspace,
            tenant_id="tenant",
            run_id="run",
            pack_version="1.0.0",
            ci=True,
        )

    assert raised.value.code == "pack_id_confirmation_required"
    assert "--confirm-pack-id" in raised.value.recovery


def test_policy_version_is_authoritative_but_pack_id_still_confirmed(
    tmp_path: Path,
) -> None:
    source, brief, workspace = _inputs(tmp_path)
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            (
                "schema_version: bfcl-authoring-policy-v1",
                "pack_id: inventory",
                'pack_version: "2.0.0"',
                "required_certification_tier: A1",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_authoring_config(
        adapter_kind="local_python",
        source=source,
        domain_brief=brief,
        workspace=workspace,
        tenant_id="tenant",
        run_id="run",
        policy_path=policy,
        confirm_pack_id=True,
        ci=True,
    )

    assert resolved.semantic_payload.pack_id.value == "inventory"
    assert resolved.semantic_payload.pack_id.origin == "user"
    assert resolved.semantic_payload.pack_version.value == "2.0.0"
    assert resolved.semantic_payload.pack_version.origin == "policy"
    assert resolved.confirmations.pack_version_confirmed.value is False
    assert resolved.semantic_payload.required_certification_tier.value == "A1"


def test_server_prose_never_supplies_pack_version(tmp_path: Path) -> None:
    source = tmp_path / "mcp_intake.yaml"
    source.write_text(
        "description: server_version 99.0.0\n",
        encoding="utf-8",
    )
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")

    with pytest.raises(ResolvedConfigError) as raised:
        resolve_authoring_config(
            adapter_kind="mcp_mode_a",
            source=source,
            domain_brief=brief,
            workspace=tmp_path / "workspace",
            tenant_id="tenant",
            run_id="run",
            pack_id="mcp-tools",
            ci=True,
        )

    assert raised.value.code == "pack_version_confirmation_required"


def test_write_is_immutable_and_tampering_fails_digest_check(tmp_path: Path) -> None:
    source, brief, workspace = _inputs(tmp_path)
    resolved = resolve_authoring_config(
        adapter_kind="local_python",
        source=source,
        domain_brief=brief,
        workspace=workspace,
        tenant_id="tenant",
        run_id="run",
        pack_id="inventory",
        pack_version="1.0.0",
        ci=True,
    )
    path = write_resolved_authoring_config(
        resolved,
        workspace / "resolved_authoring_config.json",
    )
    assert load_resolved_authoring_config(path) == resolved
    assert write_resolved_authoring_config(resolved, path) == path

    document = json.loads(path.read_text(encoding="utf-8"))
    document["semantic_payload"]["pack_version"]["value"] = "1.0.1"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ResolvedConfigError) as raised:
        load_resolved_authoring_config(path)
    assert raised.value.code == "resolved_config_invalid"


def test_config_digest_changes_model_exposure_authorization() -> None:
    base = {
        "evidence_digest": "sha256:" + "1" * 64,
        "domain_brief_content_digest": "sha256:" + "2" * 64,
        "domain_brief_source_digest": "sha256:" + "3" * 64,
        "domain_brief_redaction_report_digest": "sha256:" + "4" * 64,
        "held_out_decision_digest": "sha256:" + "5" * 64,
        "held_out_policy_digest": None,
        "held_out_redaction_report_digest": "sha256:" + "6" * 64,
    }
    first = authorize_model_exposure_by_human(
        ExposureSubject(
            **base,
            resolved_authoring_config_digest="sha256:" + "a" * 64,
        ),
        authorized_by="reviewer@example.test",
    )
    second = authorize_model_exposure_by_human(
        ExposureSubject(
            **base,
            resolved_authoring_config_digest="sha256:" + "b" * 64,
        ),
        authorized_by="reviewer@example.test",
    )

    assert first.authorization_digest != second.authorization_digest


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Customer Inventory", "customer-inventory"),
        ("123 tools", "pack-123-tools"),
        ("Đơn hàng", "on-hang"),
    ],
)
def test_pack_id_candidate_is_deterministic(value: str, expected: str) -> None:
    assert slug_pack_id_candidate(value) == expected
