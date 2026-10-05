from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.authoring_release.contracts import (
    AdapterReviewContribution,
    FreezeHookContext,
)
from nemotron.steps.byob.runtime.authoring_release.freeze import (
    AuthoringFreezeError,
    FreezeInputsV2,
    FrozenReleaseV2,
    freeze_canonical_pack,
    load_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.handoff import AuthoringHandoffError, handoff_frozen_release
from nemotron.steps.byob.runtime.authoring_release.publication import (
    BfclPublicationAdapter,
    publication_adapter_for_release,
)
from nemotron.steps.byob.runtime.authoring_release.review import (
    REQUIRED_CHECKLIST_V2,
    build_review_approval,
    build_review_packet,
    write_review_approval,
    write_review_packet,
)
from nemotron.steps.byob.runtime.authoring_release.versions import (
    FREEZE_MANIFEST_VERSION_V3,
)
from nemotron.steps.byob.runtime.authoring_workflow.events import (
    EVENT_FILE_NAME,
    load_authoring_events,
)
from nemotron.steps.byob.runtime.authoring_workflow.resume import (
    bind_artifact,
    build_session_state,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
    BYOB_ROOT,
    OraclePackRef,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.origin_provenance import (
    load_origin_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.mcp.release.adapter import McpReleaseAdapter
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import (
    BundleError,
    load_evidence_bundle,
)
from nemotron.steps.byob.runtime.release_seal import ReleaseSealAuthority
from nemotron.steps.byob.scripts import bfcl_author
from tests.steps.byob.test_bfcl_authoring_generalized_review import _candidate_pack
from tests.steps.byob.test_bfcl_authoring_revisions import _committed_session
from tests.steps.byob.test_bfcl_mcp_release_review import _validation
from tests.steps.byob.test_bfcl_source_intake import _contract_case

SEAL_AUTHORITY = ReleaseSealAuthority(
    issuer="bfcl-release-authority",
    key_id="release-key-1",
    private_key=Ed25519PrivateKey.generate(),
)
SEAL_VERIFICATION = {
    "trusted_seal_keys": {SEAL_AUTHORITY.key_id: SEAL_AUTHORITY.public_key},
    "expected_seal_issuer": SEAL_AUTHORITY.issuer,
}


def _write_seal_public_key(root: Path) -> Path:
    path = root / "release-seal-public-key.pem"
    path.write_bytes(
        SEAL_AUTHORITY.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return path


_SHA = "sha256:" + "a" * 64


@dataclass(frozen=True)
class _EnvelopeAdapter:
    adapter_kind: str
    tier: str = "A2"

    @property
    def kind(self) -> str:
        return self.adapter_kind

    def validate_pack(self, pack_root: Path) -> str:
        if self.adapter_kind == "mcp_mode_a":
            return McpReleaseAdapter(identity_digest=_SHA).validate_pack(pack_root)
        return BfclPublicationAdapter(self.adapter_kind).validate_pack(pack_root)  # type: ignore[arg-type]

    def review(
        self,
        pack_root: Path,
        source_digests: Mapping[str, str],
    ) -> AdapterReviewContribution:
        del pack_root, source_digests
        return AdapterReviewContribution(
            identity_digest=_SHA,
            certification_tier=self.tier,
            review_data={
                "certification": {"report_digest": _SHA},
                "authoring": {
                    "model_exposure_authorization_digest": _SHA,
                    "questions_status": "not_required",
                },
                "freeze_sidecars": {},
            },
        )

    def freeze_sidecars(
        self,
        context: FreezeHookContext,
    ) -> Mapping[str, bytes]:
        del context
        return {}


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _approved_release(
    tmp_path: Path,
    adapter_kind: str,
    *,
    tier: str = "A2",
    pack_source: Path | None = None,
) -> tuple[FrozenReleaseV2, Path, Path]:
    root = tmp_path / adapter_kind
    root.mkdir(parents=True)
    if pack_source is None:
        pack = _candidate_pack(root, adapter_kind, "acme-inventory", "1.0.0")
    else:
        pack = root / "pack"
        shutil.copytree(pack_source, pack)
    source_record = root / "reviewed-source.json"
    source_record.write_text('{"reviewed":true}\n', encoding="utf-8")
    adapter = _EnvelopeAdapter(adapter_kind, tier)
    packet = build_review_packet(
        adapter=adapter,
        pack_root=pack,
        source_digests={
            "certification_report": _digest(source_record),
            "model_exposure_authorization": _digest(source_record),
        },
    )
    packet_path = write_review_packet(packet, root / "review-packet.json")
    approval = build_review_approval(
        packet,
        approved_by="release-reviewer",
        reviewed_at="2026-08-29T15:00:00+07:00",
        checklist={name: True for name in REQUIRED_CHECKLIST_V2},
    )
    approval_path = write_review_approval(approval, root / "review-approval.json")
    release = freeze_canonical_pack(
        FreezeInputsV2(
            pack_root=pack,
            review_packet_path=packet_path,
            review_approval_path=approval_path,
            source_records={
                "certification_report": source_record,
                "model_exposure_authorization": source_record,
            },
            seal_authority=SEAL_AUTHORITY,
        ),
        root / "release",
        adapter=adapter,
    )
    return release, packet_path, approval_path


def _install_frozen_session(
    tmp_path: Path,
    release: FrozenReleaseV2,
    packet_path: Path,
    approval_path: Path,
) -> tuple[Path, FrozenReleaseV2]:
    (tmp_path / "session").mkdir()
    workspace, gate, parent_digest, _ = _committed_session(
        tmp_path / "session",
        phase="evidence_approved",
    )
    frozen_root = workspace / "release"
    shutil.copytree(release.root, frozen_root, copy_function=shutil.copyfile)
    packet = workspace / "review_packet.json"
    approval = workspace / "release_approval.json"
    shutil.copyfile(packet_path, packet)
    shutil.copyfile(approval_path, approval)
    draft_root = workspace / "drafts"
    draft_root.mkdir()
    (draft_root / "task_templates.yaml").write_text("[]\n", encoding="utf-8")
    provenance_document: dict[str, Any] = {"schema_version": "test-draft-v1"}
    provenance_document["record_digest"] = sha256_json(provenance_document)
    provenance = write_canonical_json(
        provenance_document,
        workspace / "draft_provenance.json",
    )
    parent = gate.load_state(parent_digest)
    bindings = parent.bindings.model_copy(
        update={
            "draft_root": "drafts",
            "draft_provenance": bind_artifact(
                workspace,
                provenance,
                digest_kind="canonical_json",
            ),
            "review_packet": bind_artifact(
                workspace,
                packet,
                digest_kind="canonical_json",
            ),
            "release_approval": bind_artifact(
                workspace,
                approval,
                digest_kind="canonical_json",
            ),
            "frozen_manifest": bind_artifact(
                workspace,
                frozen_root / "freeze_manifest.json",
                digest_kind="canonical_json",
            ),
        }
    )
    state = build_session_state(
        tenant_id="tenant-a",
        run_id="run-a",
        phase="frozen",
        bindings=bindings,
        parent_session_digest=parent_digest,
    )
    lease = gate.workspace_lock.acquire()
    try:
        gate.commit_state(state, lease=lease)
    finally:
        lease.release()
    write_canonical_json(
        {
            "schema_version": "bfcl-authoring-head-v1",
            "tenant_id": "tenant-a",
            "run_id": "run-a",
            "phase": "frozen",
            "session_digest": state.session_digest,
        },
        workspace / "authoring_head.json",
    )
    loaded = load_frozen_release(frozen_root, **SEAL_VERIFICATION)
    assert isinstance(loaded, FrozenReleaseV2)
    return workspace, loaded


def test_all_adapters_share_the_v2_release_envelope_through_freeze(
    tmp_path: Path,
) -> None:
    releases = [_approved_release(tmp_path, adapter) for adapter in ("local_python", "http_package", "mcp_mode_a")]

    assert {release.manifest["schema_version"] for release, _, _ in releases} == {FREEZE_MANIFEST_VERSION_V3}
    assert len({frozenset(release.manifest) for release, _, _ in releases}) == 1
    assert len({frozenset(release.manifest["source_records"]) for release, _, _ in releases}) == 1
    with pytest.raises(AuthoringHandoffError) as http_publication:
        publication_adapter_for_release(
            releases[1][0].root,
            **SEAL_VERIFICATION,
        )
    assert http_publication.value.code == "publication_adapter_unsupported"
    for release, _, _ in releases:
        paths = resolve_declared_pack_paths(
            OraclePackRef(manifest_path=release.pack_root / "manifest.yaml"),
            (release.pack_root,),
        )
        origin = load_origin_provenance(
            paths,
            None,
            pack_fingerprint=release.pack_fingerprint,
            pack_id="acme-inventory",
            pack_version="1.0.0",
        )
        assert origin is not None
        assert origin["adapter_kind"] == release.adapter_kind
        assert origin["frozen_pack_fingerprint"] == release.pack_fingerprint


@pytest.mark.parametrize("adapter_kind", ["local_python", "mcp_mode_a"])
def test_guided_publish_completes_stage_all_with_fresh_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    adapter_kind: str,
) -> None:
    built, packet_path, approval_path = _approved_release(tmp_path, adapter_kind)
    workspace, release = _install_frozen_session(
        tmp_path,
        built,
        packet_path,
        approval_path,
    )
    output = workspace / f"{adapter_kind}-published"
    published_output = output / f"{adapter_kind}-e2e"
    config = workspace / f"{adapter_kind}-config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "family": "bfcl",
                "expt_name": f"{adapter_kind}-e2e",
                "stage": "all",
                "random_seed": 7,
                "config_status": "resolved",
                "output_dir": str(output),
                "oracle_pack": {"manifest_path": str(release.pack_root / "manifest.yaml")},
                "oracle_runtime": {
                    "clock": "2026-08-29T15:00:00+07:00",
                    "allowed_roots": [str(release.pack_root)],
                },
                "lineage": {
                    "policy": "smoke_no_publication",
                    "profile_influenced_surface": False,
                    "judge_advisory": None,
                    "roles": {
                        role: {"enabled": False, "model_config": None}
                        for role in ("profile", "paraphrase", "surface_judge")
                    },
                },
                "surface_generation": {
                    "model_paraphrase_enabled": False,
                    "paraphrases_per_template": 0,
                    "preserve_slot_values": True,
                    "prevent_tool_name_leakage": True,
                },
                "surface_quality_validation": {
                    "enabled": False,
                    "drop_authority": False,
                },
                "task_generation": {"tasks_per_category": 1},
                "semantic_deduplication_config": {"enabled": False},
                "exports": {
                    "bfcl_json": False,
                    "nemo_evaluator_bundle": False,
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_prepare(config_path: Path, *, force_validation: bool) -> Path:
        assert force_validation is True
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["stage"] == "all"
        calls.append("prepare")
        report = tmp_path / f"{adapter_kind}-validation.json"
        validation = _validation(ready=True)
        validation["pack_fingerprint"] = release.pack_fingerprint.removeprefix("sha256:")
        validation["stats"] = {
            "has_oracle": True,
            "n_templates": 1,
            "n_assertions": 1,
            "n_tools": 1,
        }
        report.write_text(
            json.dumps(validation),
            encoding="utf-8",
        )
        return report

    def fake_generate(config_path: Path) -> Path:
        del config_path
        calls.append("generate")
        published_output.mkdir(parents=True)
        benchmark = published_output / "benchmark.parquet"
        benchmark.write_bytes(b"published")
        (published_output / "benchmark_raw.parquet").write_bytes(b"raw")
        manifest: dict[str, Any] = {
            "run_id": f"{adapter_kind}-run",
            "pack": {"content_hash": release.pack_fingerprint},
            "tier": "gold",
            "gold_eligible": True,
            "gold_ineligibility_reasons": [],
            "oracle": {
                "origin": {
                    "provider_kind": "bfcl_authoring",
                    "adapter_kind": adapter_kind,
                    "frozen_pack_fingerprint": release.pack_fingerprint,
                }
            },
        }
        (published_output / "run_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return benchmark

    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.authoring_release.publication.prepare_bfcl",
        fake_prepare,
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.authoring_release.publication.generate_bfcl",
        fake_generate,
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.mcp.release.adapter.prepare_bfcl",
        fake_prepare,
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.mcp.release.adapter.generate_bfcl",
        fake_generate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "publish",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--adapter-kind",
            adapter_kind,
            "--release",
            str(release.root),
            "--config",
            str(config),
            "--seal-issuer",
            SEAL_AUTHORITY.issuer,
            "--seal-public-key",
            str(_write_seal_public_key(tmp_path)),
            "--seal-key-id",
            SEAL_AUTHORITY.key_id,
        ],
    )

    bfcl_author.main()

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "published"
    assert result["adapter"] == adapter_kind
    assert calls == ["prepare", "generate"]
    assert Path(result["benchmark"]).is_file()
    assert Path(result["run_manifest"]).is_file()


def test_real_local_guided_publication_runs_stage_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tiny_pack = BYOB_ROOT / "data" / "tiny_oracle_pack"
    built, packet_path, approval_path = _approved_release(
        tmp_path,
        "local_python",
        pack_source=tiny_pack,
    )
    workspace, release = _install_frozen_session(
        tmp_path,
        built,
        packet_path,
        approval_path,
    )
    config_document = yaml.safe_load((BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    output = workspace / "real-stage-all"
    config_document["expt_name"] = "local-authoring-real-e2e"
    config_document["output_dir"] = str(output)
    config_document["oracle_pack"]["manifest_path"] = str(release.pack_root / "manifest.yaml")
    config_document["oracle_runtime"]["allowed_roots"] = [str(release.pack_root)]
    config_document["lineage"]["policy"] = "strict_separation"
    config = workspace / "real-local-config.yaml"
    config.write_text(yaml.safe_dump(config_document), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "publish",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--release",
            str(release.root),
            "--config",
            str(config),
            "--seal-issuer",
            SEAL_AUTHORITY.issuer,
            "--seal-public-key",
            str(_write_seal_public_key(tmp_path)),
            "--seal-key-id",
            SEAL_AUTHORITY.key_id,
        ],
    )

    bfcl_author.main()

    result = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))
    assert result["status"] == "published"
    assert manifest["tier"] == "gold"
    assert manifest["gold_eligible"] is True
    assert manifest["gold_ineligibility_reasons"] == []
    assert manifest["oracle"]["origin"]["adapter_kind"] == "local_python"
    events = load_authoring_events(workspace / ".events" / EVENT_FILE_NAME)
    assert [event.event_type for event in events] == ["validation_verdict"]
    assert events[0].payload["stage"] == "publication"


def test_below_a2_and_stale_approval_fail_before_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "negative"
    root.mkdir()
    pack = _candidate_pack(root, "local_python", "acme-inventory", "1.0.0")
    source = root / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    a1_adapter = _EnvelopeAdapter("local_python", "A1")
    packet = build_review_packet(
        adapter=a1_adapter,
        pack_root=pack,
        source_digests={
            "certification_report": _digest(source),
            "model_exposure_authorization": _digest(source),
        },
    )
    packet_path = write_review_packet(packet, root / "packet-a1.json")
    approval = build_review_approval(
        packet,
        approved_by="reviewer",
        reviewed_at="2026-08-29T15:00:00+07:00",
        checklist={name: True for name in REQUIRED_CHECKLIST_V2},
    )
    approval_path = write_review_approval(approval, root / "approval-a1.json")
    inputs = FreezeInputsV2(
        pack_root=pack,
        review_packet_path=packet_path,
        review_approval_path=approval_path,
        source_records={
            "certification_report": source,
            "model_exposure_authorization": source,
        },
        seal_authority=SEAL_AUTHORITY,
    )
    with pytest.raises(AuthoringFreezeError) as under_certified:
        freeze_canonical_pack(inputs, root / "under-certified", adapter=a1_adapter)
    assert under_certified.value.code == "adapter_under_certified"

    a2_adapter = _EnvelopeAdapter("local_python")
    fresh_packet = build_review_packet(
        adapter=a2_adapter,
        pack_root=pack,
        source_digests=packet.document["source_digests"],
    )
    fresh_packet_path = write_review_packet(fresh_packet, root / "packet-a2.json")
    stale_inputs = FreezeInputsV2(
        pack_root=pack,
        review_packet_path=fresh_packet_path,
        review_approval_path=approval_path,
        source_records=inputs.source_records,
        seal_authority=SEAL_AUTHORITY,
    )
    with pytest.raises(AuthoringFreezeError) as stale:
        freeze_canonical_pack(stale_inputs, root / "stale", adapter=a2_adapter)
    assert stale.value.code == "review_approval_stale"


def test_changed_domain_brief_invalidates_evidence_before_publication(
    tmp_path: Path,
) -> None:
    case = _contract_case(tmp_path, "local_python")
    load_evidence_bundle(
        case.evidence_path,
        certification_report_path=case.certification_path,
        trusted_certification_keys={
            case.authority.key_id: case.authority.public_key,
        },
        domain_brief_source_path=case.domain_brief_source_path,
        domain_brief_report_path=case.domain_brief_report_path,
        held_out_redaction_report_path=case.held_out_redaction_path,
        source_observations_path=case.observations_path,
    )
    case.domain_brief_source_path.write_text(
        "Changed scope after model-exposure authorization.\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleError):
        load_evidence_bundle(
            case.evidence_path,
            certification_report_path=case.certification_path,
            trusted_certification_keys={
                case.authority.key_id: case.authority.public_key,
            },
            domain_brief_source_path=case.domain_brief_source_path,
            domain_brief_report_path=case.domain_brief_report_path,
            held_out_redaction_report_path=case.held_out_redaction_path,
            source_observations_path=case.observations_path,
        )


def test_handoff_rejects_a_non_publishable_generated_manifest(
    tmp_path: Path,
) -> None:
    release, _, _ = _approved_release(tmp_path, "local_python")
    output = tmp_path / "not-publishable"

    class Adapter:
        kind = "local_python"

        def validate_pack(self, _pack_root: Path) -> str:
            return release.pack_fingerprint

        def bind_config(self, *_args: object) -> None:
            return None

        def prepare(self, _config_path: Path) -> Path:
            report = tmp_path / "validation.json"
            report.write_text("{}", encoding="utf-8")
            return report

        def load_validation_report(self, _report_path: Path) -> Mapping[str, Any]:
            return {}

        def require_fresh_gold(self, *_args: object) -> None:
            return None

        def generate(self, _config_path: Path) -> Path:
            output.mkdir()
            benchmark = output / "benchmark.parquet"
            benchmark.write_bytes(b"benchmark")
            (output / "benchmark_raw.parquet").write_bytes(b"raw")
            write_canonical_json(
                {
                    "pack": {"content_hash": release.pack_fingerprint},
                    "tier": "gold",
                    "gold_eligible": False,
                    "gold_ineligibility_reasons": ["smoke_no_publication"],
                },
                output / "run_manifest.json",
            )
            return benchmark

        def verify_publication_manifest(self, *_args: object) -> None:
            pytest.fail("ineligible manifest reached adapter verification")

    with pytest.raises(AuthoringHandoffError) as raised:
        handoff_frozen_release(
            release.root,
            tmp_path / "config.yaml",
            adapter=Adapter(),
            **SEAL_VERIFICATION,
        )
    assert raised.value.code == "publication_not_gold_eligible"
