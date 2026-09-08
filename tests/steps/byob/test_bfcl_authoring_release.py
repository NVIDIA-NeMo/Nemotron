from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nemotron.steps.byob.runtime.authoring_release.freeze import (
    AuthoringFreezeError,
    FreezeInputsV2,
    FrozenReleaseV2,
    freeze_canonical_pack,
    load_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.handoff import (
    AuthoringHandoffError,
    handoff_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.publication import (
    BfclPublicationAdapter,
)
from nemotron.steps.byob.runtime.authoring_release.review import (
    REQUIRED_CHECKLIST_V2,
    AuthoringReviewError,
    ReviewApprovalV2,
    ReviewPacketV2,
    build_review_approval,
    build_review_packet,
    load_review_approval,
    load_review_packet,
    write_review_approval,
    write_review_packet,
)
from nemotron.steps.byob.runtime.authoring_release.versions import (
    FREEZE_MANIFEST_VERSION_V3,
)
from nemotron.steps.byob.runtime.mcp.release.adapter import McpReleaseAdapter
from nemotron.steps.byob.runtime.mcp.release.freeze import (
    FreezeInputs as McpFreezeInputsV1,
)
from nemotron.steps.byob.runtime.mcp.release.freeze import (
    freeze_canonical_pack as freeze_mcp_v1,
)
from nemotron.steps.byob.runtime.mcp.release.freeze import (
    load_frozen_release as load_mcp_v1,
)
from nemotron.steps.byob.runtime.mcp.release.review import (
    ReviewApproval as McpReviewApprovalV1,
)
from nemotron.steps.byob.runtime.mcp.release.review import (
    ReviewPacket as McpReviewPacketV1,
)
from nemotron.steps.byob.runtime.mcp.release.review import (
    build_review_packet as build_mcp_review_v1,
)
from nemotron.steps.byob.runtime.mcp.release.review import (
    load_review_packet as load_mcp_review_v1,
)
from nemotron.steps.byob.runtime.release_seal import ReleaseSealAuthority
from tests.steps.byob.test_bfcl_mcp_release_review import (
    SHA_A,
    _approved_freeze_inputs,
    _inputs,
)

SEAL_AUTHORITY = ReleaseSealAuthority(
    issuer="bfcl-release-authority",
    key_id="release-key-1",
    private_key=Ed25519PrivateKey.generate(),
)
SEAL_VERIFICATION = {
    "trusted_seal_keys": {SEAL_AUTHORITY.key_id: SEAL_AUTHORITY.public_key},
    "expected_seal_issuer": SEAL_AUTHORITY.issuer,
}


def _adapter() -> McpReleaseAdapter:
    return McpReleaseAdapter(
        identity_digest=SHA_A,
        review_data=_review_data(),
    )


def _review_data() -> dict:
    return {
        "certification": {"report_digest": SHA_A},
        "authoring": {
            "model_exposure_authorization_digest": SHA_A,
            "questions_status": "not_required",
        },
        "freeze_sidecars": {},
    }


def _file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _source_records(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "certification_report": paths["evidence"],
        "evidence": paths["evidence"],
        "model_exposure_authorization": paths["evidence"],
    }


def _approved_v2(
    tmp_path: Path,
) -> tuple[dict[str, Path], ReviewPacketV2, ReviewApprovalV2, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _inputs(tmp_path)
    packet = build_review_packet(
        adapter=_adapter(),
        pack_root=paths["pack"],
        source_digests={
            "certification_report": _file_digest(paths["evidence"]),
            "evidence": _file_digest(paths["evidence"]),
            "model_exposure_authorization": _file_digest(paths["evidence"]),
        },
    )
    packet_path = write_review_packet(packet, tmp_path / "review_packet.v2.json")
    approval = build_review_approval(
        packet,
        approved_by="domain-reviewer",
        reviewed_at="2026-08-26T17:00:00+07:00",
        checklist={name: True for name in REQUIRED_CHECKLIST_V2},
    )
    approval_path = write_review_approval(
        approval,
        tmp_path / "review_approval.v2.json",
    )
    return paths, packet, approval, packet_path, approval_path


def test_mcp_v1_public_api_remains_import_compatible(tmp_path: Path) -> None:
    assert issubclass(McpFreezeInputsV1, object)
    assert inspect.signature(build_mcp_review_v1).parameters["evidence_path"]
    legacy_root = tmp_path / "v1"
    legacy_root.mkdir()
    inputs = _approved_freeze_inputs(legacy_root)
    release = freeze_mcp_v1(inputs, tmp_path / "release-v1")

    compatibility_loaded = load_frozen_release(release.root)
    assert type(compatibility_loaded) is type(load_mcp_v1(release.root))
    assert compatibility_loaded.pack_fingerprint == release.pack_fingerprint
    assert isinstance(
        load_review_packet(release.root / "pack/provenance/review_packet.json"),
        McpReviewPacketV1,
    )
    assert isinstance(
        load_review_approval(release.root / "pack/provenance/review_approval.json"),
        McpReviewApprovalV1,
    )


def test_v2_release_is_deterministic_and_semantically_equivalent_to_v1(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    v1_inputs = _approved_freeze_inputs(legacy_root)
    v1 = freeze_mcp_v1(v1_inputs, tmp_path / "legacy-release")

    paths, packet, approval, packet_path, approval_path = _approved_v2(tmp_path / "current")
    inputs = FreezeInputsV2(
        pack_root=paths["pack"],
        review_packet_path=packet_path,
        review_approval_path=approval_path,
        source_records=_source_records(paths),
        seal_authority=SEAL_AUTHORITY,
    )
    first = freeze_canonical_pack(
        inputs,
        tmp_path / "release-v2-a",
        adapter=_adapter(),
    )
    second = freeze_canonical_pack(
        inputs,
        tmp_path / "release-v2-b",
        adapter=_adapter(),
    )

    assert first.manifest == second.manifest
    assert first.manifest["schema_version"] == FREEZE_MANIFEST_VERSION_V3
    assert first.manifest["manifest_digest"] == second.manifest["manifest_digest"]
    assert first.manifest["seal_issuer"] == SEAL_AUTHORITY.issuer
    assert first.manifest["signing_key_id"] == SEAL_AUTHORITY.key_id
    assert isinstance(first.manifest["signature"], str)
    with pytest.raises(AuthoringFreezeError) as untrusted:
        load_frozen_release(first.root, adapter=_adapter())
    assert untrusted.value.code == "release_seal_trust_required"
    legacy_packet = load_mcp_review_v1(v1_inputs.review_packet_path)
    assert (
        packet.document["candidate_pack"]["fingerprint"] == legacy_packet.document["source_digests"]["canonical_pack"]
    )
    assert v1.manifest["review_packet_digest"] == legacy_packet.digest
    assert first.manifest["review_packet_digest"] == packet.digest
    assert first.manifest["review_approval_digest"] == approval.digest
    loaded = load_frozen_release(
        first.root,
        adapter=_adapter(),
        **SEAL_VERIFICATION,
    )
    assert isinstance(loaded, FrozenReleaseV2)
    assert loaded.manifest == first.manifest


def test_adapter_hook_facts_are_digest_bound_and_canonical(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    adapter = McpReleaseAdapter(
        identity_digest=SHA_A,
        blockers=({"code": "missing_probe"},),
        risks=({"code": "metadata_warning"},),
        review_data={**_review_data(), "transport": "stdio"},
    )
    first = build_review_packet(
        adapter=adapter,
        pack_root=paths["pack"],
        source_digests={
            "certification_report": SHA_A,
            "model_exposure_authorization": SHA_A,
            "z": SHA_A,
        },
    )
    second = build_review_packet(
        adapter=adapter,
        pack_root=paths["pack"],
        source_digests={
            "model_exposure_authorization": SHA_A,
            "z": SHA_A,
            "certification_report": SHA_A,
        },
    )

    assert first.document == second.document
    assert list(first.document["source_digests"]) == [
        "certification_report",
        "model_exposure_authorization",
        "z",
    ]
    assert first.document["blockers"][0]["blocker_id"].startswith("blocker:")
    assert first.document["risks"][0]["risk_id"].startswith("risk:")
    with pytest.raises(AuthoringReviewError) as raised:
        build_review_approval(
            first,
            approved_by="reviewer",
            reviewed_at="2026-08-26T17:00:00Z",
            checklist={name: True for name in REQUIRED_CHECKLIST_V2},
            acknowledged_risks=[risk["risk_id"] for risk in first.document["risks"]],
        )
    assert raised.value.code == "review_packet_blocked"


def test_freeze_requires_every_reviewed_adapter_sidecar(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    data = b'{"lineage":"reviewed"}\n'
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    reviewing = McpReleaseAdapter(
        identity_digest=SHA_A,
        review_data={
            **_review_data(),
            "freeze_sidecars": {"provenance/adapter.json": digest},
        },
        sidecars={"provenance/adapter.json": data},
    )
    packet = build_review_packet(
        adapter=reviewing,
        pack_root=paths["pack"],
        source_digests={
            "certification_report": _file_digest(paths["evidence"]),
            "model_exposure_authorization": _file_digest(paths["evidence"]),
        },
    )
    approval = build_review_approval(
        packet,
        approved_by="reviewer",
        reviewed_at="2026-08-29T15:00:00+07:00",
        checklist={name: True for name in REQUIRED_CHECKLIST_V2},
    )
    packet_path = write_review_packet(packet, tmp_path / "packet.json")
    approval_path = write_review_approval(approval, tmp_path / "approval.json")
    omitting = McpReleaseAdapter(
        identity_digest=SHA_A,
        review_data=reviewing.review_data,
        sidecars={},
    )
    with pytest.raises(AuthoringFreezeError) as raised:
        freeze_canonical_pack(
            FreezeInputsV2(
                pack_root=paths["pack"],
                review_packet_path=packet_path,
                review_approval_path=approval_path,
                source_records={
                    "certification_report": paths["evidence"],
                    "model_exposure_authorization": paths["evidence"],
                },
                seal_authority=SEAL_AUTHORITY,
            ),
            tmp_path / "release",
            adapter=omitting,
        )
    assert raised.value.code == "adapter_sidecar_set_mismatch"


def test_unknown_and_tampered_v2_manifests_fail_closed(tmp_path: Path) -> None:
    paths, _, _, packet_path, approval_path = _approved_v2(tmp_path / "inputs")
    release = freeze_canonical_pack(
        FreezeInputsV2(
            pack_root=paths["pack"],
            review_packet_path=packet_path,
            review_approval_path=approval_path,
            source_records=_source_records(paths),
            seal_authority=SEAL_AUTHORITY,
        ),
        tmp_path / "release",
        adapter=_adapter(),
    )
    manifest_path = release.root / "freeze_manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter_kind"] = "local_python"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AuthoringFreezeError) as raised:
        load_frozen_release(
            release.root,
            adapter=_adapter(),
            **SEAL_VERIFICATION,
        )
    assert raised.value.code == "freeze_manifest_tampered"


def test_v2_loader_seals_every_pack_file_without_adapter(tmp_path: Path) -> None:
    paths, _, _, packet_path, approval_path = _approved_v2(tmp_path / "inputs")
    release = freeze_canonical_pack(
        FreezeInputsV2(
            pack_root=paths["pack"],
            review_packet_path=packet_path,
            review_approval_path=approval_path,
            source_records=_source_records(paths),
            seal_authority=SEAL_AUTHORITY,
        ),
        tmp_path / "release",
        adapter=_adapter(),
    )
    tools = release.pack_root / "tools.json"
    tools.chmod(0o644)
    tools.write_text("[]\n", encoding="utf-8")

    with pytest.raises(AuthoringFreezeError) as raised:
        load_frozen_release(release.root, **SEAL_VERIFICATION)
    assert raised.value.code == "frozen_release_tampered"


def test_v2_approval_cannot_float_to_changed_packet(tmp_path: Path) -> None:
    paths, _, _, packet_path, approval_path = _approved_v2(tmp_path / "inputs")
    packet_document = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_document["adapter_review"]["changed"] = True
    packet_path.write_text(json.dumps(packet_document), encoding="utf-8")

    with pytest.raises(AuthoringReviewError) as raised:
        freeze_canonical_pack(
            FreezeInputsV2(
                pack_root=paths["pack"],
                review_packet_path=packet_path,
                review_approval_path=approval_path,
                source_records={},
                seal_authority=SEAL_AUTHORITY,
            ),
            tmp_path / "release",
            adapter=_adapter(),
        )
    assert raised.value.code == "release_digest_mismatch"


@pytest.mark.parametrize("adapter_kind", ["local_python", "mcp_mode_a"])
def test_publication_refuses_fresh_validation_for_a_different_pack(
    adapter_kind: str,
) -> None:
    frozen = "sha256:" + "1" * 64
    gold = {
        "checks": [{"status": "pass", "failures": []}],
        "stats": {
            "has_oracle": True,
            "n_templates": 1,
            "n_assertions": 1,
            "n_tools": 1,
        },
        "pack_fingerprint": "2" * 64,
    }
    adapter = (
        McpReleaseAdapter(identity_digest=SHA_A)
        if adapter_kind == "mcp_mode_a"
        else BfclPublicationAdapter("local_python")
    )

    with pytest.raises(AuthoringHandoffError) as stale:
        adapter.require_fresh_gold(gold, frozen)
    assert stale.value.code == "fresh_validation_stale"

    # A report that covers the frozen pack passes the staleness boundary, so the
    # refusal above is specific to the fingerprint rather than the report shape.
    matching = {**gold, "pack_fingerprint": frozen.removeprefix("sha256:")}
    if adapter_kind == "local_python":
        adapter.require_fresh_gold(matching, frozen)
    else:
        with pytest.raises(AuthoringHandoffError) as conformance:
            adapter.require_fresh_gold(matching, frozen)
        assert conformance.value.code != "fresh_validation_stale"

    failed = {**matching, "checks": [{"status": "fail", "failures": ["replay"]}]}
    with pytest.raises(AuthoringHandoffError) as not_gold:
        adapter.require_fresh_gold(failed, frozen)
    assert not_gold.value.code == "fresh_gold_required"


def test_v2_freeze_requires_a2_certification(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    adapter = McpReleaseAdapter(
        identity_digest=SHA_A,
        certification_tier="A1",
        review_data=_review_data(),
    )
    packet = build_review_packet(
        adapter=adapter,
        pack_root=paths["pack"],
        source_digests={name: _file_digest(path) for name, path in _source_records(paths).items()},
    )
    packet_path = write_review_packet(packet, tmp_path / "packet.json")
    approval = build_review_approval(
        packet,
        approved_by="reviewer",
        reviewed_at="2026-08-26T17:00:00Z",
        checklist={name: True for name in REQUIRED_CHECKLIST_V2},
    )
    approval_path = write_review_approval(approval, tmp_path / "approval.json")

    with pytest.raises(AuthoringFreezeError) as raised:
        freeze_canonical_pack(
            FreezeInputsV2(
                pack_root=paths["pack"],
                review_packet_path=packet_path,
                review_approval_path=approval_path,
                source_records=_source_records(paths),
                seal_authority=SEAL_AUTHORITY,
            ),
            tmp_path / "release",
            adapter=adapter,
        )
    assert raised.value.code == "adapter_under_certified"


def test_v2_handoff_uses_typed_publication_hooks(tmp_path: Path) -> None:
    paths, _, _, packet_path, approval_path = _approved_v2(tmp_path / "inputs")
    release = freeze_canonical_pack(
        FreezeInputsV2(
            pack_root=paths["pack"],
            review_packet_path=packet_path,
            review_approval_path=approval_path,
            source_records=_source_records(paths),
            seal_authority=SEAL_AUTHORITY,
        ),
        tmp_path / "release",
        adapter=_adapter(),
    )
    output = tmp_path / "publication"

    class RecordedPublicationAdapter:
        kind = "mcp_mode_a"

        def validate_pack(self, pack_root: Path) -> str:
            return _adapter().validate_pack(pack_root)

        def bind_config(
            self,
            config_path: Path,
            pack_root: Path,
            frozen_pack_fingerprint: str,
        ) -> None:
            assert config_path.name == "config.yaml"
            assert pack_root == release.pack_root
            assert frozen_pack_fingerprint == release.pack_fingerprint

        def prepare(self, config_path: Path) -> Path:
            del config_path
            return write_json(output / "validation.json", {"gold": True})

        def load_validation_report(self, report_path: Path) -> dict:
            return json.loads(report_path.read_text(encoding="utf-8"))

        def require_fresh_gold(
            self,
            report: dict,
            frozen_pack_fingerprint: str,
        ) -> None:
            assert report["gold"] is True
            assert frozen_pack_fingerprint == release.pack_fingerprint

        def generate(self, config_path: Path) -> Path:
            del config_path
            output.mkdir(parents=True, exist_ok=True)
            benchmark = output / "benchmark.parquet"
            benchmark.write_bytes(b"benchmark")
            (output / "benchmark_raw.parquet").write_bytes(b"raw")
            write_json(
                output / "run_manifest.json",
                {
                    "pack": {"content_hash": release.pack_fingerprint},
                    "tier": "gold",
                    "gold_eligible": True,
                    "gold_ineligibility_reasons": [],
                    "oracle": {
                        "mcp": {
                            "frozen_pack_fingerprint": release.pack_fingerprint,
                        }
                    },
                },
            )
            return benchmark

        def verify_publication_manifest(
            self,
            manifest: dict,
            frozen_pack_fingerprint: str,
        ) -> None:
            assert manifest["oracle"]["mcp"]["frozen_pack_fingerprint"] == frozen_pack_fingerprint

    config = tmp_path / "config.yaml"
    config.write_text("stage: all\n", encoding="utf-8")
    handoff = handoff_frozen_release(
        release.root,
        config,
        adapter=RecordedPublicationAdapter(),
        **SEAL_VERIFICATION,
    )

    assert handoff.benchmark_path.is_file()
    assert handoff.raw_benchmark_path.is_file()
    assert handoff.run_manifest_path.is_file()


def write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
