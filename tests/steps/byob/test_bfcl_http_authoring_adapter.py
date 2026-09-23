from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.authoring_workflow.credentials import (
    CredentialReference,
    build_authorization_context,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    HTTP_PROFILE_VERSION,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    PROTOCOL_VERSION,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterTier,
    certification_input_digest,
    derive_attained_tier,
    http_package_reference_profile,
    project_probe_executions,
)
from nemotron.steps.byob.runtime.source_adapters.http_package import (
    HttpPackageError,
    inspect_http_package,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up one reviewed item.",
                "parameters": {
                    "type": "object",
                    "properties": {"item_id": {"type": "string"}},
                    "required": ["item_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _attestation(catalog_digest: str) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_KIND,
        "provider_kind": "http",
        "profile_version": HTTP_PROFILE_VERSION,
        "level": "L0",
        "gateway_evidence_kind": "locally_verified",
        "gateway_evidence_issuer": "bfcl-http-verifier",
        "state_observability": "diagnostic",
        "read_only_boundary": None,
        "effective_content_digest": DIGEST_A,
        "gateway_artifact_digest": DIGEST_B,
        "tool_catalog_digest": catalog_digest,
        "probe_report_digest": DIGEST_B,
        "gateway_conformance_report_digest": DIGEST_C,
        "shim_artifact_digest": None,
        "server_content_digest": DIGEST_A,
        "snapshot_digest": None,
        "checks": [
            {
                "id": "H1",
                "requirement": "required",
                "status": "pass",
                "reason": None,
            }
        ],
    }


def _package(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    tools = _tools()
    (tmp_path / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    attestation = _attestation(sha256_json(tools))
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "base_url": "https://oracle.example",
        "expected": {
            "oracle_id": "reviewed-oracle",
            "oracle_version": "1.0.0",
            "content_digest": DIGEST_A,
        },
        "attestation": {
            "kind": ATTESTATION_KIND,
            "expected_digest": attestation_digest(attestation),
        },
        "max_request_bytes": 4096,
        "max_response_bytes": 4096,
    }
    (tmp_path / "endpoint_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    return tmp_path, attestation


class _Client:
    def __init__(
        self,
        attestation: dict[str, Any],
        *,
        names: list[str] | None = None,
        metadata_digest: str = DIGEST_A,
        principal_digest: str | None = None,
        permission_digest: str | None = None,
        authorization_context_digest: str | None = None,
    ) -> None:
        self.attestation = attestation
        self.names = ["lookup"] if names is None else names
        self.metadata_digest = metadata_digest
        self.principal_digest = principal_digest
        self.permission_digest = permission_digest
        self.authorization_context_digest = authorization_context_digest
        self.closed = False

    def metadata(self) -> dict[str, str]:
        if self.metadata_digest != DIGEST_A:
            raise RuntimeError(
                "endpoint metadata does not match the expected oracle identity"
            )
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "oracle_id": "reviewed-oracle",
            "oracle_version": "1.0.0",
            "content_digest": self.metadata_digest,
        }
        if self.principal_digest is not None:
            result["principal_digest"] = self.principal_digest
        if self.permission_digest is not None:
            result["permission_digest"] = self.permission_digest
        if self.authorization_context_digest is not None:
            result["authorization_context_digest"] = self.authorization_context_digest
        return result

    def list_tools(self) -> list[str]:
        self.metadata()
        return self.names

    def conformance(self) -> dict[str, Any]:
        return self.attestation

    def close(self, *, suppress_errors: bool = False) -> None:
        self.closed = True


def _factory(client: _Client) -> Any:
    def build(_config: Any, _headers: Any, _timeout_s: float) -> _Client:
        return client

    return build


def test_http_package_reaches_a0_from_reviewed_schema_and_live_identity(
    tmp_path: Path,
) -> None:
    package, attestation = _package(tmp_path)
    client = _Client(attestation)

    inspection = inspect_http_package(
        package,
        allowed_roots=(tmp_path,),
        environ={},
        client_factory=_factory(client),
    )
    profile = http_package_reference_profile()
    input_digest = certification_input_digest(
        inspection.descriptor,
        source_identity_digest=inspection.source_identity_digest,
        profile=profile,
    )
    outcomes = project_probe_executions(
        profile,
        inspection.execution_records,
        input_digest=input_digest,
    )

    assert client.closed is True
    assert inspection.descriptor.kind == "http_package"
    assert inspection.identity.effective_content_digest == DIGEST_A
    assert [tool.published_name for tool in inspection.tools] == ["lookup"]
    assert derive_attained_tier(profile, outcomes) is AdapterTier.A0


def test_http_package_requires_a_reviewed_companion_schema(tmp_path: Path) -> None:
    package, attestation = _package(tmp_path)
    (package / "tools.json").unlink()

    with pytest.raises(HttpPackageError) as error:
        inspect_http_package(
            package,
            allowed_roots=(tmp_path,),
            environ={},
            client_factory=_factory(_Client(attestation)),
        )
    assert error.value.code == "reviewed_schema_missing"


def test_http_package_refuses_live_catalog_or_identity_drift(tmp_path: Path) -> None:
    package, attestation = _package(tmp_path)

    with pytest.raises(HttpPackageError) as catalog:
        inspect_http_package(
            package,
            allowed_roots=(tmp_path,),
            environ={},
            client_factory=_factory(_Client(attestation, names=["other"])),
        )
    assert catalog.value.code == "catalog_mismatch"

    with pytest.raises(HttpPackageError) as identity:
        inspect_http_package(
            package,
            allowed_roots=(tmp_path,),
            environ={},
            client_factory=_factory(
                _Client(attestation, metadata_digest=DIGEST_B)
            ),
        )
    assert identity.value.code == "identity_drift"


def test_http_package_refuses_attestation_or_schema_drift(tmp_path: Path) -> None:
    package, attestation = _package(tmp_path)
    drifted = {**attestation, "effective_content_digest": DIGEST_B}

    with pytest.raises(HttpPackageError) as mismatch:
        inspect_http_package(
            package,
            allowed_roots=(tmp_path,),
            environ={},
            client_factory=_factory(_Client(drifted)),
        )
    assert mismatch.value.code == "attestation_mismatch"

    package, _ = _package(tmp_path)
    (package / "tools.json").write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                    "inferred_schema": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(HttpPackageError) as invalid:
        inspect_http_package(
            package,
            allowed_roots=(tmp_path,),
            environ={},
            client_factory=_factory(_Client(attestation)),
        )
    assert invalid.value.code == "reviewed_schema_invalid"


def test_http_identity_binds_credential_reference_without_persisting_secret(
    tmp_path: Path,
) -> None:
    package, attestation = _package(tmp_path)
    config_path = package / "endpoint_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["auth"] = {"bearer_token_env": "TOKEN_A"}
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    with pytest.raises(HttpPackageError) as missing_principal:
        inspect_http_package(
            package,
            allowed_roots=(tmp_path,),
            environ={"TOKEN_A": "first-secret"},
            client_factory=_factory(_Client(attestation)),
        )
    assert missing_principal.value.code == "unsupported_auth"

    first_context = build_authorization_context(
        (CredentialReference.environment("TOKEN_A"),),
        principal_digest=DIGEST_B,
        permission_digest=DIGEST_C,
    )
    config["expected"].update(
        {
            "principal_digest": DIGEST_B,
            "permission_digest": DIGEST_C,
            "authorization_context_digest": first_context.authorization_context_digest,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    first = inspect_http_package(
        package,
        allowed_roots=(tmp_path,),
        environ={"TOKEN_A": "first-secret"},
        client_factory=_factory(
            _Client(
                attestation,
                principal_digest=DIGEST_B,
                permission_digest=DIGEST_C,
                authorization_context_digest=first_context.authorization_context_digest,
            )
        ),
    )

    config["auth"] = {"bearer_token_env": "TOKEN_B"}
    second_context = build_authorization_context(
        (CredentialReference.environment("TOKEN_B"),),
        principal_digest=DIGEST_B,
        permission_digest=DIGEST_C,
    )
    config["expected"]["authorization_context_digest"] = (
        second_context.authorization_context_digest
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    second = inspect_http_package(
        package,
        allowed_roots=(tmp_path,),
        environ={"TOKEN_B": "second-secret"},
        client_factory=_factory(
            _Client(
                attestation,
                principal_digest=DIGEST_B,
                permission_digest=DIGEST_C,
                authorization_context_digest=second_context.authorization_context_digest,
            )
        ),
    )

    assert first.identity.source_config_digest != second.identity.source_config_digest
    config["expected"]["principal_digest"] = DIGEST_C
    third_context = build_authorization_context(
        (CredentialReference.environment("TOKEN_B"),),
        principal_digest=DIGEST_C,
        permission_digest=DIGEST_C,
    )
    config["expected"]["authorization_context_digest"] = (
        third_context.authorization_context_digest
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    third = inspect_http_package(
        package,
        allowed_roots=(tmp_path,),
        environ={"TOKEN_B": "third-secret"},
        client_factory=_factory(
            _Client(
                attestation,
                principal_digest=DIGEST_C,
                permission_digest=DIGEST_C,
                authorization_context_digest=third_context.authorization_context_digest,
            )
        ),
    )
    assert second.identity.source_config_digest != third.identity.source_config_digest
    persisted = json.dumps(
        [
            first.identity.model_dump(mode="json"),
            second.identity.model_dump(mode="json"),
            third.identity.model_dump(mode="json"),
        ]
    )
    assert "first-secret" not in persisted
    assert "second-secret" not in persisted
    assert "third-secret" not in persisted


def test_http_inspection_redacts_credential_reflection_from_errors(
    tmp_path: Path,
) -> None:
    package, attestation = _package(tmp_path)
    config_path = package / "endpoint_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = build_authorization_context(
        (CredentialReference.environment("TOKEN_A"),),
        principal_digest=DIGEST_B,
        permission_digest=DIGEST_C,
    )
    config["auth"] = {"bearer_token_env": "TOKEN_A"}
    config["expected"].update(
        {
            "principal_digest": DIGEST_B,
            "permission_digest": DIGEST_C,
            "authorization_context_digest": context.authorization_context_digest,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    class ReflectingClient(_Client):
        def metadata(self) -> dict[str, str]:
            raise RuntimeError("server reflected reflected-secret")

    with pytest.raises(HttpPackageError) as failure:
        inspect_http_package(
            package,
            allowed_roots=(tmp_path,),
            environ={"TOKEN_A": "reflected-secret"},
            client_factory=_factory(ReflectingClient(attestation)),
        )
    assert "reflected-secret" not in str(failure.value)
    assert failure.value.__cause__ is None
