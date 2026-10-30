from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.authoring_workflow.credentials import (
    CredentialLifecycleError,
    CredentialReference,
    CredentialResolver,
    build_authorization_context,
    require_current_authorization_context,
)
from nemotron.steps.byob.runtime.authoring_workflow.events import (
    AdapterIdentityPayload,
    build_authoring_event,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    PROTOCOL_VERSION,
    load_endpoint_config,
    resolve_endpoint_headers,
)
from nemotron.steps.byob.runtime.mcp.client import (
    resolve_http_headers as resolve_mcp_http_headers,
)
from nemotron.steps.byob.runtime.mcp.config import (
    HttpAuthConfig,
    StreamableHttpTransportConfig,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


class _SecretManager:
    def __init__(self, value: str, *, fail: bool = False) -> None:
        self.value = value
        self.fail = fail
        self.requests: list[tuple[str, str | None]] = []

    def resolve(self, name: str, *, version: str | None) -> str:
        self.requests.append((name, version))
        if self.fail:
            raise RuntimeError(f"backend accidentally echoed {self.value}")
        return self.value


def _manager_ref() -> CredentialReference:
    return CredentialReference(
        resolver="secret_manager",
        provider="test-vault",
        name="bfcl/oracle/token",
        version="current",
    )


def test_environment_and_secret_manager_values_are_memory_only() -> None:
    secret = "credential-super-secret"
    environment_ref = CredentialReference.environment("BFCL_ORACLE_TOKEN")
    manager_ref = _manager_ref()
    manager = _SecretManager(secret)
    resolver = CredentialResolver(
        environ={"BFCL_ORACLE_TOKEN": secret},
        secret_managers={"test-vault": manager},
    )

    environment_value = resolver.resolve(environment_ref)
    manager_value = resolver.resolve(manager_ref)

    assert environment_value.reveal() == secret
    assert manager_value.reveal() == secret
    assert manager.requests == [("bfcl/oracle/token", "current")]
    persisted = json.dumps(
        {
            "environment": environment_ref.model_dump(mode="json"),
            "manager": manager_ref.model_dump(mode="json"),
            "environment_repr": repr(environment_value),
            "manager_repr": repr(manager_value),
        }
    )
    assert secret not in persisted


def test_secret_manager_failure_does_not_echo_backend_exception() -> None:
    secret = "backend-leaked-secret"
    resolver = CredentialResolver(
        secret_managers={"test-vault": _SecretManager(secret, fail=True)}
    )
    with pytest.raises(CredentialLifecycleError) as failure:
        resolver.resolve(_manager_ref())
    assert failure.value.code == "credential_unresolved"
    assert secret not in str(failure.value)
    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__ is True


def test_invalid_reference_error_hides_supplied_value() -> None:
    secret = "looks-like-a-secret-token"
    with pytest.raises(ValueError) as failure:
        CredentialReference.model_validate(
            {"resolver": "environment", "name": secret}
        )
    assert secret not in str(failure.value)


def test_rotation_is_stable_but_principal_permission_or_reference_drift_is_stale() -> None:
    reference = CredentialReference.environment("BFCL_ORACLE_TOKEN")
    original = build_authorization_context(
        (reference,),
        principal_digest=SHA_A,
        permission_digest=SHA_B,
    )
    rotated_secret_same_context = build_authorization_context(
        (reference,),
        principal_digest=SHA_A,
        permission_digest=SHA_B,
    )
    require_current_authorization_context(
        original.authorization_context_digest,
        rotated_secret_same_context,
    )

    changed_contexts = (
        build_authorization_context(
            (reference,),
            principal_digest=SHA_C,
            permission_digest=SHA_B,
        ),
        build_authorization_context(
            (reference,),
            principal_digest=SHA_A,
            permission_digest=SHA_C,
        ),
        build_authorization_context(
            (CredentialReference.environment("BFCL_ROTATED_TOKEN_REF"),),
            principal_digest=SHA_A,
            permission_digest=SHA_B,
        ),
    )
    for changed in changed_contexts:
        with pytest.raises(CredentialLifecycleError) as stale:
            require_current_authorization_context(
                original.authorization_context_digest,
                changed,
            )
        assert stale.value.code == "credential_context_stale"


def test_endpoint_resolves_secret_manager_reference_without_persisting_value(
    tmp_path: Path,
) -> None:
    secret = "vault-only-token"
    reference = _manager_ref()
    context = build_authorization_context(
        (reference,),
        principal_digest=SHA_A,
        permission_digest=SHA_B,
    )
    document: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "base_url": "https://oracle.example",
        "auth": {"bearer_token_ref": reference.model_dump(mode="json")},
        "expected": {
            "oracle_id": "oracle",
            "oracle_version": "1",
            "content_digest": SHA_C,
            "principal_digest": SHA_A,
            "permission_digest": SHA_B,
            "authorization_context_digest": context.authorization_context_digest,
        },
    }
    path = tmp_path / "endpoint_config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    config = load_endpoint_config(path, allowed_roots=(tmp_path,))
    resolver = CredentialResolver(
        secret_managers={"test-vault": _SecretManager(secret)}
    )

    headers = resolve_endpoint_headers(config, credential_resolver=resolver)

    assert headers == {"Authorization": f"Bearer {secret}"}
    assert secret not in path.read_text(encoding="utf-8")
    assert secret not in repr(config)


def test_authorization_context_event_is_digest_only() -> None:
    secret = "must-not-enter-events"
    context = build_authorization_context(
        (CredentialReference.environment("BFCL_ORACLE_TOKEN"),),
        principal_digest=SHA_A,
        permission_digest=SHA_B,
    )
    event = build_authoring_event(
        "adapter_identity_bound",
        AdapterIdentityPayload(
            adapter_kind="http_package",
            source_identity_digest=SHA_A,
            evidence_bundle_digest=SHA_B,
            authorization_context_digest=context.authorization_context_digest,
        ),
        tenant_id="tenant-a",
        run_id="run-a",
        session_digest=SHA_C,
    )
    assert secret not in json.dumps(event.model_dump(mode="json"))
    assert (
        event.payload["authorization_context_digest"]
        == context.authorization_context_digest
    )


def test_mcp_http_transport_uses_the_same_secret_manager_resolver() -> None:
    secret = "mcp-vault-token"
    reference = _manager_ref()
    transport = StreamableHttpTransportConfig(
        kind="streamable_http",
        url="https://mcp.example/mcp",
        auth=HttpAuthConfig(bearer_token_ref=reference),
    )
    resolver = CredentialResolver(
        secret_managers={"test-vault": _SecretManager(secret)}
    )

    headers, registered_secrets = resolve_mcp_http_headers(
        transport,
        {},
        credential_resolver=resolver,
    )

    assert headers == {"Authorization": f"Bearer {secret}"}
    assert registered_secrets == (secret,)
    assert secret not in json.dumps(transport.model_dump(mode="json"))
