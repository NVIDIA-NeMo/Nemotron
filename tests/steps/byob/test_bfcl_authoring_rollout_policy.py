from __future__ import annotations

from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.authoring_workflow.resolved_config import (
    ResolvedAuthoringConfig,
    resolve_authoring_config,
)
from nemotron.steps.byob.runtime.authoring_workflow.rollout import (
    ADAPTER_ROLLOUT_ENV,
    LEGACY_MCP_ROLLOUT_ENV,
    RolloutPolicyError,
    require_adapter_rollout,
    require_no_rollout_revocation,
    resolve_adapter_rollout,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json


@pytest.mark.parametrize("adapter_kind", sorted(ADAPTER_ROLLOUT_ENV))
def test_omitted_and_explicit_false_fail_closed(adapter_kind: str) -> None:
    omitted = resolve_adapter_rollout(adapter_kind, environ={})
    explicit = resolve_adapter_rollout(
        adapter_kind,
        environ={ADAPTER_ROLLOUT_ENV[adapter_kind]: " false "},
    )

    assert omitted.enabled is False
    assert omitted.origin == "default"
    assert explicit.enabled is False
    with pytest.raises(RolloutPolicyError) as raised:
        require_adapter_rollout(adapter_kind, environ={})
    assert raised.value.code == "adapter_rollout_disabled"


@pytest.mark.parametrize("adapter_kind", sorted(ADAPTER_ROLLOUT_ENV))
def test_malformed_per_kind_values_fail_closed(adapter_kind: str) -> None:
    with pytest.raises(RolloutPolicyError) as raised:
        resolve_adapter_rollout(
            adapter_kind,
            environ={ADAPTER_ROLLOUT_ENV[adapter_kind]: "enabled-ish"},
        )
    assert raised.value.code == "rollout_value_malformed"


def test_unknown_adapter_and_policy_kind_fail_closed() -> None:
    with pytest.raises(RolloutPolicyError) as adapter_error:
        resolve_adapter_rollout("plugin.transport", environ={})
    assert adapter_error.value.code == "rollout_adapter_unknown"

    with pytest.raises(RolloutPolicyError) as policy_error:
        resolve_adapter_rollout(
            "local_python",
            environ={},
            policy={"plugin.transport": True},
        )
    assert policy_error.value.code == "rollout_adapter_unknown"


def test_mcp_legacy_alias_is_supported_but_conflicts_fail_closed() -> None:
    legacy = resolve_adapter_rollout(
        "mcp_mode_a",
        environ={LEGACY_MCP_ROLLOUT_ENV: "yes"},
    )
    assert legacy.enabled is True
    assert legacy.legacy_alias_used is True
    assert legacy.environment_variable == LEGACY_MCP_ROLLOUT_ENV

    agreed = resolve_adapter_rollout(
        "mcp_mode_a",
        environ={
            LEGACY_MCP_ROLLOUT_ENV: "1",
            ADAPTER_ROLLOUT_ENV["mcp_mode_a"]: "true",
        },
    )
    assert agreed.enabled is True
    assert agreed.legacy_alias_used is True
    assert agreed.environment_variable == ADAPTER_ROLLOUT_ENV["mcp_mode_a"]

    with pytest.raises(RolloutPolicyError) as raised:
        resolve_adapter_rollout(
            "mcp_mode_a",
            environ={
                LEGACY_MCP_ROLLOUT_ENV: "1",
                ADAPTER_ROLLOUT_ENV["mcp_mode_a"]: "0",
            },
        )
    assert raised.value.code == "rollout_settings_conflict"


def test_environment_overrides_reviewed_policy() -> None:
    decision = resolve_adapter_rollout(
        "http_package",
        environ={ADAPTER_ROLLOUT_ENV["http_package"]: "0"},
        policy={"http_package": True},
    )
    assert decision.enabled is False
    assert decision.origin == "environment"


def test_current_environment_can_revoke_a_resolved_enablement() -> None:
    require_no_rollout_revocation("local_python", environ={})
    require_no_rollout_revocation(
        "local_python",
        environ={ADAPTER_ROLLOUT_ENV["local_python"]: "1"},
    )
    with pytest.raises(RolloutPolicyError) as raised:
        require_no_rollout_revocation(
            "local_python",
            environ={ADAPTER_ROLLOUT_ENV["local_python"]: "0"},
        )
    assert raised.value.code == "adapter_rollout_disabled"


def test_rollout_decision_is_digest_bound_in_resolved_config(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate reviewed tools.", encoding="utf-8")
    common = {
        "adapter_kind": "local_python",
        "source": source,
        "domain_brief": brief,
        "workspace": tmp_path / "workspace",
        "tenant_id": "tenant",
        "run_id": "run",
        "pack_id": "tools",
        "pack_version": "1.0.0",
        "ci": True,
    }

    disabled = resolve_authoring_config(**common, environ={})
    enabled = resolve_authoring_config(
        **common,
        environ={ADAPTER_ROLLOUT_ENV["local_python"]: "1"},
    )

    assert disabled.semantic_payload.rollout_policy.live_authoring_enabled.value is False
    assert enabled.semantic_payload.rollout_policy.live_authoring_enabled.value is True
    assert enabled.semantic_payload.rollout_policy.live_authoring_enabled.origin == "user"
    assert (
        disabled.resolved_authoring_config_digest
        != enabled.resolved_authoring_config_digest
    )

    legacy = enabled.model_dump(mode="json")
    legacy["schema_version"] = "bfcl-resolved-authoring-config-v1"
    legacy["semantic_payload"].pop("rollout_policy")
    legacy["resolved_authoring_config_digest"] = sha256_json(
        {
            key: value
            for key, value in legacy.items()
            if key != "resolved_authoring_config_digest"
        }
    )
    loaded_legacy = ResolvedAuthoringConfig.model_validate(legacy)
    assert loaded_legacy.semantic_payload.rollout_policy is None
