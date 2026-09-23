from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

from nemotron.steps.byob.runtime.mcp.client import (
    build_stdio_environment,
    resolve_http_headers,
    resolve_stdio_launch,
)
from nemotron.steps.byob.runtime.mcp.config import (
    HttpAuthConfig,
    StreamableHttpTransportConfig,
    load_mcp_oracle_config,
    load_trusted_executable_policies,
)
from nemotron.steps.byob.runtime.mcp.errors import (
    McpConfigError,
    McpCredentialError,
)
from nemotron.steps.byob.runtime.mcp.rollout import (
    MCP_FEATURE_ENV,
    mcp_feature_enabled,
    require_mcp_feature,
)

DIGEST = "sha256:" + "0" * 64


def _config() -> dict:
    return {
        "profile_version": "bfcl-mcp-oracle-v1",
        "mode": "C",
        "mcp_protocol_versions": ["2026-07-28", "2025-11-25"],
        "transport": {
            "kind": "streamable_http",
            "url": "https://mcp.example.test/mcp",
            "auth": {
                "bearer_token_env": "MCP_TOKEN",
                "headers": {"X-Tenant": "MCP_TENANT"},
            },
        },
        "expected": {
            "server_name": "catalog",
            "server_version": "1.0.0",
            "tool_catalog_digest": DIGEST,
            "oracle_id": "catalog-oracle",
            "oracle_version": "1.0.0",
        },
        "control": {
            "reset_strategy": "no_op_verified",
            "state_strategy": "read_only_projection",
            "episode_binding": "transport",
            "state_projection": [{"tool": "lookup", "arguments": {"id": "A"}}],
        },
        "fixtures": {
            "direction": "snapshot",
            "snapshot_calls": [
                {"tool": "list_items", "arguments": {}, "collection": "items"}
            ],
        },
        "tools": {
            "include": ["lookup"],
            "aliases": {},
            "mutates": [],
            "requires_confirmation": [],
            "trust_annotations": False,
        },
        "isolation": "read_only",
    }


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "mcp_oracle.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_mcp_config_loads_strict_read_only_profile(tmp_path: Path) -> None:
    loaded = load_mcp_oracle_config(_write(tmp_path, _config()))

    assert loaded.value.mode == "C"
    assert loaded.value.transport.kind == "streamable_http"
    assert loaded.value.tools.published_name("lookup") == "lookup"
    assert loaded.value.limits.max_catalog_pages == 20


def test_mcp_config_rejects_unknown_fields_and_insecure_remote_http(
    tmp_path: Path,
) -> None:
    unknown = _config()
    unknown["surprise"] = True
    with pytest.raises(McpConfigError, match="extra"):
        load_mcp_oracle_config(_write(tmp_path, unknown))

    insecure = _config()
    insecure["transport"]["url"] = "http://mcp.example.test/mcp"
    with pytest.raises(McpConfigError, match="HTTPS"):
        load_mcp_oracle_config(_write(tmp_path, insecure))


def test_mcp_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "mcp_oracle.yaml"
    path.write_text(
        yaml.safe_dump(_config()) + "\nmode: A\n",
        encoding="utf-8",
    )

    with pytest.raises(McpConfigError, match="duplicate key 'mode'"):
        load_mcp_oracle_config(path)


def test_mcp_config_wraps_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "mcp_oracle.yaml"
    path.write_text("profile_version: [unterminated\n", encoding="utf-8")

    with pytest.raises(McpConfigError, match="cannot load MCP oracle config"):
        load_mcp_oracle_config(path)


def test_http_credentials_are_environment_references_only() -> None:
    transport = StreamableHttpTransportConfig(
        kind="streamable_http",
        url="https://mcp.example.test/mcp",
        auth=HttpAuthConfig(
            bearer_token_env="MCP_TOKEN",
            headers={"X-Tenant": "MCP_TENANT"},
        ),
    )
    headers, secrets = resolve_http_headers(
        transport,
        {"MCP_TOKEN": "secret-token", "MCP_TENANT": "tenant-a"},
    )
    assert headers == {
        "Authorization": "Bearer secret-token",
        "X-Tenant": "tenant-a",
    }
    assert secrets == ("secret-token", "tenant-a")

    with pytest.raises(McpCredentialError, match="MCP_TOKEN"):
        resolve_http_headers(transport, {})


def test_http_auth_rejects_case_insensitive_duplicate_headers() -> None:
    with pytest.raises(ValueError, match="case-insensitive duplicate"):
        HttpAuthConfig(
            headers={
                "X-Tenant": "MCP_TENANT",
                "x-tenant": "MCP_TENANT_FALLBACK",
            }
        )


def test_stdio_launch_requires_exact_host_policy(tmp_path: Path) -> None:
    executable = Path(sys.executable).resolve()
    digest = "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest()
    policy_path = tmp_path / "trusted.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "bfcl-trusted-executables-v1",
                "policies": {
                    "python-fixture": {
                        "executable": str(executable),
                        "sha256": digest,
                        "allowed_argv": [["-m", "fixture_server"]],
                        "allowed_cwd_roots": [str(tmp_path)],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    policies = load_trusted_executable_policies(policy_path)
    data = _config()
    data["mode"] = "B"
    data["transport"] = {
        "kind": "stdio",
        "command": [executable.name, "-m", "fixture_server"],
        "cwd": str(tmp_path),
        "env_passthrough": ["MCP_FIXTURE"],
        "executable_policy": "python-fixture",
    }
    data["control"]["reset_strategy"] = "process_restart"
    data["isolation"] = "process_per_episode"
    loaded = load_mcp_oracle_config(_write(tmp_path, data))

    command, arguments = resolve_stdio_launch(loaded.value.transport, policies)  # type: ignore[arg-type]
    assert command == str(executable)
    assert arguments == ["-m", "fixture_server"]
    assert build_stdio_environment(
        loaded.value.transport,  # type: ignore[arg-type]
        {"MCP_FIXTURE": "fixture", "UNRELATED_SECRET": "do-not-forward"},
    ) == {"MCP_FIXTURE": "fixture"}


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes "])
def test_experimental_mcp_feature_accepts_only_explicit_true_values(
    value: str,
) -> None:
    assert mcp_feature_enabled({MCP_FEATURE_ENV: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no "])
def test_experimental_mcp_feature_accepts_explicit_false_values(
    value: str,
) -> None:
    assert mcp_feature_enabled({MCP_FEATURE_ENV: value}) is False


def test_experimental_mcp_feature_is_off_by_default_and_rejects_typos() -> None:
    assert mcp_feature_enabled({}) is False
    with pytest.raises(McpConfigError, match=MCP_FEATURE_ENV):
        require_mcp_feature({})
    with pytest.raises(McpConfigError, match="must be one of"):
        mcp_feature_enabled({MCP_FEATURE_ENV: "enabled-ish"})
