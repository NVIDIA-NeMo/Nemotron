from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.client import (
    McpServerIdentity,
    McpToolPage,
    SdkConnectedMcpClient,
    SecretRegistry,
    mcp_error_boundary,
)
from nemotron.steps.byob.runtime.mcp.config import (
    LoadedMcpOracleConfig,
    McpOracleConfig,
)
from nemotron.steps.byob.runtime.mcp.discovery import (
    catalog_identity_document,
    discover_mcp_oracle,
    write_discovery_report,
)
from nemotron.steps.byob.runtime.mcp.errors import (
    McpCatalogError,
    McpConfigError,
    McpCredentialError,
    McpIdentityMismatchError,
    McpNormalizationError,
    McpProtocolError,
    McpTransportError,
)
from nemotron.steps.byob.runtime.mcp.normalization import (
    normalize_catalog,
)

DIGEST = "sha256:" + "0" * 64
CONTENT_DIGEST = "sha256:" + "a" * 64


def _raw_config() -> dict[str, Any]:
    return {
        "profile_version": "bfcl-mcp-oracle-v1",
        "mode": "C",
        "mcp_protocol_versions": ["2026-07-28", "2025-11-25"],
        "transport": {
            "kind": "streamable_http",
            "url": "https://mcp.example.test/mcp",
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
            "episode_binding": "argument",
            "episode_argument": "episode_id",
            "state_projection": [{"tool": "lookup", "arguments": {"id": "A"}}],
        },
        "fixtures": {
            "direction": "snapshot",
            "snapshot_calls": [
                {"tool": "list_items", "arguments": {}, "collection": "items"}
            ],
        },
        "tools": {
            "include": ["inventory.lookup"],
            "aliases": {"inventory.lookup": "inventory_lookup"},
            "mutates": [],
            "requires_confirmation": [],
            "trust_annotations": False,
        },
        "isolation": "read_only",
        "limits": {
            "connect_timeout_s": 1,
            "handshake_timeout_s": 1,
            "tool_timeout_s": 1,
            "reset_timeout_s": 1,
            "episode_timeout_s": 5,
            "max_response_bytes": 100000,
            "max_tools": 8,
            "max_catalog_pages": 4,
            "max_concurrent_episodes": 1,
            "session_idle_ttl_s": 30,
        },
    }


TOOLS = [
    {
        "name": "ignored.admin",
        "description": "Not selected",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "inventory.lookup",
        "description": "Look up one item.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "episode_id": {
                    "type": "string",
                    "x-mcp-header": "X-Episode",
                },
            },
            "required": ["episode_id", "id"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
        "annotations": {"readOnlyHint": True},
    },
]


class _FakeClient:
    sdk_version = "2.1.0-test"
    protocol_version = "2026-07-28"
    server_identity = McpServerIdentity("catalog", "1.0.0")
    capabilities = {"tools": {"listChanged": False}}

    def __init__(self, *, cursor_cycle: bool = False):
        self.cursor_cycle = cursor_cycle
        self.cursors: list[str | None] = []

    async def list_tools(self, cursor: str | None = None) -> McpToolPage:
        self.cursors.append(cursor)
        if cursor is None:
            return McpToolPage(tools=(TOOLS[0],), next_cursor="page-2")
        if self.cursor_cycle:
            return McpToolPage(tools=(), next_cursor="page-2")
        return McpToolPage(tools=(TOOLS[1],), next_cursor=None)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("mode C fixture should not call describe_oracle")


def _digest_config(config: McpOracleConfig) -> McpOracleConfig:
    catalog = normalize_catalog(TOOLS, config)
    document = catalog_identity_document(
        config,
        negotiated_mcp_version="2026-07-28",
        server_name="catalog",
        server_version="1.0.0",
        catalog=catalog,
    )
    digest = "sha256:" + hashlib.sha256(canonical_json(document).encode()).hexdigest()
    return config.model_copy(
        update={
            "expected": config.expected.model_copy(
                update={"tool_catalog_digest": digest}
            )
        }
    )


def test_catalog_identity_digest_matches_independent_golden_value() -> None:
    config = McpOracleConfig.model_validate(_raw_config())
    catalog = normalize_catalog(TOOLS, config)
    document = catalog_identity_document(
        config,
        negotiated_mcp_version="2026-07-28",
        server_name="catalog",
        server_version="1.0.0",
        catalog=catalog,
    )

    assert (
        "sha256:" + hashlib.sha256(canonical_json(document).encode()).hexdigest()
        == "sha256:a5522e4a95e8ad99d38e34074f6ff8d4f300e02799859c84d9ea6e5daec79e56"
    )


def _loaded(config: McpOracleConfig | None = None) -> LoadedMcpOracleConfig:
    value = (
        _digest_config(McpOracleConfig.model_validate(_raw_config()))
        if config is None
        else config
    )
    raw = value.model_dump(mode="json", exclude_none=False)
    return LoadedMcpOracleConfig(
        path=Path("/tmp/mcp_oracle.yaml"),
        value=value,
        raw_document=raw,
    )


def test_normalization_filters_aliases_and_removes_gateway_argument() -> None:
    catalog = normalize_catalog(TOOLS, _loaded().value)

    assert catalog.source_to_published == {"inventory.lookup": "inventory_lookup"}
    function = catalog.bfcl_tools[0]["function"]
    assert set(function["parameters"]["properties"]) == {"id"}
    assert function["parameters"]["required"] == ["id"]
    assert catalog.exclusions[0].source_name == "ignored.admin"
    assert catalog.warnings[0].code == "ignored_mcp_header"
    assert catalog.tools[0].annotations == {"readOnlyHint": True}


def test_selected_tool_with_unsupported_schema_fails_closed() -> None:
    broken_schema = {
        **TOOLS[1]["inputSchema"],
        "oneOf": [{"type": "object"}],
    }
    broken = [TOOLS[0], {**TOOLS[1], "inputSchema": broken_schema}]
    with pytest.raises(McpNormalizationError, match="outside BFCL"):
        normalize_catalog(broken, McpOracleConfig.model_validate(_raw_config()))


def test_selected_tool_must_explicitly_declare_object_input_schema() -> None:
    missing_type = {
        **TOOLS[1]["inputSchema"],
    }
    missing_type.pop("type")
    tools = [TOOLS[0], {**TOOLS[1], "inputSchema": missing_type}]

    with pytest.raises(McpNormalizationError, match="explicitly declare type=object"):
        normalize_catalog(tools, McpOracleConfig.model_validate(_raw_config()))


def test_selected_tool_does_not_silently_repair_duplicate_required_entries() -> None:
    duplicate_required = {
        **TOOLS[1]["inputSchema"],
        "required": ["episode_id", "id", "id"],
    }
    tools = [TOOLS[0], {**TOOLS[1], "inputSchema": duplicate_required}]

    with pytest.raises(McpNormalizationError, match="required contains duplicates"):
        normalize_catalog(tools, McpOracleConfig.model_validate(_raw_config()))


def test_discovered_tool_name_must_match_the_exact_callable_name() -> None:
    tools = [TOOLS[0], {**TOOLS[1], "name": "inventory.lookup "}]

    with pytest.raises(McpCatalogError, match="surrounding whitespace"):
        normalize_catalog(tools, McpOracleConfig.model_validate(_raw_config()))


def test_episode_argument_must_accept_the_gateway_string_identifier() -> None:
    integer_episode = {
        **TOOLS[1]["inputSchema"],
        "properties": {
            **TOOLS[1]["inputSchema"]["properties"],
            "episode_id": {"type": "integer"},
        },
    }
    tools = [TOOLS[0], {**TOOLS[1], "inputSchema": integer_episode}]

    with pytest.raises(McpNormalizationError, match="must explicitly declare type=string"):
        normalize_catalog(tools, McpOracleConfig.model_validate(_raw_config()))


def test_discovery_paginates_and_writes_deterministic_report(tmp_path: Path) -> None:
    fake = _FakeClient()

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield fake

    report = asyncio.run(discover_mcp_oracle(_loaded(), connection_factory=connect))
    first = write_discovery_report(report, tmp_path / "first.json")
    second = write_discovery_report(report, tmp_path / "second.json")

    assert fake.cursors == [None, "page-2"]
    assert report.document["attained_level"] == "L0"
    assert report.document["implementation"]["sdk_version"] == "2.1.0-test"
    assert report.document["catalog"]["page_count"] == 2
    assert first.read_bytes() == second.read_bytes()
    parsed = json.loads(first.read_text())
    assert parsed["report_digest"] == report.document["report_digest"]


def test_report_export_refuses_a_document_mutated_after_signing(tmp_path: Path) -> None:
    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield _FakeClient()

    report = asyncio.run(discover_mcp_oracle(_loaded(), connection_factory=connect))
    report.document["status"] = "tampered"

    with pytest.raises(McpProtocolError, match="modified after report_digest"):
        report.to_dict()
    with pytest.raises(McpProtocolError, match="modified after report_digest"):
        write_discovery_report(report, tmp_path / "tampered.json")
    assert not (tmp_path / "tampered.json").exists()


def test_discovery_rejects_catalog_drift() -> None:
    loaded = _loaded()
    drifted = loaded.value.model_copy(
        update={
            "expected": loaded.value.expected.model_copy(
                update={"tool_catalog_digest": DIGEST}
            )
        }
    )

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield _FakeClient()

    with pytest.raises(McpIdentityMismatchError, match="catalog digest"):
        asyncio.run(
            discover_mcp_oracle(_loaded(drifted), connection_factory=connect)
        )


def test_bootstrap_discovery_reports_observed_catalog_digest() -> None:
    loaded = _loaded()
    unpinned = loaded.value.model_copy(
        update={
            "expected": loaded.value.expected.model_copy(
                update={"tool_catalog_digest": DIGEST}
            )
        }
    )

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield _FakeClient()

    report = asyncio.run(
        discover_mcp_oracle(
            _loaded(unpinned),
            connection_factory=connect,
            verify_catalog_digest=False,
        )
    )
    assert report.document["status"] == "needs_catalog_pin"
    assert report.document["attained_level"] is None
    assert report.tool_catalog_digest != DIGEST
    assert report.document["checks"][1]["status"] == "fail"


def test_discovery_rejects_pagination_cycles() -> None:
    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield _FakeClient(cursor_cycle=True)

    with pytest.raises(McpCatalogError, match="cursor cycle"):
        asyncio.run(discover_mcp_oracle(_loaded(), connection_factory=connect))


def test_source_config_digest_tracks_the_reviewed_document_not_the_host() -> None:
    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield _FakeClient()

    here = _loaded()
    # Same reviewed document, different checkout location on another host.
    elsewhere = LoadedMcpOracleConfig(
        path=Path("/srv/other-host/packs/mcp_oracle.yaml"),
        value=here.value,
        raw_document=here.raw_document,
    )

    first = asyncio.run(discover_mcp_oracle(here, connection_factory=connect))
    second = asyncio.run(discover_mcp_oracle(elsewhere, connection_factory=connect))

    expected = "sha256:" + hashlib.sha256(
        canonical_json(here.raw_document).encode()
    ).hexdigest()
    assert first.document["source_config_digest"] == expected
    assert second.document["source_config_digest"] == expected


def test_discovery_refuses_mismatched_source_and_runtime_config() -> None:
    loaded = _loaded()
    mismatched_raw = loaded.value.model_dump(mode="json", exclude_none=False)
    mismatched_raw["expected"]["server_version"] = "different-reviewed-version"

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        raise AssertionError("a mismatched config must fail before connecting")
        yield _FakeClient()

    with pytest.raises(McpConfigError, match="does not match the effective runtime config"):
        asyncio.run(
            discover_mcp_oracle(
                LoadedMcpOracleConfig(
                    path=loaded.path,
                    value=loaded.value,
                    raw_document=mismatched_raw,
                ),
                connection_factory=connect,
            )
        )


def _mode_a_raw() -> dict[str, Any]:
    raw = _raw_config()
    raw["mode"] = "A"
    raw["control"] = {
        "reset_strategy": "control_tool",
        "state_strategy": "control_tool",
        "describe_oracle": "bfcl_describe_oracle",
        "reset_episode": "bfcl_reset",
        "get_episode_state": "bfcl_get_state",
        "episode_binding": "argument",
        "episode_argument": "episode_id",
    }
    raw["fixtures"] = {"direction": "pushed"}
    raw["isolation"] = "process_per_episode"
    raw["expected"] = {**raw["expected"], "server_content_digest": CONTENT_DIGEST}
    return raw


class _ModeAClient(_FakeClient):
    def __init__(self, result: dict[str, Any]):
        super().__init__()
        self._result = result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._result


def _mode_a_loaded() -> LoadedMcpOracleConfig:
    config = _digest_config(McpOracleConfig.model_validate(_mode_a_raw()))
    raw = config.model_dump(mode="json", exclude_none=False)
    return LoadedMcpOracleConfig(
        path=Path("/tmp/mcp_oracle.yaml"),
        value=config,
        raw_document=raw,
    )


def test_describe_oracle_accepts_structured_identity() -> None:
    client = _ModeAClient(
        {
            "structuredContent": {
                "oracle_id": "catalog-oracle",
                "oracle_version": "1.0.0",
                "content_digest": CONTENT_DIGEST.upper().replace("SHA256", "sha256"),
            }
        }
    )

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield client

    report = asyncio.run(
        discover_mcp_oracle(_mode_a_loaded(), connection_factory=connect)
    )

    assert report.document["identity"]["server_content_digest"] == CONTENT_DIGEST


def test_describe_oracle_refuses_free_text_identity() -> None:
    client = _ModeAClient(
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "oracle_id": "catalog-oracle",
                            "oracle_version": "1.0.0",
                            "content_digest": CONTENT_DIGEST,
                        }
                    ),
                }
            ]
        }
    )

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield client

    with pytest.raises(McpProtocolError, match="structuredContent"):
        asyncio.run(discover_mcp_oracle(_mode_a_loaded(), connection_factory=connect))


def _mode_b_raw() -> dict[str, Any]:
    raw = _raw_config()
    raw["mode"] = "B"
    raw["control"] = {
        "reset_strategy": "process_restart",
        "state_strategy": "read_only_projection",
        "episode_binding": "argument",
        "episode_argument": "episode_id",
        "state_projection": [{"tool": "lookup", "arguments": {}}],
    }
    raw["fixtures"] = {"direction": "pushed"}
    raw["isolation"] = "process_per_episode"
    return raw


def test_confirmation_parameter_name_comes_from_reviewed_config() -> None:
    raw = _mode_b_raw()
    raw["tools"]["requires_confirmation"] = ["inventory_lookup"]
    raw["results"] = {"confirmation_parameter": "confirmed"}
    config = McpOracleConfig.model_validate(raw)

    source_schema = TOOLS[1]["inputSchema"]
    assert isinstance(source_schema, dict)
    gated = [
        TOOLS[0],
        {
            **TOOLS[1],
            "inputSchema": {
                **source_schema,
                "properties": {
                    **source_schema["properties"],
                    "confirmed": {"type": "boolean"},
                },
            },
        },
    ]

    catalog = normalize_catalog(gated, config)
    assert catalog.bfcl_tools[0]["x-requires-confirmation"] is True

    with pytest.raises(McpNormalizationError, match="'confirmed' input"):
        normalize_catalog(TOOLS, config)


@pytest.mark.parametrize(
    "constraint",
    [
        {"const": True},
        {"enum": [True]},
        {"enum": [False]},
    ],
)
def test_confirmation_parameter_must_allow_both_decisions(
    constraint: dict[str, Any],
) -> None:
    raw = _mode_b_raw()
    raw["tools"]["requires_confirmation"] = ["inventory_lookup"]
    config = McpOracleConfig.model_validate(raw)
    source_schema = TOOLS[1]["inputSchema"]
    constrained = [
        TOOLS[0],
        {
            **TOOLS[1],
            "inputSchema": {
                **source_schema,
                "properties": {
                    **source_schema["properties"],
                    "confirm": {"type": "boolean", **constraint},
                },
            },
        },
    ]

    with pytest.raises(McpNormalizationError, match="must allow both false and true"):
        normalize_catalog(constrained, config)


def test_confirmation_parameter_allows_explicit_boolean_enum() -> None:
    raw = _mode_b_raw()
    raw["tools"]["requires_confirmation"] = ["inventory_lookup"]
    config = McpOracleConfig.model_validate(raw)
    source_schema = TOOLS[1]["inputSchema"]
    tools = [
        TOOLS[0],
        {
            **TOOLS[1],
            "inputSchema": {
                **source_schema,
                "properties": {
                    **source_schema["properties"],
                    "confirm": {"type": "boolean", "enum": [True, False]},
                },
            },
        },
    ]

    catalog = normalize_catalog(tools, config)
    assert catalog.bfcl_tools[0]["x-requires-confirmation"] is True


def test_read_only_mode_refuses_to_trust_server_annotations() -> None:
    raw = _raw_config()
    raw["tools"]["trust_annotations"] = True

    with pytest.raises(ValueError, match="mutation from server annotations"):
        McpOracleConfig.model_validate(raw)


def test_episode_argument_may_not_shadow_the_confirmation_parameter() -> None:
    raw = _raw_config()
    raw["results"] = {"confirmation_parameter": "episode_id"}

    with pytest.raises(ValueError, match="must differ from results.confirmation_parameter"):
        McpOracleConfig.model_validate(raw)


def test_error_boundary_preserves_typed_failures() -> None:
    async def raise_catalog_error() -> None:
        async with mcp_error_boundary(SecretRegistry()):
            raise McpCatalogError("duplicate tool name")

    with pytest.raises(McpCatalogError, match="duplicate tool name"):
        asyncio.run(raise_catalog_error())


def test_error_boundary_redacts_typed_failures_without_losing_taxonomy() -> None:
    async def raise_catalog_error() -> None:
        async with mcp_error_boundary(SecretRegistry(values=("s3cr3t",))):
            raise McpCatalogError("server echoed s3cr3t in its catalog")

    with pytest.raises(McpCatalogError) as caught:
        asyncio.run(raise_catalog_error())

    assert "s3cr3t" not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_error_boundary_wraps_and_redacts_unexpected_failures() -> None:
    async def raise_runtime_error() -> None:
        async with mcp_error_boundary(SecretRegistry(values=("s3cr3t",))):
            raise RuntimeError("socket closed while sending s3cr3t")

    with pytest.raises(McpTransportError) as caught:
        asyncio.run(raise_runtime_error())

    assert "s3cr3t" not in str(caught.value)
    assert "<redacted>" in str(caught.value)


class _SdkPage:
    """Mimic the MCP SDK result model, whose fields mirror the camelCase wire schema."""

    def __init__(self, *, cursor_attribute: str | None):
        self.tools = [TOOLS[1]]
        if cursor_attribute is not None:
            setattr(self, cursor_attribute, "page-2")


class _SdkClient:
    def __init__(self, page: _SdkPage):
        self._page = page

    async def list_tools(self, cursor: str | None = None) -> _SdkPage:
        return self._page


def _facade(page: _SdkPage) -> SdkConnectedMcpClient:
    return SdkConnectedMcpClient(
        _SdkClient(page),
        max_response_bytes=100_000,
        sdk_version="2.1.0-test",
    )


def test_sdk_facade_refuses_to_persist_reflected_credentials() -> None:
    page = _SdkPage(cursor_attribute="nextCursor")
    page.tools = [
        {
            **TOOLS[1],
            "description": "credential accidentally echoed: s3cr3t-token",
        }
    ]
    facade = SdkConnectedMcpClient(
        _SdkClient(page),
        max_response_bytes=100_000,
        sdk_version="2.1.0-test",
        secrets=SecretRegistry(values=("s3cr3t-token",)),
    )

    with pytest.raises(McpCredentialError, match="reflected a configured credential"):
        asyncio.run(facade.list_tools())


def test_sdk_facade_classifies_non_json_payload_as_protocol_failure() -> None:
    page = _SdkPage(cursor_attribute="nextCursor")
    page.tools = [{**TOOLS[1], "annotations": {"score": float("nan")}}]

    with pytest.raises(McpProtocolError, match="not strict JSON"):
        asyncio.run(_facade(page).list_tools())


def test_short_header_value_does_not_match_unrelated_substrings() -> None:
    page = _SdkPage(cursor_attribute="nextCursor")
    page.tools = [{**TOOLS[1], "description": "Production catalog"}]
    facade = SdkConnectedMcpClient(
        _SdkClient(page),
        max_response_bytes=100_000,
        sdk_version="2.1.0-test",
        secrets=SecretRegistry(values=("prod",)),
    )

    discovered = asyncio.run(facade.list_tools())
    assert len(discovered.tools) == 1


@pytest.mark.parametrize("cursor_attribute", ["nextCursor", "next_cursor"])
def test_sdk_facade_reads_either_pagination_spelling(cursor_attribute: str) -> None:
    page = asyncio.run(_facade(_SdkPage(cursor_attribute=cursor_attribute)).list_tools())

    assert page.next_cursor == "page-2"


def test_sdk_facade_refuses_an_unrecognized_pagination_surface() -> None:
    # Defaulting a missing cursor to None would end pagination after page one and pin a
    # digest over a catalog BFCL never finished reading.
    with pytest.raises(McpProtocolError, match="continuation cursor"):
        asyncio.run(_facade(_SdkPage(cursor_attribute=None)).list_tools())


@pytest.mark.parametrize("invalid_cursor", ["", 7, [], {}])
def test_sdk_facade_refuses_invalid_pagination_cursor_types(
    invalid_cursor: Any,
) -> None:
    page = _SdkPage(cursor_attribute="nextCursor")
    page.nextCursor = invalid_cursor

    with pytest.raises(McpProtocolError, match="non-empty string or null"):
        asyncio.run(_facade(page).list_tools())


@pytest.mark.parametrize("injected", ["\u202e", "\u200b", "\ufeff", "\x07"])
def test_invisible_description_characters_fail_closed(injected: str) -> None:
    hidden = [TOOLS[0], {**TOOLS[1], "description": f"Look up{injected} one item."}]

    with pytest.raises(McpNormalizationError, match="invisible or"):
        normalize_catalog(hidden, McpOracleConfig.model_validate(_raw_config()))


def test_non_latin_descriptions_and_script_joiners_stay_publishable() -> None:
    # Vietnamese diacritics, an RTL script, a zero-width joiner, and a newline are all
    # ordinary description content; only overrides and truly invisible padding are not.
    text = "Tra cứu một mặt hàng.\nمرحبا\u200dبك"
    allowed = [TOOLS[0], {**TOOLS[1], "description": text}]

    catalog = normalize_catalog(allowed, McpOracleConfig.model_validate(_raw_config()))

    assert catalog.bfcl_tools[0]["function"]["description"] == text


def test_description_warnings_do_not_depend_on_the_description_language() -> None:
    smuggled = [
        TOOLS[0],
        {**TOOLS[1], "description": "Tra cứu.\n```\nxem https://evil.test\n```"},
    ]

    catalog = normalize_catalog(smuggled, McpOracleConfig.model_validate(_raw_config()))

    codes = {issue.code for issue in catalog.warnings}
    assert {"description_embeds_block", "description_embeds_url"} <= codes


def test_mutation_flag_records_that_reviewed_config_declared_it() -> None:
    raw = _mode_b_raw()
    raw["tools"]["mutates"] = ["inventory_lookup"]

    catalog = normalize_catalog(TOOLS, McpOracleConfig.model_validate(raw))

    assert catalog.tools[0].mutation_source == "config"
    assert catalog.bfcl_tools[0]["x-mutates"] is True
    # The fixture annotates readOnlyHint: true, so the two sides disagree.
    assert "mutation_disagreement" in {issue.code for issue in catalog.warnings}


def test_server_annotation_supplies_mutation_only_when_explicitly_trusted() -> None:
    raw = _mode_b_raw()
    mutating = [TOOLS[0], {**TOOLS[1], "annotations": {"readOnlyHint": False}}]

    raw["tools"]["trust_annotations"] = True
    trusted = normalize_catalog(mutating, McpOracleConfig.model_validate(raw))
    assert trusted.tools[0].mutation_source == "server_annotation"
    assert trusted.bfcl_tools[0]["x-mutates"] is True

    raw["tools"]["trust_annotations"] = False
    untrusted = normalize_catalog(mutating, McpOracleConfig.model_validate(raw))
    assert untrusted.tools[0].mutation_source is None
    assert "x-mutates" not in untrusted.bfcl_tools[0]
    assert "undeclared_mutation_hint" in {issue.code for issue in untrusted.warnings}


def test_describe_oracle_records_extra_declared_fields() -> None:
    client = _ModeAClient(
        {
            "structuredContent": {
                "oracle_id": "catalog-oracle",
                "oracle_version": "1.0.0",
                "content_digest": CONTENT_DIGEST,
                "build_id": "2026-08-25.1",
            }
        }
    )

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield client

    report = asyncio.run(
        discover_mcp_oracle(_mode_a_loaded(), connection_factory=connect)
    )

    assert report.document["identity"]["server_content_digest"] == CONTENT_DIGEST
    assert report.document["oracle_declaration"]["build_id"] == "2026-08-25.1"


def test_describe_oracle_requires_every_pinned_identity_field() -> None:
    client = _ModeAClient({"structuredContent": {"oracle_id": "catalog-oracle"}})

    @asynccontextmanager
    async def connect(config: McpOracleConfig):
        yield client

    with pytest.raises(McpProtocolError, match="omitted required identity field"):
        asyncio.run(discover_mcp_oracle(_mode_a_loaded(), connection_factory=connect))


def test_limits_reject_a_seconds_field_written_in_milliseconds() -> None:
    raw = _raw_config()
    raw["limits"]["tool_timeout_s"] = 5000

    with pytest.raises(ValueError, match="unit mistake"):
        McpOracleConfig.model_validate(raw)


def test_limits_reject_a_ceiling_that_disables_the_limit() -> None:
    raw = _raw_config()
    raw["limits"]["max_tools"] = 10**9

    with pytest.raises(ValueError, match="disables the limit"):
        McpOracleConfig.model_validate(raw)


def test_error_path_must_be_a_dotted_path() -> None:
    raw = _raw_config()
    raw["results"] = {"error_path": "result error"}

    with pytest.raises(ValueError, match="dotted path"):
        McpOracleConfig.model_validate(raw)
