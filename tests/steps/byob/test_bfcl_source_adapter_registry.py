from __future__ import annotations

import builtins
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.source_adapters.registry import (
    BUILTIN_ADAPTER_REGISTRY,
    AdapterRegistration,
    AdapterRegistry,
    AdapterResolutionError,
    SourceDeclaration,
    resolve_source_adapter,
)


def _declaration(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "declaration_version": "bfcl-source-declaration-v1",
        "adapter": None,
        "local_python": None,
        "http_package": None,
        "mcp_mode_a": {"path": "configs/mcp-intake.yaml"},
        "extensions": {},
    }
    document.update(overrides)
    return document


@pytest.mark.parametrize(
    ("field", "adapter_id"),
    [
        ("local_python", "local_python"),
        ("http_package", "http_package"),
        ("mcp_mode_a", "mcp_mode_a"),
    ],
)
def test_each_builtin_adapter_resolves_from_inert_declaration_data(
    field: str,
    adapter_id: str,
) -> None:
    declaration = _declaration(mcp_mode_a=None)
    declaration[field] = {"path": f"sources/{field}"}

    resolved = resolve_source_adapter(declaration)

    assert resolved.adapter_id == adapter_id
    assert resolved.descriptor_kind == adapter_id
    assert resolved.source.path == f"sources/{field}"
    assert resolved.declaration_digest == SourceDeclaration.model_validate(
        declaration
    ).digest


def test_explicit_adapter_must_be_allowlisted_and_match_its_source_block() -> None:
    resolved = resolve_source_adapter(
        _declaration(adapter="mcp_mode_a")
    )
    assert resolved.adapter_id == "mcp_mode_a"

    with pytest.raises(AdapterResolutionError) as unknown:
        resolve_source_adapter(_declaration(adapter="third_party.plugin"))
    assert unknown.value.code == "adapter_not_allowlisted"

    with pytest.raises(AdapterResolutionError) as mismatch:
        resolve_source_adapter(
            _declaration(
                adapter="http_package",
                mcp_mode_a=None,
                local_python={"path": "backend.py"},
            )
        )
    assert mismatch.value.code == "adapter_source_mismatch"


def test_zero_and_multiple_matches_have_stable_fail_closed_codes() -> None:
    with pytest.raises(AdapterResolutionError) as zero:
        resolve_source_adapter(_declaration(mcp_mode_a=None))
    assert zero.value.code == "adapter_not_detected"

    with pytest.raises(AdapterResolutionError) as multiple:
        resolve_source_adapter(
            _declaration(http_package={"path": "endpoint-package"})
        )
    assert multiple.value.code == "adapter_detection_ambiguous"
    assert "http_package, mcp_mode_a" in multiple.value.detail


def test_explicit_adapter_cannot_hide_an_ambiguous_second_source() -> None:
    with pytest.raises(AdapterResolutionError) as error:
        resolve_source_adapter(
            _declaration(
                adapter="mcp_mode_a",
                http_package={"path": "endpoint-package"},
            )
        )

    assert error.value.code == "adapter_detection_ambiguous"


def test_detection_performs_no_io_network_subprocess_or_runtime_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = SourceDeclaration.model_validate(_declaration())

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("adapter detection attempted a side effect")

    with monkeypatch.context() as isolated:
        isolated.setattr(builtins, "open", forbidden)
        isolated.setattr(socket, "socket", forbidden)
        isolated.setattr(socket, "create_connection", forbidden)
        isolated.setattr(subprocess, "Popen", forbidden)
        isolated.setattr(subprocess, "run", forbidden)
        isolated.setattr(Path, "exists", forbidden)
        isolated.setattr(Path, "is_file", forbidden)
        # Patch imports last because monkeypatch itself lazily imports ``inspect``
        # while installing dotted/object attributes.
        isolated.setattr(builtins, "__import__", forbidden)

        resolved = BUILTIN_ADAPTER_REGISTRY.resolve(declaration)

    assert resolved.adapter_id == "mcp_mode_a"


def test_extensions_are_namespaced_json_bounded_and_non_authoritative() -> None:
    resolved = resolve_source_adapter(
        _declaration(
            extensions={
                "acme.v1": {
                    "display": {"color": "green"},
                    "retry_hint": 2,
                }
            }
        )
    )
    assert resolved.adapter_id == "mcp_mode_a"
    assert resolved.extensions["acme.v1"]["retry_hint"] == 2
    with pytest.raises(TypeError):
        resolved.extensions["other.v1"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        resolved.extensions["acme.v1"]["display"]["color"] = "red"  # type: ignore[index]

    for extensions, message in (
        ({"not_namespaced": {}}, "dotted lowercase namespaces"),
        ({"bfcl.v1": {}}, "reserved bfcl namespace"),
        ({"acme.v1": {"attained_tier": "A2"}}, "authoritative field"),
        ({"acme.v1": {"nested": {"gold_eligible": True}}}, "authoritative field"),
        ({"acme.v1": {"value": object()}}, "JSON values only"),
        ({"acme.v1": {"value": float("nan")}}, "non-finite"),
    ):
        with pytest.raises(AdapterResolutionError, match=message) as error:
            resolve_source_adapter(_declaration(extensions=extensions))
        assert error.value.code == "source_declaration_invalid"


def test_extensions_change_declaration_identity_but_not_adapter_selection() -> None:
    plain = resolve_source_adapter(_declaration())
    extended = resolve_source_adapter(
        _declaration(extensions={"acme.v1": {"label": "inventory"}})
    )

    assert plain.adapter_id == extended.adapter_id == "mcp_mode_a"
    assert plain.declaration_digest != extended.declaration_digest


def test_unknown_top_level_authority_fields_are_rejected() -> None:
    with pytest.raises(AdapterResolutionError) as error:
        resolve_source_adapter(
            {
                **_declaration(),
                "attained_tier": "A2",
            }
        )

    assert error.value.code == "source_declaration_invalid"
    assert "Extra inputs are not permitted" in error.value.detail


def test_registry_metadata_cannot_name_factories_modules_or_unknown_fields() -> None:
    assert [item.adapter_id for item in BUILTIN_ADAPTER_REGISTRY.registrations] == [
        "http_package",
        "local_python",
        "mcp_mode_a",
    ]
    assert all(
        set(item.__dataclass_fields__)  # type: ignore[attr-defined]
        == {"adapter_id", "declaration_field", "descriptor_kind"}
        for item in BUILTIN_ADAPTER_REGISTRY.registrations
    )
    with pytest.raises(AttributeError, match="immutable"):
        BUILTIN_ADAPTER_REGISTRY._registrations = ()  # type: ignore[misc]

    with pytest.raises(AdapterResolutionError) as invalid:
        AdapterRegistration(
            adapter_id="plugin",
            declaration_field="plugin",
            descriptor_kind="plugin",
        )
    assert invalid.value.code == "adapter_registry_invalid"

    with pytest.raises(AdapterResolutionError) as duplicate:
        AdapterRegistry(
            (
                AdapterRegistration(
                    adapter_id="first",
                    declaration_field="mcp_mode_a",
                    descriptor_kind="mcp_mode_a",
                ),
                AdapterRegistration(
                    adapter_id="second",
                    declaration_field="mcp_mode_a",
                    descriptor_kind="mcp_mode_a",
                ),
            )
        )
    assert duplicate.value.code == "adapter_registry_invalid"


def test_mutated_nested_declaration_is_revalidated_before_resolution() -> None:
    declaration = SourceDeclaration.model_validate(
        _declaration(extensions={"acme.v1": {"display": {"color": "green"}}})
    )
    declaration.extensions["acme.v1"]["attained_tier"] = "A2"

    with pytest.raises(AdapterResolutionError, match="authoritative field"):
        resolve_source_adapter(declaration)
