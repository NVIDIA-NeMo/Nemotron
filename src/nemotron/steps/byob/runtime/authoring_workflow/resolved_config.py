# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical, provenance-carrying configuration for guided BFCL authoring."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator

from nemotron.steps.byob.runtime.authoring_workflow.rollout import (
    resolve_adapter_rollout,
)
from nemotron.steps.byob.runtime.mcp.config import load_unique_yaml_mapping
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.source_adapters.certification import AdapterTier
from nemotron.steps.byob.runtime.source_adapters.evidence import PackIdentity
from nemotron.steps.byob.runtime.source_adapters.registry import SourceDeclaration

RESOLVED_AUTHORING_CONFIG_VERSION: Literal["bfcl-resolved-authoring-config-v2"] = (
    "bfcl-resolved-authoring-config-v2"
)
LEGACY_RESOLVED_AUTHORING_CONFIG_VERSION = "bfcl-resolved-authoring-config-v1"
RESOLVED_AUTHORING_CONFIG_FILE = "resolved_authoring_config.json"
AUTHORING_POLICY_VERSION: Literal["bfcl-authoring-policy-v1"] = (
    "bfcl-authoring-policy-v1"
)
ConfigOrigin = Literal["user", "policy", "adapter", "derived"]
AdapterKind = Literal["local_python", "http_package", "mcp_mode_a"]
_SAFE_SLUG = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ResolvedConfigError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResolvedString(_StrictModel):
    value: StrictStr
    origin: ConfigOrigin
    derivation: StrictStr | None = None


class ResolvedStringList(_StrictModel):
    value: tuple[StrictStr, ...]
    origin: ConfigOrigin
    derivation: StrictStr | None = None

    @field_validator("value")
    @classmethod
    def _canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("resolved string lists must be non-empty, unique, and sorted")
        return value


class ResolvedOptionalString(_StrictModel):
    value: StrictStr | None
    origin: ConfigOrigin
    derivation: StrictStr | None = None


class ResolvedBool(_StrictModel):
    value: StrictBool
    origin: ConfigOrigin
    derivation: StrictStr | None = None


class ResolvedRolloutPolicy(_StrictModel):
    live_authoring_enabled: ResolvedBool
    environment_variable: ResolvedOptionalString
    legacy_alias_used: ResolvedBool


class ResolvedInputs(_StrictModel):
    source_declaration_digest: ResolvedString
    domain_brief_source_digest: ResolvedString
    policy_digest: ResolvedOptionalString


class ResolvedSemanticPayload(_StrictModel):
    adapter_kind: ResolvedString
    pack_id: ResolvedString
    pack_id_candidates: ResolvedStringList
    pack_version: ResolvedString
    required_certification_tier: ResolvedString
    tenant_id: ResolvedString
    run_id: ResolvedString
    rollout_policy: ResolvedRolloutPolicy | None = None


class ResolvedPaths(_StrictModel):
    source: ResolvedString
    domain_brief: ResolvedString
    workspace: ResolvedString
    policy: ResolvedOptionalString


class ResolvedConfirmations(_StrictModel):
    pack_id_confirmed: ResolvedBool
    pack_version_confirmed: ResolvedBool


class AuthoringPolicy(_StrictModel):
    schema_version: Literal["bfcl-authoring-policy-v1"]
    pack_id: StrictStr | None = None
    pack_version: StrictStr | None = None
    required_certification_tier: Literal["A0", "A1", "A2"] = "A0"
    adapter_rollout: dict[StrictStr, StrictBool] = Field(default_factory=dict)

    @field_validator("pack_id")
    @classmethod
    def _pack_id(cls, value: str | None) -> str | None:
        if value is not None:
            PackIdentity(pack_id=value, version="0")
        return value

    @field_validator("pack_version")
    @classmethod
    def _pack_version(cls, value: str | None) -> str | None:
        if value is not None:
            PackIdentity(pack_id="pack", version=value)
        return value


class ResolvedAuthoringConfig(_StrictModel):
    schema_version: Literal[
        "bfcl-resolved-authoring-config-v1",
        "bfcl-resolved-authoring-config-v2",
    ]
    inputs: ResolvedInputs
    semantic_payload: ResolvedSemanticPayload
    resolved_paths: ResolvedPaths
    confirmations: ResolvedConfirmations
    resolved_authoring_config_digest: StrictStr

    @field_validator("resolved_authoring_config_digest")
    @classmethod
    def _digest_shape(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("resolved authoring config digest must be lowercase SHA-256")
        return value

    def model_post_init(self, __context: Any) -> None:
        semantic = self.semantic_payload
        if (
            self.schema_version == RESOLVED_AUTHORING_CONFIG_VERSION
            and semantic.rollout_policy is None
        ):
            raise ValueError("resolved authoring config v2 requires rollout policy")
        if (
            self.schema_version == LEGACY_RESOLVED_AUTHORING_CONFIG_VERSION
            and semantic.rollout_policy is not None
        ):
            raise ValueError("resolved authoring config v1 cannot carry rollout policy")
        if semantic.adapter_kind.value not in {
            "local_python",
            "http_package",
            "mcp_mode_a",
        }:
            raise ValueError("resolved adapter kind is not built in")
        PackIdentity(
            pack_id=semantic.pack_id.value,
            version=semantic.pack_version.value,
        )
        if semantic.required_certification_tier.value not in {"A0", "A1", "A2"}:
            raise ValueError("resolved certification tier is invalid")
        unsigned = self.model_dump(
            mode="json",
            exclude={"resolved_authoring_config_digest"},
        )
        if semantic.rollout_policy is None:
            unsigned["semantic_payload"].pop("rollout_policy", None)
        if self.resolved_authoring_config_digest != sha256_json(unsigned):
            raise ValueError("resolved authoring config digest mismatch")


Prompt = Callable[[str], str]


def slug_pack_id_candidate(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.casefold()).strip("-")
    if not slug:
        slug = "pack"
    if not slug[0].isalpha():
        slug = f"pack-{slug}"
    slug = slug[:64].rstrip("-")
    if _SAFE_SLUG.fullmatch(slug) is None:
        raise ResolvedConfigError(
            "pack_id_candidate_invalid",
            f"cannot derive a safe pack ID from {value!r}",
            recovery="provide an explicit safe --pack-id",
        )
    return slug


def _load_policy(path: Path | None) -> tuple[AuthoringPolicy | None, str | None]:
    if path is None:
        return None, None
    source = path.resolve()
    document = load_unique_yaml_mapping(source, "authoring policy")
    try:
        policy = AuthoringPolicy.model_validate(document)
    except ValueError as exc:
        raise ResolvedConfigError(
            "authoring_policy_invalid",
            str(exc),
            recovery="repair the reviewed authoring policy",
        ) from exc
    return policy, sha256_json(document)


def _confirmed_value(
    *,
    name: str,
    explicit: str | None,
    policy_value: str | None,
    derived_value: str | None,
    confirm_derived: bool,
    ci: bool,
    prompt: Prompt,
) -> tuple[str, ConfigOrigin, bool]:
    if explicit is not None:
        return explicit, "user", True
    if policy_value is not None and name == "pack_version":
        return policy_value, "policy", False
    candidate = policy_value if policy_value is not None else derived_value
    if candidate is None:
        if ci:
            raise ResolvedConfigError(
                f"{name}_confirmation_required",
                f"{name.replace('_', ' ')} cannot be derived safely",
                recovery=f"provide --{name.replace('_', '-')}",
            )
        candidate = prompt(f"Enter {name.replace('_', ' ')}: ").strip()
        if not candidate:
            raise ResolvedConfigError(
                f"{name}_confirmation_required",
                f"{name.replace('_', ' ')} was not confirmed",
                recovery=f"provide --{name.replace('_', '-')}",
            )
        return candidate, "user", True
    if confirm_derived:
        return candidate, "user", True
    if ci:
        raise ResolvedConfigError(
            f"{name}_confirmation_required",
            f"candidate {candidate!r} requires explicit confirmation",
            recovery=(
                f"provide --{name.replace('_', '-')} {candidate} or "
                f"--confirm-{name.replace('_', '-')}"
            ),
        )
    answer = prompt(f"Confirm {name.replace('_', ' ')} {candidate!r} [y/N]: ")
    if answer.strip().casefold() not in {"y", "yes"}:
        raise ResolvedConfigError(
            f"{name}_confirmation_required",
            f"candidate {candidate!r} was not confirmed",
            recovery=f"rerun with an explicit --{name.replace('_', '-')}",
        )
    return candidate, "user", True


def resolve_authoring_config(
    *,
    adapter_kind: AdapterKind,
    source: Path,
    domain_brief: Path,
    workspace: Path,
    tenant_id: str,
    run_id: str,
    pack_id: str | None = None,
    pack_version: str | None = None,
    policy_path: Path | None = None,
    required_certification_tier: AdapterTier | None = None,
    confirm_pack_id: bool = False,
    confirm_pack_version: bool = False,
    ci: bool = False,
    prompt: Prompt = input,
    environ: Mapping[str, str] | None = None,
) -> ResolvedAuthoringConfig:
    source_path = source.resolve()
    brief_path = domain_brief.resolve()
    workspace_path = workspace.resolve()
    if not source_path.exists() or not brief_path.is_file():
        raise ResolvedConfigError(
            "authoring_input_missing",
            "source and domain brief must exist before configuration is resolved",
            recovery="provide reviewed local --source and --brief inputs",
        )
    policy, policy_digest = _load_policy(policy_path)
    rollout = resolve_adapter_rollout(
        adapter_kind,
        environ=environ,
        policy=policy.adapter_rollout if policy is not None else None,
    )
    candidate = slug_pack_id_candidate(
        source_path.name if source_path.is_dir() else source_path.stem
    )
    policy_pack_id = policy.pack_id if policy is not None else None
    candidates = tuple(sorted({candidate, *([policy_pack_id] if policy_pack_id else [])}))
    selected_pack_id, pack_id_origin, pack_id_confirmed = _confirmed_value(
        name="pack_id",
        explicit=pack_id,
        policy_value=policy_pack_id,
        derived_value=candidate,
        confirm_derived=confirm_pack_id,
        ci=ci,
        prompt=prompt,
    )
    selected_version, version_origin, version_confirmed = _confirmed_value(
        name="pack_version",
        explicit=pack_version,
        policy_value=policy.pack_version if policy is not None else None,
        derived_value=None,
        confirm_derived=confirm_pack_version,
        ci=ci,
        prompt=prompt,
    )
    PackIdentity(pack_id=selected_pack_id, version=selected_version)
    tier = required_certification_tier or AdapterTier(
        policy.required_certification_tier if policy is not None else "A0"
    )
    declaration = SourceDeclaration.model_validate(
        {
            "declaration_version": "bfcl-source-declaration-v1",
            adapter_kind: {"path": str(source_path)},
        }
    )
    unsigned: dict[str, Any] = {
        "schema_version": RESOLVED_AUTHORING_CONFIG_VERSION,
        "inputs": {
            "source_declaration_digest": {
                "value": declaration.digest,
                "origin": "derived",
                "derivation": "canonical source declaration",
            },
            "domain_brief_source_digest": {
                "value": f"sha256:{hashlib.sha256(brief_path.read_bytes()).hexdigest()}",
                "origin": "derived",
                "derivation": "domain brief bytes",
            },
            "policy_digest": {
                "value": policy_digest,
                "origin": "user" if policy_path is not None else "derived",
                "derivation": "canonical policy document" if policy_path else "no policy",
            },
        },
        "semantic_payload": {
            "adapter_kind": {
                "value": adapter_kind,
                "origin": "derived",
                "derivation": "static source adapter resolution",
            },
            "pack_id": {
                "value": selected_pack_id,
                "origin": pack_id_origin,
                "derivation": None,
            },
            "pack_id_candidates": {
                "value": candidates,
                "origin": "derived",
                "derivation": "normalized source name and reviewed policy hint",
            },
            "pack_version": {
                "value": selected_version,
                "origin": version_origin,
                "derivation": (
                    "authoring policy pack_version"
                    if version_origin == "policy"
                    else None
                ),
            },
            "required_certification_tier": {
                "value": tier.value,
                "origin": (
                    "user"
                    if required_certification_tier is not None
                    else "policy"
                    if policy is not None
                    else "derived"
                ),
                "derivation": "authoring certification policy",
            },
            "tenant_id": {"value": tenant_id, "origin": "user", "derivation": None},
            "run_id": {"value": run_id, "origin": "user", "derivation": None},
            "rollout_policy": {
                "live_authoring_enabled": {
                    "value": rollout.enabled,
                    "origin": (
                        "user"
                        if rollout.origin == "environment"
                        else "policy"
                        if rollout.origin == "policy"
                        else "derived"
                    ),
                    "derivation": f"rollout policy for {adapter_kind}",
                },
                "environment_variable": {
                    "value": rollout.environment_variable,
                    "origin": "derived",
                    "derivation": "selected per-adapter rollout control",
                },
                "legacy_alias_used": {
                    "value": rollout.legacy_alias_used,
                    "origin": "derived",
                    "derivation": "legacy MCP compatibility alias detection",
                },
            },
        },
        "resolved_paths": {
            "source": {"value": str(source_path), "origin": "user", "derivation": None},
            "domain_brief": {
                "value": str(brief_path),
                "origin": "user",
                "derivation": None,
            },
            "workspace": {
                "value": str(workspace_path),
                "origin": "derived",
                "derivation": "guided command workspace",
            },
            "policy": {
                "value": str(policy_path.resolve()) if policy_path is not None else None,
                "origin": "user" if policy_path is not None else "derived",
                "derivation": None,
            },
        },
        "confirmations": {
            "pack_id_confirmed": {
                "value": pack_id_confirmed,
                "origin": "user",
                "derivation": "explicit value or confirmation",
            },
            "pack_version_confirmed": {
                "value": version_confirmed,
                "origin": "user" if version_confirmed else "policy",
                "derivation": (
                    "explicit value or confirmation"
                    if version_confirmed
                    else "policy-provided version"
                ),
            },
        },
    }
    unsigned["resolved_authoring_config_digest"] = sha256_json(unsigned)
    return ResolvedAuthoringConfig.model_validate(unsigned)


def write_resolved_authoring_config(
    config: ResolvedAuthoringConfig,
    path: Path,
) -> Path:
    target = path.resolve()
    if target.exists():
        existing = load_resolved_authoring_config(target)
        if existing != config:
            raise ResolvedConfigError(
                "resolved_config_already_exists",
                f"immutable resolved config differs at {target}",
                recovery="create a new authoring revision or workspace",
            )
        return target
    return write_canonical_json(config.model_dump(mode="json"), target)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResolvedConfigError(
                "resolved_config_invalid",
                f"duplicate JSON key {key!r}",
                recovery="restore the canonical resolved config",
            )
        result[key] = value
    return result


def load_resolved_authoring_config(path: Path) -> ResolvedAuthoringConfig:
    source = path.resolve()
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        return ResolvedAuthoringConfig.model_validate(document)
    except ResolvedConfigError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResolvedConfigError(
            "resolved_config_invalid",
            f"cannot load {source}: {exc}",
            recovery="restore or regenerate the canonical resolved config",
        ) from exc


def verify_resolved_authoring_inputs(
    config: ResolvedAuthoringConfig,
    *,
    adapter_kind: str,
    source: Path,
    domain_brief: Path,
) -> None:
    """Reject input substitution after the authoring configuration was resolved."""
    source_path = source.resolve()
    brief_path = domain_brief.resolve()
    semantic = config.semantic_payload
    if semantic.adapter_kind.value != adapter_kind:
        raise ResolvedConfigError(
            "resolved_adapter_mismatch",
            "live adapter differs from the resolved authoring configuration",
            recovery="create a new authoring revision for the selected adapter",
        )
    if (
        config.resolved_paths.source.value != str(source_path)
        or config.resolved_paths.domain_brief.value != str(brief_path)
    ):
        raise ResolvedConfigError(
            "resolved_input_path_mismatch",
            "live source or domain brief path differs from resolved configuration",
            recovery="use the exact resolved inputs or create a new revision",
        )
    try:
        brief_digest = f"sha256:{hashlib.sha256(brief_path.read_bytes()).hexdigest()}"
        declaration = SourceDeclaration.model_validate(
            {
                "declaration_version": "bfcl-source-declaration-v1",
                adapter_kind: {"path": str(source_path)},
            }
        )
    except (OSError, ValueError) as exc:
        raise ResolvedConfigError(
            "resolved_input_unreadable",
            f"cannot verify resolved authoring inputs: {exc}",
            recovery="restore the exact reviewed source and domain brief",
        ) from exc
    if (
        brief_digest != config.inputs.domain_brief_source_digest.value
        or declaration.digest != config.inputs.source_declaration_digest.value
    ):
        raise ResolvedConfigError(
            "resolved_input_digest_mismatch",
            "source declaration or domain brief changed after configuration resolution",
            recovery="restore the reviewed bytes or create a new authoring revision",
        )


def resolved_config_digest(path: Path) -> str:
    return load_resolved_authoring_config(path).resolved_authoring_config_digest
