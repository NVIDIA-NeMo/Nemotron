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

"""Bind certified source, compiled drafts, and reviewed semantics into a candidate pack.

Drafting stops at proposals. `runner.py` writes its plans *next to* a pack rather than into
one, and `compile_assertions.py` emits only trace predicates, because a model that has
never probed a server must not be able to state what that server returns. What a loadable
pack still needs on top of that — slots bound to fixture columns, turn policies, per
language user turns — is deliberately absent from the draft schema for the same reason.

So this step is a binder rather than an author. Everything mechanical is derived from
artifacts that are already trusted: pack identity from verified evidence, `backend.py`,
`tools.json`, and `fixtures.json` from the exact source tree certification fingerprinted,
and `assertions.py` from drafts that compiled without a blocker. The semantics a model
cannot supply arrive as one reviewed supplement, and every tool and assertion that
supplement names is checked back against those artifacts, so it cannot introduce a tool the
source never published or an assertion nobody compiled.

A source reached over a session — an HTTP endpoint or an MCP Mode A gateway — is bound the
same way but from different files, because the oracle is not in the tree. `endpoint_config.yaml`
replaces `backend.py` and is accepted only when it pins the identity the evidence certified,
`tools.json` is checked tool by tool against the published surface rather than trusted for
sitting in the right directory, and `fixtures.json` comes from the reviewed probe plan whose
digest signed evidence already carries, because a session is handed its world at open rather
than reading it from the pack.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.assertion_capabilities import (
    AssertionCapabilityError,
    read_literal_assertion_capabilities,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    load_endpoint_config,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    SourceEvidenceDocument,
    ToolEvidence,
    load_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.local_python import (
    LocalPythonError,
    inspect_local_python_package,
)
from nemotron.steps.byob.runtime.source_adapters.probe_engine import AdapterProbePlan
from nemotron.steps.byob.runtime.source_adapters.reviewed_catalog import (
    ReviewedCatalogError,
    load_reviewed_tool_catalog,
)

SUPPLEMENT_VERSION: Literal[
    "bfcl-candidate-pack-supplement-v1"
] = "bfcl-candidate-pack-supplement-v1"
RECORD_VERSION: Literal[
    "bfcl-candidate-pack-record-v1"
] = "bfcl-candidate-pack-record-v1"
PACK_DIRECTORY_NAME = "pack"
RECORD_FILE_NAME = "candidate_pack_provenance.json"
ASSERTIONS_FILE_NAME = "assertions.py"
DRAFT_PROVENANCE_FILE_NAME = "draft_provenance.json"

ENDPOINT_CONFIG_FILE_NAME = "endpoint_config.yaml"
FIXTURES_FILE_NAME = "fixtures.json"
TOOLS_FILE_NAME = "tools.json"

# The oracle for these kinds is reached over a session, so the pack pins an endpoint rather
# than importing a package, and its world arrives from the reviewed probe plan.
SESSION_ADAPTER_KINDS: frozenset[str] = frozenset({"http_package", "mcp_mode_a"})

# Copied verbatim from the certified source tree. `fixtures.json` is optional there, so it
# is the only one that may be absent.
_LOCAL_SOURCE_FILES: tuple[tuple[str, bool], ...] = (
    ("backend.py", True),
    (TOOLS_FILE_NAME, True),
    (FIXTURES_FILE_NAME, False),
)
_SESSION_SOURCE_FILES: tuple[tuple[str, bool], ...] = (
    (ENDPOINT_CONFIG_FILE_NAME, True),
    (TOOLS_FILE_NAME, True),
)


class PackAssemblyError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


class CandidatePackSupplement(BaseModel):
    """The pack semantics no draft can express, supplied once and reviewed as a unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bfcl-candidate-pack-supplement-v1"]
    languages: tuple[StrictStr, ...]
    clock: StrictStr
    task_templates: tuple[dict[str, Any], ...]
    validation_cases: tuple[dict[str, Any], ...]
    absent_ids: dict[StrictStr, tuple[StrictStr, ...]] = {}
    primary_keys: dict[StrictStr, StrictStr] = {}
    assistant_turn_templates: dict[StrictStr, dict[StrictStr, StrictStr]] = {}
    confirmation: dict[StrictStr, StrictStr] | None = None
    system_prompt: StrictStr | None = None

    @field_validator("languages")
    @classmethod
    def _languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("supplement must declare at least one language")
        if len(set(value)) != len(value):
            raise ValueError("supplement languages must be unique")
        if any(not language.strip() for language in value):
            raise ValueError("supplement languages must be non-empty")
        return value

    @field_validator("clock")
    @classmethod
    def _clock(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("supplement clock must be non-empty")
        return value

    @field_validator("task_templates", "validation_cases")
    @classmethod
    def _non_empty(cls, value: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        if not value:
            raise ValueError("supplement must declare at least one entry")
        return value


@dataclass(frozen=True)
class AssembledCandidatePack:
    output_root: Path
    pack_root: Path
    record_path: Path
    record: Mapping[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.pack_root / "manifest.yaml"


def load_candidate_pack_supplement(path: Path) -> CandidatePackSupplement:
    """Read the reviewed supplement, refusing anything the schema does not name."""
    source = path.resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PackAssemblyError(
            "supplement_unreadable",
            f"cannot read pack supplement {source}: {type(exc).__name__}: {exc}",
            recovery="supply a readable YAML supplement",
        ) from exc
    if not isinstance(document, dict):
        raise PackAssemblyError(
            "supplement_invalid",
            "pack supplement must be a mapping",
            recovery=f"write a {SUPPLEMENT_VERSION} mapping",
        )
    try:
        return CandidatePackSupplement.model_validate(document)
    except ValueError as exc:
        raise PackAssemblyError(
            "supplement_invalid",
            f"pack supplement does not satisfy {SUPPLEMENT_VERSION}: {exc}",
            recovery="correct the supplement against the documented schema",
        ) from exc


def _digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackAssemblyError(
            "draft_provenance_unreadable",
            f"cannot read {label} {path}: {type(exc).__name__}: {exc}",
            recovery="assemble from a drafting run that completed",
        ) from exc
    if not isinstance(document, dict):
        raise PackAssemblyError(
            "draft_provenance_unreadable",
            f"{label} must be a mapping",
            recovery="assemble from a drafting run that completed",
        )
    return document


def _verified_evidence(evidence_path: Path) -> SourceEvidenceDocument:
    try:
        return load_source_evidence(evidence_path)
    except ValueError as exc:
        raise PackAssemblyError(
            "evidence_invalid",
            f"cannot load verified source evidence: {exc}",
            recovery="assemble from the evidence bundle intake published",
        ) from exc


def _bind_source(
    source_root: Path,
    evidence: SourceEvidenceDocument,
) -> None:
    """Refuse unless the source tree is the one certification fingerprinted."""
    try:
        inspection = inspect_local_python_package(
            source_root,
            allowed_roots=(source_root,),
        )
    except LocalPythonError as exc:
        raise PackAssemblyError(
            "source_package_invalid",
            f"cannot inspect the local Python source package: {exc}",
            recovery="assemble from the source package intake certified",
        ) from exc
    if inspection.source_identity_digest != sha256_json(
        evidence.identity.model_dump(mode="json")
    ):
        raise PackAssemblyError(
            "source_identity_mismatch",
            "source package identity differs from the verified evidence",
            recovery="assemble from the exact source revision intake certified",
        )


def _tool_surface(tool: ToolEvidence) -> dict[str, Any]:
    """The part of a tool a benchmark publishes, and therefore the part that must match."""
    return {
        "published_name": tool.published_name,
        "description": tool.description.untrusted_text,
        "parameter_schema": tool.parameter_schema,
        "mutates": tool.mutates,
        "requires_confirmation": tool.requires_confirmation,
    }


def _bind_session_catalog(tools_path: Path, evidence: SourceEvidenceDocument) -> None:
    """Refuse unless the pack's catalog is the surface certification saw."""
    try:
        catalog = load_reviewed_tool_catalog(tools_path)
    except ReviewedCatalogError as exc:
        raise PackAssemblyError(
            "source_catalog_invalid",
            f"reviewed {TOOLS_FILE_NAME} is not loadable: {exc}",
            recovery=f"assemble from the {TOOLS_FILE_NAME} intake published",
        ) from exc
    observed = {tool.published_name: _tool_surface(tool) for tool in catalog.tools}
    certified = {tool.published_name: _tool_surface(tool) for tool in evidence.tools}
    if divergent := sorted(
        name
        for name in set(observed) | set(certified)
        if observed.get(name) != certified.get(name)
    ):
        raise PackAssemblyError(
            "source_catalog_mismatch",
            f"{TOOLS_FILE_NAME} differs from the certified tool surface: "
            + ", ".join(divergent),
            recovery=f"assemble the exact {TOOLS_FILE_NAME} intake certified",
        )


def _bind_session_source(
    source_root: Path,
    evidence: SourceEvidenceDocument,
) -> EndpointConfig:
    """Refuse unless the endpoint declaration pins the identity evidence certified."""
    endpoint_path = source_root / ENDPOINT_CONFIG_FILE_NAME
    try:
        config = load_endpoint_config(endpoint_path, allowed_roots=(source_root,))
    except (OSError, ValueError) as exc:
        raise PackAssemblyError(
            "endpoint_config_invalid",
            f"cannot load {ENDPOINT_CONFIG_FILE_NAME}: {type(exc).__name__}: {exc}",
            recovery="assemble from the endpoint declaration intake certified",
        ) from exc
    if config.expected.content_digest != evidence.identity.effective_content_digest:
        raise PackAssemblyError(
            "source_identity_mismatch",
            "endpoint declaration pins a different effective content digest "
            "than the verified evidence",
            recovery="assemble from the exact endpoint revision intake certified",
        )
    attested = next(
        (
            artifact.digest
            for artifact in evidence.identity.artifacts
            if artifact.role == "attestation"
        ),
        None,
    )
    if attested is not None and (
        config.attestation is None or config.attestation.expected_digest != attested
    ):
        raise PackAssemblyError(
            "source_identity_mismatch",
            "endpoint declaration does not pin the conformance attestation "
            "certification bound",
            recovery="assemble from the endpoint declaration intake published",
        )
    _bind_session_catalog(source_root / TOOLS_FILE_NAME, evidence)
    return config


def _session_fixtures(
    probe_plan_path: Path | None,
    evidence: SourceEvidenceDocument,
) -> dict[str, Any] | None:
    """Return the reviewed world a session is handed, bound to signed evidence."""
    if evidence.fixtures.direction in {"none", "read_only"}:
        if probe_plan_path is not None:
            raise PackAssemblyError(
                "fixtures_not_applicable",
                "evidence declares no session fixtures, so a probe plan cannot supply them",
                recovery="certify the source with a reviewed probe plan, then assemble",
            )
        return None
    if probe_plan_path is None:
        raise PackAssemblyError(
            "fixtures_missing",
            "a session-backed pack needs the reviewed probe plan that carries its fixtures",
            recovery="pass the same probe plan intake certified this source with",
        )
    try:
        document = json.loads(probe_plan_path.resolve().read_text(encoding="utf-8"))
        plan = AdapterProbePlan.model_validate(document)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PackAssemblyError(
            "probe_plan_invalid",
            f"cannot read the reviewed probe plan: {type(exc).__name__}: {exc}",
            recovery="pass the probe plan intake certified this source with",
        ) from exc
    if plan.fixtures is None:
        raise PackAssemblyError(
            "fixtures_missing",
            "the reviewed probe plan declares no fixtures",
            recovery="review a probe plan that states the world sessions are handed",
        )
    if evidence.fixtures.content_digest != sha256_json(plan.fixtures):
        raise PackAssemblyError(
            "fixtures_mismatch",
            "probe plan fixtures differ from the snapshot the evidence certified",
            recovery="pass the exact probe plan intake certified this source with",
        )
    return plan.fixtures


def _compiled_assertion_names(assertions_path: Path) -> frozenset[str]:
    try:
        declared = read_literal_assertion_capabilities(assertions_path)
    except AssertionCapabilityError as exc:
        raise PackAssemblyError(
            "compiled_assertions_invalid",
            f"compiled assertions do not declare readable capabilities: {exc}",
            recovery="recompile the assertion specifications",
        ) from exc
    if not declared:
        raise PackAssemblyError(
            "compiled_assertions_invalid",
            "compiled assertions declare no capabilities",
            recovery="recompile the assertion specifications",
        )
    return frozenset(declared)


@dataclass(frozen=True)
class _BoundDrafts:
    """The compiled assertions, and the drafting record they are accountable to."""

    assertions_path: Path
    provenance_digest: str


def _bind_drafts(draft_root: Path, evidence: SourceEvidenceDocument) -> _BoundDrafts:
    """Return the compiled assertions of a drafting run of this exact evidence."""
    provenance_path = draft_root.parent / DRAFT_PROVENANCE_FILE_NAME
    if not provenance_path.is_file():
        raise PackAssemblyError(
            "draft_provenance_missing",
            f"no drafting record beside the draft root: {provenance_path}",
            recovery="assemble from the output root a drafting run wrote",
        )
    provenance = _load_json_mapping(provenance_path, "draft provenance")
    recorded = provenance.get("evidence")
    bundle_digest = (
        recorded.get("bundle_digest") if isinstance(recorded, dict) else None
    )
    if bundle_digest != evidence.bundle_digest:
        raise PackAssemblyError(
            "draft_evidence_mismatch",
            "drafts were produced from a different evidence revision",
            recovery="redraft the current evidence revision before assembling",
        )
    if provenance.get("blocked_on"):
        raise PackAssemblyError(
            "draft_blocked",
            "drafts are still blocked on "
            + ", ".join(str(item) for item in provenance["blocked_on"]),
            recovery="certify the probes that resolve those unknowns and redraft",
        )
    if not provenance.get("assertions_compiled"):
        raise PackAssemblyError(
            "compiled_assertions_missing",
            "the drafting run compiled no assertions",
            recovery="resolve the compilation refusals and redraft",
        )
    assertions_path = draft_root / ASSERTIONS_FILE_NAME
    if not assertions_path.is_file():
        raise PackAssemblyError(
            "compiled_assertions_missing",
            f"no compiled assertions in the draft root: {assertions_path}",
            recovery="redraft so compilation writes assertions.py",
        )
    # The record digests this rather than each draft in turn: the drafting record already
    # carries `artifact_digests` for all four plans, so one digest here makes the whole set
    # tamper-evident, and a pack can name which proposals it was assembled from.
    return _BoundDrafts(
        assertions_path=assertions_path,
        provenance_digest=_digest_file(provenance_path),
    )


def _referenced_tools(supplement: CandidatePackSupplement) -> set[str]:
    names: set[str] = set()
    for template in supplement.task_templates:
        for key in ("required_tools", "tools_present"):
            names.update(_string_items(template.get(key)))
        for milestone in _mapping_items(template.get("assistant_milestones")):
            tool = milestone.get("tool")
            if isinstance(tool, str) and tool.strip():
                names.add(tool)
    for case in supplement.validation_cases:
        tool = case.get("tool")
        if isinstance(tool, str) and tool.strip():
            names.add(tool)
    return names


def _referenced_assertions(supplement: CandidatePackSupplement) -> set[str]:
    names: set[str] = set()
    for template in supplement.task_templates:
        names.update(_string_items(template.get("success_assertions")))
    return names


def _string_items(value: Any) -> Iterable[str]:
    if not isinstance(value, list):
        return ()
    return [item for item in value if isinstance(item, str) and item.strip()]


def _mapping_items(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, list):
        return ()
    return [item for item in value if isinstance(item, Mapping)]


def _check_supplement_references(
    supplement: CandidatePackSupplement,
    *,
    evidence: SourceEvidenceDocument,
    assertion_names: frozenset[str],
) -> None:
    published = {tool.published_name for tool in evidence.tools}
    if unknown := sorted(_referenced_tools(supplement) - published):
        raise PackAssemblyError(
            "supplement_tool_unknown",
            "supplement names tools the certified source never published: "
            + ", ".join(unknown),
            recovery="reference only the tools the evidence bundle publishes",
        )
    if unknown := sorted(_referenced_assertions(supplement) - assertion_names):
        raise PackAssemblyError(
            "supplement_assertion_unknown",
            "supplement names assertions the drafts never compiled: "
            + ", ".join(unknown),
            recovery="draft and compile those assertion specifications first",
        )


def _manifest_document(
    supplement: CandidatePackSupplement,
    *,
    evidence: SourceEvidenceDocument,
    has_fixtures: bool,
) -> dict[str, Any]:
    paths: dict[str, str] = {
        "tools": TOOLS_FILE_NAME,
        "templates": "task_templates.yaml",
        "assertions": ASSERTIONS_FILE_NAME,
        "validation_cases": "validation_cases.yaml",
    }
    # The loader accepts exactly one oracle, so the pack names the one its source is.
    if evidence.source_adapter.kind in SESSION_ADAPTER_KINDS:
        paths["endpoint"] = ENDPOINT_CONFIG_FILE_NAME
    else:
        paths["backend"] = "backend.py"
    if has_fixtures:
        paths["fixtures"] = FIXTURES_FILE_NAME
    manifest: dict[str, Any] = {
        "pack_id": evidence.pack.pack_id,
        "version": evidence.pack.version,
        "languages": list(supplement.languages),
        "clock": supplement.clock,
        "paths": paths,
    }
    if supplement.absent_ids:
        manifest["absent_ids"] = {
            collection: list(identifiers)
            for collection, identifiers in sorted(supplement.absent_ids.items())
        }
    if supplement.primary_keys:
        manifest["primary_keys"] = dict(sorted(supplement.primary_keys.items()))
    if supplement.assistant_turn_templates:
        manifest["assistant_turn_templates"] = {
            name: dict(sorted(turns.items()))
            for name, turns in sorted(supplement.assistant_turn_templates.items())
        }
    confirmation = supplement.confirmation
    if confirmation is None:
        # Certified evidence already states this vocabulary for a session source, and a
        # pack whose confirmation names differ from the oracle's would gate nothing.
        declared = {
            "parameter": evidence.vocabulary.parameter,
            "status_field": evidence.vocabulary.status_field,
            "pending_status": evidence.vocabulary.pending_status,
        }
        if all(value is not None for value in declared.values()):
            confirmation = cast(dict[str, str], declared)
    if confirmation is not None:
        manifest["confirmation"] = dict(sorted(confirmation.items()))
    if supplement.system_prompt is not None:
        # Stated inline rather than as a path: nothing copies a sidecar prompt file into
        # the pack, and a pack whose render language is not the default prompt's cannot
        # publish a gold row without its own prompt.
        manifest["system_prompt"] = supplement.system_prompt
    return manifest


def _write_yaml(document: Any, path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def assemble_candidate_pack(
    *,
    evidence_path: Path,
    source_root: Path,
    draft_root: Path,
    supplement_path: Path,
    output_root: Path,
    probe_plan_path: Path | None = None,
) -> AssembledCandidatePack:
    """Write one candidate oracle pack, or refuse and name the binding that failed."""
    final_root = output_root.resolve()
    if final_root.exists():
        raise PackAssemblyError(
            "pack_output_exists",
            f"candidate pack output already exists: {final_root}",
            recovery="assemble into a fresh directory so no earlier pack is overwritten",
        )
    evidence = _verified_evidence(evidence_path)
    kind = evidence.source_adapter.kind
    session_backed = kind in SESSION_ADAPTER_KINDS
    if kind != "local_python" and not session_backed:
        raise PackAssemblyError(
            "adapter_not_supported",
            f"candidate pack assembly has no binder for adapter kind {kind}",
            recovery="assemble a built-in adapter, or compile the pack by hand",
        )
    source = source_root.resolve()
    endpoint: EndpointConfig | None = None
    fixtures: dict[str, Any] | None = None
    if session_backed:
        endpoint = _bind_session_source(source, evidence)
        fixtures = _session_fixtures(probe_plan_path, evidence)
    else:
        if probe_plan_path is not None:
            raise PackAssemblyError(
                "fixtures_not_applicable",
                "a local Python pack carries its own fixtures, so a probe plan "
                "cannot supply them",
                recovery="assemble a local source without --probe-plan",
            )
        _bind_source(source, evidence)
    drafts = _bind_drafts(draft_root.resolve(), evidence)
    assertion_names = _compiled_assertion_names(drafts.assertions_path)
    supplement = load_candidate_pack_supplement(supplement_path)
    _check_supplement_references(
        supplement,
        evidence=evidence,
        assertion_names=assertion_names,
    )

    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(
        dir=final_root.parent,
        prefix=f".{final_root.name}.staging-",
    )
    try:
        root = Path(staging.name)
        pack_root = root / PACK_DIRECTORY_NAME
        pack_root.mkdir()
        copied: list[str] = []
        for name, required in (
            _SESSION_SOURCE_FILES if session_backed else _LOCAL_SOURCE_FILES
        ):
            origin = source / name
            if not origin.is_file():
                if required:
                    raise PackAssemblyError(
                        "source_package_invalid",
                        f"certified source package has no {name}",
                        recovery="assemble from a complete certified source package",
                    )
                continue
            # copyfile rather than copy2: a pack containing a symlink is refused later,
            # and metadata from the source tree is not part of the pack contract.
            shutil.copyfile(origin, pack_root / name)
            copied.append(name)
        if endpoint is not None and endpoint.ca_bundle_path is not None:
            # The endpoint loader requires the bundle inside the pack tree, so the pack
            # has to carry it under the relative name its own config already names.
            bundle_name = Path(endpoint.ca_bundle_path).name
            shutil.copyfile(endpoint.ca_bundle_path, pack_root / bundle_name)
            copied.append(bundle_name)
        if fixtures is not None:
            write_canonical_json(fixtures, pack_root / FIXTURES_FILE_NAME)
            copied.append(FIXTURES_FILE_NAME)
        shutil.copyfile(drafts.assertions_path, pack_root / ASSERTIONS_FILE_NAME)
        _write_yaml(
            [dict(template) for template in supplement.task_templates],
            pack_root / "task_templates.yaml",
        )
        _write_yaml(
            [dict(case) for case in supplement.validation_cases],
            pack_root / "validation_cases.yaml",
        )
        _write_yaml(
            _manifest_document(
                supplement,
                evidence=evidence,
                has_fixtures=FIXTURES_FILE_NAME in copied,
            ),
            pack_root / "manifest.yaml",
        )
        record = _record_document(
            pack_root,
            evidence=evidence,
            source=source,
            supplement_path=supplement_path.resolve(),
            assertion_names=assertion_names,
            draft_provenance_digest=drafts.provenance_digest,
            probe_plan_path=(
                probe_plan_path.resolve() if probe_plan_path is not None else None
            ),
        )
        write_canonical_json(record, root / RECORD_FILE_NAME)
        root.replace(final_root)
    except Exception:
        staging.cleanup()
        raise
    staging.cleanup()
    return AssembledCandidatePack(
        output_root=final_root,
        pack_root=final_root / PACK_DIRECTORY_NAME,
        record_path=final_root / RECORD_FILE_NAME,
        record=record,
    )


def _record_document(
    pack_root: Path,
    *,
    evidence: SourceEvidenceDocument,
    source: Path,
    supplement_path: Path,
    assertion_names: Sequence[str] | frozenset[str],
    draft_provenance_digest: str,
    probe_plan_path: Path | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": RECORD_VERSION,
        "adapter_kind": evidence.source_adapter.kind,
        "pack": evidence.pack.model_dump(mode="json"),
        "evidence_digest": evidence.bundle_digest,
        "source_identity_digest": sha256_json(
            evidence.identity.model_dump(mode="json")
        ),
        "source_root": str(source),
        "draft_provenance_digest": draft_provenance_digest,
        "supplement_digest": _digest_file(supplement_path),
        "probe_plan_digest": (
            _digest_file(probe_plan_path) if probe_plan_path is not None else None
        ),
        "compiled_assertions": sorted(assertion_names),
        "pack_files": {
            path.relative_to(pack_root).as_posix(): _digest_file(path)
            for path in sorted(pack_root.rglob("*"))
            if path.is_file()
        },
    }
    document["record_digest"] = sha256_json(document)
    return document
