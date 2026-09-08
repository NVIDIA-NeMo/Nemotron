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

"""What a verified evaluation source is (schema 1.0).

Config resolution recorded which publication tree the operator
*named*, by content hash. This module defines the object that says the tree is
still that publication, read back from disk — the manifest, both benchmark
tables, the relationship between them, the addressable task set, the oracle pack
an executable run would replay against, and which models read the rows while
they were being built.

Three properties shape these models.

*A handle, not a description.* A runner receives a :class:`VerifiedEvalSource`
instead of paths out of a YAML file. There is no constructor for one that skips
verification, so "the runner read an unpublished parquet" is not a state the
code can reach.

*Identity without location.* :attr:`VerifiedEvalSource.verification_identity`
covers hashes, row counts, task ids, and pack fingerprints, and no absolute
path or timestamp. Moving an intact publication tree to another host must not
change what was verified; changing one byte inside it must.

*Nothing here is a secret.* Endpoint identity arrives through the loader that
keeps environment variable names only, so the models cannot hold a token even
if a pack tried to put one in a config.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.identity import (
    ModelIdentityClaim,
    VerifiedModelExposure,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
    ContentHash,
    FrozenDict,
    NonNegativeInt,
    PositiveInt,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

SOURCE_VERIFICATION_CONTRACT_VERSION: Final = "1.0"
BFCL_TRANSLATION_CONTRACT_VERSION: Final = "bfcl-translation/1.0"

SOURCE_VERIFICATION_REPORT_FILE: Final = "source_verification_report.json"
# A failed verification writes a differently named artifact, so no reader can
# mistake a diagnosis for a pass by looking at the file that is present.
SOURCE_VERIFICATION_FAILURE_FILE: Final = "source_verification_failure.json"

# How far a score taken against this source may claim to reach. ``trace_only``
# compares proposed calls against the gold trace; ``trace_and_executable`` also
# replays them against the oracle the source run was generated with.
ClaimScope = Literal["trace_only", "trace_and_executable"]

# Row fields a translated benchmark must state exactly as its source does: the
# truth a scorer compares a candidate against. Everything outside this tuple is
# surface a translation exists to change — approved message content, the stated
# intent, metadata.language, and the function-description projection.
# A translation that edits one of these is not a translation of this benchmark,
# because a candidate could pass it while failing the source.
TRANSLATION_PRESERVED_FIELDS: Final = (
    "call_order",
    "call_order_prefix",
    "category",
    "difficulty",
    "expected_tool_calls",
    "fixture_refs",
    "gold_eligible",
    "held_out_hit",
    "is_multi_turn",
    "num_tool_calls",
    "pack_id",
    "pack_version",
    "required_tools",
    "required_tools_fingerprint",
    "seed",
    "src",
    "success_assertions",
    "system_prompt_id",
    "task_id",
    "template_id",
    "tier",
    "tools",
    "tools_present",
    "turn_policy",
    "validated_by",
    "variant_index",
    "messages",
    "metadata",
)


def translation_tool_truth(tools: Any) -> Any:
    """Project tools without their one localizable field.

    ``function.description`` is model-facing prose. The function name and the
    complete parameters schema remain executable truth and are compared exactly.
    """
    projected: list[Any] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            projected.append(tool)
            continue
        item = {str(key): value for key, value in tool.items()}
        function = item.get("function")
        if isinstance(function, Mapping):
            item["function"] = {str(key): value for key, value in function.items() if key != "description"}
        projected.append(item)
    return projected


def translation_message_truth(messages: Any) -> Any:
    """Project messages while removing only approved localizable content."""
    projected: list[Any] = []
    for message in messages:
        if not isinstance(message, Mapping):
            projected.append(message)
            continue
        item = {str(key): value for key, value in message.items()}
        role = item.get("role")
        tool_calls = item.get("tool_calls")
        if role in {"system", "user"} or (role == "assistant" and not tool_calls):
            item["content"] = None
        projected.append(item)
    return projected


def translation_metadata_truth(metadata: Any) -> Any:
    """Project row metadata without its target-locale label."""
    if not isinstance(metadata, Mapping):
        return metadata
    return {str(key): value for key, value in metadata.items() if key != "language"}


def translation_preserved_projection(row: CanonicalExportRow) -> dict[str, Any]:
    """Return the exact row truth a localization is forbidden to change."""
    dumped = row.model_dump(mode="json")
    return {
        field: (
            translation_tool_truth(dumped[field])
            if field == "tools"
            else translation_message_truth(dumped[field])
            if field == "messages"
            else translation_metadata_truth(dumped[field])
            if field == "metadata"
            else dumped[field]
        )
        for field in TRANSLATION_PRESERVED_FIELDS
    }


def _sha256_json(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"


class _Verified(BaseModel):
    """Frozen, closed, non-coercing base for every verification model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _frozen_counts(value: Mapping[str, int]) -> FrozenDict:
    return FrozenDict({str(key): int(count) for key, count in sorted(value.items())})


class SourceCheck(_Verified):
    """One verification step that passed, named so a report can be read.

    Only passing checks are ever recorded: a failing check raises, and the run
    never reaches a report. The list exists to show what was actually proven,
    which is the difference between "verified" and "did not crash".
    """

    name: StrictStr
    detail: StrictStr
    status: Literal["passed"] = "passed"

    def as_document(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


class VerifiedBenchmarkArtifact(_Verified):
    """One benchmark table, read back and held to the hash the manifest declares."""

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    file: StrictStr
    path: Path
    content_hash: ContentHash
    rows: PositiveInt
    benchmark_schema_version: StrictStr
    schema_fingerprint: ContentHash

    @field_validator("file")
    @classmethod
    def _plain_file_name(cls, value: StrictStr) -> str:
        if not value or Path(value).name != value:
            raise ValueError("must be a plain file name beside the run manifest")
        return str(value)

    @field_validator("path")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"must be an absolute path, got {value!s}")
        return value

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "content_hash": self.content_hash,
            "rows": self.rows,
            "benchmark_schema_version": self.benchmark_schema_version,
            "schema_fingerprint": self.schema_fingerprint,
        }


class VerifiedPublication(_Verified):
    """The relationship between ``benchmark_raw.parquet`` and ``benchmark.parquet``.

    The values are the ones the manifest declares; the proof that the two tables
    on disk actually stand in this relationship is the publication contract,
    which source verification runs against both files rather than reimplementing
    here.
    """

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    publication_contract_version: StrictStr
    raw_file: StrictStr
    raw_rows: PositiveInt
    raw_content_hash: ContentHash
    published_file: StrictStr
    published_rows: PositiveInt
    published_content_hash: ContentHash
    surface_gate: StrictStr
    ordering: StrictStr
    dedup_balancing_applied: StrictBool
    held_out_evaluated: StrictBool

    @model_validator(mode="after")
    def _selection_not_rewrite(self) -> VerifiedPublication:
        if self.published_rows > self.raw_rows:
            raise ValueError("publication selects from the raw table, so it cannot carry more rows")
        if self.raw_file == self.published_file:
            raise ValueError("the audit table and the published table cannot be the same file")
        if self.raw_content_hash == self.published_content_hash and self.raw_rows != self.published_rows:
            raise ValueError("two tables with different row counts cannot have the same content hash")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "publication_contract_version": self.publication_contract_version,
            "raw": {
                "file": self.raw_file,
                "rows": self.raw_rows,
                "content_hash": self.raw_content_hash,
            },
            "published": {
                "file": self.published_file,
                "rows": self.published_rows,
                "content_hash": self.published_content_hash,
                "surface_gate": self.surface_gate,
                "ordering": self.ordering,
                "dedup_balancing_applied": self.dedup_balancing_applied,
                "held_out_evaluated": self.held_out_evaluated,
            },
        }


class SourceTaskIndex(_Verified):
    """The addressable task set, in publication order.

    Order is preserved rather than sorted because it is meaningful: under
    ``selection_rank`` it is the rank deduplication and balancing fixed, so a consumer that
    evaluates the first N rows must get the same N rows the benchmark's own
    order gives. :attr:`task_ids_hash` therefore hashes the sequence, not a set,
    and is the value contamination analysis and the candidate runner agree on.
    """

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    task_ids: tuple[StrictStr, ...]
    gold_task_ids: tuple[StrictStr, ...]
    category_counts: dict[str, NonNegativeInt]
    difficulty_counts: dict[str, NonNegativeInt]
    turn_policy_counts: dict[str, NonNegativeInt]

    @field_validator("category_counts", "difficulty_counts", "turn_policy_counts")
    @classmethod
    def _ordered_and_frozen(cls, value: Mapping[str, int]) -> FrozenDict:
        return _frozen_counts(value)

    @model_validator(mode="after")
    def _addressable(self) -> SourceTaskIndex:
        if not self.task_ids:
            raise ValueError("a benchmark with no rows asks no questions, so it cannot be evaluated")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task ids must be unique; two rows sharing an id cannot be scored apart")
        gold = set(self.gold_task_ids)
        if unknown := sorted(gold - set(self.task_ids)):
            raise ValueError(f"gold task(s) {unknown[:5]} are not published rows")
        if tuple(task_id for task_id in self.task_ids if task_id in gold) != self.gold_task_ids:
            raise ValueError("gold task ids must keep publication order")
        if sum(self.turn_policy_counts.values()) != len(self.task_ids):
            raise ValueError("every published row declares a turn policy, so the counts must cover all of them")
        # ``category`` and ``difficulty`` are nullable columns, so their counts
        # may cover fewer rows than the benchmark carries, but never more.
        for label, counts in (
            ("category_counts", self.category_counts),
            ("difficulty_counts", self.difficulty_counts),
        ):
            if sum(counts.values()) > len(self.task_ids):
                raise ValueError(f"{label} counts more rows than the benchmark carries")
        return self

    @property
    def task_count(self) -> int:
        return len(self.task_ids)

    @property
    def task_ids_hash(self) -> str:
        return _sha256_json(list(self.task_ids))

    @property
    def gold_task_ids_hash(self) -> str:
        return _sha256_json(list(self.gold_task_ids))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "task_ids_hash": self.task_ids_hash,
            "gold_task_count": len(self.gold_task_ids),
            "gold_task_ids_hash": self.gold_task_ids_hash,
            "category_counts": dict(self.category_counts),
            "difficulty_counts": dict(self.difficulty_counts),
            "turn_policy_counts": dict(self.turn_policy_counts),
        }

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "task_ids": list(self.task_ids)}


class VerifiedEndpointIdentity(_Verified):
    """The oracle an endpoint pack pins, as declared by its config.

    Only the pinned identity is here. Whether the endpoint is reachable, and
    whether it still answers with this identity, is an execution-time question:
    requiring a live endpoint to verify a trace-only source would make an
    offline evaluation impossible for no gain.
    """

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    protocol_version: StrictStr
    oracle_id: StrictStr
    oracle_version: StrictStr
    content_digest: ContentHash
    base_url: StrictStr

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "content_digest": self.content_digest,
            "base_url": self.base_url,
        }


class VerifiedOracleSource(_Verified):
    """The oracle pack an executable evaluation will replay against.

    ``actual_pack_content_hash`` is recomputed from the pack's files with the
    generation fingerprint, so it is evidence rather than a restatement of the
    manifest. The two hashes are equal by construction here: a mismatch raises
    during verification and never reaches this model.
    """

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    kind: Literal["python", "endpoint"]
    pack_id: StrictStr
    pack_version: StrictStr
    expected_pack_content_hash: ContentHash
    actual_pack_content_hash: ContentHash
    pack_root: Path
    pack_manifest_path: Path
    pack_file_count: PositiveInt
    resource_role: Literal["backend", "endpoint_config"]
    resource_path: Path
    resource_content_hash: ContentHash
    interface_probed: StrictBool
    backend_interface: tuple[StrictStr, ...] = ()
    endpoint: VerifiedEndpointIdentity | None = None

    @model_validator(mode="after")
    def _coherent(self) -> VerifiedOracleSource:
        if self.actual_pack_content_hash != self.expected_pack_content_hash:
            raise ValueError("a pack whose fingerprint moved is not a verified oracle source")
        if self.kind == "python":
            if self.resource_role != "backend" or self.endpoint is not None:
                raise ValueError("a python oracle executes a backend module, not an endpoint")
            if self.interface_probed and not self.backend_interface:
                raise ValueError("a probed backend must report the interface it exposes")
        elif self.resource_role != "endpoint_config" or self.endpoint is None:
            raise ValueError("an endpoint oracle is identified by its endpoint config and pinned identity")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "pack_content_hash": self.actual_pack_content_hash,
            "pack_file_count": self.pack_file_count,
            "resource_role": self.resource_role,
            "resource_content_hash": self.resource_content_hash,
            "backend_interface": list(self.backend_interface),
            "endpoint": self.endpoint.semantic_payload() if self.endpoint is not None else None,
        }

    @property
    def verification_identity(self) -> str:
        """Path-free identity of the executable oracle resource that was verified."""
        return _sha256_json(self.semantic_payload())


class VerifiedTranslationSource(_Verified):
    """A translated benchmark that derives from this source run without changing its truth."""

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    manifest_path: Path
    manifest_content_hash: ContentHash
    source_run_id: StrictStr
    language: StrictStr
    benchmark: VerifiedBenchmarkArtifact
    task_ids_hash: ContentHash
    preserved_fields: tuple[StrictStr, ...] = TRANSLATION_PRESERVED_FIELDS
    # A translator read every localized row, so its identity is mandatory
    # contamination evidence under the BFCL translation contract.
    translator: ModelIdentityClaim
    tool_descriptions_localized: StrictBool = False

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "manifest_content_hash": self.manifest_content_hash,
            "source_run_id": self.source_run_id,
            "language": self.language,
            "benchmark": self.benchmark.semantic_payload(),
            "task_ids_hash": self.task_ids_hash,
            "preserved_fields": list(self.preserved_fields),
            "translator": self.translator.semantic_payload(),
            "tool_descriptions_localized": self.tool_descriptions_localized,
        }


class VerifiedEvalSource(_Verified):
    """A publication tree proven to be the one an eval config resolved.

    This is the only handle a runner is given. It carries the expected hashes
    for :func:`...source_verification.assert_source_unchanged`, the task set to
    iterate, and the oracle to replay against — everything needed to run, and
    nothing that would let a runner reach an unverified file.
    """

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    eval_config_hash: ContentHash
    source_run_id: StrictStr
    generation_config_hash: StrictStr
    resolved_config_hash: StrictStr
    lineage_policy: StrictStr
    gold_eligible: StrictBool
    publication_dir: Path
    source_manifest_path: Path
    source_manifest_hash: ContentHash
    benchmark: VerifiedBenchmarkArtifact
    raw_benchmark: VerifiedBenchmarkArtifact
    publication: VerifiedPublication
    task_index: SourceTaskIndex
    oracle: VerifiedOracleSource | None = None
    translation: VerifiedTranslationSource | None = None
    # Which models read these rows while the benchmark was being built. Carried
    # here rather than re-read later so the contamination gate reasons about a
    # typed, verified inventory instead of parsing the manifest a second time.
    exposures: tuple[VerifiedModelExposure, ...] = ()
    modes: tuple[StrictStr, ...]
    claim_scope: ClaimScope
    checks: tuple[SourceCheck, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> VerifiedEvalSource:
        if not self.modes:
            raise ValueError("a verified source records the modes it was verified for")
        executable = "executable" in self.modes or "held_out_eval" in self.modes
        if self.claim_scope != ("trace_and_executable" if executable else "trace_only"):
            raise ValueError("claim scope must follow the modes the source was verified for")
        if executable and self.oracle is None:
            raise ValueError("an executable claim requires a verified oracle source")
        if self.benchmark.file != self.publication.published_file:
            raise ValueError("the projected table is not the table the publication declares")
        if self.benchmark.content_hash != self.publication.published_content_hash:
            raise ValueError("the projected table's bytes are not the published bytes")
        if self.raw_benchmark.content_hash != self.publication.raw_content_hash:
            raise ValueError("the audit table's bytes are not the raw bytes the publication declares")
        if self.benchmark.rows != self.publication.published_rows:
            raise ValueError("the projected row count is not the published row count")
        if self.raw_benchmark.rows != self.publication.raw_rows:
            raise ValueError("the audit table's row count is not the raw row count")
        if self.task_index.task_count != self.benchmark.rows:
            raise ValueError("the task index does not cover the published rows exactly")
        if self.translation is not None and self.translation.source_run_id != self.source_run_id:
            raise ValueError("a translation of another run is not a translation of this source")
        published = set(self.task_index.task_ids)
        seen: set[tuple[str, str]] = set()
        for exposure in self.exposures:
            if (exposure.role, exposure.scope) in seen:
                raise ValueError(f"role {exposure.role} declares the scope {exposure.scope} twice")
            seen.add((exposure.role, exposure.scope))
            if unknown := sorted(set(exposure.task_ids) - published):
                raise ValueError(f"exposure {exposure.role} covers unpublished row(s) {unknown[:3]}")
            if tuple(task_id for task_id in self.task_index.task_ids if task_id in set(exposure.task_ids)) != (
                exposure.task_ids
            ):
                raise ValueError(f"exposure {exposure.role} must list its rows in publication order")
        return self

    @property
    def evaluation_benchmark(self) -> VerifiedBenchmarkArtifact:
        """The table a runner reads: the translation when one is configured."""
        return self.benchmark if self.translation is None else self.translation.benchmark

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.task_index.task_ids

    @property
    def executable(self) -> bool:
        return self.claim_scope == "trace_and_executable"

    def semantic_payload(self) -> dict[str, Any]:
        """What was verified, with nothing that depends on where the files live."""
        return {
            "schema_version": self.schema_version,
            "eval_config_hash": self.eval_config_hash,
            "source_run_id": self.source_run_id,
            "source_manifest_hash": self.source_manifest_hash,
            "generation_config_hash": self.generation_config_hash,
            "resolved_config_hash": self.resolved_config_hash,
            "lineage_policy": self.lineage_policy,
            "gold_eligible": self.gold_eligible,
            "benchmark": self.benchmark.semantic_payload(),
            "raw_benchmark": self.raw_benchmark.semantic_payload(),
            "publication": self.publication.semantic_payload(),
            "task_index": self.task_index.semantic_payload(),
            "oracle": self.oracle.semantic_payload() if self.oracle is not None else None,
            "translation": self.translation.semantic_payload() if self.translation is not None else None,
            "exposures": [exposure.semantic_payload() for exposure in self.exposures],
            "modes": list(self.modes),
            "claim_scope": self.claim_scope,
        }

    @property
    def verification_identity(self) -> str:
        """One hash for "this exact source, verified for these modes"."""
        return _sha256_json(self.semantic_payload())

    def resolved_paths(self) -> dict[str, Any]:
        """Where the verified bytes were read from, outside the hashed payload."""
        return {
            "publication_dir": str(self.publication_dir),
            "source_run_manifest": str(self.source_manifest_path),
            "benchmark": str(self.benchmark.path),
            "raw_benchmark": str(self.raw_benchmark.path),
            "oracle_pack_manifest": (str(self.oracle.pack_manifest_path) if self.oracle is not None else None),
            "oracle_resource": (str(self.oracle.resource_path) if self.oracle is not None else None),
            "translation_manifest": (str(self.translation.manifest_path) if self.translation is not None else None),
            "translation_benchmark": (str(self.translation.benchmark.path) if self.translation is not None else None),
        }


class SourceVerificationReport(_Verified):
    """The auditable artifact a passing verification writes.

    ``verified_at`` is recorded here and nowhere else: when a source was checked
    is worth knowing, and it is not part of what was checked, so it must not
    reach :attr:`VerifiedEvalSource.verification_identity`.
    """

    schema_version: Literal["1.0"] = SOURCE_VERIFICATION_CONTRACT_VERSION
    status: Literal["passed"] = "passed"
    verified_at: StrictStr
    source: VerifiedEvalSource

    @property
    def verification_identity(self) -> str:
        return self.source.verification_identity

    def as_document(self) -> dict[str, Any]:
        source = self.source
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "verified_at": self.verified_at,
            "verification_identity": self.verification_identity,
            "eval_config_hash": source.eval_config_hash,
            "source_run_id": source.source_run_id,
            "source_manifest_hash": source.source_manifest_hash,
            "generation_config_hash": source.generation_config_hash,
            "resolved_config_hash": source.resolved_config_hash,
            "lineage_policy": source.lineage_policy,
            "gold_eligible": source.gold_eligible,
            "benchmark": source.benchmark.semantic_payload(),
            "raw_benchmark": source.raw_benchmark.semantic_payload(),
            "publication": {**source.publication.semantic_payload(), "verified": True},
            "task_index": source.task_index.as_document(),
            "oracle": (
                {
                    "required": source.executable,
                    **source.oracle.semantic_payload(),
                    "expected_pack_content_hash": source.oracle.expected_pack_content_hash,
                    "interface_probed": source.oracle.interface_probed,
                }
                if source.oracle is not None
                else {"required": False}
            ),
            "translation": (source.translation.semantic_payload() if source.translation is not None else None),
            "exposures": [exposure.as_document() for exposure in source.exposures],
            "modes": list(source.modes),
            "claim_scope": source.claim_scope,
            "checks": [check.as_document() for check in source.checks],
            "resolved_paths": source.resolved_paths(),
        }
