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

"""Prove that an eval config names the committed benchmark publication.

Config resolution recorded what the operator *named*. This module reads it back
from disk and holds it to that record before a candidate token is paid for:

1. ``run_manifest.json`` is the publication commit marker, and its bytes still hash to
   what the config resolved. Without a manifest, a parquet beside it is an
   unpublished artifact, not a benchmark.
2. Both benchmark tables hash to what the publication declares — in three
   independent places (the publication section, the artifact section, and the
   resolved config), which must all agree.
3. The two tables satisfy the publication contract: publication *selects*
   raw rows, it does not rewrite them, and no held-out row ships.
4. The published parquet decodes under the benchmark schema this build reads,
   into a unique, addressable task set.
5. For ``executable`` mode, the oracle pack still fingerprints to what generation
   certified, and the resource that will be executed is the one the pack's own
   manifest selects.
6. Every model that read a published row while it was being built is named, with
   the rows it read. This is the inventory the contamination gate uses, and it
   is collected here because "which models shaped this benchmark" is a fact
   about the source, provable from the same manifest and the same rows.

Nothing here is a second implementation of those contracts. The publication
semantics come from :mod:`...publication_contract`, the row decode from
:mod:`...export_projection`, the pack file set and fingerprint from
:mod:`...pack_loader`, and the endpoint identity from :mod:`...endpoint`. A
verifier that re-derived any of them could disagree with the pipeline that wrote
the artifact, and then the disagreement would be the bug.

Two things are deliberately out of scope. A live endpoint is never contacted:
requiring one would make an offline trace-only evaluation impossible, and the
endpoint's *current* identity is an execution-time question. And no oracle task
is replayed: verification proves the backend imports and exposes its interface,
while replay is the runner's work.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import yaml
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
    BENCHMARK_SCHEMA_VERSIONS,
    OraclePackRef,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import load_endpoint_config
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.identity import (
    ModelIdentityClaim,
    VerifiedModelExposure,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    BfclEvalConfig,
    EvalLimits,
    EvalOracleResource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    BFCL_TRANSLATION_CONTRACT_VERSION,
    SOURCE_VERIFICATION_CONTRACT_VERSION,
    SOURCE_VERIFICATION_FAILURE_FILE,
    SOURCE_VERIFICATION_REPORT_FILE,
    TRANSLATION_PRESERVED_FIELDS,
    SourceCheck,
    SourceTaskIndex,
    SourceVerificationReport,
    VerifiedBenchmarkArtifact,
    VerifiedEndpointIdentity,
    VerifiedEvalSource,
    VerifiedOracleSource,
    VerifiedPublication,
    VerifiedTranslationSource,
    translation_preserved_projection,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_errors import (
    BenchmarkHashMismatchError,
    BenchmarkSchemaMismatchError,
    ModelExposureError,
    OraclePackDriftError,
    OracleResourceMismatchError,
    PublicationSemanticsError,
    SourceChangedDuringEvalError,
    SourceManifestDriftError,
    SourceManifestSchemaError,
    SourceTaskIndexError,
    SourceVerificationError,
    TranslationLineageError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import CanonicalExportRow
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ExportProjectionError,
    project_published_benchmark,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import PackTrustError, ProcessWorker
from nemotron.steps.byob.runtime.benchmark_families.bfcl.origin_provenance import (
    OriginProvenanceError,
    load_origin_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    ResolvedPackPaths,
    declared_pack_inputs,
    load_held_out_policy,
    oracle_runtime_fixtures,
    pack_file_hashes,
    pack_files,
    pack_fingerprint,
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    PUBLICATION_CONTRACT_VERSION,
    PUBLICATION_RESTATED_FIELDS,
    PublicationContractError,
    PublicationPlan,
    verify_written_benchmarks,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import benchmark_schema, canonical_json
from nemotron.steps.byob.runtime.config import AVAILABLE_QUALITY_METRICS
from nemotron.steps.byob.runtime.translation.locales import (
    SUPPORTED_SCRIPTS,
    contains_script,
    normalize_locale,
)

RUN_MANIFEST_FILE: Final = "run_manifest.json"

# What every publication manifest carries. A document missing one is not a
# commit marker this evaluator can reason about: each one is either lineage a
# score has to cite, or a declaration verification checks against the bytes.
REQUIRED_MANIFEST_FIELDS: Final = (
    "artifacts",
    "created_at",
    "generation_config_hash",
    "gold_eligible",
    "held_out",
    "lineage_policy",
    "models",
    "oracle",
    "oracle_clock",
    "pack",
    "publication",
    "resolved_config_hash",
    "run_id",
    "schema_version",
    "tier",
)

# The roles a generation run can drive a model in, exactly as publication records
# them. The key set is checked for equality rather than membership: a newer
# pipeline that adds a role would otherwise have its exposure silently dropped
# here, and a dropped exposure reads as "this candidate is clean".
GENERATION_EXPOSURE_ROLES: Final = ("profile", "paraphrase", "surface_judge")

# The symbols a Python oracle backend must expose for the runner to drive it.
BACKEND_INTERFACE: Final = ("call_tool", "get_state", "list_tools", "reset")

# Characters that make an id unsafe as a path component or a log token, plus the
# two reserved directory names. Non-ASCII letters are deliberately allowed: a
# pack may be authored in any language, and its task ids are still safe.
_UNSAFE_TASK_ID_CHARS: Final = frozenset('/\\:*?"<>|')
_RESERVED_TASK_IDS: Final = frozenset({".", ".."})
_MAX_TASK_ID_LENGTH: Final = 200
_BFCL_PROTECTED_PLACEHOLDER: Final = re.compile(r"__BFCL_PROTECTED_[0-9]{6}__")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_json(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"


def benchmark_schema_fingerprint(schema_version: str) -> str:
    """Fingerprint the Arrow schema an evaluation reads rows through.

    Covers the schema version *and* the column names and types, because a build
    that reads the same declared version with a different column order would
    decode a different benchmark from identical bytes.
    """
    return _sha256_json(
        {
            "benchmark_schema_version": schema_version,
            "fields": [[field.name, str(field.type)] for field in benchmark_schema()],
        }
    )


def _mapping(value: Any, artifact: str, *, recovery: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceManifestSchemaError(
            artifact,
            "is not a JSON object",
            actual=value,
            expected="a JSON object",
            recovery=recovery,
        )
    return {str(key): child for key, child in value.items()}


def _text(value: Any, artifact: str, *, expected: str, recovery: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestSchemaError(
            artifact,
            "is missing or is not a non-empty string",
            actual=value,
            expected=expected,
            recovery=recovery,
        )
    return value.strip()


def _plain_file_name(value: Any, artifact: str) -> str:
    name = _text(
        value,
        artifact,
        expected="a plain file name beside the run manifest",
        recovery="point at an unmodified run_manifest.json; publication artifacts live beside their manifest",
    )
    if Path(name).name != name:
        raise SourceManifestSchemaError(
            artifact,
            "names a path rather than a file beside the manifest",
            actual=name,
            expected="a plain file name",
            recovery="point at an unmodified run_manifest.json; a manifest that names a path could send an "
            "evaluation outside the publication tree",
        )
    return name


def _row_count(value: Any, artifact: str) -> int:
    if type(value) is not int or value < 0:
        raise PublicationSemanticsError(
            artifact,
            "does not declare a row count",
            actual=value,
            expected="a non-negative integer",
            recovery="point at an unmodified run_manifest.json",
        )
    return value


def _content_hash_field(value: Any, artifact: str) -> str:
    hash_value = _text(
        value,
        artifact,
        expected="sha256:<64 hex characters>",
        recovery="point at an unmodified run_manifest.json",
    )
    if not hash_value.startswith("sha256:") or len(hash_value) != 71:
        raise SourceManifestSchemaError(
            artifact,
            "is not a sha256 content hash",
            actual=hash_value,
            expected="sha256:<64 hex characters>",
            recovery="point at an unmodified run_manifest.json",
        )
    return hash_value


def _boolean(value: Any, artifact: str, *, recovery: str) -> bool:
    if type(value) is not bool:
        raise SourceManifestSchemaError(
            artifact,
            "does not record a boolean verdict",
            actual=value,
            expected="true or false",
            recovery=recovery,
        )
    return value


def verify_eval_source(
    config: BfclEvalConfig,
    *,
    probe_oracle: bool = True,
    revocation_check: Callable[[str], object] | None = None,
) -> VerifiedEvalSource:
    """Verify the source an eval config resolved, and return the runner's handle.

    ``probe_oracle`` controls only whether a Python backend is imported in a
    throwaway worker process to confirm its interface. Every hash, schema, and
    lineage check runs regardless, and the resulting handle records whether the
    probe happened, so a caller cannot quietly turn verification off.
    """
    checks: list[SourceCheck] = []
    manifest = _verify_commit_marker(config, checks)
    if revocation_check is not None:
        pack = manifest.get("pack")
        fingerprint = pack.get("content_hash") if isinstance(pack, Mapping) else None
        if not isinstance(fingerprint, str):
            raise SourceVerificationError(
                "source_verification",
                "source manifest lacks a release fingerprint for revocation policy",
                expected="pack.content_hash bound to the frozen release",
                recovery="regenerate the source publication with release lineage",
            )
        verdict = revocation_check(fingerprint)
        checks.append(
            SourceCheck(
                name="release_revocation",
                detail=(
                    "authenticated revocation policy applied"
                    + ("; release_revoked warning accepted" if getattr(verdict, "revoked", False) else "")
                ),
            )
        )
    publication, raw_path, published_path = _verify_published_bytes(config, manifest, checks)
    projection = _verify_publication_semantics(publication, raw_path, published_path, checks)
    index = _build_task_index(projection, checks)
    benchmark, raw_benchmark = _benchmark_artifacts(config, publication, raw_path, published_path, projection)
    oracle = _verify_oracle(config, manifest, checks, probe=probe_oracle)
    translation = _verify_translation(config, manifest, projection, index, checks)
    exposures = _build_exposure_inventory(manifest, projection, index, translation, checks)
    executable = config.settings.executable
    try:
        return VerifiedEvalSource(
            eval_config_hash=config.eval_config_hash,
            source_run_id=str(manifest["run_id"]).strip(),
            generation_config_hash=str(manifest["generation_config_hash"]),
            resolved_config_hash=str(manifest["resolved_config_hash"]),
            lineage_policy=str(manifest["lineage_policy"]),
            gold_eligible=bool(manifest["gold_eligible"]),
            publication_dir=config.source.publication_dir,
            source_manifest_path=config.source.run_manifest.path,
            source_manifest_hash=config.source.run_manifest.content_hash,
            benchmark=benchmark,
            raw_benchmark=raw_benchmark,
            publication=publication,
            task_index=index,
            oracle=oracle,
            translation=translation,
            exposures=exposures,
            modes=config.settings.modes,
            claim_scope="trace_and_executable" if executable else "trace_only",
            checks=tuple(checks),
        )
    except ValidationError as exc:
        raise SourceVerificationError(
            "source_verification",
            f"the verified facts do not form a coherent source: {exc.errors()[0].get('msg', 'invalid')}",
            expected="a source whose manifest, tables, task index, and oracle all describe one publication",
            recovery="regenerate the benchmark; a publication whose own records disagree cannot be scored",
        ) from exc


def _verify_commit_marker(config: BfclEvalConfig, checks: list[SourceCheck]) -> dict[str, Any]:
    """Re-read ``run_manifest.json`` and hold it to what the config resolved."""
    path = config.source.run_manifest.path
    artifact = "source_run_manifest"
    if path.name != RUN_MANIFEST_FILE:
        raise SourceManifestSchemaError(
            artifact,
            f"is not named {RUN_MANIFEST_FILE}",
            actual=path.name,
            expected=f"a file named {RUN_MANIFEST_FILE}",
            recovery=f"point at the published {RUN_MANIFEST_FILE}; a renamed copy is not a commit marker",
        )
    if not path.is_file():
        raise SourceManifestDriftError(
            artifact,
            "no longer exists, so the publication it committed can no longer be verified",
            expected="the manifest that was present when the eval config resolved",
            recovery="restore the publication tree, or resolve the eval config against the run that still exists",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceManifestSchemaError(
            artifact,
            f"could not be read as JSON: {type(exc).__name__}",
            expected="a readable JSON document",
            recovery="point at an unmodified run_manifest.json",
        ) from exc
    manifest = _mapping(
        payload,
        artifact,
        recovery="point at run_manifest.json, not at a list, a scalar, or a stage report",
    )
    if missing := sorted(set(REQUIRED_MANIFEST_FIELDS) - set(manifest)):
        raise SourceManifestSchemaError(
            artifact,
            f"does not carry required publication field(s): {', '.join(missing)}",
            expected=f"every one of {', '.join(REQUIRED_MANIFEST_FIELDS)}",
            recovery="point at a run_manifest.json written by this pipeline; a partial manifest cannot establish "
            "what was published or which oracle certified it",
        )
    # Structure first, then identity: "this file is not a manifest" and "this is a
    # different manifest" call for different fixes, and reporting the second when
    # the first is true sends the reader looking for a drift that did not happen.
    actual_hash = _sha256_file(path)
    if actual_hash != config.source.run_manifest.content_hash:
        raise SourceManifestDriftError(
            artifact,
            "changed after the eval config resolved it",
            actual=actual_hash,
            expected=config.source.run_manifest.content_hash,
            recovery="re-resolve the eval config against the current publication, then verify again; a manifest "
            "that moves mid-run means two different benchmarks would be scored as one",
        )
    schema_version = _text(
        manifest["schema_version"],
        f"{artifact}.schema_version",
        expected=f"one of {', '.join(sorted(BENCHMARK_SCHEMA_VERSIONS))}",
        recovery="evaluate with the pipeline revision that published the run",
    )
    if schema_version not in BENCHMARK_SCHEMA_VERSIONS:
        raise SourceManifestSchemaError(
            f"{artifact}.schema_version",
            "names a benchmark schema this build cannot read",
            actual=schema_version,
            expected=f"one of {', '.join(sorted(BENCHMARK_SCHEMA_VERSIONS))}",
            recovery="evaluate with the pipeline revision that published the run, or regenerate the benchmark",
        )
    if schema_version != config.source.benchmark_schema_version:
        raise SourceManifestDriftError(
            f"{artifact}.schema_version",
            "is not the schema version the eval config resolved",
            actual=schema_version,
            expected=config.source.benchmark_schema_version,
            recovery="re-resolve the eval config against this publication",
        )
    run_id = _text(
        manifest["run_id"],
        f"{artifact}.run_id",
        expected="the run id the eval config resolved",
        recovery="point at an unmodified run_manifest.json",
    )
    if run_id != config.source.run_id:
        raise SourceManifestDriftError(
            f"{artifact}.run_id",
            f"identifies run {run_id!r}, not the run this config resolved ({config.source.run_id!r})",
            expected=config.source.run_id,
            recovery="re-resolve the eval config against the publication being evaluated",
        )
    _boolean(
        manifest["gold_eligible"],
        f"{artifact}.gold_eligible",
        recovery="point at a run_manifest.json written by this pipeline",
    )
    for field in ("generation_config_hash", "resolved_config_hash", "lineage_policy", "tier", "created_at"):
        _text(
            manifest[field],
            f"{artifact}.{field}",
            expected="a non-empty lineage value",
            recovery="point at an unmodified run_manifest.json",
        )
    _verify_oracle_declaration(config.source.oracle, manifest)
    checks.append(
        SourceCheck(
            name="commit_marker",
            detail=f"run_manifest.json for {run_id} matches the hash the eval config resolved",
        )
    )
    return manifest


def _verify_oracle_declaration(oracle: EvalOracleResource | None, manifest: Mapping[str, Any]) -> None:
    """Hold the config's oracle kind to the kind the source run recorded."""
    declared = _mapping(
        manifest["oracle"],
        "source_run_manifest.oracle",
        recovery="point at a run_manifest.json written by this pipeline",
    )
    kind = _text(
        declared.get("kind"),
        "source_run_manifest.oracle.kind",
        expected="'python' or 'endpoint'",
        recovery="point at an unmodified run_manifest.json",
    )
    if kind not in {"python", "endpoint"}:
        raise SourceManifestSchemaError(
            "source_run_manifest.oracle.kind",
            "does not name an oracle execution kind this build supports",
            actual=kind,
            expected="'python' or 'endpoint'",
            recovery="evaluate with the pipeline revision that published the run",
        )
    if oracle is not None and oracle.kind != kind:
        raise OracleResourceMismatchError(
            "source_oracle.kind",
            f"declares a {oracle.kind} oracle, but the source run was generated against a {kind} oracle",
            actual=oracle.kind,
            expected=kind,
            recovery="evaluate against the same oracle kind that generated and replay-validated the benchmark",
        )


def _verify_published_bytes(
    config: BfclEvalConfig,
    manifest: Mapping[str, Any],
    checks: list[SourceCheck],
) -> tuple[VerifiedPublication, Path, Path]:
    """Hold both tables' bytes to every hash the publication declares for them."""
    publication = _mapping(
        manifest["publication"],
        "source_run_manifest.publication",
        recovery="point at a run_manifest.json written by this pipeline",
    )
    contract_version = _text(
        publication.get("schema_version"),
        "source_run_manifest.publication.schema_version",
        expected=f"the publication contract version {PUBLICATION_CONTRACT_VERSION}",
        recovery="evaluate with the pipeline revision that published the run",
    )
    if contract_version != PUBLICATION_CONTRACT_VERSION:
        raise PublicationSemanticsError(
            "source_run_manifest.publication.schema_version",
            "declares a publication contract this build does not verify",
            actual=contract_version,
            expected=PUBLICATION_CONTRACT_VERSION,
            recovery="evaluate with the pipeline revision that published the run; the meaning of the two tables "
            "is versioned, and scoring a table under the wrong contract scores the wrong rows",
        )
    if not _boolean(
        publication.get("verified"),
        "source_run_manifest.publication.verified",
        recovery="point at a run_manifest.json written by this pipeline",
    ):
        raise PublicationSemanticsError(
            "source_run_manifest.publication.verified",
            "records that publication did not verify its own tables",
            actual=False,
            expected="true",
            recovery="regenerate the benchmark; an unverified publication is not evaluable output",
        )
    restated = publication.get("restated_fields")
    if list(restated or []) != sorted(PUBLICATION_RESTATED_FIELDS):
        raise PublicationSemanticsError(
            "source_run_manifest.publication.restated_fields",
            "claims the published table may restate fields this contract does not allow",
            actual=restated,
            expected=f"{sorted(PUBLICATION_RESTATED_FIELDS)}",
            recovery="evaluate with the pipeline revision that published the run; a published row that restates "
            "truth cannot be compared against the audit table",
        )

    raw_declared = _mapping(
        publication.get("raw"),
        "source_run_manifest.publication.raw",
        recovery="point at a run_manifest.json written by this pipeline",
    )
    published_declared = _mapping(
        publication.get("published"),
        "source_run_manifest.publication.published",
        recovery="point at a run_manifest.json written by this pipeline",
    )
    artifacts = _mapping(
        manifest["artifacts"],
        "source_run_manifest.artifacts",
        recovery="point at a run_manifest.json written by this pipeline",
    )
    held_out = _mapping(
        manifest["held_out"],
        "source_run_manifest.held_out",
        recovery="point at a run_manifest.json written by this pipeline",
    )

    raw_file = _plain_file_name(raw_declared.get("file"), "source_run_manifest.publication.raw.file")
    published_file = _plain_file_name(published_declared.get("file"), "source_run_manifest.publication.published.file")
    if published_file != config.source.benchmark.path.name:
        raise SourceManifestDriftError(
            "source_run_manifest.publication.published.file",
            f"names {published_file!r}, but the eval config resolved {config.source.benchmark.path.name!r}",
            expected=config.source.benchmark.path.name,
            recovery="re-resolve the eval config against this publication",
        )

    raw_path = _publication_file(config, raw_file, "publication.raw.file")
    published_path = _publication_file(config, published_file, "publication.published.file")
    actual = {raw_file: _sha256_file(raw_path), published_file: _sha256_file(published_path)}
    declarations = {
        raw_file: {
            "publication.raw.content_hash": _content_hash_field(
                raw_declared.get("content_hash"), "source_run_manifest.publication.raw.content_hash"
            ),
            "artifacts.benchmark_raw_parquet.content_hash": _artifact_hash(artifacts, "benchmark_raw_parquet"),
        },
        published_file: {
            "publication.published.content_hash": _content_hash_field(
                published_declared.get("content_hash"),
                "source_run_manifest.publication.published.content_hash",
            ),
            "artifacts.benchmark_parquet.content_hash": _artifact_hash(artifacts, "benchmark_parquet"),
            "eval_config.source.benchmark.content_hash": config.source.benchmark.content_hash,
        },
    }
    for file_name, declared in declarations.items():
        for source_field, declared_hash in declared.items():
            if declared_hash != actual[file_name]:
                raise BenchmarkHashMismatchError(
                    f"{file_name} ({source_field})",
                    "does not match the bytes on disk",
                    actual=actual[file_name],
                    expected=declared_hash,
                    recovery="evaluate the publication the manifest describes; a table that changed after "
                    "publication is a different benchmark, whatever its file name says",
                )

    if _boolean(
        held_out.get("evaluated"),
        "source_run_manifest.held_out.evaluated",
        recovery="point at a run_manifest.json written by this pipeline",
    ) != _boolean(
        published_declared.get("held_out_evaluated"),
        "source_run_manifest.publication.published.held_out_evaluated",
        recovery="point at a run_manifest.json written by this pipeline",
    ):
        raise PublicationSemanticsError(
            "source_run_manifest.held_out.evaluated",
            "disagrees with publication.published.held_out_evaluated about whether a held-out policy ran",
            actual=held_out.get("evaluated"),
            expected="the same verdict in both sections",
            recovery="regenerate the benchmark; a manifest that contradicts itself cannot establish whether "
            "held-out material was kept out of the published table",
        )

    try:
        verified = VerifiedPublication(
            publication_contract_version=contract_version,
            raw_file=raw_file,
            raw_rows=_row_count(raw_declared.get("rows"), "source_run_manifest.publication.raw.rows"),
            raw_content_hash=actual[raw_file],
            published_file=published_file,
            published_rows=_row_count(
                published_declared.get("rows"), "source_run_manifest.publication.published.rows"
            ),
            published_content_hash=actual[published_file],
            surface_gate=_text(
                published_declared.get("surface_gate"),
                "source_run_manifest.publication.published.surface_gate",
                expected="the gate that decided publication",
                recovery="point at an unmodified run_manifest.json",
            ),
            ordering=_text(
                published_declared.get("ordering"),
                "source_run_manifest.publication.published.ordering",
                expected="raw_order or selection_rank",
                recovery="point at an unmodified run_manifest.json",
            ),
            dedup_balancing_applied=_boolean(
                published_declared.get("dedup_balancing_applied"),
                "source_run_manifest.publication.published.dedup_balancing_applied",
                recovery="point at an unmodified run_manifest.json",
            ),
            held_out_evaluated=_boolean(
                published_declared.get("held_out_evaluated"),
                "source_run_manifest.publication.published.held_out_evaluated",
                recovery="point at an unmodified run_manifest.json",
            ),
        )
    except ValidationError as exc:
        raise PublicationSemanticsError(
            "source_run_manifest.publication",
            f"does not describe a valid publication: {exc.errors()[0].get('msg', 'invalid')}",
            expected="a raw table the published table selects from",
            recovery="regenerate the benchmark",
        ) from exc
    checks.append(
        SourceCheck(
            name="published_bytes",
            detail=f"{published_file} and {raw_file} match every hash the manifest declares for them",
        )
    )
    return verified, raw_path, published_path


def _artifact_hash(artifacts: Mapping[str, Any], name: str) -> str:
    entry = _mapping(
        artifacts.get(name),
        f"source_run_manifest.artifacts.{name}",
        recovery="point at a run_manifest.json written by this pipeline; the eval run identifies a table by the "
        "hash the manifest published for it",
    )
    return _content_hash_field(entry.get("content_hash"), f"source_run_manifest.artifacts.{name}.content_hash")


def _publication_file(config: BfclEvalConfig, name: str, field: str) -> Path:
    """Resolve one declared artifact inside the publication tree, and only there."""
    directory = config.source.publication_dir
    path = directory / name
    if path.is_symlink():
        raise BenchmarkHashMismatchError(
            f"{field} ({name})",
            "is a symbolic link, so its bytes are not the publication's bytes",
            expected="a regular file inside the publication tree",
            recovery="evaluate the committed publication tree directly; a link can be re-pointed at another "
            "benchmark without changing anything the manifest records",
        )
    if not path.is_file():
        raise BenchmarkHashMismatchError(
            f"{field} ({name})",
            f"is not present in the publication tree at {directory}",
            expected="the table the manifest declares, beside its manifest",
            recovery="restore the publication tree; a manifest without its tables published nothing",
        )
    if path.resolve() != path:
        raise BenchmarkHashMismatchError(
            f"{field} ({name})",
            "resolves outside the path the manifest names",
            expected="a regular file directly inside the publication tree",
            recovery="evaluate the committed publication tree directly",
        )
    return path


def _verify_publication_semantics(
    publication: VerifiedPublication,
    raw_path: Path,
    published_path: Path,
    checks: list[SourceCheck],
) -> CanonicalExportProjection:
    """Hold both tables to the publication contract, then decode published rows.

    The plan is built from the manifest's own declarations — which gate ran,
    whether selection ranking ordered the rows — and the task ids read off disk.
    The publication contract then proves the part an eval run depends on: every
    published row is a raw row, unchanged, and no held-out row shipped.
    """
    raw_ids = _task_id_column(raw_path, publication.raw_file)
    published_ids = _task_id_column(published_path, publication.published_file)
    if len(raw_ids) != publication.raw_rows or len(published_ids) != publication.published_rows:
        raise PublicationSemanticsError(
            "publication.rows",
            f"declares {publication.raw_rows} raw and {publication.published_rows} published rows, "
            f"but the tables carry {len(raw_ids)} and {len(published_ids)}",
            expected="row counts that match the tables on disk",
            recovery="evaluate the publication the manifest describes",
        )
    try:
        plan = PublicationPlan(
            raw_task_ids=tuple(raw_ids),
            published_task_ids=tuple(published_ids),
            surface_gate=publication.surface_gate,
            dedup_balancing_applied=publication.dedup_balancing_applied,
            held_out_evaluated=publication.held_out_evaluated,
            ordering=publication.ordering,
        )
    except ValidationError as exc:
        raise PublicationSemanticsError(
            "publication.plan",
            f"the two tables cannot form the publication the manifest declares: "
            f"{exc.errors()[0].get('msg', 'invalid')}",
            expected="a published table that selects raw rows in the order the manifest declares",
            recovery="regenerate the benchmark; a publication whose order or membership does not follow its own "
            "declarations cannot be scored reproducibly",
        ) from exc
    try:
        report = verify_written_benchmarks(raw_path=raw_path, publication_path=published_path, plan=plan)
    except PublicationContractError as exc:
        raise PublicationSemanticsError(
            "publication.semantics",
            str(exc),
            expected="publication selects raw rows without rewriting them, and ships no held-out row",
            recovery="regenerate the benchmark; a published row that differs from its audit row makes the "
            "audit table useless for explaining a score",
        ) from exc
    if (
        report.raw_content_hash != publication.raw_content_hash
        or report.publication_content_hash != publication.published_content_hash
    ):  # pragma: no cover - the same bytes were hashed twice, from the same paths
        raise BenchmarkHashMismatchError(
            "publication.semantics",
            "the tables changed while they were being verified",
            actual=report.publication_content_hash,
            expected=publication.published_content_hash,
            recovery="stop writing into the publication tree during an evaluation",
        )

    try:
        projection = project_published_benchmark(
            published_path,
            expected_content_hash=publication.published_content_hash,
        )
    except ExportProjectionError as exc:
        raise BenchmarkSchemaMismatchError(
            f"{publication.published_file} rows",
            str(exc),
            expected="every published row decoding under the benchmark schema this build reads",
            recovery="regenerate the benchmark with this pipeline revision; a row the evaluator cannot decode "
            "cannot be scored, and skipping it would change the task set",
        ) from exc
    checks.append(
        SourceCheck(
            name="publication_semantics",
            detail=(
                f"{publication.published_rows} of {publication.raw_rows} raw rows are published unchanged, "
                f"in {publication.ordering}, and decode under the benchmark schema"
            ),
        )
    )
    return projection


def _task_id_column(path: Path, file_name: str) -> list[str]:
    """Check the file's schema, then read just the task ids.

    The ids are needed before the tables can be held to the publication contract,
    and reading them from a file whose columns are not the benchmark's would
    report a publication failure for what is really a schema failure.
    """
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(path)
    except Exception as exc:  # noqa: BLE001 - any parquet failure is the same verdict here
        raise BenchmarkSchemaMismatchError(
            file_name,
            f"cannot be read as a parquet table: {type(exc).__name__}",
            expected="a parquet file written with the benchmark schema",
            recovery="regenerate the benchmark; a table this build cannot read cannot be scored",
        ) from exc
    if not schema.equals(benchmark_schema()):
        raise BenchmarkSchemaMismatchError(
            file_name,
            "is not written with the benchmark schema this build reads",
            expected="the published benchmark schema, in its declared column order and types",
            recovery="regenerate the benchmark with this pipeline revision; a column that moved or changed type "
            "would be decoded as a different value",
        )
    column = pq.read_table(path, columns=["task_id"]).column("task_id").to_pylist()
    if any(not isinstance(task_id, str) for task_id in column):
        raise BenchmarkSchemaMismatchError(
            f"{file_name}.task_id",
            "carries a row without a string task id",
            expected="a string task id in every row",
            recovery="regenerate the benchmark",
        )
    return column


def _build_task_index(projection: CanonicalExportProjection, checks: list[SourceCheck]) -> SourceTaskIndex:
    """Index the published rows into the task set every later stage addresses."""
    rows = projection.rows
    for row in rows:
        _require_addressable_task_id(row.task_id)
    counts: dict[str, dict[str, int]] = {"category": {}, "difficulty": {}, "turn_policy": {}}
    for row in rows:
        for field, value in (
            ("category", row.category),
            ("difficulty", row.difficulty),
            ("turn_policy", row.turn_policy),
        ):
            if value is None:
                continue
            counts[field][value] = counts[field].get(value, 0) + 1
    try:
        index = SourceTaskIndex(
            task_ids=tuple(row.task_id for row in rows),
            gold_task_ids=tuple(row.task_id for row in rows if row.gold_eligible),
            category_counts=counts["category"],
            difficulty_counts=counts["difficulty"],
            turn_policy_counts=counts["turn_policy"],
        )
    except ValidationError as exc:
        raise SourceTaskIndexError(
            "benchmark.task_index",
            f"the published rows do not form an addressable task set: {exc.errors()[0].get('msg', 'invalid')}",
            expected="unique task ids, in publication order, covering every published row",
            recovery="regenerate the benchmark",
        ) from exc
    checks.append(
        SourceCheck(
            name="task_index",
            detail=(
                f"{index.task_count} task ids ({len(index.gold_task_ids)} gold) indexed in publication order "
                f"as {index.task_ids_hash}"
            ),
        )
    )
    return index


def _require_addressable_task_id(task_id: str) -> None:
    """Refuse an id that cannot safely become a file name or a log token.

    Non-ASCII letters are allowed on purpose: a pack may be authored in any
    language, and rejecting its ids would make this pipeline English-only. What
    is refused is what actually breaks a consumer — path separators, control
    characters, the reserved directory names, and ids that would be read as
    command-line flags or hidden files.
    """
    problem: str | None = None
    if not task_id:
        problem = "is empty"
    elif len(task_id) > _MAX_TASK_ID_LENGTH:
        problem = f"is longer than {_MAX_TASK_ID_LENGTH} characters"
    elif task_id in _RESERVED_TASK_IDS:
        problem = "is a reserved directory name"
    elif task_id != task_id.strip() or any(character.isspace() for character in task_id):
        problem = "carries whitespace"
    elif task_id[0] in "-.":
        problem = "starts with a dash or a dot"
    elif unsafe := sorted(set(task_id) & _UNSAFE_TASK_ID_CHARS):
        problem = f"carries the path character(s) {''.join(unsafe)!r}"
    elif any(unicodedata.category(character).startswith("C") for character in task_id):
        problem = "carries a control or format character"
    if problem is None:
        return
    raise SourceTaskIndexError(
        f"benchmark.task_id {task_id!r}",
        f"{problem}, so per-task artifacts and logs cannot address it",
        expected="a task id usable as a path component and a log token",
        recovery="fix the pack's pack_id or template_id and regenerate; task ids are derived from them",
    )


def _benchmark_artifacts(
    config: BfclEvalConfig,
    publication: VerifiedPublication,
    raw_path: Path,
    published_path: Path,
    projection: CanonicalExportProjection,
) -> tuple[VerifiedBenchmarkArtifact, VerifiedBenchmarkArtifact]:
    fingerprint = benchmark_schema_fingerprint(config.source.benchmark_schema_version)
    return (
        VerifiedBenchmarkArtifact(
            file=publication.published_file,
            path=published_path,
            content_hash=publication.published_content_hash,
            rows=projection.source.rows,
            benchmark_schema_version=config.source.benchmark_schema_version,
            schema_fingerprint=fingerprint,
        ),
        VerifiedBenchmarkArtifact(
            file=publication.raw_file,
            path=raw_path,
            content_hash=publication.raw_content_hash,
            rows=publication.raw_rows,
            benchmark_schema_version=config.source.benchmark_schema_version,
            schema_fingerprint=fingerprint,
        ),
    )


def _pack_file_drift(paths: ResolvedPackPaths, recorded: Any) -> str:
    """Name the pack files whose bytes moved, and separate inputs from the rest.

    The fingerprint is one rolling digest, so on its own it can only say the
    directory changed. Without this an operator is told to restore a revision
    nobody has identified, and the only way to learn that a doc edit — not the
    backend — caused it is to bisect the pack's history.

    Undeclared files are still hashed and still fail the run: nothing stops a
    backend from reading one. Saying which side a file is on reports that fact
    without softening it into a judgement about whether the oracle really moved.
    """
    current = pack_file_hashes(paths)
    if not isinstance(recorded, Mapping) or not recorded:
        return (
            f" (generation recorded no per-file hashes, so the drifted file cannot be named: "
            f"the pack hashes {len(current)} files today)"
        )

    declared = declared_pack_inputs(paths)

    def named(names: Iterable[str]) -> str:
        return ", ".join(
            name if name in declared else f"{name} [not a declared oracle input]" for name in sorted(names)
        )

    changed = [name for name, digest in current.items() if name in recorded and recorded[name] != digest]
    added = [name for name in current if name not in recorded]
    removed = [name for name in recorded if name not in current]

    parts: list[str] = []
    if changed:
        parts.append(f"changed {named(changed)}")
    if added:
        parts.append(f"added {named(added)}")
    if removed:
        parts.append(f"removed {named(removed)}")
    if not parts:
        # The per-file map agrees while the aggregate does not, so the drift is
        # in the hashed set itself rather than in any file's bytes.
        return " (every recorded file still matches, so the hashed set itself changed)"
    if not any(name in declared for name in (*changed, *added, *removed)):
        parts.append("every declared oracle input is unchanged")
    return " (" + "; ".join(parts) + ")"


def _verify_oracle(
    config: BfclEvalConfig,
    manifest: Mapping[str, Any],
    checks: list[SourceCheck],
    *,
    probe: bool,
) -> VerifiedOracleSource | None:
    """Prove the oracle pack still is the one that certified the source run.

    A trace-only config that names no oracle verifies nothing here; the eval
    config contract already refuses ``executable`` without one.
    """
    resource = config.source.oracle
    if resource is None:
        return None
    pack = _mapping(
        manifest["pack"],
        "source_run_manifest.pack",
        recovery="point at a run_manifest.json written by this pipeline",
    )
    pack_id = _text(
        pack.get("pack_id"),
        "source_run_manifest.pack.pack_id",
        expected="the source oracle pack id",
        recovery="point at an unmodified run_manifest.json",
    )
    pack_version = _text(
        pack.get("version"),
        "source_run_manifest.pack.version",
        expected="the source oracle pack version",
        recovery="point at an unmodified run_manifest.json",
    )
    expected_fingerprint = _content_hash_field(pack.get("content_hash"), "source_run_manifest.pack.content_hash")
    if (pack_id, pack_version) != (resource.pack_id, resource.pack_version):
        raise OracleResourceMismatchError(
            "source_oracle",
            f"was resolved for pack {resource.pack_id!r} {resource.pack_version!r}, but the source run "
            f"records {pack_id!r} {pack_version!r}",
            expected=f"{pack_id} {pack_version}",
            recovery="re-resolve the eval config against this publication",
        )
    if expected_fingerprint != resource.expected_pack_content_hash:
        raise SourceManifestDriftError(
            "source_run_manifest.pack.content_hash",
            "changed after the eval config resolved it",
            actual=expected_fingerprint,
            expected=resource.expected_pack_content_hash,
            recovery="re-resolve the eval config against this publication",
        )

    for label, reference in (
        ("source_oracle.pack_manifest", resource.pack_manifest),
        ("source_oracle.resource", resource.execution_resource),
    ):
        if not reference.path.is_file():
            raise OraclePackDriftError(
                label,
                f"no longer exists at {reference.path}",
                expected="the file the eval config resolved",
                recovery="restore the oracle pack revision the source run was generated from",
            )
        current = _sha256_file(reference.path)
        if current != reference.content_hash:
            raise OraclePackDriftError(
                label,
                "changed after the eval config resolved it",
                actual=current,
                expected=reference.content_hash,
                recovery="restore the pack revision that generated the benchmark; an oracle that changed cannot "
                "confirm the gold trace it certified",
            )

    paths = _resolve_pack(resource)
    declared_pack = _mapping(
        yaml.safe_load(paths.manifest_path.read_text(encoding="utf-8")) or {},
        "source_oracle.pack_manifest",
        recovery="point at the source pack's manifest.yaml",
    )
    if declared_pack.get("pack_id") != pack_id or str(declared_pack.get("version")) != pack_version:
        raise OracleResourceMismatchError(
            "source_oracle.pack_manifest",
            "does not identify the pack the source run recorded",
            expected=f"pack_id {pack_id!r} and version {pack_version!r}",
            recovery="use the manifest from the exact pack revision that generated the source benchmark",
        )

    role = "backend" if resource.kind == "python" else "endpoint_config"
    selected = paths.backend_path if resource.kind == "python" else paths.endpoint_config_path
    if selected is None or selected != resource.execution_resource.path:
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            f"is not the {role} the pack manifest selects ({selected})",
            expected=f"the {role} declared by the pack's own manifest.yaml",
            recovery="declare the executed resource in the pack manifest and point source_oracle.resource at it; "
            "a resource chosen only by the eval config could execute code the source run never ran",
        )

    actual_fingerprint = f"sha256:{pack_fingerprint(paths)}"
    if actual_fingerprint != expected_fingerprint:
        raise OraclePackDriftError(
            "source_oracle.pack",
            f"pack {pack_id} {pack_version} no longer fingerprints to what generation certified"
            f"{_pack_file_drift(paths, pack.get('files'))}",
            actual=actual_fingerprint,
            expected=expected_fingerprint,
            recovery="restore the pack revision the benchmark was generated from; every file in the pack tree "
            "counts, because a helper module the backend imports changes what the oracle does",
        )

    endpoint_config = (
        load_endpoint_config(paths.endpoint_config_path, allowed_roots=(paths.pack_root,))
        if paths.endpoint_config_path is not None
        else None
    )
    try:
        observed_provider_origin = load_origin_provenance(
            paths,
            endpoint_config,
            pack_fingerprint=actual_fingerprint,
            pack_id=pack_id,
            pack_version=pack_version,
        )
    except OriginProvenanceError as exc:
        raise OraclePackDriftError(
            "source_oracle.provider_origin",
            str(exc),
            expected="provider lineage consistent with the frozen oracle pack",
            recovery="restore the exact frozen provider pack used for generation",
        ) from exc
    manifest_oracle = _mapping(
        manifest["oracle"],
        "source_run_manifest.oracle",
        recovery="restore the run manifest written by generation",
    )
    origin_field = (
        "mcp"
        if observed_provider_origin is not None and observed_provider_origin.get("provider_kind") == "mcp"
        else "origin"
    )
    if manifest_oracle.get(origin_field) != observed_provider_origin:
        raise OraclePackDriftError(
            "source_oracle.provider_origin",
            "publication provider provenance differs from the frozen pack lineage",
            actual=canonical_json(manifest_oracle.get(origin_field)),
            expected=canonical_json(observed_provider_origin),
            recovery="restore the matching run manifest and frozen provider pack",
        )

    endpoint: VerifiedEndpointIdentity | None = None
    interface: tuple[str, ...] = ()
    probed = False
    if resource.kind == "endpoint":
        endpoint = _verify_endpoint_identity(paths, manifest)
    elif config.settings.executable and not probe:
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            "was not interface-probed, so executable verification cannot prove the backend can be driven",
            expected=f"an isolated probe confirming callables for {', '.join(BACKEND_INTERFACE)}",
            recovery="run source verification with the Python backend probe enabled; skipping the probe is "
            "only valid when no executable claim is requested",
        )
    elif probe:
        interface = _probe_backend_interface(paths, manifest, config.limits)
        probed = True

    try:
        verified = VerifiedOracleSource(
            kind=resource.kind,
            pack_id=pack_id,
            pack_version=pack_version,
            expected_pack_content_hash=expected_fingerprint,
            actual_pack_content_hash=actual_fingerprint,
            pack_root=paths.pack_root,
            pack_manifest_path=paths.manifest_path,
            pack_file_count=len(pack_files(paths)),
            resource_role=role,
            resource_path=resource.execution_resource.path,
            resource_content_hash=resource.execution_resource.content_hash,
            interface_probed=probed,
            backend_interface=interface,
            endpoint=endpoint,
        )
    except ValidationError as exc:
        raise OracleResourceMismatchError(
            "source_oracle",
            f"does not describe an executable oracle: {exc.errors()[0].get('msg', 'invalid')}",
            expected="a pack whose fingerprint matches and whose resource matches its kind",
            recovery="restore the pack revision the benchmark was generated from",
        ) from exc
    checks.append(
        SourceCheck(
            name="oracle_pack",
            detail=(
                f"{resource.kind} oracle for pack {pack_id} {pack_version} fingerprints to "
                f"{actual_fingerprint} across {verified.pack_file_count} files"
                + (f", interface {list(interface)}" if probed else "")
            ),
        )
    )
    return verified


def _resolve_pack(resource: EvalOracleResource) -> ResolvedPackPaths:
    """Resolve the pack through the generation resolver, from its manifest alone.

    No eval-side override is applied. Generation may resolve a backend from a
    config override, but an evaluation cannot tell an override apart from a
    substitution, and the fingerprint does not distinguish two modules that both
    live in the pack tree. Requiring the pack manifest to name what runs is what
    makes "the resource the source run executed" checkable at all.
    """
    pack_root = resource.pack_manifest.path.parent
    try:
        return resolve_declared_pack_paths(OraclePackRef(manifest_path=resource.pack_manifest.path), (pack_root,))
    except (PackTrustError, FileNotFoundError, ValueError, OSError, yaml.YAMLError) as exc:
        raise OracleResourceMismatchError(
            "source_oracle.pack_manifest",
            f"does not resolve into a complete oracle pack: {type(exc).__name__}: {exc}",
            expected="a pack whose manifest resolves every file it declares, inside the pack tree",
            recovery="restore the complete pack revision the benchmark was generated from",
        ) from exc


def _verify_endpoint_identity(
    paths: ResolvedPackPaths,
    manifest: Mapping[str, Any],
) -> VerifiedEndpointIdentity:
    """Check the endpoint config's pinned identity against the source run's.

    The loader owns the transport rules — HTTPS only, no credentials in the URL,
    secrets by environment variable name, a CA bundle inside the trust root — so
    what is left here is lineage: the oracle this config pins must be the oracle
    the source run was validated against.
    """
    config_path = paths.endpoint_config_path
    assert config_path is not None  # resolve_declared_pack_paths guarantees exactly one kind
    try:
        endpoint = load_endpoint_config(config_path, allowed_roots=(paths.pack_root,))
    except (PackTrustError, FileNotFoundError, ValueError, OSError, yaml.YAMLError) as exc:
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            f"is not a usable BFCL Oracle HTTP v1 endpoint config: {type(exc).__name__}: {exc}",
            expected="an endpoint config pinning protocol version, oracle identity, and content digest",
            recovery="restore the endpoint config the source run was validated against",
        ) from exc
    recorded = _mapping(manifest["oracle"], "source_run_manifest.oracle", recovery="regenerate the benchmark").get(
        "endpoint_metadata"
    )
    if not isinstance(recorded, Mapping):
        raise OracleResourceMismatchError(
            "source_run_manifest.oracle.endpoint_metadata",
            "records no oracle identity for an endpoint-generated run",
            actual=recorded,
            expected="the oracle identity the generation run validated against",
            recovery="regenerate the benchmark; an endpoint run that recorded no oracle identity cannot be "
            "replayed against a known oracle",
        )
    expected = endpoint.expected.as_dict()
    if dict(recorded) != expected:
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            "pins an oracle identity the source run was not generated against",
            expected=f"oracle_id {recorded.get('oracle_id')!r} version {recorded.get('oracle_version')!r} "
            f"digest {recorded.get('content_digest')}",
            actual=expected.get("content_digest"),
            recovery="point at the endpoint config for the oracle revision that generated the benchmark",
        )
    if paths.endpoint_ca_bundle_path is not None and not paths.endpoint_ca_bundle_path.is_file():
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            "declares a CA bundle that is not present",
            expected="the CA bundle the endpoint config names, inside the pack trust root",
            recovery="restore the pack's CA bundle; without it the runner cannot verify the oracle's certificate",
        )
    return VerifiedEndpointIdentity(
        protocol_version=expected["protocol_version"],
        oracle_id=expected["oracle_id"],
        oracle_version=expected["oracle_version"],
        content_digest=expected["content_digest"],
        base_url=endpoint.base_url,
    )


def _probe_backend_interface(
    paths: ResolvedPackPaths,
    manifest: Mapping[str, Any],
    limits: EvalLimits,
) -> tuple[str, ...]:
    """Import the backend in a throwaway worker and confirm its interface.

    The import runs in the same process worker the runner will use, never in the
    evaluator process: a pack must not be able to change the evaluator's imports
    or environment by being verified. No task is replayed — the probe answers
    "can this backend be driven at all", and every deadline it runs under comes
    from the eval config's own limits.
    """
    backend_path = paths.backend_path
    assert backend_path is not None  # resolve_declared_pack_paths guarantees exactly one kind
    clock = _text(
        manifest.get("oracle_clock"),
        "source_run_manifest.oracle_clock",
        expected="the frozen clock the source run was generated with",
        recovery="point at an unmodified run_manifest.json",
    )
    try:
        datetime.fromisoformat(clock)
    except ValueError as exc:
        raise SourceManifestSchemaError(
            "source_run_manifest.oracle_clock",
            "is not an ISO-8601 timestamp, so the oracle cannot be driven with the source run's clock",
            expected="an ISO-8601 timestamp with an offset",
            recovery="point at an unmodified run_manifest.json",
        ) from exc

    worker = ProcessWorker(default_timeout_s=limits.tool_timeout_s, worker="process")
    try:
        fixtures = None
        fixture_source_path = None
        if paths.fixtures_path is not None:
            fixtures = json.loads(paths.fixtures_path.read_text(encoding="utf-8"))
            if paths.held_out_path is not None:
                pack_manifest = (
                    yaml.safe_load(paths.manifest_path.read_text(encoding="utf-8")) or {}
                )
                templates = (
                    yaml.safe_load(paths.templates_path.read_text(encoding="utf-8")) or []
                )
                held_out = load_held_out_policy(
                    paths.held_out_path,
                    source=str(pack_manifest.get("held_out")),
                    manifest=pack_manifest,
                    fixtures=fixtures,
                    templates=templates,
                )
                fixtures = oracle_runtime_fixtures(
                    manifest=pack_manifest,
                    fixtures=fixtures,
                    held_out=held_out,
                )
                if (
                    (held_out.get("policy") or {}).get(
                        "fixtures_in_backend_state", True
                    )
                    is False
                ):
                    fixture_source_path = paths.fixtures_path
        outputs = worker.run_episode(
            backend_path=backend_path,
            fixtures=fixtures,
            clock_iso=clock,
            seed=0,
            task_id="__source_verification__",
            steps=[{"op": "inspect_backend"}],
            import_root=paths.pack_root,
            import_timeout_s=limits.tool_timeout_s,
            tool_timeout_s=limits.tool_timeout_s,
            episode_timeout_s=limits.episode_timeout_s,
            fixture_source_path=fixture_source_path,
        )
    except (RuntimeError, TimeoutError, OSError) as exc:
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            f"cannot be imported in an isolated worker: {type(exc).__name__}",
            expected="a backend module that imports cleanly inside the pack tree",
            recovery="restore the complete pack revision, including any helper module the backend imports; "
            "an oracle that cannot start cannot confirm an executable claim",
        ) from exc
    inspection = outputs[0] if outputs else {}
    if not isinstance(inspection, Mapping):
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            "did not report its interface when inspected",
            expected=f"callables for {', '.join(BACKEND_INTERFACE)}",
            recovery="restore the backend the source run was generated with",
        )
    if missing := sorted(name for name in BACKEND_INTERFACE if not inspection.get(name)):
        raise OracleResourceMismatchError(
            "source_oracle.resource",
            f"does not expose the backend interface: missing {', '.join(missing)}",
            expected=f"callables for {', '.join(BACKEND_INTERFACE)}",
            recovery="point at the backend module the source run executed; a module that cannot be driven "
            "cannot replay the gold trace",
        )
    return tuple(sorted(name for name, present in inspection.items() if present))


def _verify_translation(
    config: BfclEvalConfig,
    manifest: Mapping[str, Any],
    projection: CanonicalExportProjection,
    index: SourceTaskIndex,
    checks: list[SourceCheck],
) -> VerifiedTranslationSource | None:
    """Verify a translated benchmark derives from this source without changing its truth."""
    reference = config.source.translation_manifest
    if reference is None:
        return None
    artifact = "translation_manifest"
    if not reference.path.is_file():
        raise TranslationLineageError(
            artifact,
            f"no longer exists at {reference.path}",
            expected="the translation manifest the eval config resolved",
            recovery="restore the translation output, or evaluate the source benchmark directly",
        )
    current = _sha256_file(reference.path)
    if current != reference.content_hash:
        raise TranslationLineageError(
            artifact,
            "changed after the eval config resolved it",
            actual=current,
            expected=reference.content_hash,
            recovery="re-resolve the eval config against the current translation",
        )
    try:
        payload = json.loads(reference.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TranslationLineageError(
            artifact,
            f"could not be read as JSON: {type(exc).__name__}",
            expected="a readable JSON document",
            recovery="point at the translation manifest the translation run wrote",
        ) from exc
    if not isinstance(payload, Mapping):
        raise TranslationLineageError(
            artifact,
            "is not a JSON object",
            expected="a JSON object",
            recovery="point at the translation manifest the translation run wrote",
        )
    document = {str(key): value for key, value in payload.items()}

    run_id = str(manifest["run_id"]).strip()
    declared_run = document.get("source_run_id")
    declared_manifest_hash = document.get("source_run_manifest_content_hash")
    run_reference_present = "source_run_id" in document
    hash_reference_present = "source_run_manifest_content_hash" in document
    matches_run = isinstance(declared_run, str) and declared_run.strip() == run_id
    matches_hash = (
        isinstance(declared_manifest_hash, str)
        and declared_manifest_hash.strip() == config.source.run_manifest.content_hash
    )
    # Either reference is enough to establish lineage, but a reference that is
    # *present and wrong* is refused even when the other one matches: a manifest
    # naming one run while carrying another run's hash cannot be relied on to say
    # which benchmark it translated.
    if (
        not (matches_run or matches_hash)
        or (run_reference_present and not matches_run)
        or (hash_reference_present and not matches_hash)
    ):
        raise TranslationLineageError(
            artifact,
            "does not derive from the source run this config evaluates",
            expected=f"source_run_id {run_id!r}, or the source manifest's content hash",
            recovery="use the translation produced from this source run; scoring a translation against another "
            "run's gold trace compares two different benchmarks",
        )

    schema_version = document.get("schema_version")
    if schema_version != SOURCE_VERIFICATION_CONTRACT_VERSION:
        raise TranslationLineageError(
            f"{artifact}.schema_version",
            "declares a translation contract this build does not verify",
            actual=schema_version,
            expected=SOURCE_VERIFICATION_CONTRACT_VERSION,
            recovery="regenerate the translation with this pipeline revision, or evaluate it with the revision "
            "that owns its declared contract",
        )

    translation_contract = document.get("translation_contract")
    if translation_contract != BFCL_TRANSLATION_CONTRACT_VERSION:
        raise TranslationLineageError(
            f"{artifact}.translation_contract",
            "does not declare the required BFCL translation contract",
            actual=translation_contract,
            expected=BFCL_TRANSLATION_CONTRACT_VERSION,
            recovery="regenerate the localization with BFCLTranslationAdapter",
        )
    if translation_contract == BFCL_TRANSLATION_CONTRACT_VERSION:
        required_content_addressed = {
            "translation_id",
            "model",
            "contamination",
            "field_policy",
            "protected_tokens",
            "source_benchmark_content_hash",
            "source_oracle_pack",
            "artifacts",
            "quality",
            "source_language",
            "localization_validation",
        }
        if missing := sorted(required_content_addressed - set(document)):
            raise TranslationLineageError(
                artifact,
                f"omits content-addressed BFCL translation field(s): {', '.join(missing)}",
                expected="complete model, contamination, field, source, and artifact provenance",
                recovery="regenerate the localization with BFCLTranslationAdapter",
            )
        body = {key: value for key, value in document.items() if key != "translation_id"}
        expected_translation_id = _sha256_json(body)
        if document["translation_id"] != expected_translation_id:
            raise TranslationLineageError(
                f"{artifact}.translation_id",
                "does not identify the translation manifest body",
                actual=document["translation_id"],
                expected=expected_translation_id,
                recovery="restore the unmodified content-addressed translation manifest",
            )
        model = document["model"]
        if not isinstance(model, Mapping):
            raise TranslationLineageError(
                f"{artifact}.model",
                "does not declare complete translator provenance",
                expected="provider, model, canonical_id, source, immutable weights, and config_hash",
                recovery="regenerate the localization with a resolved translator identity",
            )
        for field in ("provider", "model", "canonical_id", "source"):
            _translation_text(model.get(field), f"{artifact}.model.{field}")
        config_hash = model.get("config_hash")
        revision = model.get("revision")
        weights_digest = model.get("weights_digest")
        if (
            not isinstance(config_hash, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", config_hash)
            or (
                not (isinstance(weights_digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", weights_digest))
                and not (isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision))
            )
        ):
            raise TranslationLineageError(
                f"{artifact}.model",
                "does not pin the translator config and immutable weights",
                expected="config_hash plus weights_digest or a 40-64 hex revision",
                recovery="resolve complete translator provenance and regenerate",
            )
        field_policy = document["field_policy"]
        required_localized = [
            "messages.system.content",
            "messages.user.content",
            "messages.assistant_without_tool_calls.content",
            "intent",
            "metadata.language",
        ]
        allowed_localized_policies = [
            required_localized,
            [*required_localized, "tools.function.description"],
        ]
        localized_fields = field_policy.get("localized_fields") if isinstance(field_policy, Mapping) else None
        if (
            not isinstance(field_policy, Mapping)
            or field_policy.get("flattening_contract") != "bfcl-translation-fields/1.0"
            or field_policy.get("preserved_fields") != list(TRANSLATION_PRESERVED_FIELDS)
            or not isinstance(localized_fields, list)
            or localized_fields not in allowed_localized_policies
            or type(field_policy.get("unit_count")) is not int
            or field_policy["unit_count"] <= 0
            or not isinstance(field_policy.get("field_paths_hash"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", field_policy["field_paths_hash"])
        ):
            raise TranslationLineageError(
                f"{artifact}.field_policy",
                "does not declare the closed stable-path localization policy",
                expected="the BFCL field policy with preserved fields, paths hash, and unit count",
                recovery="regenerate the localization with BFCLTranslationAdapter",
            )
        localized_fields = cast(list[str], localized_fields)
        protection = document["protected_tokens"]
        if (
            not isinstance(protection, Mapping)
            or protection.get("contract") != "bfcl-protected-tokens/1.0"
            or type(protection.get("occurrences")) is not int
            or protection["occurrences"] < 0
            or type(protection.get("fields_with_tokens")) is not int
            or not 0 <= protection["fields_with_tokens"] <= field_policy["unit_count"]
        ):
            raise TranslationLineageError(
                f"{artifact}.protected_tokens",
                "does not declare valid ordered-placeholder evidence",
                expected="non-negative token and field counts under the protected-token contract",
                recovery="regenerate the localization with BFCLTranslationAdapter",
            )
        quality = document["quality"]
        quality_metrics = quality.get("metrics") if isinstance(quality, Mapping) else None
        if (
            not isinstance(quality, Mapping)
            or quality.get("backtranslation") is not True
            or quality.get("row_filtering") is not False
            or not isinstance(quality_metrics, list)
            or not quality_metrics
            or any(
                not isinstance(metric, Mapping)
                or metric.get("type") not in AVAILABLE_QUALITY_METRICS
                or not isinstance(metric.get("threshold"), int | float)
                or isinstance(metric.get("threshold"), bool)
                or metric["threshold"] < 0
                for metric in quality_metrics
            )
            or len({metric["type"] for metric in quality_metrics}) != len(quality_metrics)
        ):
            raise TranslationLineageError(
                f"{artifact}.quality",
                "does not declare backtranslation metrics without row filtering",
                expected="backtranslation true, non-empty metrics, and row_filtering false",
                recovery="regenerate the complete BFCL localization",
            )
        localization_validation = document["localization_validation"]
        deterministic_fixes = (
            localization_validation.get("deterministic_fixes")
            if isinstance(localization_validation, Mapping)
            else None
        )
        forbidden_patterns = (
            localization_validation.get("forbidden_patterns") if isinstance(localization_validation, Mapping) else None
        )
        required_script = (
            localization_validation.get("required_script") if isinstance(localization_validation, Mapping) else None
        )
        minimum_changed_fraction = (
            localization_validation.get("minimum_changed_fraction")
            if isinstance(localization_validation, Mapping)
            else None
        )
        changed_fraction = (
            localization_validation.get("changed_fraction") if isinstance(localization_validation, Mapping) else None
        )
        if (
            not isinstance(localization_validation, Mapping)
            or localization_validation.get("contract") != "bfcl-localization-validation/1.0"
            or localization_validation.get("model_role") != "translator"
            or localization_validation.get("executable_replay") != "truth_projection_and_parquet_readback"
            or not isinstance(deterministic_fixes, Mapping)
            or set(deterministic_fixes) != {"line_endings", "trailing_whitespace", "unicode"}
            or deterministic_fixes.get("line_endings") != "lf"
            or deterministic_fixes.get("trailing_whitespace") != "removed"
            or deterministic_fixes.get("unicode") not in {"NFC", "unchanged"}
            or not isinstance(minimum_changed_fraction, int | float)
            or isinstance(minimum_changed_fraction, bool)
            or not 0 < float(minimum_changed_fraction) <= 1
            or not isinstance(changed_fraction, int | float)
            or isinstance(changed_fraction, bool)
            or not 0 <= float(changed_fraction) <= 1
            or not isinstance(forbidden_patterns, list)
            or any(not isinstance(pattern, str) or not pattern for pattern in forbidden_patterns)
            or (required_script is not None and required_script not in SUPPORTED_SCRIPTS)
        ):
            raise TranslationLineageError(
                f"{artifact}.localization_validation",
                "does not declare deterministic localization and language gates",
                expected="the BFCL localization-validation contract",
                recovery="regenerate the localization with BFCLTranslationAdapter",
            )
        try:
            compiled_forbidden_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in forbidden_patterns]
        except re.error as exc:
            raise TranslationLineageError(
                f"{artifact}.localization_validation.forbidden_patterns",
                f"contains invalid regex: {exc}",
                recovery="regenerate the localization with valid response guards",
            ) from exc
        if document["source_benchmark_content_hash"] != projection.source.content_hash:
            raise TranslationLineageError(
                f"{artifact}.source_benchmark_content_hash",
                "does not identify the source benchmark this localization derives from",
                actual=document["source_benchmark_content_hash"],
                expected=projection.source.content_hash,
                recovery="regenerate the localization from this source publication",
            )
        declared_pack = document["source_oracle_pack"]
        source_pack = manifest.get("pack")
        expected_pack = (
            {
                "pack_id": source_pack.get("pack_id"),
                "version": source_pack.get("version"),
                "content_hash": source_pack.get("content_hash"),
            }
            if isinstance(source_pack, Mapping)
            else None
        )
        if not isinstance(declared_pack, Mapping) or canonical_json(declared_pack) != canonical_json(expected_pack):
            raise TranslationLineageError(
                f"{artifact}.source_oracle_pack",
                "does not preserve the source Oracle-pack identity",
                actual=declared_pack,
                expected=canonical_json(expected_pack),
                recovery="regenerate the localization from the immutable source release",
            )
        declared_artifacts = document["artifacts"]
        expected_artifacts = {
            "translation_units": "stage_cache/translation_units.parquet",
            "backtranslation_units": "stage_cache/backtranslation_units.parquet",
            "quality_metrics": "stage_cache/quality_metrics.parquet",
        }
        if not isinstance(declared_artifacts, Mapping) or set(declared_artifacts) != set(expected_artifacts):
            raise TranslationLineageError(
                f"{artifact}.artifacts",
                "does not declare translation, backtranslation, and quality evidence",
                expected=f"exactly {sorted(expected_artifacts)}",
                recovery="regenerate the localization with BFCLTranslationAdapter",
            )
        evidence_rows: dict[str, list[dict[str, Any]]] = {}
        for name, entry in declared_artifacts.items():
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"file", "rows", "content_hash"}
                or entry.get("file") != expected_artifacts[name]
                or entry.get("rows") != field_policy["unit_count"]
            ):
                raise TranslationLineageError(
                    f"{artifact}.artifacts.{name}",
                    "does not cover every stable translation field",
                    expected=(f"{expected_artifacts[name]}, {field_policy['unit_count']} rows, and content_hash"),
                    recovery="restore the unmodified translation manifest",
                )
            relative = Path(str(entry.get("file", "")))
            evidence_path = reference.path.parent / relative
            expected_hash = entry.get("content_hash")
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or not evidence_path.is_file()
                or evidence_path.is_symlink()
                or evidence_path.resolve() != reference.path.parent.resolve() / relative
                or not isinstance(expected_hash, str)
                or _sha256_file(evidence_path) != expected_hash
            ):
                raise TranslationLineageError(
                    f"{artifact}.artifacts.{name}",
                    "is missing, unsafe, or does not match its declared hash",
                    expected="an immutable evidence file inside the translation tree",
                    recovery="restore the complete translation output",
                )
            try:
                import pyarrow.parquet as pq  # type: ignore[import-untyped]

                table = pq.read_table(evidence_path)
                artifact_rows = table.num_rows
                evidence_rows[str(name)] = table.to_pylist()
            except Exception as exc:  # noqa: BLE001 - all unreadable evidence fails alike
                raise TranslationLineageError(
                    f"{artifact}.artifacts.{name}",
                    "cannot be read as Parquet evidence",
                    expected=f"{field_policy['unit_count']} readable rows",
                    recovery="restore the complete translation output",
                ) from exc
            if artifact_rows != entry["rows"]:
                raise TranslationLineageError(
                    f"{artifact}.artifacts.{name}.rows",
                    "does not match its Parquet evidence",
                    actual=entry["rows"],
                    expected=str(artifact_rows),
                    recovery="restore the unmodified translation manifest",
                )
        forward_rows = evidence_rows["translation_units"]
        backward_rows = evidence_rows["backtranslation_units"]
        quality_rows = evidence_rows["quality_metrics"]
        translation_columns = {
            "translation_id",
            "field_path",
            "text",
            "source_language_code",
            "target_language_code",
            "translation",
        }
        quality_columns = {
            "translation_id",
            "field_path",
            "source_text",
            "translation",
            "backtranslation",
            "is_quality_metric_passed",
            *(f"score_{metric['type']}" for metric in quality_metrics),
            *(f"score_{metric['type']}_passed" for metric in quality_metrics),
        }
        if (
            any(set(row) != translation_columns for row in forward_rows)
            or any(set(row) != translation_columns for row in backward_rows)
            or any(not quality_columns <= set(row) for row in quality_rows)
        ):
            raise TranslationLineageError(
                f"{artifact}.artifacts",
                "does not carry the stable translation, backtranslation, and quality columns",
                expected="field identities, paths, texts, languages, translations, and metric verdicts",
                recovery="restore evidence written by BFCLTranslationAdapter",
            )
        for row_number, quality_row in enumerate(quality_rows):
            expected_passes: list[bool] = []
            for metric in quality_metrics:
                metric_type = str(metric["type"])
                score = quality_row.get(f"score_{metric_type}")
                passed = quality_row.get(f"score_{metric_type}_passed")
                if (
                    not isinstance(score, int | float)
                    or isinstance(score, bool)
                    or not math.isfinite(float(score))
                    or type(passed).__name__ not in {"bool", "bool_"}
                ):
                    raise TranslationLineageError(
                        f"{artifact}.artifacts.quality_metrics row {row_number}",
                        f"has an invalid {metric_type} score or verdict",
                        recovery="regenerate quality evidence from the configured metrics",
                    )
                threshold = float(metric["threshold"])
                expected_pass = float(score) <= threshold if metric_type == "ter" else float(score) >= threshold
                if bool(passed) != expected_pass:
                    raise TranslationLineageError(
                        f"{artifact}.artifacts.quality_metrics row {row_number}",
                        f"has an inconsistent {metric_type} threshold verdict",
                        actual=passed,
                        expected=str(expected_pass),
                        recovery="regenerate quality evidence from the configured thresholds",
                    )
                expected_passes.append(expected_pass)
            aggregate = quality_row.get("is_quality_metric_passed")
            expected_aggregate = all(expected_passes)
            if type(aggregate).__name__ not in {"bool", "bool_"} or bool(aggregate) != expected_aggregate:
                raise TranslationLineageError(
                    f"{artifact}.artifacts.quality_metrics row {row_number}",
                    "has an inconsistent aggregate quality verdict",
                    actual=aggregate,
                    expected=str(expected_aggregate),
                    recovery="regenerate quality evidence from per-metric verdicts",
                )
        identifiers = [str(row["translation_id"]) for row in forward_rows]
        paths = [str(row["field_path"]) for row in forward_rows]
        expected_paths = _expected_translation_paths(projection.rows, localized_fields)
        if (
            len(set(identifiers)) != len(identifiers)
            or len(set(paths)) != len(paths)
            or paths != expected_paths
            or _sha256_json(paths) != field_policy["field_paths_hash"]
        ):
            raise TranslationLineageError(
                f"{artifact}.artifacts.translation_units",
                "does not carry unique stable field identities in the declared order",
                expected=field_policy["field_paths_hash"],
                recovery="restore the original translation evidence",
            )
        by_name = {name: {str(row["translation_id"]): row for row in rows} for name, rows in evidence_rows.items()}
        if any(set(rows) != set(identifiers) for rows in by_name.values()):
            raise TranslationLineageError(
                f"{artifact}.artifacts",
                "translation, backtranslation, and quality evidence cover different fields",
                expected="the same stable translation IDs in all three artifacts",
                recovery="restore the complete translation output",
            )
        placeholder_occurrences = 0
        fields_with_placeholders = 0
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.translation import (
            protected_translation_field,
        )

        source_rows_by_id = {row.task_id: row for row in projection.rows}
        evidence_task_ids: set[str] = set()
        field_bindings: dict[str, dict[str, str]] = {}
        changed_fields = 0
        localized_texts: list[str] = []
        for identifier, path in zip(identifiers, paths, strict=True):
            forward_row = by_name["translation_units"][identifier]
            backward_row = by_name["backtranslation_units"][identifier]
            quality_row = by_name["quality_metrics"][identifier]
            source_text = forward_row["text"]
            forward_text = forward_row["translation"]
            backward_text = backward_row["translation"]
            if not all(
                isinstance(value, str)
                for value in (
                    path,
                    source_text,
                    forward_text,
                    backward_text,
                )
            ):
                raise TranslationLineageError(
                    f"{artifact}.artifacts field {identifier}",
                    "carries a non-text field path or translation",
                    expected="UTF-8 text under one stable JSON field path",
                    recovery="restore the original translation evidence",
                )
            localized_text = _BFCL_PROTECTED_PLACEHOLDER.sub("", forward_text)
            if (
                "\r" in forward_text
                or any(line != line.rstrip() for line in forward_text.split("\n"))
                or (
                    deterministic_fixes["unicode"] == "NFC"
                    and unicodedata.normalize("NFC", forward_text) != forward_text
                )
                or any(pattern.search(localized_text) for pattern in compiled_forbidden_patterns)
            ):
                raise TranslationLineageError(
                    f"{artifact}.artifacts field {identifier}",
                    "fails its declared deterministic localization or response guards",
                    recovery="regenerate and validate localized text before publication",
                )
            changed_fields += forward_text != source_text
            localized_texts.append(localized_text)
            try:
                evidence_task_id, original_text = _translation_path_value(source_rows_by_id, path)
            except (IndexError, KeyError, TypeError) as exc:
                raise TranslationLineageError(
                    f"{artifact}.artifacts field {identifier}",
                    "does not name an approved field on its source task",
                    actual=path,
                    expected="approved message content, intent, or function.description",
                    recovery="regenerate deterministic field paths from the source benchmark",
                ) from exc
            expected_protected_text = protected_translation_field(
                source_rows_by_id[evidence_task_id],
                original_text,
                path=path,
            )
            if source_text != expected_protected_text:
                raise TranslationLineageError(
                    f"{artifact}.artifacts field {identifier}",
                    "does not apply the exact protected-token policy to its source field",
                    actual=source_text,
                    expected=expected_protected_text,
                    recovery="regenerate placeholders from the immutable source row",
                )
            bindings = _protected_bindings(source_text, original_text)
            if bindings is None:
                raise TranslationLineageError(
                    f"{artifact}.artifacts field {identifier}",
                    "protected source text does not reconstruct the source benchmark field",
                    actual=path,
                    expected="source text with only protected tokens replaced by placeholders",
                    recovery="restore evidence written from the immutable source row",
                )
            evidence_task_ids.add(evidence_task_id)
            field_bindings[identifier] = bindings
            source_placeholders = _BFCL_PROTECTED_PLACEHOLDER.findall(source_text)
            if (
                _BFCL_PROTECTED_PLACEHOLDER.findall(forward_text) != source_placeholders
                or _BFCL_PROTECTED_PLACEHOLDER.findall(backward_row["text"]) != source_placeholders
                or _BFCL_PROTECTED_PLACEHOLDER.findall(backward_text) != source_placeholders
                or backward_row["text"] != forward_text
                or quality_row.get("field_path") != path
                or quality_row.get("source_text") != source_text
                or quality_row.get("translation") != forward_text
                or quality_row.get("backtranslation") != backward_text
                or forward_row["source_language_code"] != document["source_language"]
                or forward_row["target_language_code"] != document["language"]
                or backward_row["source_language_code"] != document["language"]
                or backward_row["target_language_code"] != document["source_language"]
            ):
                raise TranslationLineageError(
                    f"{artifact}.artifacts field {identifier}",
                    "breaks stable path, language, text, or protected-placeholder lineage",
                    expected="one identical field record through translation, backtranslation, and quality",
                    recovery="restore evidence written by BFCLTranslationAdapter",
                )
            expected_identifier = _sha256_json(
                {
                    "contract": field_policy["flattening_contract"],
                    "source_manifest": config.source.run_manifest.content_hash,
                    "path": path,
                    "text": original_text,
                }
            )
            if identifier != expected_identifier:
                raise TranslationLineageError(
                    f"{artifact}.artifacts field {identifier}",
                    "does not derive its stable identity from source, path, and text",
                    expected=expected_identifier,
                    recovery="regenerate the localization from the immutable source",
                )
            placeholder_occurrences += len(source_placeholders)
            fields_with_placeholders += bool(source_placeholders)
        if evidence_task_ids != set(index.task_ids):
            raise TranslationLineageError(
                f"{artifact}.artifacts.translation_units",
                "does not cover every source task",
                actual=sorted(evidence_task_ids),
                expected=index.task_ids_hash,
                recovery="flatten every published row without adding or dropping a task",
            )
        if (
            placeholder_occurrences != protection["occurrences"]
            or fields_with_placeholders != protection["fields_with_tokens"]
        ):
            raise TranslationLineageError(
                f"{artifact}.protected_tokens",
                "does not match the placeholders in translation evidence",
                actual={
                    "occurrences": protection["occurrences"],
                    "fields_with_tokens": protection["fields_with_tokens"],
                },
                expected=(f"{placeholder_occurrences} occurrences in {fields_with_placeholders} fields"),
                recovery="restore the original content-addressed translation output",
            )
        actual_changed_fraction = changed_fields / len(forward_rows)
        if (
            actual_changed_fraction < float(minimum_changed_fraction)
            or not math.isclose(
                actual_changed_fraction,
                float(changed_fraction),
                rel_tol=0,
                abs_tol=1e-12,
            )
            or (required_script is not None and not contains_script("\n".join(localized_texts), required_script))
        ):
            raise TranslationLineageError(
                f"{artifact}.localization_validation",
                "does not match the localized evidence language gates",
                actual={
                    "changed_fraction": actual_changed_fraction,
                    "required_script": required_script,
                },
                expected=localization_validation,
                recovery="regenerate the localization and its validation evidence",
            )

    for field in ("language", "benchmark", "task_ids_hash"):
        if field not in document:
            raise TranslationLineageError(
                f"{artifact}.{field}",
                "is absent, so the translation does not declare what it produced",
                expected="language, benchmark (file, rows, content_hash), and task_ids_hash",
                recovery="regenerate the translation with a pipeline that writes the source translation contract; "
                "an evaluation must never silently fall back to the source benchmark",
            )
    language = _translation_text(document["language"], f"{artifact}.language")
    declared_benchmark = document["benchmark"]
    if not isinstance(declared_benchmark, Mapping):
        raise TranslationLineageError(
            f"{artifact}.benchmark",
            "is not a mapping",
            expected="a mapping with file, rows, and content_hash",
            recovery="regenerate the translation with a pipeline that writes the source translation contract",
        )
    missing_benchmark_fields = sorted({"file", "rows", "content_hash"} - set(declared_benchmark))
    if missing_benchmark_fields:
        raise TranslationLineageError(
            f"{artifact}.benchmark",
            f"is missing required field(s): {', '.join(missing_benchmark_fields)}",
            expected="file, rows, and content_hash",
            recovery="regenerate the translation with a pipeline that writes the source translation contract",
        )
    declared_rows = declared_benchmark.get("rows")
    if type(declared_rows) is not int or declared_rows < 0:
        raise TranslationLineageError(
            f"{artifact}.benchmark.rows",
            "does not declare a valid translated row count",
            actual=declared_rows,
            expected="a non-negative integer",
            recovery="regenerate the translation manifest from the translated benchmark",
        )
    file_name = _translation_text(declared_benchmark.get("file"), f"{artifact}.benchmark.file")
    if Path(file_name).name != file_name:
        raise TranslationLineageError(
            f"{artifact}.benchmark.file",
            "names a path rather than a file beside the translation manifest",
            expected="a plain file name",
            recovery="write the translated table beside its manifest",
        )
    declared_hash = _translation_text(declared_benchmark.get("content_hash"), f"{artifact}.benchmark.content_hash")
    benchmark_path = reference.path.parent / file_name
    if not benchmark_path.is_file() or benchmark_path.is_symlink() or benchmark_path.resolve() != benchmark_path:
        raise TranslationLineageError(
            f"{artifact}.benchmark.file",
            f"is not a regular file beside the translation manifest ({benchmark_path})",
            expected="the translated table, beside its manifest",
            recovery="restore the translated benchmark the manifest declares",
        )
    actual_hash = _sha256_file(benchmark_path)
    if actual_hash != declared_hash:
        raise TranslationLineageError(
            f"{artifact}.benchmark.content_hash",
            "does not match the translated table on disk",
            actual=actual_hash,
            expected=declared_hash,
            recovery="regenerate the translation; a table that changed after its manifest was written is not "
            "the benchmark the manifest describes",
        )
    try:
        translated = project_published_benchmark(benchmark_path, expected_content_hash=actual_hash)
    except ExportProjectionError as exc:
        raise TranslationLineageError(
            f"{artifact}.benchmark",
            str(exc),
            expected="a translated table written with the benchmark schema",
            recovery="regenerate the translation; the translated table must carry the same schema as its source",
        ) from exc
    if translated.source.rows != declared_rows:
        raise TranslationLineageError(
            f"{artifact}.benchmark.rows",
            "does not match the translated table on disk",
            actual=declared_rows,
            expected=str(translated.source.rows),
            recovery="regenerate the translation manifest; its row count must describe the table it commits",
        )
    if tuple(row.task_id for row in translated.rows) != index.task_ids:
        raise TranslationLineageError(
            f"{artifact}.benchmark",
            "does not carry the source task ids, in publication order",
            actual=_sha256_json([row.task_id for row in translated.rows]),
            expected=index.task_ids_hash,
            recovery="translate every published row and keep the publication order; a translation that adds, "
            "drops, or reorders rows is a different benchmark",
        )
    source_language = _translation_text(document["source_language"], f"{artifact}.source_language")
    try:
        source_locale = normalize_locale(source_language)
        translated_locale = normalize_locale(language)
        source_languages = {normalize_locale(str(row.metadata.get("language"))).primary for row in projection.rows}
        translated_languages = {normalize_locale(str(row.metadata.get("language"))).primary for row in translated.rows}
    except ValueError as exc:
        raise TranslationLineageError(
            f"{artifact}.language",
            f"contains an invalid locale tag: {exc}",
            recovery="regenerate the localization with valid BCP-47 locale tags",
        ) from exc
    if source_languages != {source_locale.primary}:
        raise TranslationLineageError(
            f"{artifact}.source_language",
            "does not identify every source row's language",
            actual=sorted(source_languages),
            expected=source_language,
            recovery="regenerate the localization with the source language declared correctly",
        )
    if translated_languages != {translated_locale.primary}:
        raise TranslationLineageError(
            f"{artifact}.language",
            "does not identify every localized row's metadata.language",
            actual=sorted(translated_languages),
            expected=language,
            recovery="regenerate the localization without mixed or mislabeled rows",
        )
    translated_rows_by_id = {row.task_id: row for row in translated.rows}
    for identifier, path in zip(identifiers, paths, strict=True):
        try:
            _, localized_text = _translation_path_value(translated_rows_by_id, path)
        except (IndexError, KeyError, TypeError) as exc:
            raise TranslationLineageError(
                f"{artifact}.artifacts field {identifier}",
                "does not name the same approved field in the localized benchmark",
                actual=path,
                expected="the source field path on the row with the same task id",
                recovery="reassemble localization only through its stable field paths",
            ) from exc
        restored = _restore_evidence_text(
            str(by_name["translation_units"][identifier]["translation"]),
            field_bindings[identifier],
        )
        if restored != localized_text:
            raise TranslationLineageError(
                f"{artifact}.artifacts field {identifier}",
                "forward translation does not reconstruct the localized benchmark field",
                actual=path,
                expected="the exact localized value committed in benchmark Parquet",
                recovery="restore the content-addressed benchmark and translation evidence together",
            )

    declared_index_hash = _translation_text(document["task_ids_hash"], f"{artifact}.task_ids_hash")
    if declared_index_hash != index.task_ids_hash:
        raise TranslationLineageError(
            f"{artifact}.task_ids_hash",
            "does not declare the source task set",
            actual=declared_index_hash,
            expected=index.task_ids_hash,
            recovery="regenerate the translation from this source publication",
        )
    if translation_contract == BFCL_TRANSLATION_CONTRACT_VERSION:
        contamination = document["contamination"]
        if (
            not isinstance(contamination, Mapping)
            or contamination.get("role") != "translator"
            or contamination.get("scope") != "all_translated_rows"
            or contamination.get("task_ids_hash") != index.task_ids_hash
            or contamination.get("task_count") != index.task_count
            or contamination.get("model_canonical_id") != document["model"]["canonical_id"]
        ):
            raise TranslationLineageError(
                f"{artifact}.contamination",
                "does not declare translator exposure over the complete source task set",
                expected="translator exposure over all translated rows",
                recovery="regenerate the localization with BFCLTranslationAdapter",
            )
    verified_localized_fields = cast(list[str], localized_fields)
    _require_preserved_truth(
        projection.rows,
        translated.rows,
        allow_tool_descriptions=("tools.function.description" in verified_localized_fields),
    )
    translator = _translator_identity(document, artifact)

    checks.append(
        SourceCheck(
            name="translation_lineage",
            detail=(
                f"{file_name} translates {translated.source.rows} rows of run {run_id} into {language} "
                f"without changing {len(TRANSLATION_PRESERVED_FIELDS)} truth fields"
            ),
        )
    )
    return VerifiedTranslationSource(
        manifest_path=reference.path,
        manifest_content_hash=reference.content_hash,
        source_run_id=run_id,
        language=language,
        benchmark=VerifiedBenchmarkArtifact(
            file=file_name,
            path=benchmark_path,
            content_hash=actual_hash,
            rows=translated.source.rows,
            benchmark_schema_version=config.source.benchmark_schema_version,
            schema_fingerprint=benchmark_schema_fingerprint(config.source.benchmark_schema_version),
        ),
        task_ids_hash=index.task_ids_hash,
        translator=translator,
        tool_descriptions_localized=("tools.function.description" in verified_localized_fields),
    )


def _translator_identity(document: Mapping[str, Any], artifact: str) -> ModelIdentityClaim:
    """Read the mandatory model block naming the translator."""
    if "model" not in document:
        raise TranslationLineageError(
            f"{artifact}.model",
            "does not identify the model that read every translated row",
            expected="complete translator provenance",
            recovery="regenerate the localization with BFCLTranslationAdapter",
        )
    declared = document["model"]
    if not isinstance(declared, Mapping):
        raise TranslationLineageError(
            f"{artifact}.model",
            "is not a mapping, so the translating model cannot be identified",
            expected="a mapping with provider, model, source, revision, weights_digest, or canonical_id",
            recovery="declare the translating model, or omit the block entirely and accept that the translation "
            "cannot be published against a candidate it may share weights with",
        )
    claim = ModelIdentityClaim(
        provider=_optional_text(declared.get("provider")),
        served_model=_optional_text(declared.get("model")),
        weight_source=_optional_text(declared.get("source")),
        weight_model=_optional_text(declared.get("model")),
        revision=_optional_text(declared.get("revision")),
        weights_digest=_optional_text(declared.get("weights_digest")),
        label=_optional_text(declared.get("canonical_id")),
    )
    if not claim.names_a_model:
        raise TranslationLineageError(
            f"{artifact}.model",
            "declares a translating model without naming it",
            expected="at least one of provider, model, or canonical_id",
            recovery="name the model that produced the translation, or remove the block",
        )
    return claim


def _translation_text(value: Any, artifact: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TranslationLineageError(
            artifact,
            "is missing or is not a non-empty string",
            actual=value,
            expected="a non-empty string",
            recovery="regenerate the translation with a pipeline that writes the source translation contract",
        )
    return value.strip()


def _translation_path_value(
    rows: Mapping[str, CanonicalExportRow],
    path: str,
) -> tuple[str, str]:
    prefix, separator, suffix = path.partition("/")
    if prefix != "tasks" or not separator:
        raise KeyError(path)
    escaped_task_id, separator, field_path = suffix.partition("/")
    if not separator:
        raise KeyError(path)
    task_id = escaped_task_id.replace("~1", "/").replace("~0", "~")
    row = rows[task_id]
    parts = field_path.split("/")
    value: Any
    if len(parts) == 1 and parts[0] == "intent":
        value = row.intent
    elif len(parts) == 3 and parts[0] == "messages" and parts[1].isdigit() and parts[2] == "content":
        message = row.messages[int(parts[1])]
        if message.role not in {"system", "user"} and not (message.role == "assistant" and not message.tool_calls):
            raise KeyError(path)
        value = message.content
    elif len(parts) == 4 and parts[0] == "tools" and parts[1].isdigit() and parts[2:] == ["function", "description"]:
        value = row.tools[int(parts[1])]["function"].get("description")
    else:
        raise KeyError(path)
    if not isinstance(value, str) or not value:
        raise KeyError(path)
    return task_id, value


def _expected_translation_paths(
    rows: Sequence[CanonicalExportRow],
    localized_fields: Sequence[str],
) -> list[str]:
    include_tool_descriptions = "tools.function.description" in localized_fields
    paths: list[str] = []
    for row in rows:
        task_id = row.task_id.replace("~", "~0").replace("/", "~1")
        for index, message in enumerate(row.messages):
            if message.content and (
                message.role in {"system", "user"} or (message.role == "assistant" and not message.tool_calls)
            ):
                paths.append(f"tasks/{task_id}/messages/{index}/content")
        if row.intent:
            paths.append(f"tasks/{task_id}/intent")
        if include_tool_descriptions:
            for index, tool in enumerate(row.tools):
                function = tool.get("function")
                description = function.get("description") if isinstance(function, Mapping) else None
                if isinstance(description, str) and description:
                    paths.append(f"tasks/{task_id}/tools/{index}/function/description")
    return paths


def _protected_bindings(protected: str, original: str) -> dict[str, str] | None:
    placeholders = _BFCL_PROTECTED_PLACEHOLDER.findall(protected)
    if len(set(placeholders)) != len(placeholders):
        return None
    cursor = 0
    pattern = ""
    for index, placeholder in enumerate(placeholders):
        position = protected.find(placeholder, cursor)
        pattern += re.escape(protected[cursor:position])
        pattern += f"(?P<token_{index}>.*?)"
        cursor = position + len(placeholder)
    pattern += re.escape(protected[cursor:])
    match = re.fullmatch(pattern, original, flags=re.DOTALL)
    if match is None:
        return None
    return {placeholder: match.group(f"token_{index}") for index, placeholder in enumerate(placeholders)}


def _restore_evidence_text(text: str, bindings: Mapping[str, str]) -> str | None:
    result = text
    for placeholder, value in bindings.items():
        if result.count(placeholder) != 1:
            return None
        result = result.replace(placeholder, value)
    return result if "__BFCL_PROTECTED_" not in result else None


def _require_preserved_truth(
    source_rows: Sequence[CanonicalExportRow],
    translated_rows: Sequence[CanonicalExportRow],
    *,
    allow_tool_descriptions: bool,
) -> None:
    """Compare every field a translation may not change, row by row.

    Comparison is on canonical JSON rather than Python equality, so a re-encoded
    argument or a reordered call is caught even where ``==`` would call the two
    values equal.
    """
    for source, translated in zip(source_rows, translated_rows, strict=True):
        source_view = translation_preserved_projection(source)
        translated_view = translation_preserved_projection(translated)
        if not allow_tool_descriptions:
            source_view["tools"] = source.model_dump(mode="json")["tools"]
            translated_view["tools"] = translated.model_dump(mode="json")["tools"]
        changed = sorted(
            field
            for field in TRANSLATION_PRESERVED_FIELDS
            if canonical_json(source_view[field]) != canonical_json(translated_view[field])
        )
        if changed:
            raise TranslationLineageError(
                f"translation_manifest.benchmark row {source.task_id!r}",
                f"restates {changed}, which a translation may not change",
                expected=(
                    "identical message structure, tool schemas, gold calls, "
                    "assertions, identifiers, lineage, and gating fields"
                ),
                recovery=(
                    "translate only approved message content, intent display text, "
                    "metadata.language, and function.description"
                ),
            )


def _build_exposure_inventory(
    manifest: Mapping[str, Any],
    projection: CanonicalExportProjection,
    index: SourceTaskIndex,
    translation: VerifiedTranslationSource | None,
    checks: list[SourceCheck],
) -> tuple[VerifiedModelExposure, ...]:
    """Record which models read these rows, and which rows each one read.

    This is contamination analysis input, collected here because it is a fact about the source:
    the contamination gate decides what an exposure means, but it must not have
    to re-parse a manifest to find out that one happened.

    Scope is taken from the rows wherever the benchmark schema records it — a
    profile that shaped no published surface and a paraphraser that wrote three
    of fifty rows are both narrower than "the whole benchmark", and excluding
    fifty rows for three rows' worth of exposure would throw away a benchmark
    for no gain. Where the schema records nothing, the scope is every published
    row, because a judge read the whole surface it gated.
    """
    models = _mapping(
        manifest["models"],
        "source_run_manifest.models",
        recovery="point at a run_manifest.json written by this pipeline; a publication that does not say which "
        "models shaped it cannot clear any candidate",
    )
    if set(models) != set(GENERATION_EXPOSURE_ROLES):
        raise ModelExposureError(
            "source_run_manifest.models",
            f"declares roles {sorted(models)} rather than the ones this build knows",
            expected=f"exactly {', '.join(GENERATION_EXPOSURE_ROLES)}",
            recovery="evaluate with the pipeline revision that published the run; a role this build does not "
            "read would be treated as a model that never saw the benchmark",
        )

    rows = projection.rows
    exposures: list[VerifiedModelExposure] = []
    paraphrased_rows: set[str] = set()
    profile_shaped_rows = tuple(row.task_id for row in rows if row.metadata.get("profile_hash"))
    # Held against the rows whether or not the role is enabled: a manifest that
    # claims a profile shaped surfaces it cannot point at, or disclaims one the
    # rows carry, is not a record this gate can clear a candidate against.
    _cross_check_profile_influence(manifest, bool(profile_shaped_rows))
    enabled_roles: set[str] = set()
    for role in GENERATION_EXPOSURE_ROLES:
        entry = _mapping(
            models[role],
            f"source_run_manifest.models.{role}",
            recovery="point at a run_manifest.json written by this pipeline",
        )
        if not _boolean(
            entry.get("enabled"),
            f"source_run_manifest.models.{role}.enabled",
            recovery="point at a run_manifest.json written by this pipeline",
        ):
            continue
        enabled_roles.add(role)
        claim = _role_identity(role, entry)
        if role == "profile":
            task_ids = profile_shaped_rows
            scope = "profile_shaped_rows"
        elif role == "paraphrase":
            task_ids = tuple(row.task_id for row in rows if _paraphrased_by(row, claim.label))
            scope = "paraphrased_rows"
            paraphrased_rows.update(task_ids)
        else:
            task_ids = _judged_rows(manifest, index)
            scope = "all_published_rows"
        if not task_ids:
            # The role ran, but nothing it produced survived into the published
            # table. It read no published row, so it cannot have leaked one.
            continue
        exposures.append(
            VerifiedModelExposure(
                role=role,
                scope=scope,
                identity=claim,
                task_ids=task_ids,
                evidence=f"run_manifest.json models.{role}",
            )
        )

    _require_paraphrase_attribution(rows, paraphrased_rows)
    if profile_shaped_rows and "profile" not in enabled_roles:
        raise ModelExposureError(
            "benchmark.metadata.profile_hash",
            f"records that a reference profile shaped {len(profile_shaped_rows)} row(s), but the manifest "
            "declares no profile model",
            expected="a models.profile entry naming the model whose style the rows carry",
            recovery="regenerate the benchmark; a style no model is named for cannot be checked against a "
            "candidate, and would be scored as if the candidate had never seen these surfaces",
        )
    if translation is not None:
        exposures.append(
            VerifiedModelExposure(
                role="translator",
                scope="translated_rows",
                identity=translation.translator,
                task_ids=index.task_ids,
                evidence=f"{translation.manifest_path.name} model",
            )
        )
    checks.append(
        SourceCheck(
            name="model_exposure",
            detail=(
                ", ".join(
                    f"{exposure.role} read {len(exposure.task_ids)} row(s) as {exposure.display_name}"
                    for exposure in exposures
                )
                if exposures
                else "no model read the published rows: every declared role was disabled or shaped nothing"
            ),
        )
    )
    return tuple(exposures)


def _role_identity(role: str, entry: Mapping[str, Any]) -> ModelIdentityClaim:
    """Read one enabled generation role as an identity claim.

    A generation config is not required to pin a revision or a digest — it only
    had to name what it called — so a claim built here is often weaker than a
    candidate's. What it may never be is empty: a role that ran without
    recording any name at all would compare as "unknown" against every
    candidate, and the operator would have no field to go and fix.
    """
    identity = entry.get("model_identity")
    fields = identity if isinstance(identity, Mapping) else {}
    claim = ModelIdentityClaim(
        provider=_optional_text(entry.get("provider")),
        served_model=_optional_text(fields.get("model")),
        weight_source=_optional_text(fields.get("source")),
        weight_model=_optional_text(fields.get("model")),
        revision=_optional_text(fields.get("revision")),
        weights_digest=_optional_text(fields.get("weights_digest")),
        label=_optional_text(entry.get("canonical_id")),
    )
    if not claim.names_a_model:
        raise ModelExposureError(
            f"source_run_manifest.models.{role}",
            "is enabled but names no model, so no candidate can be told apart from it",
            expected="provider, model_identity.model, or canonical_id",
            recovery="regenerate the benchmark with lineage.roles configured, or evaluate a publication whose "
            "manifest records the models that built it",
        )
    return claim


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _paraphrased_by(row: CanonicalExportRow, label: str | None) -> bool:
    """Whether this row's surface was written by the model that declares ``label``.

    Publication normalizes the role's canonical id while the row keeps the case the
    pack config used, so the two are compared case-insensitively. A role with no
    canonical id at all claims every paraphrased row, which over-attributes
    rather than losing one.
    """
    written_by = row.paraphrase_model_canonical
    if not written_by:
        return False
    return label is None or written_by.strip().casefold() == label.strip().casefold()


def _judged_rows(manifest: Mapping[str, Any], index: SourceTaskIndex) -> tuple[str, ...]:
    """Every published row, unless the manifest records that no surface gate ran.

    ``models.surface_judge.enabled`` says a judge was configured;
    ``surface_quality_validation.enabled`` says it actually scored surfaces. A
    manifest that records neither is read as "it ran", because an evaluation may
    not clear a candidate on the strength of a field the publication omitted.

    Every published row is the exact scope, not a conservative one. Surface-quality validation
    withholds a surface from the judge only when a deterministic guard already
    failed it, and a surface a guard failed is dropped before publication — so a
    row that reached the published table is a row the judge read.
    """
    validation = manifest.get("surface_quality_validation")
    if isinstance(validation, Mapping) and type(validation.get("enabled")) is bool:
        if not validation["enabled"]:
            return ()
    return index.task_ids


def _cross_check_profile_influence(manifest: Mapping[str, Any], shaped_rows: bool) -> None:
    """Hold ``profile_influenced_surface`` to what the published rows record."""
    declared = manifest.get("profile_influenced_surface")
    if type(declared) is bool and declared != shaped_rows:
        raise ModelExposureError(
            "source_run_manifest.profile_influenced_surface",
            f"declares {declared}, but the published rows record the opposite",
            actual=declared,
            expected="the same verdict the rows' profile_hash carries",
            recovery="regenerate the benchmark; a manifest that disagrees with its own rows about which model "
            "shaped them cannot establish what a candidate has already seen",
        )


def _require_paraphrase_attribution(
    rows: Sequence[CanonicalExportRow],
    attributed: set[str],
) -> None:
    """Refuse a row written by a model the manifest does not declare."""
    unattributed = sorted(
        {
            str(row.paraphrase_model_canonical)
            for row in rows
            if row.paraphrase_model_canonical and row.task_id not in attributed
        }
    )
    if unattributed:
        raise ModelExposureError(
            "benchmark.paraphrase_model_canonical",
            f"names model(s) {unattributed[:3]} that no enabled role in the manifest declares",
            expected="every paraphrased row attributed to a model the manifest records",
            recovery="regenerate the benchmark; a row whose author is not in the manifest cannot be checked "
            "against a candidate, and would be scored as if no model had written it",
        )


def source_verification_report(
    source: VerifiedEvalSource,
    *,
    verified_at: datetime | None = None,
) -> SourceVerificationReport:
    """Wrap a verified source into the artifact a later stage can cite."""
    moment = verified_at or datetime.now(timezone.utc)
    return SourceVerificationReport(verified_at=moment.isoformat(), source=source)


def write_source_verification_report(
    config: BfclEvalConfig,
    source: VerifiedEvalSource,
    *,
    verified_at: datetime | None = None,
) -> tuple[Path, str]:
    """Write the passing report into the eval output tree, atomically.

    Returns the path and content hash that the candidate runtime records in the evaluation
    manifest: a score that cannot name the source verification it ran under is
    not auditable.
    """
    report = source_verification_report(source, verified_at=verified_at)
    return write_eval_artifact(
        config,
        SOURCE_VERIFICATION_REPORT_FILE,
        report.as_document(),
        supersedes=SOURCE_VERIFICATION_FAILURE_FILE,
    )


def write_source_failure_diagnostic(
    config: BfclEvalConfig,
    error: Exception,
) -> tuple[Path, str]:
    """Record why verification failed, under a name no reader can mistake for a pass."""
    document: dict[str, Any] = {
        "schema_version": SOURCE_VERIFICATION_CONTRACT_VERSION,
        "status": "failed",
        "diagnosed_at": datetime.now(timezone.utc).isoformat(),
        "eval_config_hash": config.eval_config_hash,
        "source_run_id": config.source.run_id,
        "error": (
            error.as_report()
            if isinstance(error, SourceVerificationError)
            else {"code": "eval_source_invalid", "problem": type(error).__name__}
        ),
    }
    return write_eval_artifact(
        config,
        SOURCE_VERIFICATION_FAILURE_FILE,
        document,
        supersedes=SOURCE_VERIFICATION_REPORT_FILE,
    )


def write_eval_artifact(
    config: BfclEvalConfig,
    name: str,
    document: Mapping[str, Any],
    *,
    supersedes: str | None = None,
) -> tuple[Path, str]:
    """Write one eval artifact into ``outputs.output_dir`` and nowhere else.

    ``supersedes`` names the opposite verdict for the same decision — a failure
    diagnostic for a report, or the other way round. Passing it is what keeps a
    stale pass from sitting beside a fresh failure, which is the one way a reader
    of these files can be actively misled.
    """
    output_dir = config.outputs.output_dir.resolve()
    publication_dir = config.source.publication_dir.resolve()
    if output_dir == publication_dir or publication_dir in output_dir.parents or output_dir in publication_dir.parents:
        raise SourceVerificationError(
            "outputs.output_dir",
            f"overlaps the source publication tree at {publication_dir}",
            expected="an eval output directory outside the generation run's tree",
            recovery="write eval artifacts to their own directory; verification must never be able to overwrite "
            "the benchmark it verified",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / name
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        # A crash may leave no verdict, but must never leave a stale passing
        # report beside a newer failure (or vice versa). Remove the old verdict
        # only after the replacement bytes are durable enough to promote.
        if supersedes is not None:
            (output_dir / supersedes).unlink(missing_ok=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def assert_source_unchanged(source: VerifiedEvalSource) -> None:
    """Re-check the verified bytes immediately before they are used.

    Verification and execution are separated in time, and the gap is exactly
    where a source can be replaced — a re-run of generation into the same
    directory, a pack edited to make a failing task pass. This is the second pin:
    every hash the handle recorded is recomputed, including the oracle pack's
    fingerprint, because a pack file is what an executable claim rests on.
    """
    expectations: list[tuple[str, Path, str]] = [
        ("source_run_manifest", source.source_manifest_path, source.source_manifest_hash),
        (source.benchmark.file, source.benchmark.path, source.benchmark.content_hash),
        (source.raw_benchmark.file, source.raw_benchmark.path, source.raw_benchmark.content_hash),
    ]
    if source.translation is not None:
        expectations.extend(
            [
                (
                    "translation_manifest",
                    source.translation.manifest_path,
                    source.translation.manifest_content_hash,
                ),
                (
                    source.translation.benchmark.file,
                    source.translation.benchmark.path,
                    source.translation.benchmark.content_hash,
                ),
            ]
        )
    if source.oracle is not None:
        expectations.append(
            (source.oracle.resource_role, source.oracle.resource_path, source.oracle.resource_content_hash)
        )
    for label, path, expected in expectations:
        if not path.is_file():
            raise SourceChangedDuringEvalError(
                label,
                f"is no longer present at {path}",
                expected=expected,
                recovery="verify the source again before running; an evaluation may not span two sources",
            )
        actual = _sha256_file(path)
        if actual != expected:
            raise SourceChangedDuringEvalError(
                label,
                "changed after the source was verified",
                actual=actual,
                expected=expected,
                recovery="stop the run and verify the source again; results from before and after the change "
                "would be reported as one score for one benchmark",
            )
    if source.oracle is None:
        return
    try:
        paths = resolve_declared_pack_paths(
            OraclePackRef(manifest_path=source.oracle.pack_manifest_path),
            (source.oracle.pack_root,),
        )
        actual_fingerprint = f"sha256:{pack_fingerprint(paths)}"
    except (PackTrustError, FileNotFoundError, ValueError, OSError, yaml.YAMLError) as exc:
        raise SourceChangedDuringEvalError(
            "source_oracle.pack",
            f"can no longer be resolved after verification: {type(exc).__name__}",
            expected=source.oracle.actual_pack_content_hash,
            recovery="stop the run and verify the source again; a pack that disappears or becomes invalid "
            "mid-run cannot support an executable result",
        ) from exc
    if actual_fingerprint != source.oracle.actual_pack_content_hash:
        raise SourceChangedDuringEvalError(
            "source_oracle.pack",
            f"pack {source.oracle.pack_id} changed after the source was verified",
            actual=actual_fingerprint,
            expected=source.oracle.actual_pack_content_hash,
            recovery="stop the run and verify the source again; an oracle that changes mid-run makes the "
            "executable results incomparable",
        )
