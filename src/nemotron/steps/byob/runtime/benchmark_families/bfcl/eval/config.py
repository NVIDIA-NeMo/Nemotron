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

"""Strict loader for ``eval_config.yaml`` (schemas 1.1 and 1.2).

The resolution order is fixed, and each step exists because skipping it would let
an invalid evaluation start spending tokens:

1. Parse the YAML strictly and close every section, so a misspelled key is an
   error rather than a setting that had no effect.
2. Refuse template configs and ``REPLACE_ME_*`` values: a config nobody finished
   editing must not resolve into a run that looks finished.
3. Refuse credentials written into the config, before anything is logged or
   hashed. Only environment variable *names* survive.
4. Resolve relative paths against the config's own directory, then validate the
   files they name: the source run manifest, the benchmark it published, the
   optional translation manifest, and the scoring contract document.
5. Replace those paths with content hashes and build the frozen
   :class:`~...schemas.BfclEvalConfig`, whose validators own every cross-field
   rule (candidate identity, publication gates, mode coherence).

Config resolution deliberately does *not* verify that the benchmark parquet still
matches the hash its manifest claims, compute contamination overlap, or contact a
candidate endpoint. Those are source verification, authorization, and runtime
concerns. A missing ``NVIDIA_API_KEY`` is not a config error either — the config
names the variable, and execution reads it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar

import yaml
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
    BENCHMARK_SCHEMA_VERSIONS,
    BYOB_ROOT,
    BfclConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import (
    CandidateIdentityError,
    EvalConfigError,
    EvalConfigPathError,
    EvalConfigSchemaError,
    SecretInConfigError,
    redact_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.held_out_eval import (
    HeldOutEvalConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    EVAL_CONFIG_SCHEMA_VERSION,
    EVAL_CONFIG_SCHEMA_VERSIONS,
    WEIGHT_MANIFEST_DIGEST_SCHEME,
    BfclEvalConfig,
    CandidateApi,
    CandidateInference,
    CandidateModelIdentity,
    ContaminationPolicy,
    EvalCandidate,
    EvalFileRef,
    EvalLimits,
    EvalOracleResource,
    EvalOutputConfig,
    EvalPublicationPolicy,
    EvalScoringConfig,
    EvalSettings,
    EvalSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    PUBLICATION_BENCHMARK_TABLE,
    RAW_BENCHMARK_TABLE,
)

EVAL_CONFIG_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "config_status",
        "source_run_manifest",
        "source_oracle",
        "translation_manifest",
        "eval",
        "held_out_eval",
        "scoring",
        "limits",
        "candidates",
        "contamination",
        "publication",
        "outputs",
    }
)
_SOURCE_ORACLE_KEYS: Final = frozenset({"kind", "pack_manifest", "resource"})
_EVAL_KEYS: Final = frozenset({"mode"})
_HELD_OUT_EVAL_KEYS: Final = frozenset(
    {
        "contract_version",
        "policy_hash",
        "fixture_refs",
        "template_ids",
        "seed",
        "pack_version",
        "max_tasks_per_template",
    }
)
_SCORING_KEYS: Final = frozenset(
    {
        "contract",
        "argument_matching",
        "insert_declared_defaults",
        "respect_call_order",
        "respect_call_group",
        "allow_llm_repair",
        "task_success",
        "intermediate_text_matching",
    }
)
# A config written before intermediate text had a policy meant the one policy
# there was, so its absence resolves to the publication value rather than
# refusing the file.
_SCORING_OPTIONAL: Final = frozenset({"intermediate_text_matching"})
_LIMITS_KEYS: Final = frozenset(
    {
        "max_turns",
        "tool_timeout_s",
        "candidate_timeout_s",
        "episode_timeout_s",
        "max_parallel_tasks",
        "max_retries",
    }
)
_CANDIDATE_KEYS: Final = frozenset(
    {"alias", "model", "provider", "provider_api_version", "api", "model_identity", "inference"}
)
_API_KEYS: Final = frozenset({"base_url", "api_key_env"})
_IDENTITY_KEYS: Final = frozenset({"source", "model", "revision", "weights_digest"})
_IDENTITY_OPTIONAL: Final = frozenset({"revision", "weights_digest"})
_INFERENCE_KEYS: Final = frozenset(
    {"temperature", "top_p", "max_tokens", "seed", "tool_choice", "provider_extensions"}
)
_INFERENCE_OPTIONAL: Final = frozenset({"seed", "provider_extensions"})
_CONTAMINATION_KEYS: Final = frozenset({"enforce", "on_violation", "comparison_set"})
_PUBLICATION_KEYS: Final = frozenset({"requested", "require_same_task_ids"})
_OUTPUT_KEYS: Final = frozenset(
    {
        "output_dir",
        "write_task_results",
        "write_eval_manifest",
        "cache_candidate_responses",
        "cache_tool_results",
    }
)

# Key names that hold a credential rather than a reference to one. ``api_key_env``
# is absent on purpose: naming the variable is the supported way to pass a key.
_SECRET_KEY_NAMES: Final = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "credentials",
        "private_key",
    }
)
_SECRET_KEY_SUFFIXES: Final = ("_api_key", "_password", "_secret", "_access_token", "_auth_token", "_token")
# Values that are recognizably credentials whatever field they were pasted into.
# A key leaked through a misnamed field is still a leaked key.
_SECRET_VALUE_PREFIXES: Final = (
    "nvapi-",
    "sk-",
    "sk_live_",
    "hf_",
    "ghp_",
    "gho_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "aiza",
    "bearer ",
)
_PLACEHOLDER: Final = "REPLACE_ME_"

# Artifacts that identify a generation publication tree. Eval output may not land
# in a directory holding any of them: publication owns those bytes, and an eval run
# that writes beside them makes it impossible to say which run published what.
_PUBLICATION_ARTIFACTS: Final = (
    "run_manifest.json",
    PUBLICATION_BENCHMARK_TABLE,
    RAW_BENCHMARK_TABLE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_mapping(value: Any, field: str, *, recovery: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvalConfigSchemaError(
            field,
            "must be a mapping",
            value=value,
            expected="a YAML mapping",
            recovery=recovery,
        )
    return {str(key): child for key, child in value.items()}


def _section(
    data: Mapping[str, Any],
    name: str,
    allowed: frozenset[str],
    *,
    optional: frozenset[str] = frozenset(),
    field: str | None = None,
) -> dict[str, Any]:
    """Read one closed section, with every non-optional key present.

    Absent keys are errors rather than defaults: a limit or a gate the operator
    never wrote is a limit nobody chose, and a run must not be able to claim a
    setting the config does not state.
    """
    label = field or name
    if name not in data:
        raise EvalConfigSchemaError(
            label,
            "section is required",
            expected=f"a mapping with keys: {', '.join(sorted(allowed))}",
            recovery=f"add the {name}: section; eval configs pin every setting instead of inheriting defaults",
        )
    section = _require_mapping(data[name], label, recovery=f"write {name} as a mapping of settings")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise EvalConfigSchemaError(
            label,
            f"unknown key(s): {', '.join(unknown)}",
            expected=f"only {', '.join(sorted(allowed))}",
            recovery="remove the unknown key, or upgrade to a schema version that declares it",
        )
    missing = sorted(allowed - optional - set(section))
    if missing:
        raise EvalConfigSchemaError(
            label,
            f"missing required key(s): {', '.join(missing)}",
            expected=f"every one of {', '.join(sorted(allowed - optional))}",
            recovery="state the value explicitly; nothing here falls back to a provider or pipeline default",
        )
    return section


def _walk(value: Any, field: str) -> list[tuple[str, Any, Any]]:
    """Flatten a config into ``(field_path, key, value)`` triples for policy scans."""
    found: list[tuple[str, Any, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{field}.{key}" if field else str(key)
            found.append((path, key, child))
            found.extend(_walk(child, path))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            path = f"{field}[{index}]"
            found.append((path, None, child))
            found.extend(_walk(child, path))
    return found


def _reject_placeholders(data: Mapping[str, Any]) -> None:
    for path, _key, value in _walk(data, ""):
        if isinstance(value, str) and _PLACEHOLDER in value:
            raise EvalConfigSchemaError(
                path,
                "still carries a template placeholder",
                value=value,
                expected=f"a real value with no {_PLACEHOLDER}* placeholder",
                recovery="fill in the value; a resolved config is what an eval manifest claims was run",
            )


def _reject_secrets(data: Mapping[str, Any]) -> None:
    """Refuse credential values before the config is resolved, hashed, or logged."""
    for path, key, value in _walk(data, ""):
        if key is not None:
            lowered = str(key).lower().replace("-", "_")
            if lowered in _SECRET_KEY_NAMES or lowered.endswith(_SECRET_KEY_SUFFIXES):
                raise SecretInConfigError(
                    path,
                    "names a credential value rather than a reference to one",
                    value=value,
                    secret=True,
                    expected="candidates[].api.api_key_env naming an environment variable",
                    recovery="delete this key and set api_key_env to the variable name the runner should read",
                )
        if isinstance(value, str) and value.strip().lower().startswith(_SECRET_VALUE_PREFIXES):
            raise SecretInConfigError(
                path,
                "looks like a literal credential",
                value=value,
                secret=True,
                expected="no credential values anywhere in the config",
                recovery="move the credential into an environment variable and reference it with api_key_env; "
                "rotate the key that was written to this file",
            )


def _resolve(raw: Any, field: str, base_dir: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise EvalConfigSchemaError(
            field,
            "must be a non-empty path string",
            value=raw,
            expected="a path, absolute or relative to the eval config's own directory",
            recovery=f"set {field} to the file or directory the run should use",
        )
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return Path(path).resolve()


def _existing_file(path: Path, field: str, *, recovery: str) -> Path:
    if not path.exists():
        raise EvalConfigPathError(
            field,
            f"does not exist: {path}",
            value=str(path),
            expected="an existing file",
            recovery=recovery,
        )
    if not path.is_file():
        raise EvalConfigPathError(
            field,
            f"is not a file: {path}",
            value=str(path),
            expected="a file",
            recovery=recovery,
        )
    return path


def _file_ref(path: Path, field: str) -> EvalFileRef:
    return _model(EvalFileRef, {"path": path, "content_hash": _sha256_file(path)}, field)


def _json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalConfigPathError(
            field,
            f"could not be read as JSON: {type(exc).__name__}",
            value=str(path),
            expected="a readable JSON document",
            recovery="point the field at the JSON file the generation run published, unmodified",
        ) from exc
    if not isinstance(payload, Mapping):
        raise EvalConfigPathError(
            field,
            "is not a JSON object",
            value=str(path),
            expected="a JSON object",
            recovery="point the field at run_manifest.json, not at a list or a scalar",
        )
    return dict(payload)


def _model(model: Any, payload: Mapping[str, Any], field: str) -> Any:
    """Build one strict model, translating pydantic's report into our taxonomy."""
    try:
        return model(**payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(
            f"[{part}]" if isinstance(part, int) else str(part) for part in first.get("loc", ())
        ).replace(".[", "[")
        raise EvalConfigSchemaError(
            f"{field}.{location}" if location else field,
            first.get("msg", "is not valid"),
            value=first.get("input"),
            expected=f"a value matching eval config schema {EVAL_CONFIG_SCHEMA_VERSION}",
            recovery="correct the value; quoted booleans and numbers are refused on purpose so a config "
            "cannot silently mean something else",
        ) from exc


def _manifest_string(manifest: Mapping[str, Any], key: str, field: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalConfigPathError(
            field,
            f"does not record {key!r}, so it is not a BFCL run manifest",
            value=value,
            expected=f"a run manifest carrying a non-empty {key}",
            recovery="point source_run_manifest at run_manifest.json from a completed stage=generate run",
        )
    return value.strip()


def _resolve_source(data: Mapping[str, Any], base_dir: Path) -> EvalSource:
    """Resolve the published run this evaluation reads, from its manifest alone."""
    field = "source_run_manifest"
    if field not in data:
        raise EvalConfigSchemaError(
            field,
            "is required",
            expected="a path to the run_manifest.json of a published generation run",
            recovery="run stage=generate first, then point source_run_manifest at its run_manifest.json",
        )
    raw = data[field]
    manifest_path = _resolve(raw, field, base_dir)
    if manifest_path.suffix.lower() == ".parquet":
        raise EvalConfigPathError(
            field,
            "points at a benchmark table instead of a manifest",
            value=str(manifest_path),
            expected="run_manifest.json, which names the table it published",
            recovery="point at run_manifest.json; a parquet without a manifest beside it is not published output",
        )
    _existing_file(
        manifest_path,
        field,
        recovery="point at run_manifest.json from a completed stage=generate run; a run whose manifest is "
        "absent published nothing",
    )
    manifest = _json_object(manifest_path, field)

    schema_version = _manifest_string(manifest, "schema_version", field)
    if schema_version not in BENCHMARK_SCHEMA_VERSIONS:
        raise EvalConfigPathError(
            f"{field}.schema_version",
            f"names a benchmark schema this build cannot read: {schema_version!r}",
            value=schema_version,
            expected=f"one of {', '.join(sorted(BENCHMARK_SCHEMA_VERSIONS))}",
            recovery="evaluate with the pipeline revision that published the run, or regenerate the benchmark",
        )
    run_id = _manifest_string(manifest, "run_id", field)
    lineage_policy = _manifest_string(manifest, "lineage_policy", field)
    gold_eligible = manifest.get("gold_eligible")
    if not isinstance(gold_eligible, bool):
        raise EvalConfigPathError(
            f"{field}.gold_eligible",
            "does not record a boolean gold_eligible verdict",
            value=gold_eligible,
            expected="true or false",
            recovery="point at a run_manifest.json written by this pipeline; publication eligibility is a "
            "generation verdict, not an eval setting",
        )

    oracle_resource = _resolve_oracle_resource(data, manifest, base_dir)

    publication = manifest.get("publication")
    published_file = PUBLICATION_BENCHMARK_TABLE
    benchmark_claimed_hash: str | None = None
    if isinstance(publication, Mapping):
        published = publication.get("published")
        if isinstance(published, Mapping):
            name = published.get("file")
            if isinstance(name, str) and name.strip():
                published_file = name.strip()
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        entry = artifacts.get("benchmark_parquet")
        if isinstance(entry, Mapping) and isinstance(entry.get("content_hash"), str):
            benchmark_claimed_hash = str(entry["content_hash"])
    if benchmark_claimed_hash is None:
        raise EvalConfigPathError(
            f"{field}.artifacts.benchmark_parquet",
            "does not record the published table's content hash",
            expected="artifacts.benchmark_parquet.content_hash",
            recovery="point at a run_manifest.json written by this pipeline; the eval run identifies the "
            "benchmark by the hash the manifest published",
        )
    if Path(published_file).name != published_file:
        raise EvalConfigPathError(
            f"{field}.publication.published.file",
            f"names a path instead of a file beside the manifest: {published_file!r}",
            value=published_file,
            expected="a plain file name",
            recovery="point at an unmodified run_manifest.json",
        )
    benchmark_path = _existing_file(
        manifest_path.parent / published_file,
        f"{field}.publication.published.file",
        recovery="keep the published table beside its manifest; evaluation reads the committed publication tree",
    )
    # The hash comes from the manifest, not from re-reading the parquet: config
    # resolution pins what the run claims and source verification proves the bytes.
    benchmark = _model(
        EvalFileRef,
        {"path": benchmark_path, "content_hash": benchmark_claimed_hash},
        f"{field}.artifacts.benchmark_parquet",
    )

    translation_ref: EvalFileRef | None = None
    translation_raw = data.get("translation_manifest")
    if translation_raw is not None:
        translation_field = "translation_manifest"
        translation_path = _existing_file(
            _resolve(translation_raw, translation_field, base_dir),
            translation_field,
            recovery="point at the translation manifest of a run derived from this same source run, or set it to null",
        )
        translation = _json_object(translation_path, translation_field)
        declared_run = translation.get("source_run_id")
        declared_hash = translation.get("source_run_manifest_content_hash")
        manifest_hash = _sha256_file(manifest_path)
        matches_run = isinstance(declared_run, str) and declared_run.strip() == run_id
        matches_hash = isinstance(declared_hash, str) and declared_hash.strip() == manifest_hash
        if not (matches_run or matches_hash):
            raise EvalConfigPathError(
                translation_field,
                "does not reference the source run this config evaluates",
                value=str(translation_path),
                expected=f"source_run_id == {run_id!r} or source_run_manifest_content_hash == the manifest's hash",
                recovery="use the translation manifest produced from this source run; scoring a translation "
                "against another run's gold trace compares two different benchmarks",
            )
        translation_ref = _file_ref(translation_path, translation_field)

    return _model(
        EvalSource,
        {
            "run_manifest": _file_ref(manifest_path, field),
            "benchmark": benchmark,
            "translation_manifest": translation_ref,
            "run_id": run_id,
            "benchmark_schema_version": schema_version,
            "publication_dir": manifest_path.parent,
            "gold_eligible": gold_eligible,
            "lineage_policy": lineage_policy,
            "oracle": oracle_resource,
        },
        "source_run_manifest",
    )


def _resolve_oracle_resource(
    data: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    base_dir: Path,
) -> EvalOracleResource | None:
    """Resolve the pack and concrete resource an executable run will open."""
    raw = data.get("source_oracle")
    if raw is None:
        return None
    section = _require_mapping(
        raw,
        "source_oracle",
        recovery="set source_oracle to null for trace-only evaluation, or declare kind, pack_manifest, and resource",
    )
    unknown = sorted(set(section) - _SOURCE_ORACLE_KEYS)
    missing = sorted(_SOURCE_ORACLE_KEYS - set(section))
    if unknown:
        raise EvalConfigSchemaError(
            "source_oracle",
            f"unknown key(s): {', '.join(unknown)}",
            expected=f"only {', '.join(sorted(_SOURCE_ORACLE_KEYS))}",
            recovery="remove the unknown key; the pack manifest owns all other oracle paths",
        )
    if missing:
        raise EvalConfigSchemaError(
            "source_oracle",
            f"missing required key(s): {', '.join(missing)}",
            expected=f"every one of {', '.join(sorted(_SOURCE_ORACLE_KEYS))}",
            recovery="pin both the pack manifest and the backend.py or endpoint config used by generation",
        )

    kind = section["kind"]
    if kind not in {"python", "endpoint"}:
        raise EvalConfigSchemaError(
            "source_oracle.kind",
            "does not name a supported oracle execution kind",
            value=kind,
            expected="'python' or 'endpoint'",
            recovery="use python for backend.py or endpoint for a BFCL Oracle HTTP v1 endpoint config",
        )
    declared_oracle = source_manifest.get("oracle")
    source_kind = declared_oracle.get("kind") if isinstance(declared_oracle, Mapping) else None
    if source_kind != kind:
        raise EvalConfigPathError(
            "source_oracle.kind",
            "does not match oracle.kind in the source run manifest",
            value=kind,
            expected=f"the source run's declared kind ({redact_value(source_kind)})",
            recovery="use the same oracle kind that generated and replay-validated the benchmark",
        )

    pack = source_manifest.get("pack")
    if not isinstance(pack, Mapping):
        raise EvalConfigPathError(
            "source_run_manifest.pack",
            "does not carry source pack identity",
            expected="pack_id, version, and content_hash",
            recovery="point at run_manifest.json from a completed BFCL publication",
        )
    pack_id = pack.get("pack_id")
    pack_version = pack.get("version")
    pack_hash = pack.get("content_hash")
    if not isinstance(pack_id, str) or not pack_id.strip():
        raise EvalConfigPathError(
            "source_run_manifest.pack.pack_id",
            "is missing or empty",
            value=pack_id,
            expected="the source oracle pack id",
            recovery="point at an unmodified BFCL run manifest",
        )
    if not isinstance(pack_version, str) or not pack_version.strip():
        raise EvalConfigPathError(
            "source_run_manifest.pack.version",
            "is missing or empty",
            value=pack_version,
            expected="the source oracle pack version",
            recovery="point at an unmodified BFCL run manifest",
        )

    pack_manifest_path = _existing_file(
        _resolve(section["pack_manifest"], "source_oracle.pack_manifest", base_dir),
        "source_oracle.pack_manifest",
        recovery="point at manifest.yaml from the exact oracle pack used by the source generation run",
    )
    try:
        pack_manifest = yaml.safe_load(pack_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvalConfigPathError(
            "source_oracle.pack_manifest",
            f"could not be read as YAML: {type(exc).__name__}",
            value=str(pack_manifest_path),
            expected="a readable oracle-pack manifest mapping",
            recovery="point at the source pack's manifest.yaml",
        ) from exc
    if not isinstance(pack_manifest, Mapping):
        raise EvalConfigPathError(
            "source_oracle.pack_manifest",
            "is not a YAML mapping",
            value=str(pack_manifest_path),
            expected="an oracle-pack manifest mapping",
            recovery="point at the source pack's manifest.yaml",
        )
    if pack_manifest.get("pack_id") != pack_id or str(pack_manifest.get("version")) != pack_version:
        raise EvalConfigPathError(
            "source_oracle.pack_manifest",
            "does not identify the pack recorded by the source run",
            value=str(pack_manifest_path),
            expected=f"pack_id {pack_id!r} and version {pack_version!r}",
            recovery="use the manifest from the exact pack revision that generated the source benchmark",
        )

    resource_path = _existing_file(
        _resolve(section["resource"], "source_oracle.resource", base_dir),
        "source_oracle.resource",
        recovery="point at backend.py for kind=python or the endpoint config for kind=endpoint",
    )
    if kind == "python" and resource_path.suffix != ".py":
        raise EvalConfigPathError(
            "source_oracle.resource",
            "is not a Python backend file",
            value=str(resource_path),
            expected="a .py file for source_oracle.kind=python",
            recovery="point at the backend.py used by the source oracle pack",
        )
    if kind == "endpoint" and resource_path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise EvalConfigPathError(
            "source_oracle.resource",
            "is not an endpoint configuration file",
            value=str(resource_path),
            expected="a YAML or JSON endpoint config for source_oracle.kind=endpoint",
            recovery="point at the immutable endpoint config used by the source pack",
        )

    return _model(
        EvalOracleResource,
        {
            "kind": kind,
            "pack_manifest": _file_ref(pack_manifest_path, "source_oracle.pack_manifest"),
            "execution_resource": _file_ref(resource_path, "source_oracle.resource"),
            "pack_id": pack_id.strip(),
            "pack_version": pack_version.strip(),
            "expected_pack_content_hash": pack_hash,
        },
        "source_oracle",
    )


def _resolve_scoring(data: Mapping[str, Any], base_dir: Path) -> EvalScoringConfig:
    section = _section(data, "scoring", _SCORING_KEYS, optional=_SCORING_OPTIONAL)
    contract_path = _existing_file(
        _resolve(section["contract"], "scoring.contract", base_dir),
        "scoring.contract",
        recovery="point scoring.contract at the document that defines how a call is compared; it is "
        "content-hashed, so the rules cannot change without changing eval_config_hash",
    )
    payload = {**section, "contract": _file_ref(contract_path, "scoring.contract")}
    return _model(EvalScoringConfig, payload, "scoring")


_T = TypeVar("_T")


def _resolve_eval_settings(data: Mapping[str, Any]) -> EvalSettings:
    modes = _section(data, "eval", _EVAL_KEYS)["mode"]
    if isinstance(modes, str) or not isinstance(modes, Sequence):
        raise EvalConfigSchemaError(
            "eval.mode",
            "must be a list of modes",
            value=modes,
            expected="a list such as [trace] or [trace, executable]",
            recovery="write eval.mode as a YAML list, even for a single mode",
        )
    return _model(EvalSettings, {"modes": tuple(modes)}, "eval")


def _resolve_held_out_eval(data: Mapping[str, Any]) -> HeldOutEvalConfig | None:
    if "held_out_eval" not in data:
        return None
    return _model(
        HeldOutEvalConfig,
        _section(data, "held_out_eval", _HELD_OUT_EVAL_KEYS, optional=frozenset({"contract_version"})),
        "held_out_eval",
    )


class _Violations:
    """Run independent config sections and keep every refusal, not just the first.

    A preflight sends no candidate request, so stopping at the first bad section
    saved nothing and cost a round trip per problem. The first refusal in config
    order is still the one raised, so a config with a single fault reports
    exactly what it always did.
    """

    def __init__(self) -> None:
        self.found: list[EvalConfigError] = []

    def run(self, resolve: Callable[[], _T]) -> _T | None:
        try:
            return resolve()
        except EvalConfigError as exc:
            self.found.append(exc)
            return None

    def add(self, violation: EvalConfigError) -> None:
        self.found.append(violation)

    def raise_if_any(self) -> None:
        if not self.found:
            return
        first, *rest = self.found
        raise first.also_report(rest)


def _resolve_candidates(data: Mapping[str, Any], violations: _Violations) -> tuple[EvalCandidate, ...]:
    """Resolve every candidate, recording each one's fault rather than raising it.

    One candidate's fault says nothing about the next one's. The caller owns the
    single collector for the whole config, so the refusals stay in config order
    and no collector nests inside another.
    """
    raw = data.get("candidates")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        violations.add(
            EvalConfigSchemaError(
                "candidates",
                "must be a list of candidate mappings",
                value=raw,
                expected="a non-empty list",
                recovery="declare one entry per model under evaluation",
            )
        )
        return ()
    candidates: list[EvalCandidate] = []
    for index, entry in enumerate(raw):
        resolved = violations.run(lambda index=index, entry=entry: _resolve_candidate(entry, index))
        if resolved is not None:
            candidates.append(resolved)
    return tuple(candidates)


def _resolve_candidate(entry: Any, index: int) -> EvalCandidate:
    field = f"candidates[{index}]"
    candidate = _require_mapping(entry, field, recovery="write each candidate as a mapping")
    unknown = sorted(set(candidate) - _CANDIDATE_KEYS)
    if unknown:
        raise EvalConfigSchemaError(
            field,
            f"unknown key(s): {', '.join(unknown)}",
            expected=f"only {', '.join(sorted(_CANDIDATE_KEYS))}",
            recovery="remove the unknown key; decoding settings belong under inference",
        )
    missing = sorted(_CANDIDATE_KEYS - set(candidate))
    if missing:
        raise EvalConfigSchemaError(
            field,
            f"missing required key(s): {', '.join(missing)}",
            expected=f"every one of {', '.join(sorted(_CANDIDATE_KEYS))}",
            recovery="a candidate states its serving route and the weights it serves; run "
            "resolve_bfcl_model_identity to fill model_identity in from the registry or the provider",
        )
    api = _section(candidate, "api", _API_KEYS, field=f"{field}.api")
    identity = _section(
        candidate,
        "model_identity",
        _IDENTITY_KEYS,
        optional=_IDENTITY_OPTIONAL,
        field=f"{field}.model_identity",
    )
    inference = _section(
        candidate,
        "inference",
        _INFERENCE_KEYS,
        optional=_INFERENCE_OPTIONAL,
        field=f"{field}.inference",
    )
    try:
        return _model(
            EvalCandidate,
            {
                **{key: candidate[key] for key in ("alias", "model", "provider", "provider_api_version")},
                "api": _model(CandidateApi, api, f"{field}.api"),
                "model_identity": _model(CandidateModelIdentity, identity, f"{field}.model_identity"),
                "inference": _model(CandidateInference, inference, f"{field}.inference"),
            },
            field,
        )
    except EvalConfigError as exc:
        # The candidate models are built without any list context, so they name
        # the field "candidates[]". Now that every candidate is reported in one
        # pass, two faults would otherwise arrive under the same field path.
        indexed = f"candidates[{index}]"
        exc.field = exc.field.replace("candidates[]", indexed)
        exc.args = (str(exc.args[0]).replace("candidates[]", indexed),)
        raise


def _refuse_capabilities_the_declared_schema_lacks(
    candidates: Sequence[EvalCandidate],
    schema_version: str,
    violations: _Violations,
) -> None:
    """Hold a config to the version it declares, not to the newest one this build reads.

    The candidate models accept everything 1.2 allows, because they are also
    built by callers that have no config version at all. A file that says 1.1 is
    a promise to every other reader of 1.1 that it contains nothing they would
    refuse, so the two widenings 1.2 introduced are refused here rather than
    silently accepted under the older label.
    """
    if schema_version != "1.1":
        return
    for index, candidate in enumerate(candidates):
        field = f"candidates[{index}].model_identity"
        identity = candidate.model_identity
        if identity.assurance != "weights_pinned":
            violations.add(
                CandidateIdentityError(
                    field,
                    "neither revision nor weights_digest is set, and schema 1.1 required every candidate to "
                    "pin the weights it was scored on",
                    expected="revision (immutable) or weights_digest, under schema 1.1",
                    recovery='set schema_version: "1.2" to record a provider-managed candidate and accept a '
                    "non-publishable run, or pin the revision or digest the endpoint serves",
                )
            )
        digest = identity.weights_digest
        if digest is not None and not digest.startswith("sha256:"):
            violations.add(
                CandidateIdentityError(
                    field,
                    f"weights_digest names the {digest.split(':', 1)[0]} scheme, which schema 1.1 cannot read",
                    expected="a sha256:<64 hex> digest under schema 1.1",
                    recovery=f'set schema_version: "1.2", which reads {WEIGHT_MANIFEST_DIGEST_SCHEME} digests, '
                    "or record the sha256 digest of the served weights",
                )
            )


def _resolve_outputs(data: Mapping[str, Any], base_dir: Path, source: EvalSource) -> EvalOutputConfig:
    section = _section(data, "outputs", _OUTPUT_KEYS)
    output_dir = _resolve(section["output_dir"], "outputs.output_dir", base_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise EvalConfigPathError(
            "outputs.output_dir",
            "exists but is not a directory",
            value=str(output_dir),
            expected="a directory path",
            recovery="choose a new eval output directory, or remove the file occupying this path",
        )
    publication_dir = source.publication_dir
    if output_dir == publication_dir or _within(output_dir, publication_dir) or _within(publication_dir, output_dir):
        raise EvalConfigPathError(
            "outputs.output_dir",
            f"overlaps the source publication tree at {publication_dir}",
            value=str(output_dir),
            expected="a directory outside the generation run's output tree",
            recovery="write eval artifacts to their own directory; an eval run must never be able to overwrite "
            "run_manifest.json or the benchmark it is scoring",
        )
    for artifact in _PUBLICATION_ARTIFACTS:
        if (output_dir / artifact).exists():
            raise EvalConfigPathError(
                "outputs.output_dir",
                f"already holds a generation artifact ({artifact})",
                value=str(output_dir),
                expected="an empty directory, or one holding only eval artifacts",
                recovery="choose a directory that is not a published generation tree",
            )
    return _model(EvalOutputConfig, {**section, "output_dir": output_dir}, "outputs")


def _within(path: Path, parent: Path) -> bool:
    return parent in path.parents


def load_eval_config_mapping(
    data: Mapping[str, Any],
    *,
    base_dir: Path,
    origin: str = "eval config",
) -> BfclEvalConfig:
    """Validate and resolve one already-parsed eval config mapping.

    ``base_dir`` is the directory relative paths resolve against: the eval
    config's own directory for a standalone file, and the BYOB root for a legacy
    block inlined into a generation config, matching how generation resolves its
    own paths.
    """
    data = _require_mapping(data, origin, recovery="write the eval config as a YAML mapping")
    unknown = sorted(set(data) - EVAL_CONFIG_TOP_LEVEL_KEYS)
    if unknown:
        raise EvalConfigSchemaError(
            origin,
            f"unknown top-level key(s): {', '.join(unknown)}",
            expected=f"only {', '.join(sorted(EVAL_CONFIG_TOP_LEVEL_KEYS))}",
            recovery="remove the unknown key; generation settings stay in the generation config",
        )

    schema_version = data.get("schema_version")
    if schema_version not in EVAL_CONFIG_SCHEMA_VERSIONS:
        raise EvalConfigSchemaError(
            "schema_version",
            "is not an eval config schema this build reads",
            value=schema_version,
            expected="one of " + ", ".join(f'"{version}"' for version in EVAL_CONFIG_SCHEMA_VERSIONS),
            recovery=f'set schema_version: "{EVAL_CONFIG_SCHEMA_VERSION}" (quoted, so YAML keeps it a string)',
        )
    config_status = data.get("config_status")
    if config_status == "template":
        raise EvalConfigSchemaError(
            "config_status",
            "a template is not runnable: it names placeholders instead of a source run and candidates",
            value=config_status,
            expected="resolved",
            recovery="copy the template, fill in every REPLACE_ME_* value, then set config_status: resolved",
        )
    if config_status != "resolved":
        raise EvalConfigSchemaError(
            "config_status",
            "must state whether the config is a template or a resolved run input",
            value=config_status,
            expected="'resolved' to run, 'template' for an unfilled example",
            recovery="set config_status: resolved once every value is filled in",
        )
    _reject_placeholders(data)
    _reject_secrets(data)

    # These sections constrain different things, so one being wrong says nothing
    # about the rest. Reading all of them before refusing turns a config with
    # three faults into one report instead of three runs.
    violations = _Violations()
    source = violations.run(lambda: _resolve_source(data, base_dir))
    settings = violations.run(lambda: _resolve_eval_settings(data))
    held_out_settings = violations.run(lambda: _resolve_held_out_eval(data))
    scoring = violations.run(lambda: _resolve_scoring(data, base_dir))
    limits = violations.run(lambda: _model(EvalLimits, _section(data, "limits", _LIMITS_KEYS), "limits"))
    candidates = _resolve_candidates(data, violations)
    _refuse_capabilities_the_declared_schema_lacks(candidates, str(schema_version), violations)
    contamination = violations.run(
        lambda: _model(
            ContaminationPolicy,
            _section(data, "contamination", _CONTAMINATION_KEYS),
            "contamination",
        )
    )
    publication = violations.run(
        lambda: _model(
            EvalPublicationPolicy,
            _section(data, "publication", _PUBLICATION_KEYS),
            "publication",
        )
    )
    # Outputs resolve against the source run, so a broken source has nothing to
    # say here and the section is left to the next run.
    outputs = violations.run(lambda: _resolve_outputs(data, base_dir, source)) if source is not None else None
    violations.raise_if_any()

    return _model(
        BfclEvalConfig,
        {
            "schema_version": schema_version,
            "config_status": "resolved",
            "source": source,
            "settings": settings,
            "held_out_eval": held_out_settings,
            "scoring": scoring,
            "limits": limits,
            "candidates": candidates,
            "contamination": contamination,
            "publication": publication,
            "outputs": outputs,
        },
        origin,
    )


def load_eval_config(path: str | Path, *, base_dir: Path | None = None) -> BfclEvalConfig:
    """Load, validate, resolve, and freeze one ``eval_config.yaml``."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (base_dir or Path.cwd()) / config_path
    config_path = config_path.resolve()
    _existing_file(
        config_path,
        "eval_config_path",
        recovery="point at the eval config YAML file",
    )
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvalConfigSchemaError(
            "eval_config_path",
            f"could not be parsed as YAML: {type(exc).__name__}",
            value=str(config_path),
            expected="a readable YAML document",
            recovery="fix the YAML syntax reported by the parser",
        ) from exc
    return load_eval_config_mapping(data, base_dir=config_path.parent, origin=str(config_path))


def eval_config_reference(config: BfclConfig) -> tuple[str, Any] | None:
    """Say where a generation config points for its eval input, if anywhere.

    Returns ``("path", Path)`` for ``eval_config_path`` or ``("inline", mapping)``
    for the legacy inline block. Carrying both is refused rather than resolved by
    precedence: two eval configs in one file cannot both be the one that ran.
    """
    inline = config.inline_eval
    declared_path = config.eval_config_path
    if inline is not None and declared_path is not None:
        raise EvalConfigSchemaError(
            "eval",
            "the generation config carries both eval_config_path and an inline eval block",
            expected="exactly one eval config input",
            recovery="keep eval_config_path and delete the inline eval block; a standalone eval config keeps "
            "candidate changes out of generation lineage",
        )
    if declared_path is not None:
        return ("path", Path(declared_path))
    if inline is not None:
        return ("inline", inline)
    return None


def load_eval_config_for_generation(config: BfclConfig) -> BfclEvalConfig | None:
    """Resolve the eval config a generation config references, if it references one.

    Both inputs go through the same validation: there is one implementation of
    what an eval config means, so the legacy inline form cannot drift into a
    second dialect.
    """
    reference = eval_config_reference(config)
    if reference is None:
        return None
    kind, value = reference
    if kind == "path":
        return load_eval_config(value, base_dir=BYOB_ROOT)
    return load_eval_config_mapping(value, base_dir=BYOB_ROOT, origin="eval (inline in generation config)")


def resolved_eval_config_document(config: BfclEvalConfig) -> dict[str, Any]:
    """Auditable JSON view: the semantic payload plus where the files came from.

    The paths live in a separate ``resolved_paths`` block, outside the hashed
    payload, so a reader can see what was opened without being misled into
    thinking a directory move changed the evaluation.
    """
    return {
        "schema_version": config.schema_version,
        "config_status": config.config_status,
        "eval_config_hash": config.eval_config_hash,
        "publication_allowed": config.publication_allowed,
        "publication_scope": config.publication_scope,
        "non_publication_reasons": list(config.non_publication_reasons),
        "candidate_aliases": list(config.candidate_aliases),
        "semantic_payload": config.semantic_payload(),
        "resolved_paths": {
            "source_run_manifest": str(config.source.run_manifest.path),
            "benchmark": str(config.source.benchmark.path),
            "translation_manifest": (
                str(config.source.translation_manifest.path)
                if config.source.translation_manifest is not None
                else None
            ),
            "oracle_pack_manifest": (
                str(config.source.oracle.pack_manifest.path) if config.source.oracle is not None else None
            ),
            "oracle_execution_resource": (
                str(config.source.oracle.execution_resource.path) if config.source.oracle is not None else None
            ),
            "scoring_contract": str(config.scoring.contract.path),
            "output_dir": str(config.outputs.output_dir),
        },
    }


def write_resolved_eval_config(config: BfclEvalConfig, path: str | Path) -> str:
    """Write the audit document deterministically and return its content hash."""
    output_root = config.outputs.output_dir.resolve()
    target = Path(path)
    if not target.is_absolute():
        target = output_root / target
    target = target.resolve()
    if target == output_root or not _within(target, output_root):
        raise EvalConfigPathError(
            "resolved_eval_config.path",
            "is outside outputs.output_dir",
            value=str(target),
            expected=f"a file below the configured eval output directory ({output_root})",
            recovery="write resolved_eval_config.json inside outputs.output_dir",
        )
    publication_dir = config.source.publication_dir.resolve()
    if target == publication_dir or _within(target, publication_dir):
        raise EvalConfigPathError(
            "resolved_eval_config.path",
            "overlaps the source publication tree",
            value=str(target),
            expected="a file in the isolated eval output tree",
            recovery="never write eval audit artifacts beside run_manifest.json or the source benchmark",
        )
    if target.exists() and target.is_dir():
        raise EvalConfigPathError(
            "resolved_eval_config.path",
            "is a directory rather than a file",
            value=str(target),
            expected="a JSON file path",
            recovery="append a file name such as resolved_eval_config.json",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(resolved_eval_config_document(config), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def describe_eval_config_error(exc: Exception) -> str:
    """One-line, secret-free summary for a CLI or a step report."""
    if isinstance(exc, EvalConfigError):
        report = exc.as_report()
        summary = f"[{report['code']}] {report['field']}: {report['problem']}"
        # A one-line summary that named only the first violation would send the
        # operator back for another run over problems already found. Fields are
        # paths, so listing them adds no value the redaction rules exclude.
        if exc.other_violations:
            fields = ", ".join(other.field for other in exc.other_violations)
            summary += f" (and {len(exc.other_violations)} more: {fields})"
        return summary
    return f"[eval_config_invalid] {type(exc).__name__}: {redact_value(str(exc))}"
