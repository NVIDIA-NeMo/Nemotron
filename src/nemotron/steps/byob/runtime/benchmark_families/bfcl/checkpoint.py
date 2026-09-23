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

"""Fail-closed, versioned checkpoints for verified BFCL generation resume."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.runtime_metadata import runtime_metadata
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
    generation_config_hash,
)

CHECKPOINT_CONTRACT = "bfcl-generation-checkpoint/1.0"
STATE_CONTRACT = "bfcl-generation-state/1.0"
CHECKPOINT_DIRECTORY = "checkpoints"

CANONICAL_STAGES = (
    "reference_profile",
    "expand",
    "state_machine",
    "render",
    "expected_trace",
    "schema_validation",
    "executable_replay",
    "surface_quality",
    "dedup_balancing",
    "final_output",
)
STAGE_CONTRACT_VERSIONS = {stage: "1.0" for stage in CANONICAL_STAGES}
_STAGE_MUTABLE_OUTPUTS = {
    "expand": {
        "stage_cache/task_instances.parquet",
        "stage_cache/held_out_bindings.json",
    },
    "state_machine": {"stage_cache/conversation_plans.parquet"},
    "render": {
        "stage_cache/task_instances.parquet",
        "stage_cache/conversation_plans.parquet",
        "stage_cache/rendered_conversations.parquet",
        "stage_cache/paraphrase_rejections.json",
        "stage_cache/paraphrase_io_cache.jsonl",
    },
    "expected_trace": {"stage_cache/expected_traces.parquet"},
    "schema_validation": {"stage_cache/schema_validated_traces.parquet"},
    "executable_replay": {
        "stage_cache/replay_validated_tasks.parquet",
        "stage_cache/paraphrase_rejections.json",
        "stage_cache/rendered_conversations.parquet",
    },
    "surface_quality": {
        "stage_cache/surface_validated_tasks.parquet",
        "stage_cache/surface_quality_rejections.json",
        "stage_cache/surface_judge_cache_usage.json",
    },
    "dedup_balancing": {
        "stage_cache/balanced_tasks.parquet",
        "stage_cache/dedup_balancing_report.json",
    },
    "final_output": set(),
}
# Model I/O caches are append-only and shared across runs, so a resumed stage has to
# replay the recorded responses instead of paying for new ones. Treating them as stale
# stage output would make a resumed run render different surfaces than the original.
APPEND_ONLY_CACHES = frozenset(
    {
        "reference_profile_io_cache.jsonl",
        "paraphrase_io_cache.jsonl",
        "surface_judge_io_cache.jsonl",
    }
)
APPEND_ONLY_CACHE_LOCKS = frozenset(f".{name}.lock" for name in APPEND_ONLY_CACHES)
_STAGE_STALE_CACHE_OUTPUTS = {
    stage: frozenset(
        name for name in (source.removeprefix("stage_cache/") for source in sources) if name not in APPEND_ONLY_CACHES
    )
    for stage, sources in _STAGE_MUTABLE_OUTPUTS.items()
}
_STAGE_STALE_CACHE_OUTPUTS["final_output"] = frozenset({"held_out_scan.json"})
_REQUIRED_PUBLICATION_SOURCES = {
    "publication/benchmark_raw.parquet",
    "publication/benchmark.parquet",
    "publication/run_manifest.json",
}
_REQUIRED_STATE_KEYS = {
    "reference_profile": {"profile"},
    "expand": {"profile", "tasks", "canonical_expanded"},
    "state_machine": {"profile", "tasks", "canonical_expanded", "plans"},
    "render": {
        "profile",
        "tasks",
        "canonical_expanded",
        "plans",
        "surfaces",
        "prompt_bundle",
        "paraphrase_report",
        "expanded_tasks",
        "expanded",
    },
    "expected_trace": {"traces", "drop_reasons"},
    "schema_validation": {"schema_failures"},
    "executable_replay": {"verdicts"},
    "surface_quality": {"surface_quality_records", "surface_quality_report"},
    "dedup_balancing": {"dedup_balancing_result"},
    "final_output": {"stage_counts"},
}


class CheckpointError(RuntimeError):
    """A checkpoint is absent, corrupt, incompatible, or no longer verified."""


def enabled_stages(config: Any) -> tuple[str, ...]:
    stages = list(CANONICAL_STAGES[:7])
    if config.surface_quality_validation.get("enabled"):
        stages.append("surface_quality")
    if config.semantic_deduplication_config.get("enabled"):
        stages.append("dedup_balancing")
    stages.append("final_output")
    return tuple(stages)


def validate_resume_target(config: Any, target: str) -> tuple[str, ...]:
    if target not in CANONICAL_STAGES:
        raise CheckpointError(f"unknown BFCL resume stage {target!r}; choose one of: " + ", ".join(CANONICAL_STAGES))
    stages = enabled_stages(config)
    if target not in stages:
        raise CheckpointError(f"BFCL resume stage {target!r} is disabled by the current configuration")
    return stages


def predecessor(config: Any, stage: str) -> str | None:
    stages = validate_resume_target(config, stage)
    index = stages.index(stage)
    return stages[index - 1] if index else None


def checkpoints_dir(config: Any) -> Path:
    return stage_cache_dir(config) / CHECKPOINT_DIRECTORY


def clear_checkpoints(config: Any) -> None:
    shutil.rmtree(checkpoints_dir(config), ignore_errors=True)


def clear_from_stage(config: Any, stage: str) -> None:
    stages = enabled_stages(config)
    root = checkpoints_dir(config)
    for name in stages[stages.index(stage) :]:
        shutil.rmtree(root / name, ignore_errors=True)


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _json_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_shape(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        shapes = {_hash_bytes(canonical_json(_json_shape(child)).encode()): _json_shape(child) for child in value}
        return {"list": [shapes[key] for key in sorted(shapes)]}
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise CheckpointError(f"checkpoint JSON contains unsupported value {type(value).__name__}")


def _json_schema_fingerprint(value: Any) -> str:
    return _hash_bytes(canonical_json(_json_shape(value)).encode("utf-8"))


def _task_ids_from_state(state: dict[str, Any]) -> list[str]:
    tasks = state.get("tasks")
    if tasks is None:
        return []
    if not isinstance(tasks, list):
        raise CheckpointError("checkpoint state.tasks must be a list")
    ids: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or not isinstance(task.get("task_id"), str):
            raise CheckpointError(f"checkpoint state.tasks[{index}] has no string task_id")
        ids.append(task["task_id"])
    if len(ids) != len(set(ids)):
        raise CheckpointError("checkpoint state contains duplicate task IDs")
    return ids


def _json_safe(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return value


def _task_set_hash(task_ids: list[str]) -> str:
    return _hash_bytes(canonical_json(task_ids).encode("utf-8"))


def _expected_parquet_schema(source: str) -> Any | None:
    name = Path(source).name
    if name in {"benchmark.parquet", "benchmark_raw.parquet"}:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
            benchmark_schema,
        )

        return benchmark_schema()
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import stage_tables

    factories = {
        stage_tables.REFERENCE_SAMPLES: stage_tables.reference_samples_schema,
        stage_tables.TASK_INSTANCES: stage_tables.task_instances_schema,
        stage_tables.CONVERSATION_PLANS: stage_tables.conversation_plans_schema,
        stage_tables.RENDERED_CONVERSATIONS: stage_tables.rendered_conversations_schema,
        stage_tables.EXPECTED_TRACES: stage_tables.expected_traces_schema,
        stage_tables.SCHEMA_VALIDATED_TRACES: stage_tables.schema_validated_traces_schema,
        stage_tables.REPLAY_VALIDATED_TASKS: stage_tables.replay_validated_tasks_schema,
        stage_tables.SURFACE_VALIDATED_TASKS: stage_tables.surface_validated_tasks_schema,
        stage_tables.BALANCED_TASKS: stage_tables.balanced_tasks_schema,
    }
    factory = factories.get(name)
    return factory() if factory is not None else None


def _artifact_metadata(path: Path, *, source: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": source,
        "content_hash": _hash_file(path),
        "size": path.stat().st_size,
        "format": path.suffix.removeprefix(".") or "binary",
        "row_count": None,
        "schema_fingerprint": None,
        "task_ids": None,
        "task_set_hash": None,
    }
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"checkpoint JSON artifact is invalid: {path}") from exc
        metadata["schema_fingerprint"] = _json_schema_fingerprint(value)
    elif path.suffix == ".jsonl":
        values = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    values.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"checkpoint JSONL artifact is invalid: {path}") from exc
        metadata["row_count"] = len(values)
        metadata["schema_fingerprint"] = _json_schema_fingerprint(values)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        try:
            table = pq.read_table(path)
        except Exception as exc:  # noqa: BLE001 - normalize corrupt parquet failures
            raise CheckpointError(f"checkpoint Parquet artifact is invalid: {path}") from exc
        expected_schema = _expected_parquet_schema(source)
        if expected_schema is not None and not table.schema.equals(expected_schema):
            raise CheckpointError(f"checkpoint Parquet schema does not match its stage contract: {source}")
        metadata["row_count"] = table.num_rows
        metadata["schema_fingerprint"] = _hash_bytes(table.schema.serialize().to_pybytes())
        if "task_id" in table.column_names:
            ids = table.column("task_id").to_pylist()
            if any(not isinstance(item, str) or not item for item in ids):
                raise CheckpointError(f"checkpoint artifact has invalid task_id values: {path}")
            if len(ids) != len(set(ids)):
                raise CheckpointError(f"checkpoint artifact has duplicate task IDs: {path}")
            metadata["task_ids"] = ids
            metadata["task_set_hash"] = _task_set_hash(ids)
    return metadata


def _validate_publication_declarations(
    artifact_dir: Path,
    artifacts: Mapping[str, dict[str, Any]],
) -> None:
    by_source = {str(metadata["source"]): (name, metadata) for name, metadata in artifacts.items()}
    manifest_entry = by_source.get("publication/run_manifest.json")
    if manifest_entry is None:
        raise CheckpointError("BFCL Stage 12 checkpoint has no run manifest")
    try:
        run_manifest = json.loads((artifact_dir / manifest_entry[0]).read_text(encoding="utf-8"))
        exports = run_manifest["exports"]
        references = []
        validation_report = exports.get("validation_report")
        if validation_report is not None:
            references.append(validation_report)
        for item in exports["formats"].values():
            if item.get("enabled"):
                references.append(item)
        for reference in references:
            path = reference["path"]
            declared_hash = reference["content_hash"]
            entry = by_source.get(f"publication/{path}")
            if entry is not None:
                actual_hashes = {entry[1]["content_hash"]}
            else:
                prefix = f"publication/{path.rstrip('/')}/"
                members = {
                    source.removeprefix("publication/"): (artifact_dir / name).read_bytes()
                    for source, (name, _metadata) in by_source.items()
                    if source.startswith(prefix)
                }
                if not members:
                    actual_hashes = set()
                else:
                    from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
                        export_content_hash,
                    )

                    root_prefix = f"{path.rstrip('/')}/"
                    relative_members = {
                        member.removeprefix(root_prefix): payload for member, payload in members.items()
                    }
                    actual_hashes = {
                        export_content_hash(members),
                        export_content_hash(relative_members),
                    }
            if declared_hash not in actual_hashes:
                raise CheckpointError(f"BFCL Stage 12 checkpoint export is missing or changed: {path}")
    except CheckpointError:
        raise
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("BFCL Stage 12 run manifest has an invalid export contract") from exc


def _identity(
    config: Any,
    *,
    pack_fingerprint: str,
    endpoint_metadata: dict[str, str] | None,
    endpoint_config_path: Path | None,
) -> dict[str, Any]:
    endpoint_config_hash = (
        _hash_file(endpoint_config_path)
        if endpoint_config_path is not None and endpoint_config_path.is_file()
        else None
    )
    endpoint_content_digest = None
    if endpoint_metadata is not None:
        endpoint_content_digest = _hash_bytes(canonical_json(endpoint_metadata).encode("utf-8"))
    runtime = runtime_metadata()
    source_hash = runtime.get("pipeline_source_hash")
    if not isinstance(source_hash, str) or not source_hash.startswith("sha256:"):
        raise CheckpointError("BFCL checkpoint cannot establish the current pipeline source identity")
    if endpoint_config_path is not None and endpoint_metadata is None:
        raise CheckpointError("BFCL endpoint checkpoint requires verified endpoint identity metadata")
    return {
        "generation_config_hash": generation_config_hash(config),
        "pack_fingerprint": (
            pack_fingerprint if pack_fingerprint.startswith("sha256:") else f"sha256:{pack_fingerprint}"
        ),
        "endpoint": {
            "identity": endpoint_metadata,
            "identity_digest": endpoint_content_digest,
            "config_content_hash": endpoint_config_hash,
        },
        "pipeline": {
            "git_sha": runtime.get("pipeline_git_sha"),
            "source_hash": source_hash,
            "dependency_lock_hash": runtime.get("dependency_lock_hash"),
            "worker_image_digest": runtime.get("worker_image_digest"),
        },
    }


def current_identity(
    config: Any,
    *,
    pack_fingerprint: str,
    endpoint_metadata: dict[str, str] | None,
    endpoint_config_path: Path | None,
) -> dict[str, Any]:
    return _identity(
        config,
        pack_fingerprint=pack_fingerprint,
        endpoint_metadata=endpoint_metadata,
        endpoint_config_path=endpoint_config_path,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointError(f"required BFCL checkpoint is missing: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"BFCL checkpoint manifest is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"BFCL checkpoint manifest must be an object: {path}")
    required = {
        "contract",
        "stage",
        "stage_index",
        "stage_contract_version",
        "dependency",
        "identity",
        "state",
        "task_ids",
        "task_count",
        "task_set_hash",
        "input_artifacts",
        "output_artifacts",
        "produced_artifacts",
        "parent",
        "checkpoint_id",
    }
    if set(value) != required:
        raise CheckpointError(f"BFCL checkpoint manifest fields do not match contract at {path}")
    body = {key: child for key, child in value.items() if key != "checkpoint_id"}
    if value["checkpoint_id"] != _hash_bytes(canonical_json(body).encode("utf-8")):
        raise CheckpointError(f"BFCL checkpoint manifest identity mismatch: {path}")
    return value


def write_checkpoint(
    config: Any,
    stage: str,
    state: dict[str, Any],
    *,
    identity: dict[str, Any],
    artifact_names: Sequence[str],
    publication_paths: tuple[Path, ...] = (),
    publication_root: Path | None = None,
) -> Path:
    state = _json_safe(state)
    validate_resume_target(config, stage)
    required_state = set().union(
        *(
            _REQUIRED_STATE_KEYS[name]
            for name in CANONICAL_STAGES[: CANONICAL_STAGES.index(stage) + 1]
            if name in enabled_stages(config)
        )
    )
    if missing := sorted(required_state - set(state)):
        raise CheckpointError(f"BFCL checkpoint state for {stage!r} is incomplete: {', '.join(missing)}")
    expected_parent = predecessor(config, stage)
    root = checkpoints_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    parent = None
    input_artifacts: dict[str, Any] = {}
    if expected_parent is not None:
        parent_manifest_path = root / expected_parent / "manifest.json"
        parent_manifest = _read_manifest(parent_manifest_path)
        input_artifacts = parent_manifest["output_artifacts"]
        parent = {
            "stage": expected_parent,
            "checkpoint_id": parent_manifest["checkpoint_id"],
        }

    destination = root / stage
    temporary = root / f".{stage}-{uuid.uuid4().hex}.tmp"
    snapshots = temporary / "artifacts"
    snapshots.mkdir(parents=True)
    try:
        artifact_paths: list[tuple[Path, str]] = []
        cache = stage_cache_dir(config)
        for name in sorted(set(artifact_names)):
            path = cache / name
            if not path.is_file() or path.is_symlink():
                raise CheckpointError(f"BFCL checkpoint output artifact is missing or unsafe: {path}")
            artifact_paths.append((path, f"stage_cache/{path.name}"))
        for path in publication_paths:
            if not path.is_file() or path.is_symlink():
                raise CheckpointError(f"BFCL checkpoint publication artifact is missing or unsafe: {path}")
            try:
                relative = path.relative_to(publication_root) if publication_root is not None else Path(path.name)
            except ValueError as exc:
                raise CheckpointError(f"BFCL publication artifact is outside its staging root: {path}") from exc
            artifact_paths.append((path, f"publication/{relative.as_posix()}"))

        artifacts: dict[str, Any] = {}
        for source_path, source in artifact_paths:
            snapshot_name = f"{hashlib.sha256(source.encode('utf-8')).hexdigest()}-{source_path.name}"
            snapshot_path = snapshots / snapshot_name
            shutil.copy2(source_path, snapshot_path)
            artifacts[snapshot_name] = _artifact_metadata(snapshot_path, source=source)
        artifacts_by_source = {str(metadata["source"]): metadata for metadata in artifacts.values()}
        inputs_by_source = {str(metadata["source"]): metadata for metadata in input_artifacts.values()}
        changed_sources = {
            source
            for source in set(inputs_by_source) | set(artifacts_by_source)
            if inputs_by_source.get(source) != artifacts_by_source.get(source)
        }
        allowed_changes = _STAGE_MUTABLE_OUTPUTS.get(stage, set())
        if stage == "reference_profile":
            allowed_changes = changed_sources
        elif stage == "final_output":
            allowed_changes = {source for source in artifacts_by_source if source.startswith("publication/")}
            if not _REQUIRED_PUBLICATION_SOURCES <= set(artifacts_by_source):
                missing = sorted(_REQUIRED_PUBLICATION_SOURCES - set(artifacts_by_source))
                raise CheckpointError(
                    "BFCL Stage 12 checkpoint is missing publication artifacts: " + ", ".join(missing)
                )
            _validate_publication_declarations(snapshots, artifacts)
        if unexpected := sorted(changed_sources - allowed_changes):
            raise CheckpointError(f"BFCL stage {stage!r} changed inherited artifacts: " + ", ".join(unexpected))

        task_ids = _task_ids_from_state(state)
        state_payload = {
            "contract": STATE_CONTRACT,
            "stage": stage,
            "task_ids": task_ids,
            "task_set_hash": _task_set_hash(task_ids),
            "payload": state,
        }
        state_path = temporary / "state.json"
        state_path.write_text(
            json.dumps(state_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        state_meta = _artifact_metadata(state_path, source="state.json")
        body = {
            "contract": CHECKPOINT_CONTRACT,
            "stage": stage,
            "stage_index": CANONICAL_STAGES.index(stage) + 3,
            "stage_contract_version": STAGE_CONTRACT_VERSIONS[stage],
            "dependency": expected_parent,
            "identity": identity,
            "state": state_meta,
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "task_set_hash": _task_set_hash(task_ids),
            "input_artifacts": input_artifacts,
            "output_artifacts": artifacts,
            "produced_artifacts": sorted(changed_sources),
            "parent": parent,
        }
        manifest = {
            **body,
            "checkpoint_id": _hash_bytes(canonical_json(body).encode("utf-8")),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise CheckpointError(f"BFCL checkpoint destination already exists: {destination}")
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination / "manifest.json"


def _validate_artifact(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise CheckpointError(f"checkpoint artifact is missing or unsafe: {path}")
    actual = _artifact_metadata(path, source=str(expected.get("source")))
    if actual != expected:
        raise CheckpointError(f"checkpoint artifact metadata mismatch: {path}")


def _validate_chain(
    config: Any,
    stage: str,
    *,
    identity: dict[str, Any],
    seen: set[str],
) -> dict[str, Any]:
    if stage in seen:
        raise CheckpointError("BFCL checkpoint parent chain contains a cycle")
    seen.add(stage)
    manifest_path = checkpoints_dir(config) / stage / "manifest.json"
    manifest = _read_manifest(manifest_path)
    expected_parent = predecessor(config, stage)
    if (
        manifest["contract"] != CHECKPOINT_CONTRACT
        or manifest["stage"] != stage
        or manifest["stage_index"] != CANONICAL_STAGES.index(stage) + 3
        or manifest["stage_contract_version"] != STAGE_CONTRACT_VERSIONS[stage]
        or manifest["dependency"] != expected_parent
        or manifest["identity"] != identity
    ):
        raise CheckpointError(f"BFCL checkpoint is incompatible with this run: {manifest_path}")
    expected_parent_ref = manifest["parent"]
    if expected_parent is None:
        if expected_parent_ref is not None or manifest["input_artifacts"]:
            raise CheckpointError("root BFCL checkpoint unexpectedly declares a parent")
    else:
        parent_manifest = _validate_chain(config, expected_parent, identity=identity, seen=seen)
        if expected_parent_ref != {
            "stage": expected_parent,
            "checkpoint_id": parent_manifest["checkpoint_id"],
        }:
            raise CheckpointError(f"BFCL checkpoint parent identity mismatch: {manifest_path}")
        if manifest["input_artifacts"] != parent_manifest["output_artifacts"]:
            raise CheckpointError(f"BFCL checkpoint input lineage mismatch: {manifest_path}")
    inputs_by_source = {str(metadata["source"]): metadata for metadata in manifest["input_artifacts"].values()}
    outputs_by_source = {str(metadata["source"]): metadata for metadata in manifest["output_artifacts"].values()}
    changed_sources = {
        source
        for source in set(inputs_by_source) | set(outputs_by_source)
        if inputs_by_source.get(source) != outputs_by_source.get(source)
    }
    allowed_changes = _STAGE_MUTABLE_OUTPUTS.get(stage, set())
    if stage == "reference_profile":
        allowed_changes = changed_sources
    elif stage == "final_output":
        allowed_changes = {source for source in outputs_by_source if source.startswith("publication/")}
        if not _REQUIRED_PUBLICATION_SOURCES <= set(outputs_by_source):
            raise CheckpointError("BFCL Stage 12 checkpoint publication set is incomplete")
    if manifest["produced_artifacts"] != sorted(changed_sources) or changed_sources - allowed_changes:
        raise CheckpointError(f"BFCL checkpoint output lineage mismatch: {manifest_path}")
    directory = manifest_path.parent
    _validate_artifact(directory / "state.json", manifest["state"])
    artifact_dir = directory / "artifacts"
    actual_names = sorted(path.name for path in artifact_dir.iterdir()) if artifact_dir.is_dir() else []
    if actual_names != sorted(manifest["output_artifacts"]):
        raise CheckpointError(f"BFCL checkpoint artifact set mismatch: {artifact_dir}")
    for name, expected in manifest["output_artifacts"].items():
        _validate_artifact(artifact_dir / name, expected)
    if stage == "final_output":
        _validate_publication_declarations(artifact_dir, manifest["output_artifacts"])
    return manifest


def restore_predecessor(
    config: Any,
    target: str,
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    required = predecessor(config, target)
    if required is None:
        return {}
    manifest = _validate_chain(config, required, identity=identity, seen=set())
    directory = checkpoints_dir(config) / required
    try:
        state_payload = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("BFCL checkpoint state is not valid JSON") from exc
    if (
        not isinstance(state_payload, dict)
        or set(state_payload) != {"contract", "stage", "task_ids", "task_set_hash", "payload"}
        or state_payload["contract"] != STATE_CONTRACT
        or state_payload["stage"] != required
        or not isinstance(state_payload["payload"], dict)
    ):
        raise CheckpointError("BFCL checkpoint state does not match its contract")
    task_ids = _task_ids_from_state(state_payload["payload"])
    if (
        state_payload["task_ids"] != task_ids
        or state_payload["task_set_hash"] != _task_set_hash(task_ids)
        or manifest["task_ids"] != task_ids
        or manifest["task_count"] != len(task_ids)
        or manifest["task_set_hash"] != _task_set_hash(task_ids)
    ):
        raise CheckpointError("BFCL checkpoint state task IDs, order, or count changed")
    required_state = set().union(
        *(
            _REQUIRED_STATE_KEYS[name]
            for name in CANONICAL_STAGES[: CANONICAL_STAGES.index(required) + 1]
            if name in enabled_stages(config)
        )
    )
    if missing := sorted(required_state - set(state_payload["payload"])):
        raise CheckpointError(f"BFCL checkpoint state for {required!r} is incomplete: {', '.join(missing)}")

    cache = stage_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    restored = {
        str(meta["source"]).removeprefix("stage_cache/"): snapshot_name
        for snapshot_name, meta in manifest["output_artifacts"].items()
        if str(meta["source"]).startswith("stage_cache/")
    }
    # Only an output of a stage that runs again can be stale. Removing anything else
    # would delete the append-only caches and evidence a resumed run still depends on.
    stages = enabled_stages(config)
    stale = {
        name for stage in stages[stages.index(target) :] for name in _STAGE_STALE_CACHE_OUTPUTS.get(stage, frozenset())
    }
    for name in sorted(stale - set(restored)):
        (cache / name).unlink(missing_ok=True)
    for name, snapshot_name in sorted(restored.items()):
        shutil.copy2(directory / "artifacts" / snapshot_name, cache / name)
    return state_payload["payload"]


__all__ = [
    "APPEND_ONLY_CACHES",
    "CANONICAL_STAGES",
    "CHECKPOINT_CONTRACT",
    "STAGE_CONTRACT_VERSIONS",
    "CheckpointError",
    "clear_checkpoints",
    "clear_from_stage",
    "current_identity",
    "enabled_stages",
    "restore_predecessor",
    "validate_resume_target",
    "write_checkpoint",
]
