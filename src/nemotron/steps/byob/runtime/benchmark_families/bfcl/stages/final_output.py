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

"""Assemble messages, write the benchmark parquets, and stamp the run manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.bfcl_json_export import (
    BfclJsonArtifact,
    write_bfcl_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
    BYOB_ROOT,
    DEFAULT_BENCHMARK_SCHEMA_VERSION,
    EVAL_REFERENCE_KEYS,
    BfclConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    EXPORT_DIRECTORY,
    EXPORT_FORMATS,
    export_manifest_section,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    project_published_benchmark,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_validation import (
    EXPORT_VALIDATION_REPORT_FILE,
    ExportArtifact,
    validate_and_write_export_report,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NemoEvaluatorArtifact,
    write_nemo_evaluator_bundle,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.origin_provenance import (
    load_origin_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    PACK_FINGERPRINT_CONTRACT,
    LoadedPack,
    pack_file_hashes,
    pack_fingerprint,
    project_model_facing_tools,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    plan_publication,
    publication_manifest_section,
    verify_written_benchmarks,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
    encode_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.runtime_metadata import (
    runtime_metadata,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    BALANCED_TASKS,
    REFERENCE_SAMPLES,
    STAGE_TABLES,
    SURFACE_VALIDATED_TASKS,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.held_out import (
    HELD_OUT_BINDINGS,
    HELD_OUT_SCAN,
    enforce_no_leak,
    held_out_policy,
    load_binding_report,
    manifest_section,
    scan_rows,
    write_scan_report,
)

logger = logging.getLogger(__name__)


def _write_bfcl_json_export(projection: CanonicalExportProjection, output_dir: Path) -> BfclJsonArtifact:
    return write_bfcl_json(projection, output_dir)


def _write_nemo_evaluator_export(projection: CanonicalExportProjection, output_dir: Path) -> NemoEvaluatorArtifact:
    return write_nemo_evaluator_bundle(projection, output_dir)


# One place says which formats this pipeline can actually write. Config validation
# reads the same mapping, so a format declared in the contract but never wired to a
# writer is refused at startup instead of silently producing no file.
EXPORT_WRITERS: dict[str, Callable[[CanonicalExportProjection, Path], ExportArtifact]] = {
    "bfcl_json": _write_bfcl_json_export,
    "nemo_evaluator_bundle": _write_nemo_evaluator_export,
}
if set(EXPORT_WRITERS) - set(EXPORT_FORMATS):  # pragma: no cover - import-time contract
    raise RuntimeError("an export writer is registered for a format the export contract does not declare")


def _sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _file_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_parquet_atomic(table: Any, path: Path, parquet: Any) -> None:
    """Replace one parquet only after its temporary file is fully written."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        parquet.write_table(table, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _discard(*paths: Path) -> None:
    """Remove outputs from an attempt that must not be published.

    Directories as well as files, so an abandoned attempt cannot leave an export
    tree behind for the next reader to mistake for this run's output.
    """
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _commit_staged_publication(staging_dir: Path, output_dir: Path) -> Path:
    """Promote a complete Stage 12 tree, writing the manifest commit marker last."""
    manifest_name = "run_manifest.json"
    artifact_names = ("benchmark_raw.parquet", "benchmark.parquet", EXPORT_DIRECTORY)
    final_manifest = output_dir / manifest_name
    # A reader treats the manifest as the commit marker. Remove an older marker
    # before replacing any payload, even when this helper is called outside the
    # top-level pipeline invalidation path.
    final_manifest.unlink(missing_ok=True)
    try:
        for name in artifact_names:
            source = staging_dir / name
            destination = output_dir / name
            _discard(destination)
            if source.exists():
                source.replace(destination)
        (staging_dir / manifest_name).replace(final_manifest)
    except Exception:
        _discard(
            final_manifest,
            *(output_dir / name for name in artifact_names),
        )
        raise
    finally:
        _discard(staging_dir)
    return output_dir / "benchmark.parquet"


def _require_pack_fingerprint(
    pack: LoadedPack,
    expected: str,
    *,
    phase: str,
    cleanup: tuple[Path, ...] = (),
) -> str:
    """Abort publication if pack inputs drift, removing outputs from this attempt."""
    current = pack_fingerprint(pack.paths)
    if current == expected:
        return current
    _discard(*cleanup)
    raise RuntimeError(f"oracle pack changed {phase}; refusing to publish content that validation did not certify")


def _require_endpoint_identity(
    config: BfclConfig,
    pack: LoadedPack,
    expected: dict[str, Any] | None,
    *,
    cleanup: tuple[Path, ...] = (),
) -> None:
    """Abort publication when a remote oracle no longer matches validation."""
    if pack.endpoint_config is None:
        return
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

    runtime = config.oracle_runtime
    outputs = ProcessWorker(
        default_timeout_s=runtime.episode_timeout_s,
        worker=runtime.worker,
    ).run_episode(
        endpoint_config=pack.endpoint_config,
        fixtures=None,
        clock_iso=runtime.clock,
        seed=int(config.random_seed or 0),
        task_id="final-output-endpoint-metadata",
        steps=[{"op": "metadata"}],
        import_timeout_s=runtime.import_timeout_s,
        tool_timeout_s=runtime.tool_timeout_s,
        episode_timeout_s=runtime.episode_timeout_s,
    )
    if outputs[0] == expected:
        return
    _discard(*cleanup)
    raise RuntimeError(
        "endpoint identity changed during final output; refusing to publish "
        "artifacts from an uncertified oracle revision"
    )


def _write_endpoint_manifest_atomic(
    text: str,
    path: Path,
    *,
    config: BfclConfig,
    pack: LoadedPack,
    expected_endpoint_metadata: dict[str, Any] | None,
    cleanup: tuple[Path, ...],
) -> None:
    """Recheck endpoint identity immediately before replacing the manifest."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        _require_endpoint_identity(
            config,
            pack,
            expected_endpoint_metadata,
            cleanup=cleanup,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(
    value: Any,
    *,
    roots: tuple[tuple[str, Path], ...] | None = None,
) -> Any:
    """Normalize config-only Python values before provenance hashing.

    Paths are recorded relative to the framework root when they live under it, so the
    same config hashes the same on two machines with different checkouts.
    """
    if isinstance(value, Path):
        return _portable_path(value, roots=roots)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(child, roots=roots) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child, roots=roots) for child in value]
    if isinstance(value, str) and value.startswith("/"):
        return _portable_path(Path(value), roots=roots)
    return value


def _portable_path(
    path: Path,
    *,
    roots: tuple[tuple[str, Path], ...] | None = None,
) -> str:
    candidates = roots or (("<byob>", BYOB_ROOT),)
    # Roots arrive already resolved, so a config that spells a path through a symlink
    # only matches its own root in resolved form: macOS states /tmp while the root is
    # /private/tmp. Without the second form the same logical config hashes one way on
    # macOS and another on Linux, which is the drift this labelling exists to prevent.
    # A relative path is left alone, because resolving it would fold in the working
    # directory and make the hash depend on where the run was started from.
    forms = [path]
    if path.is_absolute() and (resolved := path.resolve()) != path:
        forms.append(resolved)
    for label, root in candidates:
        for form in forms:
            try:
                relative = form.relative_to(root)
            except ValueError:
                continue
            return label if relative == Path(".") else f"{label}/{relative.as_posix()}"
    return str(path)


def _config_roots(config: BfclConfig) -> tuple[tuple[str, Path], ...]:
    roots: list[tuple[str, Path]] = [
        ("<pack>", config.oracle_pack.manifest_path.parent),
        ("<output>", config.output_dir),
        ("<byob>", BYOB_ROOT),
    ]
    roots.extend(
        (f"<allowed_root_{index}>", root)
        for index, root in enumerate(config.oracle_runtime.allowed_roots)
        if root not in {config.oracle_pack.manifest_path.parent, config.output_dir, BYOB_ROOT}
    )
    return tuple(roots)


def _without_eval_reference(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the eval inputs a generation config only carries.

    Generation lineage answers "what produced these rows". A candidate model, its
    revision, or an eval config's location answers "who was scored on them", and
    letting either move the generation hash would make a benchmark look like a
    different benchmark every time someone evaluated a new model on it.
    """
    return {key: value for key, value in payload.items() if key not in EVAL_REFERENCE_KEYS and key != "inline_eval"}


def _resolved_config(config: BfclConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("raw", None)
    return _jsonable(_without_eval_reference(payload), roots=_config_roots(config))


def _generation_config(config: BfclConfig) -> dict[str, Any]:
    """Portable view of the YAML input, so two hosts with the same logical config match."""
    return _jsonable(_without_eval_reference(dict(config.raw or {})), roots=_config_roots(config))


def generation_config_hash(config: BfclConfig) -> str:
    """Return the exact portable generation-config identity used by Stage 12."""
    return _sha256(canonical_json(_generation_config(config)))


def generation_mode(config: BfclConfig) -> str:
    """Derive the manifest generation_mode from lineage roles and surface flags."""
    if config.lineage.policy == "smoke_no_publication":
        return "smoke_no_publication"
    paraphrase = (config.lineage.roles or {}).get("paraphrase")
    if paraphrase and paraphrase.enabled and config.surface_generation.get("model_paraphrase_enabled"):
        return "template_plus_paraphrase"
    return "template_only"


def build_messages(
    surface: dict[str, Any],
    expected_calls: list[dict[str, Any]],
    results: list[Any],
) -> list[dict[str, Any]]:
    """Assemble the OpenAI-style message array from surface text and replay results.

    One assistant message per call_group carries every call in that group, and each
    call is followed immediately by its own tool result message.
    """
    if len(results) != len(expected_calls):
        raise ValueError(f"replay returned {len(results)} results for {len(expected_calls)} expected calls")
    messages: list[dict[str, Any]] = [{"role": "system", "content": surface["system_prompt"]}]
    # Walk the calls in trace order so one assistant message consumes exactly the
    # calls of its own step, even if two steps were to carry the same group value.
    remaining = deque(enumerate(expected_calls))

    call_counter = 0
    for step in surface["steps"]:
        if step["kind"] == "user":
            messages.append({"role": "user", "content": step["content"]})
        elif step["kind"] == "assistant_text":
            messages.append({"role": "assistant", "content": step["content"]})
        else:
            group_value = int(step["call_group"])
            group: list[tuple[int, dict[str, Any]]] = []
            while remaining and int(remaining[0][1]["call_group"]) == group_value:
                group.append(remaining.popleft())
            if not group:
                raise ValueError(f"surface call_group {group_value} has no expected calls")
            tool_calls = []
            tool_messages = []
            for index, call in sorted(group, key=lambda item: item[1]["position_in_group"]):
                call_id = f"call_{call_counter}"
                call_counter += 1
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["function_name"],
                            "arguments": canonical_json(call["arguments"]),
                        },
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": canonical_json(results[index]),
                    }
                )
            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            messages.extend(tool_messages)
    if remaining:
        groups = [call.get("call_group") for _, call in remaining]
        raise ValueError(f"expected calls were not represented by surface steps: call_groups={groups}")
    return messages


def build_row(
    config: BfclConfig,
    pack: LoadedPack,
    task: dict[str, Any],
    surface: dict[str, Any],
    expected_calls: list[dict[str, Any]],
    verdict: dict[str, Any],
    tier: str,
    model_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one benchmark row with its provenance stamped top-level."""
    return {
        "task_id": task["task_id"],
        "template_id": task["template_id"],
        "variant_index": int(task["variant_index"]),
        "messages": build_messages(surface, expected_calls, verdict["results"]),
        "tools": canonical_json(model_tools),
        "expected_tool_calls": [{**call, "arguments": encode_arguments(call["arguments"])} for call in expected_calls],
        "success_assertions": list(task.get("success_assertions") or []),
        "fixture_refs": list(task.get("fixture_refs") or []),
        "intent": task.get("intent"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "required_tools": list(task.get("required_tools") or []),
        "required_tools_fingerprint": canonical_json(sorted(task.get("required_tools") or [])),
        "tools_present": list(task.get("tools_present") or []),
        "turn_policy": task.get("turn_policy"),
        "is_multi_turn": bool(task.get("is_multi_turn")),
        "num_tool_calls": int(task.get("num_tool_calls") or 0),
        "call_order": task.get("call_order", "strict"),
        "call_order_prefix": task.get("call_order_prefix"),
        "system_prompt_id": surface["system_prompt_id"],
        "tier": tier,
        "gold_eligible": tier == "gold" and config.lineage.policy != "smoke_no_publication",
        "validated_by": ["schema", "replay", "assertions"] if task.get("success_assertions") else ["schema", "replay"],
        "pack_id": task["pack_id"],
        "pack_version": task["pack_version"],
        "seed": int(task.get("seed") or 0),
        "paraphrase_model": surface.get("paraphrase_model"),
        "paraphrase_model_canonical": surface.get("paraphrase_model_canonical"),
        # Null, not False: no held-out source is configured, so the row was never
        # checked against one and must not claim it was.
        "held_out_hit": None,
        "src": f"{task['pack_id']}:{task['template_id']}",
        "metadata": canonical_json(
            {
                "language": surface["language"],
                "expt_name": config.expt_name,
                "base_task_id": surface.get("base_task_id"),
                "surface_source": surface.get("source", "template"),
                "profile_hash": surface.get("profile_hash"),
            }
        ),
    }


def _model_role(name: str, config: BfclConfig) -> dict[str, Any]:
    role = (config.lineage.roles or {}).get(name)
    enabled = bool(role and role.enabled)
    if not enabled:
        return {
            "alias": None,
            "provider": None,
            "model_identity": None,
            "canonical_id": None,
            "config_hash": None,
            "enabled": False,
        }
    model_config = dict(role.model_config or {}) if role else {}
    return {
        "alias": model_config.get("alias"),
        "provider": model_config.get("provider"),
        "model_identity": {
            "source": model_config.get("source"),
            "model": model_config.get("model"),
            "revision": model_config.get("revision"),
            "weights_digest": model_config.get("weights_digest"),
        },
        "canonical_id": (
            str(model_config["canonical_id"]).strip().lower() if model_config.get("canonical_id") else None
        ),
        "config_hash": _sha256(canonical_json(_jsonable(model_config))),
        "enabled": True,
    }


def _bias_applicability(pack: LoadedPack) -> dict[str, dict[str, str]]:
    """Describe which audit dimensions the pack can meaningfully exercise."""
    result = {f"B{index}": {"status": "applicable"} for index in range(1, 17)}
    policy = held_out_policy(pack)
    if policy is None:
        result["B7"] = {
            "status": "na",
            "reason": "pack declares no held_out policy",
        }
    elif policy.reserves_nothing:
        # The policy was enforced, but it withholds nothing, so a leakage result
        # from this run would say nothing about generalization.
        result["B7"] = {
            "status": "na",
            "reason": "held_out policy reserves no fixture row or template",
        }
    policies = {str(template.get("turn_policy")) for template in pack.templates}
    has_parallel_group = any(
        count > 1
        for template in pack.templates
        for count in Counter(
            milestone.get("call_group")
            for milestone in template.get("assistant_milestones") or []
            if milestone.get("type") == "tool_call" and milestone.get("call_group") is not None
        ).values()
    )
    for policy in (
        "single_turn",
        "missing_slot",
        "confirmation",
        "correction",
        "multi_tool",
        "dependent_call",
        "negative_path",
        "clarify_only",
        "irrelevant",
    ):
        key = f"B3.{policy}"
        applicable = policy in policies or (policy == "multi_tool" and has_parallel_group)
        result[key] = (
            {"status": "applicable"}
            if applicable
            else {
                "status": "na",
                "reason": f"pack declares no {policy} template",
            }
        )
    has_distractor = any(
        set(template.get("tools_present") or []) - set(template.get("required_tools") or [])
        for template in pack.templates
    )
    result["B3.distractor_present"] = (
        {"status": "applicable"}
        if has_distractor
        else {
            "status": "na",
            "reason": "pack templates expose no distractor tools",
        }
    )
    return result


def _reference_benchmark_manifest(config: BfclConfig) -> dict[str, Any] | None:
    reference = config.reference_benchmark
    if reference is None:
        return None
    return {
        "name": reference.name,
        "samples_path": _jsonable(
            reference.samples_path,
            roots=_config_roots(config),
        ),
        "content_hash": reference.content_hash,
    }


def run_final_output(
    config: BfclConfig,
    pack: LoadedPack,
    tasks: list[dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
    verdicts: dict[str, dict[str, Any]],
    validation_report: dict[str, Any],
    prompt_bundle: dict[str, Any],
    stage_counts: dict[str, int],
    expected_pack_fingerprint: str,
    trace_drop_reasons: dict[str, str] | None = None,
    surface_quality_records: list[dict[str, Any]] | None = None,
    surface_quality_report: dict[str, Any] | None = None,
    dedup_balancing_decisions: list[Any] | None = None,
    dedup_balancing_report: dict[str, Any] | None = None,
    expected_artifact_hashes: dict[str, str] | None = None,
    before_publication_commit: Callable[[Path], None] | None = None,
) -> Path:
    """Write benchmark_raw.parquet, benchmark.parquet, and run_manifest.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tier = str(validation_report.get("tier", "prototype"))
    model_tools = project_model_facing_tools(pack.tools)
    model_tools_by_name = {str((tool.get("function") or {}).get("name")): tool for tool in model_tools}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        verdict = verdicts.get(task_id) or {}
        surface = surfaces[task_id]
        if not verdict.get("passed"):
            continue
        task_tools = [
            model_tools_by_name[name] for name in task.get("tools_present") or [] if name in model_tools_by_name
        ]
        rows.append(build_row(config, pack, task, surface, traces[task_id], verdict, tier, task_tools))

    if not rows:
        raise RuntimeError(
            "BFCL final_output has no replay-validated rows; inspect stage_cache/replay_validated_tasks.parquet"
        )

    cache = stage_cache_dir(config)
    for artifact_name, expected_hash in sorted((expected_artifact_hashes or {}).items()):
        artifact_path = cache / artifact_name
        try:
            actual_hash = _file_hash(artifact_path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"stage artifact {artifact_name} disappeared before publication") from exc
        if actual_hash != expected_hash:
            raise RuntimeError(f"stage artifact {artifact_name} changed after its producing stage completed")
    if surface_quality_report is not None:
        try:
            stored_surface_report = json.loads((cache / "surface_quality_rejections.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError("Stage 10 requires surface_quality_rejections.json") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Stage 10 surface_quality_rejections.json is not valid JSON") from exc
        if stored_surface_report != surface_quality_report:
            raise ValueError("surface_quality_rejections.json does not match the Stage 10 verdict")
    # Hash the intermediates first: a missing stage artifact must stop the run before
    # a published parquet exists without the manifest that explains it.
    stage_artifacts = {
        name.removesuffix(".parquet"): {"content_hash": _file_hash(cache / name)} for name in STAGE_TABLES
    }
    tool_artifacts = {
        "tools_internal": {"content_hash": _file_hash(cache / "tools_normalized_internal.json")},
        "tools_model_facing": {"content_hash": _file_hash(cache / "tools_normalized.json")},
    }
    # Stage 10 ran, so its outputs are evidence the manifest must carry. Hash them here,
    # with the other mandatory intermediates, so a deleted artifact stops the run before
    # a published parquet exists: hashing only when present would let a missing file
    # publish a quality claim that nothing substantiates.
    surface_quality_artifacts = (
        {
            "surface_validated_tasks": {"content_hash": _file_hash(cache / SURFACE_VALIDATED_TASKS)},
            "surface_quality_rejections": {"content_hash": _file_hash(cache / "surface_quality_rejections.json")},
            **(
                {"surface_judge_cache_usage": {"content_hash": _file_hash(cache / "surface_judge_cache_usage.json")}}
                if (judge_role := (config.lineage.roles or {}).get("surface_judge")) and judge_role.enabled
                else {}
            ),
        }
        if surface_quality_records is not None
        else {}
    )
    dedup_enabled = bool(config.semantic_deduplication_config.get("enabled"))
    if dedup_enabled != (dedup_balancing_decisions is not None):
        raise ValueError("semantic_deduplication_config.enabled must match the presence of Stage 11 decisions")
    if (dedup_balancing_decisions is None) != (dedup_balancing_report is None):
        raise ValueError("Stage 11 decisions and report must be provided together")
    dedup_balancing_artifacts: dict[str, dict[str, str]] = {}
    if dedup_balancing_report is not None:
        report_path = cache / "dedup_balancing_report.json"
        stored_report = json.loads(report_path.read_text(encoding="utf-8"))
        if stored_report != dedup_balancing_report:
            raise ValueError("Stage 11 report does not match dedup_balancing_report.json")
        current_balanced_tasks_hash = _file_hash(cache / BALANCED_TASKS)
        reported_balanced_tasks_hash = (
            (stored_report.get("artifacts") or {}).get(BALANCED_TASKS, {}).get("content_hash")
        )
        if reported_balanced_tasks_hash != current_balanced_tasks_hash:
            raise ValueError("balanced_tasks.parquet content hash does not match the Stage 11 report")
        dedup_balancing_artifacts = {
            "balanced_tasks": {"content_hash": current_balanced_tasks_hash},
            "dedup_balancing_report": {"content_hash": _file_hash(report_path)},
        }
    current_pack_fingerprint = _require_pack_fingerprint(
        pack,
        expected_pack_fingerprint,
        phase="before final output was assembled",
    )

    output_dir = Path(config.output_dir) / config.expt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_endpoint_identity(
        config,
        pack,
        validation_report.get("endpoint_metadata"),
    )
    schema = benchmark_schema()

    # Raw means executable/schema-valid. Publication additionally applies Stage 10
    # when enabled; the disabled path preserves the legacy deterministic guard gate.
    guard_violations: dict[str, bool] | None = None
    quality_decisions: dict[str, str] | None = None
    if surface_quality_records is None:
        if surface_quality_report is not None:
            raise ValueError("a surface-quality report requires surface-quality records")
        guard_violations = {
            str(row["task_id"]): bool(surfaces[str(row["task_id"])]["guard_violations"]) for row in rows
        }
        published = [row for row in rows if not surfaces[str(row["task_id"])]["guard_violations"]]
    else:
        if surface_quality_report is None:
            raise ValueError("surface-quality records require a surface-quality report")
        quality_by_task: dict[str, dict[str, Any]] = {}
        for record in surface_quality_records:
            task_id = str(record["task_id"])
            if task_id in quality_by_task:
                raise ValueError(f"duplicate surface-quality record for task {task_id!r}")
            quality_by_task[task_id] = record
        raw_ids = {str(row["task_id"]) for row in rows}
        if set(quality_by_task) != raw_ids:
            missing = sorted(raw_ids - set(quality_by_task))
            extra = sorted(set(quality_by_task) - raw_ids)
            raise ValueError(
                f"surface-quality records must match replay-validated rows exactly (missing={missing}, extra={extra})"
            )
        quality_decisions = {
            str(row["task_id"]): str(quality_by_task[str(row["task_id"])]["decision"]) for row in rows
        }
        published = [row for row in rows if quality_by_task[str(row["task_id"])]["decision"] == "kept"]
    stage_eleven_decisions: list[Any] | None = None
    if dedup_balancing_decisions is not None:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
            DedupBalancingDecision,
        )

        stage_eleven_decisions = [
            value if isinstance(value, DedupBalancingDecision) else DedupBalancingDecision.model_validate(value)
            for value in dedup_balancing_decisions
        ]
        stage_eleven_by_task: dict[str, DedupBalancingDecision] = {}
        for decision in stage_eleven_decisions:
            if decision.task_id in stage_eleven_by_task:
                raise ValueError(f"duplicate Stage 11 decision for task {decision.task_id!r}")
            stage_eleven_by_task[decision.task_id] = decision
        stage_ten_ids = {str(row["task_id"]) for row in published}
        if set(stage_eleven_by_task) != stage_ten_ids:
            missing = sorted(stage_ten_ids - set(stage_eleven_by_task))
            extra = sorted(set(stage_eleven_by_task) - stage_ten_ids)
            raise ValueError(
                "Stage 11 decisions must match Stage 10 publication candidates exactly "
                f"(missing={missing}, extra={extra})"
            )
        published = sorted(
            (row for row in published if stage_eleven_by_task[str(row["task_id"])].selected),
            key=lambda row: stage_eleven_by_task[str(row["task_id"])].selection_rank,
        )
        expected_selected = int(dedup_balancing_report["counts"]["selected"])
        if expected_selected != len(published):
            raise ValueError("Stage 11 report selected count does not match its decisions")
    stage_eleven_gold_eligible = True
    if dedup_balancing_report is not None:
        unmet_targets = dedup_balancing_report.get("unmet_targets") or []
        release_policy = dedup_balancing_report.get("release_policy")
        if not isinstance(release_policy, dict):
            raise ValueError("Stage 11 report requires release_policy")
        policy = release_policy.get("unmet_target_policy")
        if policy not in {"abort", "publish_non_gold"}:
            raise ValueError("Stage 11 report carries an invalid unmet_target_policy")
        if policy != config.semantic_deduplication_config.get("unmet_target_policy", "abort"):
            raise ValueError("Stage 11 report unmet_target_policy does not match config")
        expected_action = policy if unmet_targets else "none"
        if release_policy.get("unmet_target_action") != expected_action:
            raise ValueError("Stage 11 report carries an inconsistent unmet_target_action")
        if bool(unmet_targets) == bool(release_policy.get("gold_eligible")):
            raise ValueError("Stage 11 release eligibility is inconsistent with unmet targets")
        if unmet_targets and policy == "abort":
            raise RuntimeError(
                "Stage 11 has unmet balancing targets under abort policy; "
                "inspect stage_cache/dedup_balancing_report.json"
            )
        stage_eleven_gold_eligible = bool(release_policy["gold_eligible"])
        if not stage_eleven_gold_eligible:
            for row in rows:
                row["gold_eligible"] = False
    if not published:
        if surface_quality_records is not None:
            if dedup_balancing_decisions is not None:
                source = "Stage 11 deduplication and balancing policy"
                recovery = "inspect stage_cache/balanced_tasks.parquet and stage_cache/dedup_balancing_report.json"
            else:
                source = "Stage 10 surface-quality policy"
                recovery = (
                    "inspect stage_cache/surface_validated_tasks.parquet and "
                    "stage_cache/surface_quality_rejections.json"
                )
        else:
            source = "deterministic surface guards"
            recovery = "inspect stage_cache/rendered_conversations.parquet"
        raise RuntimeError(f"BFCL final_output has no publication rows after {source}; {recovery}")
    held_out = held_out_policy(pack)
    held_out_scan_report: dict[str, Any] | None = None
    held_out_hits: dict[str, bool] | None = None
    if held_out is not None:
        normalized_path = cache / "held_out_normalized.json"
        try:
            stored_policy = json.loads(normalized_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError("a held-out run requires a valid stage_cache/held_out_normalized.json") from exc
        if stored_policy != pack.held_out:
            raise ValueError("held_out_normalized.json does not match the loaded held-out policy")
        binding_report = load_binding_report(
            config,
            held_out,
            expected_tasks_expanded=int(stage_counts["canonical_expanded"]),
        )
        # Only tasks that actually recorded their bindings are mapped; a task missing
        # the key is left out so the scan reports it instead of clearing it by default.
        fixture_refs_by_task = {
            str(task["task_id"]): [str(reference) for reference in task["fixture_refs"]]
            for task in tasks
            if isinstance(task.get("fixture_refs"), list)
        }
        # Scan every executable row, not only the published ones: a reserved binding
        # anywhere means Stage 4 enforcement failed, and benchmark_raw.parquet ships
        # beside the manifest that would claim it did not.
        held_out_decisions = scan_rows(held_out, rows, fixture_refs_by_task=fixture_refs_by_task)
        held_out_scan_report = write_scan_report(
            config,
            held_out,
            held_out_decisions,
            binding_report=binding_report,
            rows_published=len(published),
        )
        stored_scan_report = json.loads((cache / HELD_OUT_SCAN).read_text(encoding="utf-8"))
        if stored_scan_report != held_out_scan_report:
            raise ValueError("held_out_scan.json does not match the Stage 12 scan")
        enforce_no_leak(config, held_out_scan_report)
        held_out_by_task = {decision.task_id: decision for decision in held_out_decisions}
        for row in rows:
            row["held_out_hit"] = held_out_by_task[str(row["task_id"])].held_out_hit
        held_out_hits = {task_id: bool(decision.held_out_hit) for task_id, decision in held_out_by_task.items()}

    # Derived from the stage decisions rather than from ``published``: the plan and
    # the filtered list agreeing is the check, and a plan copied from the list it is
    # supposed to police would confirm any mistake that list already made.
    publication_plan = plan_publication(
        raw_task_ids=[str(row["task_id"]) for row in rows],
        replay_validated_rows=int(stage_counts["replay_passed"]),
        guard_violations=guard_violations,
        surface_quality_decisions=quality_decisions,
        dedup_decisions=stage_eleven_decisions,
        held_out_hits=held_out_hits,
    )
    requested_exports = tuple(name for name in EXPORT_FORMATS if config.exports.get(name))
    export_validation_report = None
    export_validation_report_hash = None

    surface_rejections: dict[str, int] = {}
    for surface in surfaces.values():
        if surface["guard_violations"]:
            template_id = str(surface["template_id"])
            surface_rejections[template_id] = surface_rejections.get(template_id, 0) + 1
    trace_drop_reasons = trace_drop_reasons or {}
    trace_drop_summary = dict(sorted(Counter(trace_drop_reasons.values()).items()))
    # A pack whose run never reached the paraphrase stage still needs a manifest, so an
    # absent report reads as "nothing was requested" rather than stopping publication.
    paraphrase_report_path = cache / "paraphrase_rejections.json"
    paraphrase_report: dict[str, Any] = (
        json.loads(paraphrase_report_path.read_text(encoding="utf-8")) if paraphrase_report_path.is_file() else {}
    )
    created_at = datetime.now(timezone.utc)
    schema_version = str(config.schema_version or DEFAULT_BENCHMARK_SCHEMA_VERSION)
    _require_pack_fingerprint(
        pack,
        current_pack_fingerprint,
        phase="while final output was being assembled",
        cleanup=(),
    )
    # Hash prepare/validation intermediates the published rows depend on, not only
    # the six stage tables — otherwise a tampered validation report would not show.
    lineage_artifacts = {
        "oracle_validation_report": {"content_hash": _file_hash(cache / "oracle_validation_report.json")},
        "pack_manifest": {"content_hash": _file_hash(cache / "pack_manifest.json")},
        "fixtures_normalized": {"content_hash": _file_hash(cache / "fixtures_normalized.json")},
        "task_templates_normalized": {"content_hash": _file_hash(cache / "task_templates_normalized.yaml")},
        "reference_profile": {"content_hash": _file_hash(cache / "reference_profile.json")},
        "reference_samples": {"content_hash": _file_hash(cache / REFERENCE_SAMPLES)},
    }
    optional_artifacts = [
        "reference_profile_io_cache.jsonl",
        "paraphrase_io_cache.jsonl",
        "paraphrase_rejections.json",
    ]
    # A declared policy makes its evidence mandatory: hashing it only when the file
    # happens to exist would let a deleted scan publish an unbacked leakage claim.
    held_out_artifacts = (
        {
            "held_out_normalized": {"content_hash": _file_hash(cache / "held_out_normalized.json")},
            "held_out_bindings": {"content_hash": _file_hash(cache / HELD_OUT_BINDINGS)},
            "held_out_scan": {"content_hash": _file_hash(cache / HELD_OUT_SCAN)},
        }
        if held_out is not None
        else {}
    )
    lineage_artifacts.update(surface_quality_artifacts)
    lineage_artifacts.update(dedup_balancing_artifacts)
    lineage_artifacts.update(held_out_artifacts)
    for artifact_name in optional_artifacts:
        artifact_path = cache / artifact_name
        if artifact_path.is_file():
            lineage_artifacts[artifact_name.rsplit(".", 1)[0]] = {"content_hash": _file_hash(artifact_path)}
    gold_ineligibility_reasons: list[str] = []
    if tier != "gold":
        gold_ineligibility_reasons.append("pack_tier_not_gold")
    if config.lineage.policy == "smoke_no_publication":
        gold_ineligibility_reasons.append("smoke_no_publication")
    if not stage_eleven_gold_eligible:
        gold_ineligibility_reasons.append("stage_eleven_unmet_targets")

    # No final path is touched until every source artifact above has been read and
    # hashed successfully. From here on, failures delete one private staging tree.
    staging_dir = output_dir / f".stage12-{uuid.uuid4().hex}"
    _discard(staging_dir)
    staging_dir.mkdir(parents=True)
    raw_path = staging_dir / "benchmark_raw.parquet"
    benchmark_path = staging_dir / "benchmark.parquet"
    try:
        _write_parquet_atomic(pa.Table.from_pylist(rows, schema=schema), raw_path, pq)
        _write_parquet_atomic(pa.Table.from_pylist(published, schema=schema), benchmark_path, pq)
        publication_report = verify_written_benchmarks(
            raw_path=raw_path,
            publication_path=benchmark_path,
            plan=publication_plan,
        )

        # Every enabled format re-encodes one projection of the file just verified,
        # so the parquet is decoded once no matter how many writers run.
        if requested_exports:
            projection = project_published_benchmark(
                benchmark_path,
                expected_content_hash=publication_report.publication_content_hash,
                expected_task_ids=publication_plan.published_task_ids,
            )
            export_artifacts: dict[str, ExportArtifact] = {}
            for name in requested_exports:
                export_artifacts[name] = EXPORT_WRITERS[name](projection, staging_dir)
            export_validation_report, export_validation_report_hash = validate_and_write_export_report(
                projection,
                export_artifacts,
                staging_dir,
            )
        exports_manifest = export_manifest_section(
            enabled={name: bool(config.exports.get(name)) for name in EXPORT_FORMATS},
            report=export_validation_report,
            validation_report_path=EXPORT_VALIDATION_REPORT_FILE if export_validation_report is not None else None,
            validation_report_content_hash=export_validation_report_hash,
        )
    except Exception:
        _discard(staging_dir)
        raise

    provider_origin = load_origin_provenance(
        pack.paths,
        pack.endpoint_config,
        pack_fingerprint=current_pack_fingerprint,
        pack_id=str(pack.manifest.get("pack_id")),
        pack_version=str(pack.manifest.get("version")),
    )
    manifest = {
        "schema_version": schema_version,
        # The timestamp is part of the id: two runs of the same config are different
        # runs, and a manifest that cannot be told from its predecessor is not lineage.
        "run_id": (
            f"{config.expt_name}-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{prompt_bundle['system_prompt_id'].split(':')[-1][:12]}-{uuid.uuid4().hex}"
        ),
        "created_at": created_at.isoformat(),
        "oracle_clock": config.oracle_runtime.clock,
        "generation_mode": generation_mode(config),
        "pipeline_version": "bfcl-runtime-0.1.0",
        "generation_config_hash": generation_config_hash(config),
        "resolved_config_hash": _sha256(canonical_json(_resolved_config(config))),
        "runtime": runtime_metadata(),
        "pack": {
            "pack_id": pack.manifest.get("pack_id"),
            "version": pack.manifest.get("version"),
            "fingerprint_contract": PACK_FINGERPRINT_CONTRACT,
            "content_hash": f"sha256:{current_pack_fingerprint}",
            # content_hash alone proves the pack changed but names no file, which
            # leaves a later eval telling an operator to restore a revision it
            # cannot describe. The per-file map is what turns that into "the
            # README moved, the backend did not".
            "files": pack_file_hashes(pack.paths),
        },
        "oracle": {
            "kind": "endpoint" if pack.endpoint_config is not None else "python",
            "endpoint_metadata": validation_report.get("endpoint_metadata"),
            **(
                {("mcp" if provider_origin.get("provider_kind") == "mcp" else "origin"): provider_origin}
                if provider_origin is not None
                else {}
            ),
        },
        "reference_benchmark": _reference_benchmark_manifest(config),
        "prompt_bundle_hash": prompt_bundle["prompt_bundle_hash"],
        "tier": tier,
        "gold_eligible": (
            tier == "gold" and config.lineage.policy != "smoke_no_publication" and stage_eleven_gold_eligible
        ),
        "gold_ineligibility_reasons": gold_ineligibility_reasons,
        "lineage_policy": config.lineage.policy,
        # Reported from what ships, not from what was attempted: a profile that shaped
        # only rejected candidates influenced no published surface.
        "profile_influenced_surface": any(
            bool(surfaces[str(row["task_id"])].get("profile_hash")) for row in published
        ),
        "models": {name: _model_role(name, config) for name in ("profile", "paraphrase", "surface_judge")},
        "judge_advisory": config.lineage.judge_advisory,
        "surface_quality_validation": {
            "contract_version": config.surface_quality_validation.get("contract_version"),
            "enabled": bool(config.surface_quality_validation.get("enabled")),
            "drop_authority": bool(config.surface_quality_validation.get("drop_authority")),
            "report": surface_quality_report,
        },
        "semantic_deduplication": {
            "contract_version": config.semantic_deduplication_config.get("contract_version"),
            "enabled": dedup_enabled,
            "model_identifier": (
                config.semantic_deduplication_config.get("model_identifier") if dedup_enabled else None
            ),
            "settings_hash": (
                (dedup_balancing_report.get("lineage") or {}).get("settings_hash")
                if dedup_balancing_report is not None
                else None
            ),
            "embedding_signature": (
                (dedup_balancing_report.get("lineage") or {}).get("embedding_signature")
                if dedup_balancing_report is not None
                else None
            ),
            "report": dedup_balancing_report,
        },
        "seeds": {
            "global": int(config.random_seed or 0),
            "derivation": (
                "uint64_be(sha256(canonical_json(global_seed,pack_id,pack_version,"
                "template_id,sorted_fixture_refs,slot_bindings,variant_index))[0:8])"
            ),
        },
        "held_out": manifest_section(held_out, held_out_scan_report),
        "publication": publication_manifest_section(publication_report),
        "exports": exports_manifest,
        "bias_targets": _jsonable(config.task_generation),
        "bias_applicability": _bias_applicability(pack),
        "stage_counts": {**stage_counts, "published": len(published)},
        # Guard rejections are counted per template, because a template is what an
        # author fixes; paraphrase does not run, so these are surface guards only.
        "surface_guard_rejections": {"by_template": surface_rejections},
        "paraphrase_rejections": {
            "requested_candidates": int(paraphrase_report.get("requested_candidates", 0)),
            "accepted_candidates": int(paraphrase_report.get("accepted_candidates", 0)),
            "rejected_candidates": int(paraphrase_report.get("rejected_candidates", 0)),
            "by_reason": dict(paraphrase_report.get("by_reason") or {}),
            "by_template": dict(paraphrase_report.get("by_template") or {}),
        },
        "trace_drop_rejections": {
            "count": len(trace_drop_reasons),
            "by_reason": trace_drop_summary,
        },
        "artifacts": {
            **tool_artifacts,
            **lineage_artifacts,
            # One entry per stage artifact, so a published row can be traced back to
            # the exact intermediate that produced it.
            **stage_artifacts,
            "benchmark_raw_parquet": {"content_hash": _file_hash(raw_path)},
            "benchmark_parquet": {"content_hash": _file_hash(benchmark_path)},
            **(
                {
                    "export_validation_report": {
                        "content_hash": export_validation_report_hash,
                    }
                }
                if export_validation_report_hash is not None
                else {}
            ),
        },
    }
    _require_pack_fingerprint(
        pack,
        current_pack_fingerprint,
        phase="before the final manifest was stamped",
        cleanup=(staging_dir,),
    )
    manifest_path = staging_dir / "run_manifest.json"
    _write_endpoint_manifest_atomic(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        manifest_path,
        config=config,
        pack=pack,
        expected_endpoint_metadata=validation_report.get("endpoint_metadata"),
        cleanup=(staging_dir,),
    )
    if before_publication_commit is not None:
        before_publication_commit(staging_dir)
    final_benchmark_path = _commit_staged_publication(staging_dir, output_dir)
    logger.info(
        "BFCL final_output wrote %d rows (%d published) to %s",
        len(rows),
        len(published),
        final_benchmark_path,
    )
    return final_benchmark_path
