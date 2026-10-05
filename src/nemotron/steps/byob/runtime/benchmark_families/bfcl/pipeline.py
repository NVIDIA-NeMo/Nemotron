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

"""BFCL benchmark-family orchestration.

The generic CLI dispatcher lives in `nemotron.steps.byob.scripts.runtime`.
This module owns the BFCL-specific stage order and cache paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import SURFACE_GENERATION_KEYS

if TYPE_CHECKING:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack

logger = logging.getLogger(__name__)


def _artifact_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _verify_stage_artifact_matches_report(artifact: Path, report: Path) -> None:
    """Refuse a stage artifact that no longer matches the hash its report declares.

    Stage 12 repeats this before publishing, but the checkpoint written right after a
    stage reads its artifacts, so tampering has to be named by the integrity check
    rather than by whichever reader happens to touch the bytes first.
    """
    declared = (
        (json.loads(report.read_text(encoding="utf-8")).get("artifacts") or {})
        .get(artifact.name, {})
        .get("content_hash")
    )
    if declared != _artifact_hash(artifact):
        raise ValueError(f"{artifact.name} content hash does not match the {report.name} report")


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _output_lock(config: BfclConfig) -> Iterator[None]:
    """Prevent two writers from interleaving one experiment directory."""
    output_dir = Path(config.output_dir) / config.expt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".bfcl.lock"
    handle = lock_path.open("a+b")
    unlock: Callable[[], None]
    try:
        try:
            if os.name == "nt":
                import msvcrt

                if lock_path.stat().st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                unlock = partial(
                    msvcrt.locking,
                    handle.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                unlock = partial(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise RuntimeError(f"another BFCL process is writing experiment {config.expt_name!r}") from exc
        yield
    finally:
        try:
            if "unlock" in locals():
                unlock()
        finally:
            handle.close()


# Reports written to the stage cache are editable, so a stored gold claim is never
# trusted across runs. Validation results are instead remembered for the lifetime of
# this process, keyed by the pack and config fingerprints they were computed from, so
# `stage=all` does not pay for the same episodes twice.
_VALIDATED_THIS_PROCESS: dict[tuple[str, str, str, str], dict] = {}
_FINAL_ARTIFACTS = (
    "run_manifest.json",
    "run_manifest.json.tmp",
    "benchmark.parquet",
    "benchmark.parquet.tmp",
    "benchmark_raw.parquet",
    "benchmark_raw.parquet.tmp",
)
# The whole export tree, so a run that disables a format cannot inherit the one a
# previous run left behind and publish it beside a manifest that never mentions it.
_FINAL_EXPORT_DIRECTORIES = ("exports",)
# Derived Stage 10 outputs, rewritten whenever the stage runs. A previous run must not
# leave them behind for a run that disables the stage, or for one that aborts on pack
# drift: either way the next manifest would hash a verdict this run never reached.
_SURFACE_QUALITY_ARTIFACTS = (
    "surface_validated_tasks.parquet",
    "surface_quality_rejections.json",
    "surface_judge_cache_usage.json",
)
_DEDUP_BALANCING_ARTIFACTS = (
    "balanced_tasks.parquet",
    "balanced_tasks.parquet.tmp",
    "dedup_balancing_report.json",
    "dedup_balancing_report.json.tmp",
)
_HELD_OUT_ARTIFACTS = (
    "held_out_normalized.json",
    "held_out_bindings.json",
    "held_out_bindings.json.tmp",
    "held_out_scan.json",
    "held_out_scan.json.tmp",
)


def _unsupported_requests(config: BfclConfig) -> list[str]:
    """List config features that are requested but that generation cannot honor.

    Generation refuses these instead of ignoring them, so a run never reports
    lineage or quality guarantees that no stage actually applied.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import EXPORT_WRITERS

    requested: list[str] = []
    requested.extend(
        f"exports.{name}" for name, on in sorted(config.exports.items()) if on and name not in EXPORT_WRITERS
    )
    supported_task_generation = {
        "tasks_per_category",
        "candidate_tasks_per_category",
        "target_published_tasks",
        "target_published_tasks",
    }
    if config.semantic_deduplication_config.get("enabled"):
        supported_task_generation.update(
            {
                "difficulty_mix",
                "turn_mix",
                "tool_call_count_mix",
                "max_intent_share",
                "policy_mix",
                "max_turns",
                "max_tool_calls",
            }
        )
    requested.extend(
        f"task_generation.{name}" for name in sorted(config.task_generation) if name not in supported_task_generation
    )
    # Asking for work a disabled stage would have done. Reading these is what keeps
    # "refuse, do not ignore" true for the whole config and not just the gate flags.
    # Evaluation is a separate run over a published benchmark: the eval config is
    # parsed and hashed by bfcl.eval, so generation refuses it rather than accepting
    # a config no stage of this run will honor.
    if config.eval_config_path is not None:
        requested.append("eval_config_path")
    if config.inline_eval is not None:
        requested.append("eval")
    if config.translation_config_path is not None:
        requested.append("translation_config_path")
    # Shared BYOB fields that BFCL loads for schema compatibility but never applies.
    # Naming them keeps "refuse, do not ignore" true for leftover MCQ keys too.
    if config.input_dir is not None:
        requested.append("input_dir")
    if config.generation_model_config:
        requested.append("generation_model_config")
    if config.judge_model_config:
        requested.append("judge_model_config")
    # A key no stage reads is almost always a typo for one that matters, so name it
    # rather than letting the run proceed with a setting that had no effect.
    requested.extend(
        f"surface_generation.{name}"
        for name in sorted(config.surface_generation)
        if name not in SURFACE_GENERATION_KEYS
    )
    return requested


def _validate_pack(config: BfclConfig, *, force: bool = False) -> tuple[dict, Path]:
    """Validate the pack, reusing only results this process computed itself."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import pack_fingerprint
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        run_oracle_validation,
        validation_config_fingerprint,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.prepare import prepare_oracle_pack

    pack = prepare_oracle_pack(config)
    report_path = stage_cache_dir(config) / "oracle_validation_report.json"
    fingerprint_before = pack_fingerprint(pack.paths)
    try:
        endpoint_identity_key = _endpoint_metadata(config, pack)
    except Exception as exc:  # noqa: BLE001 — validation records the endpoint failure
        # Endpoint availability and identity are validation outcomes. Keep a stable
        # cache key for the failed probe, then let oracle_validation write the
        # actionable failure report instead of aborting prepare before it can do so.
        endpoint_identity_key = f"error:{type(exc).__name__}:{exc}"
    key = (
        str(report_path),
        fingerprint_before,
        validation_config_fingerprint(config),
        (
            repr(sorted(endpoint_identity_key.items()))
            if isinstance(endpoint_identity_key, dict)
            else str(endpoint_identity_key or "")
        ),
    )
    remembered = None if force else _VALIDATED_THIS_PROCESS.get(key)
    if remembered is not None:
        # The in-process verdict is authoritative; repair a deleted or edited
        # report so the manifest cannot hash evidence different from that verdict.
        try:
            stored = json.loads(report_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            stored = None
        if stored != remembered:
            _write_json_atomic(report_path, remembered)
        return remembered, report_path
    report = run_oracle_validation(config, pack)
    fingerprint_after = pack_fingerprint(pack.paths)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            "oracle pack changed while it was being validated; pack code and concurrent "
            "writers must not modify fingerprinted inputs"
        )
    _VALIDATED_THIS_PROCESS[key] = report
    return report, report_path


def _invalidate_final_outputs(config: BfclConfig) -> None:
    """Remove a previous completed run before reusing its experiment directory.

    The manifest is removed first: if this generation later fails, no old manifest can
    make stale parquets look like the result of the failed invocation.
    """
    _invalidate_publication_outputs(config)
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

    cache = stage_cache_dir(config)
    for name in (*_SURFACE_QUALITY_ARTIFACTS, *_DEDUP_BALANCING_ARTIFACTS, *_HELD_OUT_ARTIFACTS):
        (cache / name).unlink(missing_ok=True)


def _invalidate_publication_outputs(config: BfclConfig) -> None:
    """Remove only Stage 12 publication commit artifacts, preserving checkpoints."""
    output_dir = Path(config.output_dir) / config.expt_name
    for name in _FINAL_ARTIFACTS:
        (output_dir / name).unlink(missing_ok=True)
    for name in _FINAL_EXPORT_DIRECTORIES:
        shutil.rmtree(output_dir / name, ignore_errors=True)
    for staging_dir in output_dir.glob(".stage12-*"):
        shutil.rmtree(staging_dir, ignore_errors=True)


def _endpoint_metadata(config: BfclConfig, pack: LoadedPack) -> dict[str, str] | None:
    """Read and verify remote oracle identity through the normal process worker."""
    endpoint_config = pack.endpoint_config
    if endpoint_config is None:
        return None
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

    runtime = config.oracle_runtime
    outputs = ProcessWorker(
        default_timeout_s=runtime.episode_timeout_s,
        worker=runtime.worker,
    ).run_episode(
        endpoint_config=endpoint_config,
        fixtures=None,
        clock_iso=runtime.clock,
        seed=int(config.random_seed or 0),
        task_id="endpoint-metadata",
        steps=[{"op": "metadata"}],
        import_timeout_s=runtime.import_timeout_s,
        tool_timeout_s=runtime.tool_timeout_s,
        episode_timeout_s=runtime.episode_timeout_s,
    )
    metadata = outputs[0]
    if not isinstance(metadata, dict):
        raise RuntimeError("endpoint metadata response was not an object")
    return {str(key): str(value) for key, value in metadata.items()}


def _prepare_bfcl_unlocked(
    config_path: str | os.PathLike[str],
    *,
    force_validation: bool = False,
) -> Path:
    """Normalize and validate the configured oracle pack."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
    )

    config = BfclConfig.from_yaml(str(config_path))
    # Prepare rewrites the lineage cache. Remove any completed publication first
    # so an old benchmark cannot remain beside validation evidence for a new pack.
    _invalidate_final_outputs(config)
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.checkpoint import (
        clear_checkpoints,
    )

    clear_checkpoints(config)
    report, report_path = _validate_pack(config, force=force_validation)
    gold_eligible, tier = derive_pack_tier(report)
    if not gold_eligible:
        logger.warning(
            "BFCL prepare completed but pack is not gold_eligible (tier=%s)",
            tier,
        )
    return report_path


def prepare_bfcl(
    config_path: str | os.PathLike[str],
    *,
    force_validation: bool = False,
) -> Path:
    """Normalize and validate an oracle pack under an output lock."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig

    config = BfclConfig.from_yaml(str(config_path))
    with _output_lock(config):
        return _prepare_bfcl_unlocked(
            config_path,
            force_validation=force_validation,
        )


def _generate_bfcl_unlocked(
    config_path: str | os.PathLike[str],
    *,
    skip_until: str | None = None,
) -> Path:
    """Run the supported BFCL generation phases."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
        pack_fingerprint,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
        run_dedup_balancing_stage,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.executable_replay import (
        run_executable_replay,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import run_expand
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        run_expected_trace,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import run_final_output
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.paraphrase import (
        apply_expected_result_guards,
        run_paraphrase,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import run_render
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.schema_validation import (
        run_schema_validation,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.surface_quality import (
        run_surface_quality_validation,
    )

    config = BfclConfig.from_yaml(str(config_path))
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.checkpoint import (
        APPEND_ONLY_CACHE_LOCKS,
        APPEND_ONLY_CACHES,
        clear_checkpoints,
        clear_from_stage,
        current_identity,
        enabled_stages,
        restore_predecessor,
        validate_resume_target,
        write_checkpoint,
    )

    stages = enabled_stages(config)
    target = skip_until or stages[0]
    if skip_until is not None:
        validate_resume_target(config, skip_until)
    unsupported = _unsupported_requests(config)
    if unsupported:
        raise NotImplementedError(
            "BFCL generate cannot honor these config settings: "
            + ", ".join(unsupported)
            + ". Disable them to run the template-only path."
        )
    if skip_until is None:
        _invalidate_final_outputs(config)
        clear_checkpoints(config)
    else:
        # A failed resume must never leave a previous publication looking current.
        _invalidate_publication_outputs(config)
    report, _ = _validate_pack(config, force=skip_until is not None)
    cache = stage_cache_dir(config)
    expected_artifact_hashes: dict[str, str] = {}

    def pin_artifacts(*names: str) -> None:
        for name in names:
            path = cache / name
            if path.is_file():
                expected_artifact_hashes[name] = _artifact_hash(path)

    pin_artifacts(
        "oracle_validation_report.json",
        "tools_normalized_internal.json",
        "tools_normalized.json",
        "fixtures_normalized.json",
        "task_templates_normalized.yaml",
        "validation_cases_normalized.yaml",
        "pack_manifest.json",
        "pack_paths.json",
        "held_out_normalized.json",
    )

    # Derive the gate from the individual checks so a summary flag cannot stand alone.
    gold_eligible, tier = derive_pack_tier(report)
    report["gold_eligible"] = gold_eligible
    report["tier"] = tier
    if not gold_eligible:
        raise RuntimeError(
            f"BFCL generate refuses non-gold pack (tier={tier!r}). "
            "Re-run stage=prepare and fix oracle_validation failures."
        )
    pack = load_pack(config)
    current_fingerprint = pack_fingerprint(pack.paths)
    expected_endpoint_metadata = report.get("endpoint_metadata")
    if _endpoint_metadata(config, pack) != expected_endpoint_metadata:
        raise RuntimeError(
            "endpoint metadata changed after validation; refusing to generate from "
            "an oracle revision the gold report did not certify"
        )
    identity = current_identity(
        config,
        pack_fingerprint=current_fingerprint,
        endpoint_metadata=expected_endpoint_metadata,
        endpoint_config_path=pack.paths.endpoint_config_path,
    )
    state = restore_predecessor(config, target, identity=identity) if skip_until is not None else {}
    if skip_until is not None:
        clear_from_stage(config, target)
        # Pin the restored cache before any resumed stage can consume it. Append-only
        # caches stay unpinned exactly as on a full run, because a resumed stage may
        # still record a miss the earlier run never reached.
        for path in sorted(cache.iterdir()):
            if (
                path.is_file()
                and not path.name.endswith(".tmp")
                and path.name not in APPEND_ONLY_CACHES
                and path.name not in APPEND_ONLY_CACHE_LOCKS
            ):
                expected_artifact_hashes[path.name] = _artifact_hash(path)

    templates_by_id = {str(template["template_id"]): template for template in pack.templates}

    if "reference_profile" in stages[stages.index(target) :]:
        profile = run_reference_profile(config)
        state["profile"] = profile
        pin_artifacts("reference_samples.parquet", "reference_profile.json")
        write_checkpoint(
            config,
            "reference_profile",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
        )
    else:
        profile = state["profile"]

    if "expand" in stages[stages.index(target) :]:
        tasks = run_expand(config, pack)
        state["tasks"] = tasks
        state["canonical_expanded"] = len(tasks)
        pin_artifacts("task_instances.parquet", "held_out_bindings.json")
        write_checkpoint(
            config,
            "expand",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
        )
    else:
        tasks = state["tasks"]
    pin_artifacts("task_instances.parquet", "held_out_bindings.json")
    canonical_expanded = int(state["canonical_expanded"])

    if "state_machine" in stages[stages.index(target) :]:
        plans = run_state_machine(config, templates_by_id, tasks)
        state["plans"] = plans
        pin_artifacts("conversation_plans.parquet")
        write_checkpoint(
            config,
            "state_machine",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
        )
    else:
        plans = state["plans"]
    pin_artifacts("conversation_plans.parquet")

    if "render" in stages[stages.index(target) :]:
        surfaces, prompt_bundle = run_render(config, pack, templates_by_id, tasks, plans)
        pin_artifacts("rendered_conversations.parquet")
        tasks, plans, surfaces, paraphrase_report = run_paraphrase(
            config,
            pack,
            templates_by_id,
            tasks,
            plans,
            surfaces,
            profile,
        )
        pin_artifacts(
            "task_instances.parquet",
            "conversation_plans.parquet",
            "rendered_conversations.parquet",
            "paraphrase_rejections.json",
            "paraphrase_io_cache.jsonl",
        )
        state.update(
            {
                "tasks": tasks,
                "plans": plans,
                "surfaces": surfaces,
                "prompt_bundle": prompt_bundle,
                "paraphrase_report": paraphrase_report,
                "expanded_tasks": list(tasks),
                "expanded": len(tasks),
            }
        )
        write_checkpoint(
            config,
            "render",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
        )
    else:
        tasks = state["tasks"]
        plans = state["plans"]
        surfaces = state["surfaces"]
        prompt_bundle = state["prompt_bundle"]
        paraphrase_report = state["paraphrase_report"]
    pin_artifacts(
        "task_instances.parquet",
        "conversation_plans.parquet",
        "rendered_conversations.parquet",
        "paraphrase_rejections.json",
        "paraphrase_io_cache.jsonl",
    )
    expanded_tasks = list(state["expanded_tasks"])
    expanded = int(state["expanded"])
    # Drops an instance whose own fixture data cannot bind a trace, so `tasks` shrinks
    # to the kept set. Schema/replay still receive every expanded task so stage tables
    # keep a joinable row for each drop.
    if "expected_trace" in stages[stages.index(target) :]:
        traces, drop_reasons = run_expected_trace(config, pack, tasks, plans)
        state["traces"] = traces
        state["drop_reasons"] = drop_reasons
        pin_artifacts("expected_traces.parquet")
        write_checkpoint(
            config,
            "expected_trace",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
        )
    else:
        traces = state["traces"]
        drop_reasons = state["drop_reasons"]
    pin_artifacts("expected_traces.parquet")

    if "schema_validation" in stages[stages.index(target) :]:
        schema_failures = run_schema_validation(config, pack, expanded_tasks, traces, skipped=drop_reasons)
        state["schema_failures"] = schema_failures
        pin_artifacts("schema_validated_traces.parquet")
        write_checkpoint(
            config,
            "schema_validation",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
        )
    else:
        schema_failures = state["schema_failures"]
    pin_artifacts("schema_validated_traces.parquet")

    if "executable_replay" in stages[stages.index(target) :]:
        verdicts = run_executable_replay(
            config,
            pack,
            expanded_tasks,
            traces,
            schema_failures,
            skipped=drop_reasons,
        )
        paraphrase_report = apply_expected_result_guards(
            config,
            expanded_tasks,
            surfaces,
            verdicts,
            paraphrase_report,
        )
        state["verdicts"] = verdicts
        state["paraphrase_report"] = paraphrase_report
        pin_artifacts(
            "replay_validated_tasks.parquet",
            "paraphrase_rejections.json",
            "rendered_conversations.parquet",
        )
        write_checkpoint(
            config,
            "executable_replay",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
        )
    else:
        verdicts = state["verdicts"]
        paraphrase_report = state["paraphrase_report"]
    pin_artifacts(
        "replay_validated_tasks.parquet",
        "paraphrase_rejections.json",
        "rendered_conversations.parquet",
    )
    replay_tasks = [task for task in expanded_tasks if (verdicts.get(str(task["task_id"])) or {}).get("passed")]
    surface_quality_records: list[dict] | None = None
    surface_quality_report: dict | None = None
    if config.surface_quality_validation.get("enabled"):
        if "surface_quality" in stages[stages.index(target) :]:
            surface_quality_records, surface_quality_report = run_surface_quality_validation(
                config,
                replay_tasks,
                surfaces,
                profile=profile,
            )
            state["surface_quality_records"] = surface_quality_records
            state["surface_quality_report"] = surface_quality_report
            pin_artifacts(
                "surface_validated_tasks.parquet",
                "surface_quality_rejections.json",
                "surface_judge_cache_usage.json",
            )
            write_checkpoint(
                config,
                "surface_quality",
                state,
                identity=identity,
                artifact_names=tuple(expected_artifact_hashes),
            )
        else:
            surface_quality_records = state["surface_quality_records"]
            surface_quality_report = state["surface_quality_report"]
        pin_artifacts(
            "surface_validated_tasks.parquet",
            "surface_quality_rejections.json",
            "surface_judge_cache_usage.json",
        )
    dedup_balancing_result: dict | None = None
    if config.semantic_deduplication_config.get("enabled"):
        if surface_quality_records is None:
            raise RuntimeError("Stage 11 requires Stage 10 surface-quality records")
        quality_by_task = {str(record["task_id"]): record for record in surface_quality_records}
        stage_eleven_tasks = [
            task for task in replay_tasks if quality_by_task[str(task["task_id"])]["decision"] == "kept"
        ]
        stage_eleven_quality = [quality_by_task[str(task["task_id"])] for task in stage_eleven_tasks]
        if "dedup_balancing" in stages[stages.index(target) :]:
            dedup_balancing_result = run_dedup_balancing_stage(
                config,
                stage_eleven_tasks,
                surfaces,
                stage_eleven_quality,
            )
            state["dedup_balancing_result"] = dedup_balancing_result
            _verify_stage_artifact_matches_report(
                cache / "balanced_tasks.parquet",
                cache / "dedup_balancing_report.json",
            )
            pin_artifacts("balanced_tasks.parquet", "dedup_balancing_report.json")
            write_checkpoint(
                config,
                "dedup_balancing",
                state,
                identity=identity,
                artifact_names=tuple(expected_artifact_hashes),
            )
        else:
            dedup_balancing_result = state["dedup_balancing_result"]
        pin_artifacts("balanced_tasks.parquet", "dedup_balancing_report.json")
    current_fingerprint = pack_fingerprint(pack.paths)
    if current_fingerprint != report.get("pack_fingerprint"):
        raise RuntimeError(
            "oracle pack changed after validation; refusing to publish artifacts derived "
            "from content that the gold report did not certify"
        )
    if _endpoint_metadata(config, pack) != expected_endpoint_metadata:
        raise RuntimeError(
            "endpoint metadata changed during generation; refusing to publish artifacts from multiple oracle revisions"
        )

    # Each count reports its own stage, so a reader can tell a backend disagreement from
    # a paraphrase the guards rejected. ``published`` is where the stages meet.
    stage_counts = {
        "expanded": expanded,
        "canonical_expanded": canonical_expanded,
        "paraphrase_requested": int(paraphrase_report["requested_candidates"]),
        "paraphrase_accepted": int(paraphrase_report["accepted_candidates"]),
        "paraphrase_rejected": int(paraphrase_report["rejected_candidates"]),
        "surface_passed": sum(1 for surface in surfaces.values() if not surface["guard_violations"]),
        "trace_derived": len(traces),
        "trace_dropped": len(drop_reasons),
        "schema_passed": sum(
            1 for task_id, failures in schema_failures.items() if not failures and task_id not in drop_reasons
        ),
        "replay_passed": sum(1 for task in expanded_tasks if (verdicts.get(str(task["task_id"])) or {}).get("passed")),
    }
    if surface_quality_report is not None:
        stage_counts.update(
            {
                "surface_quality_evaluated": int(surface_quality_report["evaluated"]),
                "surface_quality_kept": int(surface_quality_report["kept"]),
                "surface_quality_dropped_python": int(surface_quality_report["dropped_by_python"]),
                "surface_quality_dropped_judge": int(surface_quality_report["dropped_by_surface_judge"]),
                "surface_quality_judge_errors": sum(
                    int(count) for count in surface_quality_report["judge_error_counts"].values()
                ),
            }
        )
    if dedup_balancing_result is not None:
        dedup_report = dedup_balancing_result["artifacts"]["report"]
        stage_counts.update(
            {
                "dedup_balancing_input": int(dedup_report["counts"]["stage_ten_survivors"]),
                "semantic_duplicates": int(dedup_report["counts"]["final_duplicates"]),
                "dedup_balancing_selected": int(dedup_report["counts"]["selected"]),
                "dedup_balancing_dropped": int(dedup_report["counts"]["dropped"]),
            }
        )
    state["stage_counts"] = stage_counts

    def checkpoint_staged_publication(staging_dir: Path) -> None:
        publication_paths = tuple(path for path in sorted(staging_dir.rglob("*")) if path.is_file())
        write_checkpoint(
            config,
            "final_output",
            state,
            identity=identity,
            artifact_names=tuple(expected_artifact_hashes),
            publication_paths=publication_paths,
            publication_root=staging_dir,
        )

    try:
        final_path = run_final_output(
            config,
            pack,
            tasks,
            surfaces,
            traces,
            verdicts,
            report,
            prompt_bundle,
            stage_counts,
            trace_drop_reasons=drop_reasons,
            surface_quality_records=surface_quality_records,
            surface_quality_report=surface_quality_report,
            dedup_balancing_decisions=(
                dedup_balancing_result["decisions"] if dedup_balancing_result is not None else None
            ),
            dedup_balancing_report=(
                dedup_balancing_result["artifacts"]["report"] if dedup_balancing_result is not None else None
            ),
            expected_pack_fingerprint=current_fingerprint,
            expected_artifact_hashes=expected_artifact_hashes,
            before_publication_commit=checkpoint_staged_publication,
        )
    except BaseException:
        # Stage 12 is complete only when its checkpoint can prove the publication.
        clear_from_stage(config, "final_output")
        _invalidate_publication_outputs(config)
        raise
    return final_path


def generate_bfcl(config_path: str | os.PathLike[str], *, skip_until: str | None = None) -> Path:
    """Run BFCL generation under an exclusive experiment-directory lock."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig

    config = BfclConfig.from_yaml(str(config_path))
    with _output_lock(config):
        return _generate_bfcl_unlocked(config_path, skip_until=skip_until)
