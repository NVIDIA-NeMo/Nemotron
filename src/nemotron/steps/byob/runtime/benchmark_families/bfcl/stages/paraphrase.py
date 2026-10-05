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

"""Create guarded surface variants without changing locked task semantics."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
    request_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack
from nemotron.steps.byob.runtime.benchmark_families.bfcl.response_model import (
    ParaphraseResult,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    CONVERSATION_PLANS,
    RENDERED_CONVERSATIONS,
    TASK_INSTANCES,
    conversation_plan_row,
    conversation_plans_schema,
    rendered_conversation_row,
    rendered_conversations_schema,
    task_instance_row,
    task_instances_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
    task_id_for,
    task_seed_for,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import (
    TOOL_NAME_RULE,
    check_surface_guards,
    mentions_value,
    resolve_render_contract,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_styles import (
    SURFACE_STYLE_AXES,
    style_plan,
)

logger = logging.getLogger(__name__)

PARAPHRASE_PROMPT_VERSION = "bfcl-controlled-paraphrase-v2"
PARAPHRASE_SYSTEM_PROMPT = """Rewrite only user-facing wording.
Do not add, remove, infer, normalize, or alter protected values or conversation turns.
Write variant i in the sentence form and register named by surface_styles[i]; a style
selects how the request is phrased, never what it asks for or which values it carries.
Follow style_hints and avoid every pattern in style_avoid when those lists are supplied.
Do not mention forbidden implementation names. Return only the requested structured variants."""
PARAPHRASE_PROMPT = """Create the requested ordered variants from this canonical JSON contract:
{{ model_input }}"""
ParaphraseRunner = Callable[..., dict[str, dict[str, Any]]]
_LITERAL = re.compile(r"(?<!\w)(?:[A-Z]{2,}[-_][A-Z0-9_-]+|\d[\d.,:/-]*)(?!\w)")


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def resolve_style_axes(config: BfclConfig) -> tuple[str, ...]:
    """Return the axes this run requests, letting a pack profile declare its own."""
    declared = config.surface_generation.get("surface_style_axes")
    if declared:
        return tuple(str(axis) for axis in declared)
    return SURFACE_STYLE_AXES


def _identity_bindings(task: dict[str, Any]) -> dict[str, Any]:
    bindings = dict(task.get("slots_initial") or task.get("slots") or {})
    for update in task.get("slot_updates") or []:
        entry_index = update["entry_index"]
        for name, value in update.get("values", {}).items():
            bindings[f"{name}@correction{entry_index}"] = value
    return bindings


def _variant_task(
    config: BfclConfig,
    task: dict[str, Any],
    variant_index: int,
) -> dict[str, Any]:
    variant = deepcopy(task)
    arguments = {
        "pack_id": str(task["pack_id"]),
        "pack_version": str(task["pack_version"]),
        "template_id": str(task["template_id"]),
        "fixture_refs": list(task.get("fixture_refs") or []),
        "slot_bindings": _identity_bindings(task),
        "variant_index": variant_index,
    }
    variant["task_id"] = task_id_for(**arguments)
    variant["variant_index"] = variant_index
    variant["seed"] = task_seed_for(
        global_seed=int(config.random_seed or 0),
        **arguments,
    )
    variant["base_task_id"] = task["task_id"]
    return variant


def _guard_model_input(
    template: dict[str, Any],
    task: dict[str, Any],
    user_turns: list[str],
    *,
    language: str,
    requested_variants: int,
    tool_names: list[str],
    surface_styles: list[str],
    style_hints: list[str],
    style_avoid: list[str],
) -> dict[str, Any]:
    paraphrase = template.get("paraphrase") or {}
    slots = template.get("slots") or {}
    initial = task.get("slots_initial") or task.get("slots") or {}
    preserve_names = set(paraphrase.get("must_preserve") or [])
    preserve_names.update(
        name
        for name, definition in slots.items()
        if definition.get("visible_in_first_turn") is True
    )
    protected_values = [
        str(initial[name])
        for name in sorted(preserve_names)
        if name in initial
    ]
    for update in task.get("slot_updates") or []:
        protected_values.extend(
            str(update["values"][name])
            for name in sorted(set(update.get("values") or {}) & preserve_names)
        )
    forbidden = [
        str(value)
        for value in paraphrase.get("must_not_mention") or []
        if value != TOOL_NAME_RULE
    ]
    forbidden.extend(tool_names)
    # Slot values the template withholds are deliberately absent from this contract: the
    # canonical turn never states them, so a model that cannot see them cannot leak them.
    # The must_omit guard still checks the rewrite in case the model invents one.
    return {
        "language": language,
        "canonical_user_turns": user_turns,
        "surface_styles": surface_styles,
        "style_hints": style_hints,
        "style_avoid": style_avoid,
        "must_preserve": list(dict.fromkeys(protected_values)),
        "must_not_mention": list(dict.fromkeys(forbidden)),
        "requested_variants": requested_variants,
    }


def _canonical_literal(value: str) -> str:
    return re.sub(r"\D", "", value) if value[:1].isdigit() else value.lower()


def _novel_literals(canonical: list[str], candidate: list[str]) -> list[str]:
    existing = {
        _canonical_literal(match.group(0))
        for text in canonical
        for match in _LITERAL.finditer(text)
    }
    return sorted(
        {
            match.group(0)
            for text in candidate
            for match in _LITERAL.finditer(text)
            if _canonical_literal(match.group(0)) not in existing
        }
    )


def _paraphrase_contract_error(response: Any, requested: int) -> str | None:
    """Return why a response cannot be safely reused as structured paraphrases."""
    if not isinstance(response, dict):
        return "response_not_object"
    variants = response.get("variants")
    if not isinstance(variants, list):
        return "variants_not_list"
    if len(variants) != requested:
        return "variant_count_mismatch"
    for index, candidate in enumerate(variants):
        if not isinstance(candidate, dict):
            return f"variant_{index}_not_object"
        user_turns = candidate.get("user_turns")
        if not isinstance(user_turns, list) or any(
            not isinstance(turn, str) for turn in user_turns
        ):
            return f"variant_{index}_user_turns_not_string_list"
    return None


def _candidate_surface(
    canonical: dict[str, Any],
    *,
    task: dict[str, Any],
    template: dict[str, Any],
    user_turns: list[str],
    tool_names: list[str],
    variant_index: int,
    model_config: dict[str, Any],
    profile_hash: str | None,
    preserve_slot_values: bool,
    prevent_tool_name_leakage: bool,
) -> dict[str, Any]:
    surface = deepcopy(canonical)
    canonical_user_turns = [
        str(step["content"]) for step in canonical["steps"] if step["kind"] == "user"
    ]
    violations: list[dict[str, Any]] = []
    if len(user_turns) != len(canonical_user_turns):
        violations.append(
            {
                "guard": "semantic_shape",
                "reason": "user_turn_count_changed",
                "expected": len(canonical_user_turns),
                "actual": len(user_turns),
            }
        )
    elif any(not isinstance(text, str) or not text.strip() for text in user_turns):
        violations.append(
            {"guard": "semantic_shape", "reason": "empty_user_turn"}
        )
    else:
        cleaned_user_turns = [text.strip() for text in user_turns]
        replacements = iter(cleaned_user_turns)
        for step in surface["steps"]:
            if step["kind"] == "user":
                step["content"] = next(replacements)
        if cleaned_user_turns == canonical_user_turns:
            violations.append(
                {"guard": "semantic_shape", "reason": "unchanged_surface"}
            )
        violations.extend(
            check_surface_guards(
                template,
                task,
                cleaned_user_turns,
                tool_names,
                preserve_slot_values=preserve_slot_values,
                prevent_tool_name_leakage=prevent_tool_name_leakage,
            )
        )
        for value in _novel_literals(canonical_user_turns, cleaned_user_turns):
            # A literal the canonical turn never stated is a new fact, whatever its
            # source; the post-replay guard names the narrower case it can prove.
            violations.append({"guard": "novel_literal", "value": value})
    surface.update(
        {
            "task_id": task["task_id"],
            "base_task_id": task["base_task_id"],
            "variant_index": variant_index,
            "source": "model",
            "guard_violations": violations,
            "paraphrase_model": model_config.get("alias"),
            "paraphrase_model_canonical": model_config.get("canonical_id"),
            "profile_hash": profile_hash,
        }
    )
    return surface


def _write_report(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _scalar_result_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [
            item
            for child in value.values()
            for item in _scalar_result_values(child)
        ]
    if isinstance(value, list):
        return [item for child in value for item in _scalar_result_values(child)]
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (str, int, float)):
        text = str(value)
        return [text] if text else []
    return []


def apply_expected_result_guards(
    config: BfclConfig,
    tasks: list[dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Reject a model surface that states a replay result absent from its base."""
    for task in tasks:
        task_id = str(task["task_id"])
        surface = surfaces[task_id]
        if surface.get("source") != "model" or surface["guard_violations"]:
            continue
        verdict = verdicts.get(task_id) or {}
        if not verdict.get("passed"):
            continue
        base_surface = surfaces[str(surface["base_task_id"])]
        base_text = "\n".join(
            str(step["content"])
            for step in base_surface["steps"]
            if step["kind"] == "user"
        )
        candidate_text = "\n".join(
            str(step["content"])
            for step in surface["steps"]
            if step["kind"] == "user"
        )
        leaked = sorted(
            {
                value
                for result in verdict.get("results") or []
                for value in _scalar_result_values(result)
                if not mentions_value(base_text, value)
                and mentions_value(candidate_text, value)
            }
        )
        if not leaked:
            continue
        violations = [
            {"guard": "expected_result_leakage", "value": value}
            for value in leaked
        ]
        surface["guard_violations"].extend(violations)
        template_id = str(task["template_id"])
        event = {
            "base_task_id": str(surface["base_task_id"]),
            "template_id": template_id,
            "variant_index": int(task["variant_index"]),
            "reason": "expected_result_leakage",
            "detail": violations,
            "count": 1,
        }
        report.setdefault("events", []).append(event)
        report["accepted_candidates"] = int(report.get("accepted_candidates", 0)) - 1
        report["rejected_candidates"] = int(report.get("rejected_candidates", 0)) + 1
        by_reason = report.setdefault("by_reason", {})
        by_reason["expected_result_leakage"] = (
            int(by_reason.get("expected_result_leakage", 0)) + 1
        )
        template_counts = report.setdefault("by_template", {}).setdefault(
            template_id,
            {"requested": 0, "accepted": 0, "rejected": 0},
        )
        template_counts["accepted"] -= 1
        template_counts["rejected"] += 1

    cache = stage_cache_dir(config)
    write_stage_table(
        cache / RENDERED_CONVERSATIONS,
        [
            rendered_conversation_row(surfaces[str(task["task_id"])])
            for task in tasks
        ],
        rendered_conversations_schema(),
    )
    report["by_reason"] = dict(sorted(report.get("by_reason", {}).items()))
    _write_report(cache / "paraphrase_rejections.json", report)
    return report


def run_paraphrase(
    config: BfclConfig,
    pack: LoadedPack,
    templates_by_id: dict[str, dict[str, Any]],
    tasks: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    *,
    model_runner: ParaphraseRunner | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Fan out accepted model surfaces while retaining every canonical variant."""
    enabled = bool(config.surface_generation.get("model_paraphrase_enabled"))
    requested_default = int(
        config.surface_generation.get("paraphrases_per_template", 0)
    )
    contract = resolve_render_contract(config, pack, templates_by_id)
    tool_names = contract["tool_names"]
    style_axes = resolve_style_axes(config)
    role = (config.lineage.roles or {}).get("paraphrase")
    model_config = dict(role.model_config or {}) if role else {}
    if model_config.get("canonical_id"):
        model_config["canonical_id"] = (
            str(model_config["canonical_id"]).strip().lower()
        )
    profile_available = bool(
        enabled
        and config.lineage.profile_influenced_surface
        and profile.get("status") == "completed"
    )
    style_hints = list(profile.get("style_hints") or []) if profile_available else []
    style_avoid = list(profile.get("avoid") or []) if profile_available else []
    profile_hash = str(profile.get("output_hash")) if profile_available else None
    io_cache = ImmutableModelIOCache(
        stage_cache_dir(config) / "paraphrase_io_cache.jsonl"
    )
    prompt_hash = _sha256(
        PARAPHRASE_PROMPT_VERSION
        + "\n"
        + PARAPHRASE_SYSTEM_PROMPT
        + "\n"
        + PARAPHRASE_PROMPT
    )

    output_tasks = list(tasks)
    output_plans = dict(plans)
    output_surfaces = dict(surfaces)
    pending: list[dict[str, Any]] = []
    rejection_events: list[dict[str, Any]] = []
    if enabled:
        assert role is not None and role.enabled
        for task in tasks:
            template = templates_by_id[str(task["template_id"])]
            paraphrase = template.get("paraphrase") or {}
            if paraphrase.get("allowed") is not True:
                continue
            requested = requested_default
            max_variants = paraphrase.get("max_variants")
            if isinstance(max_variants, int) and not isinstance(max_variants, bool):
                requested = min(requested, max(0, max_variants))
            if requested == 0:
                continue
            base_id = str(task["task_id"])
            canonical_user_turns = [
                str(step["content"])
                for step in surfaces[base_id]["steps"]
                if step["kind"] == "user"
            ]
            model_input = _guard_model_input(
                template,
                task,
                canonical_user_turns,
                language=str(surfaces[base_id]["language"]),
                requested_variants=requested,
                tool_names=tool_names,
                surface_styles=style_plan(task, requested, style_axes),
                style_hints=style_hints,
                style_avoid=style_avoid,
            )
            key = request_hash(
                model_canonical=str(model_config["canonical_id"]),
                prompt_hash=prompt_hash,
                model_input=model_input,
                inference_parameters=dict(
                    model_config.get("inference_parameters") or {}
                ),
                output_schema=ParaphraseResult.model_json_schema(),
                seed=int(task["seed"]),
            )
            pending.append(
                {
                    "key": key,
                    "task": task,
                    "template": template,
                    "model_input": model_input,
                    "input_json": canonical_json(model_input),
                    "requested": requested,
                    "response": io_cache.get(key),
                }
            )

    # A profile that no eligible template consumes cannot make a surface inconsistent.
    # Delay this gate until after eligibility/max-variant filtering, and compare
    # normalized BCP-47 tags because their spelling is case-insensitive.
    if profile_available and pending:
        profile_languages = {
            str(language).strip().casefold()
            for language in profile.get("languages") or []
        }
        surface_languages = {
            str(surfaces[str(item["task"]["task_id"])]["language"])
            .strip()
            .casefold()
            for item in pending
        }
        if len(profile_languages) != 1 or not surface_languages <= profile_languages:
            raise ValueError(
                "reference profile language must match every rendered surface language "
                f"(profile={sorted(profile_languages)}, surfaces={sorted(surface_languages)})"
            )

    missing = [item for item in pending if item["response"] is None]
    if missing:
        if model_runner is None:
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_runner import (
                run_structured_model,
            )

            model_runner = run_structured_model
        def request_batch(batch: list[dict[str, Any]]) -> None:
            """Fill in the responses of one batch, attributing failures per request."""
            try:
                responses = model_runner(
                    config,
                    stage_name="paraphrase",
                    model_config=model_config,
                    requests=[
                        {
                            "request_id": item["key"],
                            "model_input": item["input_json"],
                        }
                        for item in batch
                    ],
                    system_prompt=PARAPHRASE_SYSTEM_PROMPT,
                    prompt=PARAPHRASE_PROMPT,
                    output_format=ParaphraseResult,
                )
            except Exception as exc:  # noqa: BLE001 - canonical rows must survive
                # A failed call is an infrastructure event, not a model observation. It
                # stays out of the immutable cache so a fixed endpoint can be retried;
                # the rejection report is what records what produced nothing. One
                # failing request must not discard the variants of the rest of its
                # batch, so a failed batch is retried one request at a time.
                if len(batch) > 1:
                    for item in batch:
                        request_batch([item])
                    return
                batch[0]["response"] = {"_model_error": type(exc).__name__}
                return
            for item in batch:
                response = responses.get(item["key"])
                if response is None:
                    item["response"] = {"_model_error": "missing_response"}
                    continue
                item["response"] = response
                if _paraphrase_contract_error(response, int(item["requested"])) is None:
                    io_cache.put(
                        item["key"],
                        response,
                        model_canonical=str(model_config["canonical_id"]),
                        input_hash=_sha256(item["input_json"]),
                    )

        batch_size = max(1, int(config.ndd_batch_size))
        for start in range(0, len(missing), batch_size):
            request_batch(missing[start : start + batch_size])

    for item in pending:
        task = item["task"]
        base_id = str(task["task_id"])
        response = item["response"]
        if isinstance(response, dict) and response.get("_model_error"):
            rejection_events.append(
                {
                    "base_task_id": base_id,
                    "template_id": str(task["template_id"]),
                    "reason": "model_error",
                    "detail": response["_model_error"],
                    "count": item["requested"],
                }
            )
            continue
        contract_error = _paraphrase_contract_error(response, int(item["requested"]))
        if contract_error is not None:
            rejection_events.append(
                {
                    "base_task_id": base_id,
                    "template_id": str(task["template_id"]),
                    "reason": "model_contract",
                    "detail": contract_error,
                    "count": item["requested"],
                }
            )
            continue
        variants = response["variants"]
        # Each variant of one binding is asked for a different style, so two identical
        # variants add no wording and only consume the diversity budget of a surface.
        accepted_turns: set[tuple[str, ...]] = set()
        for variant_index, candidate in enumerate(variants, 1):
            user_turns = (
                candidate.get("user_turns") if isinstance(candidate, dict) else None
            )
            if not isinstance(user_turns, list):
                user_turns = []
            variant_task = _variant_task(config, task, variant_index)
            variant_id = str(variant_task["task_id"])
            variant_plan = deepcopy(plans[base_id])
            variant_plan["task_id"] = variant_id
            variant_surface = _candidate_surface(
                surfaces[base_id],
                task=variant_task,
                template=item["template"],
                user_turns=user_turns,
                tool_names=tool_names,
                variant_index=variant_index,
                model_config=model_config,
                profile_hash=profile_hash,
                preserve_slot_values=bool(
                    config.surface_generation.get("preserve_slot_values", True)
                ),
                prevent_tool_name_leakage=bool(
                    config.surface_generation.get(
                        "prevent_tool_name_leakage",
                        True,
                    )
                ),
            )
            if variant_surface["guard_violations"]:
                primary = variant_surface["guard_violations"][0]
                rejection_events.append(
                    {
                        "base_task_id": base_id,
                        "template_id": str(task["template_id"]),
                        "variant_index": variant_index,
                        "reason": str(primary.get("guard", "unknown")),
                        "detail": variant_surface["guard_violations"],
                        "count": 1,
                    }
                )
                continue
            variant_turns = tuple(
                str(step["content"])
                for step in variant_surface["steps"]
                if step["kind"] == "user"
            )
            if variant_turns in accepted_turns:
                rejection_events.append(
                    {
                        "base_task_id": base_id,
                        "template_id": str(task["template_id"]),
                        "variant_index": variant_index,
                        "reason": "semantic_shape",
                        "detail": [
                            {
                                "guard": "semantic_shape",
                                "reason": "duplicate_variant_surface",
                            }
                        ],
                        "count": 1,
                    }
                )
                continue
            accepted_turns.add(variant_turns)
            output_tasks.append(variant_task)
            output_plans[variant_id] = variant_plan
            output_surfaces[variant_id] = variant_surface

    cache = stage_cache_dir(config)
    write_stage_table(
        cache / TASK_INSTANCES,
        [task_instance_row(task) for task in output_tasks],
        task_instances_schema(),
    )
    write_stage_table(
        cache / CONVERSATION_PLANS,
        [
            conversation_plan_row(task, output_plans[str(task["task_id"])])
            for task in output_tasks
        ],
        conversation_plans_schema(),
    )
    write_stage_table(
        cache / RENDERED_CONVERSATIONS,
        [
            rendered_conversation_row(output_surfaces[str(task["task_id"])])
            for task in output_tasks
        ],
        rendered_conversations_schema(),
    )
    by_reason = Counter()
    by_template: dict[str, dict[str, int]] = {}
    for item in pending:
        template_counts = by_template.setdefault(
            str(item["task"]["template_id"]),
            {"requested": 0, "accepted": 0, "rejected": 0},
        )
        template_counts["requested"] += int(item["requested"])
    accepted_candidates = 0
    # Count what a variant is rather than where it sits in the list, so accounting stays
    # correct no matter how a later stage orders canonical rows against their variants.
    for task in output_tasks:
        if int(task.get("variant_index", 0)) == 0:
            continue
        accepted_candidates += 1
        by_template.setdefault(
            str(task["template_id"]),
            {"requested": 0, "accepted": 0, "rejected": 0},
        )["accepted"] += 1
    for event in rejection_events:
        count = int(event["count"])
        by_reason[str(event["reason"])] += count
        template_counts = by_template.setdefault(
            str(event["template_id"]),
            {"requested": 0, "accepted": 0, "rejected": 0},
        )
        template_counts["rejected"] += count
    requested_candidates = sum(int(item["requested"]) for item in pending)
    report = {
        "enabled": enabled,
        # The profile only counts as consumed once a surface it shaped survives the
        # guards; a request that every candidate failed carried it nowhere.
        "profile_consumed": profile_available and accepted_candidates > 0,
        "requested_candidates": requested_candidates,
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": requested_candidates - accepted_candidates,
        "by_reason": dict(sorted(by_reason.items())),
        "by_template": dict(sorted(by_template.items())),
        "events": rejection_events,
    }
    _write_report(cache / "paraphrase_rejections.json", report)
    logger.info(
        "BFCL paraphrase kept %d canonical and %d model variants",
        len(tasks),
        accepted_candidates,
    )
    return output_tasks, output_plans, output_surfaces, report
