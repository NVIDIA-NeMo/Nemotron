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

"""Render deterministic surface text and enforce the per-template surface guards."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    RENDERED_CONVERSATIONS,
    rendered_conversation_row,
    rendered_conversations_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.templating import (
    PlaceholderError,
    placeholder_names,
    substitute,
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "default_system.txt"
# The one reserved must_not_mention entry; every other entry is a forbidden phrase.
TOOL_NAME_RULE = "tool_names"
# The placeholder a text milestone uses to name the slot it is asking about.
SLOT_NAME_PLACEHOLDER = "slot_name"
# The language the frozen fallback prompt is written in.
DEFAULT_PROMPT_LANGUAGE = "en"


class RenderError(ValueError):
    """Raised when a template cannot be rendered deterministically."""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_language(config: BfclConfig, pack: LoadedPack, available: list[str]) -> str:
    """Pick the surface language from config, then manifest, then the single option."""
    requested = config.surface_generation.get("language") or pack.manifest.get("default_language")
    if requested:
        if str(requested) not in available:
            raise RenderError(f"language {requested!r} is not available; pack offers {available}")
        return str(requested)
    languages = [str(item) for item in (pack.manifest.get("languages") or []) if str(item) in available]
    if languages:
        return languages[0]
    if len(available) == 1:
        return available[0]
    raise RenderError(f"cannot pick a surface language from {available}; set surface_generation.language")


def resolve_prompt_bundle(pack: LoadedPack) -> dict[str, Any]:
    """Resolve the system prompt text plus its bundle hash."""
    inline = pack.manifest.get("system_prompt")
    declared_path = pack.manifest.get("system_prompt_path")
    if isinstance(inline, str) and inline.strip():
        text = inline
        origin = "manifest.system_prompt"
    elif declared_path and pack.paths.system_prompt_path is not None:
        path = pack.paths.system_prompt_path
        text = path.read_text(encoding="utf-8")
        origin = str(path)
    else:
        text = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        origin = "bfcl/prompts/default_system.txt"
    text = text.strip()
    digest = _digest(text)
    return {
        "system_prompt": text,
        "system_prompt_id": f"sha256:{digest[0:16]}",
        "prompt_bundle_hash": f"sha256:{digest}",
        "origin": origin,
    }


def _substitute(text: str, values: dict[str, Any], *, what: str = "surface text") -> str:
    try:
        return substitute(text, values, what=what)
    except PlaceholderError as exc:
        raise RenderError(str(exc)) from exc


def _require_nonempty_text(text: str, *, what: str) -> str:
    """Reject pack text that would create an empty user-facing turn."""
    if not text.strip():
        raise RenderError(f"{what} rendered an empty user-facing turn")
    return text


def _localized(block: Any, language: str, what: str) -> str:
    if not isinstance(block, dict):
        raise RenderError(f"{what} must be a language mapping")
    if language in block:
        return str(block[language])
    raise RenderError(f"{what} has no entry for language {language!r}")


def _asked_slot(milestone: dict[str, Any], template: dict[str, Any], language: str) -> str:
    """Return how a text milestone should name the slot it is asking about.

    Naming the slot explicitly on the milestone is what lets a template ask for one of
    several withheld values; the single-hidden-slot shortcut covers the common case. A
    slot may carry a per-language ``label`` so the question reads as prose instead of
    quoting an identifier.
    """
    slots = template.get("slots") or {}
    declared = milestone.get("slot")
    if declared is not None:
        if str(declared) not in slots:
            raise RenderError(f"template {template.get('template_id')!r} asks for undeclared slot {declared!r}")
        name = str(declared)
    else:
        hidden = [name for name, slot in slots.items() if slot.get("visible_in_first_turn") is False]
        if len(hidden) != 1:
            raise RenderError(
                f"template {template.get('template_id')!r} uses {{{SLOT_NAME_PLACEHOLDER}}} in a "
                f"{milestone.get('type')!r} milestone but withholds {len(hidden)} slots; declare "
                "slot: <name> on the milestone so the question names one of them"
            )
        name = hidden[0]
    label = (slots.get(name) or {}).get("label")
    if label is None:
        return name
    return _localized(label, language, f"slot {name!r} label")


def _edge_guard(character: str, *, preceding: bool) -> str:
    """Return the lookaround that keeps ``character`` from merging into a longer token.

    The class to exclude follows the value's own edge. A number tolerates a unit written
    against it ("200000đ" states 200000) but not another digit, while a word must not sit
    inside a longer word ("status" does not state "us").
    """
    if character.isdigit():
        forbidden = r"\d"
    elif character.isalpha() or character == "_":
        forbidden = r"\w"
    else:
        return ""
    return rf"(?<!{forbidden})" if preceding else rf"(?!{forbidden})"


_GROUPED_NUMBER = re.compile(r"(?<!\d)\d{1,3}(?:[.,\u00a0\u202f ]\d{3})+(?!\d)")


def _mentions(haystack: str, value: str) -> bool:
    """Report whether ``value`` appears in ``haystack`` as its own value.

    A value found inside a longer token was never stated: an amount of 4 does not appear
    in "400000" and a status of "us" does not appear in "status". Reading either as
    present would report the wrong verdict for both guards that use this — a value the
    user never stated would satisfy ``must_preserve``, and a withheld one would look
    leaked to ``must_omit``.
    """
    if not value:
        return False
    if value.isdigit():
        # Natural-language amount formatting may add grouping separators without changing
        # the protected value (500000 -> 500.000 -> 500,000 -> 500 000). Only true
        # three-digit grouping counts: reading any digit run joined by punctuation as one
        # number would let "1, 2, 3" stand in for 123, which states no amount at all.
        for candidate in _GROUPED_NUMBER.findall(haystack):
            if re.sub(r"\D", "", candidate) == value:
                return True
    pattern = _edge_guard(value[0], preceding=True) + re.escape(value) + _edge_guard(value[-1], preceding=False)
    return re.search(pattern, haystack) is not None


def _mentions_name(haystack: str, name: str) -> bool:
    """Report whether a tool name appears as a word rather than inside a longer one."""
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", haystack) is not None


def mentions_value(haystack: str, value: str) -> bool:
    """Public value-aware matcher for deterministic post-replay surface guards."""
    return _mentions(haystack, value)


def _assistant_text_templates(pack: LoadedPack, template: dict[str, Any]) -> dict[str, Any]:
    pack_wide = pack.manifest.get("assistant_turn_templates") or {}
    overrides = template.get("assistant_turn_templates") or {}
    merged = dict(pack_wide)
    merged.update(overrides)
    return merged


def _language_gaps(pack: LoadedPack, template: dict[str, Any], language: str) -> list[str]:
    """Name the blocks of one template that carry no text for ``language``."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import TEXT_MILESTONES

    template_id = template.get("template_id")
    gaps: list[str] = []
    if language not in (template.get("user_turn_templates") or {}):
        gaps.append(f"template {template_id!r} user_turn_templates")
    for turn in template.get("user_simulator_turns") or []:
        if language not in (turn.get("content_template") or {}):
            gaps.append(f"template {template_id!r} user_simulator_turns after {turn.get('after')!r}")
    assistant_templates = _assistant_text_templates(pack, template)
    for milestone in template.get("assistant_milestones") or []:
        milestone_type = str(milestone.get("type"))
        if milestone_type not in TEXT_MILESTONES:
            continue
        block = milestone.get("content_template") or assistant_templates.get(milestone_type)
        # A missing block is a separate contract error that render_task reports with
        # the manifest key to declare, so only judge the languages of one that exists.
        if isinstance(block, dict) and language not in block:
            gaps.append(f"assistant_turn_templates.{milestone_type} for template {template_id!r}")
    return gaps


def check_surface_guards(
    template: dict[str, Any],
    task: dict[str, Any],
    user_texts: list[str],
    tool_names: list[str],
    *,
    preserve_slot_values: bool = True,
    prevent_tool_name_leakage: bool = True,
) -> list[dict[str, Any]]:
    """Re-check rendered user turns against the template's paraphrase guards.

    The two flags are run-wide defaults; a template's own ``must_preserve`` /
    ``must_not_mention`` entries are enforced on top of them. ``must_omit`` applies to
    the first turn only: a withheld slot may legitimately arrive in a later user
    reply, which is exactly what makes a withheld-slot conversation gold. A
    corrected slot must show every value the user stated, superseded ones included,
    otherwise the conversation would not explain the value the calls end up using.
    """
    paraphrase = template.get("paraphrase") or {}
    slots = template.get("slots") or {}
    bound = task.get("slots") or {}
    initial = task.get("slots_initial") or bound
    surface = " \n".join(user_texts)
    first_turn = user_texts[0] if user_texts else ""
    violations: list[dict[str, Any]] = []

    declared_preserve = set(paraphrase.get("must_preserve") or [])
    visible_slots = {name for name, slot in slots.items() if slot.get("visible_in_first_turn")}
    # The visibility flag describes the opening request specifically. Searching the
    # whole conversation would accept a template that withholds the value first and
    # only states it in a later simulator reply.
    auto_preserve = visible_slots if preserve_slot_values else set()
    if preserve_slot_values:
        for name in sorted(auto_preserve):
            if name in initial and not _mentions(first_turn, str(initial[name])):
                violations.append({"guard": "must_preserve", "slot": name})

    must_preserve = declared_preserve | auto_preserve
    for name in sorted(must_preserve):
        stated: list[Any] = []
        # An explicitly declared preserve rule also applies to an initial value that
        # is not governed by the run-wide visible-slot rule.
        if name in declared_preserve and name not in auto_preserve and name in initial:
            stated.append(initial[name])
        stated.extend(update["values"][name] for update in task.get("slot_updates") or [] if name in update["values"])
        if not stated and name in declared_preserve and name not in auto_preserve and name in bound:
            stated = [bound[name]]
        for value in dict.fromkeys(str(item) for item in stated):
            if not _mentions(surface, value):
                violations.append({"guard": "must_preserve", "slot": name})

    must_omit = set(paraphrase.get("must_omit") or [])
    must_omit.update(name for name, slot in slots.items() if slot.get("visible_in_first_turn") is False)
    for name in sorted(must_omit):
        if name in bound and _mentions(first_turn, str(bound[name])):
            violations.append({"guard": "must_omit", "slot": name})

    lowered = surface.lower()
    declared = [str(rule) for rule in paraphrase.get("must_not_mention") or []]
    # A declared phrase adds to the tool-name rule instead of replacing it: a template
    # that names one forbidden term must not thereby stop leak detection.
    if TOOL_NAME_RULE in declared or prevent_tool_name_leakage:
        violations.extend(
            {"guard": "must_not_mention", "tool": name} for name in tool_names if _mentions_name(lowered, name.lower())
        )
    violations.extend(
        {"guard": "must_not_mention", "phrase": phrase}
        for phrase in declared
        if phrase != TOOL_NAME_RULE and phrase.lower() in lowered
    )
    return violations


def render_task(
    pack: LoadedPack,
    template: dict[str, Any],
    task: dict[str, Any],
    plan: dict[str, Any],
    *,
    language: str,
    prompt_bundle: dict[str, Any],
    tool_names: list[str],
    preserve_slot_values: bool = True,
    prevent_tool_name_leakage: bool = True,
) -> dict[str, Any]:
    """Produce system/user/assistant text for one task without calling a model.

    Text is rendered with the values in force at that point in the conversation, so a
    turn that replaces a slot renders the old value before it and the new value after.
    """
    updates = {update["entry_index"]: update for update in task.get("slot_updates") or []}
    slots = dict(task.get("slots_initial") or task.get("slots") or {})
    assistant_templates = _assistant_text_templates(pack, template)
    user_texts: list[str] = []
    rendered_steps: list[dict[str, Any]] = []

    for step in plan["steps"]:
        if step["kind"] == "user":
            if step["source"] == "first_turn":
                block = template.get("user_turn_templates")
                text = _substitute(
                    _localized(block, language, f"template {template.get('template_id')!r} user_turn_templates"),
                    slots,
                )
            else:
                entry = step["turn"]
                update_index = step.get("update_index")
                values = dict(slots)
                if update_index is not None:
                    update = updates.get(update_index)
                    if update is None:
                        raise RenderError(
                            f"task {task['task_id']!r} plans a slot correction that expansion did not bind"
                        )
                    slots.update(update["values"])
                    # The alias names the replacement so the turn can read naturally
                    # while the canonical slot key stays the one calls bind from.
                    values = {**slots, **update["aliases"]}
                text = _substitute(
                    _localized(
                        entry.get("content_template"),
                        language,
                        f"user_simulator_turns after {entry.get('after')!r} content_template",
                    ),
                    values,
                )
            text = _require_nonempty_text(
                text,
                what=(f"template {template.get('template_id')!r} {step.get('source')!r} user turn"),
            )
            user_texts.append(text)
            rendered_steps.append({"kind": "user", "content": text})
        elif step["kind"] == "text":
            milestone_type = step["milestone_type"]
            block = step["milestone"].get("content_template") or assistant_templates.get(milestone_type)
            if block is None:
                raise RenderError(
                    f"no assistant_turn_templates entry for {milestone_type!r}; declare it on the pack "
                    "manifest or the template"
                )
            raw = _localized(block, language, f"assistant_turn_templates.{milestone_type}")
            values = dict(slots)
            if SLOT_NAME_PLACEHOLDER in placeholder_names(raw) and SLOT_NAME_PLACEHOLDER not in values:
                values[SLOT_NAME_PLACEHOLDER] = _asked_slot(step["milestone"], template, language)
            text = _require_nonempty_text(
                _substitute(raw, values),
                what=(f"template {template.get('template_id')!r} assistant milestone {milestone_type!r}"),
            )
            rendered_steps.append({"kind": "assistant_text", "content": text})
        else:
            rendered_steps.append({"kind": "calls", "call_group": step["call_group"]})

    violations = check_surface_guards(
        template,
        task,
        user_texts,
        tool_names,
        preserve_slot_values=preserve_slot_values,
        prevent_tool_name_leakage=prevent_tool_name_leakage,
    )
    return {
        "task_id": task["task_id"],
        "base_task_id": task["task_id"],
        "template_id": task["template_id"],
        "variant_index": int(task.get("variant_index", 0)),
        "source": "template",
        "language": language,
        "system_prompt": prompt_bundle["system_prompt"],
        "system_prompt_id": prompt_bundle["system_prompt_id"],
        "steps": rendered_steps,
        "guard_violations": violations,
        "paraphrase_model": None,
        "paraphrase_model_canonical": None,
        "profile_hash": None,
    }


def resolve_render_contract(
    config: BfclConfig,
    pack: LoadedPack,
    templates_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Settle the run-wide render inputs: language, system prompt, guarded tool names.

    These decisions depend on the pack and the config rather than on any one task, so
    validation resolves them before granting gold instead of letting a missing text
    block or an unlocalized prompt abort a run halfway through rendering.
    """
    available: set[str] = set()
    for template in pack.templates:
        available.update((template.get("user_turn_templates") or {}).keys())
    language = resolve_language(config, pack, sorted(available))
    # The opening turn is the only block every task renders, so it decides which
    # languages are on offer. Report the rest of the blocks now: a pack that states
    # its simulator turns in one language and its assistant turns in another should
    # hear about every gap at once, not one RenderError per run.
    gaps = sorted(gap for template in templates_by_id.values() for gap in _language_gaps(pack, template, language))
    if gaps:
        raise RenderError(
            f"no entry for language {language!r} in: "
            + ", ".join(gaps)
            + "; state these blocks in the render language or narrow languages on the manifest"
        )
    prompt_bundle = resolve_prompt_bundle(pack)
    if prompt_bundle["origin"].startswith("bfcl/prompts/") and language != DEFAULT_PROMPT_LANGUAGE:
        # A row whose system prompt and user turns disagree on language is not a clean
        # gold row. A smoke run says up front that it publishes nothing, so it may keep
        # exercising the pipeline while the pack's prompt is still missing.
        complaint = (
            f"language {language!r} cannot use the {DEFAULT_PROMPT_LANGUAGE!r} default system "
            "prompt in a gold row; declare manifest.system_prompt or system_prompt_path in "
            "the pack's render language"
        )
        if config.lineage.policy == "smoke_no_publication":
            logger.warning("BFCL render: %s", complaint)
        else:
            raise RenderError(complaint)
    tool_names = sorted(
        str((tool.get("function") or {}).get("name"))
        for tool in pack.tools
        if (tool.get("function") or {}).get("name")
    )
    # A tool named after an ordinary word of the domain language — "book", "transfer" —
    # would make every natural user turn look like a tool-name leak. A pack may exempt
    # such a name, and only a name it actually declares.
    guards = pack.manifest.get("surface_guards") or {}
    exempt = {str(name) for name in (guards.get("tool_names_exempt") or [])}
    unknown = exempt - set(tool_names)
    if unknown:
        raise RenderError(
            "manifest surface_guards.tool_names_exempt names tools the pack does not declare: "
            + ", ".join(sorted(unknown))
        )
    tool_names = [name for name in tool_names if name not in exempt]
    return {"language": language, "prompt_bundle": prompt_bundle, "tool_names": tool_names}


def run_render(
    config: BfclConfig,
    pack: LoadedPack,
    templates_by_id: dict[str, dict[str, Any]],
    tasks: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Render every task verbatim and drop rows whose surface breaks a guard."""
    contract = resolve_render_contract(config, pack, templates_by_id)
    language = contract["language"]
    prompt_bundle = contract["prompt_bundle"]
    tool_names = contract["tool_names"]

    surfaces: dict[str, dict[str, Any]] = {}
    for task in tasks:
        template = templates_by_id[str(task["template_id"])]
        surfaces[str(task["task_id"])] = render_task(
            pack,
            template,
            task,
            plans[str(task["task_id"])],
            language=language,
            prompt_bundle=prompt_bundle,
            tool_names=tool_names,
            preserve_slot_values=bool(config.surface_generation.get("preserve_slot_values", True)),
            prevent_tool_name_leakage=bool(config.surface_generation.get("prevent_tool_name_leakage", True)),
        )

    write_stage_table(
        stage_cache_dir(config) / RENDERED_CONVERSATIONS,
        [rendered_conversation_row(surfaces[str(task["task_id"])]) for task in tasks],
        rendered_conversations_schema(),
    )
    rejected = {
        task_id: surface["guard_violations"] for task_id, surface in surfaces.items() if surface["guard_violations"]
    }
    logger.info(
        "BFCL render produced %d surfaces (language=%s, guard rejections=%d)",
        len(surfaces),
        language,
        len(rejected),
    )
    return surfaces, prompt_bundle
