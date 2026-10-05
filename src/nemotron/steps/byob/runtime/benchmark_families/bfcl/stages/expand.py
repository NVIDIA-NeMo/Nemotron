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

"""Bind template slots against pack sources into locked task instances."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
from collections import Counter
from typing import Any, Literal

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.fixture_filter import evaluate_filter
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    HeldOutPolicy,
    fixture_ref,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    TASK_INSTANCES,
    task_instance_row,
    task_instances_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.held_out import (
    BindingLedger,
    held_out_policy,
    write_binding_report,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
    simulator_delivery_order,
)

logger = logging.getLogger(__name__)


def task_id_for(
    *,
    pack_id: str,
    pack_version: str,
    template_id: str,
    fixture_refs: list[str],
    slot_bindings: dict[str, Any],
    variant_index: int,
) -> str:
    payload = canonical_json(
        {
            "pack_id": pack_id,
            "pack_version": pack_version,
            "template_id": template_id,
            "fixture_refs": sorted(fixture_refs),
            "slot_bindings": slot_bindings,
            "variant_index": variant_index,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{pack_id}__{template_id}__{digest[0:16]}"


def task_seed_for(
    *,
    global_seed: int,
    pack_id: str,
    pack_version: str,
    template_id: str,
    fixture_refs: list[str],
    slot_bindings: dict[str, Any],
    variant_index: int,
) -> int:
    payload = canonical_json(
        {
            "global_seed": global_seed,
            "pack_id": pack_id,
            "pack_version": pack_version,
            "template_id": template_id,
            "fixture_refs": sorted(fixture_refs),
            "slot_bindings": slot_bindings,
            "variant_index": variant_index,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[0:8], byteorder="big", signed=False)


class ExpansionError(ValueError):
    """Raised when a template cannot be bound against the pack."""


def _permuted_indices(size: int, seed_material: Any) -> list[int]:
    """Return a deterministic full-cycle permutation without sorting scores."""
    if size <= 1:
        return list(range(size))
    digest = hashlib.sha256(canonical_json(seed_material).encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "big") % size
    step = int.from_bytes(digest[8:16], "big") % size or 1
    while math.gcd(step, size) != 1:
        step = (step + 1) % size or 1
    return [(start + offset * step) % size for offset in range(size)]


def _bounded_extend(
    combinations: list[list[tuple[Any, str | None]]],
    usable_by_combination: list[list[tuple[Any, str | None]]],
    *,
    limit: int,
    seed_material: Any,
) -> list[list[tuple[Any, str | None]]]:
    """Extend combinations with deterministic, coverage-oriented sampling."""
    cap = max(limit, 1)
    total = sum(len(candidates) for candidates in usable_by_combination)
    if total <= cap:
        return [
            combination + [candidate]
            for combination, candidates in zip(
                combinations,
                usable_by_combination,
                strict=True,
            )
            for candidate in candidates
        ]

    result: list[list[tuple[Any, str | None]]] = []
    combination_order = _permuted_indices(
        len(combinations),
        {"seed": seed_material, "kind": "combinations"},
    )
    candidate_orders = {
        index: _permuted_indices(
            len(usable_by_combination[index]),
            {"seed": seed_material, "kind": "candidates", "combination": index},
        )
        for index in combination_order
    }
    max_width = max(len(order) for order in candidate_orders.values())
    for round_index in range(max_width):
        for combination_index in combination_order:
            order = candidate_orders[combination_index]
            if round_index >= len(order):
                continue
            candidate = usable_by_combination[combination_index][order[round_index]]
            result.append(combinations[combination_index] + [candidate])
            if len(result) == cap:
                return result
    return result


def primary_key_for(manifest: dict[str, Any], collection: str, rows: list[dict[str, Any]]) -> str:
    """Resolve a collection's primary id field, declared or by convention.

    The key names the row a slot value came from, so guessing wrong attributes a task
    to the wrong record. Convention is only trusted when it is unambiguous — the
    collection's own ``<singular>_id`` or ``id`` — and a collection holding several
    ``*_id`` fields (its own plus a foreign key) must declare which one is its key.
    """
    declared = ((manifest.get("primary_keys") or {}) if isinstance(manifest, dict) else {}).get(collection)
    fields = next((list(row) for row in rows if isinstance(row, dict)), [])
    if isinstance(declared, str):
        if fields and declared not in fields:
            raise ExpansionError(
                f"manifest declares primary key {declared!r} for {collection!r}, which its rows do "
                f"not carry; rows hold {', '.join(fields)}"
            )
        return declared

    singular = collection[:-1] if collection.endswith("s") else collection
    for candidate in (f"{singular}_id", "id"):
        if candidate in fields:
            return candidate
    id_fields = [field for field in fields if field.endswith("_id")]
    if len(id_fields) == 1:
        return id_fields[0]
    raise ExpansionError(
        f"cannot tell which field identifies a row of {collection!r} (candidates: "
        f"{', '.join(id_fields) or 'none'}); declare primary_keys.{collection} in the manifest"
    )


def _tool_parameter(pack: LoadedPack, tool_name: str, param: str) -> dict[str, Any]:
    for tool in pack.tools:
        function = tool.get("function") or {}
        if function.get("name") == tool_name:
            return ((function.get("parameters") or {}).get("properties") or {}).get(param) or {}
    return {}


def _candidates(
    pack: LoadedPack,
    slot_name: str,
    slot: dict[str, Any],
    *,
    held_out: HeldOutPolicy | None = None,
    ledger: BindingLedger | None = None,
    fixture_selection: Literal["exclude", "include", "all"] = "exclude",
) -> list[tuple[Any, str | None]]:
    """Return ordered ``(value, fixture_ref)`` candidates for one slot.

    A held-out policy is applied here rather than after expansion because a task
    that never binds a reserved row cannot leak it downstream, and dropping the
    row afterwards would silently shrink the budget the pack asked for.
    """
    source = str(slot.get("source") or "")
    kind, _, rest = source.partition(":")
    if not rest:
        kind, rest = "fixture", source

    if kind == "fixture":
        collection, _, field = rest.partition(".")
        rows = (pack.fixtures or {}).get(collection)
        if not isinstance(rows, list):
            raise ExpansionError(f"slot {slot_name!r} references unknown collection {collection!r}")
        key = primary_key_for(pack.manifest, collection, rows)
        matched = [
            row
            for row in rows
            if isinstance(row, dict) and field in row and evaluate_filter(row, slot.get("filter"))
        ]
        if not matched:
            raise ExpansionError(f"slot {slot_name!r} filter matched zero rows in {collection!r}")
        missing = [index for index, row in enumerate(matched) if row.get(key) is None]
        if missing:
            raise ExpansionError(
                f"rows {missing} of {collection!r} carry no {key!r}, so the tasks they bind could not "
                "record which record they came from"
            )
        referenced = [(row, fixture_ref(collection, row[key])) for row in matched]
        if ledger is not None:
            ledger.examined(len(referenced))
        if held_out is not None and fixture_selection != "all":
            reserved = sorted(
                reference for _, reference in referenced if held_out.blocks_fixture(reference)
            )
            collection_has_reservations = (
                fixture_selection == "include"
                and any(
                    json.loads(reference)[0] == collection
                    for reference in getattr(held_out, "fixture_refs", ())
                )
            )
            if reserved:
                if ledger is not None:
                    ledger.blocked(reserved)
                if fixture_selection == "include":
                    referenced = [
                        (row, reference)
                        for row, reference in referenced
                        if held_out.blocks_fixture(reference)
                    ]
                else:
                    referenced = [
                        (row, reference)
                        for row, reference in referenced
                        if not held_out.blocks_fixture(reference)
                    ]
                if not referenced:
                    raise ExpansionError(
                        f"slot {slot_name!r} can bind no row of {collection!r}: the held-out policy "
                        f"reserves every row its filter matched ({', '.join(reserved)}); declare more "
                        "fixture rows or release some from held_out.fixtures"
                    )
            elif fixture_selection == "include" and collection_has_reservations:
                # This template simply contributes no fixture-held-out instance:
                # another template may expose the reserved rows through a compatible
                # filter, so private expansion must not turn this local miss into a
                # pack-wide failure.
                referenced = []
        return [(row[field], reference) for row, reference in referenced]

    if kind == "enum":
        tool_name, _, param = rest.partition(".")
        values = _tool_parameter(pack, tool_name, param).get("enum")
        if not values:
            raise ExpansionError(f"slot {slot_name!r} has no enum on {tool_name}.{param}")
        return [(value, None) for value in values]

    if kind == "literal":
        raw = rest.strip()
        if raw.startswith("["):
            values = ast.literal_eval(raw)
        else:
            # A bare literal keeps its Python type, so literal:200000 binds an integer
            # and passes an integer-typed tool parameter. Anything unparsable is text.
            try:
                values = [ast.literal_eval(raw)]
            except (ValueError, SyntaxError):
                values = [raw]
        if not values:
            raise ExpansionError(f"slot {slot_name!r} declares an empty literal set")
        return [(value, None) for value in values]

    if kind == "range":
        bounds = ast.literal_eval(rest.strip()) if rest.strip().startswith("{") else None
        if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
            raise ExpansionError(f"slot {slot_name!r} declares a degenerate range")
        start = bounds["min"]
        end = bounds["max"]
        step = bounds.get("step", 1)
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (start, end, step)
        ):
            raise ExpansionError(f"slot {slot_name!r} range min, max, and step must be integers")
        if step == 0:
            raise ExpansionError(f"slot {slot_name!r} range step must not be zero")
        if (end - start) * step < 0:
            raise ExpansionError(
                f"slot {slot_name!r} range step points away from max"
            )
        stop = end + (1 if step > 0 else -1)
        values = list(range(start, stop, step))
        if not values:
            raise ExpansionError(f"slot {slot_name!r} declares an empty range")
        return [(value, None) for value in values]

    if kind == "absent":
        collection = rest.strip()
        declared = (pack.manifest.get("absent_ids") or {}).get(collection)
        values = [declared] if isinstance(declared, str) else list(declared or [])
        if not values:
            raise ExpansionError(f"pack declares no absent id for collection {collection!r}")
        return [(value, None) for value in values]

    raise ExpansionError(f"slot {slot_name!r} uses unsupported source kind {kind!r}")


def _source_kind(source: Any) -> str:
    kind, _, rest = str(source or "").partition(":")
    return kind if rest else "fixture"


def declared_slot_updates(template: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    """Return ``(simulator_turn_index, slot_name, definition)`` for every replacement."""
    declared: list[tuple[int, str, dict[str, Any]]] = []
    for index, entry in enumerate(template.get("user_simulator_turns") or []):
        updates = entry.get("slot_updates")
        if updates is None:
            continue
        if not isinstance(updates, dict):
            raise ExpansionError(
                f"template {template.get('template_id')!r} declares slot_updates that is not a mapping"
            )
        for slot_name, definition in updates.items():
            if not isinstance(definition, dict) or not definition.get("source"):
                raise ExpansionError(
                    f"slot_updates for {slot_name!r} must declare the source its replacement resolves through"
                )
            declared.append((index, str(slot_name), definition))
    return declared


def _check_correction_contract(
    template: dict[str, Any], slots: dict[str, Any], updates: list[tuple[int, str, dict[str, Any]]]
) -> None:
    """Enforce the rules that keep a correction meaningful and bindable."""
    template_id = template.get("template_id")
    policy = str(template.get("turn_policy"))
    if updates and policy != "correction":
        raise ExpansionError(
            f"template {template_id!r} replaces a slot mid-conversation but declares turn_policy "
            f"{policy!r}; declare correction so the edge stays visible to balancing and scoring"
        )
    if policy == "correction" and not updates:
        raise ExpansionError(
            f"template {template_id!r} is correction but declares no slot_updates; a correction is a "
            "later user turn that replaces a value the user already gave"
        )
    for _, slot_name, definition in updates:
        slot = slots.get(slot_name)
        if slot is None:
            raise ExpansionError(
                f"template {template_id!r} replaces undeclared slot {slot_name!r}"
            )
        if not slot.get("visible_in_first_turn"):
            raise ExpansionError(
                f"template {template_id!r} replaces hidden slot {slot_name!r}; a correction can only "
                "replace a value the user already stated"
            )
        if _source_kind(definition.get("source")) != _source_kind(slot.get("source")):
            raise ExpansionError(
                f"template {template_id!r} replaces {slot_name!r} through source kind "
                f"{_source_kind(definition.get('source'))!r} instead of "
                f"{_source_kind(slot.get('source'))!r}; a replacement must resolve the same way"
            )


def _grouped_updates(
    updates: list[tuple[int, str, dict[str, Any]]],
    values: dict[tuple[int, str], Any],
    delivery_order: dict[int, int],
) -> list[dict[str, Any]]:
    """Order bound replacements by the turn that delivers them, not by declaration.

    Two corrections declared in one order can reach the user in another, and the value
    in force is whichever landed last in the conversation.
    """
    grouped: dict[int, dict[str, Any]] = {}
    aliases: dict[int, dict[str, Any]] = {}
    for entry_index, slot_name, definition in updates:
        value = values[(entry_index, slot_name)]
        grouped.setdefault(entry_index, {})[slot_name] = value
        alias = definition.get("bind_as")
        if alias:
            aliases.setdefault(entry_index, {})[str(alias)] = value
    return [
        {
            "entry_index": entry_index,
            "values": grouped[entry_index],
            "aliases": aliases.get(entry_index, {}),
        }
        for entry_index in sorted(grouped, key=lambda index: delivery_order.get(index, index))
    ]


def expand_template(
    pack: LoadedPack,
    template: dict[str, Any],
    limit: int,
    global_seed: int,
    *,
    held_out: HeldOutPolicy | None = None,
    ledger: BindingLedger | None = None,
    fixture_selection: Literal["exclude", "include", "all"] = "exclude",
    allow_held_out_template: bool = False,
) -> list[dict]:
    """Bind one template into at most ``limit`` deterministic task instances."""
    slots = template.get("slots") or {}
    slot_names = sorted(slots)
    updates = declared_slot_updates(template)
    _check_correction_contract(template, slots, updates)
    if (
        held_out is not None
        and held_out.blocks_template(template.get("template_id"))
        and not allow_held_out_template
    ):
        raise ExpansionError(
            f"template {template.get('template_id')!r} is reserved by the held-out policy and "
            "must not be bound"
        )

    delivery_order = simulator_delivery_order(template)
    ordered_updates = sorted(
        updates,
        key=lambda update: delivery_order.get(update[0], update[0]),
    )
    update_keys = [(entry_index, name) for entry_index, name, _ in ordered_updates]
    keys: list[Any] = [*slot_names, *update_keys]
    candidate_lists = [
        _candidates(
            pack,
            name,
            slots[name],
            held_out=held_out,
            ledger=ledger,
            fixture_selection=fixture_selection,
        )
        for name in slot_names
    ]
    candidate_lists.extend(
        _candidates(
            pack,
            f"{name} (correction)",
            definition,
            held_out=held_out,
            ledger=ledger,
            fixture_selection=fixture_selection,
        )
        for _, name, definition in ordered_updates
    )

    combinations: list[list[tuple[Any, str | None]]] = [[]]
    for position, candidates in enumerate(candidate_lists):
        # Drop a replacement equal to the value it replaces before capping, or the cap
        # would keep only the pairs that correct nothing and the template would look
        # unbindable. When every candidate is a restatement, keep them: a sibling slot
        # in the same correction turn may still change.
        replaced = keys[position][1] if isinstance(keys[position], tuple) else None
        held_at = None
        if replaced is not None:
            # Compare with the most recent value for this slot in conversation order,
            # which may itself be an earlier correction rather than the initial slot.
            for prior_position in range(position - 1, -1, -1):
                prior_key = keys[prior_position]
                prior_name = prior_key[1] if isinstance(prior_key, tuple) else prior_key
                if prior_name == replaced:
                    held_at = prior_position
                    break
        usable_by_combination: list[list[tuple[Any, str | None]]] = []
        for combo in combinations:
            if held_at is None:
                usable = candidates
            else:
                differing = [
                    candidate
                    for candidate in candidates
                    if combo[held_at][0] != candidate[0]
                ]
                usable = differing if differing else candidates
            usable_by_combination.append(usable)
        # Bound breadth without repeatedly retaining the row-major prefix. A
        # seeded full-cycle ordering spreads the budget across both existing
        # combinations and the new slot's values without materializing the full
        # Cartesian product.
        combinations = _bounded_extend(
            combinations,
            usable_by_combination,
            limit=limit,
            seed_material={
                "template_id": template.get("template_id"),
                    "seed": global_seed,
                "position": position,
            },
        )

    pack_id = str(pack.manifest.get("pack_id"))
    pack_version = str(pack.manifest.get("version"))
    tasks: list[dict[str, Any]] = []
    for combination in combinations[: max(limit, 1)]:
        raw = {key: value for key, (value, _) in zip(keys, combination, strict=True)}
        initial = {name: raw[name] for name in slot_names}
        replacements = {key: raw[key] for key in update_keys}
        bound_updates = _grouped_updates(ordered_updates, replacements, delivery_order)
        final = dict(initial)
        # A multi-slot correction turn is useful whenever any one slot changes. Treating
        # a single unchanged sibling as a no-op for the whole instance would discard
        # valid corrections that happen to restated another field.
        corrects_anything = False
        for update in bound_updates:
            for name, value in update["values"].items():
                if final.get(name) != value:
                    corrects_anything = True
            final.update(update["values"])
        if bound_updates and not corrects_anything:
            continue
        bindings = {
            **initial,
            **{f"{name}@correction{entry_index}": raw[(entry_index, name)] for entry_index, name in update_keys},
        }
        fixture_refs = sorted({ref for _, ref in combination if ref})
        tasks.append(
            {
                "slots_initial": initial,
                "slot_updates": bound_updates,
                "task_id": task_id_for(
                    pack_id=pack_id,
                    pack_version=pack_version,
                    template_id=str(template.get("template_id")),
                    fixture_refs=fixture_refs,
                    slot_bindings=bindings,
                    variant_index=0,
                ),
                "template_id": template.get("template_id"),
                "variant_index": 0,
                "seed": task_seed_for(
                    global_seed=global_seed,
                    pack_id=pack_id,
                    pack_version=pack_version,
                    template_id=str(template.get("template_id")),
                    fixture_refs=fixture_refs,
                    slot_bindings=bindings,
                    variant_index=0,
                ),
                "slots": final,
                "fixture_refs": fixture_refs,
                "pack_id": pack_id,
                "pack_version": pack_version,
                "intent": template.get("intent"),
                "category": template.get("category"),
                "difficulty": template.get("difficulty"),
                "turn_policy": template.get("turn_policy"),
                "call_order": template.get("call_order", "strict"),
                "call_order_prefix": template.get("call_order_prefix"),
                "required_tools": list(template.get("required_tools") or []),
                "tools_present": list(template.get("tools_present") or []),
                "success_assertions": list(template.get("success_assertions") or []),
                "edge_signatures": list(template.get("edge_signatures") or []),
                "mutates": bool(template.get("mutates", False)),
            }
        )
    if updates and not tasks:
        raise ExpansionError(
            f"template {template.get('template_id')!r} bound no correction instance because every "
            "replacement equalled the original; declare replacement values disjoint from the "
            "original source"
        )
    return tasks


def category_of(template: dict[str, Any]) -> str:
    """Return the balancing bucket a template expands into."""
    declared = template.get("category")
    return str(declared) if declared else str(template.get("template_id"))


def group_by_category(templates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket templates the way the expansion budget is spent."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for template in templates:
        grouped.setdefault(category_of(template), []).append(template)
    return grouped


def check_category_budgets(templates: list[dict[str, Any]], budget: int) -> None:
    """Raise when a category's budget cannot keep one instance per template.

    Validation calls this before granting gold so a pack does not learn about an
    unsatisfiable budget only once generation is underway.
    """
    for category, grouped in group_by_category(templates).items():
        if budget < len(grouped):
            raise ExpansionError(
                f"category {category!r} declares {len(grouped)} templates but "
                f"tasks_per_category is {budget}; raise the budget so every template "
                "keeps at least one instance"
            )


def _select_round_robin(pools: list[list[dict[str, Any]]], budget: int) -> list[dict[str, Any]]:
    """Take instances one per template per pass so every template stays represented."""
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < budget and any(depth < len(pool) for pool in pools):
        for pool in pools:
            if len(selected) >= budget:
                break
            if depth < len(pool):
                selected.append(pool[depth])
        depth += 1
    return selected


def run_expand(config: BfclConfig, pack: LoadedPack) -> list[dict[str, Any]]:
    """Expand every template and cache the locked task instances.

    ``candidate_tasks_per_category`` controls the Stage-4 inventory when
    declared; ``tasks_per_category`` remains the Stage-11 publication cap.
    Keeping these separate lets balancing choose a challenge mix from a larger
    deterministic pool instead of retaining every easy row.
    """
    selected_budget = int(
        config.task_generation.get("tasks_per_category", 1) or 1
    )
    budget = int(
        config.task_generation.get(
            "candidate_tasks_per_category",
            selected_budget,
        )
        or selected_budget
    )
    global_seed = int(config.random_seed or 0)

    policy = held_out_policy(pack)
    bindable = list(pack.templates)
    blocked_templates: list[str] = []
    if policy is not None:
        blocked_templates = sorted(
            str(template["template_id"])
            for template in bindable
            if policy.blocks_template(template["template_id"])
        )
        bindable = [
            template
            for template in bindable
            if not policy.blocks_template(template["template_id"])
        ]
        if not bindable:
            raise ExpansionError(
                "the held-out policy reserves every template in the pack, so generation would "
                "have nothing to publish; release a template or drop it from held_out.templates"
            )

    original_by_category = group_by_category(pack.templates)
    by_category = group_by_category(bindable)
    check_category_budgets(bindable, budget)

    tasks: list[dict[str, Any]] = []
    ledger = BindingLedger() if policy is not None else None
    for category, original_templates in original_by_category.items():
        templates = by_category.get(category, [])
        blocked_in_category = sorted(
            str(template["template_id"])
            for template in original_templates
            if policy is not None and policy.blocks_template(template["template_id"])
        )
        if not templates:
            raise ExpansionError(
                f"category {category!r} has no bindable template after the held-out policy "
                f"reserved {', '.join(blocked_in_category)}; declare enough non-held-out "
                "templates to retain the category"
            )
        # Tally per category, then merge: a row two categories both reserve is withheld
        # from each of them, and a shared tally would hide the second shortfall.
        category_ledger = BindingLedger() if policy is not None else None
        pools = [
            expand_template(
                pack,
                template,
                budget,
                global_seed,
                held_out=policy,
                ledger=category_ledger,
            )
            for template in templates
        ]
        selected = _select_round_robin(pools, budget)
        withheld = sorted(category_ledger.blocked_refs) if category_ledger is not None else []
        if ledger is not None and category_ledger is not None:
            ledger.examined(category_ledger.attempts)
            ledger.blocked(withheld)
        # A budget the pack cannot meet once its reservations are honoured is a pack
        # error, not a quiet reduction: publishing fewer rows would change the mix the
        # run claims without saying so.
        if (withheld or blocked_in_category) and len(selected) < budget:
            causes = [
                *(f"template:{template_id}" for template_id in blocked_in_category),
                *(f"fixture:{reference}" for reference in withheld),
            ]
            raise ExpansionError(
                f"category {category!r} bound {len(selected)} of {budget} instances once the "
                f"held-out policy withheld {', '.join(causes)}; declare enough templates "
                "and fixtures to meet tasks_per_category while keeping held-out reserved"
            )
        tasks.extend(selected)
    task_ids = [str(task["task_id"]) for task in tasks]
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        raise ExpansionError(f"expansion produced duplicate task_id values: {duplicates}")

    write_stage_table(
        stage_cache_dir(config) / TASK_INSTANCES,
        [task_instance_row(task) for task in tasks],
        task_instances_schema(),
    )
    if policy is not None and ledger is not None:
        write_binding_report(
            config,
            policy,
            blocked_templates=blocked_templates,
            blocked_fixture_refs=sorted(ledger.blocked_refs),
            bind_attempts=ledger.attempts,
            tasks_expanded=len(tasks),
        )
        logger.info(
            "BFCL expand honoured a held-out policy: %d template(s) and %d fixture row(s) withheld "
            "across %d binding attempts",
            len(blocked_templates),
            len(ledger.blocked_refs),
            ledger.attempts,
        )
    logger.info(
        "BFCL expand produced %d task instances across %d categories",
        len(tasks),
        len(by_category),
    )
    return tasks
