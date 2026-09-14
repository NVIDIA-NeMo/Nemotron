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

"""Project Stage 10 survivors into the text Stage 11 clusters on.

Only user-authored turns reach the projection: assistant milestones, tool-call
payloads, and oracle results stay out, so a near-duplicate is decided on what a
person actually said. Slot literals are masked by their pack-declared slot name,
so two tasks that differ only in a bound id collapse into one cluster instead of
surviving as separate publications.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_balancing_contract import (
    BALANCING_DIMENSIONS,
    DEDUP_BALANCING_CONTRACT_VERSION,
    DedupBalancingDecision,
    Stage11Coverage,
    validate_complete_decision_set,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    BALANCED_TASKS,
    balanced_tasks_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.surface_quality import (
    user_facing_turns,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.surface_quality_contract import (
    SurfaceQualityCheckResult,
    validate_complete_check_set,
)

USER_TURN_MARKER = "[user]"
# The backend embeds text and reports which ids are near-duplicates. It is a
# parameter so a run can be reproduced, and tested, without standing up Ray.
DuplicateFinder = Callable[..., Mapping[str, Any]]
DEDUP_BALANCING_REPORT = "dedup_balancing_report.json"


class DedupBalancingPolicyError(RuntimeError):
    """Stage 11 completed diagnostics but policy forbids publication."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_id(task: Mapping[str, Any]) -> str:
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("a Stage 11 projection requires a non-empty task_id")
    return task_id.strip()


def user_turn_texts(surface: Mapping[str, Any]) -> list[str]:
    """Return the user turns in conversation order, whitespace normalized.

    Whitespace is collapsed because a paraphrase that only re-wraps a line is the
    same surface to a reader, and an embedding should not treat it as new.
    """
    return [" ".join(turn["content"].split()) for turn in user_facing_turns(surface) if turn["role"] == "user"]


def _bound_values(task: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Collect every value the pack bound into this task, correction included."""
    values: list[tuple[str, Any]] = []
    for key in ("slots_initial", "slots"):
        bound = task.get(key)
        if bound is None:
            continue
        if not isinstance(bound, Mapping):
            raise ValueError(f"task {_task_id(task)!r} has a {key} that is not a mapping")
        values.extend((str(name), value) for name, value in bound.items())
    updates = task.get("slot_updates")
    if updates is None:
        updates = []
    if not isinstance(updates, Sequence) or isinstance(updates, str | bytes):
        raise ValueError(f"task {_task_id(task)!r} has slot_updates that is not a list")
    for update in updates:
        if not isinstance(update, Mapping):
            raise ValueError(f"task {_task_id(task)!r} has a slot update that is not a mapping")
        for key in ("values", "aliases"):
            replaced = update.get(key)
            if replaced is None:
                replaced = {}
            if not isinstance(replaced, Mapping):
                raise ValueError(f"task {_task_id(task)!r} has slot update {key} that is not a mapping")
            values.extend((str(name), value) for name, value in replaced.items())
    return values


def slot_literals(task: Mapping[str, Any]) -> dict[str, str]:
    """Map each literal this task bound to the slot name that masks it.

    A literal two slots share is attributed to the first slot name in sort order,
    so masking never depends on mapping iteration order.
    """
    literals: dict[str, str] = {}
    for slot_name, value in _bound_values(task):
        literal = str(value)
        if not literal.strip():
            continue
        owner = literals.get(literal)
        if owner is None or slot_name < owner:
            literals[literal] = slot_name
    return literals


def _bounded(literal: str) -> str:
    """Match a literal only as a whole token, so short values cannot corrupt words."""
    # Numeric slots are commonly rendered next to a localized unit (``500000đ``,
    # ``20kg``). A word boundary would leave the value unmasked because Unicode
    # letters and digits are both ``\w``. Digit-only values instead guard against
    # adjacent digits, which still prevents matching a prefix of a larger number.
    if literal.isdigit():
        return r"(?<!\d)" + re.escape(literal) + r"(?!\d)"
    prefix = r"(?<!\w)" if _is_word_char(literal[0]) else ""
    suffix = r"(?!\w)" if _is_word_char(literal[-1]) else ""
    return prefix + re.escape(literal) + suffix


def _is_word_char(character: str) -> bool:
    return character.isalnum() or character == "_"


# Thousands separators seen across locales a pack may render in: dot, comma, plain
# and non-breaking spaces, and the Swiss apostrophes. Only uniform three-digit groups
# are recognized, so a locale with mixed group widths keeps its value unmasked rather
# than risking a wrong match on unrelated digits.
_GROUPED_NUMBER = re.compile(r"(?<!\d)\d{1,3}(?:[.,\u00a0\u202f '\u2019]\d{3})+(?!\d)")


def mask_slot_literals(text: str, task: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Replace bound literals with ``<slot_name>``, returning the slots masked."""
    literals = slot_literals(task)
    if not literals:
        return text, []
    # One alternation pass, longest literal first: a substituted placeholder can
    # never be re-matched by a shorter literal, and a value that contains another
    # value keeps its own slot name.
    ordered = sorted(literals, key=lambda literal: (-len(literal), literal))
    pattern = re.compile("|".join(_bounded(literal) for literal in ordered))
    masked: set[str] = set()

    def replace_grouped_number(match: re.Match[str]) -> str:
        normalized = re.sub(r"\D", "", match.group(0))
        slot_name = literals.get(normalized)
        if slot_name is None or not normalized.isdigit():
            return match.group(0)
        masked.add(slot_name)
        return f"<{slot_name}>"

    text = _GROUPED_NUMBER.sub(replace_grouped_number, text)

    def replace(match: re.Match[str]) -> str:
        slot_name = literals[match.group(0)]
        masked.add(slot_name)
        return f"<{slot_name}>"

    return pattern.sub(replace, text), sorted(masked)


def project_surface_text(task: Mapping[str, Any], surface: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Render the canonical embedding text for one survivor and the slots masked."""
    texts = user_turn_texts(surface)
    if not texts:
        raise ValueError(f"task {_task_id(task)!r} projects no user turn; Stage 11 cannot embed an empty surface")
    lines: list[str] = []
    masked: set[str] = set()
    for text in texts:
        line, slots = mask_slot_literals(text, task)
        masked.update(slots)
        lines.append(f"{USER_TURN_MARKER} {line}")
    return "\n".join(lines), sorted(masked)


# What a row executes: the request, the values it carries, the tools and policy it
# runs under, and the assertions that decide it. Identity, provenance, declared
# labels, and which rendering of a case a row is are deliberately excluded, which is
# what lets a paraphrase share the identity of its canonical task.
EXECUTION_CASE_KEYS: tuple[str, ...] = (
    "intent",
    "category",
    "slots_initial",
    "slots",
    "slot_updates",
    "fixture_refs",
    "required_tools",
    "tools_present",
    "success_assertions",
    "turn_policy",
    "call_order",
    "call_order_prefix",
    "num_tool_calls",
    "has_user_confirmation",
    "confirmed_call_turns",
    "edge_signatures",
)


def execution_case_hash(task: Mapping[str, Any]) -> str:
    """Hash executable meaning separately from the masked linguistic surface.

    Model paraphrases intentionally share this identity with their canonical
    task, while different fixture bindings, call policies, assertions, or
    distractor sets remain distinct evaluation cases.
    """
    return _sha256(canonical_json({key: task.get(key) for key in EXECUTION_CASE_KEYS}))


def project_dedup_text(task: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    """Project one Stage 10 survivor into the record Stage 11 embeds."""
    text, masked = project_surface_text(task, surface)
    return {
        "task_id": _task_id(task),
        "text": text,
        "text_hash": _sha256(text),
        "execution_case_hash": execution_case_hash(task),
        "num_user_turns": text.count(f"{USER_TURN_MARKER} "),
        "masked_slots": masked,
    }


def project_dedup_texts(
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project every survivor in input order, one record per task."""
    projected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        task_id = _task_id(task)
        if task_id in seen:
            raise ValueError(f"duplicate Stage 11 projection input for task {task_id!r}")
        seen.add(task_id)
        surface = surfaces.get(task_id)
        if surface is None:
            raise ValueError(f"task {task_id!r} reached Stage 11 without a surface")
        projected.append(project_dedup_text(task, surface))
    return projected


# How often one selected group may repeat, and which bound to name when a cap keeps a
# run away from its declared publication target. Each entry pairs a balancing feature
# with the setting that bounds it, so a pack limits repeated wording, repeated requests,
# and repeated executable meaning through one mechanism instead of three.
GROUP_REUSE_CAPS: tuple[tuple[str, str, str], ...] = (
    ("surface_text_hash", "max_exact_surface_reuse", "insufficient_surface_diversity"),
    ("intent", "max_rows_per_intent", "intent_cap_limits_inventory"),
    (
        "execution_case_hash",
        "max_execution_case_reuse",
        "execution_case_cap_limits_inventory",
    ),
)


@dataclass(frozen=True)
class DedupSettings:
    """The embedding and clustering settings one run is pinned to."""

    model_identifier: str
    n_clusters: int
    eps: float
    remove_duplicates: bool
    max_exact_surface_reuse: int | None
    min_exact_surface_ratio: float | None
    max_rows_per_intent: int | None
    max_execution_case_reuse: int | None
    representative_source_preference: tuple[str, ...]
    unmet_target_policy: str

    def as_lineage(self) -> dict[str, Any]:
        return {
            "contract_version": DEDUP_BALANCING_CONTRACT_VERSION,
            "model_identifier": self.model_identifier,
            "n_clusters": self.n_clusters,
            "eps": self.eps,
            "remove_duplicates": self.remove_duplicates,
            "max_exact_surface_reuse": self.max_exact_surface_reuse,
            "min_exact_surface_ratio": self.min_exact_surface_ratio,
            "max_rows_per_intent": self.max_rows_per_intent,
            "max_execution_case_reuse": self.max_execution_case_reuse,
            "representative_source_preference": list(self.representative_source_preference),
            "unmet_target_policy": self.unmet_target_policy,
        }

    @property
    def group_caps(self) -> dict[str, int]:
        """Return the declared per-group ceilings keyed by balancing feature."""
        return {
            feature: cap for feature, attribute, _ in GROUP_REUSE_CAPS if (cap := getattr(self, attribute)) is not None
        }

    @property
    def settings_hash(self) -> str:
        return _sha256(canonical_json(self.as_lineage()))


def resolve_dedup_settings(config: BfclConfig) -> DedupSettings:
    """Read the locked Stage 11 settings; config validation already required them."""
    dedup = config.semantic_deduplication_config or {}
    if not dedup.get("enabled"):
        raise ValueError("Stage 11 requires semantic_deduplication_config.enabled")
    unmet_target_policy = dedup.get("unmet_target_policy", "abort")
    if not isinstance(unmet_target_policy, str) or unmet_target_policy not in {
        "abort",
        "publish_non_gold",
    }:
        raise ValueError("Stage 11 unmet_target_policy must be 'abort' or 'publish_non_gold'")
    return DedupSettings(
        model_identifier=str(dedup["model_identifier"]),
        n_clusters=int(dedup["n_clusters"]),
        eps=float(dedup["eps"]),
        remove_duplicates=bool(dedup["remove_duplicates"]),
        max_exact_surface_reuse=(
            int(dedup["max_exact_surface_reuse"]) if dedup.get("max_exact_surface_reuse") is not None else None
        ),
        min_exact_surface_ratio=(
            float(dedup["min_exact_surface_ratio"]) if dedup.get("min_exact_surface_ratio") is not None else None
        ),
        max_rows_per_intent=(
            int(dedup["max_rows_per_intent"]) if dedup.get("max_rows_per_intent") is not None else None
        ),
        max_execution_case_reuse=(
            int(dedup["max_execution_case_reuse"]) if dedup.get("max_execution_case_reuse") is not None else None
        ),
        representative_source_preference=tuple(dedup.get("representative_source_preference") or ("template", "model")),
        unmet_target_policy=unmet_target_policy,
    )


def effective_n_clusters(configured: int, row_count: int) -> int:
    """Choose a valid k while retaining candidates for pairwise comparison.

    Curator compares rows only inside a K-means partition. Setting ``k`` equal
    to the row count makes every distinct row a likely singleton and turns
    semantic deduplication into a no-op, so non-trivial inputs retain an average
    of at least two rows per partition.
    """
    if configured < 1:
        raise ValueError("semantic_deduplication_config.n_clusters must be positive")
    if row_count < 0:
        raise ValueError("Stage 11 row_count cannot be negative")
    if row_count == 0:
        return 0
    return min(configured, max(1, row_count // 2))


def reconcile_curator_pairwise_artifacts(
    *,
    task_ids: set[str],
    pairs: Sequence[tuple[str, str, float]],
    duplicate_ids: Sequence[str],
    eps: float,
) -> dict[str, str]:
    """Recover clusters from the Curator decisions that produced duplicates."""
    if not 0.0 < eps < 1.0:
        raise ValueError("Curator cosine dedup eps must be between 0 and 1")
    pair_ids = [task_id for task_id, _, _ in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Curator pairwise output contains repeated ids")
    if set(pair_ids) != task_ids:
        missing = sorted(task_ids - set(pair_ids))
        extra = sorted(set(pair_ids) - task_ids)
        raise ValueError(f"Curator pairwise output must cover embedded ids exactly (missing={missing}, extra={extra})")
    parent = {task_id: task_id for task_id in task_ids}

    def find(task_id: str) -> str:
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    threshold = 1.0 - eps
    pairwise_duplicate_ids: set[str] = set()
    for task_id, predecessor, score in pairs:
        if not math.isfinite(score):
            raise ValueError(f"Curator pairwise output has a non-finite score for task {task_id!r}")
        if score < threshold:
            continue
        if predecessor not in task_ids:
            raise ValueError(f"Curator pairwise output links task {task_id!r} to unknown predecessor {predecessor!r}")
        if predecessor == task_id:
            raise ValueError(f"Curator pairwise output marks task {task_id!r} duplicate of itself")
        pairwise_duplicate_ids.add(task_id)
        union(task_id, predecessor)

    duplicate_set = set(duplicate_ids)
    if len(duplicate_set) != len(duplicate_ids):
        raise ValueError("Curator duplicate output contains repeated ids")
    if duplicate_set != pairwise_duplicate_ids:
        missing = sorted(pairwise_duplicate_ids - duplicate_set)
        extra = sorted(duplicate_set - pairwise_duplicate_ids)
        raise ValueError(
            "Curator duplicate and pairwise artifacts disagree "
            f"(missing_from_duplicates={missing}, extra_in_duplicates={extra})"
        )
    return {task_id: find(task_id) for task_id in sorted(task_ids)}


def _validate_finder_result(
    result: Mapping[str, Any],
    *,
    task_ids: Sequence[str],
) -> tuple[list[str], dict[str, str], str, dict[str, dict[str, Any]]]:
    """Check a backend result before any of it can decide a publication."""
    if not isinstance(result, Mapping):
        raise ValueError("a Stage 11 duplicate finder must return a mapping")
    missing_keys = sorted({"duplicate_ids", "cluster_by_id", "embedding_signature"} - set(result))
    if missing_keys:
        raise ValueError("a Stage 11 duplicate finder result is missing keys: " + ", ".join(missing_keys))
    signature = result["embedding_signature"]
    if not isinstance(signature, str) or not signature.strip():
        raise ValueError("a Stage 11 duplicate finder must report a non-empty embedding_signature")
    cluster_by_id = result["cluster_by_id"]
    if not isinstance(cluster_by_id, Mapping):
        raise ValueError("a Stage 11 duplicate finder must return cluster_by_id as a mapping")
    if set(cluster_by_id) != set(task_ids):
        missing = [task_id for task_id in task_ids if task_id not in cluster_by_id]
        extra = sorted(set(cluster_by_id) - set(task_ids))
        raise ValueError(f"Stage 11 clusters must cover the embedded ids exactly (missing={missing}, extra={extra})")
    clusters = {}
    for task_id, label in cluster_by_id.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"task {task_id!r} has an empty Stage 11 cluster label")
        clusters[str(task_id)] = label.strip()
    duplicate_ids = result["duplicate_ids"]
    if isinstance(duplicate_ids, str) or not isinstance(duplicate_ids, Sequence):
        raise ValueError("a Stage 11 duplicate finder must return duplicate_ids as a list")
    duplicates = [str(task_id) for task_id in duplicate_ids]
    if len(set(duplicates)) != len(duplicates):
        raise ValueError("a Stage 11 duplicate finder repeated a duplicate id")
    if unknown := sorted(set(duplicates) - set(task_ids)):
        raise ValueError("a Stage 11 duplicate finder reported unknown ids: " + ", ".join(unknown))
    members: dict[str, list[str]] = {}
    for task_id, label in sorted(clusters.items()):
        members.setdefault(label, []).append(task_id)
    duplicate_set = set(duplicates)
    for task_id in sorted(duplicate_set):
        if len(members[clusters[task_id]]) < 2:
            raise ValueError(f"task {task_id!r} is marked duplicate but sits alone in its cluster")
    for label, cluster_members in sorted(members.items()):
        representatives = [member for member in cluster_members if member not in duplicate_set]
        if len(representatives) != 1:
            raise ValueError(
                f"cluster {label!r} must have exactly one non-duplicate representative, got {len(representatives)}"
            )
    pairwise_by_id: dict[str, dict[str, Any]] = {}
    raw_pairwise = result.get("pairwise_by_id")
    if raw_pairwise is not None:
        if not isinstance(raw_pairwise, Mapping) or set(raw_pairwise) != set(task_ids):
            raise ValueError("a Stage 11 duplicate finder pairwise_by_id must cover embedded ids exactly")
        known_ids = set(task_ids)
        for task_id, raw in raw_pairwise.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"task {task_id!r} pairwise metadata must be a mapping")
            predecessor = raw.get("predecessor_id")
            score = raw.get("similarity_score")
            if not isinstance(predecessor, str) or predecessor not in known_ids:
                raise ValueError(f"task {task_id!r} pairwise metadata has an unknown predecessor")
            if not isinstance(score, int | float) or isinstance(score, bool) or not math.isfinite(float(score)):
                raise ValueError(f"task {task_id!r} pairwise metadata requires a finite similarity_score")
            pairwise_by_id[str(task_id)] = {
                "predecessor_id": predecessor,
                "similarity_score": float(score),
            }
    return sorted(duplicates), clusters, signature, pairwise_by_id


def _exact_diversity(
    projected: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    values = [str(record[key]) for record in projected if record.get(key) is not None]
    if len(values) != len(projected):
        return None
    counts = Counter(values)
    total = len(values)
    unique = len(counts)
    return {
        "total": total,
        "unique": unique,
        "duplicate_rows": total - unique,
        "unique_ratio": (unique / total) if total else 1.0,
        "max_reuse": max(counts.values(), default=0),
    }


def run_semantic_dedup(
    config: BfclConfig,
    projected: Sequence[Mapping[str, Any]],
    *,
    finder: DuplicateFinder | None = None,
) -> dict[str, Any]:
    """Embed the projections and record which survivors are near-duplicates.

    Embeddings only decide duplication. Nothing a model produces reaches a task's
    text, calls, arguments, or assertions.
    """
    settings = resolve_dedup_settings(config)
    task_ids: list[str] = []
    for record in projected:
        task_id = record.get("task_id")
        text = record.get("text")
        text_hash = record.get("text_hash")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("a Stage 11 projected record requires a non-empty task_id")
        if not isinstance(text, str) or not text:
            raise ValueError(f"projected record for task {task_id!r} requires non-empty text")
        expected_text_hash = _sha256(text)
        if text_hash != expected_text_hash:
            raise ValueError(f"projected record for task {task_id!r} has a text_hash that does not match its text")
        task_ids.append(task_id)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Stage 11 cannot embed the same task twice")
    input_hash = _sha256(
        canonical_json(
            [
                {"task_id": task_id, "text_hash": record["text_hash"]}
                for task_id, record in zip(task_ids, projected, strict=True)
            ]
        )
    )
    lineage = {
        "settings": settings.as_lineage(),
        "settings_hash": settings.settings_hash,
        "input_hash": input_hash,
        "input_count": len(task_ids),
        "effective_n_clusters": effective_n_clusters(settings.n_clusters, len(task_ids)),
        "exact_surface_diversity": _exact_diversity(projected, "text_hash"),
        "execution_case_diversity": _exact_diversity(
            projected,
            "execution_case_hash",
        ),
    }
    # One row cannot duplicate anything, and zero rows have nothing to embed, so
    # neither case may pay for an embedding model or a k-means with k > n.
    if len(task_ids) < 2:
        return {
            **lineage,
            "embedded": False,
            "embedding_signature": None,
            "duplicate_ids": [],
            "clusters": {task_id: task_id for task_id in task_ids},
            "records": [
                {
                    "task_id": task_id,
                    "cluster_id": task_id,
                    "is_duplicate": False,
                    "text_hash": str(projected[index]["text_hash"]),
                }
                for index, task_id in enumerate(task_ids)
            ],
        }
    if finder is None:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.dedup_curator import (
            curator_duplicate_finder,
        )

        finder = curator_duplicate_finder
    result = finder(
        config=config,
        settings=settings,
        n_clusters=lineage["effective_n_clusters"],
        rows=[{"id": str(record["task_id"]), "text": str(record["text"])} for record in projected],
    )
    duplicates, clusters, signature, pairwise_by_id = _validate_finder_result(result, task_ids=task_ids)
    duplicate_set = set(duplicates)
    return {
        **lineage,
        "embedded": True,
        "embedding_signature": signature,
        "duplicate_ids": duplicates,
        "clusters": clusters,
        "records": [
            {
                "task_id": task_id,
                "cluster_id": clusters[task_id],
                "is_duplicate": task_id in duplicate_set,
                "text_hash": str(projected[index]["text_hash"]),
                **(
                    {
                        "curator_predecessor_id": pairwise_by_id[task_id]["predecessor_id"],
                        "curator_similarity_score": pairwise_by_id[task_id]["similarity_score"],
                    }
                    if task_id in pairwise_by_id
                    else {}
                ),
            }
            for index, task_id in enumerate(task_ids)
        ],
    }


def derive_stage11_coverage(
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    *,
    edge_signatures_by_task_id: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Stage11Coverage]:
    """Derive the coverage partition without any pack-specific branching."""
    task_ids = [_task_id(task) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Stage 11 coverage input task_id values must be unique")
    normalized_edges: dict[str, Sequence[str]] | None = None
    if edge_signatures_by_task_id is not None:
        normalized_edges = {}
        for raw_task_id, signatures in edge_signatures_by_task_id.items():
            if not isinstance(raw_task_id, str) or not raw_task_id.strip():
                raise ValueError("Stage 11 edge signature keys must be non-empty task ids")
            task_id = raw_task_id.strip()
            if task_id in normalized_edges:
                raise ValueError(f"Stage 11 edge signatures repeated task {task_id!r} after normalization")
            normalized_edges[task_id] = signatures
        if set(normalized_edges) != set(task_ids):
            missing = [task_id for task_id in task_ids if task_id not in normalized_edges]
            extra = sorted(set(normalized_edges) - set(task_ids))
            raise ValueError(f"Stage 11 edge signatures must cover inputs exactly (missing={missing}, extra={extra})")
    coverage: dict[str, Stage11Coverage] = {}
    for task, task_id in zip(tasks, task_ids, strict=True):
        surface = surfaces.get(task_id)
        if surface is None:
            raise ValueError(f"task {task_id!r} reached Stage 11 without a surface")
        language = surface.get("language")
        turn_policy = task.get("turn_policy")
        raw_edges: Any
        if normalized_edges is not None:
            raw_edges = normalized_edges[task_id]
        else:
            raw_edges = task.get("edge_signatures")
            if raw_edges is None:
                raw_edges = ()
        if isinstance(raw_edges, str) or not isinstance(raw_edges, Sequence):
            raise ValueError(f"task {task_id!r} edge_signatures must be a list")
        coverage[task_id] = Stage11Coverage(
            language=language,
            turn_policy=turn_policy,
            edge_signatures=tuple(raw_edges),
        )
    return coverage


def _string_list(task: Mapping[str, Any], key: str) -> list[str]:
    value = task.get(key)
    if value is None:
        value = []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"task {_task_id(task)!r} {key} must be a list")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"task {_task_id(task)!r} {key} must contain non-empty values")
    return sorted(set(normalized))


def capability_signature(task: Mapping[str, Any]) -> str:
    """Hash the generic executable capabilities that may not be collapsed."""
    call_order = task.get("call_order", "strict")
    if not isinstance(call_order, str) or not call_order.strip():
        raise ValueError(f"task {_task_id(task)!r} requires a non-empty call_order")
    mutates = task.get("mutates", False)
    if not isinstance(mutates, bool):
        raise ValueError(f"task {_task_id(task)!r} mutates must be a boolean")
    payload = {
        "required_tools": _string_list(task, "required_tools"),
        "tools_present": _string_list(task, "tools_present"),
        "success_assertions": _string_list(task, "success_assertions"),
        "mutates": mutates,
        "call_order": call_order.strip(),
        "call_order_prefix": task.get("call_order_prefix"),
    }
    return _sha256(canonical_json(payload))


def _records_by_task(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_task_ids: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    by_task: dict[str, Mapping[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"{label} records require non-empty task_id values")
        task_id = task_id.strip()
        if task_id in by_task:
            raise ValueError(f"{label} repeated task {task_id!r}")
        by_task[task_id] = record
    expected = set(expected_task_ids)
    if set(by_task) != expected:
        missing = [task_id for task_id in expected_task_ids if task_id not in by_task]
        extra = sorted(set(by_task) - expected)
        raise ValueError(f"{label} must cover Stage 11 inputs exactly (missing={missing}, extra={extra})")
    return by_task


def _representative_rank(
    *,
    task_id: str,
    task: Mapping[str, Any],
    surface: Mapping[str, Any],
    quality: Mapping[str, Any],
    coverage_frequency: int,
    source_preference: Sequence[str],
    seed: int,
    hard_limited: bool,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if quality.get("decision") != "kept":
        raise ValueError(f"task {task_id!r} is not a Stage 10 survivor")
    turn_policy = task.get("turn_policy")
    if not isinstance(turn_policy, str):
        raise ValueError(f"task {task_id!r} requires turn_policy for representative selection")
    if quality.get("turn_policy") != turn_policy:
        raise ValueError(f"task {task_id!r} Stage 10 turn_policy does not match its task")
    raw_checks = quality.get("checks")
    if isinstance(raw_checks, str) or not isinstance(raw_checks, Sequence):
        raise ValueError(f"task {task_id!r} requires Stage 10 checks")
    checks = validate_complete_check_set(
        [
            item if isinstance(item, SurfaceQualityCheckResult) else SurfaceQualityCheckResult.model_validate(item)
            for item in raw_checks
        ],
        turn_policy=turn_policy,
    )
    advisory = quality.get("advisory_failures") or []
    if isinstance(advisory, str) or not isinstance(advisory, Sequence):
        raise ValueError(f"task {task_id!r} advisory_failures must be a list")
    judge_problem = bool(quality.get("judge_error") is not None or advisory)
    applicable_failures = sum(check.status in {"failed", "error"} for check in checks)
    surface_source = surface.get("source") or "template"
    quality_source = quality.get("surface_source")
    if quality_source is not None and quality_source != surface_source:
        raise ValueError(f"task {task_id!r} Stage 10 surface source does not match its surface")
    source = surface_source
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"task {task_id!r} requires a surface source")
    source = source.strip()
    source_rank = source_preference.index(source) if source in source_preference else len(source_preference)
    seeded_tie_break = _sha256(canonical_json({"seed": seed, "task_id": task_id}))
    details = {
        "hard_limited": hard_limited,
        "judge_problem": judge_problem,
        "applicable_failure_count": applicable_failures,
        "coverage_frequency": coverage_frequency,
        "surface_source": source,
        "source_preference_rank": source_rank,
        "seeded_tie_break": seeded_tie_break,
    }
    return (
        (
            int(hard_limited),
            int(judge_problem),
            applicable_failures,
            coverage_frequency,
            source_rank,
            seeded_tie_break,
            task_id,
        ),
        details,
    )


def select_duplicate_representatives(
    config: BfclConfig,
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    quality_records: Sequence[Mapping[str, Any]],
    semantic_result: Mapping[str, Any],
    *,
    coverage_by_task_id: Mapping[str, Stage11Coverage | Mapping[str, object]] | None = None,
    edge_signatures_by_task_id: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[DedupBalancingDecision], list[dict[str, Any]]]:
    """Partition Curator groups and deterministically select one representative."""
    settings = resolve_dedup_settings(config)
    semantic_settings_hash = semantic_result.get("settings_hash")
    if semantic_settings_hash is not None and semantic_settings_hash != settings.settings_hash:
        raise ValueError("semantic dedup result settings_hash does not match the Stage 11 config")
    task_ids = [_task_id(task) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Stage 11 representative input task_id values must be unique")
    task_by_id = {task_id: task for task_id, task in zip(task_ids, tasks, strict=True)}
    if coverage_by_task_id is None:
        coverage = derive_stage11_coverage(
            tasks,
            surfaces,
            edge_signatures_by_task_id=edge_signatures_by_task_id,
        )
    else:
        if edge_signatures_by_task_id is not None:
            raise ValueError("provide coverage_by_task_id or edge_signatures_by_task_id, not both")
        coverage = {}
        for raw_task_id, value in coverage_by_task_id.items():
            if not isinstance(raw_task_id, str) or not raw_task_id.strip():
                raise ValueError("Stage 11 coverage keys must be non-empty task ids")
            task_id = raw_task_id.strip()
            if task_id in coverage:
                raise ValueError(f"Stage 11 coverage repeated task {task_id!r} after normalization")
            coverage[task_id] = value if isinstance(value, Stage11Coverage) else Stage11Coverage.model_validate(value)
        if set(coverage) != set(task_ids):
            missing = [task_id for task_id in task_ids if task_id not in coverage]
            extra = sorted(set(coverage) - set(task_ids))
            raise ValueError(
                f"Stage 11 coverage must cover representative inputs exactly (missing={missing}, extra={extra})"
            )
    quality_by_id = _records_by_task(
        quality_records,
        expected_task_ids=task_ids,
        label="Stage 10 quality",
    )
    raw_semantic_records = semantic_result.get("records")
    if isinstance(raw_semantic_records, str) or not isinstance(raw_semantic_records, Sequence):
        raise ValueError("semantic dedup result requires records")
    semantic_by_id = _records_by_task(
        raw_semantic_records,
        expected_task_ids=task_ids,
        label="semantic dedup",
    )
    coverage_counts = Counter(
        (
            item.language,
            item.turn_policy,
            item.edge_signatures,
        )
        for item in coverage.values()
    )
    capabilities = {task_id: capability_signature(task_by_id[task_id]) for task_id in task_ids}
    partitions: dict[tuple[str, Stage11Coverage, str], list[str]] = {}
    for task_id in task_ids:
        curator_cluster = semantic_by_id[task_id].get("cluster_id")
        if not isinstance(curator_cluster, str) or not curator_cluster.strip():
            raise ValueError(f"semantic dedup task {task_id!r} requires cluster_id")
        key = (
            curator_cluster.strip(),
            coverage[task_id],
            capabilities[task_id],
        )
        partitions.setdefault(key, []).append(task_id)

    seed = int(config.random_seed or 0)
    hard_limited = hard_limit_violations(config, tasks, surfaces)
    chosen_by_task: dict[str, tuple[str, str | None, dict[str, Any]]] = {}
    for (curator_cluster, bucket, capability), members in sorted(
        partitions.items(),
        key=lambda item: (
            item[0][0],
            item[0][1].language,
            item[0][1].turn_policy,
            item[0][1].edge_signatures,
            item[0][2],
        ),
    ):
        ranked: list[tuple[tuple[Any, ...], str, dict[str, Any]]] = []
        frequency = coverage_counts[(bucket.language, bucket.turn_policy, bucket.edge_signatures)]
        for task_id in members:
            surface = surfaces.get(task_id)
            if surface is None:
                raise ValueError(f"task {task_id!r} reached Stage 11 without a surface")
            rank, details = _representative_rank(
                task_id=task_id,
                task=task_by_id[task_id],
                surface=surface,
                quality=quality_by_id[task_id],
                coverage_frequency=frequency,
                source_preference=settings.representative_source_preference,
                seed=seed,
                hard_limited=task_id in hard_limited,
            )
            ranked.append((rank, task_id, details))
        ranked.sort()
        representative_id = ranked[0][1]
        cluster_id = (
            None
            if len(members) == 1
            else "dedup-"
            + _sha256(
                canonical_json(
                    {
                        "curator_cluster_id": curator_cluster,
                        "coverage": bucket.model_dump(mode="json"),
                        "capability_signature": capability,
                        "members": sorted(members),
                    }
                )
            )[:20]
        )
        details_by_id = {task_id: details for _, task_id, details in ranked}
        for task_id in members:
            chosen_by_task[task_id] = (
                representative_id,
                cluster_id,
                details_by_id[task_id],
            )

    decisions: list[DedupBalancingDecision] = []
    metadata: list[dict[str, Any]] = []
    selected_rank = 0
    for input_index, task_id in enumerate(task_ids):
        representative_id, cluster_id, rank_details = chosen_by_task[task_id]
        is_duplicate = representative_id != task_id
        selected = not is_duplicate or not settings.remove_duplicates
        decision = DedupBalancingDecision(
            task_id=task_id,
            selected=selected,
            is_duplicate=is_duplicate,
            duplicate_cluster_id=cluster_id,
            representative_task_id=(None if cluster_id is None else representative_id),
            drop_reason=("semantic_duplicate" if is_duplicate and settings.remove_duplicates else None),
            selection_rank=selected_rank if selected else input_index,
        )
        if selected:
            selected_rank += 1
        decisions.append(decision)
        semantic = semantic_by_id[task_id]
        metadata.append(
            {
                "task_id": task_id,
                "curator_cluster_id": semantic["cluster_id"],
                "curator_is_duplicate": bool(semantic.get("is_duplicate", False)),
                "duplicate_cluster_id": cluster_id,
                "representative_task_id": (None if cluster_id is None else representative_id),
                "capability_signature": capabilities[task_id],
                "coverage": coverage[task_id].model_dump(mode="json"),
                "representative_rank": rank_details,
                "text_hash": semantic.get("text_hash"),
                "curator_predecessor_id": semantic.get("curator_predecessor_id"),
                "curator_similarity_score": semantic.get("curator_similarity_score"),
            }
        )
    validated = validate_complete_decision_set(
        decisions,
        input_task_ids=task_ids,
        coverage_by_task_id=coverage,
        remove_duplicates=settings.remove_duplicates,
    )
    return validated, metadata


def _user_turn_count(task_id: str, surface: Mapping[str, Any]) -> int:
    steps = surface.get("steps")
    if isinstance(steps, str) or not isinstance(steps, Sequence):
        raise ValueError(f"task {task_id!r} surface steps must be a list")
    num_turns = sum(1 for step in steps if isinstance(step, Mapping) and step.get("kind") == "user")
    if num_turns < 1:
        raise ValueError(f"task {task_id!r} requires at least one user turn")
    return num_turns


def _tool_call_count(task_id: str, task: Mapping[str, Any]) -> int:
    raw_calls = task.get("num_tool_calls")
    if not isinstance(raw_calls, int) or isinstance(raw_calls, bool) or raw_calls < 0:
        raise ValueError(f"task {task_id!r} num_tool_calls must be a non-negative integer")
    return raw_calls


def hard_limit_violations(
    config: BfclConfig,
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Report which survivors the run's declared publication limits exclude.

    Representative selection reads this too: a limit is a property of the row,
    not of balancing, so electing a representative that publication can never
    keep would silently discard the cluster members that could have survived.
    """
    generation = config.task_generation or {}
    max_turns = generation.get("max_turns")
    max_tool_calls = generation.get("max_tool_calls")
    if max_turns is None and max_tool_calls is None:
        return {}
    violations: dict[str, str] = {}
    for task in tasks:
        task_id = _task_id(task)
        surface = surfaces.get(task_id)
        if surface is None:
            raise ValueError(f"task {task_id!r} reached Stage 11 without a surface")
        if max_turns is not None and _user_turn_count(task_id, surface) > int(max_turns):
            violations[task_id] = "max_turns_exceeded"
        elif max_tool_calls is not None and _tool_call_count(task_id, task) > int(max_tool_calls):
            violations[task_id] = "max_tool_calls_exceeded"
    return violations


def balancing_features(
    task: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one survivor onto the eight locked balancing dimensions."""
    task_id = _task_id(task)

    def required_text(key: str) -> str:
        value = task.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"task {task_id!r} requires a non-empty {key}")
        return value.strip()

    num_turns = _user_turn_count(task_id, surface)
    raw_calls = _tool_call_count(task_id, task)
    call_bucket = "3+" if raw_calls >= 3 else str(raw_calls)
    projection = project_dedup_text(task, surface)
    return {
        "intent": required_text("intent"),
        "category": required_text("category"),
        "required_tools": canonical_json(_string_list(task, "required_tools")),
        "tools_present": canonical_json(_string_list(task, "tools_present")),
        "difficulty": required_text("difficulty"),
        "turn_class": "multi_turn" if num_turns > 1 else "single_turn",
        "tool_call_count": call_bucket,
        "turn_policy": required_text("turn_policy"),
        "surface_text_hash": projection["text_hash"],
        "execution_case_hash": projection["execution_case_hash"],
        "num_turns": num_turns,
        "num_tool_calls": raw_calls,
    }


def largest_remainder_quotas(
    total: int,
    mix: Mapping[str, float],
) -> dict[str, int]:
    """Allocate an integer total with deterministic largest remainders."""
    if total < 0:
        raise ValueError("a balancing quota total cannot be negative")
    if not mix:
        return {}
    values = {str(bucket): float(weight) for bucket, weight in mix.items()}
    if any(not math.isfinite(weight) or weight < 0 for weight in values.values()):
        raise ValueError("balancing mix weights must be finite and non-negative")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("balancing mix weights must sum to 1")
    raw = {bucket: total * weight for bucket, weight in values.items()}
    quotas = {bucket: math.floor(value) for bucket, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        values,
        key=lambda bucket: (-(raw[bucket] - quotas[bucket]), bucket),
    )
    for bucket in order[:remaining]:
        quotas[bucket] += 1
    return quotas


def _target_mix_by_dimension(config: BfclConfig) -> dict[str, dict[str, float]]:
    generation = config.task_generation or {}
    return {
        dimension: {str(bucket): float(weight) for bucket, weight in (generation.get(config_key) or {}).items()}
        for dimension, config_key in (
            ("difficulty", "difficulty_mix"),
            ("turn_class", "turn_mix"),
            ("tool_call_count", "tool_call_count_mix"),
            ("turn_policy", "policy_mix"),
        )
        if generation.get(config_key)
    }


def _dimension_counts(
    task_ids: Sequence[str],
    features_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for dimension in BALANCING_DIMENSIONS:
        bucket_counts = Counter(str(features_by_id[task_id][dimension]) for task_id in task_ids)
        counts[dimension] = dict(sorted(bucket_counts.items()))
    return counts


def _publication_shortfall_reason(
    *,
    target: int,
    selected: int,
    bounds: Mapping[str, int],
    mix_exceeds_inventory: bool,
) -> str:
    """Name the bound that kept the selection away from the declared target."""
    if selected > target:
        return "coverage_requires_more_than_declared_target"
    binding = sorted(
        (name for name, bound in bounds.items() if bound < target),
        key=lambda name: (bounds[name], name),
    )
    if binding:
        return binding[0]
    if mix_exceeds_inventory:
        return "declared_mix_exceeds_inventory"
    return "balancing_constraint"


def _solve_balanced_selection(
    *,
    candidates: Sequence[str],
    candidates_by_bucket: Mapping[tuple[str, str, tuple[str, ...]], Sequence[str]],
    features: Mapping[str, Mapping[str, Any]],
    by_decision: Mapping[str, DedupBalancingDecision],
    budget: int,
    category_cap: int,
    target_counts: Mapping[str, Mapping[str, int]],
    conditional_target_mixes: Mapping[str, Mapping[str, float]],
    group_caps: Mapping[str, int],
    min_unique_surface_count: int,
    stable_key: Callable[[str], tuple[Any, ...]],
) -> set[str]:
    """Globally minimize declared mix and category deviations."""
    ordered = sorted(candidates, key=stable_key)

    def groups(feature: str) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for task_id in ordered:
            grouped.setdefault(str(features[task_id][feature]), []).append(task_id)
        return grouped

    def cost(selected: Sequence[str]) -> tuple[int, int]:
        """Rank a selection the same way the solver does: overflow, then mix."""
        overflow = sum(
            max(
                0,
                sum(features[task_id]["category"] == category for task_id in selected) - category_cap,
            )
            for category in {features[task_id]["category"] for task_id in ordered}
        )
        deviation = 0
        for dimension, quotas in target_counts.items():
            counts = Counter(features[task_id][dimension] for task_id in selected)
            for bucket in set(quotas) | set(counts):
                deviation += abs(counts[bucket] - quotas.get(bucket, 0))
        for dimension, mix in conditional_target_mixes.items():
            counts = Counter(
                str(features[task_id][dimension]) for task_id in selected if str(features[task_id][dimension]) in mix
            )
            quotas = largest_remainder_quotas(sum(counts.values()), mix)
            deviation += sum(abs(counts[bucket] - target) for bucket, target in quotas.items())
        return overflow, deviation

    try:
        import pulp  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        # Unit/minimal installations do not include the BYOB optimization extra.
        # Keep small deterministic use cases functional, but never fall back to an
        # approximate answer that could make an abort policy reject a feasible mix.
        from itertools import combinations

        if len(ordered) > 24:
            raise RuntimeError(
                "exact Stage 11 balancing requires the BYOB dependency 'pulp' when more than 24 candidates survive"
            ) from None
        best: tuple[tuple[int, int, int], tuple[str, ...]] | None = None
        rank = {task_id: index + 1 for index, task_id in enumerate(ordered)}
        for chosen_tuple in combinations(ordered, budget):
            chosen = set(chosen_tuple)
            if any(not chosen.intersection(members) for members in candidates_by_bucket.values()):
                continue
            if any(
                count > cap
                for feature, cap in group_caps.items()
                for count in Counter(str(features[task_id][feature]) for task_id in chosen_tuple).values()
            ):
                continue
            surface_counts = Counter(str(features[task_id]["surface_text_hash"]) for task_id in chosen_tuple)
            if len(surface_counts) < min_unique_surface_count:
                continue
            if any(
                decision.representative_task_id is not None
                and decision.representative_task_id != task_id
                and decision.representative_task_id in ordered
                and decision.representative_task_id not in chosen
                for task_id in chosen
                for decision in (by_decision[task_id],)
            ):
                continue
            score = (
                *cost(chosen_tuple),
                sum(rank[task_id] for task_id in chosen_tuple),
            )
            candidate = (score, chosen_tuple)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("Stage 11 publication constraints have no feasible selection")
        return set(best[1])

    variables = {
        task_id: pulp.LpVariable(f"selected_{index}", cat=pulp.LpBinary) for index, task_id in enumerate(ordered)
    }
    problem = pulp.LpProblem("bfcl_stage11_balancing", pulp.LpMinimize)
    problem += pulp.lpSum(variables.values()) == budget
    for members in candidates_by_bucket.values():
        problem += pulp.lpSum(variables[task_id] for task_id in members) >= 1
    for task_id in ordered:
        representative_id = by_decision[task_id].representative_task_id
        if representative_id is not None and representative_id != task_id and representative_id in variables:
            problem += variables[task_id] <= variables[representative_id]

    for feature, cap in sorted(group_caps.items()):
        for _, members in sorted(groups(feature).items()):
            # A group that cannot reach the cap needs no constraint, which keeps the
            # model small when a feature is nearly unique across candidates.
            if len(members) > cap:
                problem += pulp.lpSum(variables[task_id] for task_id in members) <= cap

    if min_unique_surface_count:
        surface_used: list[Any] = []
        for index, (_, members) in enumerate(sorted(groups("surface_text_hash").items())):
            count = pulp.lpSum(variables[task_id] for task_id in members)
            used = pulp.LpVariable(f"surface_used_{index}", cat=pulp.LpBinary)
            surface_used.append(used)
            problem += count >= used
            problem += count <= len(members) * used
        problem += pulp.lpSum(surface_used) >= min_unique_surface_count

    overflow_terms: list[Any] = []
    categories = sorted({str(features[task_id]["category"]) for task_id in ordered})
    for category in categories:
        count = pulp.lpSum(variables[task_id] for task_id in ordered if features[task_id]["category"] == category)
        overflow = pulp.LpVariable(
            f"category_overflow_{len(overflow_terms)}",
            lowBound=0,
            cat=pulp.LpInteger,
        )
        problem += count - category_cap <= overflow
        overflow_terms.append(overflow)

    deviation_terms: list[Any] = []
    for dimension, quotas in sorted(target_counts.items()):
        buckets = sorted(set(quotas) | {str(features[task_id][dimension]) for task_id in ordered})
        for bucket in buckets:
            count = pulp.lpSum(variables[task_id] for task_id in ordered if features[task_id][dimension] == bucket)
            under = pulp.LpVariable(
                f"under_{len(deviation_terms)}",
                lowBound=0,
                cat=pulp.LpInteger,
            )
            over = pulp.LpVariable(
                f"over_{len(deviation_terms)}",
                lowBound=0,
                cat=pulp.LpInteger,
            )
            problem += count + under - over == quotas.get(bucket, 0)
            deviation_terms.extend((under, over))
    for dimension, mix in sorted(conditional_target_mixes.items()):
        applicable_total = pulp.lpSum(
            variables[task_id] for task_id in ordered if str(features[task_id][dimension]) in mix
        )
        for bucket, weight in sorted(mix.items()):
            count = pulp.lpSum(
                variables[task_id] for task_id in ordered if str(features[task_id][dimension]) == bucket
            )
            under = pulp.LpVariable(
                f"conditional_under_{len(deviation_terms)}",
                lowBound=0,
            )
            over = pulp.LpVariable(
                f"conditional_over_{len(deviation_terms)}",
                lowBound=0,
            )
            problem += count + under - over == float(weight) * applicable_total
            deviation_terms.extend((under, over))

    # The three objectives are ranked, not weighted. The category budget is a
    # publication invariant that only coverage may break, declared mixes are targets
    # under it, and the stable order merely breaks ties. Folding that into one weighted
    # objective needs a tie-break weight of O(n^2) — millions of units at publication
    # scale — and the solver's tolerances would then decide whether a small mix
    # deviation really outranks a category overflow. Solving in order and pinning each
    # result keeps the ranking exact.
    total_overflow = pulp.lpSum(overflow_terms)
    total_deviation = pulp.lpSum(deviation_terms)
    tie_break_cost = pulp.lpSum((index + 1) * variables[task_id] for index, task_id in enumerate(ordered))

    def solve(objective: Any, phase: str) -> None:
        problem.setObjective(objective)
        status = problem.solve(pulp.PULP_CBC_CMD(msg=False, threads=1))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(
                "Stage 11 could not solve the publication balancing constraints "
                f"(phase={phase}, solver_status={pulp.LpStatus[status]!r})"
            )

    for objective, phase in (
        (total_overflow, "category_budget"),
        (total_deviation, "declared_mix"),
        (tie_break_cost, "stable_order"),
    ):
        solve(objective, phase)
        if phase != "stable_order":
            problem += objective <= float(pulp.value(objective) or 0.0) + 1e-7
    return {task_id for task_id in ordered if float(pulp.value(variables[task_id]) or 0.0) >= 0.5}


def balance_publication_set(
    config: BfclConfig,
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    representative_decisions: Sequence[DedupBalancingDecision | Mapping[str, object]],
    *,
    coverage_by_task_id: Mapping[str, Stage11Coverage | Mapping[str, object]] | None = None,
    edge_signatures_by_task_id: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[DedupBalancingDecision], list[dict[str, Any]], dict[str, Any]]:
    """Balance representative candidates across all eight locked dimensions."""
    settings = resolve_dedup_settings(config)
    task_ids = [_task_id(task) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Stage 11 balancing input task_id values must be unique")
    task_by_id = {task_id: task for task_id, task in zip(task_ids, tasks, strict=True)}
    if coverage_by_task_id is None:
        coverage = derive_stage11_coverage(
            tasks,
            surfaces,
            edge_signatures_by_task_id=edge_signatures_by_task_id,
        )
    else:
        if edge_signatures_by_task_id is not None:
            raise ValueError("provide coverage_by_task_id or edge_signatures_by_task_id, not both")
        coverage = {
            str(task_id).strip(): (
                value if isinstance(value, Stage11Coverage) else Stage11Coverage.model_validate(value)
            )
            for task_id, value in coverage_by_task_id.items()
        }
    representatives = validate_complete_decision_set(
        representative_decisions,
        input_task_ids=task_ids,
        coverage_by_task_id=coverage,
        remove_duplicates=settings.remove_duplicates,
    )
    by_decision = {decision.task_id: decision for decision in representatives}
    features: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        surface = surfaces.get(task_id)
        if surface is None:
            raise ValueError(f"task {task_id!r} reached Stage 11 balancing without a surface")
        features[task_id] = balancing_features(task_by_id[task_id], surface)

    generation = config.task_generation or {}
    hard_reason = hard_limit_violations(config, tasks, surfaces)
    initial_candidates = [
        task_id for task_id in task_ids if by_decision[task_id].selected and task_id not in hard_reason
    ]
    initial_candidate_set = set(initial_candidates)
    candidates = [
        task_id
        for task_id in initial_candidates
        if by_decision[task_id].representative_task_id is None
        or by_decision[task_id].representative_task_id in initial_candidate_set
    ]
    bucket_key = {
        task_id: (
            coverage[task_id].language,
            coverage[task_id].turn_policy,
            coverage[task_id].edge_signatures,
        )
        for task_id in task_ids
    }
    required_buckets = sorted(set(bucket_key.values()))
    candidates_by_bucket: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
    for task_id in candidates:
        candidates_by_bucket.setdefault(bucket_key[task_id], []).append(task_id)
    missing_buckets = [bucket for bucket in required_buckets if bucket not in candidates_by_bucket]
    if missing_buckets:
        raise ValueError(f"Stage 11 hard limits remove the final survivor of coverage bucket {missing_buckets[0]!r}")

    seed = int(config.random_seed or 0)

    def stable_key(task_id: str) -> tuple[Any, ...]:
        decision = by_decision[task_id]
        return (
            int(decision.is_duplicate),
            decision.selection_rank,
            _sha256(canonical_json({"seed": seed, "task_id": task_id})),
            task_id,
        )

    mandatory = {min(members, key=stable_key) for members in candidates_by_bucket.values()}
    category_cap = int(generation.get("tasks_per_category") or len(candidates))
    inventory_by_category = Counter(features[task_id]["category"] for task_id in candidates)
    category_budget = sum(min(count, category_cap) for count in inventory_by_category.values())
    group_caps = settings.group_caps
    group_inventory = {
        feature: Counter(str(features[task_id][feature]) for task_id in candidates)
        for feature, _, _ in GROUP_REUSE_CAPS
    }
    group_budgets = {
        feature: sum(min(count, cap) for count in group_inventory[feature].values())
        for feature, cap in group_caps.items()
    }
    if settings.min_exact_surface_ratio is not None:
        group_budgets["surface_text_hash"] = min(
            group_budgets.get("surface_text_hash", len(candidates)),
            math.floor(len(group_inventory["surface_text_hash"]) / settings.min_exact_surface_ratio),
        )
    target_mixes = _target_mix_by_dimension(config)
    conditional_target_mixes = {
        dimension: mix for dimension, mix in target_mixes.items() if dimension == "tool_call_count"
    }
    fixed_target_mixes = {
        dimension: mix for dimension, mix in target_mixes.items() if dimension not in conditional_target_mixes
    }

    # The inventory does not change while the budget shrinks, so counting it once keeps
    # the search linear even when the declared mix forces many steps down.
    mix_inventory = {
        dimension: Counter(features[task_id][dimension] for task_id in candidates) for dimension in target_mixes
    }

    def quotas_fit(total: int) -> bool:
        for dimension, mix in fixed_target_mixes.items():
            inventory = mix_inventory[dimension]
            quotas = largest_remainder_quotas(total, mix)
            if any(quotas[bucket] > inventory.get(bucket, 0) for bucket in quotas):
                return False
        return True

    feasible_budget = min([len(candidates), category_budget, *group_budgets.values()])
    declared_target = generation.get("target_published_tasks")
    budget = min(feasible_budget, int(declared_target)) if declared_target is not None else feasible_budget
    # A publication shortfall costs a whole run, so record which bound produced it
    # instead of reporting one fixed cause for every reason a target can be missed.
    shortfall_bounds = {
        "insufficient_candidate_inventory": len(candidates),
        "category_cap_limits_inventory": category_budget,
        **{reason: group_budgets[feature] for feature, _, reason in GROUP_REUSE_CAPS if feature in group_budgets},
    }
    mix_exceeds_inventory = False
    while budget > len(mandatory) and not quotas_fit(budget):
        budget -= 1
        mix_exceeds_inventory = True
    budget = max(budget, len(mandatory))
    target_counts = {dimension: largest_remainder_quotas(budget, mix) for dimension, mix in fixed_target_mixes.items()}
    min_unique_surface_count = (
        math.ceil(budget * settings.min_exact_surface_ratio) if settings.min_exact_surface_ratio is not None else 0
    )

    # This is a multi-dimensional cardinality problem. A local greedy/swap
    # optimizer can stop at a strict local optimum even when an exact selection
    # exists, and its nested repair loop is cubic.
    selected = _solve_balanced_selection(
        candidates=candidates,
        candidates_by_bucket=candidates_by_bucket,
        features=features,
        by_decision=by_decision,
        budget=budget,
        category_cap=category_cap,
        target_counts=target_counts,
        conditional_target_mixes=conditional_target_mixes,
        group_caps=group_caps,
        min_unique_surface_count=min_unique_surface_count,
        stable_key=stable_key,
    )
    selected_order = sorted(selected, key=stable_key)
    selected_counts = {
        dimension: Counter(features[task_id][dimension] for task_id in selected) for dimension in BALANCING_DIMENSIONS
    }
    target_counts.update(
        {
            dimension: largest_remainder_quotas(
                sum(selected_counts[dimension].get(bucket, 0) for bucket in mix),
                mix,
            )
            for dimension, mix in conditional_target_mixes.items()
        }
    )
    selected_group_counts = {
        feature: Counter(str(features[task_id][feature]) for task_id in selected_order)
        for feature, _, _ in GROUP_REUSE_CAPS
    }
    category_counts = selected_counts["category"]
    selected_by_bucket = Counter(bucket_key[task_id] for task_id in selected)

    coverage_locked_ids = {task_id for task_id in selected if selected_by_bucket[bucket_key[task_id]] == 1}
    rank_by_id = {task_id: rank for rank, task_id in enumerate(selected_order)}
    actual_counts = _dimension_counts(selected_order, features)
    decisions: list[DedupBalancingDecision] = []
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        previous = by_decision[task_id]
        if not previous.selected:
            decision = previous
            primary_dimension = None
        elif task_id in hard_reason:
            decision = DedupBalancingDecision(
                task_id=task_id,
                selected=False,
                is_duplicate=previous.is_duplicate,
                duplicate_cluster_id=previous.duplicate_cluster_id,
                representative_task_id=previous.representative_task_id,
                drop_reason=hard_reason[task_id],
                selection_rank=previous.selection_rank,
            )
            primary_dimension = None
        elif task_id in selected:
            decision = DedupBalancingDecision(
                task_id=task_id,
                selected=True,
                is_duplicate=previous.is_duplicate,
                duplicate_cluster_id=previous.duplicate_cluster_id,
                representative_task_id=previous.representative_task_id,
                selection_rank=rank_by_id[task_id],
            )
            primary_dimension = None
        else:
            category = features[task_id]["category"]
            if category_counts[category] >= category_cap:
                primary_dimension = "category"
            elif any(
                selected_group_counts[feature][str(features[task_id][feature])] >= cap
                # Whatever displaced this row shares its intent: directly for the intent
                # cap, and by construction for the executable-case cap, because intent is
                # part of the case identity. The intent dimension names that bound exactly.
                for feature, cap in group_caps.items()
                if feature in ("intent", "execution_case_hash")
            ):
                primary_dimension = "intent"
            else:
                over_target = [
                    dimension
                    for dimension, quotas in target_counts.items()
                    if selected_counts[dimension][features[task_id][dimension]]
                    >= quotas.get(features[task_id][dimension], 0)
                ]
                primary_dimension = over_target[0] if over_target else "intent"
            decision = DedupBalancingDecision(
                task_id=task_id,
                selected=False,
                is_duplicate=previous.is_duplicate,
                duplicate_cluster_id=previous.duplicate_cluster_id,
                representative_task_id=previous.representative_task_id,
                drop_reason="balance_quota",
                balance_dimension=primary_dimension,
                selection_rank=previous.selection_rank,
            )
        decisions.append(decision)
        records.append(
            {
                "task_id": task_id,
                "selected": decision.selected,
                "selection_rank": decision.selection_rank,
                "drop_reason": decision.drop_reason,
                "balance_dimension": primary_dimension,
                "dimensions": {dimension: features[task_id][dimension] for dimension in BALANCING_DIMENSIONS},
                "surface_text_hash": features[task_id]["surface_text_hash"],
                "num_turns": features[task_id]["num_turns"],
                "num_tool_calls": features[task_id]["num_tool_calls"],
                "coverage_locked": task_id in coverage_locked_ids,
            }
        )

    validated = validate_complete_decision_set(
        decisions,
        input_task_ids=task_ids,
        coverage_by_task_id=coverage,
        remove_duplicates=settings.remove_duplicates,
    )
    inventory_counts = _dimension_counts(candidates, features)
    unmet: list[dict[str, Any]] = []
    if declared_target is not None and len(selected_order) != int(declared_target):
        unmet.append(
            {
                "dimension": "publication_count",
                "bucket": "all",
                "target": int(declared_target),
                "actual": len(selected_order),
                "inventory": len(candidates),
                "reason": _publication_shortfall_reason(
                    target=int(declared_target),
                    selected=len(selected_order),
                    bounds=shortfall_bounds,
                    mix_exceeds_inventory=mix_exceeds_inventory,
                ),
            }
        )
    for category, actual in sorted(actual_counts["category"].items()):
        if actual > category_cap:
            unmet.append(
                {
                    "dimension": "category",
                    "bucket": category,
                    "target": category_cap,
                    "actual": actual,
                    "inventory": inventory_counts["category"].get(category, 0),
                    "reason": "coverage_constraint",
                }
            )
    for dimension, quotas in target_counts.items():
        conditional_mix = conditional_target_mixes.get(dimension)
        conditional_total = (
            sum(actual_counts[dimension].get(bucket, 0) for bucket in conditional_mix)
            if conditional_mix is not None
            else 0
        )
        for bucket, target in sorted(quotas.items()):
            actual = actual_counts[dimension].get(bucket, 0)
            if (
                conditional_mix is not None
                and conditional_total > 0
                and abs(actual / conditional_total - float(conditional_mix[bucket])) <= 0.05
            ):
                continue
            if actual != target:
                inventory = inventory_counts[dimension].get(bucket, 0)
                unmet.append(
                    {
                        "dimension": dimension,
                        "bucket": bucket,
                        "target": target,
                        "actual": actual,
                        "inventory": inventory,
                        "reason": (
                            "insufficient_inventory"
                            if inventory < target
                            else "coverage_or_cross_dimension_constraint"
                        ),
                    }
                )
    selected_surface_counts = selected_group_counts["surface_text_hash"]
    summary = {
        "input_count": len(task_ids),
        "candidate_count": len(candidates),
        "selected_count": len(selected_order),
        "hard_limit_drops": dict(sorted(Counter(hard_reason.values()).items())),
        "category_cap": category_cap,
        "target_counts": target_counts,
        "inventory_counts": inventory_counts,
        "actual_counts": actual_counts,
        "exact_surface_diversity": {
            "unique": len(selected_surface_counts),
            "unique_ratio": (len(selected_surface_counts) / len(selected_order) if selected_order else 1.0),
            "max_reuse": max(selected_surface_counts.values(), default=0),
            "max_exact_surface_reuse": settings.max_exact_surface_reuse,
            "min_exact_surface_ratio": settings.min_exact_surface_ratio,
        },
        # Repetition is reported for every capped feature whether or not the pack
        # declared a ceiling, so a run can be judged on how often it repeats a request
        # or an executable case before a cap is chosen.
        "group_diversity": {
            feature: {
                "unique": len(counts),
                "max_reuse": max(counts.values(), default=0),
                "cap": group_caps.get(feature),
            }
            for feature, counts in sorted(selected_group_counts.items())
        },
        "unmet_targets": unmet,
    }
    return validated, records, summary


def _content_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_dedup_balancing_artifacts(
    config: BfclConfig,
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[DedupBalancingDecision | Mapping[str, object]],
    representative_metadata: Sequence[Mapping[str, Any]],
    balancing_records: Sequence[Mapping[str, Any]],
    semantic_result: Mapping[str, Any],
    balancing_summary: Mapping[str, Any],
    *,
    coverage_by_task_id: Mapping[str, Stage11Coverage | Mapping[str, object]] | None = None,
    edge_signatures_by_task_id: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Write Stage 11's complete parquet and deterministic audit report."""
    settings = resolve_dedup_settings(config)
    semantic_settings_hash = semantic_result.get("settings_hash")
    if semantic_settings_hash != settings.settings_hash:
        raise ValueError("semantic dedup result settings_hash does not match the Stage 11 config")
    task_ids = [_task_id(task) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Stage 11 artifact input task_id values must be unique")
    task_by_id = {task_id: task for task_id, task in zip(task_ids, tasks, strict=True)}
    if coverage_by_task_id is None:
        coverage = derive_stage11_coverage(
            tasks,
            surfaces,
            edge_signatures_by_task_id=edge_signatures_by_task_id,
        )
    else:
        if edge_signatures_by_task_id is not None:
            raise ValueError("provide coverage_by_task_id or edge_signatures_by_task_id, not both")
        coverage = {
            str(task_id).strip(): (
                value if isinstance(value, Stage11Coverage) else Stage11Coverage.model_validate(value)
            )
            for task_id, value in coverage_by_task_id.items()
        }
    validated = validate_complete_decision_set(
        decisions,
        input_task_ids=task_ids,
        coverage_by_task_id=coverage,
        remove_duplicates=settings.remove_duplicates,
    )
    representative_by_id = _records_by_task(
        representative_metadata,
        expected_task_ids=task_ids,
        label="representative metadata",
    )
    semantic_records = semantic_result.get("records")
    if isinstance(semantic_records, (str, bytes)) or not isinstance(semantic_records, Sequence):
        raise ValueError("semantic dedup result must contain complete per-task records")
    semantic_by_id = _records_by_task(
        semantic_records,
        expected_task_ids=task_ids,
        label="semantic dedup result",
    )
    if int(semantic_result.get("input_count", -1)) != len(task_ids):
        raise ValueError("semantic dedup result input_count does not match the Stage 10 survivors")
    balancing_by_id = _records_by_task(
        balancing_records,
        expected_task_ids=task_ids,
        label="balancing metadata",
    )
    rows: list[dict[str, Any]] = []
    for decision in validated:
        task_id = decision.task_id
        representative = representative_by_id[task_id]
        balancing = balancing_by_id[task_id]
        if bool(balancing.get("selected")) != decision.selected:
            raise ValueError(f"task {task_id!r} balancing metadata disagrees with its decision")
        if balancing.get("drop_reason") != decision.drop_reason:
            raise ValueError(f"task {task_id!r} balancing drop_reason disagrees with its decision")
        if balancing.get("balance_dimension") != decision.balance_dimension:
            raise ValueError(f"task {task_id!r} balancing dimension disagrees with its decision")
        if int(balancing.get("selection_rank", -1)) != decision.selection_rank:
            raise ValueError(f"task {task_id!r} balancing rank disagrees with its decision")
        if representative.get("duplicate_cluster_id") != decision.duplicate_cluster_id:
            raise ValueError(f"task {task_id!r} representative cluster metadata drifted")
        if representative.get("representative_task_id") != decision.representative_task_id:
            raise ValueError(f"task {task_id!r} representative task metadata drifted")
        semantic = semantic_by_id[task_id]
        if representative.get("curator_cluster_id") != semantic.get("cluster_id"):
            raise ValueError(f"task {task_id!r} Curator cluster metadata drifted")
        if bool(representative.get("curator_is_duplicate", False)) != bool(semantic.get("is_duplicate", False)):
            raise ValueError(f"task {task_id!r} Curator duplicate metadata drifted")
        for key in ("text_hash", "curator_predecessor_id", "curator_similarity_score"):
            if representative.get(key) != semantic.get(key):
                raise ValueError(f"task {task_id!r} semantic {key} metadata drifted")
        dimensions = balancing.get("dimensions")
        if not isinstance(dimensions, Mapping) or set(dimensions) != set(BALANCING_DIMENSIONS):
            raise ValueError(f"task {task_id!r} balancing metadata must contain all eight dimensions")
        bucket = coverage[task_id]
        rows.append(
            {
                "task_id": task_id,
                "contract_version": decision.contract_version,
                "selected": decision.selected,
                "is_duplicate": decision.is_duplicate,
                "duplicate_cluster_id": decision.duplicate_cluster_id,
                "representative_task_id": decision.representative_task_id,
                "drop_reason": decision.drop_reason,
                "balance_dimension": decision.balance_dimension,
                "selection_rank": decision.selection_rank,
                "curator_cluster_id": representative.get("curator_cluster_id"),
                "curator_is_duplicate": bool(representative.get("curator_is_duplicate", False)),
                "curator_predecessor_id": representative.get("curator_predecessor_id"),
                "curator_similarity_score": representative.get("curator_similarity_score"),
                "text_hash": representative.get("text_hash"),
                "capability_signature": representative.get("capability_signature"),
                "language": bucket.language,
                "edge_signatures": list(bucket.edge_signatures),
                **{dimension: str(dimensions[dimension]) for dimension in BALANCING_DIMENSIONS},
                "num_turns": int(balancing["num_turns"]),
                "num_tool_calls": int(balancing["num_tool_calls"]),
                "coverage_locked": bool(balancing.get("coverage_locked", False)),
                "representative_rank": canonical_json(representative.get("representative_rank") or {}),
            }
        )

    cache = stage_cache_dir(config)
    artifact_path = write_stage_table(
        cache / BALANCED_TASKS,
        rows,
        balanced_tasks_schema(),
    )
    artifact_hash = _content_hash(artifact_path)
    by_decision = {decision.task_id: decision for decision in validated}

    def grouped(key: str, *, from_task: bool = False) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            task_id = row["task_id"]
            value = task_by_id[task_id].get(key) if from_task else row[key]
            bucket_name = str(value)
            group = groups.setdefault(
                bucket_name,
                {
                    "input": 0,
                    "selected": 0,
                    "dropped": 0,
                    "duplicates": 0,
                    "drop_reason_counts": {},
                },
            )
            group["input"] += 1
            group["selected" if row["selected"] else "dropped"] += 1
            group["duplicates"] += int(row["is_duplicate"])
            reason = row["drop_reason"]
            if reason is not None:
                counts = group["drop_reason_counts"]
                counts[reason] = counts.get(reason, 0) + 1
        for group in groups.values():
            group["drop_reason_counts"] = dict(sorted(group["drop_reason_counts"].items()))
        return dict(sorted(groups.items()))

    edge_inventory: Counter[str] = Counter()
    edge_selected: Counter[str] = Counter()
    for task_id in task_ids:
        for signature in coverage[task_id].edge_signatures:
            edge_inventory[signature] += 1
            if by_decision[task_id].selected:
                edge_selected[signature] += 1
    rare_edge_preservation = {
        signature: {
            "input": edge_inventory[signature],
            "selected": edge_selected[signature],
            "preserved": edge_selected[signature] > 0,
        }
        for signature in sorted(edge_inventory)
    }
    drop_reason_counts = dict(
        sorted(Counter(decision.drop_reason for decision in validated if decision.drop_reason is not None).items())
    )
    actual_counts: dict[str, dict[str, int]] = {
        dimension: dict(sorted(Counter(str(row[dimension]) for row in rows if row["selected"]).items()))
        for dimension in BALANCING_DIMENSIONS
    }
    summary_actual_counts = balancing_summary.get("actual_counts")
    if summary_actual_counts is not None and summary_actual_counts != actual_counts:
        raise ValueError("balancing summary actual_counts does not match the selected artifact rows")
    hard_limit_drops = dict(
        sorted(
            Counter(
                decision.drop_reason
                for decision in validated
                if decision.drop_reason in {"max_turns_exceeded", "max_tool_calls_exceeded"}
            ).items()
        )
    )
    unmet_targets = list(balancing_summary.get("unmet_targets") or [])
    report = {
        "schema_version": "1.0",
        "contract_version": DEDUP_BALANCING_CONTRACT_VERSION,
        "counts": {
            "stage_ten_survivors": len(task_ids),
            "curator_duplicates": sum(bool(row["curator_is_duplicate"]) for row in rows),
            "final_duplicates": sum(decision.is_duplicate for decision in validated),
            "semantic_duplicate_annotations": sum(decision.is_duplicate for decision in validated),
            "semantic_duplicate_drops": sum(decision.drop_reason == "semantic_duplicate" for decision in validated),
            "selected": sum(decision.selected for decision in validated),
            "dropped": sum(not decision.selected for decision in validated),
        },
        "drop_reason_counts": drop_reason_counts,
        "by_template": grouped("template_id", from_task=True),
        "by_category": grouped("category"),
        "by_turn_policy": grouped("turn_policy"),
        "by_difficulty": grouped("difficulty"),
        "inventory_counts": balancing_summary.get("inventory_counts") or {},
        "target_counts": balancing_summary.get("target_counts") or {},
        "actual_counts": actual_counts,
        "diversity": {
            "candidate_exact_surfaces": semantic_result.get("exact_surface_diversity"),
            "candidate_execution_cases": semantic_result.get("execution_case_diversity"),
            "published_exact_surfaces": balancing_summary.get("exact_surface_diversity"),
        },
        "unmet_targets": unmet_targets,
        "release_policy": {
            "unmet_target_policy": settings.unmet_target_policy,
            "unmet_target_action": (settings.unmet_target_policy if unmet_targets else "none"),
            "gold_eligible": not unmet_targets,
        },
        "hard_limit_drops": hard_limit_drops,
        "category_cap": balancing_summary.get("category_cap"),
        "rare_edge_preservation": rare_edge_preservation,
        "coverage_bucket_count": len(
            {
                (
                    bucket.language,
                    bucket.turn_policy,
                    bucket.edge_signatures,
                )
                for bucket in coverage.values()
            }
        ),
        "lineage": {
            "settings": settings.as_lineage(),
            "settings_hash": settings.settings_hash,
            "projected_input_hash": semantic_result.get("input_hash"),
            "embedding_signature": semantic_result.get("embedding_signature"),
            "effective_n_clusters": semantic_result.get("effective_n_clusters"),
        },
        "artifacts": {
            BALANCED_TASKS: {
                "content_hash": artifact_hash,
                "row_count": len(rows),
            }
        },
    }
    report_path = _write_json_atomic(cache / DEDUP_BALANCING_REPORT, report)
    return {
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "report_path": report_path,
        "report_hash": _content_hash(report_path),
        "report": report,
    }


def run_dedup_balancing_stage(
    config: BfclConfig,
    tasks: Sequence[Mapping[str, Any]],
    surfaces: Mapping[str, Mapping[str, Any]],
    quality_records: Sequence[Mapping[str, Any]],
    *,
    finder: DuplicateFinder | None = None,
) -> dict[str, Any]:
    """Run Stage 11 end to end and leave its auditable artifacts behind."""
    projected = project_dedup_texts(tasks, surfaces)
    semantic_result = run_semantic_dedup(config, projected, finder=finder)
    coverage = derive_stage11_coverage(tasks, surfaces)
    representative_decisions, representative_metadata = select_duplicate_representatives(
        config,
        tasks,
        surfaces,
        quality_records,
        semantic_result,
        coverage_by_task_id=coverage,
    )
    decisions, balancing_records, balancing_summary = balance_publication_set(
        config,
        tasks,
        surfaces,
        representative_decisions,
        coverage_by_task_id=coverage,
    )
    artifacts = write_dedup_balancing_artifacts(
        config,
        tasks,
        surfaces,
        decisions,
        representative_metadata,
        balancing_records,
        semantic_result,
        balancing_summary,
        coverage_by_task_id=coverage,
    )
    if balancing_summary.get("unmet_targets") and resolve_dedup_settings(config).unmet_target_policy == "abort":
        raise DedupBalancingPolicyError(
            "Stage 11 balancing targets are infeasible and unmet_target_policy='abort'; "
            f"inspect {artifacts['report_path']}"
        )
    return {
        "decisions": decisions,
        "projected": projected,
        "semantic_result": semantic_result,
        "representative_metadata": representative_metadata,
        "balancing_records": balancing_records,
        "balancing_summary": balancing_summary,
        "coverage_by_task_id": coverage,
        "artifacts": artifacts,
    }
