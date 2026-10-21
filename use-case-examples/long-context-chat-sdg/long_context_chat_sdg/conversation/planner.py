# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planner — decide each conversation's shape.

Two simple knobs drive everything (both from the engine config):
  - turn COUNT  = min_turns..max_turns, center-weighted so the average lands
    mid-range (~4-5 for a 3-6 range), with some shorter/longer for diversity.
  - turn DEPTH  = min_hops..max_steps (searches per turn), applied to every turn.

Turn KIND is for conversational diversity only (NOT depth): the opening kind and
each follow-up kind are sampled from weighted distributions, and a couple of kinds
may ask the user a clarifying question. Customers never tune per-kind numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

_KINDS = ["factual", "comparative", "multi_hop", "exploratory", "ambiguous"]
_CLARIFY_KINDS = {"exploratory", "ambiguous"}          # openings where the assistant may clarify first
_FOLLOWUP_CLARIFY = {"clarify"}                        # follow-ups where the assistant may clarify first
_NO_TOOL_KINDS = {"simplify", "acknowledge"}           # conversational follow-ups: answered from context, NO retrieval

# opening-kind distribution (sampled when the row carries no kind).
_DEFAULT_SEED_KINDS: List[Dict[str, Any]] = [
    {"kind": "factual", "weight": 3}, {"kind": "comparative", "weight": 2},
    {"kind": "multi_hop", "weight": 3}, {"kind": "exploratory", "weight": 2},
    {"kind": "ambiguous", "weight": 1},
]
# follow-up-kind distribution (labels resolve to directives in prompts.KIND_DIRECTIVES).
# `simplify`/`acknowledge` are conversational (no tool); the rest drive fresh research.
_DEFAULT_FOLLOWUPS: List[Dict[str, Any]] = [
    {"kind": "deepen", "weight": 3}, {"kind": "compare", "weight": 2},
    {"kind": "related", "weight": 2}, {"kind": "clarify", "weight": 1},
    {"kind": "simplify", "weight": 1}, {"kind": "acknowledge", "weight": 1},
]


@dataclass
class TurnSpec:
    kind: str
    min_hops: int
    max_steps: int
    clarify: bool
    no_tool: bool = False                              # conversational turn: answer from context, no retrieval


@dataclass
class ConversationPlan:
    turns: List[TurnSpec] = field(default_factory=list)

    def kinds(self) -> List[str]:
        return [t.kind for t in self.turns]

    @property
    def n_turns(self) -> int:
        return len(self.turns)


def _sample_turns(rng: random.Random, lo: int, hi: int) -> int:
    """Center-weighted turn count: the average of two uniform draws is triangular,
    so the count lands mid-range (e.g. ~4-5 for a 3-6 range) rather than flat.
    Round half up (not banker's rounding) so 4 and 5 stay balanced."""
    return int((rng.randint(lo, hi) + rng.randint(lo, hi)) / 2 + 0.5)


def plan_conversation(cfg_plan: Dict[str, Any], rng: random.Random, *, seed_kind: str = "",
                      min_turns: int = 3, max_turns: int = 6,
                      min_hops: int = 2, max_steps: int = 6) -> ConversationPlan:
    cfg_plan = cfg_plan or {}

    # ── turn count (center-weighted over the engine's turn range) ────────────
    lo, hi = int(min_turns), max(int(min_turns), int(max_turns))
    nt = cfg_plan.get("num_turns")                     # optional advanced override
    if nt:
        lo = max(lo, int(nt.get("min", lo)))
        hi = max(lo, min(hi, int(nt.get("max", hi))))
    weights = nt.get("weights") if nt else None
    if weights:
        opts = [(int(k), float(v)) for k, v in weights.items() if lo <= int(k) <= hi]
        n_turns = rng.choices([c for c, _ in opts], weights=[w for _, w in opts], k=1)[0] \
            if opts else _sample_turns(rng, lo, hi)
    else:
        n_turns = _sample_turns(rng, lo, hi)

    mh, ms = int(min_hops), max(int(min_hops), int(max_steps))   # global depth, every turn

    # ── opening turn: honor a provided kind, else sample one ─────────────────
    if not seed_kind or seed_kind not in _KINDS:
        seeds = cfg_plan.get("seed_kinds") or _DEFAULT_SEED_KINDS
        seed_kind = rng.choices([s["kind"] for s in seeds],
                                weights=[float(s.get("weight", 1)) for s in seeds], k=1)[0]
    turns = [TurnSpec(seed_kind, mh, ms, seed_kind in _CLARIFY_KINDS)]   # opening always grounds

    # ── follow-up turns (weighted kind variety) ──────────────────────────────
    fu = cfg_plan.get("follow_up_kinds") or _DEFAULT_FOLLOWUPS
    labels = [f["kind"] for f in fu]
    fweights = [float(f.get("weight", 1)) for f in fu]
    for _ in range(n_turns - 1):
        label = rng.choices(labels, weights=fweights, k=1)[0]
        no_tool = label in _NO_TOOL_KINDS
        turns.append(TurnSpec(label, 0 if no_tool else mh, ms,
                              label in _FOLLOWUP_CLARIFY, no_tool))
    return ConversationPlan(turns=turns)
