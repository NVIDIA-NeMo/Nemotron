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

"""ConversationSimulatorConfig — the single source of truth for every engine knob.

Extends Data Designer's ``SingleColumnConfig`` when DD is installed; falls back to
a plain pydantic model otherwise so the pieces stay importable and unit-testable
without the DD runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:  # real plugin path
    from data_designer.config.base import SingleColumnConfig as _Base
    _HAS_DD = True
except Exception:  # test / standalone path
    from pydantic import BaseModel as _Base  # type: ignore
    _HAS_DD = False

from pydantic import Field


class ConversationSimulatorConfig(_Base):
    """All knobs for the multi-turn retrieval-conversation simulation."""

    # ── column plumbing ───────────────────────────────────────────────────────
    column_type: Literal["long-context-chat-simulator"] = "long-context-chat-simulator"
    model_alias: str = "assistant_model"           # DD requires a primary alias
    # direct OpenAI-client bypass (DD's facade mis-parses tool calls on some
    # endpoints): {alias: {model, base_url, api_key_env, params}} injected by the pipeline.
    model_clients: Dict[str, Any] = Field(default_factory=dict)

    query_column: str = "query"                    # the seed query (from the query source)
    kind_column: str = "kind"                      # OPTIONAL per-row shape override; absent => planner samples
    cluster_id_column: str = "cluster_id"
    persona_column: str = "persona"
    tools_column: str = "tools"                    # user-defined OpenAI-format tool schemas

    # ── retrieval service (external, HTTP) ────────────────────────────────────
    retrieval_mode: Literal["http", "simulated"] = "http"
    retrieval_endpoint: str = ""                   # retrieval service POST url
    retrieval_tools: List[str] = Field(default_factory=lambda: ["search"])  # tool names routed to the service
    retrieval_field_map: Dict[str, Any] = Field(default_factory=dict)       # request/response schema mapping
    retrieval_timeout: int = 120                    # generous: service can be slow under load
    retrieval_max_retries: int = 2                   # bounded retries on transient endpoint blips (long runs)
    retrieval_headers: Dict[str, str] = Field(default_factory=dict)
    top_k: int = 4                                 # chunks handed to the assistant per hop
    oversample_factor: int = 2                     # request top_k*factor, keep a random top_k
    query_arg_names: List[str] = Field(default_factory=lambda: ["query", "q", "search_query"])
    top_k_arg_names: List[str] = Field(default_factory=lambda: ["top_k", "k", "n"])

    # ── depth & loop control ──────────────────────────────────────────────────
    min_hops: int = 2                              # floor on retrieval rounds (blocks lazy answers)
    max_steps: int = 6                             # hard ceiling on hops per turn
    max_turns: int = 3                             # ceiling (superseded by the plan)
    min_turns: int = 2
    max_correction_attempts: int = 3               # tool-call schema-error retries
    max_tool_calls_per_turn: int = 1               # cap tool calls/turn (many NIM endpoints reject >1)
    force_first_tool: bool = True                  # force tool_choice=required until min_hops
    allow_clarify: bool = True                     # let clarify-kind turns ask a question first

    # ── conversation plan (per-row shape from the query kind) ─────────────────
    conversation_plan: Dict[str, Any] = Field(default_factory=dict)

    # ── context compression ───────────────────────────────────────────────────
    context_window_k: int = 2                      # recent tool responses kept RAW
    compaction_mode: Literal["reference", "summary", "drop"] = "reference"
    compression_token_limit: int = 2000
    use_scratchpad: bool = True

    # ── quality gates ─────────────────────────────────────────────────────────
    gate_query: bool = True                        # judge/skip weak opening queries
    inline_judge: bool = True                      # judge trajectories during generation
    majority_vote_n: int = 1                       # assistant self-consistency samples
    max_tools: int = 12                            # tool subset offered per row
    persona_voice: bool = True                     # re-voice the seed query in the persona

    @property
    def side_effect_columns(self) -> List[str]:
        return [
            "conversation_messages", "conversation_messages_raw", "conversation_status",
            "conversation_metadata", "user_query", "conversation_plan", "hops_taken",
            "retrieval_log", "trajectory_judgment", "cluster_id", "compression",
            "retrieval_mode",
        ]

    @property
    def required_columns(self) -> List[str]:
        return [self.tools_column]
