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

"""Work item types passed through pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DatasetWorkItem:
    """
    Input to PlanStage - one per dataset in a blend.

    This work item carries all information needed for PlanStage to:
    - Discover input files
    - Create shard plan
    - Fan out to ShardWorkItems
    """

    dataset_name: str
    path: str
    weight: float
    split: str | None
    subset: str | None
    text_field: str

    # Run context (set by driver)
    run_hash: str
    run_dir: str
    config_hash: str
    num_shards: int
    dtype: str
    min_doc_chars: int | None
    max_doc_tokens: int | None
    max_rows: int | None
    sample: str | int | None
    sample_seed: int

    # Resolved tokenizer config (for plan creation)
    tokenizer_config: dict = field(default_factory=dict)


@dataclass
class ShardWorkItem:
    """Payload for shard processing."""

    dataset_name: str
    plan_hash: str
    shard_index: int
    assignment: dict[str, Any]
    output_dir: str
    receipts_dir: str
    text_field: str
    dtype: str
    min_doc_chars: int | None
    max_doc_tokens: int | None
    max_rows: int | None


@dataclass
class SftDatasetWorkItem:
    """Input to SFT plan stage - one per dataset."""

    # Dataset identity (matches DatasetWorkItem pattern)
    dataset_name: str
    path: str
    weight: float
    split: str | None
    subset: str | None

    # Run context (set by driver)
    run_hash: str
    run_dir: str
    config_hash: str

    # Planning/output partitioning
    num_shards: int
    dtype: str
    max_doc_tokens: int | None
    max_rows: int | None
    sample: str | int | None
    sample_seed: int

    # Resolved tokenizer config (for plan creation)
    tokenizer_config: dict = field(default_factory=dict)

    # ChatSFT parsing/tokenization options (consumed by stage 3)
    messages_field: str = "messages"
    tools_field: str = "tools"
    chat_template: str | None = None
    used_in_filter: str | None = None
    used_in_field: str = "used_in"

    # Packing options
    pack_size: int = 2048
    algorithm: str = "first_fit_shuffle"
    seed: int | None = None

    # Packed Parquet output options (per packed-sft-impl-parquet-nemotron.md)
    parquet_row_group_size: int = 1000
    parquet_compression: str = "zstd"


@dataclass
class SftShardWorkItem:
    """Payload for SFT shard processing (packed Parquet output)."""

    dataset_name: str
    plan_hash: str
    shard_index: int
    assignment: dict[str, Any]

    # Output locations
    output_dir: str
    receipts_dir: str
    spool_dir: str | None = None

    # Tokenization and filtering
    dtype: str = "int32"
    messages_field: str = "messages"
    tools_field: str = "tools"
    chat_template: str | None = None
    max_doc_tokens: int | None = None
    max_rows: int | None = None
    used_in_filter: str | None = None
    used_in_field: str = "used_in"

    # Packing
    pack_size: int = 2048
    algorithm: str = "first_fit_shuffle"
    seed: int | None = None

    # Packed Parquet output options
    parquet_row_group_size: int = 1000
    parquet_compression: str = "zstd"


@dataclass
class VideoExtractWorkItem:
    """Input to AudioExtractStage — one per video to extract audio from.

    Used by Omni-style video-language SFT pipelines (see
    nemotron.data_prep.recipes.sft_omni). The youtube_id field carries the
    bare key needed by downstream shard planning to join with QA records.

    Note: this work item type is consumed by ``make_wandb_stats_hook`` (called
    directly with hard-coded dataset_names) rather than ``pipeline_wandb_hook``
    (which expects per-dataset/per-shard items). It deliberately does NOT
    carry dataset_name / num_shards fields — those would be dead weight here.
    """

    video_path: str
    audio_path: str
    youtube_id: str

    # Run context (set by driver)
    run_hash: str
    run_dir: str


@dataclass
class VlmPreferencePrepWorkItem:
    """Input to VlmPreferencePrepStage — one per dataset prep run.

    Drives the single-worker, single-stage VLM-preference cache build
    used by Omni RL data prep. The stage extracts a media archive,
    copies the source records (parquet), validates rows against the
    extracted media, and writes the sentinel files NeMo-RL's response
    datasets consume.

    Note: the work item carries a coarse ``flavor`` discriminator rather
    than a full schema-adapter callback. The current implementation
    knows MMPR's parquet schema inline; if a second VLM preference
    dataset arrives with a different schema, factor the row-validation
    helpers out and replace ``flavor`` with a per-dataset adapter.
    "Locality over DRY" until then.

    Fields:
        flavor: ``"tiny"`` (MMPR-Tiny → vision RL) or ``"mpo"`` (full MMPR
            → MPO). Tiny runs vendored MMPR-Tiny logic; MPO defers to the
            upstream prep script via subprocess until that logic can be
            vendored cleanly.
        raw_dir: Local directory holding the raw HF download
            (``images.zip`` and ``mmpr_tiny.parquet`` for tiny).
        output_dir: Target cache directory. Receives the extracted images,
            the parquet copy, the preview JSONL, the summary, and the
            ``.mmpr_ready`` sentinel.
        meta_name: Output metadata filename (MPO flavor only; ignored for
            tiny).
        plan_hash: Stable hash of the prep configuration. Used by
            ``ReceiptManager`` for resumability + cache invalidation.
        receipts_dir: Receipt directory. MUST live outside ``output_dir``
            so non-force resume can wipe the published cache without
            dropping receipts.
        builder_command: Optional shell command for the MPO flavor when
            the vendored logic is not available (TODO: remove once
            ``prepare_public_mmpr_for_mpo.py`` is vendored). The command
            is formatted with ``{input_dir}``, ``{output_dir}``, and
            ``{meta_name}`` before invocation.
    """

    flavor: Literal["tiny", "mpo"]
    raw_dir: str
    output_dir: str
    meta_name: str
    plan_hash: str
    receipts_dir: str

    # Run context
    run_hash: str
    run_dir: str

    # MPO-only escape hatch until the upstream script can be vendored.
    builder_command: str | None = None


@dataclass
class WebDatasetShardWorkItem:
    """Input to WebDatasetShardStage — one per output Energon shard.

    Each shard is independent: it owns a tuple of QA records along with the
    resolved (video, audio) paths needed to assemble the tar. Shards are
    grouped on the driver so that all QA pairs for a given video land in the
    same shard — this lets the audio-extract pipeline run independently and
    avoids any per-video work in the shard-write stage.

    Fields:
        split: "train" | "val" | "test".
        shard_index: 0-based shard index within the split.
        plan_hash: Stable hash of the shard plan (used by ReceiptManager).
        output_dir: Directory where shard tar is written.
        receipts_dir: Directory for shard receipts (lives outside output_dir
            so cleanup of the published dataset doesn't drop the receipts).
        records: Tuple of (video_path, audio_path, conversation_json) triples.
    """

    split: str
    shard_index: int
    plan_hash: str
    output_dir: str
    receipts_dir: str
    records: tuple[tuple[str, str, str], ...]

    # Run context
    run_hash: str
    run_dir: str


@dataclass
class JsonlDatasetWorkItem:
    """
    Input to JsonlPlanStage - one per dataset/split in a JSONL pipeline.

    This work item carries all information needed for JsonlPlanStage to:
    - Discover input files
    - Create JSONL shard plan (without tokenizer resolution)
    - Fan out to JsonlShardWorkItems
    """

    dataset_name: str
    path: str
    weight: float
    split: str | None
    subset: str | None
    text_field: str

    # Run context (set by driver)
    run_hash: str
    run_dir: str
    config_hash: str

    num_shards: int
    compression: Literal["none", "zstd"] = "none"
    max_rows: int | None = None
    resolve_hf_placeholders: bool = False


@dataclass
class JsonlShardWorkItem:
    """Payload for JSONL shard processing."""

    dataset_name: str
    plan_hash: str
    shard_index: int
    assignment: dict[str, Any]
    output_dir: str
    receipts_dir: str
    text_field: str
    compression: str
    max_rows: int | None
    resolve_hf_placeholders: bool = False


__all__ = [
    "DatasetWorkItem",
    "ShardWorkItem",
    "SftDatasetWorkItem",
    "SftShardWorkItem",
    "VideoExtractWorkItem",
    "WebDatasetShardWorkItem",
    "JsonlDatasetWorkItem",
    "JsonlShardWorkItem",
    "VlmPreferencePrepWorkItem",
]
