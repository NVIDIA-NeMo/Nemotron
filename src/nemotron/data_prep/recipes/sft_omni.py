# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Omni SFT Pipeline Recipe — multimodal video-language SFT to Energon WebDataset.

This recipe composes the audio-extract and webdataset-shard stages into a
two-pipeline build of a video-QA Energon dataset:

    Pipeline 1: [VideoExtractWorkItem] → AudioExtractStage → ffmpeg per video
    Pipeline 2: [WebDatasetShardWorkItem] → WebDatasetShardStage → tar per shard

    + Driver-side finalize (restructure shards, sqlite index, dataset.yaml,
      receipt aggregation for sample counts)

Counterpart to ``recipes/sft.py`` (text/chat SFT to packed parquet). Uses the
same ``setup_*_run`` / ``finalize_*_run`` shape so a recipe-level reader sees
the same idiom across families.

Key Design Decisions:
    - Two pipelines, not one: per-video and per-shard fan-in shapes don't
      unify cleanly under PlanStage (shard planning needs whole-dataset
      visibility for video-boundary preservation).
    - Receipts live OUTSIDE dataset_path (caller-supplied ``runs_root``)
      so cleanup of the published Energon dir doesn't drop the receipts.
    - Driver-side `make_wandb_stats_hook` instead of `pipeline_wandb_hook`:
      our work items are per-video / per-shard, not per-dataset, and don't
      fit the pipeline_wandb_hook abstraction. Same observability surface.
    - Generic over QA shape: caller resolves their dataset's QA records to
      ``(video_id, question, answer)`` triples. Recipe-specific logic
      (URLs, filename layouts, MCQ-vs-free-form QA) lives in the cookbook
      script, not here.

Usage:
    from nemotron.data_prep.recipes.sft_omni import run_sft_omni_pipeline

    result = run_sft_omni_pipeline(
        videos_by_id={"abc": Path("/data/videos/abc_0_10.mp4"), ...},
        qa_records={"train": [("abc", "What is happening?", "A cat plays."), ...]},
        audio_dir=Path("/data/audio"),
        dataset_path=Path("/data/energon"),
        runs_root=Path("/data/runs"),
        samples_per_shard=100,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import tarfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.core.work_items import (
    VideoExtractWorkItem,
    WebDatasetShardWorkItem,
)
from nemotron.data_prep.observability import make_wandb_stats_hook
from nemotron.data_prep.recipes.execution_mode import ExecutionModeRequest, resolve_execution_mode
from nemotron.data_prep.stages import (
    AudioExtractStage,
    AudioExtractStageConfig,
    PipelineContext,
    WebDatasetShardStage,
    WebDatasetShardStageConfig,
)
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars

logger = logging.getLogger(__name__)


# Energon dataset.yaml describing how to decode each WebDataset sample.
#
# Schema contract: ``ChatMLWebdataset`` (in megatron.bridge.data.energon.
# task_encoder_utils) expects each sample's conversation.json to be the
# OpenAI-style structured ChatML emitted by ``_build_conversation_json``
# below — a list of {"role", "content": [{"type": ..., ...}]} turns. If
# megatron-bridge changes this contract the dataset will build cleanly but
# training silently misreads. There is no contract test today; if you bump
# megatron-bridge across an Energon major version, run a one-sample smoke
# load through ChatMLWebdataset before promoting the dataset.
_DATASET_YAML = {
    "__module__": "megatron.bridge.data.energon.task_encoder_utils",
    "__class__": "ChatMLWebdataset",
    "field_map": {
        "conversation": "conversation.json",
        "audio": "audio.wav",
        "videos": "video.mp4",
    },
    "subflavors": {},
}


# =============================================================================
# Run Context + Result
# =============================================================================


@dataclass(frozen=True)
class SftOmniRunContext:
    """Metadata for the run — passed to finalize."""

    run_hash: str
    run_dir: str
    dataset_path: str
    splits: tuple[str, ...]


@dataclass(frozen=True)
class SftOmniFormatResult:
    """Result returned by ``finalize_sft_omni_run``."""

    run_hash: str
    run_dir: str
    dataset_path: Path
    split_sample_counts: dict[str, int]
    num_shards: int


# =============================================================================
# ChatML conversation builder (recipe contract — see schema note above)
# =============================================================================


def _build_conversation_json(question: str, answer: str) -> str:
    """Serialize a single QA turn into Energon ChatML conversation JSON."""
    conversation = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return json.dumps(conversation)


# =============================================================================
# Driver: Setup
# =============================================================================


def _build_run_hash(
    *,
    dataset_path: Path,
    samples_per_shard: int,
    sample: int | None,
    force: bool,
    splits: tuple[str, ...],
) -> str:
    """Deterministic hash for the run; ``force`` salts with current time."""
    payload = {
        "kind": "sft_omni",
        "dataset_path": str(dataset_path),
        "samples_per_shard": int(samples_per_shard),
        "sample": sample,
        "splits": list(splits),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return digest if not force else f"{digest}_{int(time.time())}"


def setup_sft_omni_run(
    *,
    videos_by_id: dict[str, Path],
    qa_records: dict[str, list[tuple[str, str, str]]],
    audio_dir: Path,
    dataset_path: Path,
    runs_root: Path,
    samples_per_shard: int,
    sample: int | None = None,
    force: bool = False,
) -> tuple[list[VideoExtractWorkItem], list[WebDatasetShardWorkItem], SftOmniRunContext]:
    """Build work items for both pipelines.

    Caller-supplied inputs:
        videos_by_id: mapping from a stable video key (e.g. youtube_id) to the
            local MP4 path. The recipe doesn't care how filenames are laid out;
            the caller resolves any layout-specific naming before calling here.
        qa_records: per-split list of ``(video_id, question, answer)`` triples.
            ``video_id`` must be a key in ``videos_by_id``; entries pointing at
            unknown videos are dropped silently. Caller is responsible for any
            QA-shape conversion (MCQ → text answer, free-form → text, etc.).
        audio_dir: where extracted WAVs land. Audio path for each video is
            ``audio_dir / f"{video_path.stem}.wav"``.
        dataset_path: where the published Energon dataset goes.
        runs_root: where this run's receipt namespace lives. MUST be outside
            ``dataset_path`` — otherwise a non-force resumption that wipes the
            dataset for cleanup would also drop the receipts and defeat
            ReceiptManager-based resumability.

    Audio-extract items are deduplicated by video_id (each video extracted
    once even if multiple QA pairs reference it). Shard work items are
    grouped by video_id so all QA pairs for one video land in the same tar
    shard — this makes the audio-extract pipeline strictly upstream of the
    shard-write pipeline, with no per-video work duplicated.
    """
    audio_dir = audio_dir.expanduser().resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_path.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()

    splits = tuple(qa_records.keys())

    run_hash = _build_run_hash(
        dataset_path=dataset_path,
        samples_per_shard=samples_per_shard,
        sample=sample,
        force=force,
        splits=splits,
    )
    run_dir = str(runs_root / run_hash)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    plan_hash = run_hash  # one plan per run for now
    receipts_root = f"{run_dir}/receipts"
    Path(receipts_root).mkdir(parents=True, exist_ok=True)

    # Optional sample cap — bound the work to the first N video keys
    # (sorted for determinism).
    effective_videos = videos_by_id
    if sample is not None:
        keys = sorted(videos_by_id)[:sample]
        effective_videos = {k: videos_by_id[k] for k in keys}

    # --- per-video audio-extract items ---------------------------------------
    audio_items: list[VideoExtractWorkItem] = []
    for video_id, video_path in effective_videos.items():
        audio_path = audio_dir / f"{video_path.stem}.wav"
        audio_items.append(
            VideoExtractWorkItem(
                video_path=str(video_path),
                audio_path=str(audio_path),
                youtube_id=video_id,
                run_hash=run_hash,
                run_dir=run_dir,
            )
        )

    # --- per-shard write items -----------------------------------------------
    # Shards group QA records up to samples_per_shard each, but never split a
    # single video across shards. This guarantees that once a video's audio is
    # extracted no future shard re-needs it.
    shard_items: list[WebDatasetShardWorkItem] = []
    for split in splits:
        per_video: dict[str, list[tuple[str, str]]] = defaultdict(list)
        order: list[str] = []
        for video_id, question, answer in qa_records[split]:
            if video_id not in effective_videos:
                continue
            if video_id not in per_video:
                order.append(video_id)
            per_video[video_id].append((question, answer))

        current: list[tuple[str, str, str]] = []
        shard_index = 0

        def _flush() -> None:
            nonlocal current, shard_index
            if not current:
                return
            shard_items.append(
                WebDatasetShardWorkItem(
                    split=split,
                    shard_index=shard_index,
                    plan_hash=plan_hash,
                    output_dir=str(dataset_path),  # flat layout: {split}-shard-NNNNNN.tar at root
                    receipts_dir=f"{receipts_root}/shards/{split}",
                    records=tuple(current),
                    run_hash=run_hash,
                    run_dir=run_dir,
                )
            )
            shard_index += 1
            current = []

        for video_id in order:
            video_path = effective_videos[video_id]
            audio_path = audio_dir / f"{video_path.stem}.wav"
            for question, answer in per_video[video_id]:
                current.append(
                    (str(video_path), str(audio_path), _build_conversation_json(question, answer))
                )
                if len(current) >= samples_per_shard:
                    _flush()
        _flush()

    Path(f"{run_dir}/config.json").write_text(json.dumps({
        "run_hash": run_hash,
        "dataset_path": str(dataset_path),
        "samples_per_shard": samples_per_shard,
        "sample": sample,
        "splits": list(splits),
        "num_videos": len(audio_items),
        "num_shards": len(shard_items),
    }, indent=2))

    context = SftOmniRunContext(
        run_hash=run_hash,
        run_dir=run_dir,
        dataset_path=str(dataset_path),
        splits=splits,
    )
    return audio_items, shard_items, context


# =============================================================================
# Driver: Finalize
# =============================================================================


def _restructure_shards_to_flat_layout(dataset_path: Path) -> dict[str, list[str]]:
    """Move shards from per-split subdirs into the Energon flat root layout.

    Driver-side step (no benefit from parallelism). The shard-write stage emits
    tars directly to the dataset root using the ``{split}-shard-NNNNNN.tar``
    naming, so this is mostly a no-op now — kept as a guard for older runs
    or for the case where output_dir contains stray per-split subdirs.
    """
    split_shards: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        flat = sorted(p.name for p in dataset_path.glob(f"{split}-shard-*.tar"))
        if flat:
            split_shards[split] = flat
            continue
        split_dir = dataset_path / split
        if split_dir.is_dir():
            tars = sorted(p.name for p in split_dir.iterdir() if p.suffix == ".tar")
            renamed: list[str] = []
            for tar in tars:
                new_name = f"{split}-{tar}"
                (split_dir / tar).rename(dataset_path / new_name)
                renamed.append(new_name)
            split_dir.rmdir()
            split_shards[split] = renamed
    return split_shards


def _build_energon_index(dataset_path: Path, split_shards: dict[str, list[str]]) -> int:
    """Build .nv-meta/{.info.yaml,index.sqlite,index.uuid,split.yaml,dataset.yaml}.

    Returns total sample count across all shards.

    Bypasses ``energon prepare`` (deadlocks in all modes on the pinned
    megatron-energon: AggregatorPool.close → aggregator_process.join blocks
    indefinitely; the sqlite is always complete before the hang). Uses stdlib
    tarfile to read member byte offsets directly instead.
    """
    import yaml

    meta_dir = dataset_path / ".nv-meta"
    meta_dir.mkdir(exist_ok=True)

    ordered_shards = [
        name for split in ("train", "val", "test") for name in split_shards.get(split, [])
    ]

    db_path = meta_dir / "index.sqlite"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE samples ("
        "  id           INTEGER PRIMARY KEY,"
        "  tar_file_id  INTEGER,"
        "  sample_key   TEXT,"
        "  sample_index INTEGER,"
        "  byte_offset  INTEGER,"
        "  byte_size    INTEGER)"
    )
    conn.execute("CREATE INDEX idx_samples_sample_key ON samples(sample_key)")

    shard_counts: dict[str, int] = {}
    for tar_file_id, shard_name in enumerate(ordered_shards):
        with tarfile.open(dataset_path / shard_name) as tf:
            members = tf.getmembers()

        groups: dict[str, list] = defaultdict(list)
        for m in members:
            groups[m.name.split(".", 1)[0]].append(m)

        ordered_groups = sorted(groups.items(), key=lambda kv: min(m.offset for m in kv[1]))
        rows = []
        for sample_index, (sample_key, mems) in enumerate(ordered_groups):
            mems_sorted = sorted(mems, key=lambda m: m.offset)
            byte_offset = mems_sorted[0].offset
            if sample_index + 1 < len(ordered_groups):
                byte_size = min(m.offset for m in ordered_groups[sample_index + 1][1]) - byte_offset
            else:
                last = mems_sorted[-1]
                byte_size = last.offset_data + ((last.size + 511) // 512) * 512 - byte_offset
            rows.append((tar_file_id, sample_key, sample_index, byte_offset, byte_size))

        conn.executemany(
            "INSERT INTO samples (tar_file_id, sample_key, sample_index, byte_offset, byte_size)"
            " VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
        shard_counts[shard_name] = len(rows)

    conn.close()
    total = sum(shard_counts.values())
    logger.info(f"index.sqlite: {total} samples across {len(ordered_shards)} shards")

    with open(meta_dir / ".info.yaml", "w") as f:
        yaml.dump({"shard_counts": shard_counts}, f)
    with open(meta_dir / "index.uuid", "w") as f:
        f.write(str(uuid.uuid4()))
    with open(meta_dir / "split.yaml", "w") as f:
        yaml.dump({"split_parts": dict(split_shards), "exclude": []}, f)
    with open(meta_dir / "dataset.yaml", "w") as f:
        yaml.dump(_DATASET_YAML, f, sort_keys=False)

    return total


def finalize_sft_omni_run(context: SftOmniRunContext) -> SftOmniFormatResult:
    """Driver-side finalize: restructure shards, write Energon metadata, count samples."""
    dataset_path = Path(context.dataset_path)
    split_shards = _restructure_shards_to_flat_layout(dataset_path)
    _build_energon_index(dataset_path, split_shards)

    import yaml as pyyaml
    info = pyyaml.safe_load((dataset_path / ".nv-meta" / ".info.yaml").read_text())
    counts = info.get("shard_counts", {}) or {}

    split_sample_counts: dict[str, int] = {split: 0 for split in context.splits}
    for shard_name, n in counts.items():
        for split in context.splits:
            if shard_name.startswith(f"{split}-"):
                split_sample_counts[split] += int(n)
                break

    num_shards = sum(len(v) for v in split_shards.values())

    return SftOmniFormatResult(
        run_hash=context.run_hash,
        run_dir=context.run_dir,
        dataset_path=dataset_path,
        split_sample_counts=split_sample_counts,
        num_shards=num_shards,
    )


def clear_dataset_path(dataset_path: Path) -> None:
    """Remove any stale Energon output before a forced rebuild.

    Only intended for ``force=True`` callers — a non-force resumption should
    NOT call this, because the shard-write stage's ReceiptManager will skip
    already-complete shards. Callers must ensure receipts live OUTSIDE
    ``dataset_path`` (see ``setup_sft_omni_run`` ``runs_root`` parameter);
    receipts colocated with the dataset would be wiped here.

    ``raw_dir``-style intermediate directories are preserved by design — the
    audio extraction step is expensive and idempotent across runs.
    """
    if dataset_path.exists():
        shutil.rmtree(dataset_path)


# =============================================================================
# Convenience Entry Point
# =============================================================================


def _run_pipeline_with_wandb(
    *,
    spec: pipelines_v1.PipelineSpec,
    pipeline_kind: str,
    pipeline_ctx: PipelineContext,
    num_items: int,
) -> None:
    """Run a pipeline with the lower-level wandb stats hook attached.

    Uses ``make_wandb_stats_hook`` directly rather than ``pipeline_wandb_hook``
    because the latter expects per-dataset / per-shard work items, while our
    audio-extract pipeline has per-video items that don't fit that abstraction.
    Same observability output, different attachment point.
    """
    hook = make_wandb_stats_hook(
        observability=pipeline_ctx.observability,
        pipeline_kind=pipeline_kind,
        run_hash=pipeline_ctx.run_hash,
        run_dir=pipeline_ctx.run_dir,
        dataset_names=["sft_omni"],
        dataset_num_shards={"sft_omni": num_items},
    )
    if hook:
        with hook:
            pipelines_v1.run_pipeline(spec)
    else:
        pipelines_v1.run_pipeline(spec)


def run_sft_omni_pipeline(
    *,
    videos_by_id: dict[str, Path],
    qa_records: dict[str, list[tuple[str, str, str]]],
    audio_dir: Path,
    dataset_path: Path,
    runs_root: Path,
    samples_per_shard: int,
    sample: int | None = None,
    force: bool = False,
    audio_workers_per_node: float = 4.0,
    shard_workers_per_node: float = 2.0,
    audio_extract_config: AudioExtractStageConfig | None = None,
    shard_write_config: WebDatasetShardStageConfig | None = None,
    observability: ObservabilityConfig | None = None,
    execution_mode: ExecutionModeRequest = "auto",
) -> SftOmniFormatResult:
    """Convenience wrapper: setup → run audio + shard pipelines → finalize.

    For full control over the pipeline stages (e.g. attaching extra hooks
    between phases), use ``setup_sft_omni_run`` and ``finalize_sft_omni_run``
    with explicit ``PipelineSpec`` construction in your driver script.
    """
    audio_extract_cfg = audio_extract_config or AudioExtractStageConfig()
    shard_write_cfg = shard_write_config or WebDatasetShardStageConfig()
    observability_cfg = observability or ObservabilityConfig()

    # Phase 1: Setup
    audio_items, shard_items, context = setup_sft_omni_run(
        videos_by_id=videos_by_id,
        qa_records=qa_records,
        audio_dir=audio_dir,
        dataset_path=dataset_path,
        runs_root=runs_root,
        samples_per_shard=samples_per_shard,
        sample=sample,
        force=force,
    )

    pipeline_ctx = PipelineContext(
        output_root=str(dataset_path),
        run_hash=context.run_hash,
        run_dir=context.run_dir,
        config_hash=None,
        observability=observability_cfg,
        hf_env=detect_hf_env_vars(),
    )

    # Phase 2: Audio-extract pipeline
    if audio_items:
        audio_specs = [
            pipelines_v1.StageSpec(
                AudioExtractStage(audio_extract_cfg, pipeline_ctx),
                num_workers_per_node=audio_workers_per_node,
            ),
        ]
        audio_spec = pipelines_v1.PipelineSpec(
            input_data=audio_items,
            stages=audio_specs,
            config=pipelines_v1.PipelineConfig(
                execution_mode=resolve_execution_mode(audio_specs, execution_mode),
                logging_interval_s=observability_cfg.pipeline_logging_interval_s,
            ),
        )
        _run_pipeline_with_wandb(
            spec=audio_spec,
            pipeline_kind="sft_omni-audio",
            pipeline_ctx=pipeline_ctx,
            num_items=len(audio_items),
        )

    # Phase 3: Shard-write pipeline
    if shard_items:
        shard_specs = [
            pipelines_v1.StageSpec(
                WebDatasetShardStage(shard_write_cfg, pipeline_ctx),
                num_workers_per_node=shard_workers_per_node,
            ),
        ]
        shard_spec = pipelines_v1.PipelineSpec(
            input_data=shard_items,
            stages=shard_specs,
            config=pipelines_v1.PipelineConfig(
                execution_mode=resolve_execution_mode(shard_specs, execution_mode),
                logging_interval_s=observability_cfg.pipeline_logging_interval_s,
            ),
        )
        _run_pipeline_with_wandb(
            spec=shard_spec,
            pipeline_kind="sft_omni-shards",
            pipeline_ctx=pipeline_ctx,
            num_items=len(shard_items),
        )

    # Phase 4: Finalize
    return finalize_sft_omni_run(context)


__all__ = [
    "SftOmniFormatResult",
    "SftOmniRunContext",
    "clear_dataset_path",
    "finalize_sft_omni_run",
    "run_sft_omni_pipeline",
    "setup_sft_omni_run",
]
