#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/data/prep/sft"
# image = "anyscale/ray:2.49.2-py312"
# setup = """
# Requires the full nemotron repository synced to the worker. Install with:
#   uv sync --reinstall-package nemotron --extra audio
# The `audio` extra pulls in `webdataset` and `imageio-ffmpeg` (a static ffmpeg
# binary) which the Valor32k flow needs. The HF and generic-Energon flows do
# not need the extra; bare `uv sync --reinstall-package nemotron` is enough
# for those.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra xenna python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/data_prep"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
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

"""Data preparation for Omni3 SFT — three explicit flows, dispatched by config.

Flows:
    HF Hub flow (``hf_dataset_id`` set) — open data, no shard building. The
        training container pulls from the Hub on-demand; this step records
        the dataset ID in a manifest so artifact lineage flows through W&B.

    Valor32k flow (``dataset_name == "valor32k"``) — fully self-contained
        Cosmos-Xenna pipeline that extracts audio from a source video tar
        and assembles an Energon WebDataset. Mirrors the super3/nano3 SFT
        pattern: setup → run_pipeline (× 2 for audio + shards) → finalize
        → artifact. Builders live in ``nemotron.data_prep.recipes.sft_omni``
        and ``nemotron.data_prep.stages.{audio_extract,webdataset_shard}``.

    Generic Energon flow (``dataset_path`` set, no special dataset_name) —
        validates a pre-built Energon directory exists at the given path
        and emits a manifest. Used when shards already live on shared
        storage (e.g. another team built them).

The default config uses the HF flow (CORD-v2) so ``nemotron omni3 data prep
sft`` works without internal dataset access. Use ``-c valor32k`` for the
Valor32k self-build path; ``-c tiny`` for an HF smoke test.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from nemo_runspec.artifacts import ArtifactTrackingResult, log_artifact, setup_artifact_tracking
from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.recipes.sft_omni import clear_dataset_path, run_sft_omni_pipeline
from nemotron.data_prep.stages import AudioExtractStageConfig, WebDatasetShardStageConfig
from nemotron.kit import EnergonWebDatasetArtifact, print_step_complete, wandb_kit
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    init_wandb_from_env,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

logger = logging.getLogger(__name__)

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "default.yaml"
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))

# Module-level flag for Ray execution (used by nemotron CLI dispatcher).
RAY = True

# ---- Valor32k-specific constants -------------------------------------------
# All knowledge of Valor32k's source layout (URL, tar structure, filename
# convention, QA shape) lives in this file. The shared SFT-Omni recipe
# (nemotron.data_prep.recipes.sft_omni) is generic over these.
VALOR32K_QA_ZIP_URL = "https://github.com/inesriahi/valor32k-avqa-2/raw/refs/heads/main/data.zip"
# Valor32k filenames have the form {youtube_id}_{start.sss}_{end.sss}.mp4.
# QA records key by the bare youtube_id, so we strip the timestamp suffix.
_VALOR32K_TS_SUFFIX_RE = re.compile(r"^(.+)_\d+\.\d+_\d+\.\d+$")


@dataclass
class Omni3SFTDataPrepConfig:
    """Omni SFT data preparation config.

    Mirrors the structure of super3/nano3 SFT data-prep configs:
    - dataset selection (mutually exclusive flow knobs)
    - per-stage nested configs (audio_extract, shard_write)
    - run control (sample, force, execution_mode, config_name)
    - observability
    """

    # --- Flow selection (exactly one) ---
    dataset_name: str = "cord_v2"
    """Dataset identifier. ``valor32k`` triggers the self-build pipeline;
    other values fall through to the HF or generic-Energon flow."""

    hf_dataset_id: str | None = None
    """HuggingFace dataset ID for the HF-Hub flow (e.g. ``naver-clova-ix/cord-v2``)."""

    dataset_path: Path | None = None
    """Energon dataset directory. Required for the valor32k and
    generic-Energon flows; ignored for the HF-Hub flow."""

    # --- Valor32k-specific knobs ---
    raw_dir: Path | None = None
    """Intermediate workdir for extracted videos + audio. Defaults to
    ``<dataset_path>/../raw`` when null."""

    source_tar: Path | None = None
    """Path to the VALOR-32K source tar. Optional if ``raw_dir/videos/`` is
    already populated by hand."""

    samples_per_shard: int = 100
    """QA records per Energon WebDataset shard. Valor32k-specific."""

    qa_zip_url: str = VALOR32K_QA_ZIP_URL
    """URL of the QA annotations zip. Override for a custom QA fork."""

    strip_components: int = 4
    """``--strip-components`` value when extracting the source tar.
    Default 4 matches ``raid/datasets/audioset/valor_videos/*.mp4``."""

    audio_workers_per_node: float = 4.0
    """Audio-extract workers per node. ffmpeg is one-CPU per worker."""

    shard_workers_per_node: float = 2.0
    """Shard-write workers per node. I/O bound; 2 per node is plenty."""

    # --- Common bookkeeping ---
    metadata_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "stage0_sft/data_prep")
    """Where the staging manifest lands (NEMO_RUN_DIR-relative for cluster runs)."""

    link_path: Path | None = None
    """Optional convenience symlink pointing at the Energon output dir."""

    modality_filter: str | None = None
    """Reserved for downstream filtering by sample modality (audio/visual/text)."""

    sample: int | None = None
    """Quick-test cap on number of videos processed."""

    force: bool = False
    """Force a rebuild even when outputs look complete."""

    execution_mode: str = "auto"
    """Execution mode: 'auto' (default), 'streaming', or 'batch'."""

    config_name: str = "default"
    """Config name used for artifact naming."""

    # --- Stage configs (nested) ---
    audio_extract: AudioExtractStageConfig = field(default_factory=AudioExtractStageConfig)
    """Configuration for the per-video ffmpeg stage."""

    shard_write: WebDatasetShardStageConfig = field(default_factory=WebDatasetShardStageConfig)
    """Configuration for the per-shard tar-build stage."""

    # --- Pipeline-level config ---
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    """Pipeline observability settings."""

    def __post_init__(self) -> None:
        # Ensure paths are Path objects (omegaconf gives us strings)
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.raw_dir, str):
            self.raw_dir = Path(self.raw_dir)
        if isinstance(self.source_tar, str):
            self.source_tar = Path(self.source_tar)
        if isinstance(self.metadata_dir, str):
            self.metadata_dir = Path(self.metadata_dir)
        if isinstance(self.link_path, str):
            self.link_path = Path(self.link_path)


# ----------------------------------------------------------------------------
# Shared helpers (manifest, link)
# ----------------------------------------------------------------------------


def _materialize_link(source: Path, target: Path, force: bool) -> None:
    if target.exists() or target.is_symlink():
        if not force:
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def _write_manifest(metadata_dir: Path, manifest: dict) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest_path


# ----------------------------------------------------------------------------
# HF-Hub flow (manifest only)
# ----------------------------------------------------------------------------


def _run_hf_prep(cfg: Omni3SFTDataPrepConfig, tracking: ArtifactTrackingResult | None) -> Path:
    """Emit a manifest referencing the HF dataset; no shard building."""
    return _write_manifest(
        cfg.metadata_dir.expanduser(),
        {
            "dataset_name": cfg.dataset_name,
            "source": "huggingface",
            "hf_dataset_id": cfg.hf_dataset_id,
            "modality_filter": cfg.modality_filter,
            "sample": cfg.sample,
        },
    )


# ----------------------------------------------------------------------------
# Generic Energon flow (validate-only)
# ----------------------------------------------------------------------------


def _run_energon_validate(cfg: Omni3SFTDataPrepConfig, tracking: ArtifactTrackingResult | None) -> Path:
    """Validate a pre-built Energon dataset and emit a manifest."""
    assert cfg.dataset_path is not None
    dataset_path = cfg.dataset_path.expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Energon dataset not found at {dataset_path}. Either set "
            "dataset_name=valor32k for the self-build flow, or point "
            "dataset_path at a pre-built Energon directory."
        )
    if cfg.link_path:
        _materialize_link(dataset_path.resolve(), cfg.link_path.expanduser(), cfg.force)
    return _write_manifest(
        cfg.metadata_dir.expanduser(),
        {
            "dataset_name": cfg.dataset_name,
            "source": "energon",
            "dataset_path": str(dataset_path.resolve()),
            "modality_filter": cfg.modality_filter,
            "sample": cfg.sample,
        },
    )


# ----------------------------------------------------------------------------
# Valor32k-specific helpers (URL, tar layout, filename convention, QA shape)
# ----------------------------------------------------------------------------
#
# These four helpers own everything Valor32k-flavored about this recipe.
# The shared SFT-Omni pipeline doesn't know about any of this — it takes
# already-resolved video paths and (video_id, question, answer) triples.
# A future video-language SFT family would write their own equivalents here
# (different URL, different tar layout, different QA shape) and reuse the
# generic recipe unchanged. "Locality over DRY."


def _download_valor32k_qa(output_dir: Path, qa_zip_url: str, *, timeout_sec: int = 60) -> None:
    """Fetch the Valor32k QA zip and unpack the per-split JSONs (idempotent)."""
    logger.info(f"Downloading QA annotations from {qa_zip_url}")
    with urlopen(qa_zip_url, timeout=timeout_sec) as resp:
        payload = resp.read()
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        for member in zf.namelist():
            if not member.endswith(".json"):
                continue
            target = output_dir / Path(member).name
            if target.exists():
                logger.info(f"  Already exists: {target.name}")
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            logger.info(f"  Extracted {target.name}")


def _extract_valor32k_tar(
    source_tar: Path,
    videos_dir: Path,
    strip_components: int,
) -> None:
    """Extract the VALOR-32K source tar; idempotent and validated.

    The canonical tar layout is ``raid/datasets/audioset/valor_videos/*.mp4``
    (4 path segments — hence the default ``strip_components=4``). Custom tars
    may have a different depth; we fail fast with a layout-mismatch error
    rather than letting downstream stages chew through an empty directory.
    """
    if videos_dir.exists() and any(videos_dir.glob("*.mp4")):
        return
    source_tar = source_tar.expanduser().resolve()
    if not source_tar.is_file():
        raise FileNotFoundError(f"source_tar not found: {source_tar}")

    videos_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Extracting source tar {source_tar} into {videos_dir} "
        "(this can take 5-10 min on shared storage)"
    )
    subprocess.check_call(
        ["tar", "xf", str(source_tar), "-C", str(videos_dir),
         f"--strip-components={strip_components}"]
    )

    if any(videos_dir.glob("*.mp4")):
        return
    nested = list(videos_dir.rglob("*.mp4"))
    if nested:
        depth = len(nested[0].relative_to(videos_dir).parts) - 1
        raise RuntimeError(
            f"Tar extracted but no *.mp4 files at {videos_dir}. Found {len(nested)} "
            f"MP4s nested {depth} extra level(s) deep (first: {nested[0]}). "
            f"Adjust strip_components (currently {strip_components}) for your tar layout."
        )
    raise RuntimeError(
        f"Tar extraction produced no MP4 files under {videos_dir}. "
        f"strip_components={strip_components} did not match this tar's layout."
    )


def _index_valor32k_videos(videos_dir: Path) -> dict[str, Path]:
    """Map bare youtube_id → video path, stripping ``_start_end`` timestamp suffix."""
    index: dict[str, Path] = {}
    for p in videos_dir.glob("*.mp4"):
        m = _VALOR32K_TS_SUFFIX_RE.match(p.stem)
        key = m.group(1) if m else p.stem
        index[key] = p
    return index


def _load_valor32k_qa_records(
    raw_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, list[tuple[str, str, str]]]:
    """Read Valor32k MCQ JSONs and resolve to ``(video_id, question, answer)`` triples.

    Valor32k stores each QA as ``{video_id, question, options, correct_answer_idx}``
    — i.e. multiple-choice. We resolve ``options[correct_answer_idx]`` here so
    the shared recipe sees only the textual question/answer pair.
    """
    out: dict[str, list[tuple[str, str, str]]] = {}
    for split in splits:
        qa_file = raw_dir / f"combined_dataset_{split}_flattened.json"
        if not qa_file.exists():
            continue
        with open(qa_file) as f:
            data = json.load(f)
        out[split] = [
            (str(qa["video_id"]), qa["question"], qa["options"][qa["correct_answer_idx"]])
            for qa in data
        ]
    return out


# ----------------------------------------------------------------------------
# Valor32k flow (orchestrates the generic SFT-Omni pipeline)
# ----------------------------------------------------------------------------


def _run_valor32k_pipeline(
    cfg: Omni3SFTDataPrepConfig,
    tracking: ArtifactTrackingResult | None,
) -> EnergonWebDatasetArtifact:
    """Drive the shared SFT-Omni pipeline against Valor32k inputs.

    Phases:
        1. Valor32k-specific prep — source-tar extract, QA annotation download
        2. Resolve to generic structures — video index + QA triples
        3. Run the shared pipeline — ``run_sft_omni_pipeline`` does setup,
           audio extract, shard write, and finalize
        4. Build the artifact and emit the staging manifest

    All Valor32k-specific knowledge (URL, tar layout, filename regex, MCQ
    answer extraction) lives in the helper functions above; this orchestrator
    just wires them up.
    """
    if cfg.dataset_path is None:
        raise ValueError("valor32k flow requires dataset_path (Energon output directory)")

    start_time = time.time()
    wandb_kit.add_run_tags(["data-prep", "sft", "valor32k", cfg.config_name])
    wandb_kit.log_wandb_config(cfg)

    dataset_path = cfg.dataset_path.expanduser().resolve()
    raw_dir = (
        cfg.raw_dir.expanduser().resolve()
        if cfg.raw_dir is not None
        else dataset_path.parent / "raw"
    )
    metadata_dir = cfg.metadata_dir.expanduser()
    sentinel = dataset_path / ".nv-meta" / "dataset.yaml"
    needs_build = cfg.force or not sentinel.exists()

    # Receipts live outside dataset_path so non-force resume preserves them.
    runs_root = (metadata_dir / "runs").resolve()

    # ---- Phase 1: Valor32k-specific prep ----------------------------------
    raw_dir.mkdir(parents=True, exist_ok=True)
    if needs_build:
        # Force-only: wipe stale Energon output (orphan-shard footgun).
        # Non-force resume leaves the dir alone so ReceiptManager can skip
        # already-complete shards.
        if cfg.force:
            clear_dataset_path(dataset_path)

        videos_dir = raw_dir / "videos"
        if cfg.source_tar is not None and not (
            videos_dir.exists() and any(videos_dir.glob("*.mp4"))
        ):
            _extract_valor32k_tar(cfg.source_tar, videos_dir, cfg.strip_components)
        elif not videos_dir.exists() or not any(videos_dir.glob("*.mp4")):
            raise FileNotFoundError(
                f"{videos_dir} contains no MP4 files and source_tar is unset. "
                "Either populate videos/ manually, or set source_tar / "
                "OMNI3_VALOR32K_VIDEOS_TAR to the source tar path."
            )

        _download_valor32k_qa(raw_dir, cfg.qa_zip_url)

    # ---- Phase 2: resolve to generic structures ---------------------------
    videos_by_id = _index_valor32k_videos(raw_dir / "videos")
    qa_records = _load_valor32k_qa_records(raw_dir)

    # ---- Phase 3: run the shared SFT-Omni pipeline ------------------------
    format_result = run_sft_omni_pipeline(
        videos_by_id=videos_by_id,
        qa_records=qa_records,
        audio_dir=raw_dir / "audio",
        dataset_path=dataset_path,
        runs_root=runs_root,
        samples_per_shard=cfg.samples_per_shard,
        sample=cfg.sample,
        force=cfg.force,
        audio_workers_per_node=cfg.audio_workers_per_node,
        shard_workers_per_node=cfg.shard_workers_per_node,
        audio_extract_config=cfg.audio_extract,
        shard_write_config=cfg.shard_write,
        observability=cfg.observability,
        execution_mode=cfg.execution_mode,
    )

    if not sentinel.exists():
        raise FileNotFoundError(
            f"Valor32k build did not produce a complete Energon dataset at {dataset_path} "
            f"(missing {sentinel}). Check the pipeline logs above."
        )

    if cfg.link_path:
        _materialize_link(dataset_path, cfg.link_path.expanduser(), cfg.force)

    # ---- Phase 4: artifact + manifest -------------------------------------
    elapsed_sec = time.time() - start_time
    artifact_name = f"omni3/sft/valor32k{('?sample=' + str(cfg.sample)) if cfg.sample else ''}"
    source_uri_parts = [cfg.qa_zip_url]
    if cfg.source_tar:
        source_uri_parts.append(f"file://{cfg.source_tar}")

    artifact = EnergonWebDatasetArtifact.from_run(
        dataset_path=dataset_path,
        split_sample_counts=format_result.split_sample_counts,
        num_shards=format_result.num_shards,
        source_datasets=source_uri_parts,
        elapsed_sec=elapsed_sec,
        name=artifact_name,
    )
    if tracking is not None:
        log_artifact(artifact, tracking)
    else:
        artifact.save()

    # Also drop the staging manifest for symmetry with other flows.
    _write_manifest(
        metadata_dir,
        {
            "dataset_name": cfg.dataset_name,
            "source": "energon-valor32k",
            "dataset_path": str(dataset_path),
            "raw_dir": str(raw_dir),
            "source_tar": str(cfg.source_tar) if cfg.source_tar else None,
            "samples_per_shard": cfg.samples_per_shard,
            "sample": cfg.sample,
            "split_sample_counts": format_result.split_sample_counts,
            "num_shards": format_result.num_shards,
            "elapsed_sec": elapsed_sec,
        },
    )

    wandb_kit.finish_run(exit_code=0)
    print_step_complete(data_prep=artifact)
    return artifact


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------


def run_data_prep(
    cfg: Omni3SFTDataPrepConfig,
    tracking: ArtifactTrackingResult | None = None,
) -> Path | EnergonWebDatasetArtifact:
    """Dispatch to the right flow based on dataset_name / which fields are set."""
    if cfg.hf_dataset_id and cfg.dataset_path:
        raise ValueError(
            "Set exactly one of hf_dataset_id (HF-Hub flow) or dataset_path "
            "(Energon flow), not both."
        )
    if cfg.dataset_name == "valor32k" and cfg.hf_dataset_id:
        raise ValueError(
            "dataset_name='valor32k' triggers the self-build flow; hf_dataset_id "
            "is incompatible. Drop one."
        )
    if cfg.dataset_name == "valor32k":
        return _run_valor32k_pipeline(cfg, tracking)
    if cfg.hf_dataset_id:
        return _run_hf_prep(cfg, tracking)
    if cfg.dataset_path:
        return _run_energon_validate(cfg, tracking)
    raise ValueError(
        "No data prep flow selected. Set hf_dataset_id (HF flow), or "
        "dataset_name=valor32k + dataset_path (Valor32k self-build), or "
        "dataset_path alone (validate a pre-built Energon dataset)."
    )


def main(cfg: Omni3SFTDataPrepConfig | None = None) -> Path | EnergonWebDatasetArtifact:
    """Entry point for omni3 SFT data prep.

    Mirrors the pattern used by super3/nano3 SFT data prep: when run directly
    we parse the config ourselves; when run via the CLI framework we receive
    an already-constructed dataclass.
    """
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)
        tracking = setup_artifact_tracking(config)
        cfg = omegaconf_to_dataclass(config, Omni3SFTDataPrepConfig)
    else:
        tracking = None

    if tracking is None or tracking.wandb:
        init_wandb_from_env()

    return run_data_prep(cfg, tracking=tracking)


if __name__ == "__main__":
    main()
