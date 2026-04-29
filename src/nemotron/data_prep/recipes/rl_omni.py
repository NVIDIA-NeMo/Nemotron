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
Omni RL Multimodal Pipeline Recipe — single-stage MMPR cache build.

This recipe drives the MMPR-Tiny (vision RL) and MMPR-public (MPO) data
prep flows for Nemotron-3 Omni:

    Pipeline: [VlmPreferencePrepWorkItem] → VlmPreferencePrepStage  (single worker)

    + Driver-side finalize (read receipt, materialize result paths)

Counterpart to ``recipes/sft_omni.py`` (video-language SFT) and
``recipes/rl.py`` (text RL with HF placeholder resolution). Same
``setup_*_run`` / ``finalize_*_run`` shape so a recipe-level reader sees
a uniform idiom across families.

Key design decisions:

- **Single-worker, single-stage by design.** The current VLM-preference
  cache build is bounded by one zip extraction and one parquet
  validation; multi-worker fan-out would add ceremony with no throughput
  win at current dataset sizes. The framework value is consistency
  (run-hash caching, receipts, artifact lineage), not parallelism. When
  a future multimodal preference dataset arrives that genuinely benefits
  from per-archive fan-out, the recipe surface stays — only the
  work-item shape and ``num_workers`` change.

- **Two flavors share one stage.** ``flavor="tiny"`` runs vendored logic
  for MMPR-Tiny → vision RL; ``flavor="mpo"`` shells out to the upstream
  prep script via ``builder_command``. This is a deliberate transitional
  state: until ``prepare_public_mmpr_for_mpo.py`` can be vendored cleanly,
  the MPO path keeps subprocess semantics while still gaining the
  framework's resumability and lineage. The migration is mechanical when
  the script becomes available.

- **Receipts live OUTSIDE ``output_dir``** (caller-supplied
  ``runs_root``). Same lesson as ``recipes/sft_omni.py``: cleanup of the
  published cache must not drop receipts. Without this, non-force resume
  becomes a footgun.

Usage::

    from nemotron.data_prep.recipes.rl_omni import run_rl_omni_pipeline

    result = run_rl_omni_pipeline(
        flavor="tiny",
        raw_dir=Path("/data/mmpr_tiny/raw"),
        output_dir=Path("/data/mmpr_tiny/processed"),
        runs_root=Path("/data/mmpr_tiny/runs"),
        meta_name="meta_public.json",
        builder_command=None,
        force=False,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.core.work_items import VlmPreferencePrepWorkItem
from nemotron.data_prep.observability import make_wandb_stats_hook
from nemotron.data_prep.recipes.execution_mode import ExecutionModeRequest, resolve_execution_mode
from nemotron.data_prep.stages import (
    PipelineContext,
    VlmPreferencePrepStage,
    VlmPreferencePrepStageConfig,
)
from nemotron.data_prep.utils.filesystem import get_filesystem, read_json
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars

logger = logging.getLogger(__name__)


# =============================================================================
# Run Context + Result
# =============================================================================


@dataclass(frozen=True)
class RlOmniRunContext:
    """Metadata for the run — passed to finalize."""

    run_hash: str
    run_dir: str
    output_dir: str
    receipts_dir: str
    flavor: Literal["tiny", "mpo"]
    meta_name: str


@dataclass(frozen=True)
class RlOmniPrepResult:
    """Result returned by ``finalize_rl_omni_run``."""

    run_hash: str
    run_dir: str
    output_dir: Path
    flavor: Literal["tiny", "mpo"]
    files: dict[str, str]
    stats: dict[str, int | str]


# =============================================================================
# Driver: Setup
# =============================================================================


# Per-flavor required-file lists used by ``ensure_raw_dir`` to decide
# whether a download is needed. The tiny flavor's vendored prep
# (``VlmPreferencePrepStage._run_tiny``) reads these exact filenames; the
# MPO flavor delegates to the upstream ``prepare_public_mmpr_for_mpo.py``
# script which doesn't expose its required inputs declaratively, so we
# treat "raw_dir is empty / non-existent" as the trigger and download
# the full snapshot. Update _RAW_DIR_REQUIREMENTS when a flavor's prep
# logic changes its file expectations.
_RAW_DIR_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "tiny": ("images.zip", "mmpr_tiny.parquet"),
    "mpo": (),  # opaque to us; presence-of-any-file is the trigger
}


def _normalize_hf_repo_id(source_uri: str) -> str:
    """Strip optional ``hf://`` prefix from a source URI."""
    return source_uri.removeprefix("hf://") if source_uri.startswith("hf://") else source_uri


def _raw_dir_is_present(raw_dir: Path, flavor: str) -> bool:
    """Return True iff ``raw_dir`` already has the files this flavor needs."""
    requirements = _RAW_DIR_REQUIREMENTS.get(flavor, ())
    if requirements:
        return all((raw_dir / name).exists() for name in requirements)
    # No declarative requirements — accept any non-empty directory as
    # "user has staged something."
    return raw_dir.is_dir() and any(raw_dir.iterdir())


def ensure_raw_dir(
    *,
    flavor: Literal["tiny", "mpo"],
    raw_dir: Path,
    source_uri: str | None,
) -> None:
    """Materialize the raw HF download under ``raw_dir`` if absent.

    Idempotent: a return-without-download means ``raw_dir`` already has
    everything the flavor's prep stage needs. Otherwise, when
    ``source_uri`` is set, snapshot-download the HF repo into ``raw_dir``
    and return; when it isn't, raise a ``FileNotFoundError`` with the
    exact remediation step.

    Args:
        flavor: ``"tiny"`` (MMPR-Tiny) or ``"mpo"`` (full MMPR).
        raw_dir: Where the prep stage expects raw inputs to live.
        source_uri: HF repo id (with or without ``hf://`` prefix) to
            download from. Read from the per-stage YAML
            (``vision.yaml`` / ``mpo.yaml``) by callers.
    """
    raw_dir = raw_dir.expanduser().resolve()
    if _raw_dir_is_present(raw_dir, flavor):
        return

    if not source_uri:
        required = _RAW_DIR_REQUIREMENTS.get(flavor, ())
        hint = (
            f"Stage required files {list(required)} under {raw_dir}; "
            "set ``source_uri`` in the stage YAML so the dispatcher "
            "can auto-download from Hugging Face."
            if required
            else f"Stage required content under {raw_dir}; "
            "set ``source_uri`` in the stage YAML so the dispatcher "
            "can auto-download from Hugging Face, or pre-stage the "
            "directory manually."
        )
        raise FileNotFoundError(hint)

    repo_id = _normalize_hf_repo_id(source_uri)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "ensure_raw_dir: downloading %s into %s (flavor=%s) ...",
        repo_id,
        raw_dir,
        flavor,
    )
    # Imported lazily so import cycles in the test path don't drag in
    # huggingface_hub when it isn't needed.
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    if not _raw_dir_is_present(raw_dir, flavor):
        # Defensive: HF snapshots can place files at unexpected paths
        # depending on the dataset. Surface this clearly rather than
        # letting the prep stage emit a more confusing error two layers
        # down.
        required = _RAW_DIR_REQUIREMENTS.get(flavor, ())
        raise FileNotFoundError(
            f"snapshot_download({repo_id!r}) completed but {raw_dir} "
            f"still does not contain the expected files for flavor "
            f"{flavor!r}: {list(required)}. Inspect the downloaded "
            "layout and either symlink the files into place or update "
            "_RAW_DIR_REQUIREMENTS to match the new layout."
        )


def _build_run_hash(
    *,
    flavor: str,
    raw_dir: Path,
    output_dir: Path,
    meta_name: str,
    builder_command: str | None,
    force: bool,
) -> str:
    """Deterministic hash for the run; ``force`` salts with current time."""
    payload = {
        "kind": "rl_omni",
        "flavor": flavor,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "meta_name": meta_name,
        "builder_command": builder_command,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return digest if not force else f"{digest}_{int(time.time())}"


def setup_rl_omni_run(
    *,
    flavor: Literal["tiny", "mpo"],
    raw_dir: Path,
    output_dir: Path,
    runs_root: Path,
    meta_name: str,
    builder_command: str | None = None,
    force: bool = False,
    source_uri: str | None = None,
) -> tuple[list[VlmPreferencePrepWorkItem], RlOmniRunContext]:
    """Build the single work item driving the prep stage.

    Args:
        flavor: ``"tiny"`` (MMPR-Tiny) or ``"mpo"`` (full MMPR).
        raw_dir: Local directory holding the raw HF download.
        output_dir: Target cache directory; receives the prep outputs.
        runs_root: Where this run's receipts live. MUST be outside
            ``output_dir`` so non-force resume preserves receipts after
            cleanup of the published cache.
        meta_name: Name of the metadata file the trainer reads. For tiny,
            this is informational; for mpo, the stage verifies it after the
            builder command runs.
        builder_command: Required for ``flavor="mpo"`` until that flavor's
            logic is vendored. Format string supports ``{input_dir}``,
            ``{output_dir}``, ``{meta_name}``.
        force: Salt the run hash with the current time so cached results
            are bypassed.
        source_uri: HF repo id (e.g. ``hf://OpenGVLab/MMPR-Tiny``) used
            to auto-download into ``raw_dir`` when absent. Pre-staged
            ``raw_dir`` paths skip the download.

    Returns:
        (work_items, context). Work items contain exactly one element.
    """
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()

    # Materialize raw inputs before constructing the work item so the
    # prep stage's missing-file diagnostics never fire on a
    # source_uri-equipped run. Pre-staged operators (raw_dir already
    # populated) get a no-op.
    ensure_raw_dir(flavor=flavor, raw_dir=raw_dir, source_uri=source_uri)

    # Receipts MUST live outside output_dir so a forced rebuild
    # (clear_output_dir(output_dir)) doesn't drop them. We exposed
    # runs_root as a config override on Omni3RLDataPrepConfig, which
    # opened the door to operators accidentally pointing it inside
    # output_dir; check explicitly.
    if runs_root.is_relative_to(output_dir):
        raise ValueError(
            f"runs_root ({runs_root}) must not live inside output_dir "
            f"({output_dir}). A forced rebuild would drop the receipts. "
            "Set runs_root to a sibling or scratch path."
        )

    if flavor == "mpo" and not builder_command:
        raise ValueError(
            "rl_omni flavor='mpo' requires builder_command until the upstream "
            "prep logic is vendored."
        )

    run_hash = _build_run_hash(
        flavor=flavor,
        raw_dir=raw_dir,
        output_dir=output_dir,
        meta_name=meta_name,
        builder_command=builder_command,
        force=force,
    )
    run_dir = str(runs_root / run_hash)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    receipts_dir = f"{run_dir}/receipts"
    Path(receipts_dir).mkdir(parents=True, exist_ok=True)

    plan_hash = run_hash  # one work item per run

    work_item = VlmPreferencePrepWorkItem(
        flavor=flavor,
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
        meta_name=meta_name,
        plan_hash=plan_hash,
        receipts_dir=receipts_dir,
        run_hash=run_hash,
        run_dir=run_dir,
        builder_command=builder_command,
    )

    Path(f"{run_dir}/config.json").write_text(
        json.dumps(
            {
                "run_hash": run_hash,
                "flavor": flavor,
                "raw_dir": str(raw_dir),
                "output_dir": str(output_dir),
                "meta_name": meta_name,
                "builder_command": builder_command,
            },
            indent=2,
        )
    )

    context = RlOmniRunContext(
        run_hash=run_hash,
        run_dir=run_dir,
        output_dir=str(output_dir),
        receipts_dir=receipts_dir,
        flavor=flavor,
        meta_name=meta_name,
    )
    return [work_item], context


# =============================================================================
# Driver: Finalize
# =============================================================================


def finalize_rl_omni_run(context: RlOmniRunContext) -> RlOmniPrepResult:
    """Read the lone receipt, materialize the result for artifact construction."""
    fs, _ = get_filesystem(context.output_dir)
    receipt_path = f"{context.receipts_dir.rstrip('/')}/shard_{0:06d}.json"

    if not fs.exists(receipt_path):
        raise FileNotFoundError(
            f"rl_omni run {context.run_hash} finished but receipt not found "
            f"at {receipt_path}. The pipeline may have failed silently."
        )

    receipt = read_json(fs, receipt_path)
    status = receipt.get("status")
    if status != "completed":
        err_msg = receipt.get("error_message", "<unknown>")
        raise RuntimeError(
            f"rl_omni run {context.run_hash} did not complete (status={status}). "
            f"Receipt error: {err_msg}"
        )

    files = receipt.get("files", {}) or {}
    stats = receipt.get("stats", {}) or {}

    return RlOmniPrepResult(
        run_hash=context.run_hash,
        run_dir=context.run_dir,
        output_dir=Path(context.output_dir),
        flavor=context.flavor,
        files=files,
        stats=stats,
    )


# =============================================================================
# Convenience Entry Point
# =============================================================================


def run_rl_omni_pipeline(
    *,
    flavor: Literal["tiny", "mpo"],
    raw_dir: Path,
    output_dir: Path,
    runs_root: Path,
    meta_name: str = "meta_public.json",
    builder_command: str | None = None,
    force: bool = False,
    stage_config: VlmPreferencePrepStageConfig | None = None,
    observability: ObservabilityConfig | None = None,
    execution_mode: ExecutionModeRequest = "auto",
    source_uri: str | None = None,
) -> RlOmniPrepResult:
    """Convenience wrapper: setup → run pipeline → finalize.

    For full control over the pipeline (e.g. attaching extra hooks),
    call ``setup_rl_omni_run`` and ``finalize_rl_omni_run`` directly with
    explicit ``PipelineSpec`` construction in your driver script.
    """
    stage_cfg = stage_config or VlmPreferencePrepStageConfig()
    observability_cfg = observability or ObservabilityConfig()

    work_items, context = setup_rl_omni_run(
        flavor=flavor,
        raw_dir=raw_dir,
        output_dir=output_dir,
        runs_root=runs_root,
        meta_name=meta_name,
        builder_command=builder_command,
        force=force,
        source_uri=source_uri,
    )

    pipeline_ctx = PipelineContext(
        output_root=str(output_dir),
        run_hash=context.run_hash,
        run_dir=context.run_dir,
        config_hash=None,
        observability=observability_cfg,
        hf_env=detect_hf_env_vars(),
    )

    stage_specs = [
        pipelines_v1.StageSpec(
            VlmPreferencePrepStage(stage_cfg, pipeline_ctx),
            num_workers=1,
        ),
    ]
    spec = pipelines_v1.PipelineSpec(
        input_data=work_items,
        stages=stage_specs,
        config=pipelines_v1.PipelineConfig(
            execution_mode=resolve_execution_mode(stage_specs, execution_mode),
            logging_interval_s=observability_cfg.pipeline_logging_interval_s,
        ),
    )

    hook = make_wandb_stats_hook(
        observability=observability_cfg,
        pipeline_kind=f"rl_omni-{flavor}",
        run_hash=context.run_hash,
        run_dir=context.run_dir,
        dataset_names=[f"mmpr_{flavor}"],
        dataset_num_shards={f"mmpr_{flavor}": len(work_items)},
    )
    if hook:
        with hook:
            pipelines_v1.run_pipeline(spec)
    else:
        pipelines_v1.run_pipeline(spec)

    return finalize_rl_omni_run(context)


def clear_output_dir(output_dir: Path) -> None:
    """Remove the published MMPR cache before a forced rebuild.

    Only intended for ``force=True`` callers. Receipts are kept under
    ``runs_root`` (set by ``setup_rl_omni_run``) which lives outside
    ``output_dir`` by contract — this call will not drop them.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)


__all__ = [
    "RlOmniPrepResult",
    "RlOmniRunContext",
    "clear_output_dir",
    "finalize_rl_omni_run",
    "run_rl_omni_pipeline",
    "setup_rl_omni_run",
]
