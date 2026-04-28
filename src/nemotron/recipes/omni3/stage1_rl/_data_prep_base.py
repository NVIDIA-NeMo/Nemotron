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

"""Shared base for Omni RL data preparation.

Dispatches across the three Omni RL data-prep variants:

- ``stage="mpo"`` — full MMPR for Mixed-Preference Optimization. Currently
  routes through ``recipes.rl_omni`` with ``flavor="mpo"``, which shells
  out to the upstream ``prepare_public_mmpr_for_mpo.py`` builder command.
  TODO: replace ``builder_command`` with vendored logic when the upstream
  script can be brought in-tree.

- ``stage="text"`` — text-only RL. Reuses Nano3's existing
  ``recipes.rl.run_rl_resolve_pipeline`` because the source dataset
  (``nvidia/Nemotron-3-Nano-RL-Training-Blend``), placeholder semantics
  (DAPO + Skywork via ``HFPlaceholderResolver``), and output schema
  (``responses_create_params``-shaped JSONL) are identical to Nano3's
  text-only RL stage.

- ``stage="vision"`` — MMPR-Tiny for vision RL. Routes through
  ``recipes.rl_omni`` with ``flavor="tiny"``; the prep logic is vendored
  in ``stages.vlm_preference_prep`` (single-worker, single-stage Xenna pipeline).

Why the asymmetry between flavors? Text and Vision get vendored
implementations because the prep logic is small, owned, testable, and
already in-tree. MPO keeps the subprocess escape hatch because
``prepare_public_mmpr_for_mpo.py`` lives in the upstream NeMo-RL
``nano-v3-omni-recipes`` branch and is not yet available for clean
vendoring. The MPO path still gets the framework's run-hash caching,
receipts, and W&B lineage — only the work-doing core is a subprocess.
This is a deliberate transitional state, not a permanent design.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from nemo_runspec.artifacts import ArtifactTrackingResult, log_artifact, setup_artifact_tracking
from nemotron.data_prep.blend import DataBlend
from nemotron.data_prep.config import DatasetConfig
from nemotron.data_prep.recipes.rl import run_rl_resolve_pipeline
from nemotron.data_prep.recipes.rl_omni import (
    clear_output_dir,
    run_rl_omni_pipeline,
)
from nemotron.data_prep.utils.discovery import get_dataset_metadata
from nemotron.data_prep.utils.hf_placeholder import HFPlaceholderResolver, NANO3_TARGET_DATASETS
from nemotron.kit import Artifact, SplitJsonlDataArtifact, print_step_complete, wandb_kit
from nemotron.kit.trackers import InputDatasetInfo
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    init_wandb_from_env,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

logger = logging.getLogger(__name__)

_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class Omni3RLDataArtifact(Artifact):
    """Artifact for single-file or directory-based Omni RL data outputs.

    Used for MPO and Vision flavors where the trainer reads a directory
    layout (parquet + image cache, or annotations + images + meta JSON)
    rather than a JSONL manifest.
    """

    stage: str
    dataset_name: str
    source_uri: str | None = None

    def _get_output_dir(self) -> Path:
        return self.path.parent if self.path.is_file() or self.path.suffix else self.path

    def get_wandb_files(self) -> list[tuple[str, str]]:
        files: list[tuple[str, str]] = []
        if self.path.is_file() and self.path.exists():
            files.append((str(self.path), self.path.name))
        metadata_path = self._get_output_dir() / "metadata.json"
        if metadata_path.exists():
            files.append((str(metadata_path), "metadata.json"))
        return files

    def get_wandb_references(self) -> list[tuple[str, str]]:
        target = self.path.parent if self.path.is_file() else self.path
        return [(f"file://{target.resolve()}", "output")]

    def get_input_uris(self) -> list[str]:
        return [self.source_uri] if self.source_uri else []


@dataclass
class Omni3RLDataPrepConfig:
    """Configuration for Omni RL data preparation.

    A single config dataclass spans all three flavors because the CLI
    dispatcher (``omni3 data prep rl -c {mpo|text|vision}``) selects a
    YAML and we'd rather have stage-specific fields be optional than
    split into three configs that look 80% identical.

    Common fields:
        stage: One of ``"mpo"``, ``"text"``, ``"vision"``.
        dataset_name: Logical dataset identifier for artifact lineage.
        source_uri: Upstream HF dataset URI for lineage tracking.
        output_dir: Where the prepared cache / JSONL lands.
        force: Salt the run hash with the current time to bypass cache.
        sample: Optional row cap for quick tests (text flavor only).
        execution_mode: Xenna execution mode hint (text flavor only).

    Flavor-specific fields:
        input_dir: Local raw-download directory (mpo, vision).
        meta_name: Metadata filename the trainer reads (mpo).
        builder_command: Subprocess command for the upstream prep
            script (mpo only — TODO: drop once vendored).
        blend_path: JSON blend specification (text only).
    """

    stage: str = "vision"
    dataset_name: str = "mmpr_tiny"
    source_uri: str | None = None
    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "stage1_rl/data_prep")
    force: bool = False

    # Common to MPO + Vision (raw HF download dir)
    input_dir: Path = Path("data")

    # Vision/MPO: name of the meta file the trainer reads
    meta_name: str = "meta_public.json"

    # MPO escape hatch (subprocess shim until upstream script is vendored).
    # Format string supports {input_dir}, {output_dir}, {meta_name}.
    builder_command: str | None = None

    # Text-only: data blend JSON path. Identical schema to Nano3's
    # stage2_rl/config/data_prep/data_blend_raw.json.
    blend_path: Path | None = None

    # Text-only: optional row cap for quick tests.
    sample: int | None = None

    # Text-only: Xenna execution mode hint (auto/streaming/batch).
    execution_mode: str = "auto"

    # Optional override for the receipts directory (mpo + vision flavors).
    # If unset, defaults to ``output_dir.parent / "runs"`` — fine for most
    # cluster layouts but exposed here so an operator debugging a stuck
    # run can redirect receipts to a writable scratch path without
    # touching the published cache.
    runs_root: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.blend_path, str):
            self.blend_path = Path(self.blend_path)
        if isinstance(self.runs_root, str):
            self.runs_root = Path(self.runs_root)


# =============================================================================
# Stage handlers
# =============================================================================


def _prepare_mpo(
    cfg: Omni3RLDataPrepConfig,
    tracking: ArtifactTrackingResult | None,
) -> Omni3RLDataArtifact:
    """MPO: full MMPR cache build via the rl_omni recipe (subprocess flavor).

    Until ``prepare_public_mmpr_for_mpo.py`` is vendored, we route the
    upstream script through ``VlmPreferencePrepStage(flavor="mpo")``. The pipeline
    gets run-hash caching, receipts, and lineage — the script itself runs
    as a subprocess inside the stage.
    """
    if not cfg.builder_command:
        raise ValueError(
            "stage='mpo' requires builder_command in the config until the "
            "upstream prepare_public_mmpr_for_mpo.py logic is vendored."
        )

    output_dir = cfg.output_dir.expanduser()
    runs_root = (
        cfg.runs_root.expanduser().resolve()
        if cfg.runs_root is not None
        else (output_dir.parent / "runs").resolve()
    )

    if cfg.force:
        clear_output_dir(output_dir)

    start_time = time.time()
    wandb_kit.add_run_tags(["data-prep", "rl", "mpo"])

    result = run_rl_omni_pipeline(
        flavor="mpo",
        raw_dir=cfg.input_dir,
        output_dir=output_dir,
        runs_root=runs_root,
        meta_name=cfg.meta_name,
        builder_command=cfg.builder_command,
        force=cfg.force,
        source_uri=cfg.source_uri,
    )

    meta_path = result.output_dir / cfg.meta_name
    artifact = Omni3RLDataArtifact(
        path=meta_path,
        stage=cfg.stage,
        dataset_name=cfg.dataset_name,
        source_uri=cfg.source_uri,
        name="omni3/rl/mpo/data",
    )
    artifact.metadata["elapsed_sec"] = time.time() - start_time
    artifact.metadata["run_hash"] = result.run_hash
    _emit(artifact, tracking)
    return artifact


def _prepare_text(
    cfg: Omni3RLDataPrepConfig,
    tracking: ArtifactTrackingResult | None,
) -> SplitJsonlDataArtifact:
    """Text RL: reuses Nano3's HF-placeholder-resolution pipeline.

    The omni3 text RL stage consumes the same dataset
    (``nvidia/Nemotron-3-Nano-RL-Training-Blend``) with the same
    placeholder format and the same output schema as Nano3's stage2_rl,
    so we share the recipe verbatim. Only the artifact name and
    output_dir default differ.
    """
    if cfg.blend_path is None:
        raise ValueError(
            "stage='text' requires blend_path pointing at a data blend JSON "
            "(see stage1_rl/config/data_prep/data_blend_raw.json)."
        )

    blend = DataBlend.load(cfg.blend_path)
    output_dir = cfg.output_dir.expanduser()
    if cfg.sample is not None:
        output_dir = output_dir / f"sample-{cfg.sample}"

    start_time = time.time()
    wandb_kit.add_run_tags(["data-prep", "rl", "text"])

    # Source-dataset lineage (same shape as Nano3 stage2_rl).
    source_datasets: list[InputDatasetInfo] = []
    seen_keys: set[str] = set()
    for dataset in blend.datasets:
        key = f"{dataset.path}|{dataset.subset or ''}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        ds_config = DatasetConfig(
            name=dataset.name,
            path=dataset.path,
            split=dataset.split,
            subset=dataset.subset,
            text_field=dataset.text_field,
        )
        hf_metadata = get_dataset_metadata(ds_config)
        source_datasets.append(
            InputDatasetInfo(
                uri=dataset.path,
                name=dataset.name,
                weight=dataset.weight,
                split=dataset.split,
                subset=dataset.subset,
                text_field=dataset.text_field,
                num_rows=hf_metadata.num_rows,
                size_bytes=hf_metadata.size_bytes,
            )
        )

    # Note: cfg.force here is forwarded into the recipe's run_hash salt so a
    # forced run produces a fresh runs/{hash} subdir. Unlike _prepare_mpo /
    # _prepare_vision we do NOT wipe `output_dir` because the JSONL recipe
    # writes per-run subdirs (Nano3's pattern) and stale runs are harmless.
    result = run_rl_resolve_pipeline(
        blend=blend,
        output_dir=output_dir,
        sample=cfg.sample,
        force=cfg.force,
        compression="none",
        num_shards_per_split=1,
        resolve_hf_placeholders=True,
        execution_mode=cfg.execution_mode,
        hf_placeholder_targets=NANO3_TARGET_DATASETS,
    )

    # Add external HF placeholder datasets (DAPO, Skywork) for lineage.
    # The resolver's PyArrow tables aren't picklable, so we re-load on the
    # driver — same pattern as Nano3 stage2_rl.
    print("Loading external HuggingFace dataset metadata for lineage tracking...")
    resolver = HFPlaceholderResolver.create(target_datasets=NANO3_TARGET_DATASETS)
    for ext_ds_info in resolver.get_loaded_datasets_info():
        source_datasets.append(
            InputDatasetInfo(
                uri=ext_ds_info["uri"],
                name=ext_ds_info["name"],
                split=ext_ds_info["split"],
                num_rows=ext_ds_info["num_rows"],
            )
        )

    elapsed = time.time() - start_time
    artifact = SplitJsonlDataArtifact(
        path=Path(result.manifest_path),
        total_sequences=result.total_records,
        elapsed_sec=elapsed,
        source_datasets=source_datasets,
        train=result.split_paths.get("train"),
        val=result.split_paths.get("val"),
        test=result.split_paths.get("test"),
    )
    artifact.name = (
        f"omni3/rl/text/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"
    )
    _emit(artifact, tracking)
    return artifact


def _prepare_vision(
    cfg: Omni3RLDataPrepConfig,
    tracking: ArtifactTrackingResult | None,
) -> Omni3RLDataArtifact:
    """Vision RL: MMPR-Tiny cache via vendored VlmPreferencePrepStage."""
    output_dir = cfg.output_dir.expanduser()
    runs_root = (
        cfg.runs_root.expanduser().resolve()
        if cfg.runs_root is not None
        else (output_dir.parent / "runs").resolve()
    )

    if cfg.force:
        clear_output_dir(output_dir)

    start_time = time.time()
    wandb_kit.add_run_tags(["data-prep", "rl", "vision"])

    result = run_rl_omni_pipeline(
        flavor="tiny",
        raw_dir=cfg.input_dir,
        output_dir=output_dir,
        runs_root=runs_root,
        meta_name=cfg.meta_name,
        builder_command=None,  # tiny flavor uses vendored logic
        force=cfg.force,
        source_uri=cfg.source_uri,
    )

    artifact = Omni3RLDataArtifact(
        path=result.output_dir,
        stage=cfg.stage,
        dataset_name=cfg.dataset_name,
        source_uri=cfg.source_uri,
        name="omni3/rl/vision/data",
    )
    artifact.metadata["elapsed_sec"] = time.time() - start_time
    artifact.metadata["run_hash"] = result.run_hash
    artifact.metadata["row_stats"] = result.stats
    _emit(artifact, tracking)
    return artifact


# =============================================================================
# Common helpers
# =============================================================================


def _emit(
    artifact: Artifact | SplitJsonlDataArtifact,
    tracking: ArtifactTrackingResult | None,
) -> None:
    if tracking is not None:
        log_artifact(artifact, tracking)
    else:
        artifact.save()
    # Mark the W&B run finished cleanly. Without this, drivers can leave
    # the run in ``running`` state until process teardown — long enough
    # for W&B's heartbeat to mark it crashed in some setups. Mirrors the
    # finalization step in nano3/stage2_rl/data_prep.py.
    wandb_kit.finish_run(exit_code=0)
    print_step_complete(data_prep=artifact)


# =============================================================================
# Dispatch + main
# =============================================================================


_STAGE_TO_HANDLER = {
    "mpo": _prepare_mpo,
    "text": _prepare_text,
    "vision": _prepare_vision,
}

# Legacy ``dataset_name``-only routing fallback. Used only when ``stage`` is
# empty/missing in the config; ``stage`` is otherwise authoritative. Kept so
# pre-existing YAMLs that set only ``dataset_name`` still resolve, but the
# explicit ``stage`` field is the supported way to dispatch.
_DATASET_NAME_TO_STAGE = {
    "mmpr": "mpo",
    "mmpr_public": "mpo",
    "nemotron_3_nano_rl_training_blend": "text",
    "text_only_rl_stage1": "text",
    "mmpr_tiny": "vision",
}


def run_data_prep(
    cfg: Omni3RLDataPrepConfig,
    tracking: ArtifactTrackingResult | None = None,
):
    """Run the selected Omni RL data-prep variant.

    ``cfg.stage`` is authoritative. ``cfg.dataset_name`` is consulted ONLY
    if ``stage`` is empty/missing (legacy fallback for YAMLs that set just
    a dataset name).
    """
    stage = (cfg.stage or "").lower()
    if not stage:
        dataset_name = (cfg.dataset_name or "").lower()
        stage = _DATASET_NAME_TO_STAGE.get(dataset_name, "")

    handler = _STAGE_TO_HANDLER.get(stage)
    if handler is None:
        raise ValueError(
            f"Unsupported omni3 RL data-prep stage={cfg.stage!r} "
            f"dataset_name={cfg.dataset_name!r}. "
            f"Supported stages: {sorted(_STAGE_TO_HANDLER)}."
        )
    return handler(cfg, tracking)


def _validate_known_keys(config) -> None:
    """Reject YAMLs that set fields the dataclass doesn't define.

    The dispatcher schema changed when omni3 RL prep moved onto
    ``nemotron.data_prep``: the old fields ``input_file``, ``meta_name``-
    for-text, ``train_file_name``, ``val_file_name``,
    ``validation_strategy``, and ``builder_command`` (for text/vision)
    are no longer recognized. ``omegaconf_to_dataclass`` typically
    silently drops unknown keys, which is the worst failure mode —
    silent misconfiguration. We surface stale keys explicitly here.

    The ``run`` and ``artifacts`` top-level blocks are consumed by the
    nemo-runspec / artifact-tracking layers respectively and are not
    expected to map onto the dataclass.
    """
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        logger.warning(
            "omegaconf unavailable; skipping omni3 RL config unknown-key validation. "
            "Stale YAML keys may be silently ignored."
        )
        return
    if not isinstance(config, DictConfig):
        return
    known = {f.name for f in fields(Omni3RLDataPrepConfig)}
    framework_keys = {"run", "artifacts"}
    raw = OmegaConf.to_container(config, resolve=False) or {}
    if not isinstance(raw, dict):
        return
    unknown = set(raw.keys()) - known - framework_keys
    if unknown:
        raise ValueError(
            "Stale or unknown keys in omni3 RL data-prep config: "
            f"{sorted(unknown)}. Supported fields: {sorted(known)}. "
            "If you're upgrading from a pre-framework YAML, see the "
            "config files under stage1_rl/config/data_prep/ for the "
            "current schema."
        )


def main(
    default_config: Path,
    cfg: Omni3RLDataPrepConfig | None = None,
):
    """Generic entry point for Omni RL data-prep wrappers."""
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=default_config)
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        tracking = setup_artifact_tracking(config)
        _validate_known_keys(config)
        cfg = omegaconf_to_dataclass(config, Omni3RLDataPrepConfig)
    else:
        tracking = None

    if tracking is None or tracking.wandb:
        init_wandb_from_env()

    return run_data_prep(cfg, tracking=tracking)
