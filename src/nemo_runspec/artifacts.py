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

"""Artifact tracking setup for training and data prep scripts.

Provides two entry points:

1. ``setup_artifact_tracking`` — parses config, initialises backends, resolves
   artifact references.  Returns a result object used to decide which
   stage-specific monkey-patches to apply.

2. ``log_artifact`` — saves an artifact to **all** active backends.
   Replaces ``artifact.save()`` in data-prep scripts where the full
   ``nemo_runspec`` API is available at runtime (code packager).

Example (train script — monkey-patches applied by caller)::

    from nemo_runspec.artifacts import setup_artifact_tracking

    tracking = setup_artifact_tracking(config, artifacts_key="run")

    if tracking.wandb:
        patch_wandb_local_file_handler_skip_digest_verification()

    if tracking.manifest and tracking.wandb:
        patch_checkpoint_logging_both()
    elif tracking.wandb:
        patch_wandb_checkpoint_logging()
    elif tracking.manifest:
        patch_manifest_checkpoint_logging()

    if tracking.wandb:
        patch_wandb_init_for_lineage(
            artifact_qualified_names=tracking.qualified_names,
            tags=["pretrain"],
        )

Example (data-prep script — full API available)::

    from nemo_runspec.artifacts import setup_artifact_tracking, log_artifact

    tracking = setup_artifact_tracking(config)

    if tracking.wandb:
        init_wandb_from_env()          # creates wandb run for metrics

    # ... run pipeline ...

    log_artifact(artifact, tracking)   # logs to manifest + wandb
    wandb_kit.finish_run(exit_code=0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from nemotron.kit.artifacts.base import Artifact
    from nemo_runspec.manifest_tracker import ManifestTracker

logger = logging.getLogger(__name__)


@dataclass
class ArtifactTrackingResult:
    """Result of artifact tracking setup.

    Attributes:
        manifest: Whether manifest-based tracking is active.
        wandb: Whether W&B tracking is active.
        qualified_names: W&B artifact qualified names for lineage registration.
            Empty when wandb is not active.
    """

    manifest: bool = False
    wandb: bool = False
    qualified_names: list[str] = field(default_factory=list)

    # Private: tracker instance for use by log_artifact().
    # Not part of the public API — callers should not access this directly.
    _manifest_tracker: ManifestTracker | None = field(default=None, repr=False)


def setup_artifact_tracking(
    config: DictConfig,
    *,
    artifacts_key: str = "run",
) -> ArtifactTrackingResult:
    """Initialize artifact tracking from config.

    Reads the top-level ``artifacts:`` section and sets up the appropriate
    backends.  Handles:

    1. Parsing ``config.artifacts`` to determine active backends.
    2. Initializing ``ManifestTracker`` via ``kit.init()`` when manifest is enabled.
    3. Clearing the artifact cache when W&B is active.
    4. Resolving ``${art:...}`` artifact references from the config.

    The caller is responsible for applying monkey-patches (checkpoint logging,
    wandb bug workarounds) based on the returned flags — those are
    stage-specific and intentionally not hidden here.

    Args:
        config: Full training config (OmegaConf DictConfig).
        artifacts_key: Config key containing artifact references (default: ``"run"``).

    Returns:
        ArtifactTrackingResult with flags and resolved artifact names.

    Example config::

        artifacts:
          manifest:
            root: /lustre/artifacts
          wandb: true               # credentials from [wandb] in env.toml

        run:
          data: nano3-sft-data:latest
          model: nano3-pretrain-model:latest
    """
    from nemo_runspec.config.resolvers import clear_artifact_cache, register_resolvers_from_config

    # -------------------------------------------------------------------------
    # 1. Parse artifacts config
    # -------------------------------------------------------------------------
    artifacts_cfg: dict[str, Any] = OmegaConf.to_container(
        config.get("artifacts", OmegaConf.create()), resolve=True
    ) or {}

    manifest_cfg = artifacts_cfg.get("manifest")
    use_manifest = (
        isinstance(manifest_cfg, dict) and bool(manifest_cfg.get("root"))
    )
    # wandb: true (bool) or wandb: {project: ...} (dict for override)
    use_wandb = bool(artifacts_cfg.get("wandb"))

    result = ArtifactTrackingResult(manifest=use_manifest, wandb=use_wandb)

    logger.info(
        f"[ARTIFACT] Tracking setup: manifest={use_manifest}, wandb={use_wandb}"
    )

    # -------------------------------------------------------------------------
    # 2. Initialize backends
    # -------------------------------------------------------------------------
    if use_manifest:
        from nemo_runspec.manifest_tracker import ManifestTracker
        from nemo_runspec.artifact_registry import ArtifactRegistry, set_artifact_registry
        from nemotron.kit.trackers import set_lineage_tracker

        manifest_root = artifacts_cfg["manifest"]["root"]

        # Initialize registry + tracker directly (no nemotron.kit dependency)
        registry = ArtifactRegistry(backend="fsspec", root=manifest_root)
        set_artifact_registry(registry)

        tracker = ManifestTracker(root=str(manifest_root))
        set_lineage_tracker(tracker)

        logger.info(f"[ARTIFACT] ManifestTracker initialized: root={manifest_root}")

        # Store reference so log_artifact() can use it even if the
        # global lineage tracker is later overwritten by WandbTracker.
        result._manifest_tracker = tracker

    if use_wandb:
        clear_artifact_cache()

        # Patch wandb digest verification before resolving artifacts.
        # Local file references become stale when data prep is re-run;
        # the stored checksum no longer matches the regenerated file.
        try:
            from nemotron.kit.wandb_kit import (
                patch_wandb_local_file_handler_skip_digest_verification,
            )
            patch_wandb_local_file_handler_skip_digest_verification()
        except Exception:
            pass  # wandb may not be installed in all environments

    # -------------------------------------------------------------------------
    # 3. Resolve artifact references
    # -------------------------------------------------------------------------
    if use_wandb:
        # W&B mode: resolve via wandb API (downloads artifacts, returns qualified names)
        result.qualified_names = register_resolvers_from_config(
            config,
            artifacts_key=artifacts_key,
            mode="pre_init",
            pre_init_patch_http_digest=False,
        )
    elif use_manifest:
        # Manifest mode: resolve from local/fsspec filesystem
        register_resolvers_from_config(
            config,
            artifacts_key=artifacts_key,
            mode="local",
        )
    else:
        # No tracking: still register the resolver (returns empty for ${art:...})
        register_resolvers_from_config(
            config,
            artifacts_key=artifacts_key,
            mode="local",
        )

    return result


def log_artifact(
    artifact: Artifact,
    tracking: ArtifactTrackingResult,
    *,
    name: str | None = None,
) -> None:
    """Save artifact to all active tracking backends.

    This is the data-prep counterpart to the monkey-patches used in train
    scripts.  It calls ``artifact.save()`` (which uses whichever lineage
    tracker is currently set as the global), then additionally writes to
    any other active backend.

    Typical flow when **both** manifest and wandb are active:

    1. ``setup_artifact_tracking()`` sets ManifestTracker as lineage tracker.
    2. ``init_wandb_from_env()`` overwrites it with WandbTracker.
    3. ``artifact.save()`` logs via WandbTracker (the current global).
    4. This function additionally calls ManifestTracker (stored in *tracking*).

    When only one backend is active, ``artifact.save()`` handles everything
    and this function is effectively a pass-through.

    Args:
        artifact: The artifact to save and log.
        tracking: Result from ``setup_artifact_tracking()``.
        name: Optional name override for the artifact in registries.
    """
    # artifact.save() writes metadata.json locally, then logs via the
    # global lineage tracker (WandbTracker when wandb is active,
    # ManifestTracker when manifest-only, no-op otherwise).
    artifact.save(name=name)

    # When both backends are active, artifact.save() went through
    # WandbTracker (since init_wandb_from_env sets it as global).
    # We additionally write to ManifestTracker here.
    if tracking.manifest and tracking.wandb and tracking._manifest_tracker:
        try:
            artifact_name = artifact._derive_artifact_name(name)
            used = getattr(artifact, "_used_artifacts", [])
            result = tracking._manifest_tracker.log_artifact(artifact, artifact_name, used)
            logger.info(f"[ARTIFACT] Manifest logged: {artifact_name}")
            # Store manifest info for display by print_step_complete
            artifact._manifest_path = (
                f"{tracking._manifest_tracker._root}/{artifact_name}/"
                f"v{result['artifact_id'].split(':v')[-1]}"
            )
        except Exception:
            logger.warning(
                f"[ARTIFACT] Failed to write manifest for {name or 'unknown'}",
                exc_info=True,
            )
    elif tracking.manifest and not tracking.wandb:
        # Manifest-only: artifact.save() already went through ManifestTracker.
        # Extract manifest path from the tracking result stored on artifact.
        if hasattr(artifact, "tracking") and artifact.tracking and artifact.tracking.artifact_id:
            aid = artifact.tracking.artifact_id  # e.g. "super3-pretrain-data-tiny:v1"
            art_name, version = aid.rsplit(":v", 1)
            artifact._manifest_path = (
                f"{tracking._manifest_tracker._root}/{art_name}/v{version}"
            )
