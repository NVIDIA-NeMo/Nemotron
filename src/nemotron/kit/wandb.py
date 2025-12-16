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

# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Weights & Biases configuration for experiment tracking and artifact storage.

This module provides a WandbConfig dataclass that can be passed via CLI to enable
W&B artifact tracking. When configured, it automatically initializes the kit
wandb backend.

Example:
    >>> from nemotron.kit.wandb import WandbConfig, init_wandb_if_configured
    >>>
    >>> # In your config dataclass
    >>> @dataclass
    ... class MyConfig:
    ...     wandb: WandbConfig | None = None
    >>>
    >>> # In your main function
    >>> def main(cfg: MyConfig):
    ...     init_wandb_if_configured(cfg.wandb)
    ...     # Now kit.init() has been called with wandb backend
    ...     artifact.save(name="my-artifact")  # Will track in W&B
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    """Weights & Biases configuration for experiment tracking and artifact storage.

    When project is set, enables W&B artifact tracking. All fields are optional
    to support both tracked and untracked runs.

    Example CLI usage:
        nemotron nano3 data prep pretrain --wandb.project my-project --wandb.entity my-team
    """

    project: str | None = None
    """W&B project name (required to enable tracking)"""

    entity: str | None = None
    """W&B entity/team name"""

    run_name: str | None = None
    """W&B run name (auto-generated if not specified)"""

    tags: tuple[str, ...] = ()
    """Tags for filtering runs"""

    notes: str | None = None
    """Notes/description for the run"""

    @property
    def enabled(self) -> bool:
        """Returns True if wandb is configured (project is set)."""
        return self.project is not None


def init_wandb_if_configured(
    wandb_config: WandbConfig | None,
    job_type: str = "data-prep",
    tags: list[str] | None = None,
) -> None:
    """Initialize kit with wandb backend if WandbConfig is provided and enabled.

    This should be called at the start of command handlers to enable artifact tracking.
    If wandb_config is None or project is not set, this is a no-op.

    Args:
        wandb_config: WandbConfig instance or None
        job_type: W&B job type for categorizing runs (default: "data-prep")
        tags: Additional tags to add to the run (merged with config tags)

    Example:
        >>> def main(cfg: MyConfig):
        ...     init_wandb_if_configured(cfg.wandb, job_type="training", tags=["pretrain"])
        ...     # Artifacts will now be tracked in W&B
    """
    if wandb_config is None or not wandb_config.enabled:
        return

    import nemotron.kit as kit

    # Initialize kit with wandb backend (enables artifact tracking)
    kit.init(
        backend="wandb",
        wandb_project=wandb_config.project,
        wandb_entity=wandb_config.entity,
    )

    # Initialize wandb run if not already active
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is required for W&B tracking. Install with: pip install wandb")

    if wandb.run is None:
        # Merge config tags with additional tags
        all_tags: list[str] = []
        if wandb_config.tags:
            all_tags.extend(wandb_config.tags)
        if tags:
            all_tags.extend(tags)

        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=wandb_config.run_name,
            tags=all_tags if all_tags else None,
            notes=wandb_config.notes,
            job_type=job_type,
        )


def add_wandb_tags(tags: list[str]) -> None:
    """Add tags to the active wandb run if one exists.

    This can be called after wandb is initialized to add stage-specific tags.
    Tags are merged with any existing tags on the run.

    Args:
        tags: List of tags to add to the run

    Example:
        >>> add_wandb_tags(["data-prep", "pretrain"])
    """
    try:
        import wandb

        if wandb.run is not None and tags:
            # Get existing tags and merge
            existing_tags = list(wandb.run.tags) if wandb.run.tags else []
            new_tags = list(set(existing_tags + tags))  # Deduplicate
            wandb.run.tags = new_tags
    except ImportError:
        pass
    except Exception:
        pass  # Don't fail if tags can't be added


def finish_wandb(exit_code: int = 0) -> None:
    """Finish the active wandb run if one exists.

    This should be called at the end of a successful run to properly close
    the wandb session. Without this, runs will appear as "crashed" in the
    W&B dashboard.

    Args:
        exit_code: Exit code to report. 0 for success, non-zero for failure.

    Example:
        >>> try:
        ...     # Do work
        ...     artifact.save()
        ...     finish_wandb(exit_code=0)
        ... except Exception:
        ...     finish_wandb(exit_code=1)
        ...     raise
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
    except ImportError:
        pass


_HTTP_HANDLER_PATCHED = False
_LOCAL_FILE_HANDLER_PATCHED = False
_WANDB_INIT_PATCHED = False
_LINEAGE_REGISTERED = False
_RUNID_PATCHED = False
_CHECKPOINT_LOGGING_PATCHED = False
_NEMO_RL_CHECKPOINT_LOGGING_PATCHED = False
_PENDING_ARTIFACT_QUALIFIED_NAMES: set[str] = set()
_PENDING_TAGS: set[str] = set()


def patch_wandb_http_handler_skip_digest_verification() -> None:
    """Best-effort patch to skip digest verification for HTTP reference artifacts.

    Some reference artifact backends (e.g. HuggingFace URLs) can return varying ETags
    over time, causing W&B to reject downloads due to digest mismatch.
    """
    global _HTTP_HANDLER_PATCHED
    if _HTTP_HANDLER_PATCHED:
        return

    try:
        from wandb.sdk.artifacts.storage_handlers import http_handler

        original_load_path = http_handler.HTTPHandler.load_path

        def patched_load_path(self, manifest_entry, local: bool = False):
            import os
            import tempfile

            import requests

            url = getattr(manifest_entry, "ref", None)
            if url is None:
                return original_load_path(self, manifest_entry, local=local)

            path = getattr(manifest_entry, "path", None)
            if local or path is None:
                fd, tmp_path = tempfile.mkstemp()
                os.close(fd)
                path = tmp_path

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return path

        http_handler.HTTPHandler.load_path = patched_load_path
        _HTTP_HANDLER_PATCHED = True
        logger.debug("Patched wandb HTTP handler to skip digest verification")
    except Exception as e:
        logger.warning(f"Failed to patch wandb HTTP handler: {e}")


def patch_wandb_local_file_handler_skip_digest_verification() -> None:
    """Best-effort patch to skip digest verification for local file reference artifacts.

    Local file references can become stale if data prep is re-run with different
    parameters, causing W&B to reject artifact downloads due to digest mismatch.
    """
    global _LOCAL_FILE_HANDLER_PATCHED
    if _LOCAL_FILE_HANDLER_PATCHED:
        return

    try:
        from wandb.sdk.artifacts.storage_handlers import local_file_handler

        original_load_path = local_file_handler.LocalFileHandler.load_path

        def patched_load_path(self, manifest_entry, local: bool = False):
            # Skip digest verification - just return the local path
            path = getattr(manifest_entry, "ref", None)
            if path and path.startswith("file://"):
                path = path[7:]  # Remove "file://" prefix
            if path is None:
                return original_load_path(self, manifest_entry, local=local)
            return path

        local_file_handler.LocalFileHandler.load_path = patched_load_path
        _LOCAL_FILE_HANDLER_PATCHED = True
        logger.debug("Patched wandb LocalFileHandler to skip digest verification")
    except Exception as e:
        logger.warning(f"Failed to patch wandb LocalFileHandler: {e}")


def patch_wandb_init_for_lineage(
    *,
    artifact_qualified_names: list[str],
    tags: list[str] | None = None,
) -> None:
    """Patch `wandb.init()` so that, once a run is active, lineage is registered.

    Intended for setups where another library owns `wandb.init()` (e.g. Megatron-Bridge)
    but this project resolves artifacts before that init happens.
    """
    global _WANDB_INIT_PATCHED

    if artifact_qualified_names:
        _PENDING_ARTIFACT_QUALIFIED_NAMES.update(map(str, artifact_qualified_names))
    if tags:
        _PENDING_TAGS.update(map(str, tags))

    if _WANDB_INIT_PATCHED:
        return

    import wandb

    original_init = wandb.init

    def patched_init(*args, **kwargs):
        result = original_init(*args, **kwargs)
        _register_lineage_if_possible()
        return result

    wandb.init = patched_init
    _WANDB_INIT_PATCHED = True
    logger.debug("Patched wandb.init for lineage registration")


def _register_lineage_if_possible() -> None:
    global _LINEAGE_REGISTERED
    if _LINEAGE_REGISTERED:
        return

    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    if _PENDING_TAGS:
        add_wandb_tags(sorted(_PENDING_TAGS))

    for qname in sorted(_PENDING_ARTIFACT_QUALIFIED_NAMES):
        try:
            wandb.run.use_artifact(qname)
        except Exception as e:
            logger.warning(f"Failed to register artifact lineage for {qname}: {e}")

    _LINEAGE_REGISTERED = True


def patch_wandb_runid_for_seeded_random() -> None:
    """Patch wandb's generate_fast_id to use an independent random instance.

    This fixes the "Invalid Client ID digest" error that occurs when random.seed()
    is called before artifact creation (common in ML training for reproducibility).
    See: https://github.com/wandb/wandb/pull/11039
    """
    global _RUNID_PATCHED
    if _RUNID_PATCHED:
        return

    import os
    import random as random_module

    from wandb.sdk.artifacts import artifact as artifact_module
    from wandb.sdk.lib import runid

    # Create an independent random instance seeded from OS entropy
    # This ensures it's not affected by any global random.seed() calls
    _independent_random = random_module.Random()
    _independent_random.seed(os.urandom(32))  # Seed from OS entropy

    id_chars = "abcdefghijklmnopqrstuvwxyz0123456789"

    def patched_generate_fast_id(length: int = 8) -> str:
        return "".join(_independent_random.choices(id_chars, k=length))

    # Patch both the source module AND the artifact module's imported reference
    runid.generate_fast_id = patched_generate_fast_id
    artifact_module.generate_fast_id = patched_generate_fast_id
    _RUNID_PATCHED = True
    logger.info("[WANDB] Patched generate_fast_id in both runid and artifact modules")


def _resolve_to_lustre_path(path: str) -> str:
    """Resolve a container path to the actual Lustre path.

    When running in a container, /nemo_run is a bind mount that maps to the actual
    Lustre path. This function resolves the path using:
    1. NEMO_RUN_DIR environment variable (if set)
    2. Reading /proc/mounts to find bind mount source

    Args:
        path: Path string, possibly starting with /nemo_run/

    Returns:
        Path with /nemo_run/ replaced by actual Lustre path
    """
    import os
    from pathlib import Path as PathLib

    resolved = str(PathLib(path).resolve())

    # If path doesn't start with /nemo_run, nothing to do
    if not resolved.startswith("/nemo_run"):
        return resolved

    # Method 1: Use NEMO_RUN_DIR environment variable
    nemo_run_dir = os.environ.get("NEMO_RUN_DIR")
    if nemo_run_dir and nemo_run_dir != "/nemo_run":
        logger.info(f"[WANDB] Using NEMO_RUN_DIR={nemo_run_dir} for path resolution")
        if resolved.startswith("/nemo_run/"):
            return resolved.replace("/nemo_run/", f"{nemo_run_dir}/", 1)
        elif resolved == "/nemo_run":
            return nemo_run_dir

    # Method 2: Try to find /nemo_run bind mount source from /proc/mounts
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    source, target = parts[0], parts[1]
                    if target == "/nemo_run" and source.startswith("/"):
                        logger.info(f"[WANDB] Found /nemo_run bind mount: {source}")
                        if resolved.startswith("/nemo_run/"):
                            return resolved.replace("/nemo_run/", f"{source}/", 1)
                        elif resolved == "/nemo_run":
                            return source
    except (OSError, IOError) as e:
        logger.warning(f"[WANDB] Could not read /proc/mounts: {e}")

    logger.warning(
        f"[WANDB] Could not resolve /nemo_run to Lustre path. "
        f"Set NEMO_RUN_DIR environment variable to the actual Lustre path."
    )
    return resolved


def patch_wandb_checkpoint_logging() -> None:
    """Monkey patch on_save_checkpoint_success to use add_reference like Megatron-Bridge.

    The original Megatron-Bridge code uses add_reference(checksum=False) but doesn't
    call wait(), so artifacts don't show up in real-time. This patch adds wait() to
    ensure artifacts are committed immediately.

    IMPORTANT: Uses _resolve_to_lustre_path to convert container mount paths (/nemo_run/)
    to actual Lustre paths, so artifacts can be accessed from other jobs.
    """
    from pathlib import Path
    from typing import Any

    global _CHECKPOINT_LOGGING_PATCHED
    if _CHECKPOINT_LOGGING_PATCHED:
        return

    from megatron.bridge.training.utils import wandb_utils

    def patched_on_save_checkpoint_success(
        checkpoint_path: str,
        save_dir: str,
        iteration: int,
        wandb_writer: Any | None,
    ) -> None:
        if not wandb_writer or not wandb_writer.run:
            return

        try:
            # Resolve the checkpoint path to absolute
            checkpoint_path_resolved = Path(checkpoint_path).resolve()

            # Verify checkpoint directory actually exists before logging
            if not checkpoint_path_resolved.exists():
                logger.warning(
                    f"[WANDB] Checkpoint path does not exist, skipping artifact: {checkpoint_path_resolved}"
                )
                return

            # Store the save_dir (parent of checkpoint) as absolute_path
            # Megatron-Bridge expects pretrained_checkpoint to be the save directory,
            # and it constructs the full checkpoint path by appending iter_XXXXXX
            save_dir_resolved = _resolve_to_lustre_path(str(Path(save_dir).resolve()))
            absolute_path = save_dir_resolved

            artifact_name, artifact_version = wandb_utils._get_artifact_name_and_version(
                Path(save_dir), Path(checkpoint_path)
            )

            # Create artifact with file reference
            # Use the container path that actually exists for add_reference validation,
            # but store the absolute path in metadata for cross-job access
            metadata = {"iteration": iteration, "absolute_path": absolute_path}
            artifact = wandb_writer.Artifact(artifact_name, type="model", metadata=metadata)

            # Use the resolved container path (which exists) for add_reference
            artifact.add_reference(f"file://{str(checkpoint_path_resolved)}", checksum=False)

            # Log artifact with alias
            logged = wandb_writer.run.log_artifact(artifact, aliases=[artifact_version])

            # Wait for commit (this is what was missing in Megatron-Bridge)
            logged.wait()
            logger.info(f"[WANDB] Artifact committed: {artifact_name}:{artifact_version}")

            # Write tracker file for later reference
            wandb_tracker_filename = wandb_utils._get_wandb_artifact_tracker_filename(save_dir)
            wandb_tracker_filename.write_text(
                f"{wandb_writer.run.entity}/{wandb_writer.run.project}"
            )
        except Exception as e:
            logger.error(f"[WANDB] Failed to log checkpoint artifact: {e}")

    wandb_utils.on_save_checkpoint_success = patched_on_save_checkpoint_success
    _CHECKPOINT_LOGGING_PATCHED = True
    logger.info("[WANDB] Patched checkpoint logging to add wait() call")


def patch_nemo_rl_checkpoint_logging() -> None:
    """Monkey patch NeMo-RL's CheckpointManager to log checkpoint artifacts to W&B.

    NeMo-RL uses a different checkpoint mechanism than Megatron-Bridge. This patch
    wraps CheckpointManager.finalize_checkpoint() to log the checkpoint as a W&B
    artifact after the checkpoint is finalized.

    The artifact is created with:
    - type: "model"
    - name: "rl" (to match pretrain/sft naming convention)
    - metadata: step number extracted from checkpoint path
    - file reference: local path to checkpoint directory
    """
    from pathlib import Path
    from typing import Any

    global _NEMO_RL_CHECKPOINT_LOGGING_PATCHED
    if _NEMO_RL_CHECKPOINT_LOGGING_PATCHED:
        return

    try:
        from nemo_rl.utils.checkpoint import CheckpointManager
    except ImportError:
        logger.warning("[WANDB] nemo_rl not installed, skipping checkpoint logging patch")
        return

    original_finalize_checkpoint = CheckpointManager.finalize_checkpoint

    def patched_finalize_checkpoint(self, checkpoint_path: Any) -> None:
        """Finalize checkpoint and log to W&B as artifact."""
        # Call original finalize first
        original_finalize_checkpoint(self, checkpoint_path)

        # Now log to wandb
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        try:
            checkpoint_path = Path(checkpoint_path)
            # After finalize, tmp_step_X becomes step_X
            step_str = checkpoint_path.name.split("_")[-1]
            step = int(step_str)

            # Final checkpoint path after rename
            final_checkpoint_path = checkpoint_path.parent / f"step_{step}"
            # Resolve to absolute container path (for add_reference validation)
            checkpoint_path_resolved = str(final_checkpoint_path.resolve())
            # Get the absolute shared filesystem path for cross-job access
            absolute_path = _resolve_to_lustre_path(str(final_checkpoint_path))

            # Create artifact with naming convention matching pretrain/sft
            artifact_name = "rl"
            artifact_version = f"step_{step}"

            # Store absolute_path in metadata for cross-job access
            metadata = {"step": step, "absolute_path": absolute_path}
            artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
            # Use the resolved container path (which exists) for add_reference
            artifact.add_reference(f"file://{checkpoint_path_resolved}", checksum=False)

            # Log artifact with alias
            logged = wandb.run.log_artifact(artifact, aliases=[artifact_version, "latest"])

            # Wait for commit to ensure artifact is visible immediately
            logged.wait()
            logger.info(
                f"[WANDB] RL checkpoint artifact committed: {artifact_name}:{artifact_version}"
            )

        except Exception as e:
            logger.error(f"[WANDB] Failed to log RL checkpoint artifact: {e}")

    CheckpointManager.finalize_checkpoint = patched_finalize_checkpoint
    _NEMO_RL_CHECKPOINT_LOGGING_PATCHED = True
    logger.info("[WANDB] Patched NeMo-RL CheckpointManager for artifact logging")
