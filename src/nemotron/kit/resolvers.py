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

"""OmegaConf custom resolvers for artifact resolution.

This module provides resolvers that can be used in config files to resolve
artifact paths at runtime, enabling W&B lineage tracking when running inside
containers.

Usage in config YAML:
    run:
      data: DataBlendsArtifact-pretrain
      model: ModelArtifact-pretrain:v5

    recipe:
      per_split_data_args_path: ${art.data.path}
      checkpoint_path: ${art.model.path}/model.pt

Usage in training script:
    from nemotron.kit.resolvers import register_resolvers

    # Register resolvers before loading config
    register_resolvers(artifacts={
        "data": "DataBlendsArtifact-pretrain",
        "model": "ModelArtifact-pretrain:v5",
    })

    # Now load config - ${art.X.path} will resolve with W&B lineage
    config = OmegaConf.load("train.yaml")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

# Global artifact registry for the resolver.
# Keys are user-facing (e.g. "data") and values are resolved artifact info.
_ARTIFACT_REGISTRY: dict[str, dict[str, Any]] = {}

# Internal cache for de-duplicating resolution work.
_ARTIFACT_CACHE: dict[str, dict[str, Any]] = {}

ResolverMode = Literal["active_run", "pre_init"]


def _parse_artifact_ref(artifact_ref: str) -> tuple[str, str | None]:
    if ":" in artifact_ref:
        name, version = artifact_ref.rsplit(":", 1)
        return name, version
    return artifact_ref, None


def _normalize_version(version: str | None) -> str:
    if version is None:
        return "latest"
    if version == "latest":
        return "latest"
    if version.startswith("v"):
        return version
    if version.isdigit():
        return f"v{version}"
    return version


def _get_distributed_info() -> tuple[int, int]:
    """Get rank and world_size from torchrun environment variables.

    Returns:
        Tuple of (rank, world_size). Defaults to (0, 1) for single-process runs.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _get_job_id() -> str:
    """Get a unique job identifier for the current run.

    Priority order:
    1. NEMO_EXPERIMENT_ID - unique per nemo-run execution (most reliable)
    2. SLURM_JOB_ID - unique per Slurm job
    3. TORCHELASTIC_RUN_ID + hostname - unique per torchrun execution

    Returns:
        Unique job identifier string.
    """
    # Prefer NEMO_EXPERIMENT_ID - set by nemo-run, unique per execution
    nemo_exp_id = os.environ.get("NEMO_EXPERIMENT_ID")
    if nemo_exp_id:
        return f"nemo_{nemo_exp_id}"

    # Fall back to Slurm job ID if available
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"slurm_{slurm_job_id}"

    # Fall back to hostname + parent PID (the torchrun launcher PID)
    # This ensures all workers in the same job get the same ID
    import socket

    hostname = socket.gethostname()
    # Use TORCHELASTIC_RUN_ID if available (set by torchrun)
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getppid()))
    return f"{hostname}_{run_id}"


def _get_marker_path(artifacts: dict[str, str]) -> Path:
    """Generate a unique marker file path based on artifact references AND job ID.

    Uses NEMO_RUN_DIR (shared filesystem) if available for multi-node jobs,
    otherwise falls back to TMPDIR or /tmp.

    The marker path now includes a job identifier to prevent stale marker files
    from previous runs being read by workers in the current run.

    Args:
        artifacts: Dict of artifact key -> artifact reference.

    Returns:
        Path to marker file on shared or local storage.
    """
    # Hash the artifacts dict to create a unique marker per config
    artifacts_str = json.dumps(sorted(artifacts.items()))
    hash_suffix = hashlib.md5(artifacts_str.encode()).hexdigest()[:8]

    # Include job ID to prevent reading stale marker files from previous runs
    job_id = _get_job_id()

    # Prefer NEMO_RUN_DIR (shared filesystem) for multi-node jobs
    # Fall back to TMPDIR or /tmp for single-node or local runs
    base_dir = os.environ.get("NEMO_RUN_DIR") or os.environ.get("TMPDIR", "/tmp")
    return Path(base_dir) / f".nemotron_artifacts_{hash_suffix}_{job_id}"


def _wait_for_artifacts(marker_path: Path, timeout: int = 600) -> dict[str, Any]:
    """Wait for rank 0 to complete artifact downloads and read results.

    Args:
        marker_path: Path to marker file written by rank 0.
        timeout: Maximum seconds to wait (default: 600 = 10 minutes).

    Returns:
        Dict with "results" and "qualified_names" from rank 0.

    Raises:
        TimeoutError: If rank 0 doesn't complete within timeout.
    """
    start = time.time()
    while not marker_path.exists():
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Timeout waiting for rank 0 to download artifacts (marker: {marker_path})"
            )
        time.sleep(1.0)

    # Read the artifact data written by rank 0
    data = json.loads(marker_path.read_text())
    return data


def _resolve_artifact_active_run(name: str, version: str | None = None) -> dict[str, Any]:
    """Resolve an artifact and cache the result.

    Args:
        name: Artifact name (e.g., "DataBlendsArtifact-pretrain")
        version: Optional version (e.g., "v5" or "5"). If None, uses latest.

    Returns:
        Dict with artifact info: {"path": str, "version": int, "name": str}
    """
    # Build cache key
    cache_key = f"active:{name}:{_normalize_version(version)}"

    if cache_key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[cache_key]

    import wandb

    artifact_ref = f"{name}:{_normalize_version(version)}"

    # Use artifact - this registers lineage in W&B
    artifact = wandb.use_artifact(artifact_ref)

    # Download/get local path
    local_path = artifact.download()

    result = {
        "path": local_path,
        "version": artifact.version,
        "name": artifact.name,
        "type": artifact.type,
        "qualified_name": getattr(artifact, "qualified_name", None),
    }

    _ARTIFACT_CACHE[cache_key] = result

    return result


def resolve_artifact_pre_init(
    artifact_ref: str,
    *,
    entity: str | None = None,
    project: str | None = None,
    patch_http_digest: bool = False,
) -> dict[str, Any]:
    """Resolve a W&B artifact via `wandb.Api()` without requiring an active run.

    This is used in training scripts where `wandb.init()` is handled elsewhere
    (e.g. Megatron-Bridge). It returns `qualified_name` so lineage can be
    registered once a run becomes active.
    """
    import logging

    logger = logging.getLogger(__name__)

    name, version = _parse_artifact_ref(artifact_ref)
    version_str = _normalize_version(version)

    # Don't cache :latest lookups - always fetch fresh to get actual latest version
    # Only cache explicit version references (v1, v2, etc.)
    use_cache = version_str != "latest"
    cache_key = (
        f"pre_init:{name}:{version_str}:{entity or ''}:{project or ''}:{int(patch_http_digest)}"
    )

    if use_cache and cache_key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[cache_key]

    import wandb

    if patch_http_digest:
        try:
            from nemotron.kit.wandb import patch_wandb_http_handler_skip_digest_verification

            patch_wandb_http_handler_skip_digest_verification()
        except Exception:
            # Best-effort: do not fail artifact resolution because patching failed.
            pass

    # Clear any W&B internal caches to ensure fresh artifact lookups
    # This is important when resolving :latest to get the actual latest version
    try:
        # Clear the artifact cache directory state if possible
        if hasattr(wandb, "_artifacts_cache"):
            wandb._artifacts_cache = None
    except Exception:
        pass

    # Create fresh API instance for each lookup
    # timeout=30 ensures reasonable API response time
    api = wandb.Api(timeout=30)

    resolved_entity = entity or os.environ.get("WANDB_ENTITY")
    resolved_project = project or os.environ.get("WANDB_PROJECT") or "nemotron"

    # Fully-qualified path is typically entity/project/name:version.
    # Keep compatibility with earlier behavior that allowed omitting entity.
    if resolved_entity:
        full_ref = f"{resolved_entity}/{resolved_project}/{name}:{version_str}"
    else:
        full_ref = f"{resolved_project}/{name}:{version_str}"

    # Use print for critical debug info (logger may not be configured yet)
    print(f"[ARTIFACT] Resolving artifact: {full_ref} (requested version: {version_str})")
    logger.info(f"[ARTIFACT] Resolving artifact: {full_ref}")

    # Fetch artifact metadata fresh from W&B API (no caching at API level)
    # Note: api.artifact() makes a fresh network request each time
    artifact = api.artifact(full_ref)

    # Log which version :latest actually resolved to
    print(f"[ARTIFACT] W&B resolved to version: {artifact.version} (qualified: {artifact.qualified_name})")
    logger.info(f"[ARTIFACT] Resolved to version: {artifact.version} (qualified: {artifact.qualified_name})")

    local_path = artifact.download(skip_cache=True)

    logger.info(f"[ARTIFACT] Downloaded to: {local_path}")

    # Check if artifact contains file:// references (e.g., model checkpoints on shared storage)
    # If so, extract the actual path from the reference instead of using the W&B download path
    reference_path = None
    try:
        # First check if artifact metadata has absolute_path (preferred for cross-job access)
        artifact_metadata = artifact.metadata or {}
        if "absolute_path" in artifact_metadata:
            reference_path = artifact_metadata["absolute_path"]
            print(f"[ARTIFACT] Found absolute_path in metadata: {reference_path}")
            logger.info(f"[ARTIFACT] Using absolute_path from metadata: {reference_path}")

            # Handle legacy artifacts where absolute_path points to iter_XXXXXX directory
            # instead of the save_dir. Megatron-Bridge expects pretrained_checkpoint to be
            # the save_dir, and uses ckpt_step to determine which iteration to load.
            # For legacy artifacts, extract the iteration and normalize path to save_dir.
            import re
            path_obj = Path(reference_path)
            iter_match = re.match(r"^iter_(\d+)$", path_obj.name)
            if iter_match:
                # Extract iteration from path (legacy artifact)
                extracted_iteration = int(iter_match.group(1))
                # Normalize to save_dir (parent)
                reference_path = str(path_obj.parent)
                print(f"[ARTIFACT] Legacy artifact: extracted iteration={extracted_iteration}, normalized path to save_dir: {reference_path}")
                logger.info(f"[ARTIFACT] Normalized iter_XXXXXX path to parent save_dir: {reference_path}")
                # Store extracted iteration in metadata for ${art:model,iteration} resolver
                # Only if not already present (don't override actual metadata)
                if "iteration" not in artifact_metadata:
                    artifact_metadata["iteration"] = extracted_iteration
        else:
            # Fall back to extracting from file:// references
            manifest = artifact.manifest
            if manifest and hasattr(manifest, "entries"):
                entries = manifest.entries
                # Look for file:// references in the manifest
                for entry_name, entry in entries.items():
                    if hasattr(entry, "ref") and entry.ref and entry.ref.startswith("file://"):
                        # Extract the path from the file:// URI
                        ref_path = entry.ref[7:]  # Remove "file://" prefix
                        # Get the parent directory (checkpoint directory)
                        ref_dir = str(Path(ref_path).parent)

                        # Map /nemo_run/ paths to actual Lustre path using NEMO_RUN_DIR
                        # This handles artifacts created with container mount paths
                        nemo_run_dir = os.environ.get("NEMO_RUN_DIR")
                        if nemo_run_dir and nemo_run_dir != "/nemo_run":
                            if ref_dir.startswith("/nemo_run/"):
                                ref_dir = ref_dir.replace("/nemo_run/", f"{nemo_run_dir}/", 1)
                            elif ref_dir.startswith("/nemo_run"):
                                ref_dir = ref_dir.replace("/nemo_run", nemo_run_dir, 1)

                        if reference_path is None:
                            reference_path = ref_dir
                            print(f"[ARTIFACT] Found file reference, using path: {reference_path}")
                            logger.info(f"[ARTIFACT] Found file reference: {entry.ref} -> {reference_path}")
                        break
    except Exception as e:
        logger.warning(f"[ARTIFACT] Could not check for references: {e}")

    # Use reference path if available, otherwise use download path
    effective_path = reference_path if reference_path else local_path

    # Log contents of downloaded artifact directory
    local_path_obj = Path(local_path)
    if local_path_obj.exists():
        try:
            contents = list(local_path_obj.iterdir())
            logger.info(f"[ARTIFACT] Contents: {[f.name for f in contents]}")

            # Check for metadata.json specifically
            metadata_path = local_path_obj / "metadata.json"
            if metadata_path.exists():
                logger.info(f"[ARTIFACT] metadata.json found at {metadata_path}")
            else:
                logger.warning(f"[ARTIFACT] metadata.json NOT found at {metadata_path}")
                # List all files recursively to help debug
                all_files = list(local_path_obj.rglob("*"))
                logger.info(f"[ARTIFACT] All files (recursive): {[str(f) for f in all_files[:20]]}")
        except Exception as e:
            logger.warning(f"[ARTIFACT] Could not list directory: {e}")
    else:
        logger.warning(f"[ARTIFACT] Download path does not exist: {local_path}")

    result = {
        "path": effective_path,
        "version": getattr(artifact, "version", None),
        "name": getattr(artifact, "name", name),
        "type": getattr(artifact, "type", None),
        "qualified_name": getattr(artifact, "qualified_name", None),
        # Store the W&B download path separately for metadata access
        # This is needed when the effective_path is a file reference to shared storage
        # but metadata.json was uploaded to W&B
        "metadata_dir": local_path,
    }

    # Include iteration from artifact metadata (for model checkpoint artifacts)
    # This is used by ${art:model,iteration} resolver to set checkpoint.ckpt_step
    # Check artifact_metadata (which may have been updated with extracted iteration from path)
    # or fall back to the original W&B artifact metadata
    iteration = artifact_metadata.get("iteration")
    if iteration is None:
        # Check original W&B artifact metadata
        original_metadata = artifact.metadata or {}
        iteration = original_metadata.get("iteration")
    if iteration is not None:
        result["iteration"] = iteration
        print(f"[ARTIFACT] Stored iteration={result['iteration']} for resolver access")

    # Only cache explicit version lookups, not :latest
    if use_cache:
        _ARTIFACT_CACHE[cache_key] = result
    return result


def _read_artifact_metadata(artifact_path: str) -> dict[str, Any]:
    """Read metadata.json from an artifact directory.

    Args:
        artifact_path: Path to the artifact directory.

    Returns:
        Parsed metadata dict, or empty dict if not found.
    """
    import logging

    logger = logging.getLogger(__name__)

    artifact_path_obj = Path(artifact_path)
    metadata_path = artifact_path_obj / "metadata.json"

    logger.info(f"Looking for metadata.json at: {metadata_path}")
    logger.info(f"Artifact path exists: {artifact_path_obj.exists()}")

    if artifact_path_obj.exists():
        try:
            contents = list(artifact_path_obj.iterdir())
            logger.info(f"Contents of artifact directory: {[f.name for f in contents]}")
        except Exception as e:
            logger.warning(f"Could not list artifact directory: {e}")

    if metadata_path.exists():
        logger.info(f"metadata.json found, reading...")
        return json.loads(metadata_path.read_text())

    logger.warning(f"metadata.json NOT found at {metadata_path}")
    return {}


def _art_resolver(name: str, field: str = "path") -> Any:
    """OmegaConf resolver for ${art:NAME,FIELD} syntax.

    Args:
        name: Artifact key from run.artifacts (e.g., "data", "model")
        field: Field to return (default: "path"). Options:
            - path, version, name, type: Basic artifact fields
            - Any field from metadata.json (e.g., pack_size, training_path)
            - metadata.X: Explicit metadata field access (legacy syntax)

    Returns:
        The requested field value (preserves original type: int, float, str, etc.)

    Examples:
        ${art:data,path}              -> /path/to/artifact (str)
        ${art:data,pack_size}         -> 4096 (int from metadata.json)
        ${art:data,training_path}     -> /path/to/training_4096.npy (str)
        ${art:data,metadata.pack_size} -> 4096 (int, explicit metadata syntax)
    """
    if name not in _ARTIFACT_REGISTRY:
        raise KeyError(
            f"Artifact '{name}' not found. "
            f"Available: {list(_ARTIFACT_REGISTRY.keys())}. "
            "Did you call register_resolvers() with the artifacts dict?"
        )

    artifact_info = _ARTIFACT_REGISTRY[name]

    # Use metadata_dir for reading metadata.json (W&B download location)
    # Falls back to path if metadata_dir not available (for backwards compat)
    metadata_dir = artifact_info.get("metadata_dir") or artifact_info.get("path")

    # Handle explicit metadata.* prefix (legacy syntax)
    if field.startswith("metadata."):
        metadata_field = field[len("metadata.") :]
        if not metadata_dir:
            raise KeyError(f"Artifact '{name}' has no path, cannot read metadata")

        metadata = _read_artifact_metadata(metadata_dir)
        if metadata_field not in metadata:
            raise KeyError(
                f"Field '{metadata_field}' not found in metadata.json for artifact '{name}'. "
                f"Available fields: {list(metadata.keys())}"
            )
        return metadata[metadata_field]

    # Check if field is in artifact_info first (excludes metadata_dir from direct access)
    if field in artifact_info and field != "metadata_dir":
        return artifact_info[field]

    # Fall back to reading from metadata.json
    if metadata_dir:
        metadata = _read_artifact_metadata(metadata_dir)
        if field in metadata:
            return metadata[field]

    # Field not found anywhere
    available_fields = [k for k in artifact_info.keys() if k != "metadata_dir"]
    if metadata_dir:
        metadata = _read_artifact_metadata(metadata_dir)
        available_fields.extend([f for f in metadata.keys() if f not in available_fields])

    raise KeyError(
        f"Unknown field '{field}' for artifact '{name}'. Available fields: {available_fields}"
    )


def register_resolvers(
    artifacts: dict[str, str] | None = None,
    *,
    replace: bool = True,
    mode: ResolverMode = "active_run",
    pre_init_patch_http_digest: bool = False,
) -> list[str]:
    """Register OmegaConf resolvers for artifact resolution.

    This should be called early in the training script, before loading
    any configs that use ${art.X.path} interpolations.

    Args:
        artifacts: Dict mapping artifact keys to artifact references.
            Example: {"data": "DataBlendsArtifact-pretrain:v5", "model": "ModelArtifact"}
            The key is what you use in ${art.KEY.path}, the value is the W&B artifact name.
        replace: Whether to replace existing resolver (default True).

    Example:
        >>> from nemotron.kit.resolvers import register_resolvers
        >>> register_resolvers(artifacts={
        ...     "data": "DataBlendsArtifact-pretrain",
        ...     "model": "ModelArtifact-pretrain:v5",
        ... })
        >>> config = OmegaConf.load("train.yaml")
        >>> # ${art.data.path} now resolves to the downloaded artifact path
    """
    qualified_names: list[str] = []

    # Pre-resolve all artifacts
    if artifacts:
        rank, world_size = _get_distributed_info()

        if world_size > 1:
            # Distributed mode: only rank 0 downloads, others wait
            marker_path = _get_marker_path(artifacts)

            if rank == 0:
                # Rank 0: ALWAYS delete any existing marker file first to ensure fresh download.
                # This prevents stale marker files from causing other ranks to read old cached data.
                # Even though marker_path includes job_id, we delete it unconditionally to ensure
                # the current run always gets fresh artifacts (important for :latest resolution).
                if marker_path.exists():
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(f"[ARTIFACT] Rank 0: Deleting existing marker file to force fresh download: {marker_path}")
                    marker_path.unlink()

                # Rank 0: download artifacts and write marker file
                results: dict[str, dict[str, Any]] = {}
                print(f"[ARTIFACT] Rank 0: Resolving {len(artifacts)} artifacts: {list(artifacts.keys())}")
                for key, artifact_ref in artifacts.items():
                    print(f"[ARTIFACT] Rank 0: Starting resolution of '{key}' -> '{artifact_ref}'")
                    try:
                        if mode == "active_run":
                            name, version = _parse_artifact_ref(artifact_ref)
                            result = _resolve_artifact_active_run(name, version)
                        elif mode == "pre_init":
                            result = resolve_artifact_pre_init(
                                artifact_ref,
                                patch_http_digest=pre_init_patch_http_digest,
                            )
                        else:
                            raise ValueError(f"Unknown resolver mode: {mode}")

                        _ARTIFACT_REGISTRY[key] = result
                        results[key] = result

                        qname = result.get("qualified_name")
                        if qname:
                            qualified_names.append(str(qname))
                        print(f"[ARTIFACT] Rank 0: Successfully resolved '{key}' -> {result.get('path')}")
                    except Exception as e:
                        print(f"[ARTIFACT] Rank 0: ERROR resolving '{key}' ({artifact_ref}): {e}")
                        raise

                # Signal completion to other ranks
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text(
                    json.dumps(
                        {
                            "results": results,
                            "qualified_names": qualified_names,
                        }
                    )
                )
            else:
                # Other ranks: wait for rank 0 and use shared results
                data = _wait_for_artifacts(marker_path)
                for key, result in data["results"].items():
                    _ARTIFACT_REGISTRY[key] = result
                qualified_names = data["qualified_names"]
        else:
            # Single process mode: download directly (existing behavior)
            print(f"[ARTIFACT] Single process: Resolving {len(artifacts)} artifacts: {list(artifacts.keys())}")
            for key, artifact_ref in artifacts.items():
                print(f"[ARTIFACT] Starting resolution of '{key}' -> '{artifact_ref}'")
                try:
                    if mode == "active_run":
                        name, version = _parse_artifact_ref(artifact_ref)
                        result = _resolve_artifact_active_run(name, version)
                    elif mode == "pre_init":
                        result = resolve_artifact_pre_init(
                            artifact_ref,
                            patch_http_digest=pre_init_patch_http_digest,
                        )
                    else:
                        raise ValueError(f"Unknown resolver mode: {mode}")

                    _ARTIFACT_REGISTRY[key] = result

                    qname = result.get("qualified_name")
                    if qname:
                        qualified_names.append(str(qname))
                    print(f"[ARTIFACT] Successfully resolved '{key}' -> {result.get('path')}")
                except Exception as e:
                    print(f"[ARTIFACT] ERROR resolving '{key}' ({artifact_ref}): {e}")
                    raise

    # Register the resolver
    # ${art.data.path} -> _art_resolver("data", "path")
    # ${art.model.version} -> _art_resolver("model", "version")
    OmegaConf.register_new_resolver(
        "art",
        lambda name, field="path": _art_resolver(name, field),
        replace=replace,
    )

    return qualified_names


def register_resolvers_from_config(
    config: Any,
    artifacts_key: str = "run",
    *,
    replace: bool = True,
    mode: ResolverMode = "active_run",
    pre_init_patch_http_digest: bool = False,
) -> list[str]:
    """Register artifact resolvers from a config's run section.

    This function extracts artifact references from the config's run section.
    Artifact references are string values that look like W&B artifact names
    (contain "Artifact" in the name or match the pattern Name-stage:version).

    Args:
        config: OmegaConf config (or path to YAML file)
        artifacts_key: Dotpath to section containing artifacts (default: "run")
        replace: Whether to replace existing resolver

    Example config.yaml:
        run:
          data: DataBlendsArtifact-pretrain
          model: ModelArtifact-pretrain:v5
          env:
            container: nvcr.io/nvidian/nemo:25.11-nano-v3.rc2

        recipe:
          per_split_data_args_path: ${art.data.path}

    Example usage:
        >>> config = OmegaConf.load("config.yaml")
        >>> register_resolvers_from_config(config)
        >>> # Now resolve the config
        >>> resolved = OmegaConf.to_container(config, resolve=True)
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    # Navigate to the section containing artifacts
    section = OmegaConf.select(config, artifacts_key, default=None)

    artifacts: dict[str, str] = {}

    if section is not None:
        section_dict = OmegaConf.to_container(section, resolve=False)

        # Extract artifact references from the section
        # Artifact refs are string values that look like W&B artifact names
        if isinstance(section_dict, dict):
            print(f"[ARTIFACT] Scanning config section '{artifacts_key}' for artifact references...")
            for key, value in section_dict.items():
                is_ref = _is_artifact_reference(value)
                print(f"[ARTIFACT]   {key}={repr(value)} -> is_artifact_reference={is_ref}")
                if is_ref:
                    artifacts[key] = value
            print(f"[ARTIFACT] Detected artifacts: {artifacts}")

    if artifacts:
        return register_resolvers(
            artifacts,
            replace=replace,
            mode=mode,
            pre_init_patch_http_digest=pre_init_patch_http_digest,
        )
    else:
        # Still register the resolver, just without pre-resolved artifacts
        register_resolvers(replace=replace)
        return []


def _is_artifact_reference(value: Any) -> bool:
    """Check if a value looks like a W&B artifact reference.

    Args:
        value: Value to check

    Returns:
        True if value looks like an artifact reference

    Examples:
        >>> _is_artifact_reference("DataBlendsArtifact-pretrain")
        True
        >>> _is_artifact_reference("ModelArtifact-pretrain:v5")
        True
        >>> _is_artifact_reference("nvcr.io/nvidian/nemo:25.11")
        False
        >>> _is_artifact_reference({"nested": "dict"})
        False
    """
    if not isinstance(value, str):
        return False

    # Skip container images (contain / or nvcr or docker)
    if "/" in value or "nvcr" in value.lower() or "docker" in value.lower():
        return False

    # Check for common artifact patterns
    # Pattern 1: Contains "Artifact" (e.g., DataBlendsArtifact-pretrain)
    if "Artifact" in value:
        return True

    # Pattern 2: Ends with version specifier and looks like an artifact name
    # e.g., "my-model:v5", "dataset:latest"
    if ":" in value:
        name, version = value.rsplit(":", 1)
        if version.startswith("v") or version == "latest" or version.isdigit():
            # Verify name part looks artifact-like (no slashes, dots suggesting URLs)
            if "." not in name and "/" not in name:
                return True

    return False


def clear_artifact_cache() -> None:
    """Clear the artifact cache.

    Useful for testing or when you want to re-resolve artifacts.
    """
    _ARTIFACT_REGISTRY.clear()
    _ARTIFACT_CACHE.clear()
