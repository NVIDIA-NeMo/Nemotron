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


def _get_marker_path(artifacts: dict[str, str]) -> Path:
    """Generate a unique marker file path based on artifact references.

    Uses NEMO_RUN_DIR (shared filesystem) if available for multi-node jobs,
    otherwise falls back to TMPDIR or /tmp.

    Args:
        artifacts: Dict of artifact key -> artifact reference.

    Returns:
        Path to marker file on shared or local storage.
    """
    # Hash the artifacts dict to create a unique marker per config
    artifacts_str = json.dumps(sorted(artifacts.items()))
    hash_suffix = hashlib.md5(artifacts_str.encode()).hexdigest()[:8]

    # Prefer NEMO_RUN_DIR (shared filesystem) for multi-node jobs
    # Fall back to TMPDIR or /tmp for single-node or local runs
    base_dir = os.environ.get("NEMO_RUN_DIR") or os.environ.get("TMPDIR", "/tmp")
    return Path(base_dir) / f".nemotron_artifacts_{hash_suffix}"


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
    name, version = _parse_artifact_ref(artifact_ref)
    version_str = _normalize_version(version)
    cache_key = (
        f"pre_init:{name}:{version_str}:{entity or ''}:{project or ''}:{int(patch_http_digest)}"
    )

    if cache_key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[cache_key]

    import wandb

    if patch_http_digest:
        try:
            from nemotron.kit.wandb import patch_wandb_http_handler_skip_digest_verification

            patch_wandb_http_handler_skip_digest_verification()
        except Exception:
            # Best-effort: do not fail artifact resolution because patching failed.
            pass

    api = wandb.Api()

    resolved_entity = entity or os.environ.get("WANDB_ENTITY")
    resolved_project = project or os.environ.get("WANDB_PROJECT") or "nemotron"

    # Fully-qualified path is typically entity/project/name:version.
    # Keep compatibility with earlier behavior that allowed omitting entity.
    if resolved_entity:
        full_ref = f"{resolved_entity}/{resolved_project}/{name}:{version_str}"
    else:
        full_ref = f"{resolved_project}/{name}:{version_str}"

    artifact = api.artifact(full_ref)
    local_path = artifact.download(skip_cache=True)

    result = {
        "path": local_path,
        "version": getattr(artifact, "version", None),
        "name": getattr(artifact, "name", name),
        "type": getattr(artifact, "type", None),
        "qualified_name": getattr(artifact, "qualified_name", None),
    }

    _ARTIFACT_CACHE[cache_key] = result
    return result


def _read_artifact_metadata(artifact_path: str) -> dict[str, Any]:
    """Read metadata.json from an artifact directory.

    Args:
        artifact_path: Path to the artifact directory.

    Returns:
        Parsed metadata dict, or empty dict if not found.
    """
    metadata_path = Path(artifact_path) / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _art_resolver(name: str, field: str = "path") -> str:
    """OmegaConf resolver for ${art:NAME,FIELD} syntax.

    Args:
        name: Artifact key from run.artifacts (e.g., "data", "model")
        field: Field to return (default: "path"). Options:
            - path, version, name, type: Basic artifact fields
            - metadata.X: Read field X from artifact's metadata.json

    Returns:
        The requested field value as string

    Examples:
        ${art:data,path}              -> /path/to/artifact
        ${art:data,metadata.pack_size} -> 4096
    """
    if name not in _ARTIFACT_REGISTRY:
        raise KeyError(
            f"Artifact '{name}' not found. "
            f"Available: {list(_ARTIFACT_REGISTRY.keys())}. "
            "Did you call register_resolvers() with the artifacts dict?"
        )

    artifact_info = _ARTIFACT_REGISTRY[name]

    # Handle metadata.* fields by reading from metadata.json
    if field.startswith("metadata."):
        metadata_field = field[len("metadata.") :]
        artifact_path = artifact_info.get("path")
        if not artifact_path:
            raise KeyError(f"Artifact '{name}' has no path, cannot read metadata")

        metadata = _read_artifact_metadata(artifact_path)
        if metadata_field not in metadata:
            raise KeyError(
                f"Field '{metadata_field}' not found in metadata.json for artifact '{name}'. "
                f"Available fields: {list(metadata.keys())}"
            )
        return str(metadata[metadata_field])

    if field not in artifact_info:
        raise KeyError(
            f"Unknown field '{field}' for artifact '{name}'. "
            f"Available fields: {list(artifact_info.keys())} or metadata.*"
        )

    return str(artifact_info[field])


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
                # Rank 0: download artifacts and write marker file
                results: dict[str, dict[str, Any]] = {}
                for key, artifact_ref in artifacts.items():
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
            for key, artifact_ref in artifacts.items():
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
            for key, value in section_dict.items():
                if _is_artifact_reference(value):
                    artifacts[key] = value

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
