"""
Artifact registry for nemotron.kit.

Manages artifact versions, aliases, and storage locations using fsspec or W&B backends.
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nemotron.kit.exceptions import ArtifactNotFoundError, ArtifactVersionNotFoundError


@dataclass
class ArtifactVersion:
    """A specific version of an artifact."""

    version: int
    path: str  # Can be local path or remote URL
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactEntry:
    """Registry entry for an artifact with all its versions."""

    name: str
    versions: list[ArtifactVersion] = field(default_factory=list)
    aliases: dict[str, int] = field(default_factory=dict)  # alias -> version number

    def latest_version(self) -> ArtifactVersion | None:
        """Get the latest version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version)

    def get_version(self, version: int) -> ArtifactVersion | None:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "versions": [
                {
                    "version": v.version,
                    "path": v.path,
                    "created_at": v.created_at,
                    "metadata": v.metadata,
                }
                for v in self.versions
            ],
            "aliases": self.aliases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactEntry":
        """Deserialize from dictionary."""
        versions = [
            ArtifactVersion(
                version=v["version"],
                path=v["path"],
                created_at=v["created_at"],
                metadata=v.get("metadata", {}),
            )
            for v in data.get("versions", [])
        ]
        return cls(
            name=data["name"],
            versions=versions,
            aliases=data.get("aliases", {}),
        )


class ArtifactRegistry:
    """Registry for managing artifact versions and storage.

    Supports two backends:
    - fsspec: Stores artifacts in a filesystem (local, S3, GCS, etc.)
    - wandb: Uses W&B artifacts for storage and versioning

    Example:
        >>> from nemotron.kit import init
        >>> init(backend="fsspec", root="/data/artifacts")
        >>>
        >>> from nemotron.kit.registry import get_registry
        >>> registry = get_registry()
        >>> version = registry.publish("my-dataset", Path("/tmp/data"))
        >>> local_path = registry.resolve("my-dataset", version=1)
    """

    def __init__(
        self,
        backend: str = "fsspec",
        root: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            backend: Storage backend ("fsspec" or "wandb")
            root: Root path for fsspec backend (required for fsspec)
            wandb_project: W&B project name (required for wandb)
            wandb_entity: W&B entity/team name (optional for wandb)
        """
        self.backend = backend
        self.root = Path(root) if root else None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # In-memory cache of registry entries
        self._entries: dict[str, ArtifactEntry] = {}

        if backend == "fsspec":
            if not self.root:
                raise ValueError("root is required for fsspec backend")
            self.root.mkdir(parents=True, exist_ok=True)
            self._load_index()
        elif backend == "wandb":
            if not self.wandb_project:
                raise ValueError("wandb_project is required for wandb backend")
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "wandb is required for wandb backend. Install with: pip install wandb"
                )

    def _index_path(self) -> Path:
        """Get path to registry index file."""
        return self.root / ".art_index.json"

    def _load_index(self) -> None:
        """Load registry index from disk."""
        index_path = self._index_path()
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self._entries = {
                name: ArtifactEntry.from_dict(entry) for name, entry in data.items()
            }

    def _save_index(self) -> None:
        """Save registry index to disk."""
        index_path = self._index_path()
        data = {name: entry.to_dict() for name, entry in self._entries.items()}

        # Atomic write
        temp_path = index_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(index_path)

    def publish(
        self,
        name: str,
        source_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactVersion:
        """Publish a new version of an artifact.

        Args:
            name: Artifact name
            source_path: Local path to artifact directory
            metadata: Optional metadata to attach

        Returns:
            The new ArtifactVersion
        """
        if self.backend == "fsspec":
            return self._publish_fsspec(name, source_path, metadata)
        elif self.backend == "wandb":
            return self._publish_wandb(name, source_path, metadata)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _publish_fsspec(
        self,
        name: str,
        source_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactVersion:
        """Publish using fsspec backend."""
        # Get or create entry
        entry = self._entries.get(name)
        if entry is None:
            entry = ArtifactEntry(name=name)
            self._entries[name] = entry

        # Determine next version
        latest = entry.latest_version()
        next_version = (latest.version + 1) if latest else 1

        # Create version directory
        version_dir = self.root / name / f"v{next_version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact to registry
        if source_path.is_dir():
            # Copy directory contents
            for item in source_path.iterdir():
                dest = version_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        else:
            # Copy single file
            shutil.copy2(source_path, version_dir / source_path.name)

        # Create version entry
        version = ArtifactVersion(
            version=next_version,
            path=str(version_dir),
            created_at=datetime.now().astimezone().isoformat(),
            metadata=metadata or {},
        )
        entry.versions.append(version)

        # Update latest alias
        entry.aliases["latest"] = next_version

        # Save index
        self._save_index()

        return version

    def _publish_wandb(
        self,
        name: str,
        source_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactVersion:
        """Publish using W&B backend."""
        # Create W&B artifact
        artifact = self._wandb.Artifact(
            name=name,
            type="artifact",
            metadata=metadata or {},
        )

        # Add source directory
        if source_path.is_dir():
            artifact.add_dir(str(source_path))
        else:
            artifact.add_file(str(source_path))

        # Log artifact (requires active run or creates one)
        if self._wandb.run is None:
            # Initialize a run for artifact logging
            run = self._wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                job_type="artifact-publish",
            )
            logged = run.log_artifact(artifact)
            logged.wait()
            run.finish()
        else:
            logged = self._wandb.run.log_artifact(artifact)
            logged.wait()

        # Create version entry
        version = ArtifactVersion(
            version=int(logged.version.lstrip("v")),
            path=f"{self.wandb_entity or logged.entity}/{self.wandb_project}/{name}:{logged.version}",
            created_at=datetime.now().astimezone().isoformat(),
            metadata=metadata or {},
        )

        return version

    def resolve(self, name: str, version: int | None = None) -> Path:
        """Resolve artifact to local path.

        Args:
            name: Artifact name
            version: Version number (None for latest)

        Returns:
            Local path to artifact

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist
            ArtifactVersionNotFoundError: If version doesn't exist
        """
        if self.backend == "fsspec":
            return self._resolve_fsspec(name, version)
        elif self.backend == "wandb":
            return self._resolve_wandb(name, version)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _resolve_fsspec(self, name: str, version: int | None = None) -> Path:
        """Resolve using fsspec backend."""
        entry = self._entries.get(name)
        if entry is None:
            raise ArtifactNotFoundError(name)

        if version is None:
            # Get latest
            artifact_version = entry.latest_version()
            if artifact_version is None:
                raise ArtifactVersionNotFoundError(name, "latest")
        else:
            artifact_version = entry.get_version(version)
            if artifact_version is None:
                raise ArtifactVersionNotFoundError(name, version)

        return Path(artifact_version.path)

    def _resolve_wandb(self, name: str, version: int | None = None) -> Path:
        """Resolve using W&B backend."""
        # Build artifact reference
        if version is None:
            ref = f"{name}:latest"
        else:
            ref = f"{name}:v{version}"

        if self.wandb_entity:
            ref = f"{self.wandb_entity}/{self.wandb_project}/{ref}"
        else:
            ref = f"{self.wandb_project}/{ref}"

        # Download artifact
        try:
            api = self._wandb.Api()
            artifact = api.artifact(ref)
            local_path = artifact.download()
            return Path(local_path)
        except Exception as e:
            if "not found" in str(e).lower():
                if version is None:
                    raise ArtifactNotFoundError(name)
                else:
                    raise ArtifactVersionNotFoundError(name, version)
            raise

    def get(self, name: str) -> ArtifactEntry | None:
        """Get artifact entry by name."""
        if self.backend == "fsspec":
            return self._entries.get(name)
        elif self.backend == "wandb":
            # For W&B, we'd need to query the API
            # Return None for now - resolve() handles the actual lookup
            return None
        return None

    def list(self) -> list[str]:
        """List all artifact names."""
        if self.backend == "fsspec":
            return list(self._entries.keys())
        elif self.backend == "wandb":
            # Would need to query W&B API
            return []
        return []

    def alias(self, name: str, alias: str, version: int) -> None:
        """Create an alias for a specific version.

        Args:
            name: Artifact name
            alias: Alias name (e.g., "production", "staging")
            version: Version number to alias
        """
        if self.backend == "fsspec":
            entry = self._entries.get(name)
            if entry is None:
                raise ArtifactNotFoundError(name)

            if entry.get_version(version) is None:
                raise ArtifactVersionNotFoundError(name, version)

            entry.aliases[alias] = version
            self._save_index()
        elif self.backend == "wandb":
            # W&B has its own alias system
            api = self._wandb.Api()
            artifact = api.artifact(f"{self.wandb_project}/{name}:v{version}")
            artifact.aliases.append(alias)
            artifact.save()


# Global registry instance
_registry: ArtifactRegistry | None = None


def get_registry() -> ArtifactRegistry:
    """Get the global artifact registry.

    Raises:
        RuntimeError: If kit.init() hasn't been called
    """
    if _registry is None:
        raise RuntimeError(
            "Artifact registry not initialized. Call kit.init() first."
        )
    return _registry


def set_registry(registry: ArtifactRegistry | None) -> None:
    """Set the global artifact registry."""
    global _registry
    _registry = registry
