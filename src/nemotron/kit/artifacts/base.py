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

"""Base artifact class and tracking info."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from nemotron.kit.trackers import LineageTracker


class TrackingInfo(BaseModel):
    """Information about artifact tracking in external systems."""

    artifact_id: str | None = None
    artifact_type: str | None = None
    run_id: str | None = None
    url: str | None = None
    used_artifacts: Annotated[list[str], Field(default_factory=list)]


class Artifact(BaseModel):
    """Path-centric artifact with optional typed metadata.

    Core philosophy: An artifact IS a path with metadata.

    Simple usage (no subclass needed):
        >>> artifact = Artifact(path=Path("/data/model"), type="model")
        >>> artifact.metadata["step"] = 10000

    Typed subclass for validation and IDE support:
        >>> class ModelArtifact(Artifact):
        ...     step: int
        ...     final_loss: float | None = None
        >>>
        >>> model = ModelArtifact(path=Path("/data/model"), step=10000)
        >>> model.step  # IDE autocomplete works
        >>> model.metadata["step"]  # Also accessible here
    """

    # === Core fields ===
    path: Annotated[Path, Field(description="Filesystem path to the artifact")]
    type: Annotated[str, Field(default="artifact", description="Artifact type")]
    metadata: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Artifact metadata")
    ]

    # === Provenance fields ===
    created_at: Annotated[
        str,
        Field(
            default_factory=lambda: datetime.now().astimezone().isoformat(),
            description="ISO timestamp of creation",
        ),
    ]
    producer: Annotated[str | None, Field(default=None, description="Run ID or 'local'")]
    tracking: Annotated[TrackingInfo | None, Field(default=None, description="Tracking metadata")]
    name: Annotated[
        str | None,
        Field(default=None, description="Semantic artifact name (e.g., nano3/pretrain/data)"),
    ]

    # === Private registry state ===
    _name: str | None = None
    _version: int | None = None
    _used_artifacts: list[str] = []

    @classmethod
    def _get_metadata_fields(cls) -> set[str]:
        """Get fields that should be synced to metadata dict.

        These are fields defined in subclasses but not in Artifact base.
        """
        base_fields = {"path", "type", "metadata", "created_at", "producer", "tracking", "name"}
        if hasattr(cls, "model_fields"):
            return set(cls.model_fields.keys()) - base_fields
        return set()

    @model_validator(mode="before")
    @classmethod
    def _setup_defaults(cls, data: Any) -> Any:
        """Set default type and sync metadata fields."""
        if not isinstance(data, dict):
            return data

        # Set type from class name if not provided
        if "type" not in data or data["type"] == "artifact":
            data["type"] = cls.__name__

        # Ensure metadata dict exists
        if "metadata" not in data:
            data["metadata"] = {}

        # Pull typed fields from metadata if provided there (for loading)
        metadata_fields = cls._get_metadata_fields()
        for field_name in metadata_fields:
            if field_name not in data and field_name in data["metadata"]:
                data[field_name] = data["metadata"][field_name]

        return data

    @model_validator(mode="after")
    def _sync_to_metadata(self) -> Self:
        """Push typed fields into metadata dict after validation."""
        metadata_fields = self._get_metadata_fields()
        for field_name in metadata_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if value is not None:
                    self.metadata[field_name] = value
        return self

    @property
    def metrics(self) -> dict[str, float]:
        """Extract numeric metrics from metadata for logging."""
        return {
            k: float(v)
            for k, v in self.metadata.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

    def get_wandb_files(self) -> list[tuple[str, str]]:
        """Return list of (local_path, artifact_name) tuples for files to upload.

        These files are small and will be uploaded to W&B storage.
        Override in subclasses to customize which files are uploaded.

        Returns:
            List of (local_path, artifact_name) tuples.
            Default: [("metadata.json", "metadata.json")] if it exists.
        """
        files = []
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            files.append((str(metadata_path), "metadata.json"))
        return files

    def get_wandb_references(self) -> list[tuple[str, str]]:
        """Return list of (uri, name) tuples for references to add.

        References point to data on shared storage (e.g., /lustre) without uploading.
        Override in subclasses to customize which references are added.

        Returns:
            List of (uri, name) tuples.
            Default: reference to artifact directory as "output".
        """
        # Default: add reference to the artifact directory
        artifact_path = self.path
        if artifact_path.is_file():
            artifact_path = artifact_path.parent
        return [(f"file://{artifact_path.resolve()}", "output")]

    def get_input_uris(self) -> list[str]:
        """Return list of input URIs for lineage tracking.

        Override in subclasses to specify input dependencies.

        Returns:
            List of URIs (hf://, file://, etc.) that this artifact depends on.
        """
        return []

    @property
    def uri(self) -> str | None:
        """Return art:// URI if published, None otherwise."""
        if self._name is not None and self._version is not None:
            return f"art://{self._name}:v{self._version}"
        return None

    @property
    def art_path(self) -> str:
        """Return art:// URI for downstream consumption.

        For registered artifacts: art://name:vN
        For named artifacts: art://name
        For unnamed artifacts: art:///absolute/path
        """
        if self._name is not None and self._version is not None:
            return f"art://{self._name}:v{self._version}"
        if self.name is not None:
            return f"art://{self.name}"
        # Fallback: use absolute path
        return f"art://{self.path.resolve()}"

    def _get_output_dir(self) -> Path:
        """Get the output directory for metadata.json.

        Override in subclasses where path points to a file instead of directory.
        """
        return self.path

    def _derive_artifact_name(self, name: str | None) -> str:
        """Derive W&B artifact name from semantic name or type.

        Args:
            name: Explicit name override, or None to derive from self.name

        Returns:
            Artifact name like "SFTDataArtifact-sft"
        """
        if name is not None:
            return name

        if self.name:
            # Extract stage from semantic name (e.g., "nano3/sft/data" -> "sft")
            parts = self.name.split("/")
            if len(parts) >= 2:
                stage = parts[1].split("?")[0]  # Remove query params like ?sample=100
                return f"{self.type}-{stage}"

        return self.type

    def save(self, name: str | None = None) -> None:
        """Save artifact metadata to path/metadata.json (atomic write).

        If tracking is active, also logs to tracking backend.
        If kit.init() was called, publishes to registry.

        Args:
            name: Optional name for artifact in registry. Defaults to type.
        """
        from nemotron.kit.trackers import get_lineage_tracker

        output_dir = self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set producer before writing metadata
        tracker = get_lineage_tracker()
        if tracker and tracker.is_active():
            if self.producer is None:
                self.producer = tracker.get_run_id() or "local"
        else:
            if self.producer is None:
                self.producer = "local"

        # Write metadata.json FIRST (before log_artifact) so tracker can include it
        # This is critical: tracker.log_artifact() reads metadata.json to add to W&B artifact
        metadata_file_path = output_dir / "metadata.json"
        temp_path = output_dir / ".metadata.json.tmp"

        with open(temp_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        # Atomic rename
        temp_path.rename(metadata_file_path)

        # Now log to tracker (after metadata.json exists on disk)
        if tracker and tracker.is_active():
            artifact_name = self._derive_artifact_name(name)
            tracking_metadata = tracker.log_artifact(self, artifact_name, self._used_artifacts)
            self.tracking = TrackingInfo(**tracking_metadata)

        # Publish to registry if initialized (skip for wandb backend - tracker handles it)
        self._publish_to_registry(name, output_dir)

    def _publish_to_registry(self, name: str | None, output_dir: Path) -> None:
        """Publish artifact to registry if initialized."""
        try:
            from nemotron.kit import get_config, is_initialized
            from nemotron.kit.registry import get_registry

            if is_initialized():
                config = get_config()
                # Skip registry publish for wandb backend - WandbTracker already logged it
                if config and config.backend != "wandb":
                    registry = get_registry()
                    artifact_name = name or self.type
                    version = registry.publish(artifact_name, output_dir, metadata=self.metadata)
                    self._name = artifact_name
                    self._version = version.version
        except ImportError:
            # Registry not available, skip
            pass

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        tracked_artifact: str | None = None,
    ) -> Self:
        """Load artifact from local path, tracked artifact, or stdin.

        Priority: tracked_artifact > path > stdin

        Args:
            path: Local filesystem path to artifact directory
            tracked_artifact: Tracked artifact reference (e.g., "team/project/data:v1")

        Returns:
            Loaded artifact instance
        """
        from nemotron.kit.trackers import get_lineage_tracker

        tracker = get_lineage_tracker()

        # Option 1: Load from tracked artifact
        if tracked_artifact:
            if not tracker or not tracker.is_active():
                raise ValueError(
                    "Cannot load tracked artifact: no active tracker. "
                    "Use set_lineage_tracker() to configure tracking."
                )
            # Download artifact and get local path
            path = tracker.use_artifact(tracked_artifact, cls.__name__.lower())

        # Option 2: Load from explicit path
        elif path:
            pass  # Use provided path

        # Option 3: Load from stdin (piping)
        else:
            if sys.stdin.isatty():
                raise ValueError(
                    "No input provided. Use --input-path, --input-artifact, or pipe from stdin."
                )
            # Read JSON from stdin
            stdin_data = json.loads(sys.stdin.read())
            if "path" not in stdin_data:
                raise ValueError("Invalid stdin data: missing 'path' field")
            path = Path(stdin_data["path"])

        # Load metadata.json
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Artifact metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            data = json.load(f)

        # Create artifact instance
        artifact = cls(**data)

        # Track usage for lineage (if tracker active)
        if tracker and tracker.is_active() and artifact.tracking:
            artifact._used_artifacts.append(artifact.tracking.artifact_id or str(artifact.path))

        return artifact

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        """Load artifact from art:// URI.

        Args:
            uri: Artifact URI (e.g., "art://my-dataset:v1" or "art://my-dataset:latest")

        Returns:
            Loaded artifact instance
        """
        from nemotron.kit.registry import get_registry

        registry = get_registry()

        # Parse URI: art://name:version or art://name
        if not uri.startswith("art://"):
            raise ValueError(f"Invalid art:// URI: {uri}")

        uri_path = uri[6:]  # Remove "art://"

        # Parse name and version
        version: int | str | None
        if ":" in uri_path:
            name, version_str = uri_path.rsplit(":", 1)
            if version_str == "latest":
                version = None
            elif version_str.startswith("v"):
                version = int(version_str[1:])
            else:
                # Try numeric; otherwise treat as alias
                try:
                    version = int(version_str)
                except ValueError:
                    version = version_str  # Alias string
        else:
            name = uri_path
            version = None

        # Resolve to local path
        local_path = registry.resolve(name, version)

        # Load artifact
        artifact = cls.load(path=local_path)

        # Set registry metadata
        artifact._name = name
        if version is not None:
            artifact._version = version
        else:
            # Get latest version number
            entry = registry.get(name)
            if entry and entry.versions:
                artifact._version = entry.versions[-1].version

        return artifact

    def to_json(self) -> str:
        """Serialize artifact to JSON for piping."""
        return json.dumps({"path": str(self.path), "type": self.type})

    def __str__(self) -> str:
        """String representation for piping to stdout."""
        return self.to_json()
