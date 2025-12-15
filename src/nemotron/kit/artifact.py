"""
Core artifact module for nemotron.kit.

Provides the Artifact base class, typed subclasses, and utilities.
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field, model_validator

from nemotron.kit.trackers import InputDatasetInfo, get_lineage_tracker


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
    name: Annotated[str | None, Field(default=None, description="Semantic artifact name (e.g., nano3/pretrain/data)")]

    # === Private registry state ===
    _name: str | None = None
    _version: int | None = None
    _used_artifacts: list[str] = []

    @classmethod
    def _get_metadata_fields(cls) -> set[str]:
        """Get fields that should be synced to metadata dict.

        These are fields defined in subclasses but not in Artifact base.
        """
        base_fields = {
            "path", "type", "metadata", "created_at", "producer", "tracking", "name"
        }
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

    def save(self, name: str | None = None) -> None:
        """Save artifact metadata to path/metadata.json (atomic write).

        If tracking is active, also logs to tracking backend.
        If kit.init() was called, publishes to registry.

        Args:
            name: Optional name for artifact in registry. Defaults to type.
        """
        # Ensure output directory exists
        self.path.mkdir(parents=True, exist_ok=True)

        # Get tracker if active
        tracker = get_lineage_tracker()
        if tracker and tracker.is_active():
            # Set producer to run ID
            if self.producer is None:
                self.producer = tracker.get_run_id() or "local"

            # Log to tracking backend
            artifact_name = name or self.type
            tracking_metadata = tracker.log_artifact(self, artifact_name, self._used_artifacts)

            # Update tracking info
            self.tracking = TrackingInfo(**tracking_metadata)
        else:
            # Local-only mode
            if self.producer is None:
                self.producer = "local"

        # Write metadata.json atomically (temp file + rename)
        metadata_path = self.path / "metadata.json"
        temp_path = self.path / ".metadata.json.tmp"

        with open(temp_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        # Atomic rename
        temp_path.rename(metadata_path)

        # Publish to registry if initialized (skip for wandb backend - tracker handles it)
        try:
            from nemotron.kit import get_config, is_initialized
            from nemotron.kit.registry import get_registry

            if is_initialized():
                config = get_config()
                # Skip registry publish for wandb backend - WandbTracker already logged it
                if config and config.backend != "wandb":
                    registry = get_registry()
                    artifact_name = name or self.type
                    version = registry.publish(artifact_name, self.path, metadata=self.metadata)
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


# =============================================================================
# Typed Artifact Subclasses
# =============================================================================


class DataBlendsArtifact(Artifact):
    """Tokenized data blends artifact (output of pretrain/RL data_prep).

    The path points directly to the blend.json file.

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    - tokenizer_uri: URI of the tokenizer model (hf://models/...)
    """

    total_tokens: Annotated[int, Field(ge=0, description="Total tokens processed")]
    total_sequences: Annotated[int, Field(ge=0, description="Total documents processed")]
    elapsed_sec: Annotated[float, Field(default=0.0, ge=0, description="Processing time in seconds")]

    # Per-split token counts (optional, populated in per-split mode)
    train_tokens: Annotated[int | None, Field(default=None, ge=0, description="Tokens in train split")]
    valid_tokens: Annotated[int | None, Field(default=None, ge=0, description="Tokens in valid split")]
    test_tokens: Annotated[int | None, Field(default=None, ge=0, description="Tokens in test split")]

    # Source datasets for lineage tracking
    # Accepts InputDatasetInfo (with metadata) or str (URI only, for backwards compat)
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]
    tokenizer_uri: Annotated[
        str | None, Field(default=None, description="URI of tokenizer model")
    ]

    def save(self, name: str | None = None) -> None:
        """Save artifact metadata to path's parent directory.

        Since DataBlendsArtifact.path points to blend.json (a file),
        metadata.json is written to the same directory as blend.json.
        """
        # Use parent directory since self.path is a file (blend.json)
        output_dir = self.path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get tracker if active
        tracker = get_lineage_tracker()
        if tracker and tracker.is_active():
            if self.producer is None:
                self.producer = tracker.get_run_id() or "local"

            # Derive artifact name from semantic name if set
            # e.g., "nano3/sft/data" -> "DataBlendsArtifact-sft"
            # e.g., "nano3/pretrain/data" -> "DataBlendsArtifact-pretrain"
            artifact_name = name
            if artifact_name is None and self.name:
                # Extract stage from semantic name (e.g., "nano3/sft/data" -> "sft")
                parts = self.name.split("/")
                if len(parts) >= 2:
                    stage = parts[1].split("?")[0]  # Remove query params like ?sample=100
                    artifact_name = f"{self.type}-{stage}"
            artifact_name = artifact_name or self.type

            tracking_metadata = tracker.log_artifact(self, artifact_name, self._used_artifacts)
            self.tracking = TrackingInfo(**tracking_metadata)
        else:
            if self.producer is None:
                self.producer = "local"

        # Write metadata.json atomically in the parent directory
        metadata_path = output_dir / "metadata.json"
        temp_path = output_dir / ".metadata.json.tmp"

        with open(temp_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        temp_path.rename(metadata_path)

        # Publish to registry if initialized (skip for wandb backend - tracker handles it)
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
            pass


class PretrainBlendsArtifact(Artifact):
    """Pretrain data blends artifact (output of pretrain data_prep).

    The path points to the output directory containing bin/idx files.
    The blend_path points to the blend.json file within that directory.

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    - tokenizer_uri: URI of the tokenizer model (hf://models/...)
    """

    total_tokens: Annotated[int, Field(ge=0, description="Total tokens processed")]
    total_sequences: Annotated[int, Field(ge=0, description="Total documents processed")]
    elapsed_sec: Annotated[float, Field(default=0.0, ge=0, description="Processing time in seconds")]

    # Sharding configuration
    num_shards: Annotated[int, Field(ge=1, description="Number of output shards")]

    # Path to blend.json for Megatron-Bridge per_split_data_args_path
    blend_path: Annotated[str | None, Field(default=None, description="Path to blend.json file")]

    # Per-split token counts (optional, populated in per-split mode)
    train_tokens: Annotated[int | None, Field(default=None, ge=0, description="Tokens in train split")]
    valid_tokens: Annotated[int | None, Field(default=None, ge=0, description="Tokens in valid split")]
    test_tokens: Annotated[int | None, Field(default=None, ge=0, description="Tokens in test split")]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]
    tokenizer_uri: Annotated[
        str | None, Field(default=None, description="URI of tokenizer model")
    ]

    def save(self, name: str | None = None) -> None:
        """Save artifact metadata to output directory.

        The path points to the output directory containing bin/idx files.
        The blend_path in metadata points to blend.json within that directory.
        """
        output_dir = self.path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get tracker if active
        tracker = get_lineage_tracker()
        if tracker and tracker.is_active():
            if self.producer is None:
                self.producer = tracker.get_run_id() or "local"

            # Derive artifact name from semantic name if set
            # e.g., "nano3/pretrain/data" -> "PretrainBlendsArtifact-pretrain"
            artifact_name = name
            if artifact_name is None and self.name:
                # Extract stage from semantic name (e.g., "nano3/pretrain/data" -> "pretrain")
                parts = self.name.split("/")
                if len(parts) >= 2:
                    stage = parts[1].split("?")[0]  # Remove query params like ?sample=100
                    artifact_name = f"{self.type}-{stage}"
            artifact_name = artifact_name or self.type

            tracking_metadata = tracker.log_artifact(self, artifact_name, self._used_artifacts)
            self.tracking = TrackingInfo(**tracking_metadata)
        else:
            if self.producer is None:
                self.producer = "local"

        # Write metadata.json atomically
        metadata_path = output_dir / "metadata.json"
        temp_path = output_dir / ".metadata.json.tmp"

        with open(temp_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        temp_path.rename(metadata_path)

        # Publish to registry if initialized (skip for wandb backend - tracker handles it)
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
            pass


class SFTDataArtifact(Artifact):
    """Packed SFT data artifact (output of SFT data_prep).

    Contains packed .npy files with tokenized and packed chat sequences.
    The path points to the output directory containing training.npy, validation.npy, etc.

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    - tokenizer_uri: URI of the tokenizer model (hf://models/...)
    """

    total_tokens: Annotated[int, Field(ge=0, description="Total tokens processed")]
    total_sequences: Annotated[int, Field(ge=0, description="Total sequences after packing")]
    elapsed_sec: Annotated[float, Field(default=0.0, ge=0, description="Processing time in seconds")]

    # Packing configuration
    pack_size: Annotated[int, Field(ge=1, description="Maximum tokens per packed sequence")]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]
    tokenizer_uri: Annotated[
        str | None, Field(default=None, description="URI of tokenizer model")
    ]

    def save(self, name: str | None = None) -> None:
        """Save artifact metadata to output directory."""
        output_dir = self.path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get tracker if active
        tracker = get_lineage_tracker()
        if tracker and tracker.is_active():
            if self.producer is None:
                self.producer = tracker.get_run_id() or "local"

            # Derive artifact name from semantic name if set
            artifact_name = name
            if artifact_name is None and self.name:
                parts = self.name.split("/")
                if len(parts) >= 2:
                    stage = parts[1].split("?")[0]
                    artifact_name = f"{self.type}-{stage}"
            artifact_name = artifact_name or self.type

            tracking_metadata = tracker.log_artifact(self, artifact_name, self._used_artifacts)
            self.tracking = TrackingInfo(**tracking_metadata)
        else:
            if self.producer is None:
                self.producer = "local"

        # Write metadata.json atomically
        metadata_path = output_dir / "metadata.json"
        temp_path = output_dir / ".metadata.json.tmp"

        with open(temp_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        temp_path.rename(metadata_path)

        # Publish to registry if initialized
        try:
            from nemotron.kit import get_config, is_initialized
            from nemotron.kit.registry import get_registry

            if is_initialized():
                config = get_config()
                if config and config.backend != "wandb":
                    registry = get_registry()
                    artifact_name = name or self.type
                    version = registry.publish(artifact_name, output_dir, metadata=self.metadata)
                    self._name = artifact_name
                    self._version = version.version
        except ImportError:
            pass


class PretrainDataArtifact(Artifact):
    """Pretrain data artifact (output of pretrain data_prep).

    Contains tokenized bin/idx files for pretraining.
    The path points to the output directory containing sharded data files.

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    - tokenizer_uri: URI of the tokenizer model (hf://models/...)
    """

    total_tokens: Annotated[int, Field(ge=0, description="Total tokens processed")]
    total_sequences: Annotated[int, Field(ge=0, description="Total documents processed")]
    elapsed_sec: Annotated[float, Field(default=0.0, ge=0, description="Processing time in seconds")]

    # Sharding configuration
    num_shards: Annotated[int, Field(ge=1, description="Number of output shards")]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]
    tokenizer_uri: Annotated[
        str | None, Field(default=None, description="URI of tokenizer model")
    ]

    def save(self, name: str | None = None) -> None:
        """Save artifact metadata to output directory."""
        output_dir = self.path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get tracker if active
        tracker = get_lineage_tracker()
        if tracker and tracker.is_active():
            if self.producer is None:
                self.producer = tracker.get_run_id() or "local"

            # Derive artifact name from semantic name if set
            artifact_name = name
            if artifact_name is None and self.name:
                parts = self.name.split("/")
                if len(parts) >= 2:
                    stage = parts[1].split("?")[0]
                    artifact_name = f"{self.type}-{stage}"
            artifact_name = artifact_name or self.type

            tracking_metadata = tracker.log_artifact(self, artifact_name, self._used_artifacts)
            self.tracking = TrackingInfo(**tracking_metadata)
        else:
            if self.producer is None:
                self.producer = "local"

        # Write metadata.json atomically
        metadata_path = output_dir / "metadata.json"
        temp_path = output_dir / ".metadata.json.tmp"

        with open(temp_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        temp_path.rename(metadata_path)

        # Publish to registry if initialized
        try:
            from nemotron.kit import get_config, is_initialized
            from nemotron.kit.registry import get_registry

            if is_initialized():
                config = get_config()
                if config and config.backend != "wandb":
                    registry = get_registry()
                    artifact_name = name or self.type
                    version = registry.publish(artifact_name, output_dir, metadata=self.metadata)
                    self._name = artifact_name
                    self._version = version.version
        except ImportError:
            pass


class SplitJsonlDataArtifact(Artifact):
    """Split JSONL data artifact (output of non-tokenized data_prep).

    Used for RL and other stages that output JSONL files without tokenization.
    The path points directly to the manifest.json file.

    Unlike DataBlendsArtifact, this does not track token counts since the
    data is not tokenized.

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    """

    total_sequences: Annotated[int, Field(ge=0, description="Total documents processed")]
    elapsed_sec: Annotated[float, Field(default=0.0, ge=0, description="Processing time in seconds")]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]

    def save(self, name: str | None = None) -> None:
        """Save artifact metadata to path's parent directory.

        Since SplitJsonlDataArtifact.path points to manifest.json (a file),
        metadata.json is written to the same directory.
        """
        # Use parent directory since self.path is a file (blend.json/manifest.json)
        output_dir = self.path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get tracker if active
        tracker = get_lineage_tracker()
        if tracker and tracker.is_active():
            if self.producer is None:
                self.producer = tracker.get_run_id() or "local"

            # Derive artifact name from semantic name if set
            artifact_name = name
            if artifact_name is None and self.name:
                parts = self.name.split("/")
                if len(parts) >= 2:
                    stage = parts[1].split("?")[0]
                    artifact_name = f"{self.type}-{stage}"
            artifact_name = artifact_name or self.type

            tracking_metadata = tracker.log_artifact(self, artifact_name, self._used_artifacts)
            self.tracking = TrackingInfo(**tracking_metadata)
        else:
            if self.producer is None:
                self.producer = "local"

        # Write metadata.json atomically in the parent directory
        metadata_path = output_dir / "metadata.json"
        temp_path = output_dir / ".metadata.json.tmp"

        with open(temp_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        temp_path.rename(metadata_path)

        # Publish to registry if initialized
        try:
            from nemotron.kit import get_config, is_initialized
            from nemotron.kit.registry import get_registry

            if is_initialized():
                config = get_config()
                if config and config.backend != "wandb":
                    registry = get_registry()
                    artifact_name = name or self.type
                    version = registry.publish(artifact_name, output_dir, metadata=self.metadata)
                    self._name = artifact_name
                    self._version = version.version
        except ImportError:
            pass


class ModelArtifact(Artifact):
    """Model checkpoint artifact (output of training).

    The path points to the checkpoint directory.
    """

    step: Annotated[int, Field(ge=0, description="Training step")]
    final_loss: Annotated[float | None, Field(default=None, description="Final training loss")]


# =============================================================================
# Artifact Input for CLI Commands
# =============================================================================


@dataclass
class ArtifactInput:
    """Defines an artifact input slot for a CLI command.

    Used with App.command() to specify named artifact inputs that can be
    provided via --art.<name> CLI arguments or stdin piping.

    Example:
        >>> app.command(
        ...     "pretrain",
        ...     TrainingConfig,
        ...     training_main,
        ...     artifacts={
        ...         "data": ArtifactInput(
        ...             default_name="DataBlendsArtifact-pretrain",
        ...             mappings={"path": "dataset.data_path"},
        ...         ),
        ...     },
        ... )

    Then users can run:
        nemotron nano3 pretrain --art.data v10
        nemotron nano3 pretrain --art.data DataBlendsArtifact-pretrain:latest
        nemotron nano3 pretrain --art.data romeyn/nemotron/DataBlendsArtifact-pretrain:v10
    """

    default_name: str
    """Default W&B artifact name (e.g., 'DataBlendsArtifact-pretrain').

    Used when only a version is provided (e.g., --art.data v10 or --art.data latest).
    """

    mappings: dict[str, str]
    """Mapping from artifact metadata fields to config field paths.

    Keys are field names from the artifact's metadata.json (e.g., 'path').
    Values are dot-separated config field paths (e.g., 'dataset.data_path').

    Example: {"path": "dataset.data_path"} means:
    - Load artifact metadata
    - Get metadata["path"] value
    - Set config.dataset.data_path = that value
    """


# =============================================================================
# Utilities
# =============================================================================


def apply_scale(count: int, scale: str) -> int:
    """Apply scale factor for fast iteration.

    Scale factors:
    - tiny: 1% (minimum 1, maximum 10,000)
    - small: 10%
    - medium: 30%
    - full: 100%

    Example:
        >>> apply_scale(100_000, "tiny")
        1000
        >>> apply_scale(2_000_000, "tiny")  # Capped at 10k
        10000
        >>> apply_scale(100_000, "full")
        100000
    """
    scale_factors = {
        "tiny": 0.01,
        "small": 0.10,
        "medium": 0.30,
        "full": 1.0,
    }

    if scale not in scale_factors:
        raise ValueError(f"Invalid scale: {scale}. Must be one of: {list(scale_factors.keys())}")

    scaled = int(count * scale_factors[scale])
    result = max(1, scaled)  # Ensure at least 1

    # Cap tiny at 10k for reasonable testing time
    if scale == "tiny":
        result = min(result, 10_000)

    return result


def print_step_complete(
    *args: dict[str, Artifact],
    title: str = "Complete",
    **artifacts: Artifact,
) -> None:
    """Print completion message with named artifacts.

    - Rich table to stderr (for humans)
    - JSON to stdout automatically when stdout is piped (for pipeline composition)

    Args:
        *args: Legacy dict syntax for backward compatibility
        title: Title for the completion message
        **artifacts: Named artifacts (e.g., data=artifact, model=checkpoint)

    Example:
        >>> print_step_complete(data=data_artifact)
        >>> print_step_complete(data=data_artifact, model=model_artifact)
    """
    # Support legacy dict syntax for backward compatibility
    if args and isinstance(args[0], dict):
        artifacts = args[0]

    # Auto-enable JSON output when stdout is piped
    output_json = not sys.stdout.isatty()

    # Output JSON to stdout when piped
    if output_json:
        # Output format: {"name": {"path": "...", "type": "..."}, ...}
        output = {name: json.loads(art.to_json()) for name, art in artifacts.items()}
        print(json.dumps(output), flush=True)

    # Output human-readable panel to stderr
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console(file=sys.stderr)

        panels = []
        for name, artifact in artifacts.items():
            # Build content lines - URI first for easy copy/paste
            lines = Text()
            lines.append(f"{artifact.art_path}\n\n", style="bold yellow")
            lines.append("Path:    ", style="dim")
            lines.append(f"{artifact.path.resolve()}\n", style="blue")

            # Add metrics if present
            if artifact.metrics:
                lines.append("Metrics: ", style="dim")
                metrics_parts = [
                    f"{k}={v:,.0f}" if v > 100 else f"{k}={v:.2f}"
                    for k, v in artifact.metrics.items()
                ]
                lines.append(", ".join(metrics_parts), style="green")

            panel = Panel(
                lines,
                title=f"[bold cyan]{name}[/bold cyan] [dim]({artifact.type})[/dim]",
                title_align="left",
                border_style="green",
            )
            panels.append(panel)

        # Print all panels
        console.print()
        for panel in panels:
            console.print(panel)

    except ImportError:
        # Fallback without rich
        sys.stderr.write(f"\nComplete {title}\n")
        sys.stderr.write("=" * 70 + "\n")
        for name, artifact in artifacts.items():
            sys.stderr.write(f"{name} ({artifact.type}):\n")
            sys.stderr.write(f"  {artifact.art_path}\n\n")
            sys.stderr.write(f"  Path: {artifact.path.resolve()}\n")
            if artifact.metrics:
                metrics_parts = [
                    f"{k}={v:,.0f}" if v > 100 else f"{k}={v:.2f}"
                    for k, v in artifact.metrics.items()
                ]
                sys.stderr.write(f"  Metrics: {', '.join(metrics_parts)}\n")
        sys.stderr.write("=" * 70 + "\n")
        sys.stderr.flush()
