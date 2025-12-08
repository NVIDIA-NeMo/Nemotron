"""
Core artifact module for nemotron.kit.

Provides the Artifact base class, TrackingInfo, and utilities.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field, model_validator

from nemotron.kit.trackers import get_lineage_tracker


class TrackingInfo(BaseModel):
    """Information about artifact tracking in external systems."""

    artifact_id: str | None = None
    artifact_type: str | None = None
    run_id: str | None = None
    url: str | None = None
    used_artifacts: Annotated[list[str], Field(default_factory=list)]


class Artifact(BaseModel):
    """Base class for all step outputs.

    Every nemotron step produces an Artifact with validated fields and
    automatic save/load functionality.

    Example:
        >>> from pathlib import Path
        >>> from typing import Annotated
        >>> from pydantic import Field
        >>> from nemotron.kit import Artifact
        >>>
        >>> class Dataset(Artifact):
        ...     num_examples: Annotated[int, Field(gt=0)]
        ...     train_path: Path
        >>>
        >>> dataset = Dataset(
        ...     path=Path("/tmp/data"),
        ...     num_examples=1000,
        ...     train_path=Path("/tmp/data/train.parquet"),
        ...     metrics={"num_examples": 1000}
        ... )
        >>> dataset.save()
        >>> loaded = Dataset.load(path=Path("/tmp/data"))
    """

    # Core fields (automatically included)
    schema_version: Annotated[int, Field(default=1, description="Artifact schema version")]
    type: Annotated[str, Field(default="artifact", description="Artifact type")]
    path: Annotated[Path, Field(description="Local filesystem path where artifact is stored")]
    created_at: Annotated[
        str,
        Field(
            default_factory=lambda: datetime.now().astimezone().isoformat(),
            description="ISO timestamp of creation",
        ),
    ]
    producer: Annotated[str | None, Field(default=None, description="Run ID or 'local'")]
    metrics: Annotated[
        dict[str, float], Field(default_factory=dict, description="Numeric metrics (for logging)")
    ]
    attrs: Annotated[dict[str, Any], Field(default_factory=dict, description="Additional attributes")]
    tracking: Annotated[TrackingInfo | None, Field(default=None, description="Tracking metadata")]
    name: Annotated[str | None, Field(default=None, description="Semantic artifact name (e.g., nano3/pretrain/data)")]

    # Registry metadata (set after publish)
    _name: str | None = None
    _version: int | None = None

    # Track which artifacts were used to create this one (for lineage)
    _used_artifacts: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data: Any) -> Any:
        """Set default values for type field based on class name."""
        if isinstance(data, dict):
            if "type" not in data or data["type"] == "artifact":
                # Use the actual class name (e.g., "Dataset", "Checkpoint")
                data["type"] = cls.__name__.lower()
        return data

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
        """Save artifact to path/metadata.json (atomic write).

        If tracking is active, also logs to tracking backend.
        If art.init() was called, publishes to registry.

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

        # Publish to registry if initialized
        try:
            from nemotron.kit import is_initialized
            from nemotron.kit.registry import get_registry

            if is_initialized():
                registry = get_registry()
                artifact_name = name or self.type
                version = registry.publish(artifact_name, self.path)
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

        Example:
            >>> # From local path
            >>> dataset = Dataset.load(path=Path("/tmp/data"))
            >>>
            >>> # From tracked artifact
            >>> dataset = Dataset.load(tracked_artifact="team/project/data:v1")
            >>>
            >>> # From stdin (when piping)
            >>> dataset = Dataset.load()  # Reads from stdin
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

        Example:
            >>> dataset = Dataset.from_uri("art://my-dataset:v1")
            >>> dataset = Dataset.from_uri("art://my-dataset:latest")
        """
        from nemotron.kit.registry import get_registry

        registry = get_registry()

        # Parse URI: art://name:version or art://name
        if not uri.startswith("art://"):
            raise ValueError(f"Invalid art:// URI: {uri}")

        uri_path = uri[6:]  # Remove "art://"

        # Parse name and version
        if ":" in uri_path:
            name, version_str = uri_path.rsplit(":", 1)
            if version_str == "latest":
                version = None
            elif version_str.startswith("v"):
                version = int(version_str[1:])
            else:
                version = int(version_str)
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
        >>> # New syntax (preferred)
        >>> print_step_complete(data=data_artifact)
        >>> print_step_complete(data=data_artifact, model=model_artifact)
        >>>
        >>> # Legacy syntax (still supported)
        >>> print_step_complete({"data_prep": dataset})
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
        from rich.console import Console, Group
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
                metrics_parts = [f"{k}={v:,.0f}" if v > 100 else f"{k}={v:.2f}"
                                for k, v in artifact.metrics.items()]
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
                metrics_parts = [f"{k}={v:,.0f}" if v > 100 else f"{k}={v:.2f}"
                                for k, v in artifact.metrics.items()]
                sys.stderr.write(f"  Metrics: {', '.join(metrics_parts)}\n")
        sys.stderr.write("=" * 70 + "\n")
        sys.stderr.flush()
