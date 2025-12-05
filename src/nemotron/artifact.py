"""
Core artifact module for nemotron.

Provides the Artifact base class, tracking protocol, and utilities.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Self

from pydantic import BaseModel, Field, model_validator


class TrackingInfo(BaseModel):
    """Information about artifact tracking in external systems."""

    artifact_id: str | None = None
    artifact_type: str | None = None
    run_id: str | None = None
    url: str | None = None
    used_artifacts: list[str] = Field(default_factory=list)


class LineageTracker(Protocol):
    """Protocol for lineage tracking backends (W&B, MLflow, custom).

    Implement these 4 methods to integrate with any tracking system.
    """

    def is_active(self) -> bool:
        """Check if tracking is currently active."""
        ...

    def use_artifact(self, ref: str, artifact_type: str) -> Path:
        """Mark artifact as used (for lineage). Returns local path.

        Args:
            ref: Artifact reference (e.g., "team/project/data:v1")
            artifact_type: Type of artifact (e.g., "dataset", "checkpoint")

        Returns:
            Local path where artifact is available
        """
        ...

    def log_artifact(self, artifact: "Artifact", name: str, used_refs: list[str]) -> dict[str, Any]:
        """Log artifact to tracking backend.

        Args:
            artifact: The artifact to log
            name: Name for the artifact
            used_refs: List of artifact references that were used to create this

        Returns:
            Dictionary with tracking metadata (artifact_id, url, etc.)
        """
        ...

    def get_run_id(self) -> str | None:
        """Get current run ID."""
        ...


# Global tracker instance
_tracker: LineageTracker | None = None


def set_lineage_tracker(tracker: LineageTracker | None) -> None:
    """Set the artifact tracking backend.

    Examples:
        >>> from nemotron.trackers import WandbTracker
        >>> set_lineage_tracker(WandbTracker())  # Use W&B
        >>> set_lineage_tracker(None)  # Disable tracking
    """
    global _tracker
    _tracker = tracker


def get_lineage_tracker() -> LineageTracker | None:
    """Get the current artifact tracker."""
    return _tracker


class Artifact(BaseModel):
    """Base class for all step outputs.

    Every nemotron step produces an Artifact with validated fields and
    automatic save/load functionality.

    Example:
        >>> from pathlib import Path
        >>> from pydantic import Field
        >>>
        >>> class Dataset(Artifact):
        ...     num_examples: int = Field(gt=0)
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
    schema_version: int = Field(default=1, description="Artifact schema version")
    type: str = Field(default="artifact", description="Artifact type")
    path: Path = Field(description="Local filesystem path where artifact is stored")
    created_at: str = Field(
        default_factory=lambda: datetime.now().astimezone().isoformat(),
        description="ISO timestamp of creation",
    )
    producer: str | None = Field(default=None, description="Run ID or 'local'")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Numeric metrics (for logging)"
    )
    attrs: dict[str, Any] = Field(default_factory=dict, description="Additional attributes")
    tracking: TrackingInfo | None = Field(default=None, description="Tracking metadata")

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

    def save(self, artifact_name: str | None = None) -> None:
        """Save artifact to path/metadata.json (atomic write).

        If tracking is active, also logs to tracking backend.

        Args:
            artifact_name: Optional name for tracked artifact. Defaults to type.
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
            name = artifact_name or self.type
            tracking_metadata = tracker.log_artifact(self, name, self._used_artifacts)

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


def print_complete(
    artifacts: dict[str, Artifact],
    title: str = "Complete",
) -> None:
    """Print completion message.

    - Rich table to stderr (for humans)
    - JSON to stdout (for piping)

    Args:
        artifacts: Dictionary mapping step names to artifacts
        title: Title for the completion message

    Example:
        >>> dataset = Dataset(path=Path("/tmp/data"), num_examples=1000)
        >>> print_complete({"data_prep": dataset})
    """
    # Output JSON to stdout for piping
    if len(artifacts) == 1:
        # Single artifact - output its JSON for piping
        artifact = next(iter(artifacts.values()))
        print(artifact.to_json(), flush=True)
    else:
        # Multiple artifacts - output list
        output = {name: art.to_json() for name, art in artifacts.items()}
        print(json.dumps(output), flush=True)

    # Output human-readable table to stderr
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console(file=sys.stderr)
        table = Table(title=f"✓ {title}", show_header=True, header_style="bold cyan")

        table.add_column("Step", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Path", style="blue")
        table.add_column("Metrics", style="green")
        table.add_column("Tracked", style="yellow")

        for name, artifact in artifacts.items():
            metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in artifact.metrics.items())
            tracked_str = "✓" if artifact.tracking else "✗"

            table.add_row(
                name,
                artifact.type,
                str(artifact.path),
                metrics_str or "-",
                tracked_str,
            )

        console.print(table)

    except ImportError:
        # Fallback without rich
        sys.stderr.write(f"\n✓ {title}\n")
        sys.stderr.write("=" * 70 + "\n")
        for name, artifact in artifacts.items():
            sys.stderr.write(f"{name}:\n")
            sys.stderr.write(f"  Type: {artifact.type}\n")
            sys.stderr.write(f"  Path: {artifact.path}\n")
            if artifact.metrics:
                metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in artifact.metrics.items())
                sys.stderr.write(f"  Metrics: {metrics_str}\n")
            sys.stderr.write(f"  Tracked: {'✓' if artifact.tracking else '✗'}\n")
        sys.stderr.write("=" * 70 + "\n")
        sys.stderr.flush()
