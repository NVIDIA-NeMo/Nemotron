"""
Lineage tracking backends for nemotron.kit.

Provides the LineageTracker protocol and implementations for W&B and no-op tracking.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from nemotron.kit.artifact import Artifact


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

    def log_artifact(
        self, artifact: "Artifact", name: str, used_refs: list[str]
    ) -> dict[str, Any]:
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
        >>> from nemotron.kit import WandbTracker
        >>> set_lineage_tracker(WandbTracker())  # Use W&B
        >>> set_lineage_tracker(None)  # Disable tracking
    """
    global _tracker
    _tracker = tracker


def get_lineage_tracker() -> LineageTracker | None:
    """Get the current artifact tracker."""
    return _tracker


class WandbTracker:
    """Weights & Biases (W&B) tracking backend.

    Automatically logs artifacts and tracks lineage.

    Example:
        >>> import wandb
        >>> from nemotron.kit import set_lineage_tracker, WandbTracker
        >>>
        >>> wandb.init(project="my-project")
        >>> set_lineage_tracker(WandbTracker())
        >>> # Now all artifact.save() calls log to W&B
    """

    def __init__(self) -> None:
        """Initialize W&B tracker.

        Raises:
            ImportError: If wandb is not installed
        """
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandbTracker. Install it with: pip install wandb"
            )

    def is_active(self) -> bool:
        """Check if W&B run is active."""
        return self.wandb.run is not None

    def use_artifact(self, ref: str, artifact_type: str) -> Path:
        """Download artifact from W&B and mark as used.

        Args:
            ref: W&B artifact reference (e.g., "team/project/data:v1")
            artifact_type: Type of artifact

        Returns:
            Local path where artifact is downloaded
        """
        if not self.is_active():
            raise RuntimeError("No active W&B run. Call wandb.init() first.")

        # Use artifact (tracks lineage)
        artifact = self.wandb.run.use_artifact(ref, type=artifact_type)

        # Download to local cache
        artifact_dir = artifact.download()

        return Path(artifact_dir)

    def log_artifact(
        self, artifact: "Artifact", name: str, used_refs: list[str]
    ) -> dict[str, Any]:
        """Log artifact to W&B.

        Args:
            artifact: The artifact to log
            name: Name for the W&B artifact
            used_refs: List of artifact references that were used

        Returns:
            Dictionary with tracking metadata
        """
        if not self.is_active():
            raise RuntimeError("No active W&B run. Call wandb.init() first.")

        # Create W&B artifact
        wb_artifact = self.wandb.Artifact(
            name=name,
            type=artifact.type,
            metadata={
                "created_at": artifact.created_at,
                **artifact.metadata,
            },
        )

        # Add the artifact directory
        wb_artifact.add_dir(str(artifact.path))

        # Log metrics to run
        if artifact.metrics:
            self.wandb.log(artifact.metrics)

        # Mark dependencies (for lineage)
        for ref in used_refs:
            try:
                # Try to create artifact dependency
                used_artifact = self.wandb.Api().artifact(ref)
                wb_artifact.use_artifact(used_artifact)
            except Exception:
                # If we can't find the artifact, just skip lineage tracking
                pass

        # Log to W&B
        logged = self.wandb.run.log_artifact(wb_artifact)

        # Wait for artifact to be logged (to get ID)
        logged.wait()

        return {
            "artifact_id": f"{logged.entity}/{logged.project}/{logged.name}:{logged.version}",
            "artifact_type": artifact.type,
            "run_id": self.wandb.run.id,
            "url": logged.url if hasattr(logged, "url") else None,
            "used_artifacts": used_refs,
        }

    def get_run_id(self) -> str | None:
        """Get current W&B run ID."""
        return self.wandb.run.id if self.wandb.run else None


class NoOpTracker:
    """No-op tracker that does nothing.

    Useful for testing or explicitly disabling tracking.

    Example:
        >>> from nemotron.kit import set_lineage_tracker, NoOpTracker
        >>>
        >>> set_lineage_tracker(NoOpTracker())  # Disable tracking
    """

    def is_active(self) -> bool:
        """Always returns False."""
        return False

    def use_artifact(self, ref: str, artifact_type: str) -> Path:
        """Raises error - cannot use artifacts without tracking."""
        raise RuntimeError("NoOpTracker cannot load artifacts")

    def log_artifact(
        self, artifact: "Artifact", name: str, used_refs: list[str]
    ) -> dict[str, Any]:
        """Returns empty metadata."""
        return {
            "artifact_id": None,
            "artifact_type": artifact.type,
            "run_id": None,
            "url": None,
            "used_artifacts": [],
        }

    def get_run_id(self) -> str | None:
        """Always returns None."""
        return None
