"""
Lineage tracking backends for nemotron.kit.

Provides the LineageTracker protocol and implementations for W&B and no-op tracking.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

if TYPE_CHECKING:
    from nemotron.kit.artifact import Artifact


def to_wandb_uri(path: str) -> str:
    """Convert a data path to a W&B-compatible reference URI.

    Args:
        path: Data path in various formats:
            - hf://repo/name -> https://huggingface.co/datasets/repo/name
            - s3://bucket/key -> s3://bucket/key (unchanged)
            - gs://bucket/key -> gs://bucket/key (unchanged)
            - /local/path -> file:///local/path

    Returns:
        W&B-compatible URI for add_reference()
    """
    if path.startswith("hf://"):
        # HuggingFace dataset: hf://nvidia/Nemotron-CC -> https://huggingface.co/datasets/nvidia/Nemotron-CC
        repo = path[5:]  # Remove "hf://"
        return f"https://huggingface.co/datasets/{repo}"
    elif path.startswith("s3://") or path.startswith("gs://"):
        # Cloud storage URIs are already compatible
        return path
    elif path.startswith("http://") or path.startswith("https://"):
        # HTTP URLs are already compatible
        return path
    elif path.startswith("file://"):
        # Already a file URI
        return path
    else:
        # Local path - convert to file:// URI
        # Ensure absolute path
        abs_path = Path(path).resolve()
        return f"file://{abs_path}"


def tokenizer_to_uri(model: str, revision: str | None = None) -> str:
    """Convert a tokenizer model name/path to a reference URI.

    Args:
        model: Tokenizer model name (e.g., "meta-llama/Llama-3.2-1B") or local path
        revision: Optional git revision/commit SHA for HuggingFace models

    Returns:
        URI for the tokenizer
    """
    if "/" in model and not model.startswith("/"):
        # HuggingFace model name
        base_url = f"https://huggingface.co/{quote(model, safe='/')}"
        if revision:
            return f"{base_url}/tree/{revision}"
        return base_url
    else:
        # Local path
        abs_path = Path(model).resolve()
        return f"file://{abs_path}"


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
        """Log artifact to W&B using URI references for lineage tracking.

        For DataBlendsArtifact, adds references to:
        - Source datasets (from artifact.source_datasets)
        - Tokenizer model (from artifact.tokenizer_uri)
        - The output blend.json itself

        Args:
            artifact: The artifact to log
            name: Name for the W&B artifact
            used_refs: List of artifact references that were used

        Returns:
            Dictionary with tracking metadata
        """
        if not self.is_active():
            raise RuntimeError("No active W&B run. Call wandb.init() first.")

        # Build metadata including source URIs if present
        metadata = {
            "created_at": artifact.created_at,
            **artifact.metadata,
        }

        # Create W&B artifact
        wb_artifact = self.wandb.Artifact(
            name=name,
            type=artifact.type,
            metadata=metadata,
        )

        # Check if artifact has source URIs (DataBlendsArtifact)
        source_datasets = getattr(artifact, "source_datasets", None)
        tokenizer_uri = getattr(artifact, "tokenizer_uri", None)

        if source_datasets or tokenizer_uri:
            # Use URI references for lineage tracking
            # Add reference to each source dataset
            if source_datasets:
                for i, dataset_path in enumerate(source_datasets):
                    uri = to_wandb_uri(dataset_path)
                    # Use checksum=False for remote URIs that may not support checksumming
                    checksum = uri.startswith("file://")
                    try:
                        wb_artifact.add_reference(
                            uri,
                            name=f"source_dataset_{i}",
                            checksum=checksum,
                        )
                    except Exception:
                        # Some URIs may not be supported, log and continue
                        pass

            # Add reference to tokenizer
            if tokenizer_uri:
                try:
                    wb_artifact.add_reference(
                        tokenizer_uri,
                        name="tokenizer",
                        checksum=False,  # HuggingFace models don't support checksumming
                    )
                except Exception:
                    pass

            # Add the actual output directory as a reference (not uploaded)
            artifact_path = artifact.path
            if artifact_path.is_file():
                artifact_path = artifact_path.parent
            output_uri = f"file://{artifact_path.resolve()}"
            try:
                wb_artifact.add_reference(
                    output_uri,
                    name="output",
                    checksum=True,
                )
            except Exception:
                # Fallback to add_dir if reference fails
                wb_artifact.add_dir(str(artifact_path))
        else:
            # No source URIs - fallback to add_dir for backward compatibility
            artifact_path = artifact.path
            if artifact_path.is_file():
                artifact_path = artifact_path.parent
            wb_artifact.add_dir(str(artifact_path))

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

        # Collect all tracked URIs
        tracked_uris = list(source_datasets or [])
        if tokenizer_uri:
            tracked_uris.append(tokenizer_uri)

        return {
            "artifact_id": f"{logged.entity}/{logged.project}/{logged.name}:{logged.version}",
            "artifact_type": artifact.type,
            "run_id": self.wandb.run.id,
            "url": logged.url if hasattr(logged, "url") else None,
            "used_artifacts": used_refs,
            "source_uris": tracked_uris,
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
