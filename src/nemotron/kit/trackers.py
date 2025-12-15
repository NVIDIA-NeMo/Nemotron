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

"""
Lineage tracking backends for nemotron.kit.

Provides the LineageTracker protocol and implementations for W&B and no-op tracking.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

if TYPE_CHECKING:
    from nemotron.kit.artifact import Artifact


@dataclass
class InputDatasetInfo:
    """Metadata for an input dataset to register as a W&B artifact.

    This captures all relevant metadata from the blend specification and
    discovery phase so input datasets can be properly tracked for lineage.

    Attributes:
        uri: The source URI (hf://, s3://, file://)
        name: Dataset name from blend specification
        weight: Weight in the blend (default: 1.0)
        split: HuggingFace split name
        subset: HuggingFace config/subset name
        text_field: Field containing text to tokenize
        num_rows: Number of rows (from HF metadata)
        size_bytes: Size in bytes (from HF metadata)
        num_files: Number of input files discovered
    """

    uri: str
    name: str | None = None
    weight: float = 1.0
    split: str | None = None
    subset: str | None = None
    text_field: str = "text"
    # Discovered metadata from planning phase
    num_rows: int | None = None
    size_bytes: int | None = None
    num_files: int | None = None


def _uri_to_artifact_name(uri: str, subset: str | None = None) -> str:
    """Convert a URI to a valid W&B artifact name.

    W&B artifact names can only contain alphanumeric characters, dashes,
    underscores, and dots. This sanitizes URIs for use as artifact names.

    Args:
        uri: Source URI (e.g., "hf://nvidia/Nemotron-CC", "s3://bucket/key")
        subset: Optional subset name to append (for HuggingFace datasets with subsets)

    Returns:
        Sanitized artifact name
    """
    import re

    # Remove protocol prefix
    if "://" in uri:
        name = uri.split("://", 1)[1]
    else:
        name = uri

    # Append subset if provided (important for same-path different-subset datasets)
    if subset:
        name = f"{name}-{subset}"

    # Replace path separators and invalid characters with dashes
    name = re.sub(r"[/\\:@#?&=+%]", "-", name)

    # Remove leading/trailing dashes and collapse multiple dashes
    name = re.sub(r"-+", "-", name).strip("-")

    # Ensure it starts with alphanumeric
    if name and not name[0].isalnum():
        name = "dataset-" + name

    # Truncate to max length (128 chars for W&B)
    return name[:128] if name else "dataset"


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

    def _register_input_datasets(
        self, datasets: list[InputDatasetInfo]
    ) -> tuple[list[str], list[Any]]:
        """Register input datasets as W&B artifacts for lineage tracking.

        Each dataset is registered with its full metadata so users can trace
        data provenance from the raw source through to the final processed artifact.

        For external datasets (HuggingFace, S3, etc.), we use run.use_artifact()
        with an Artifact object. This both creates the artifact (if needed) and
        marks it as an input to this run.

        Args:
            datasets: List of input dataset info with metadata

        Returns:
            Tuple of (artifact_refs, artifact_objects) where:
            - artifact_refs: List of artifact references (entity/project/name:version)
            - artifact_objects: List of W&B artifact objects for use in lineage
        """
        artifact_refs = []
        artifact_objects = []

        for ds in datasets:
            # Create artifact name from URI and subset (sanitize for wandb)
            artifact_name = _uri_to_artifact_name(ds.uri, subset=ds.subset)

            # Build metadata dict with all available dataset info
            metadata: dict[str, Any] = {
                "source_uri": ds.uri,
                "raw": True,
            }
            if ds.name:
                metadata["name"] = ds.name
            if ds.weight != 1.0:
                metadata["weight"] = ds.weight
            if ds.split:
                metadata["split"] = ds.split
            if ds.subset:
                metadata["subset"] = ds.subset
            if ds.text_field != "text":
                metadata["text_field"] = ds.text_field
            # Include discovered metadata from planning phase
            if ds.num_rows is not None:
                metadata["num_rows"] = ds.num_rows
            if ds.size_bytes is not None:
                metadata["size_bytes"] = ds.size_bytes
            if ds.num_files is not None:
                metadata["num_files"] = ds.num_files

            # Check if artifact already exists in W&B
            try:
                # Try to get existing artifact by name
                api = self.wandb.Api()
                project_path = f"{self.wandb.run.entity}/{self.wandb.run.project}"
                existing = api.artifact(f"{project_path}/{artifact_name}:latest")
                # Artifact exists - use it by reference string
                artifact_ref = (
                    f"{existing.entity}/{existing.project}/{existing.name}:{existing.version}"
                )
                self.wandb.run.use_artifact(artifact_ref)
                artifact_refs.append(artifact_ref)
                artifact_objects.append(existing)
            except Exception:
                # Artifact doesn't exist - create and use it
                # Build description with clickable HuggingFace link
                uri = to_wandb_uri(ds.uri)
                description = f"Source: {uri}"

                input_artifact = self.wandb.Artifact(
                    name=artifact_name,
                    type="dataset",
                    description=description,
                    metadata=metadata,
                )

                # Add reference to actual data location (external HF/S3/file)
                try:
                    input_artifact.add_reference(uri, name="data", checksum=False)
                except Exception:
                    pass  # Some URIs may not support references

                # use_artifact with Artifact object both creates AND marks as input
                used = self.wandb.run.use_artifact(input_artifact)
                artifact_ref = f"{used.entity}/{used.project}/{used.name}:{used.version}"
                artifact_refs.append(artifact_ref)
                artifact_objects.append(used)

        return artifact_refs, artifact_objects

    def _register_tokenizer(self, tokenizer_uri: str) -> tuple[str | None, Any | None]:
        """Register a tokenizer as a W&B artifact for lineage tracking.

        Args:
            tokenizer_uri: URI to the tokenizer (HuggingFace URL or file path)

        Returns:
            Tuple of (artifact_ref, artifact_object) or (None, None) if registration fails
        """
        import re

        # Extract tokenizer name from URI for artifact naming
        # e.g., "https://huggingface.co/meta-llama/Llama-3.2-1B" -> "meta-llama-Llama-3.2-1B"
        if "huggingface.co/" in tokenizer_uri:
            # Extract the model path after huggingface.co/
            match = re.search(r"huggingface\.co/([^/]+/[^/]+)", tokenizer_uri)
            if match:
                artifact_name = match.group(1).replace("/", "-")
            else:
                artifact_name = "tokenizer"
        elif tokenizer_uri.startswith("file://"):
            # Local path - use the last directory component
            path_part = tokenizer_uri[7:]  # Remove "file://"
            artifact_name = Path(path_part).name or "tokenizer"
        else:
            artifact_name = "tokenizer"

        # Sanitize artifact name
        artifact_name = re.sub(r"[^a-zA-Z0-9._-]", "-", artifact_name)
        artifact_name = re.sub(r"-+", "-", artifact_name).strip("-")
        artifact_name = artifact_name[:128] if artifact_name else "tokenizer"

        # Build metadata
        metadata: dict[str, Any] = {
            "source_uri": tokenizer_uri,
            "type": "tokenizer",
        }

        # Check if artifact already exists
        try:
            api = self.wandb.Api()
            project_path = f"{self.wandb.run.entity}/{self.wandb.run.project}"
            existing = api.artifact(f"{project_path}/{artifact_name}:latest")
            # Artifact exists - use it by reference
            artifact_ref = (
                f"{existing.entity}/{existing.project}/{existing.name}:{existing.version}"
            )
            self.wandb.run.use_artifact(artifact_ref)
            return artifact_ref, existing
        except Exception:
            # Artifact doesn't exist - create and use it
            # Build description with clickable link
            description = f"Source: {tokenizer_uri}"

            tokenizer_artifact = self.wandb.Artifact(
                name=artifact_name,
                type="tokenizer",
                description=description,
                metadata=metadata,
            )

            # Add reference to the tokenizer location
            try:
                tokenizer_artifact.add_reference(tokenizer_uri, name="tokenizer", checksum=False)
            except Exception:
                pass  # Some URIs may not support references

            # use_artifact creates and marks as input
            try:
                used = self.wandb.run.use_artifact(tokenizer_artifact)
                artifact_ref = f"{used.entity}/{used.project}/{used.name}:{used.version}"
                return artifact_ref, used
            except Exception:
                return None, None

    def log_artifact(self, artifact: "Artifact", name: str, used_refs: list[str]) -> dict[str, Any]:
        """Log artifact to W&B using URI references for lineage tracking.

        For DataBlendsArtifact, registers input datasets as separate artifacts
        with "raw" alias and creates lineage from inputs to this output.

        Args:
            artifact: The artifact to log
            name: Name for the W&B artifact
            used_refs: List of artifact references that were used

        Returns:
            Dictionary with tracking metadata
        """
        if not self.is_active():
            raise RuntimeError("No active W&B run. Call wandb.init() first.")

        # Check if artifact has source datasets (DataBlendsArtifact)
        source_datasets = getattr(artifact, "source_datasets", None)
        tokenizer_uri = getattr(artifact, "tokenizer_uri", None)

        # Register input datasets as separate artifacts for lineage
        input_artifact_refs: list[str] = []
        input_artifact_objects: list[Any] = []
        if source_datasets:
            # source_datasets can be list[InputDatasetInfo] or list[str]
            if source_datasets and isinstance(source_datasets[0], InputDatasetInfo):
                input_artifact_refs, input_artifact_objects = self._register_input_datasets(
                    source_datasets
                )
            else:
                # Legacy: list[str] - convert to InputDatasetInfo
                legacy_datasets = [InputDatasetInfo(uri=uri) for uri in source_datasets]
                input_artifact_refs, input_artifact_objects = self._register_input_datasets(
                    legacy_datasets
                )

        # Register tokenizer as an input artifact for lineage
        tokenizer_artifact_ref: str | None = None
        tokenizer_artifact_obj: Any | None = None
        if tokenizer_uri:
            tokenizer_artifact_ref, tokenizer_artifact_obj = self._register_tokenizer(tokenizer_uri)
            if tokenizer_artifact_ref:
                input_artifact_refs.append(tokenizer_artifact_ref)
            if tokenizer_artifact_obj:
                input_artifact_objects.append(tokenizer_artifact_obj)

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

        if source_datasets or tokenizer_uri:
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

            # For DataBlendsArtifact, add blend.json directly so it's viewable in W&B UI
            # Don't add output reference - the lineage to input datasets is sufficient
            artifact_path = artifact.path
            if artifact_path.is_file() and artifact_path.name == "blend.json":
                # Add only the blend.json file (small, viewable in UI)
                wb_artifact.add_file(str(artifact_path), name="blend.json")
            elif artifact.type == "PretrainBlendsArtifact":
                # For PretrainBlendsArtifact, add blend.json explicitly
                # The blend_path is stored in metadata
                blend_path = artifact.metadata.get("blend_path")
                if blend_path and Path(blend_path).exists():
                    wb_artifact.add_file(str(blend_path), name="blend.json")
                # Also add metadata.json for artifact loading
                metadata_path = artifact_path / "metadata.json"
                if metadata_path.exists():
                    wb_artifact.add_file(str(metadata_path), name="metadata.json")
            else:
                # Generic artifact - add directory reference
                # Use checksum=False to avoid digest mismatch errors when files change
                # between data prep runs (the path is what matters for lineage)
                if artifact_path.is_file():
                    artifact_path = artifact_path.parent
                output_uri = f"file://{artifact_path.resolve()}"
                try:
                    wb_artifact.add_reference(
                        output_uri,
                        name="output",
                        checksum=False,
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
        # First, add explicit used_refs passed by caller
        for ref in used_refs:
            try:
                dep_artifact = self.wandb.Api().artifact(ref)
                wb_artifact.use_artifact(dep_artifact)
            except Exception:
                pass

        # Then add input artifacts we just registered (use objects directly, not API lookup)
        # This creates the lineage arrows: input_artifact -> output_artifact
        for input_art in input_artifact_objects:
            try:
                wb_artifact.use_artifact(input_art)
            except Exception:
                pass

        # Log to W&B
        logged = self.wandb.run.log_artifact(wb_artifact)

        # Wait for artifact to be logged (to get ID)
        logged.wait()

        # Add experiment_id as alias for cross-task artifact discovery
        experiment_id = os.environ.get("NEMO_EXPERIMENT_ID")
        if experiment_id:
            logged.aliases.append(experiment_id)
            logged.save()

        # Collect all tracked URIs
        tracked_uris: list[str] = []
        if source_datasets:
            for ds in source_datasets:
                if isinstance(ds, InputDatasetInfo):
                    tracked_uris.append(ds.uri)
                else:
                    tracked_uris.append(ds)
        if tokenizer_uri:
            tracked_uris.append(tokenizer_uri)

        return {
            "artifact_id": f"{logged.entity}/{logged.project}/{logged.name}:{logged.version}",
            "artifact_type": artifact.type,
            "run_id": self.wandb.run.id,
            "url": logged.url if hasattr(logged, "url") else None,
            "used_artifacts": used_refs,
            "source_uris": tracked_uris,
            "input_artifact_refs": input_artifact_refs,
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

    def log_artifact(self, artifact: "Artifact", name: str, used_refs: list[str]) -> dict[str, Any]:
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
