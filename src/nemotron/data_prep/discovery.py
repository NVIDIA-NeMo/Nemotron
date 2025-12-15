"""Input file discovery for various sources (HuggingFace, S3, GCS, local)."""

import functools
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fsspec import AbstractFileSystem

from nemotron.data_prep.config import DatasetConfig, FileInfo

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=32)
def _get_split_pattern(split: str) -> re.Pattern:
    """Get compiled regex pattern for split matching. Cached for reuse."""
    return re.compile(rf"(^|/){re.escape(split)}(-|/|$)")


@dataclass
class DatasetMetadata:
    """Metadata about a dataset from HuggingFace Hub."""

    num_rows: int | None = None
    size_bytes: int | None = None
    num_rows_str: str | None = None
    size_str: str | None = None


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} EB"


def _format_rows(num_rows: int) -> str:
    """Format row count into human-readable string."""
    if num_rows >= 1_000_000_000:
        return f"{num_rows / 1_000_000_000:.2f}B"
    elif num_rows >= 1_000_000:
        return f"{num_rows / 1_000_000:.1f}M"
    elif num_rows >= 1_000:
        return f"{num_rows / 1_000:.1f}K"
    else:
        return str(num_rows)


def fetch_hf_dataset_metadata(
    repo_id: str,
    subset: str | None = None,
    split: str | None = None,
) -> DatasetMetadata:
    """
    Fetch dataset metadata from HuggingFace Dataset Viewer API.

    Args:
        repo_id: HuggingFace dataset repo ID (e.g., "nvidia/Nemotron-CC-Math-v1")
        subset: Optional subset/config name
        split: Optional split name

    Returns:
        DatasetMetadata with size and row count info
    """
    import urllib.request
    import json

    try:
        # Build API URL (don't filter by config - do it client-side)
        url = f"https://datasets-server.huggingface.co/size?dataset={repo_id}"

        # Create request with optional authentication for gated datasets
        request = urllib.request.Request(url)

        # Try to get HuggingFace token for authentication
        # Check HF_TOKEN env var first (for remote execution), then local cache
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
            except Exception:
                pass  # huggingface_hub not installed or no token
        if token:
            request.add_header("Authorization", f"Bearer {token}")

        # Fetch metadata
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())

        # Extract size info
        size_info = data.get("size", {})

        # If subset specified, look for that config
        if subset:
            configs = size_info.get("configs", [])
            for cfg in configs:
                if cfg.get("config") == subset:
                    # If split specified, look in splits list
                    if split:
                        splits = size_info.get("splits", [])
                        for sp in splits:
                            if sp.get("config") == subset and sp.get("split") == split:
                                sp_bytes = sp.get("num_bytes_parquet_files") or sp.get("num_bytes")
                                return DatasetMetadata(
                                    num_rows=sp.get("num_rows"),
                                    size_bytes=sp_bytes,
                                    num_rows_str=_format_rows(sp["num_rows"]) if sp.get("num_rows") else None,
                                    size_str=_format_size(sp_bytes) if sp_bytes else None,
                                )
                    # No split specified, use config totals directly
                    cfg_rows = cfg.get("num_rows")
                    cfg_bytes = cfg.get("num_bytes_parquet_files") or cfg.get("num_bytes")
                    return DatasetMetadata(
                        num_rows=cfg_rows,
                        size_bytes=cfg_bytes,
                        num_rows_str=_format_rows(cfg_rows) if cfg_rows else None,
                        size_str=_format_size(cfg_bytes) if cfg_bytes else None,
                    )

        # No subset, use dataset totals
        dataset_info = size_info.get("dataset", {})
        num_rows = dataset_info.get("num_rows")
        num_bytes = dataset_info.get("num_bytes_parquet_files") or dataset_info.get("num_bytes")

        return DatasetMetadata(
            num_rows=num_rows,
            size_bytes=num_bytes,
            num_rows_str=_format_rows(num_rows) if num_rows else None,
            size_str=_format_size(num_bytes) if num_bytes else None,
        )

    except Exception as e:
        logger.debug(f"Failed to fetch HF metadata for {repo_id}: {e}")
        return DatasetMetadata()


def get_dataset_metadata(config: DatasetConfig) -> DatasetMetadata:
    """
    Get metadata for a dataset config.

    Currently only supports HuggingFace datasets.
    """
    if config.path.startswith("hf://"):
        repo_id = config.path[5:]  # Remove hf:// prefix
        return fetch_hf_dataset_metadata(repo_id, config.subset, config.split)

    # For non-HF datasets, return empty metadata
    return DatasetMetadata()


def discover_input_files(
    config: DatasetConfig,
    fs: AbstractFileSystem,
) -> list[FileInfo]:
    """
    Discover and enumerate input files with local path resolution.

    Automatically detects source type from path prefix:
    - hf:// -> HuggingFace Hub
    - s3:// -> S3
    - gs:// -> GCS
    - Otherwise -> Local filesystem
    """
    path = config.path

    if path.startswith("hf://"):
        return discover_hf_files(config)
    else:
        return discover_filesystem_files(config, fs)


def discover_hf_files(config: DatasetConfig) -> list[FileInfo]:
    """
    Discover parquet files from HuggingFace dataset.

    Stores HF identity for deferred download - files will be downloaded
    inside Ray actors to support multi-node execution.
    """
    from huggingface_hub import HfApi

    hf_path = config.path[5:]  # Remove hf:// prefix

    # Get token from env var (for remote execution) or local cache
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass  # No token available

    api = HfApi(token=token)

    # Resolve revision to SHA for determinism
    dataset_info = api.dataset_info(hf_path, revision=config.revision)
    resolved_revision = dataset_info.sha

    # Find parquet files for the split
    files = []
    split = config.split

    for sibling in dataset_info.siblings:
        filename = sibling.rfilename

        # Match data files: parquet or jsonl
        # Valid extensions: .parquet, .jsonl, .json
        is_data_file = (
            filename.endswith(".parquet")
            or filename.endswith(".jsonl")
            or filename.endswith(".json")
        )
        if not is_data_file:
            continue

        # Check if subset is specified and matches
        # HuggingFace datasets use various patterns:
        # - data/{subset}-{split}-00000.parquet (subset as filename prefix)
        # - {subset}/{split}-00000.parquet (subset as directory)
        # - data/{subset}/{split}-00000.parquet (subset as subdirectory)
        # - data/{subset}.jsonl (subset as filename without split)
        if config.subset:
            subset = config.subset
            # Get the base filename without extension for matching
            base_filename = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            # Check various patterns where subset can appear
            if not (
                f"/{subset}/" in filename  # subset as directory
                or filename.startswith(f"{subset}/")  # subset at start as directory
                or f"/{subset}-" in filename  # subset as filename prefix after path
                or f"/{subset}_" in filename  # subset with underscore separator
                or filename.startswith(f"{subset}-")  # subset at start of filename
                or filename.startswith(f"{subset}_")  # subset with underscore at start
                or f"data/{subset}-" in filename  # common HF pattern: data/{subset}-
                or f"data/{subset}_" in filename  # common HF pattern with underscore
                or base_filename == subset  # exact match: data/{subset}.jsonl
                or f"/{subset}." in filename  # subset before extension: data/{subset}.jsonl
            ):
                continue

        # Check if split matches as a proper component (skip if no split specified)
        if split and not _matches_split(filename, split):
            continue

        # Store HF identity - DO NOT download here (supports multi-node)
        files.append(
            FileInfo(
                path=filename,
                local_path=None,  # Will be resolved inside actor
                size=sibling.size or 0,
                etag=sibling.lfs.sha256 if sibling.lfs else None,
                hf_repo_id=hf_path,
                hf_filename=filename,
                hf_revision=resolved_revision,
            )
        )

    return sorted(files, key=lambda f: f.path)


def _matches_split(filename: str, split: str) -> bool:
    """
    Check if split matches as a proper path/filename component.

    Valid matches:
    - "train-00000-of-00001.parquet" (split prefix with hyphen)
    - "data/train-00000.parquet" (split after path separator)
    - "train/file.parquet" (split as directory)

    Invalid matches:
    - "training-data.parquet" (split is substring)
    - "retrain-00000.parquet" (split is suffix of word)
    """
    # Use cached compiled pattern for efficiency
    return bool(_get_split_pattern(split).search(filename))


def discover_filesystem_files(
    config: DatasetConfig,
    fs: AbstractFileSystem,
) -> list[FileInfo]:
    """Discover files from filesystem (local/S3/GCS)."""
    path = config.path

    # Expand glob
    if "*" in path:
        file_paths = sorted(fs.glob(path))
    elif fs.isdir(path):
        all_files = fs.listdir(path, detail=False)
        file_paths = sorted(
            [
                f"{path.rstrip('/')}/{f}" if isinstance(f, str) else f["name"]
                for f in all_files
                if (isinstance(f, str) and f.endswith((".parquet", ".jsonl", ".json")))
                or (
                    isinstance(f, dict)
                    and f.get("name", "").endswith((".parquet", ".jsonl", ".json"))
                )
            ]
        )
    else:
        file_paths = [path]

    files = []
    for file_path in file_paths:
        try:
            info = fs.info(file_path)
            # Extract mtime for local files (stronger fingerprint)
            mtime = None
            if "mtime" in info:
                mtime = info["mtime"]
            elif "LastModified" in info:
                # S3 returns datetime
                mtime = info["LastModified"].timestamp() if hasattr(info["LastModified"], "timestamp") else None

            # Extract version_id for S3/GCS versioned objects
            version_id = info.get("VersionId") or info.get("version_id")

            files.append(
                FileInfo(
                    path=file_path,
                    local_path=file_path,  # Same for local/cloud (fsspec handles)
                    size=info.get("size", 0),
                    etag=info.get("ETag") or info.get("etag"),
                    mtime=mtime,
                    version_id=version_id,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to get info for {file_path}: {e}")

    return files
