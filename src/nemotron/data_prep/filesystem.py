"""Filesystem utilities using fsspec for cloud-native operations."""

import json
from typing import Any

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs


def get_filesystem(path: str) -> tuple[AbstractFileSystem, str]:
    """Get filesystem and normalized path from a URI."""
    fs, normalized = url_to_fs(path)
    return fs, normalized


def read_json(fs: AbstractFileSystem, path: str) -> Any:
    """Read JSON file from filesystem."""
    with fs.open(path, "r") as f:
        return json.load(f)


def write_json(
    fs: AbstractFileSystem, path: str, data: Any, indent: int = 2
) -> None:
    """Write JSON file to filesystem."""
    with fs.open(path, "w") as f:
        json.dump(data, f, indent=indent)


def ensure_dir(fs: AbstractFileSystem, path: str) -> None:
    """Ensure directory exists, creating it if necessary."""
    fs.makedirs(path, exist_ok=True)


def file_exists(fs: AbstractFileSystem, path: str) -> bool:
    """Check if a file exists."""
    try:
        return fs.exists(path)
    except Exception:
        return False
