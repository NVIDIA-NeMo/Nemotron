"""
fsspec filesystem implementation for art:// URIs.

Allows using art:// URIs with fsspec.open() and other fsspec-compatible tools.
"""

from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem

from nemotron.kit.exceptions import ArtifactNotFoundError, ArtifactVersionNotFoundError
from nemotron.kit.registry import get_registry
from nemotron.kit.track import STAGE_DIRS, get_config


class ArtifactFileSystem(AbstractFileSystem):
    """fsspec filesystem for art:// URIs.

    Enables using artifact URIs with fsspec-compatible APIs:

    Example:
        >>> import fsspec
        >>> # Read file from artifact
        >>> with fsspec.open("art://my-dataset:v1/train.json") as f:
        ...     data = json.load(f)
        >>>
        >>> # Get filesystem directly
        >>> fs = fsspec.filesystem("art")
        >>> files = fs.ls("art://my-dataset:v1")

    URI format:
        art://name:version/path/to/file
        art://name:latest/path/to/file
        art://name/path/to/file  (implies latest)
    """

    protocol = "art"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the artifact filesystem."""
        super().__init__(**kwargs)

    def _parse_uri(self, path: str) -> tuple[str, int | None, str]:
        """Parse art:// URI into components.

        Args:
            path: URI like "art://name:v1/file.txt" or "name:v1/file.txt"

        Returns:
            Tuple of (artifact_name, version, file_path)
        """
        # Remove protocol prefix if present
        if path.startswith("art://"):
            path = path[6:]

        # Split into artifact ref and file path
        parts = path.split("/", 1)
        artifact_ref = parts[0]
        file_path = parts[1] if len(parts) > 1 else ""

        # Parse artifact name and version
        if ":" in artifact_ref:
            name, version_str = artifact_ref.rsplit(":", 1)
            if version_str == "latest":
                version = None
            elif version_str.startswith("v"):
                version = int(version_str[1:])
            else:
                version = int(version_str)
        else:
            name = artifact_ref
            version = None

        return name, version, file_path

    def _resolve_semantic(self, uri: str) -> tuple[str, str] | None:
        """Resolve semantic URI like nano3/pretrain/data?sample=10000/file.json.

        Returns:
            Tuple of (base_path, file_path) or None if not a semantic URI.
        """
        if uri.startswith("art://"):
            uri = uri[6:]

        # Skip if has version specifier (name:version format)
        first_part = uri.split("?")[0].split("/")[0]
        if ":" in first_part:
            return None

        # Parse query params
        params: dict[str, str] = {}
        file_path = ""
        if "?" in uri:
            base, rest = uri.split("?", 1)
            if "/" in rest:
                params_str, file_path = rest.split("/", 1)
            else:
                params_str = rest
            for p in params_str.split("&"):
                if "=" in p:
                    k, v = p.split("=", 1)
                    params[k] = v
        else:
            parts = uri.split("/")
            if len(parts) > 3:
                base = "/".join(parts[:3])
                file_path = "/".join(parts[3:])
            else:
                base = uri

        parts = base.split("/")
        if len(parts) < 3:
            return None

        recipe, stage, step = parts[0], parts[1], parts[2]
        stage_dir = STAGE_DIRS.get(stage)
        if stage_dir is None:
            return None

        config = get_config()

        if config.backend == "wandb":
            import wandb

            artifact_name = f"{recipe}-{stage}-{step}"
            if "sample" in params:
                artifact_name += f"-sample-{params['sample']}"
            artifact = wandb.use_artifact(f"{artifact_name}:latest")
            base_path = artifact.download()  # Returns string path
        else:
            # fsspec backend - can be any fsspec URI (local, hf://, s3://, etc.)
            base_path = f"{config.output_root}/{recipe}/{stage_dir}"
            if "sample" in params:
                base_path = f"{base_path}/sample-{params['sample']}"

        return base_path, file_path

    def _resolve(self, path: str) -> tuple[Path, str]:
        """Resolve art:// path to local filesystem path.

        Returns:
            Tuple of (artifact_root_path, relative_file_path)
        """
        # Try semantic resolution first
        semantic = self._resolve_semantic(path)
        if semantic is not None:
            base_path, file_path = semantic
            return Path(base_path), file_path

        # Fall back to registry (name:version format)
        name, version, file_path = self._parse_uri(path)
        registry = get_registry()
        artifact_path = registry.resolve(name, version)
        return artifact_path, file_path

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open a file from an artifact.

        Args:
            path: art:// URI to file
            mode: File mode (only read modes supported)
            **kwargs: Additional arguments passed to open()

        Returns:
            File object
        """
        if "w" in mode or "a" in mode:
            raise ValueError("ArtifactFileSystem is read-only")

        artifact_path, file_path = self._resolve(path)
        full_path = artifact_path / file_path if file_path else artifact_path

        return open(full_path, mode, **kwargs)

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
        """List contents of artifact or directory within artifact.

        Args:
            path: art:// URI
            detail: If True, return detailed info dicts

        Returns:
            List of file paths or info dicts
        """
        artifact_path, file_path = self._resolve(path)
        target_path = artifact_path / file_path if file_path else artifact_path

        if not target_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if target_path.is_file():
            if detail:
                stat = target_path.stat()
                return [
                    {
                        "name": path,
                        "size": stat.st_size,
                        "type": "file",
                    }
                ]
            return [path]

        # List directory
        results = []
        for item in target_path.iterdir():
            item_path = f"{path.rstrip('/')}/{item.name}"
            if detail:
                stat = item.stat()
                results.append(
                    {
                        "name": item_path,
                        "size": stat.st_size if item.is_file() else 0,
                        "type": "file" if item.is_file() else "directory",
                    }
                )
            else:
                results.append(item_path)

        return results

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Get info about a path.

        Args:
            path: art:// URI

        Returns:
            Dict with file/directory info
        """
        artifact_path, file_path = self._resolve(path)
        target_path = artifact_path / file_path if file_path else artifact_path

        if not target_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        stat = target_path.stat()
        return {
            "name": path,
            "size": stat.st_size if target_path.is_file() else 0,
            "type": "file" if target_path.is_file() else "directory",
        }

    def exists(self, path: str, **kwargs: Any) -> bool:
        """Check if path exists.

        Args:
            path: art:// URI

        Returns:
            True if path exists
        """
        try:
            artifact_path, file_path = self._resolve(path)
            target_path = artifact_path / file_path if file_path else artifact_path
            return target_path.exists()
        except (ArtifactNotFoundError, ArtifactVersionNotFoundError):
            return False

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> bytes:
        """Read entire file content.

        Args:
            path: art:// URI
            start: Start byte offset
            end: End byte offset

        Returns:
            File contents as bytes
        """
        artifact_path, file_path = self._resolve(path)
        target_path = artifact_path / file_path if file_path else artifact_path

        with open(target_path, "rb") as f:
            if start is not None:
                f.seek(start)
            if end is not None:
                return f.read(end - (start or 0))
            return f.read()

    def get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        """Copy file from artifact to local path.

        Args:
            rpath: Remote art:// URI
            lpath: Local destination path
        """
        import shutil

        artifact_path, file_path = self._resolve(rpath)
        source = artifact_path / file_path if file_path else artifact_path

        if source.is_dir():
            shutil.copytree(source, lpath)
        else:
            shutil.copy2(source, lpath)

    def isfile(self, path: str) -> bool:
        """Check if path is a file."""
        try:
            artifact_path, file_path = self._resolve(path)
            target_path = artifact_path / file_path if file_path else artifact_path
            return target_path.is_file()
        except (ArtifactNotFoundError, ArtifactVersionNotFoundError, FileNotFoundError):
            return False

    def isdir(self, path: str) -> bool:
        """Check if path is a directory."""
        try:
            artifact_path, file_path = self._resolve(path)
            target_path = artifact_path / file_path if file_path else artifact_path
            return target_path.is_dir()
        except (ArtifactNotFoundError, ArtifactVersionNotFoundError, FileNotFoundError):
            return False
