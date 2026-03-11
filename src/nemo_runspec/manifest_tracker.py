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

"""Manifest-based artifact tracker.

Zero-copy tracker that writes structured JSON manifests via fsspec.
Supports local filesystems, S3, GCS, and HF Hub.

Data stays at its original location — only JSON metadata is written
to the artifact directory.

Directory layout::

    {root}/
    ├── nano3-pretrain-data/
    │   ├── v1/
    │   │   ├── manifest.json
    │   │   └── metadata.json
    │   ├── v2/
    │   │   ├── manifest.json
    │   │   └── metadata.json
    │   └── latest              # plain file containing "v2"

Usage::

    tracker = ManifestTracker(root="/lustre/artifacts")
    tracker.log_model_checkpoint(name="nano3-model", path="/lustre/ckpt", iteration=500)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec

if TYPE_CHECKING:
    from nemotron.kit.artifact import Artifact

logger = logging.getLogger(__name__)


class ManifestTracker:
    """Zero-copy manifest-based artifact tracker.

    Writes ``manifest.json`` + ``metadata.json`` to ``{root}/{name}/v{N}/``
    via fsspec. Supports local filesystems, S3, GCS, and HF Hub.

    Satisfies the :class:`nemotron.kit.trackers.LineageTracker` protocol
    via structural (duck) typing — no import required.
    """

    def __init__(self, root: str) -> None:
        self._root = root.rstrip("/")

        # Validate HF Hub roots require org/repo format
        if self._root.startswith("hf://"):
            parts = self._root.removeprefix("hf://").split("/")
            if len(parts) < 2:
                raise ValueError(
                    f"HF Hub root must be 'hf://org/repo' or 'hf://org/repo/subdir', got: {root}"
                )

        self._fs: fsspec.AbstractFileSystem
        self._fs_root: str
        self._fs, self._fs_root = fsspec.core.url_to_fs(self._root)
        self._fs.mkdirs(self._fs_root, exist_ok=True)

    # ------------------------------------------------------------------
    # LineageTracker protocol
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return True

    def get_run_id(self) -> str | None:
        return os.environ.get("NEMO_EXPERIMENT_ID", "local")

    def use_artifact(self, ref: str, artifact_type: str) -> Path:
        """Resolve artifact ref to its original data path."""
        name, version = _parse_ref(ref)
        version_dir = self._resolve_version_dir(name, version)
        manifest = self._read_manifest(version_dir)
        return Path(manifest["path"])

    def log_artifact(
        self, artifact: Artifact, name: str, used_refs: list[str]
    ) -> dict[str, Any]:
        """Publish a full Pydantic artifact — writes manifest + metadata."""
        version, version_dir = self._allocate_version(name)

        art_path = artifact.path
        if art_path is None:
            raise ValueError(f"Artifact has no path set, cannot log: {artifact}")
        path_str = str(art_path.resolve()) if hasattr(art_path, "resolve") else str(art_path)

        manifest = {
            "name": name,
            "version": version,
            "type": artifact.type,
            "path": path_str,
            "created_at": artifact.created_at,
            "producer": artifact.producer or self.get_run_id(),
            "metadata": artifact.metadata,
            "inputs": artifact.get_input_uris(),
            "used_artifacts": used_refs,
        }

        artifact_data = artifact.model_dump(mode="json")

        self._write_version_files(version_dir, manifest, artifact_data, name, version)

        return {
            "artifact_id": f"{name}:v{version}",
            "artifact_type": artifact.type,
            "run_id": self.get_run_id(),
            "url": None,
            "used_artifacts": used_refs,
        }

    # ------------------------------------------------------------------
    # Convenience: checkpoint logging without a full Artifact object
    # ------------------------------------------------------------------

    def log_model_checkpoint(
        self,
        name: str,
        path: str,
        iteration: int,
    ) -> dict[str, Any]:
        """Write manifest + metadata for a model checkpoint.

        This avoids constructing a full :class:`Artifact` Pydantic object,
        which is convenient for monkey-patched checkpoint callbacks.
        """
        version, version_dir = self._allocate_version(name)

        now = datetime.now().astimezone().isoformat()
        producer = self.get_run_id()

        manifest: dict[str, Any] = {
            "name": name,
            "version": version,
            "type": "model",
            "path": path,
            "created_at": now,
            "producer": producer,
            "metadata": {
                "iteration": iteration,
                "absolute_path": path,
            },
            "inputs": [],
            "used_artifacts": [],
        }

        metadata: dict[str, Any] = {
            "path": path,
            "type": "model",
            "created_at": now,
            "iteration": iteration,
            "absolute_path": path,
            "metadata": {
                "iteration": iteration,
                "absolute_path": path,
            },
        }

        self._write_version_files(version_dir, manifest, metadata, name, version)

        return {
            "artifact_id": f"{name}:v{version}",
            "artifact_type": "model",
            "run_id": producer,
            "url": None,
            "used_artifacts": [],
        }

    # ------------------------------------------------------------------
    # Version file writing (with HF Hub batching)
    # ------------------------------------------------------------------

    def _write_version_files(
        self,
        version_dir: str,
        manifest: dict[str, Any],
        metadata: dict[str, Any],
        name: str,
        version: int,
    ) -> None:
        """Write manifest.json, metadata.json, and update ``latest``.

        For ``hf://`` roots, batches all three files into a single HF Hub
        commit so the write is atomic from the reader's perspective.
        For all other filesystems, uses the normal temp-then-rename pattern.
        """
        if self._root.startswith("hf://"):
            self._write_version_files_hf(version_dir, manifest, metadata, name, version)
        else:
            self._atomic_json_write(f"{version_dir}/manifest.json", manifest)
            self._atomic_json_write(f"{version_dir}/metadata.json", metadata)
            self._update_latest(name, version)

    def _write_version_files_hf(
        self,
        version_dir: str,
        manifest: dict[str, Any],
        metadata: dict[str, Any],
        name: str,
        version: int,
    ) -> None:
        """Batch-write all version files as a single HF Hub commit."""
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        # Parse repo_id from root: "hf://org/repo-name/subdir" → "org/repo-name"
        parts = self._root.removeprefix("hf://").split("/")
        repo_id = f"{parts[0]}/{parts[1]}"
        # Paths within the repo are relative to repo root
        prefix = "/".join(parts[2:]) if len(parts) > 2 else ""
        base = f"{prefix}/{name}" if prefix else name

        ops = [
            CommitOperationAdd(
                path_in_repo=f"{base}/v{version}/manifest.json",
                path_or_fileobj=json.dumps(manifest, indent=2, default=str).encode(),
            ),
            CommitOperationAdd(
                path_in_repo=f"{base}/v{version}/metadata.json",
                path_or_fileobj=json.dumps(metadata, indent=2, default=str).encode(),
            ),
            CommitOperationAdd(
                path_in_repo=f"{base}/latest",
                path_or_fileobj=f"v{version}".encode(),
            ),
        ]
        api.create_commit(
            repo_id=repo_id,
            operations=ops,
            commit_message=f"artifact: {name} v{version}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_version(self, name: str) -> tuple[int, str]:
        """Create the next version directory, return ``(version, version_dir)``."""
        artifact_dir = f"{self._fs_root}/{name}"
        self._fs.mkdirs(artifact_dir, exist_ok=True)

        existing: list[int] = []
        try:
            for entry in self._fs.ls(artifact_dir, detail=False):
                basename = entry.rstrip("/").split("/")[-1]
                if basename.startswith("v") and basename[1:].isdigit():
                    existing.append(int(basename[1:]))
        except FileNotFoundError:
            pass

        version = max(existing, default=0) + 1
        version_dir = f"{artifact_dir}/v{version}"
        # Use exist_ok=False so concurrent writers surface as errors
        # rather than silently overwriting. In practice only one rank
        # writes (is_last_rank / Ray driver), but this guards against
        # unexpected concurrency.
        try:
            self._fs.mkdirs(version_dir, exist_ok=False)
        except FileExistsError:
            # Another process beat us — retry with incremented version
            existing.append(version)
            version = max(existing) + 1
            version_dir = f"{artifact_dir}/v{version}"
            self._fs.mkdirs(version_dir, exist_ok=True)
        return version, version_dir

    def _update_latest(self, name: str, version: int) -> None:
        """Write a plain text ``latest`` file with the version directory name."""
        artifact_dir = f"{self._fs_root}/{name}"
        latest_path = f"{artifact_dir}/latest"
        temp_path = f"{artifact_dir}/.latest_tmp_{os.getpid()}"
        with self._fs.open(temp_path, "w") as f:
            f.write(f"v{version}")
        self._fs.mv(temp_path, latest_path)

    def _resolve_version_dir(self, name: str, version: int | str | None) -> str:
        artifact_dir = f"{self._fs_root}/{name}"
        if not self._fs.exists(artifact_dir):
            raise FileNotFoundError(f"Artifact not found: {name}")

        if version is None or version == "latest":
            latest_path = f"{artifact_dir}/latest"
            if self._fs.exists(latest_path):
                with self._fs.open(latest_path, "r") as f:
                    version_name = f.read().strip()
                return f"{artifact_dir}/{version_name}"
            return f"{artifact_dir}/v{self._highest_version(name)}"

        if isinstance(version, str):
            if version.startswith("v") and version[1:].isdigit():
                version = int(version[1:])
            elif version.isdigit():
                version = int(version)
            else:
                raise FileNotFoundError(f"Unknown version format: {version}")

        version_dir = f"{artifact_dir}/v{version}"
        if not self._fs.exists(version_dir):
            raise FileNotFoundError(f"Version not found: {name}:v{version}")
        return version_dir

    def _highest_version(self, name: str) -> int:
        artifact_dir = f"{self._fs_root}/{name}"
        versions: list[int] = []
        for entry in self._fs.ls(artifact_dir, detail=False):
            basename = entry.rstrip("/").split("/")[-1]
            if basename.startswith("v") and basename[1:].isdigit():
                versions.append(int(basename[1:]))
        if not versions:
            raise FileNotFoundError(f"No versions found for: {name}")
        return max(versions)

    def _read_manifest(self, version_dir: str) -> dict[str, Any]:
        path = f"{version_dir}/manifest.json"
        if not self._fs.exists(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        with self._fs.open(path, "r") as f:
            return json.load(f)

    def _atomic_json_write(self, path: str, data: dict[str, Any]) -> None:
        temp = f"{path}.tmp.{os.getpid()}"
        with self._fs.open(temp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self._fs.mv(temp, path)

    @property
    def root(self) -> str:
        """The root URI/path for this tracker."""
        return self._root


def _parse_ref(ref: str) -> tuple[str, int | str | None]:
    """Parse ``'name:v5'`` or ``'name:latest'`` into ``(name, version)``."""
    if ":" not in ref:
        return ref, None
    name, version_str = ref.rsplit(":", 1)
    if version_str == "latest":
        return name, "latest"
    if version_str.startswith("v") and version_str[1:].isdigit():
        return name, int(version_str[1:])
    if version_str.isdigit():
        return name, int(version_str)
    return name, version_str
