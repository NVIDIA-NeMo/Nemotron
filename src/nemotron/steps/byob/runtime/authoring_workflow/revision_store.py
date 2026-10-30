# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Crash-safe, immutable storage for content-addressed authoring revisions."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

REVISION_MANIFEST_VERSION: Literal["bfcl-revision-manifest-v1"] = (
    "bfcl-revision-manifest-v1"
)
MANIFEST_FILE_NAME = "manifest.json"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SUPPORTED_FILESYSTEMS = frozenset(
    {"apfs", "hfs", "hfs+", "ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "overlay"}
)

CrashHook = Callable[[str], None]

_RECOVERY_BY_CODE = {
    "revision_already_exists": "use the existing verified revision or a new content address",
    "unsupported_filesystem": "move the workspace to an approved local filesystem",
    "filesystem_probe_failed": "repair filesystem inspection before retrying",
    "revision_incomplete": "preserve the revision for audit and rebuild from its parent",
    "artifact_digest_mismatch": "restore the immutable artifact or rebuild the revision",
    "manifest_invalid": "resume from the last revision with a verified manifest",
    "content_address_mismatch": "load the revision through its recorded content address",
    "workspace_invalid": "use a real, non-symlink workspace directory",
}


class RevisionStoreError(ValueError):
    """A stable, machine-classifiable revision-store refusal."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        recovery: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery or _RECOVERY_BY_CODE.get(
            code,
            "preserve the workspace and resume from the last verified revision",
        )
        super().__init__(f"{code}: {detail}; recovery: {self.recovery}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RevisionArtifact(_StrictModel):
    path: StrictStr
    digest: StrictStr
    size_bytes: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> RevisionArtifact:
        _validate_artifact_path(self.path)
        if _DIGEST.fullmatch(self.digest) is None:
            raise ValueError("artifact digest must be sha256:<64 lowercase hex>")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes cannot be negative")
        return self


class RevisionManifest(_StrictModel):
    schema_version: Literal["bfcl-revision-manifest-v1"]
    content_address: StrictStr
    artifacts: tuple[RevisionArtifact, ...]
    manifest_digest: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RevisionManifest:
        _validate_digest(self.content_address)
        paths = tuple(artifact.path for artifact in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("revision artifacts must be sorted with unique paths")
        unsigned = self.model_dump(mode="json", exclude={"manifest_digest"})
        if self.manifest_digest != sha256_json(unsigned):
            raise ValueError("revision manifest digest mismatch")
        return self


def _validate_digest(value: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise RevisionStoreError(
            "content_address_invalid",
            "content address must be sha256:<64 lowercase hex>",
        )


def _validate_artifact_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or value in {".", "..", MANIFEST_FILE_NAME}
    ):
        raise RevisionStoreError(
            "artifact_path_invalid",
            f"revision artifact must be one safe top-level file: {value!r}",
        )


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _darwin_filesystem(path: Path) -> str:
    try:
        result = subprocess.run(
            ["/sbin/mount"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RevisionStoreError(
            "filesystem_probe_failed",
            f"cannot inspect local mount table: {type(exc).__name__}",
        ) from exc
    resolved = path.resolve()
    candidates: list[tuple[int, str, bool]] = []
    for line in result.stdout.splitlines():
        if " on " not in line or " (" not in line or not line.endswith(")"):
            continue
        _device, remainder = line.split(" on ", 1)
        mount_text, options_text = remainder.rsplit(" (", 1)
        mount_point = Path(mount_text.replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        options = tuple(item.strip().lower() for item in options_text[:-1].split(","))
        candidates.append((len(mount_point.parts), options[0], "local" in options))
    if not candidates:
        raise RevisionStoreError(
            "filesystem_probe_failed",
            f"no mount entry contains {resolved}",
        )
    _depth, filesystem, local = max(candidates)
    if not local:
        raise RevisionStoreError(
            "unsupported_filesystem",
            f"revision store requires a local filesystem, got {filesystem}",
        )
    return filesystem


def _linux_filesystem(path: Path) -> str:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RevisionStoreError(
            "filesystem_probe_failed",
            f"cannot inspect local mount table: {type(exc).__name__}",
        ) from exc
    resolved = path.resolve()
    candidates: list[tuple[int, str]] = []
    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        details = right.split()
        if len(fields) < 5 or not details:
            continue
        mount_point = Path(fields[4].replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidates.append((len(mount_point.parts), details[0].lower()))
    if not candidates:
        raise RevisionStoreError(
            "filesystem_probe_failed",
            f"no mount entry contains {resolved}",
        )
    return max(candidates)[1]


def _filesystem_kind(path: Path) -> str:
    system = platform.system()
    if system == "Darwin":
        return _darwin_filesystem(path)
    if system == "Linux":
        return _linux_filesystem(path)
    raise RevisionStoreError(
        "unsupported_filesystem",
        f"revision store has no durable local backend for {system}",
    )


def _assert_supported_filesystem(path: Path) -> None:
    filesystem = _filesystem_kind(path)
    if filesystem not in _SUPPORTED_FILESYSTEMS:
        raise RevisionStoreError(
            "unsupported_filesystem",
            f"filesystem {filesystem!r} has no approved atomic-rename contract",
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable(
    path: Path,
    payload: bytes,
    *,
    crash_hook: CrashHook | None,
    write_event: str,
    fsync_event: str,
) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        if crash_hook is not None:
            crash_hook(write_event)
        os.fsync(stream.fileno())
        if crash_hook is not None:
            crash_hook(fsync_event)


def _rename_directory(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    system = platform.system()
    if system == "Darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif system == "Linux":
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise RevisionStoreError(
            "unsupported_filesystem",
            f"no exclusive directory rename primitive for {system}",
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RevisionStoreError(
                "manifest_invalid",
                f"duplicate manifest key {key!r}",
            )
        result[key] = value
    return result


class RevisionStore:
    """Persist complete revisions with a manifest-last durable commit."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def put(
        self,
        content_address: str,
        artifacts: Mapping[str, bytes],
        *,
        crash_hook: CrashHook | None = None,
    ) -> Path:
        _validate_digest(content_address)
        if not artifacts:
            raise RevisionStoreError(
                "revision_empty",
                "a revision must contain at least one artifact",
            )
        normalized = dict(artifacts)
        for name, payload in normalized.items():
            _validate_artifact_path(name)
            if not isinstance(payload, bytes):
                raise RevisionStoreError(
                    "artifact_invalid",
                    f"artifact {name!r} must be bytes",
                )
        if len(normalized) != len(artifacts):
            raise RevisionStoreError(
                "artifact_path_invalid",
                "revision artifact paths must be unique",
            )

        parent = self.root.parent
        if not parent.is_dir() or parent.is_symlink():
            raise RevisionStoreError(
                "workspace_invalid",
                f"revision-store parent must be an existing real directory: {parent}",
            )
        _assert_supported_filesystem(parent)
        if self.root.exists():
            if self.root.is_symlink() or not self.root.is_dir():
                raise RevisionStoreError(
                    "workspace_invalid",
                    f"revision-store root must be a real directory: {self.root}",
                )
        else:
            self.root.mkdir()
            _fsync_directory(parent)
        _fsync_directory(self.root)
        if crash_hook is not None:
            crash_hook("after_root_fsync")

        target = self.root / content_address.removeprefix("sha256:")
        if target.exists():
            raise RevisionStoreError(
                "revision_already_exists",
                f"revision already exists: {content_address}",
            )
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=f".{target.name}.staging-"))
        renamed = False
        try:
            records: list[RevisionArtifact] = []
            for name in sorted(normalized):
                payload = normalized[name]
                _write_durable(
                    staging / name,
                    payload,
                    crash_hook=crash_hook,
                    write_event=f"after_artifact_write:{name}",
                    fsync_event=f"after_artifact_fsync:{name}",
                )
                records.append(
                    RevisionArtifact(
                        path=name,
                        digest=_digest_bytes(payload),
                        size_bytes=len(payload),
                    )
                )
            _fsync_directory(staging)
            if crash_hook is not None:
                crash_hook("after_artifacts_directory_fsync")

            unsigned = {
                "schema_version": REVISION_MANIFEST_VERSION,
                "content_address": content_address,
                "artifacts": [
                    record.model_dump(mode="json") for record in records
                ],
            }
            manifest = RevisionManifest.model_validate(
                {**unsigned, "manifest_digest": sha256_json(unsigned)}
            )
            manifest_payload = (
                canonical_json(manifest.model_dump(mode="json")) + "\n"
            ).encode("utf-8")
            _write_durable(
                staging / MANIFEST_FILE_NAME,
                manifest_payload,
                crash_hook=crash_hook,
                write_event="after_manifest_write",
                fsync_event="after_manifest_fsync",
            )
            _fsync_directory(staging)
            if crash_hook is not None:
                crash_hook("before_rename")

            try:
                _rename_directory(staging, target)
            except OSError as exc:
                if target.exists():
                    raise RevisionStoreError(
                        "revision_already_exists",
                        f"revision already exists: {content_address}",
                    ) from exc
                raise
            renamed = True
            if crash_hook is not None:
                crash_hook("after_rename")
            _fsync_directory(self.root)
            if crash_hook is not None:
                crash_hook("after_parent_fsync")
        except Exception:
            if not renamed:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return target

    def put_json(
        self,
        content_address: str,
        artifacts: Mapping[str, Any],
        *,
        crash_hook: CrashHook | None = None,
    ) -> Path:
        payloads = {
            name: (canonical_json(document) + "\n").encode("utf-8")
            for name, document in artifacts.items()
        }
        return self.put(content_address, payloads, crash_hook=crash_hook)

    def load_manifest(self, content_address: str) -> RevisionManifest:
        _validate_digest(content_address)
        target = self.root / content_address.removeprefix("sha256:")
        manifest_path = target / MANIFEST_FILE_NAME
        try:
            document = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_mapping,
            )
            manifest = RevisionManifest.model_validate(document)
        except RevisionStoreError:
            raise
        except Exception as exc:
            raise RevisionStoreError(
                "manifest_invalid",
                f"cannot load revision manifest: {type(exc).__name__}: {exc}",
            ) from exc
        if manifest.content_address != content_address:
            raise RevisionStoreError(
                "content_address_mismatch",
                "revision manifest names a different content address",
            )
        return manifest

    def verify(self, content_address: str) -> RevisionManifest:
        manifest = self.load_manifest(content_address)
        target = self.root / content_address.removeprefix("sha256:")
        if target.is_symlink() or not target.is_dir():
            raise RevisionStoreError(
                "revision_invalid",
                "revision path must be a real directory",
            )
        expected = {MANIFEST_FILE_NAME, *(item.path for item in manifest.artifacts)}
        observed = {path.name for path in target.iterdir()}
        if observed != expected:
            raise RevisionStoreError(
                "revision_incomplete",
                f"revision entries differ: missing={sorted(expected - observed)!r}, "
                f"extra={sorted(observed - expected)!r}",
            )
        for artifact in manifest.artifacts:
            path = target / artifact.path
            if path.is_symlink() or not path.is_file():
                raise RevisionStoreError(
                    "revision_incomplete",
                    f"artifact is not a regular file: {artifact.path}",
                )
            payload = path.read_bytes()
            if len(payload) != artifact.size_bytes or _digest_bytes(payload) != artifact.digest:
                raise RevisionStoreError(
                    "artifact_digest_mismatch",
                    f"artifact changed after commit: {artifact.path}",
                )
        return manifest
