#!/usr/bin/env python3
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

"""Create a deterministic, content-addressed BFCL release artifact bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path


class ReleaseArchiveError(ValueError):
    """The requested archive would be incomplete, ambiguous, or unsafe."""


@dataclass(frozen=True)
class ArchiveSource:
    logical_root: str
    path: Path


def _hash(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _files(source: ArchiveSource) -> list[tuple[str, Path]]:
    root = source.path.resolve()
    if not root.is_dir():
        raise ReleaseArchiveError(f"archive source is not a directory: {source.path}")
    files: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ReleaseArchiveError(f"archive source contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReleaseArchiveError(f"archive source contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix()
        files.append((f"{source.logical_root}/{relative}", path))
    if not files:
        raise ReleaseArchiveError(f"archive source is empty: {source.path}")
    return files


def build_release_archive(
    *,
    release_dir: Path,
    evaluation_dirs: tuple[Path, ...],
    evidence_dirs: tuple[Path, ...],
    output_dir: Path,
    bundle_name: str,
) -> tuple[Path, Path]:
    """Write a deterministic tar.gz and its content inventory."""
    if not bundle_name or "/" in bundle_name or bundle_name in {".", ".."}:
        raise ReleaseArchiveError("bundle_name must be one safe path component")
    sources = [ArchiveSource("release", release_dir)]
    sources.extend(
        ArchiveSource(f"evaluations/{path.name}", path)
        for path in evaluation_dirs
    )
    sources.extend(
        ArchiveSource(f"evidence/{path.name}", path)
        for path in evidence_dirs
    )
    entries: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for source in sources:
        for logical_path, path in _files(source):
            if logical_path in seen:
                raise ReleaseArchiveError(f"duplicate archive path: {logical_path}")
            seen.add(logical_path)
            entries.append((logical_path, path.read_bytes()))
    inventory = {
        "schema_version": "1.0",
        "files": [
            {
                "path": path,
                "size_bytes": len(data),
                "content_hash": _hash(data),
            }
            for path, data in entries
        ],
    }
    inventory_bytes = (
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    entries.append(("artifact_inventory.json", inventory_bytes))

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for logical_path, data in entries:
            info = tarfile.TarInfo(logical_path)
            info.size = len(data)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as stream:
        stream.write(tar_buffer.getvalue())
    archive_bytes = compressed.getvalue()

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{bundle_name}.tar.gz"
    inventory_path = output_dir / f"{bundle_name}.inventory.json"
    for path, data in (
        (archive_path, archive_bytes),
        (inventory_path, inventory_bytes),
    ):
        if path.exists() and path.read_bytes() != data:
            raise ReleaseArchiveError(f"refusing to replace a different artifact: {path}")
        path.write_bytes(data)
    (output_dir / f"{bundle_name}.sha256").write_text(
        f"{hashlib.sha256(archive_bytes).hexdigest()}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return archive_path, inventory_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, action="append", default=[])
    parser.add_argument("--evidence-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-name", default="bfcl-release-bundle")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        archive, inventory = build_release_archive(
            release_dir=args.release_dir,
            evaluation_dirs=tuple(args.evaluation_dir),
            evidence_dirs=tuple(args.evidence_dir),
            output_dir=args.output_dir,
            bundle_name=args.bundle_name,
        )
    except (OSError, ReleaseArchiveError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_hash": _hash(archive.read_bytes()),
                "inventory": str(inventory),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
