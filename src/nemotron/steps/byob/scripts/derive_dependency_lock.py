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

"""Derive a local Python source's dependency lock from its imports and this environment.

Nothing here is a judgement, which is why it is computed rather than drafted. Intake walks
the same import graph and refuses any import that is neither local, stdlib, nor locked, so a
lock is either the set this script produces or a lock intake will reject. The versions and
artifact digests are facts about what is installed, and a guess at either would be a
provenance record that describes no real artifact.

The result is verified rather than asserted: the lock is written into a throwaway copy of
the source and put through the real inspector, so agreement with intake is demonstrated
before anything is written where a run would pick it up.

Exit codes carry the verdict. Zero means an empty lock, which is the only lock that lets a
source of this kind be probed. Two means the lock was produced and something about it stops
a run: either it names a dependency the execution policy will reject whatever the lock
says, or `--check` found the source disagreeing with the environment. One means the lock
could not be derived at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import sysconfig
import tempfile
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.source_adapters.local_python import (
    inspect_local_python_package,
)
from nemotron.steps.byob.runtime.source_adapters.local_python_reach import walk_source


class DerivationError(ValueError):
    """Raised when the lock cannot be computed from what is present."""


def _artifact_digest(distribution: metadata.Distribution) -> str:
    """A digest of the installed files themselves, rather than of the installer's manifest.

    Hashing RECORD is cheaper and was the first attempt, but RECORD describes the wheel as
    it shipped, not the tree as it stands: a package patched after installation, or one
    installed in editable mode, keeps its RECORD unchanged while the code that will run
    changes underneath it. A lock exists to say which code ran, so the files are read.
    """
    files = sorted(distribution.files or (), key=str)
    digest = hashlib.sha256()
    counted = 0
    for item in files:
        try:
            payload = Path(distribution.locate_file(item)).read_bytes()
        except OSError:
            # Listed but absent, which the digest should reflect rather than hide.
            continue
        digest.update(str(item).encode("utf-8"))
        digest.update(hashlib.sha256(payload).digest())
        counted += 1
    if not counted:
        name = distribution.metadata["Name"]
        raise DerivationError(
            f"{name} lists no readable installed file, so nothing identifies what would "
            "be imported; lock it by hand or reinstall it"
        )
    return "sha256:" + digest.hexdigest()


def locked_dependencies(imports: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    """Resolve each import name to the distribution that provides it, here and now."""
    provided = metadata.packages_distributions()
    locked: list[dict[str, str]] = []
    for import_name in sorted(imports):
        names = provided.get(import_name)
        if not names:
            origins = ", ".join(imports[import_name])
            raise DerivationError(
                f"import {import_name!r} at {origins} belongs to no installed "
                "distribution, so it is neither lockable nor importable at run time"
            )
        if len(set(names)) > 1:
            raise DerivationError(
                f"import {import_name!r} is provided by more than one distribution "
                f"({', '.join(sorted(set(names)))}); lock it by hand"
            )
        distribution = metadata.distribution(names[0])
        locked.append(
            {
                "import_name": import_name,
                "distribution": names[0],
                "version": distribution.version,
                "artifact_digest": _artifact_digest(distribution),
            }
        )
    return locked


def verify_against_intake(source: Path, document: dict[str, Any]) -> None:
    """Put the lock through the real inspector, on a copy, before trusting it.

    The copy keeps its symlinks rather than following them. Following them was the first
    attempt and it quietly changed the thing under test: intake refuses a source whose
    links point outside it, so a replica with the targets copied in place passes a check
    the original would fail, and the report then vouches for a source that cannot run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        replica = Path(tmp) / source.name
        shutil.copytree(source, replica, symlinks=True)
        (replica / "dependency-lock.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        inspect_local_python_package(replica, allowed_roots=(replica,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local Python source")
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the lock; defaults to dependency-lock.json in the source",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare with the lock already in the source instead of writing one",
    )
    args = parser.parse_args()
    source = args.source.resolve()

    try:
        backend = source / "backend.py"
        if not backend.is_file():
            raise DerivationError(f"no backend.py under {source}")
        document = {
            "schema_version": "bfcl-python-dependency-lock-v1",
            "dependencies": locked_dependencies(walk_source(source, backend).external),
        }
        verify_against_intake(source, document)
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "refusal_code": getattr(exc, "code", None),
                    "reason": getattr(exc, "detail", str(exc)),
                },
                default=str,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Locking an external import satisfies the A0 inspector and then loses the whole probe
    # suite: the execution policy admits only local modules and its own stdlib allowance,
    # and it is checked before the first probe runs. So a non-empty lock here is a finding
    # about the source rather than a lock to keep.
    findings = [
        {
            "code": "import_rejected_by_execution_policy",
            "impact": "blocks_all_probes",
            "detail": (
                f"{item['import_name']!r} is locked correctly and still fails "
                "local-python-v1, which raises probe_unsafe before any probe runs; a "
                "certifiable source of this kind imports only local modules and the "
                "profile's own stdlib allowance"
            ),
        }
        for item in document["dependencies"]
    ]
    serialized = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target = args.output or source / "dependency-lock.json"
    existing = target.read_text(encoding="utf-8") if target.is_file() else None
    drifted = existing is None or json.loads(existing) != document

    # A lock that disagrees with the environment and a lock that was never written are the
    # same problem for a run, but only one of them is a surprise to whoever wrote it.
    if not args.check:
        target.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                # Not "pass". The lock is right and the source still cannot be probed, so
                # calling this a pass tells a caller to carry on towards a run that has
                # already lost every probe. Whoever reads the exit code hears it too.
                "status": "blocked" if findings else "pass",
                "verified_against_intake": True,
                "dependencies": document["dependencies"],
                "findings": findings,
                "interpreter": sysconfig.get_python_version(),
                "target": str(target),
                "drifted": drifted,
                "written": not args.check,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if findings or (args.check and drifted):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
