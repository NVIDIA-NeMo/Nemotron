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

"""Walk a local Python source the way intake does, without needing a lock to do it.

Two tools have to know which files a source actually reaches and what it imports from
outside itself: the one that derives the dependency lock, and the one that reads the error
vocabulary off the code. Intake computes that closure already, but only once a lock exists,
which is no help to the tool whose job is to write one.

So the walk is repeated here with the lock left out, and every resolution decision is
delegated to intake's own resolver rather than reimplemented. That matters more than it
looks: a hand-rolled resolver that forgets a package's `__init__.py`, or that does not
follow `from package import submodule`, reports a closure that is missing whole files, and
a lock derived from it is then refused by the very check it was written for.
"""

from __future__ import annotations

import ast
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nemotron.steps.byob.runtime.source_adapters.local_python import (
    absolute_import_target,
    module_identity,
    resolve_local_module,
)


@dataclass(frozen=True)
class LocalReach:
    """What a source reaches: its own files, and the top-level names it imports."""

    modules: tuple[str, ...]
    external: Mapping[str, tuple[str, ...]]


def walk_source(root: Path, backend: Path) -> LocalReach:
    """Every local file reachable from the backend, and every import that leaves it.

    An import is external when it resolves to nothing inside the source and its top-level
    name is neither stdlib nor built in. Each one is reported with the places it was seen,
    because a lock entry a reviewer cannot trace back to a line is not worth much.
    """
    # Resolved first because the resolver hands back real paths, and on a platform where
    # the temporary or home directory is itself a link an unresolved root no longer
    # prefixes them, which turns every file it finds into one it cannot name.
    root = root.resolve()
    queue: deque[Path] = deque([backend.resolve()])
    visited: dict[str, None] = {}
    external: dict[str, list[str]] = {}
    stdlib = sys.stdlib_module_names | set(sys.builtin_module_names)

    def note_external(module: str, origin: str) -> None:
        top = module.partition(".")[0]
        if top and top not in stdlib:
            external.setdefault(top, []).append(origin)

    while queue:
        path = queue.popleft()
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in visited:
            continue
        visited[relative] = None
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        _, package = module_identity(root, path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = resolve_local_module(root, alias.name)
                    if local:
                        queue.extend(local)
                    else:
                        note_external(alias.name, f"{relative}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                base = absolute_import_target(node, package=package, source=relative)
                local = resolve_local_module(root, base)
                if local:
                    queue.extend(local)
                    # `from package import submodule` imports a file, and nothing in the
                    # statement says whether the name is a module or an attribute of one,
                    # so both readings are tried and only the one that exists is followed.
                    for alias in node.names:
                        if alias.name != "*":
                            child = resolve_local_module(root, f"{base}.{alias.name}")
                            if child:
                                queue.extend(child)
                elif node.level == 0:
                    note_external(base, f"{relative}:{node.lineno}")

    return LocalReach(
        modules=tuple(sorted(visited)),
        external={name: tuple(origins) for name, origins in sorted(external.items())},
    )
