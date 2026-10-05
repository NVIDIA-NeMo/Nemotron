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

"""Check a local Python source against its own catalogue, before intake spawns anything.

Intake and the probes already establish everything here, and that is the argument for
running it first rather than an argument against running it. The catalogue probe learns that
a backend has no `get_state` by importing it in a child process, after the identity closure
has been walked and digested, and it reports the result as a failed probe on a source that
was otherwise fine. Read statically the same defect is a line in a report and a name to fix.

What this cannot do is tell you the behaviour is right. Nothing static can. Every finding
here is about a source disagreeing with what it published, or about a scaffold that still
carries the marks left for a reviewer.

Exit codes follow the shared contract: 0 when the source agrees with its catalogue, 2 when
it does not, 1 when the check could not be run at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.pack_authoring.source_scaffolding import (
    REQUIRED_SYMBOLS,
    SourceScaffoldError,
    fixture_findings,
    interface_findings,
    read_surface,
)

_BLOCKING = {"blocks_all_probes", "blocks_intake", "blocks_a2"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local Python source package")
    parser.add_argument("--tools", type=Path, help="Defaults to tools.json inside the source")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat every finding as blocking, including the advisory ones",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    tools_path = args.tools.resolve() if args.tools else source / "tools.json"
    backend_path = source / "backend.py"
    fixtures_path = source / "fixtures.json"

    try:
        if not backend_path.is_file():
            raise ValueError(f"no backend.py under {source}")
        if not tools_path.is_file():
            raise ValueError(f"no reviewed catalogue at {tools_path}")
        surface = read_surface(json.loads(tools_path.read_text(encoding="utf-8")))
        backend = backend_path.read_text(encoding="utf-8")
        findings = interface_findings(backend, published=[tool.name for tool in surface])
        fixtures: dict[str, Any] | None = None
        if fixtures_path.is_file():
            fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
            if not isinstance(fixtures, dict):
                raise ValueError("fixtures.json must be an object of collections")
            findings.extend(fixture_findings(fixtures))
    except (OSError, SyntaxError, ValueError, SourceScaffoldError) as exc:
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

    blocking = [item for item in findings if args.strict or item["impact"] in _BLOCKING]
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "blocked" if blocking else "pass",
                "source": str(source),
                "tools": [tool.name for tool in surface],
                "required_symbols": list(REQUIRED_SYMBOLS),
                "fixture_collections": sorted(fixtures) if fixtures else [],
                "findings": findings,
                "blocking": [item["code"] for item in blocking],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if blocking:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
