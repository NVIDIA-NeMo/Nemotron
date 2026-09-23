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

"""Fill in a candidate's ``model_identity`` before an evaluation config is written.

``registry`` resolves a reference such as ``main`` to the commit it points at
right now, which is the value the eval contract wants and the one an operator
would otherwise go hunting for. ``local`` digests weights already on disk.
``provider-managed`` records a hosted route the provider pins nothing about, and
says so: that run can be scored but not published.

The resolved block is printed as JSON and, with ``--output``, written as a YAML
fragment to paste into the candidate. Nothing is invented — a route that cannot
be pinned is reported as unpinned rather than hashed into something that looks
like a pin. ``identity_publication_gate`` reports only what the identity decides;
scoring, contamination, artifact, and source gates are checked when the eval
config loads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.model_identity_resolution import (
    ModelIdentityResolutionError,
    identity_document,
    provider_managed_identity,
    resolve_local_identity,
    resolve_registry_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    registry = commands.add_parser("registry", help="Resolve a registry reference to an immutable commit.")
    registry.add_argument("--source", default="huggingface", help="Registry the weights come from.")
    registry.add_argument("--model", required=True, help="Repository id, such as org/model.")
    registry.add_argument(
        "--revision",
        help="Reference to resolve. A branch or tag is fine here; the commit it names is recorded.",
    )
    registry.add_argument("--output", type=Path, help="Write the YAML fragment here as well.")

    local = commands.add_parser("local", help="Digest weights held on this machine.")
    local.add_argument("--source", default="local", help="Label for where the weights came from.")
    local.add_argument("--model", required=True)
    local.add_argument("--weights-dir", type=Path, required=True)
    local.add_argument("--output", type=Path, help="Write the YAML fragment here as well.")

    hosted = commands.add_parser(
        "provider-managed",
        help="Record a hosted route with no published commit or digest.",
    )
    hosted.add_argument(
        "--source",
        required=True,
        help="The candidate's own provider, such as openai. An unpinned identity may name nothing else.",
    )
    hosted.add_argument("--model", required=True, help="The candidate's own model, exactly as the route names it.")
    hosted.add_argument("--output", type=Path, help="Write the YAML fragment here as well.")
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "registry":
        identity = resolve_registry_identity(source=args.source, model=args.model, revision=args.revision)
    elif args.command == "local":
        identity = resolve_local_identity(source=args.source, model=args.model, weights_dir=args.weights_dir)
    else:
        identity = provider_managed_identity(source=args.source, model=args.model)

    document = identity_document(identity)
    if args.output is not None:
        if args.output.exists():
            raise ModelIdentityResolutionError(f"refusing to overwrite an existing fragment: {args.output}")
        args.output.write_text(
            yaml.safe_dump({"model_identity": document["model_identity"]}, sort_keys=False),
            encoding="utf-8",
        )
        document["output"] = str(args.output.resolve())
    return {"status": "resolved", **document}


def main() -> None:
    try:
        result = _run(_parser().parse_args())
    except (ModelIdentityResolutionError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "fail", "error_type": type(exc).__name__, "reason": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
