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

"""CLI entrypoint for BYOB benchmark generation and translation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkRunResult
from nemotron.steps.byob.scripts.runtime import (
    STAGE_CHOICES,
    list_family_names,
    load_dispatch_config,
    resolve_dispatch_value,
    run_byob,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BYOB benchmark family stage")
    parser.add_argument("--config", type=Path, help="Path to the BYOB YAML config")
    parser.add_argument("--family", default=None, help="Benchmark family to run")
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        help=(
            "Pipeline stage to run. `eval` consumes an orchestration config; "
            "`all` chains prepare and generate only."
        ),
    )
    parser.add_argument(
        "--skip-until",
        default=None,
        help="Resume from a family-specific stage enum name, such as JUDGEMENT or QUALITY_METRICS",
    )
    parser.add_argument("--list-families", action="store_true", help="List registered benchmark families")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_families:
        for family in list_family_names():
            print(family)
        return

    if args.config is None:
        parser.error("--config is required unless --list-families is set")

    yaml_dict = load_dispatch_config(args.config)
    stage = resolve_dispatch_value(args.stage, yaml_dict, "stage")
    family = resolve_dispatch_value(args.family, yaml_dict, "family", default="mcq")
    skip_until = resolve_dispatch_value(args.skip_until, yaml_dict, "skip_until")

    if stage is None:
        parser.error("--stage is required unless the config contains `stage`")

    try:
        output = run_byob(
            config=args.config,
            stage=stage,
            family=family,
            skip_until=skip_until,
        )
    except Exception as exc:
        exit_code = getattr(exc, "cli_exit_code", None)
        if exit_code is None:
            raise
        code = getattr(exc, "code", "byob_cli_failed")
        print(f"{code}: {exc}", file=sys.stderr)
        raise SystemExit(exit_code) from exc
    if isinstance(output, BenchmarkRunResult):
        print(output.render())
    elif output is not None:
        print(output)


if __name__ == "__main__":
    main()
