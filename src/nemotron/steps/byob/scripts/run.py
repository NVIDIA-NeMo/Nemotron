"""CLI entrypoint for BYOB benchmark generation and translation."""

from __future__ import annotations

import argparse
from pathlib import Path

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
        help="Pipeline stage to run. Use `all` to chain prepare and generate.",
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

    output_path = run_byob(config=args.config, stage=stage, family=family, skip_until=skip_until)
    if output_path is not None:
        print(output_path)


if __name__ == "__main__":
    main()
