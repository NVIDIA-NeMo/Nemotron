"""CLI entrypoint for BYOB benchmark generation and translation."""

from __future__ import annotations

import argparse
from pathlib import Path

from nemotron.steps.byob.scripts.runtime import list_family_names, run_byob


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BYOB benchmark family stage")
    parser.add_argument("--config", type=Path, help="Path to the BYOB YAML config")
    parser.add_argument("--family", default="mcq", help="Benchmark family to run")
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "translate"),
        help="Pipeline stage to run",
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

    if args.config is None or args.stage is None:
        parser.error("--config and --stage are required unless --list-families is set")

    output_path = run_byob(
        config=args.config,
        stage=args.stage,
        family=args.family,
        skip_until=args.skip_until,
    )
    if output_path is not None:
        print(output_path)


if __name__ == "__main__":
    main()
