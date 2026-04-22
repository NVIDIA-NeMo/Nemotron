#!/usr/bin/env python3
"""Classify pages by visual element type and reasoning complexity."""

import argparse


UPSTREAM_URL = "https://gitlab-master.nvidia.com/sdg-research/sdg-share/-/tree/main/sdgs/long-document/public_recipes/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-endpoint", required=True, help="OpenAI-compatible vLLM endpoint URL.")
    parser.add_argument("--seed-path", required=True, help="Path to the input seed parquet.")
    parser.add_argument("--num-records", type=int, default=None, help="Optional limit on processed records.")
    parser.add_argument("--output-dir", required=True, help="Directory for classification outputs.")
    return parser


def main() -> None:
    build_parser().parse_args()
    # TODO(release): port from upstream gitlab-master sdg-share/sdgs/long-document/public_recipes at release time:
    # https://gitlab-master.nvidia.com/sdg-research/sdg-share/-/tree/main/sdgs/long-document/public_recipes/
    raise NotImplementedError("Port this script body from upstream before release")


if __name__ == "__main__":
    main()
