#!/usr/bin/env python3
"""Use a frontier model as an LLM-as-a-judge to score QA pairs on correctness, grounding, and training signal strength."""

import argparse


UPSTREAM_URL = "https://gitlab-master.nvidia.com/sdg-research/sdg-share/-/tree/main/sdgs/long-document/public_recipes/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-endpoint", required=True, help="OpenAI-compatible judge endpoint URL.")
    parser.add_argument("--input", required=True, help="Input parquet or JSONL produced by a QA recipe.")
    parser.add_argument("--output", required=True, help="Output path for judged results.")
    parser.add_argument("--num-records", type=int, default=None, help="Optional limit on judged records.")
    return parser


def main() -> None:
    build_parser().parse_args()
    # TODO(release): port from upstream gitlab-master sdg-share/sdgs/long-document/public_recipes at release time:
    # https://gitlab-master.nvidia.com/sdg-research/sdg-share/-/tree/main/sdgs/long-document/public_recipes/
    raise NotImplementedError("Port this script body from upstream before release")


if __name__ == "__main__":
    main()
