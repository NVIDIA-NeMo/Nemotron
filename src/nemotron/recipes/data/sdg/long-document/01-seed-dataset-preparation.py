#!/usr/bin/env python3
"""Download PDFs from HuggingFace FinePDFs, render pages to PNG, and produce per-page, windowed, and whole-document seed parquet files."""

import argparse


UPSTREAM_URL = "https://gitlab-master.nvidia.com/sdg-research/sdg-share/-/tree/main/sdgs/long-document/public_recipes/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory for generated seed parquet files.")
    parser.add_argument("--num-docs", type=int, default=10000, help="Number of documents to prepare.")
    parser.add_argument(
        "--dataset-id",
        default="HuggingFaceFW/finepdfs",
        help="Source Hugging Face dataset identifier.",
    )
    return parser


def main() -> None:
    build_parser().parse_args()
    # TODO(release): port from upstream gitlab-master sdg-share/sdgs/long-document/public_recipes at release time:
    # https://gitlab-master.nvidia.com/sdg-research/sdg-share/-/tree/main/sdgs/long-document/public_recipes/
    raise NotImplementedError("Port this script body from upstream before release")


if __name__ == "__main__":
    main()
