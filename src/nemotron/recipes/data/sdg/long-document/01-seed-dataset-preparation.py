# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "pymupdf",
#     "pandas",
#     "pyarrow",
#     "pydantic>=2",
#     "omegaconf>=2.3",
#     "pyyaml",
# ]
#
# [tool.runspec]
# schema = "1"
# name = "data/sdg/long-document/seed"
# image = "nvcr.io/nvidia/pytorch:25.12-py3"
# setup = "Inline PEP 723 deps resolved at runtime via `uv run --no-project`."
#
# [tool.runspec.run]
# launch = "direct"
# cmd = "uv run --no-project {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config"
# default = "01-seed"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Long-Document Understanding Seed Dataset Preparation.

Uses HuggingFace's FinePDFs dataset (HuggingFaceFW/finepdfs) as an example
data source. Downloads PDFs, renders each page to a PNG image, and produces
three seed parquet files under ``output_dir``:

  1. ``seed_per_page.parquet`` — one row per page (used by 02–06).
  2. ``seed_windowed.parquet`` — one row per sliding window of pages (used by 07).
     Window size adapts to document length (2 pages for short documents up to 8
     for long ones).
  3. ``seed_whole_document.parquet`` — one row per document (used by 08).

All three share a ``png_images_base64`` column containing a JSON array of
base64-encoded PNG strings.

Run standalone (operator-driven):

    uv run --no-project 01-seed-dataset-preparation.py \
        --config config/01-seed.yaml \
        num_docs=50 subset=fra_Latn

Run via the Nemotron CLI (nemo-run dispatch):

    nemotron data sdg long-document seed --run dlw -c 01-seed num_docs=50

CPU-only — no GPU or external endpoint required.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Pull in the shared YAML+dotlist+Pydantic loader from the sibling helper
# (folder name has a dash so it isn't an importable package).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _recipe_config import load_recipe_config  # noqa: E402  (after sys.path mutation)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

FINEPDFS_REPO = "HuggingFaceFW/finepdfs"


class SeedConfig(BaseModel):
    """Pydantic config for the long-document seed-preparation stage."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(
        default=Path("./seed_data"),
        description="Directory where the three seed parquet files are written.",
    )
    num_docs: int = Field(
        default=10,
        gt=0,
        description="Number of PDF documents to process.",
    )
    subset: str = Field(
        default="eng_Latn",
        description="FinePDFs language subset (e.g. eng_Latn, fra_Latn).",
    )
    timeout: int = Field(
        default=20,
        ge=0,
        description="HTTP download timeout in seconds.",
    )
    dpi: int = Field(
        default=144,
        gt=0,
        description="Render resolution in DPI.",
    )
    max_pages: int = Field(
        default=50,
        gt=0,
        description="Skip documents with more pages than this.",
    )
    min_window_pages: int = Field(
        default=2,
        ge=1,
        description="Minimum pages per window; shorter docs are skipped for windowed output.",
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "01-seed.yaml"


# --------------------------------------------------------------------------- #
# Recipe body — unchanged from upstream public_recipes, parameterized by cfg.
# --------------------------------------------------------------------------- #


def download_pdf(url: str, timeout: int) -> bytes | None:
    """Download a PDF from *url*, returning raw bytes or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        log.warning("Failed to download %s: %s", url, exc)
        return None


def render_pages(pdf_bytes: bytes, dpi: int) -> list[bytes]:
    """Render every page of *pdf_bytes* to PNG, returning a list of raw PNG bytes."""
    import fitz  # pymupdf — imported lazily so --help works without the heavy dep installed

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[bytes] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def png_to_base64(png_bytes: bytes) -> str:
    """Encode raw PNG bytes as a base64 string."""
    return base64.b64encode(png_bytes).decode("ascii")


def adaptive_window_size(n_pages: int) -> int:
    """Choose a window size that scales with document length.

    Short documents get small windows (2 pages) so multi-page questions
    remain feasible; longer documents get larger windows (up to 8) to
    cover more context per seed row.
    """
    if n_pages > 10 and n_pages < 20:
        return 3
    elif n_pages > 20 and n_pages < 30:
        return 4
    elif n_pages > 30 and n_pages < 40:
        return 5
    elif n_pages > 40 and n_pages < 50:
        return 6
    elif n_pages > 50 and n_pages < 60:
        return 7
    elif n_pages > 60:
        return 8
    return 2


def run_seed(cfg: SeedConfig) -> None:
    """Build the three seed parquet files described in the module docstring."""
    import pandas as pd
    from datasets import load_dataset

    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Streaming %d documents from %s (subset=%s)",
        cfg.num_docs,
        FINEPDFS_REPO,
        cfg.subset,
    )

    ds = load_dataset(
        FINEPDFS_REPO,
        name=cfg.subset,
        split="train",
        streaming=True,
    )

    per_page_rows: list[dict] = []
    windowed_rows: list[dict] = []
    whole_doc_rows: list[dict] = []

    docs_processed = 0
    for row in ds:
        if docs_processed >= cfg.num_docs:
            break

        doc_id = row.get("id", f"doc_{docs_processed:06d}")
        url = row["url"]
        date = row.get("date", "")

        pdf_bytes = download_pdf(url, timeout=cfg.timeout)
        if pdf_bytes is None:
            continue

        try:
            page_pngs = render_pages(pdf_bytes, dpi=cfg.dpi)
        except Exception as exc:
            log.warning("Failed to render %s: %s", url, exc)
            continue

        if len(page_pngs) == 0:
            log.warning("No pages rendered for %s, skipping", url)
            continue

        if len(page_pngs) > cfg.max_pages:
            log.info(
                "Skipping %s (%d pages > max_pages %d)",
                url,
                len(page_pngs),
                cfg.max_pages,
            )
            continue

        page_b64s: list[str] = []

        for page_idx, png_bytes in enumerate(page_pngs):
            b64 = png_to_base64(png_bytes)
            page_b64s.append(b64)

            per_page_rows.append(
                {
                    "id": doc_id,
                    "url": url,
                    "date": date,
                    "page_number": page_idx,
                    "total_pages": len(page_pngs),
                    "png_images_base64": json.dumps([b64]),
                }
            )

        whole_doc_rows.append(
            {
                "id": doc_id,
                "url": url,
                "date": date,
                "total_pages": len(page_pngs),
                "png_images_base64": json.dumps(page_b64s),
            }
        )

        n_pages = len(page_b64s)
        win_size = adaptive_window_size(n_pages)
        n_windows = n_pages // win_size
        for i in range(n_windows):
            win_start = i * win_size
            win_end = win_start + win_size
            if win_end - win_start < cfg.min_window_pages:
                break
            windowed_rows.append(
                {
                    "id": doc_id,
                    "url": url,
                    "date": date,
                    "total_pages": n_pages,
                    "start_page": win_start,
                    "end_page": win_end,
                    "window_size": win_end - win_start,
                    "png_images_base64": json.dumps(page_b64s[win_start:win_end]),
                }
            )

        docs_processed += 1
        log.info(
            "[%d/%d] %s — %d pages",
            docs_processed,
            cfg.num_docs,
            url,
            len(page_pngs),
        )

    if not per_page_rows:
        log.error("No documents were successfully processed. Exiting.")
        return

    per_page_path = output_dir / "seed_per_page.parquet"
    windowed_path = output_dir / "seed_windowed.parquet"
    whole_doc_path = output_dir / "seed_whole_document.parquet"

    pd.DataFrame(per_page_rows).to_parquet(per_page_path, index=False)
    if windowed_rows:
        pd.DataFrame(windowed_rows).to_parquet(windowed_path, index=False)
    pd.DataFrame(whole_doc_rows).to_parquet(whole_doc_path, index=False)

    log.info("Per-page seed:       %s (%d rows)", per_page_path, len(per_page_rows))
    log.info("Windowed seed:       %s (%d rows)", windowed_path, len(windowed_rows))
    log.info("Whole-document seed: %s (%d rows)", whole_doc_path, len(whole_doc_rows))


def main(cfg: SeedConfig | None = None) -> None:
    """Entry point. ``cfg`` is supplied when called from the Nemotron CLI;
    when called as a script we parse ``--config`` + dotlist overrides ourselves."""
    if cfg is None:
        cfg = load_recipe_config(DEFAULT_CONFIG_PATH, SeedConfig)
    run_seed(cfg)


if __name__ == "__main__":
    main()
    # Force-exit to avoid hanging on background threads from datasets/fsspec.
    os._exit(0)
