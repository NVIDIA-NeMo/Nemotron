# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer>=0.5.6",
#     "pydantic>=2",
#     "omegaconf>=2.3",
#     "pyyaml",
# ]
#
# [tool.runspec]
# schema = "1"
# name = "data/sdg/long-document/ocr"
# image = "nvcr.io/nvidia/pytorch:25.12-py3"
# setup = "Inline PEP 723 deps resolved at runtime via `uv run --no-project`."
#
# [tool.runspec.run]
# launch = "direct"
# cmd = "uv run --no-project {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config"
# default = "02-ocr"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Long-Document Understanding Nemotron-Parse OCR Recipe.

Run Nemotron-Parse v1.1 OCR over document images from a seed parquet file.
Each record produces:

  - ``transcribed_texts``: clean text extracted from the OCR output.
  - ``transcribed_texts__metadata``: bounding-box coordinates and class labels.

Prerequisites:

  - A seed parquet file containing a ``png_images_base64`` column with a JSON
    array of base64-encoded PNG images (one element per page; single-page
    seeds have a one-element array).
  - A vLLM-compatible deployment of ``nvidia/NVIDIA-Nemotron-Parse-v1.1``.
    The vLLM server must be launched with a chat template that injects the
    Nemotron-Parse special tokens.  See the long-document README and the
    upstream public_recipes README for chat-template examples and Slurm/Pyxis
    launch commands.

Run standalone (operator-driven):

    uv run --no-project 02-nemotron-parse-ocr-sdg.py \
        --config config/02-ocr.yaml \
        vllm_endpoint=http://localhost:8000/v1 \
        seed_path=./seed_data/seed_per_page.parquet \
        num_records=100

Run via the Nemotron CLI:

    nemotron data sdg long-document ocr --run dlw -c 02-ocr \
        vllm_endpoint=http://node:8000/v1 seed_path=... num_records=100

The recipe itself is a CPU client; the GPUs are on the operator-supplied vLLM
endpoint.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner
from pydantic import BaseModel, ConfigDict, Field

# Pull in the shared YAML+dotlist+Pydantic loader from the sibling helper
# (folder name has a dash so it isn't an importable package).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _recipe_config import load_recipe_config  # noqa: E402  (after sys.path mutation)


NEMOTRON_PARSE_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
VLLM_PROVIDER_NAME = "vllm"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "02-ocr.yaml"

_STRUCTURED_ELEMENT_PATTERN = re.compile(
    r"<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>",
    re.DOTALL,
)


class OcrConfig(BaseModel):
    """Pydantic config for the long-document OCR stage."""

    model_config = ConfigDict(extra="forbid")

    vllm_endpoint: str = Field(
        ...,
        description="Base URL of the vLLM server hosting nemotron-parse "
                    "(e.g. http://localhost:8000/v1).",
    )
    seed_path: Path = Field(
        ...,
        description="Path to the seed parquet file (per-page seeds expected).",
    )
    model_alias: str = Field(
        default="ocr",
        description="Alias under which the OCR model is published by the vLLM server.",
    )
    num_records: int = Field(
        default=5,
        gt=0,
        description="Number of records to generate.",
    )
    artifact_path: Path | None = Field(
        default=None,
        description="Optional directory for Data Designer artifacts; defaults to "
                    "the data-designer-managed location when omitted.",
    )


# --------------------------------------------------------------------------- #
# Recipe body — preserves upstream behavior, parameterized by ``cfg``.
# Heavy ``data_designer`` imports are deferred so the module load stays
# light enough for the CLI's importlib-based config-class discovery.
# --------------------------------------------------------------------------- #


def _extract_structured_elements(text: str) -> list[dict]:
    """Parse Nemotron-Parse bbox markup into structured dicts.

    Input format: ``<x_START><y_START>TEXT<x_END><y_END><class_LABEL>``.

    Returns a list of dicts with keys ``bbox`` (``{x1,y1,x2,y2}``), ``class_label``,
    and ``text``.
    """
    elements = []
    for match in _STRUCTURED_ELEMENT_PATTERN.finditer(text):
        x1, y1, content, x2, y2, class_label = match.groups()
        elements.append({
            "bbox": {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
            },
            "class_label": class_label,
            "text": content.strip(),
        })
    return elements


def _build_config(seed_path: str | Path, model_alias: str):
    @dd.custom_column_generator(
        required_columns=["raw_ocr_output"],
        side_effect_columns=["transcribed_texts__metadata"],
    )
    def parse_ocr_output(row: dict) -> dict:
        """Extract clean text and bbox metadata from raw Nemotron-Parse output."""
        raw = row["raw_ocr_output"]
        elements = _extract_structured_elements(raw)
        row["transcribed_texts"] = "\n".join(el["text"] for el in elements)
        row["transcribed_texts__metadata"] = [
            {"bbox": el["bbox"], "class_label": el["class_label"]}
            for el in elements
        ]
        return row

    model_configs = [
        dd.ModelConfig(
            alias=model_alias,
            model=NEMOTRON_PARSE_MODEL,
            provider=VLLM_PROVIDER_NAME,
            # Health check sends a text-only probe; this model requires image
            # input, so the check would fail. Skip it.
            skip_health_check=True,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=0,
                timeout=60,
                max_parallel_requests=32,
                extra_body={
                    "skip_special_tokens": False,
                    "top_k": 1,
                    "repetition_penalty": 1.1,
                },
            ),
        ),
    ]

    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="raw_ocr_output",
            model_alias=model_alias,
            prompt="",
            multi_modal_context=[
                dd.ImageContext(
                    # Expects a single-element JSON array from the per-page seed.
                    column_name="png_images_base64",
                    data_type=dd.ModalityDataType.BASE64,
                    image_format=dd.ImageFormat.PNG,
                ),
            ],
            drop=True,
        )
    )

    config_builder.add_column(
        dd.CustomColumnConfig(
            name="transcribed_texts",
            generator_function=parse_ocr_output,
        )
    )

    return config_builder


def run_ocr(cfg: OcrConfig) -> None:
    """Run the OCR pipeline against the operator-supplied vLLM endpoint."""
    config_builder = _build_config(seed_path=cfg.seed_path, model_alias=cfg.model_alias)

    model_providers = [
        dd.ModelProvider(name=VLLM_PROVIDER_NAME, endpoint=cfg.vllm_endpoint),
    ]
    data_designer = DataDesigner(
        artifact_path=cfg.artifact_path,
        model_providers=model_providers,
    )
    data_designer.set_run_config(dd.RunConfig(progress_bar=True, disable_early_shutdown=True))
    results = data_designer.create(
        config_builder, num_records=cfg.num_records, dataset_name="nemotron_parse_ocr"
    )

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    results.load_analysis().to_report()


def main(cfg: OcrConfig | None = None) -> None:
    """Entry point. ``cfg`` is supplied when called from the Nemotron CLI;
    when called as a script we parse ``--config`` + dotlist overrides ourselves."""
    if cfg is None:
        cfg = load_recipe_config(DEFAULT_CONFIG_PATH, OcrConfig)
    run_ocr(cfg)


if __name__ == "__main__":
    main()
