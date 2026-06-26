#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/curate/nemo_curator"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Lightweight JSONL curation via NeMo Curator."""

from __future__ import annotations

import argparse
import os
from ast import literal_eval
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def keep_language(value: str, allowed: set[str]) -> bool:
    score, lang_code = literal_eval(value)
    return lang_code in allowed and score >= 0.0


def ray_client_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg.get("ray") or {})
    if "num_cpus" not in kwargs and os.environ.get("NEMOTRON_CURATOR_RAY_NUM_CPUS"):
        kwargs["num_cpus"] = int(os.environ["NEMOTRON_CURATOR_RAY_NUM_CPUS"])
    return kwargs


def text_filter_stages():
    """Return Filter/ScoreFilter across supported NeMo Curator releases."""
    try:
        from nemo_curator.stages.text.modules import Filter, ScoreFilter
    except ImportError:
        from nemo_curator.stages.text.filters import Filter, ScoreFilter
    return Filter, ScoreFilter


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate JSONL text with NeMo Curator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    if cfg.get("dataset"):
        snapshot_download(**cfg["dataset"])
    allowed_languages = {code.upper() for code in cfg.get("language_codes") or []}
    models = cfg.get("models") or {}
    quality_filters = cfg.get("quality_filters") or {}

    pipeline = Pipeline(name="curate_nemo_curator")
    pipeline.add_stage(JsonlReader(file_paths=cfg["input_glob"], fields=[cfg["text_field"]]))
    if allowed_languages:
        Filter, ScoreFilter = text_filter_stages()
        from nemo_curator.stages.text.filters.fasttext import FastTextLangId

        pipeline.add_stage(
            ScoreFilter(
                FastTextLangId(
                    model_path=models["fasttext_langid"],
                    min_langid_score=quality_filters.get("min_langid_score", 0.0),
                ),
                text_field=cfg["text_field"],
                score_field="language",
            )
        )
        pipeline.add_stage(
            Filter(
                filter_fn=lambda value: keep_language(value, allowed_languages),
                filter_field="language",
            )
        )

    has_word_filter = any(key in quality_filters for key in ("min_words", "max_words"))
    if has_word_filter:
        if not all(key in quality_filters for key in ("min_words", "max_words")):
            raise ValueError("quality_filters must set both min_words and max_words to enable WordCountFilter")
        _, ScoreFilter = text_filter_stages()
        from nemo_curator.stages.text.filters.heuristic import WordCountFilter

        pipeline.add_stage(
            ScoreFilter(
                WordCountFilter(
                    min_words=quality_filters["min_words"],
                    max_words=quality_filters["max_words"],
                ),
                text_field=cfg["text_field"],
            )
        )
    if cfg.get("domains"):
        from nemo_curator.stages.text.classifiers import MultilingualDomainClassifier

        pipeline.add_stage(
            MultilingualDomainClassifier(
                text_field=cfg["text_field"],
                filter_by=cfg["domains"],
                cache_dir=models.get("hf_cache_dir"),
            )
        )
    pipeline.add_stage(JsonlWriter(path=cfg["output_dir"]))

    ray_client = RayClient(**ray_client_kwargs(cfg))
    ray_client.start()
    try:
        pipeline.run()
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
