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

"""Data acquisition and curation via NeMo Curator — reference implementation."""
from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.classifiers import MultilingualDomainClassifier
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.filters.fasttext import FastTextLangId
from nemo_curator.stages.text.filters.heuristic.string import WordCountFilter
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def keep_language(value: str, allowed: set[str]) -> bool:
    score, lang_code = literal_eval(value)
    return lang_code in allowed and score >= 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and curate text with NeMo Curator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    snapshot_download(**cfg["dataset"])
    allowed_languages = {code.upper() for code in cfg["language_codes"]}

    pipeline = Pipeline(name="curate_nemo_curator")
    pipeline.add_stage(JsonlReader(file_paths=cfg["input_glob"], fields=[cfg["text_field"]]))
    pipeline.add_stage(
        ScoreFilter(
            FastTextLangId(
                model_path=cfg["models"]["fasttext_langid"],
                min_langid_score=cfg["quality_filters"]["min_langid_score"],
            ),
            text_field=cfg["text_field"],
            score_field="language",
        )
    )
    pipeline.add_stage(Filter(filter_fn=lambda value: keep_language(value, allowed_languages), filter_field="language"))
    pipeline.add_stage(
        ScoreFilter(
            WordCountFilter(
                min_words=cfg["quality_filters"]["min_words"],
                max_words=cfg["quality_filters"]["max_words"],
            ),
            text_field=cfg["text_field"],
        )
    )
    pipeline.add_stage(
        MultilingualDomainClassifier(
            text_field=cfg["text_field"],
            filter_by=cfg.get("domains") or None,
            cache_dir=cfg["models"].get("hf_cache_dir"),
        )
    )
    pipeline.add_stage(JsonlWriter(path=cfg["output_dir"]))

    ray_client = RayClient()
    ray_client.start()
    try:
        pipeline.run()
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
