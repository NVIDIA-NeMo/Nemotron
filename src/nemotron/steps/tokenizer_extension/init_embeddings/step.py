#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/tokenizer_extension/init_embeddings"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
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
# gpus_per_node = 1
# ///

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attach an extended tokenizer to the base model, initialize the new embedding
rows, and save a resized HF checkpoint ready for CPT (pretrain/megatron_bridge).

The init strategy is chosen by `method:` (baseline | subword | focus) and the
arm by `arm:` (add | replace); embeddings.py dispatches to a per-technique
engine script. Single process.

GPU use: the base model is embedding surgery only (no forward pass), so every
engine loads it WITHOUT device_map -- it is host-resident (~60 GB bf16 for a
30B model). GPUs are used only by the auxiliary encoders: MuRIL-class models
for subword.input/output_averaging=bert_weighted, and Gemma (sharded with
device_map='auto') for gemma_weighted. The 8-GPU request therefore only pays
for itself on the gemma path; other methods leave them idle.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
# package root too: languages.py / script_ranges.py are shared across steps
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embeddings import run_init  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args, _ = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_init(cfg)


if __name__ == "__main__":
    main()
