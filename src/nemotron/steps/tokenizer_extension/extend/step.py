#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/tokenizer_extension/extend"
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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenizer extension (Add / Replace) — CPU-only.

Trains one BPE on the configured corpus and splices it into the base tokenizer
(Add) and/or a pruned base (Replace). The splice is rank-dead-safe. YAML drives
everything; see config/default.yaml.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# self-contained step: import the sibling modules that live next to this executor
sys.path.insert(0, str(Path(__file__).resolve().parent))
# package root too: languages.py / script_ranges.py are shared across steps
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extension import run_extension  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args, _ = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_extension(cfg)


if __name__ == "__main__":
    main()
