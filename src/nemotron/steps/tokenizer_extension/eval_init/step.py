#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/tokenizer_extension/eval_init"
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
# gpus_per_node = 8
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

"""Score extended (resized / CPT'd) checkpoints for embedding-init quality.

Reports cross-entropy loss, perplexity, and bits-per-byte (BPB) on a validation
corpus, optionally alongside the unextended base model. BPB is the cross-model
metric: per-token loss/PPL are NOT comparable across different vocabularies,
BPB (normalized by UTF-8 bytes) is. Wraps the bpb.py engine via its argparse
main().
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
# package root too: languages.py / script_ranges.py are shared across steps
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bpb as engine  # noqa: E402

log = logging.getLogger(__name__)


def _build_argv(cfg: dict) -> list[str]:
    models = cfg.get("models") or []
    if isinstance(models, str):
        models = [models]
    if not models:
        raise ValueError("config must set `models` (one or more checkpoint paths).")
    corpus = cfg.get("corpus") or {}
    data_file = cfg.get("data_file") or corpus.get("path")
    hf_dataset = cfg.get("hf_dataset") or corpus.get("hf_dataset")
    if not data_file and not hf_dataset:
        raise ValueError(
            "config must set `data_file` (local .jsonl/.txt) or a `corpus.hf_dataset` "
            "(streamed HF validation corpus, e.g. ai4bharat/samanantar)."
        )

    argv: list[str] = ["--models", *[str(m) for m in models]]
    # Off by default: executing model-repo code is opt-in, set it in the config.
    if bool(cfg.get("trust_remote_code", False)):
        argv += ["--trust-remote-code"]
    if cfg.get("base_model"):
        argv += ["--base-model", str(cfg["base_model"])]
    if data_file:
        argv += ["--data-file", str(data_file)]
    else:
        argv += ["--hf-dataset", str(hf_dataset)]
        hf_config = cfg.get("hf_config") or corpus.get("hf_config")
        if hf_config:
            argv += ["--hf-config", str(hf_config)]
        argv += ["--hf-split", str(cfg.get("hf_split") or corpus.get("hf_split", "train"))]
        argv += ["--skip-docs", str(int(cfg.get("skip_docs", corpus.get("skip_docs", 0))))]
    text_field = cfg.get("text_field") or corpus.get("text_field", "text")
    argv += ["--text-field", str(text_field)]
    argv += ["--max-docs", str(int(cfg.get("max_docs", -1)))]
    if bool(cfg.get("allow_token_cap_comparison", False)):
        argv += ["--allow-token-cap-comparison"]
    argv += ["--max-tokens", str(int(cfg.get("max_tokens", -1)))]
    argv += ["--max-length", str(int(cfg.get("max_length", 2048)))]
    argv += ["--stride", str(int(cfg.get("stride", 512)))]
    argv += ["--dtype", str(cfg.get("dtype", "bfloat16"))]
    if cfg.get("device"):
        argv += ["--device", str(cfg["device"])]
    if cfg.get("output_json"):
        Path(cfg["output_json"]).parent.mkdir(parents=True, exist_ok=True)
        argv += ["--output-json", str(cfg["output_json"])]
    return argv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", default=str(Path(__file__).parent / "config" / "default.yaml"))
    args, _ = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    argv = _build_argv(cfg)
    log.info("MILESTONE: BPB eval | argv=%s", " ".join(argv))
    engine.main(argv)
    log.info("MILESTONE: DONE — BPB eval complete (output_json=%s)", cfg.get("output_json"))


if __name__ == "__main__":
    main()
