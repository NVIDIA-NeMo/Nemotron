#!/usr/bin/env python3

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

"""Fertility evaluation for a tokenizer on an eval corpus (HF dataset or local).

Fertility = sum(tokens)/sum(words), tokenized on RAW text (the tokenizer applies
its own normalization). Same flexible corpus input as the extend step. Streams,
so it is memory-safe on 10M+ rows.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from transformers import AutoTokenizer

log = logging.getLogger(__name__)


def _raw_stream(corpus: dict) -> Iterator[str]:
    text_field = corpus.get("text_field", "text")
    fields = (text_field, "tgt", "text", "content")
    limit = int(corpus.get("num_docs", 0)) or float("inf")
    skip = int(corpus.get("skip_docs", 0))
    n = 0

    def gate(v):
        nonlocal n, skip
        if not isinstance(v, str) or not v.strip():
            return None
        if skip > 0:
            skip -= 1
            return None
        n += 1
        return v

    if corpus.get("hf_dataset"):
        from datasets import load_dataset

        ld = {
            "path": corpus["hf_dataset"],
            "name": corpus.get("hf_config", corpus.get("hf_name")),
            "split": corpus.get("hf_split", "train"),
            "streaming": bool(corpus.get("streaming", True)),
        }
        # Pinning explicit shards keeps the eval slice byte-identical across arms
        # and lets it be held out from the extend corpus.
        for key, val in (
            ("data_dir", corpus.get("hf_data_dir")),
            ("data_files", corpus.get("hf_data_files")),
            ("revision", corpus.get("hf_revision")),
        ):
            if val is not None:
                ld[key] = val
        if not ld["streaming"]:
            ld["verification_mode"] = "no_checks"
        log.info("loading HF dataset: %s", ld)
        ds = load_dataset(**ld)
        for ex in ds:
            v = next((ex[f] for f in fields if isinstance(ex.get(f), str) and ex[f].strip()), "")
            g = gate(v)
            if g is not None:
                yield g
                if n >= limit:
                    return
        return

    path = corpus["path"]
    pattern = corpus.get("glob", "*.parquet")
    files = sorted(_glob.glob(os.path.join(path, pattern))) if os.path.isdir(path) else sorted(_glob.glob(path))
    if not files:
        raise FileNotFoundError(
            f"corpus.path matched no files: {path!r} (glob {pattern!r}). A silent empty "
            f"corpus would report fertility 0.0 as if it were a result."
        )
    if files and files[0].endswith(".parquet"):
        import pyarrow.parquet as pq

        for f in files:
            for b in pq.ParquetFile(f).iter_batches(batch_size=8192, columns=[text_field]):
                for v in b.column(0).to_pylist():
                    g = gate(v)
                    if g is not None:
                        yield g
                        if n >= limit:
                            return
    else:
        for f in files:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = next((ex[fl] for fl in fields if isinstance(ex.get(fl), str) and ex[fl].strip()), "")
                g = gate(v)
                if g is not None:
                    yield g
                    if n >= limit:
                        return


def run_fertility(cfg: dict) -> dict:
    t0 = time.time()
    trust = cfg.get("trust_remote_code", False)
    try:
        tok = AutoTokenizer.from_pretrained(
            cfg["tokenizer"], use_fast=True, trust_remote_code=trust, fix_mistral_regex=True
        )
        used_fix = True
    except TypeError:
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], use_fast=True, trust_remote_code=trust)
        used_fix = False

    bs = int(cfg.get("batch_size", 1000))
    tot_words = tot_tokens = tot_chars = n = 0
    uniq: set[int] = set()
    batch: list[str] = []

    def flush(b):
        nonlocal tot_words, tot_tokens, tot_chars, n
        if not b:
            return
        for text, ids in zip(b, tok(b, add_special_tokens=False)["input_ids"]):
            tot_words += len([w for w in text.split() if w.strip()])
            tot_tokens += len(ids)
            tot_chars += len(text)
            uniq.update(ids)
        n += len(b)

    for text in _raw_stream(cfg["corpus"]):
        batch.append(text)
        if len(batch) >= bs:
            flush(batch)
            batch = []
    flush(batch)

    if not tot_words:
        raise RuntimeError(
            "no words were scored — the corpus yielded no documents. Check corpus.path / "
            "glob / text_field (a 0.0 fertility is not a valid measurement)."
        )
    fert = tot_tokens / tot_words
    report = {
        "label": cfg.get("label", cfg["tokenizer"]),
        "tokenizer": cfg["tokenizer"],
        "vocab_size": len(tok),
        "fix_mistral_regex": used_fix,
        "eval_corpus": cfg["corpus"],
        "fertility_definition": "sum(tokens)/sum(words)",
        "totals": {"docs": n, "words": tot_words, "tokens": tot_tokens, "chars": tot_chars},
        "metrics": {
            "fertility": round(fert, 4),
            "chars_per_token": round(tot_chars / tot_tokens, 3) if tot_tokens else 0,
            "unique_tokens_used": len(uniq),
            "vocab_coverage": round(len(uniq) / len(tok), 4) if len(tok) else 0,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out = cfg.get("output")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Fertility %s = %s (%d docs)", report["label"], report["metrics"]["fertility"], n)
    return report
