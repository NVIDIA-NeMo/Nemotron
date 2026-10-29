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

"""Config-driven tokenizer extension: train one BPE, emit Add and/or Replace arms.

Corpus input is flexible (config `corpus:` block):
  * HuggingFace dataset  -> hf_dataset / hf_config / hf_split (streamed)
  * Local parquet dir    -> path + glob (cross-shard sampled)
  * Local jsonl file(s)   -> path (glob *.jsonl)
plus `text_field` (the column holding the text) in every case.

Reuses the standalone cores (continued_bpe.py = Add core + splice,
replace_bpe.py = prune + NFKC wrap). The splice is the constructive/rank-dead-safe
one, and every built arm is asserted free of rank-dead tokens.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from continued_bpe import (
    batch_iterator,
    clean_text,
    compute_continued_bpe_artifacts,
    extend_tokenizer,
    find_rank_dead_tokens,
)
from languages import get_normalizer
from languages import describe as describe_language
from languages import resolve as resolve_language
from replace_bpe import (
    identify_script_tokens,
    prune_backend,
    resolve_ranges,
    wrap_fast,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Corpus reader (HF dataset name OR local parquet dir OR local jsonl)
# --------------------------------------------------------------------------- #
def _require_text_field(example: Any, text_field: str, where: str) -> None:
    """Refuse a corpus whose records do not carry the configured text column.

    Reading a different column trains a different tokenizer, and the result is
    indistinguishable from the one that was asked for, so the column is checked
    rather than guessed.
    """
    try:
        keys = sorted(example.keys())
    except AttributeError:
        return
    if text_field in keys:
        return
    raise ValueError(
        f"corpus.text_field={text_field!r} is not a column of {where}. Available: {keys}."
    )


def corpus_stream(corpus: dict, normalizer: Any) -> Iterator[str]:
    text_field = corpus.get("text_field", "text")
    if corpus.get("hf_dataset") and corpus.get("path"):
        raise ValueError(
            f"corpus: set either `hf_dataset` or `path`, not both "
            f"(hf_dataset={corpus['hf_dataset']!r}, path={corpus['path']!r}). "
            "Two sources cannot be read as one corpus, and silently picking one "
            "trains on data the config does not name."
        )
    max_samples = int(corpus.get("samples", 1_000_000))
    max_doc_chars = int(corpus.get("max_doc_chars", 0))

    def emit(raw: str) -> str | None:
        if not isinstance(raw, str) or not raw.strip():
            return None
        cleaned = clean_text(raw, normalizer)
        if max_doc_chars > 0:
            cleaned = cleaned[:max_doc_chars]
        return cleaned if len(cleaned) > 50 else None

    yielded = 0

    # 1) HuggingFace dataset. Generic: HF exposes a dataset as
    #    path -> name (config / subset) -> split, plus optional data_dir /
    #    data_files / revision. The SAME language is named differently across
    #    datasets (sangraha: name=verified, split=hin ; samanantar: name=hi,
    #    split=train), so we pass through whatever the config sets.
    if corpus.get("hf_dataset"):
        from datasets import load_dataset

        streaming = bool(corpus.get("streaming", True))
        ld: dict[str, Any] = {"path": corpus["hf_dataset"], "streaming": streaming}
        name = corpus.get("hf_name", corpus.get("hf_config"))  # the config/subset level
        for key, val in (
            ("name", name),
            ("split", corpus.get("hf_split")),
            ("data_dir", corpus.get("hf_data_dir")),
            ("data_files", corpus.get("hf_data_files")),
            ("revision", corpus.get("hf_revision")),
        ):
            if val is not None:
                ld[key] = val
        if not streaming:
            # some datasets (e.g. ai4bharat/sangraha) record 0 expected examples in
            # dataset_infos, which trips split-size verification on a real download.
            ld["verification_mode"] = "no_checks"
        log.info("MILESTONE: loading HF dataset (streaming=%s): %s", streaming, ld)
        try:
            ds = load_dataset(**ld)
        except Exception as e:  # surface the valid configs/splits to fix the config fast
            hint = ""
            try:
                from datasets import get_dataset_config_names, get_dataset_split_names

                cfgs = get_dataset_config_names(corpus["hf_dataset"])
                hint = f" | available name(config)= {cfgs}"
                if name in cfgs:
                    hint += f" ; split= {get_dataset_split_names(corpus['hf_dataset'], name)}"
            except Exception:
                pass
            raise SystemExit(f"load_dataset failed for {ld}: {e}.{hint}")
        if not streaming:
            log.info(
                "MILESTONE: dataset ready (%s rows cached); iterating first %s...",
                f"{ds.num_rows:,}" if hasattr(ds, "num_rows") else "?",
                f"{max_samples:,}",
            )
        checked = False
        for ex in ds:
            if not checked:
                _require_text_field(ex, text_field, f"HF dataset {corpus['hf_dataset']!r}")
                checked = True
            value = ex.get(text_field)
            c = emit(value if isinstance(value, str) else "")
            if c is not None:
                yield c
                yielded += 1
                if yielded % 100000 == 0:
                    log.info("MILESTONE: processed %s docs into BPE trainer...", f"{yielded:,}")
                if yielded >= max_samples:
                    return
        return

    # 2) Local path (parquet dir/glob or jsonl)
    path = corpus.get("path")
    if not path:
        raise ValueError("corpus: set either `hf_dataset` or `path`.")
    pattern = corpus.get("glob", "*.parquet")
    files = sorted(_glob.glob(os.path.join(path, pattern))) if os.path.isdir(path) else sorted(_glob.glob(path))
    if not files:
        raise SystemExit(f"No files match corpus path {path!r} glob {pattern!r}")

    if files[0].endswith(".parquet"):
        import pyarrow.parquet as pq

        diversify = bool(corpus.get("diversify", True))
        per_shard = math.ceil(max_samples / len(files)) if diversify else max_samples
        for fpath in files:
            if yielded >= max_samples:
                return
            taken = 0
            for batch in pq.ParquetFile(fpath).iter_batches(batch_size=8192, columns=[text_field]):
                for v in batch.column(0).to_pylist():
                    c = emit(v)
                    if c is None:
                        continue
                    yield c
                    yielded += 1
                    taken += 1
                    if yielded % 100000 == 0:
                        log.info("streamed %s docs for BPE training...", f"{yielded:,}")
                    if yielded >= max_samples:
                        return
                    if taken >= per_shard:
                        break
                if taken >= per_shard:
                    break
    else:  # jsonl
        checked = False
        for fpath in files:
            with open(fpath, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not checked:
                        _require_text_field(ex, text_field, fpath)
                        checked = True
                    value = ex.get(text_field)
                    c = emit(value if isinstance(value, str) else "")
                    if c is None:
                        continue
                    yield c
                    yielded += 1
                    if yielded % 100000 == 0:
                        log.info("streamed %s docs for BPE training...", f"{yielded:,}")
                    if yielded >= max_samples:
                        return


def _load_base(model_id: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=trust_remote_code, fix_mistral_regex=True
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)


def _build_arm(base_like, trained_tok, base_tok, extension_size: int):
    art = compute_continued_bpe_artifacts(base_like.backend_tokenizer, trained_tok.backend_tokenizer)
    merges = [tuple(x.split(" ")) for x in art.new_merges if len(x.split(" ")) == 2]
    before = len(base_like)
    ext = extend_tokenizer(base_like, art.new_vocab, merges, n_tokens=extension_size)
    ext = wrap_fast(ext.backend_tokenizer, base_tok)
    dead = find_rank_dead_tokens(ext, [t for t, i in ext.get_vocab().items() if i >= before])
    if dead:
        raise RuntimeError(f"{len(dead):,} spliced tokens are rank-dead, e.g. {dead[:5]}")
    return ext, len(art.new_vocab), len(ext) - before


def run_extension(cfg: dict) -> dict:
    """Train one BPE from `cfg` and build ONE arm (add OR replace).

    Add and Replace are deliberately separate jobs (run them one after another;
    the first downloads/caches the dataset, the second reuses it). Emits milestone
    logs so a long remote run is never silent.
    """
    import time

    t0 = time.time()
    root_dir = Path(cfg.get("output_dir", "./output/tokenizer_extension"))
    method = cfg.get("method")
    if method not in ("add", "replace", "expand"):
        raise ValueError(f"method must be 'add', 'replace', or 'expand', got {method!r}")

    # One arm per job, each in its own subdirectory: `add` and `replace` are built
    # from the same corpus and would otherwise overwrite each other in a shared
    # output_dir. Downstream configs reference output_dir/<method>.
    out_dir = root_dir / method
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_size = int(cfg.get("extension_size", 30000))
    corpus = cfg["corpus"]
    min_freq = int(corpus.get("min_frequency", 0))
    # `language:` resolves the normalizer and the prune script together;
    # `script_normalizer:` / `remove_script:` still override either one.
    script_norm, remove_script = resolve_language(cfg)
    language_resolution = describe_language(cfg)
    normalizer = get_normalizer(script_norm)
    log.info(
        "language=%s -> script_normalizer=%s remove_script=%s",
        cfg.get("language", "(legacy devanagari default)"),
        script_norm,
        remove_script,
    )

    log.info("MILESTONE: loading base tokenizer %s ...", cfg.get("model_id"))
    base_tok = _load_base(cfg.get("model_id"), cfg.get("trust_remote_code", False))
    base_size = len(base_tok)
    log.info(
        "MILESTONE: base loaded (vocab=%d) | method=%s ext_size=%d min_frequency=%d",
        base_size,
        method,
        ext_size,
        min_freq,
    )

    log.info("MILESTONE: corpus + BPE training start (target vocab=%d)...", base_size + ext_size)
    t_train = time.time()
    train_kwargs = {"vocab_size": base_size + ext_size}
    if min_freq > 0:
        train_kwargs["min_frequency"] = min_freq
    stream = corpus_stream(corpus, normalizer)
    trained_tok = base_tok.train_new_from_iterator(
        batch_iterator(stream, int(cfg.get("batch_size", 1000))), **train_kwargs
    )
    train_sec = round(time.time() - t_train, 1)
    log.info("MILESTONE: BPE training done in %ss (trained vocab=%d).", train_sec, len(trained_tok))

    summary: dict[str, Any] = {
        "model_id": cfg.get("model_id"),
        "method": method,
        "language": language_resolution["language"],
        "language_source": language_resolution["language_source"],
        "script_normalizer": language_resolution["script_normalizer"],
        "remove_script": language_resolution["remove_script"],
        "language_overrides": language_resolution["overrides"],
        "language_profile_defaults": language_resolution["profile_defaults"],
        "extension_size": ext_size,
        "corpus": corpus,
        "base_vocab_size": base_size,
        "timings_sec": {"train": train_sec},
    }

    log.info("MILESTONE: building %s tokenizer (splice + rank-dead check)...", method)
    t_build = time.time()
    if method == "add":
        arm_tok, cand, spliced = _build_arm(base_tok, trained_tok, base_tok, ext_size)
    elif method == "expand":
        # Naive expansion: decode the novel BPE tokens to Unicode and add_tokens()
        # them ATOMICALLY (no merge rules) — the baseline arm.
        base_keys = set(base_tok.get_vocab().keys())
        tvocab = trained_tok.get_vocab()
        # Rank order over ALL novel tokens, not a pre-truncated slice: the strip()
        # below collapses `Ġfoo` and `foo` onto one surface, so slicing to ext_size
        # first silently delivered FEWER than the requested rows. Take candidates in
        # rank order until ext_size distinct surfaces are collected.
        new_keys = sorted((t for t in tvocab if t not in base_keys), key=lambda t: tvocab[t])
        # NB: byte-level BPE encodes a leading space into word-initial tokens. Adding the
        # bare decoded surface orphans that space (stray `Ġ`), inflating fertility ~+0.24.
        # Add with lstrip=True + stripped surface so the token absorbs the preceding space.
        from tokenizers import AddedToken

        # add_tokens() silently drops surfaces already present in the base vocab, so
        # a single pass can splice fewer than ext_size rows. Keep pulling candidates
        # in rank order until ext_size are actually spliced (or we run out).
        def _mk(surface: str) -> AddedToken:
            return AddedToken(surface, lstrip=True, rstrip=False, normalized=False, single_word=False)

        seen: set[str] = set()
        pending = iter(new_keys)
        new_unicode: list[str] = []
        spliced = 0
        while spliced < ext_size:
            batch: list[str] = []
            for bt in pending:
                surface = trained_tok.convert_tokens_to_string([bt]).strip()
                if surface and surface not in seen:
                    seen.add(surface)
                    batch.append(surface)
                    if len(batch) >= ext_size - spliced:
                        break
            if not batch:
                break  # candidates exhausted
            added = int(base_tok.add_tokens([_mk(x) for x in batch]))
            new_unicode.extend(batch)
            spliced += added
        cand = len(new_unicode)
        if spliced < ext_size:
            log.warning(
                "expand: spliced %d of the requested %d tokens from %d novel "
                "candidates (%d distinct surfaces tried; the rest collided with "
                "the base vocab). Widen the corpus or lower extension_size.",
                spliced,
                ext_size,
                len(new_keys),
                cand,
            )
        arm_tok = base_tok
    else:  # replace
        ranges = resolve_ranges([s for s in remove_script.split(",") if s.strip()])
        remove_tokens = identify_script_tokens(base_tok, ranges)
        pruned_backend, old2new, removed_ids, removed_list = prune_backend(base_tok.backend_tokenizer, remove_tokens)
        pruned_base = wrap_fast(pruned_backend, base_tok)
        arm_tok, cand, spliced = _build_arm(pruned_base, trained_tok, base_tok, ext_size)
        summary["removed"] = len(removed_ids)
    # Nothing was added, so the output would be the base tokenizer under another
    # name. Saved, it is indistinguishable from a successful build and every
    # comparison made against it is a comparison with the base.
    if spliced == 0:
        raise ValueError(
            f"method={method} spliced 0 tokens from {cand} candidate(s): the result would be "
            f"identical to the base tokenizer, so nothing was written to {out_dir}. Widen the "
            "corpus (check corpus.text_field, corpus.samples, and that documents survive the "
            "50-character minimum), or lower min_frequency."
        )

    arm_tok.save_pretrained(out_dir)
    if method == "replace":
        (out_dir / "id_remap.json").write_text(json.dumps({str(k): v for k, v in old2new.items()}))
        (out_dir / "removed_tokens.txt").write_text("\n".join(removed_list))

    summary.update(
        {
            "final_vocab_size": len(arm_tok),
            "new_candidates": cand,
            "tokens_requested": int(ext_size),
            "tokens_spliced": spliced,
            "output": str(out_dir),
        }
    )
    # Record the budget alongside the result so any artifact can be audited after
    # the fact. Constructive merging can add intermediate tokens as well as the
    # target, so `spliced` is not guaranteed to equal the request; comparisons
    # across arms are only valid at a matched budget.
    if spliced != ext_size:
        log.warning(
            "tokens_spliced (%d) != extension_size (%d) for method=%s. Fertility "
            "and BPB comparisons are only valid between arms with the same "
            "tokens_spliced.",
            spliced,
            ext_size,
            method,
        )
    summary["timings_sec"]["build"] = round(time.time() - t_build, 1)
    summary["timings_sec"]["total"] = round(time.time() - t0, 1)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info(
        "MILESTONE: DONE — %s tokenizer saved -> %s (vocab=%d, spliced=%d, total=%ss).",
        method,
        out_dir,
        summary["final_vocab_size"],
        spliced,
        summary["timings_sec"]["total"],
    )
    return summary
