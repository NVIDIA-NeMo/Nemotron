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

"""Continued-BPE core: train a BPE on a corpus and splice it into a base tokenizer.

Two invariants this module maintains:

* **Bounded merge fan-out.** New tokens are added with dependency backtracking, so
  only the requested tokens and the topological parents they need to be emittable
  enter the vocabulary.
* **Device-safe embedding surgery.** ``modify_embeddings`` tracks the device of
  each tensor explicitly, so a model sharded across GPUs pools correctly.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:  # tokenizer-build path is torch-free; only the embedding-init helpers need torch
    import torch
except ModuleNotFoundError:
    torch = None
from datasets import get_dataset_split_names, interleave_datasets, load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONSTANTS & DATA STREAMING
# =============================================================================
DATASET_ID = "ai4bharat/sangraha"
DATASET_CONFIG = "verified"
DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\[\]{}|\\^`\"']+")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
MULTISPACE_RE = re.compile(r"[ \t]+")


def get_devanagari_normalizer() -> Any:
    try:
        from indicnlp.normalize.indic_normalize import DevanagariNormalizer

        return DevanagariNormalizer()
    except ImportError:
        logger.warning("indic-nlp-library missing; skipping Devanagari Normalizer.")
        return None


def clean_text(text: str, devanagari_norm: Any) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)
    if devanagari_norm:
        text = devanagari_norm.normalize(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def _normalize_lang(tag: str) -> str:
    return tag.strip().lower().replace("-", "_")


def _base_language_from_split(split_name: str) -> str:
    # Sangraha split names are typically like hin_Deva / hin_Latn.
    return split_name.split("_", 1)[0].lower()


def _resolve_sangraha_splits(requested_languages: list[str], dataset_config: str) -> list[str]:
    available_splits = get_dataset_split_names(DATASET_ID, dataset_config)
    if not requested_languages:
        return available_splits

    requested = {_normalize_lang(lang) for lang in requested_languages}
    resolved = []
    for split_name in available_splits:
        split_norm = _normalize_lang(split_name)
        base_lang = _base_language_from_split(split_name)
        if split_norm in requested or base_lang in requested:
            resolved.append(split_name)

    if not resolved:
        raise ValueError(
            f"No matching Sangraha splits for languages={requested_languages}. "
            f"Available splits include: {available_splits[:10]}..."
        )
    return resolved


def _extract_text(example: dict[str, Any]) -> str:
    for field in ("text", "content", "response", "prompt"):
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _derive_language(example: dict[str, Any], split_name: str) -> str:
    row_language = example.get("language")
    if isinstance(row_language, str) and row_language.strip():
        return row_language.strip().lower()
    return split_name.strip().lower()


def mixed_language_text_stream(
    languages: list[str],
    max_samples_per_lang: int,
    devanagari_norm: Any,
    pbar: tqdm | None,
    dataset_config: str,
    doc_id_to_language: dict[str, str] | None = None,
) -> Iterator[str]:
    source_splits = _resolve_sangraha_splits(languages, dataset_config)
    datasets = []
    if doc_id_to_language is None:
        doc_id_to_language = {}

    for split_name in source_splits:
        ds = load_dataset(DATASET_ID, dataset_config, split=split_name, streaming=True)
        ds = ds.map(lambda ex, s=split_name: {"__split__": s})
        ds = ds.take(max_samples_per_lang)
        datasets.append(ds)

    mixed_ds = interleave_datasets(datasets)
    for example in mixed_ds:
        doc_id_obj = example.get("doc_id", example.get("id"))
        if doc_id_obj is not None:
            doc_id_to_language[str(doc_id_obj)] = _derive_language(example, split_name=example.get("__split__", ""))

        raw_text = _extract_text(example)
        if isinstance(raw_text, str) and raw_text.strip():
            cleaned = clean_text(raw_text, devanagari_norm)
            if len(cleaned) > 50:
                yield cleaned
                if pbar is not None:
                    pbar.update(1)

    logger.info(f"Collected doc_id->language mappings: {len(doc_id_to_language)}")


def batch_iterator(stream: Iterable[str], batch_size: int = 1000) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in stream:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# =============================================================================
# 2. BPE ARTIFACTS EXTRACTION
# =============================================================================
@dataclass(frozen=True)
class ExtensionArtifacts:
    new_vocab: dict[str, int]
    new_merges: list[str]


def _get_bpe_state(tokenizer_backend_str: str) -> tuple[dict[str, int], list[str]]:
    obj = json.loads(tokenizer_backend_str)
    model = obj.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"Expected BPE backend, got {model.get('type')!r}")

    vocab = model.get("vocab", {})
    merges_raw = model.get("merges", [])
    merges: list[str] = []
    for m in merges_raw:
        if isinstance(m, str):
            merges.append(m)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            merges.append(f"{m[0]} {m[1]}")
    return vocab, merges


def compute_continued_bpe_artifacts(base_backend: Tokenizer, trained_backend: Tokenizer) -> ExtensionArtifacts:
    base_vocab, base_merges = _get_bpe_state(base_backend.to_str())
    trained_vocab, trained_merges = _get_bpe_state(trained_backend.to_str())

    base_merges_set = set(base_merges)
    new_merges = [m for m in trained_merges if m not in base_merges_set]

    base_vocab_set = set(base_vocab.keys())
    new_vocab = {t: int(trained_vocab[t]) for t in trained_vocab.keys() if t not in base_vocab_set}
    return ExtensionArtifacts(new_vocab=new_vocab, new_merges=new_merges)


# =============================================================================
# 3. EXTENSION APPLICATION & DEPENDENCY RESOLUTION
# =============================================================================
def _rank_merges(merges: list[str]) -> dict[tuple[str, str], int]:
    """Map each merge pair to its priority (lower fires first, first rule wins)."""
    ranks: dict[tuple[str, str], int] = {}
    for i, rule in enumerate(merges):
        parts = rule.split(" ")
        if len(parts) == 2 and (parts[0], parts[1]) not in ranks:
            ranks[(parts[0], parts[1])] = i
    return ranks


def _apply_merges(symbols: list[str], ranks: dict[tuple[str, str], int]) -> list[str]:
    """Run BPE over `symbols`, always applying the lowest-ranked applicable pair."""
    pieces = list(symbols)
    while len(pieces) > 1:
        best_rank: int | None = None
        best_at = -1
        for i in range(len(pieces) - 1):
            rank = ranks.get((pieces[i], pieces[i + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank, best_at = rank, i
        if best_rank is None:
            break
        pieces[best_at : best_at + 2] = [pieces[best_at] + pieces[best_at + 1]]
    return pieces


def _apply_bpe_extension_backend(
    base_backend: Tokenizer,
    new_vocab: dict[str, int],
    new_merges: list[str] | None,
    n_tokens: int,
    keep_added_token_positions: bool,
) -> Tokenizer:
    """Splice `new_vocab` in so that every added token is actually producible.

    Copying the freshly trained merge rules verbatim is unsafe: spliced rules land
    below every surviving base rule, so a base rule consuming the same symbols
    earlier can make the trained path unreachable. The token then sits in the vocab
    but can never be emitted inside a word, which inflates fertility while every
    graph-reachability check still passes.

    Each token is therefore built from the pieces the *current* merge table yields,
    so the rule appended is the one that will fire. Tokens are processed in trained
    priority order, and ids are appended densely in that order, which is what
    `keep_added_token_positions` used to request; `new_merges` is retained only to
    select between vocab-only and merge-based splicing.
    """
    obj = json.loads(base_backend.to_str())
    model = obj.get("model", {})
    vocab: dict[str, int] = model["vocab"]

    merges_raw = model.get("merges", [])
    merges: list[str] = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in merges_raw]

    # Candidates in trained-priority order. Deliberately NOT pre-truncated to
    # n_tokens: a candidate may have to be skipped (already present, not
    # buildable from the byte alphabet, or needing more rows than remain), and
    # slicing first would then silently deliver fewer rows than requested --
    # the same trap the expand arm documents.
    ranked = [t for (t, _) in sorted(new_vocab.items(), key=lambda kv: kv[1])]

    next_id = (max(vocab.values()) + 1) if vocab else 0
    budget = int(n_tokens)
    added = 0

    if not new_merges:
        for tok in ranked:
            if added >= budget:
                break
            if tok in vocab:
                continue
            vocab[tok] = next_id
            next_id += 1
            added += 1
        model["vocab"] = vocab
        return Tokenizer.from_str(json.dumps(obj))

    ranks = _rank_merges(merges)
    alphabet = {tok for tok in vocab if len(tok) == 1}

    for tok in ranked:
        if added >= budget:
            break
        if tok in vocab:
            continue
        # Anything outside the byte alphabet cannot be rebuilt from merges at all.
        if any(symbol not in alphabet for symbol in tok):
            continue

        pieces = _apply_merges(list(tok), ranks)

        # Building `tok` may also have to add the intermediate pieces it merges
        # through. Price the whole chain before committing to it: stopping
        # part-way would leave an intermediate in the vocab with no rule that
        # produces `tok`, which is the unreachable-token failure this function
        # exists to avoid. In trained-priority order a token's pieces are
        # already in the vocab, so `cost` is normally 1 and this never binds.
        chain: list[str] = []
        left = pieces[0]
        for right in pieces[1:]:
            left = left + right
            chain.append(left)
        if not chain:
            chain = [left]
        cost = sum(1 for piece in chain if piece not in vocab)

        if added + cost > budget:
            # This token needs more rows than remain. Skip it and keep scanning:
            # cost varies per token, so a later, lower-priority candidate can be
            # cheaper and land the arm on exactly `extension_size`. Stopping here
            # instead would underfill the budget.
            continue

        left = pieces[0]
        for right in pieces[1:]:
            if (left, right) not in ranks:
                ranks[(left, right)] = len(merges)
                merges.append(f"{left} {right}")
            combined = left + right
            if combined not in vocab:
                vocab[combined] = next_id
                next_id += 1
                added += 1
            left = combined

        if left not in vocab:
            vocab[left] = next_id
            next_id += 1
            added += 1

    model["vocab"] = vocab
    model["merges"] = merges
    return Tokenizer.from_str(json.dumps(obj))


def extend_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    new_vocab: dict[str, int],
    new_merges: list[tuple[str, str]] | None,
    n_tokens: int,
    keep_added_token_positions: bool = False,
) -> PreTrainedTokenizerFast:
    merges_list = [" ".join(x) for x in new_merges] if new_merges else None
    updated_backend = _apply_bpe_extension_backend(
        base_backend=Tokenizer.from_str(tokenizer.backend_tokenizer.to_str()),
        new_vocab=new_vocab,
        new_merges=merges_list,
        n_tokens=n_tokens,
        keep_added_token_positions=keep_added_token_positions,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=updated_backend,
        unk_token=getattr(tokenizer, "unk_token", None),
        bos_token=getattr(tokenizer, "bos_token", None),
        eos_token=getattr(tokenizer, "eos_token", None),
        pad_token=getattr(tokenizer, "pad_token", None),
    )


# =============================================================================
# 4. MODEL MODIFICATION
# =============================================================================
InitMethod = Literal["mean", "mean_of_constituents"]


def modify_embeddings(
    model: PreTrainedModel,
    old_tokenizer: PreTrainedTokenizerBase,
    new_tokenizer: PreTrainedTokenizerBase,
    init_method: InitMethod = "mean_of_constituents",
    ignore_size_mismatch: bool = False,
) -> dict[str, Any]:
    old_n, new_n = len(old_tokenizer), len(new_tokenizer)

    if not ignore_size_mismatch and model.get_input_embeddings().weight.shape[0] != old_n:
        raise ValueError(f"Model rows ({model.get_input_embeddings().weight.shape[0]}) != old tokenizer ({old_n}).")

    changes: dict[str, Any] = {"old_vocab_size": old_n, "new_vocab_size": new_n, "initialized": []}
    model.resize_token_embeddings(new_n)

    in_emb = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    out_emb = out_layer.weight if out_layer is not None else None

    # Calculate global means (detach to prevent graph buildup)
    global_mean_in = in_emb[:old_n].mean(dim=0).detach()
    global_mean_out = out_emb[:old_n].mean(dim=0).detach() if out_emb is not None else None

    with torch.no_grad():
        for tid in tqdm(range(old_n, new_n), desc="Initializing Embeddings"):
            tok = new_tokenizer.convert_ids_to_tokens(tid)
            vec_in = global_mean_in.clone()
            vec_out = global_mean_out.clone() if global_mean_out is not None else None

            if init_method == "mean_of_constituents":
                text = new_tokenizer.convert_tokens_to_string([tok])
                enc = old_tokenizer(text, add_special_tokens=False)
                ids = [i for i in enc.get("input_ids", []) if 0 <= i < old_n]
                if ids:
                    # Track each tensor's device: the model may be sharded across GPUs.
                    ids_tensor_in = torch.tensor(ids, device=in_emb.device)
                    vec_in = in_emb[ids_tensor_in].mean(dim=0)

                    if out_emb is not None:
                        ids_tensor_out = torch.tensor(ids, device=out_emb.device)
                        vec_out = out_emb[ids_tensor_out].mean(dim=0)

            # Ensure tensors are moved to the correct device partition before copying
            in_emb[tid].copy_(vec_in.to(in_emb.device))
            if out_emb is not None and vec_out is not None:
                out_emb[tid].copy_(vec_out.to(out_emb.device))

            changes["initialized"].append({"id": tid, "token": tok, "method": init_method})

    return changes


# =============================================================================
# 5. BENCHMARKING & PRUNING
# =============================================================================
def find_unreachable_tokens_merges(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    vocab, merges = _get_bpe_state(tokenizer.backend_tokenizer.to_str())
    vocab_tokens = set(vocab.keys())

    merge_pairs, merge_outputs = [], set()
    for rule in merges:
        parts = rule.split(" ")
        if len(parts) == 2:
            out = "".join(parts)
            merge_pairs.append((parts[0], parts[1], out))
            merge_outputs.add(out)

    reachable = set(vocab_tokens - merge_outputs)
    changed = True
    while changed:
        changed = False
        for a, b, out in merge_pairs:
            if out not in reachable and a in reachable and b in reachable and out in vocab_tokens:
                reachable.add(out)
                changed = True

    return sorted(vocab_tokens - reachable)


def find_rank_dead_tokens(tokenizer: PreTrainedTokenizerBase, tokens: Iterable[str] | None = None) -> list[str]:
    """Vocab tokens the merge table can never emit, accounting for merge priority.

    `find_unreachable_tokens_merges` only asks whether *some* merge chain exists.
    A chain that is always outranked by an earlier rule still never fires, so that
    check reports a clean bill of health for tokenizers whose spliced tokens are
    entirely unusable. Special/added tokens are skipped: they bypass the merge
    table by design.
    """
    obj = json.loads(tokenizer.backend_tokenizer.to_str())
    vocab, merges = _get_bpe_state(tokenizer.backend_tokenizer.to_str())
    ranks = _rank_merges(merges)
    alphabet = {tok for tok in vocab if len(tok) == 1}
    special = {entry.get("content") for entry in obj.get("added_tokens", [])}

    dead: list[str] = []
    for tok in vocab.keys() if tokens is None else tokens:
        if len(tok) < 2 or tok in special:
            continue
        if any(symbol not in alphabet for symbol in tok):
            continue
        if _apply_merges(list(tok), ranks) != [tok]:
            dead.append(tok)
    return sorted(dead)


class BasePruner(ABC):
    @abstractmethod
    def train(self, tokenizer: PreTrainedTokenizerBase, corpus: Iterable[str] | None = None) -> None: ...
    @abstractmethod
    def prune(self, tokenizer: PreTrainedTokenizerBase, n_tokens: int) -> None: ...
    @abstractmethod
    def save(self, path: str | Path) -> None: ...


@dataclass
class FrequencyPruner(BasePruner):
    freq: dict[int, int] | None = None
    token_ids_sorted: list[int] | None = None

    def train(self, tokenizer: PreTrainedTokenizerBase, corpus: Iterable[str] | None = None) -> None:
        freq = Counter()
        for text in tqdm(corpus, desc="Computing Pruner Frequencies"):
            freq.update(tokenizer(text, add_special_tokens=False).get("input_ids", []))
        self.freq = dict(freq)
        self.token_ids_sorted = sorted(freq.keys(), key=lambda tid: (freq[tid], tid))

    def prune(self, tokenizer: PreTrainedTokenizerBase, n_tokens: int) -> None:
        self.token_ids_to_prune = self.token_ids_sorted[: int(n_tokens)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"token_ids_to_prune": getattr(self, "token_ids_to_prune", [])}, indent=2))


# =============================================================================
# 6. MAIN ORCHESTRATOR
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Nemotron Tokenizer Extension Toolkit")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--languages", default="hin_Deva,ben_Beng,tam_Taml,tel_Telu")
    parser.add_argument(
        "--dataset-config",
        default=DATASET_CONFIG,
        help=f"Sangraha config name (default: {DATASET_CONFIG}).",
    )
    parser.add_argument("--samples-per-lang", type=int, default=200000)
    parser.add_argument("--extension-size", type=int, default=64000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save final model/tokenizer/artifacts. If omitted, a new timestamped folder is created.",
    )
    parser.add_argument("--tokenizer-only", action="store_true")

    parser.add_argument(
        "--keep-added-token-positions", action="store_true", help="Preserve dense relative ID placements."
    )
    parser.add_argument("--init-method", choices=["mean", "mean_of_constituents"], default="mean_of_constituents")
    parser.add_argument("--benchmark", action="store_true", help="Calculate and print unreachable tokens.")
    parser.add_argument(
        "--prune-size",
        type=int,
        default=0,
        help="If >0, calculates the lowest-frequency N tokens to prune downstream.",
    )

    args = parser.parse_args()
    if not args.out_dir:
        script_dir = Path(__file__).resolve().parent
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(script_dir / "outputs" / f"continued_bpe_{stamp}")
    os.makedirs(args.out_dir, exist_ok=True)

    langs = [lang.strip() for lang in args.languages.split(",")]
    total_docs = len(langs) * args.samples_per_lang
    devanagari_norm = get_devanagari_normalizer()
    doc_id_to_language: dict[str, str] = {}

    logger.info("Loading Base Tokenizer...")
    base_tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)

    logger.info(
        f"Phase 1: Streaming ~{total_docs} docs from {DATASET_ID}/{args.dataset_config} to learn optimal subwords..."
    )
    with tqdm(total=total_docs, desc="Corpus Streaming", unit="docs") as pbar:
        stream = mixed_language_text_stream(
            langs,
            args.samples_per_lang,
            devanagari_norm,
            pbar,
            dataset_config=args.dataset_config,
            doc_id_to_language=doc_id_to_language,
        )
        trained_tok = base_tok.train_new_from_iterator(
            batch_iterator(stream, args.batch_size), vocab_size=len(base_tok) + args.extension_size
        )

    logger.info("Phase 2: Extracting Diff Artifacts...")
    artifacts = compute_continued_bpe_artifacts(base_tok.backend_tokenizer, trained_tok.backend_tokenizer)

    merges_pairs = [tuple(x.split(" ")) for x in artifacts.new_merges if len(x.split(" ")) == 2]
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(artifacts.new_vocab, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "merges.json"), "w") as f:
        json.dump(artifacts.new_merges, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "doc_id_to_language.json"), "w") as f:
        json.dump(doc_id_to_language, f, ensure_ascii=False)

    logger.info("Phase 3: Splicing merges into base tokenizer...")
    expanded_tok = extend_tokenizer(
        base_tok,
        artifacts.new_vocab,
        merges_pairs,
        n_tokens=args.extension_size,
        keep_added_token_positions=args.keep_added_token_positions,
    )
    expanded_tok.save_pretrained(args.out_dir)

    if args.benchmark:
        logger.info("Phase 3.5: Benchmarking Unreachable Graph Tokens...")
        unreachable = find_unreachable_tokens_merges(expanded_tok)
        logger.info(f"Found {len(unreachable)} mathematically unreachable tokens in the BPE graph.")
        with open(os.path.join(args.out_dir, "unreachable.json"), "w") as f:
            json.dump(unreachable, f, ensure_ascii=False)

    if args.prune_size > 0:
        logger.info(f"Phase 3.5: Calculating {args.prune_size} tokens to prune via FrequencyPruner...")
        pruner = FrequencyPruner()
        prune_stream = mixed_language_text_stream(
            langs,
            min(args.samples_per_lang, 10000),
            devanagari_norm,
            None,
            dataset_config=args.dataset_config,
        )
        pruner.train(expanded_tok, list(prune_stream))
        pruner.prune(expanded_tok, args.prune_size)
        pruner.save(os.path.join(args.out_dir, "pruned_tokens.json"))
        logger.info("Pruned tokens list saved.")

    if args.tokenizer_only:
        logger.info("Tokenizer-only mode active. Exiting before Model initialization.")
        return

    logger.info(f"Phase 4: Loading Model & Applying '{args.init_method}' embeddings...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    modify_embeddings(model, old_tokenizer=base_tok, new_tokenizer=expanded_tok, init_method=args.init_method)

    model.save_pretrained(args.out_dir)
    logger.info(f"Success! Model and Tokenizer fully extended and saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
