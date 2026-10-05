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

"""
Replace-and-extend BPE toolkit for the Nemotron tokenizer.

Where `continued_bpe.py` *adds* fresh Indic subwords on top of the base vocab
(keeping the base's residual tokens), this script *replaces* them:

  Phase 1  Identify the base's residual tokens for a target script
           (e.g. Devanagari) -- the ~1.5k pre-existing Hindi/Marathi/... tokens.
  Phase 2  Prune those tokens + every merge that touches them, then densely
           re-index the surviving vocab (0..M-1). Special/added tokens survive
           and get re-pointed; `ignore_merges`/`unk_token`/prefixes preserved.
  Phase 3  Train a fresh BPE on a target-language corpus.
  Phase 4  Diff the fresh tokenizer against the *pruned* base and splice the
           new subwords + merges in (DAG-backtracked), re-filling the removed
           slots with corpus-optimal tokens and extending by the requested
           budget.
  Phase 5  (optional) Remap the model embeddings to the new index space and
           initialize the fresh rows.

Design notes
------------
* The base tokenizer is loaded with `fix_mistral_regex=True`. Nemotron is a
  Mistral derivative, and without this the pre-tokenizer shatters Indic words
  during BPE training.
* Self-contained: the shared continued-BPE core (Sangraha streaming, merge-diff,
  constructive splice, rank-dead check) is inlined below, so this script runs on
  its own with no dependency on continued_bpe.py.
* Embedding-init strategy is intentionally simple here (mean-of-constituents over
  the ORIGINAL embeddings, so a replaced Devanagari token warm-starts from the
  mean of the old residual tokens it replaces). Other init schemes are a
  follow-up -- `--init-method` is the hook.

Compared to the *add* path, this trades a few pre-trained embedding rows for
reclaimed vocab slots. See `../README.md` for when to choose each arm.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
# SHARED CONTINUED-BPE CORE  (inlined so this script is self-contained)
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
    text = text.replace("\x00", "").replace("﻿", "")
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
    return split_name.split("_", 1)[0].lower()


def _resolve_sangraha_splits(requested_languages: list[str], dataset_config: str) -> list[str]:
    available_splits = get_dataset_split_names(DATASET_ID, dataset_config)
    if not requested_languages:
        return available_splits
    requested = {_normalize_lang(lang) for lang in requested_languages}
    resolved = []
    for split_name in available_splits:
        if _normalize_lang(split_name) in requested or _base_language_from_split(split_name) in requested:
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
    graph-reachability check still passes. Each token is therefore built from the
    pieces the *current* merge table yields, so the rule appended is the one that
    will fire.
    """
    obj = json.loads(base_backend.to_str())
    model = obj.get("model", {})
    vocab: dict[str, int] = model["vocab"]
    merges_raw = model.get("merges", [])
    merges: list[str] = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in merges_raw]

    selected_tokens = [t for (t, _) in sorted(new_vocab.items(), key=lambda kv: kv[1])][: int(n_tokens)]
    next_id = (max(vocab.values()) + 1) if vocab else 0

    if not new_merges:
        for tok in selected_tokens:
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
        model["vocab"] = vocab
        return Tokenizer.from_str(json.dumps(obj))

    ranks = _rank_merges(merges)
    alphabet = {tok for tok in vocab if len(tok) == 1}
    budget = int(n_tokens)
    added = 0

    for tok in selected_tokens:
        if added >= budget:
            break
        if tok in vocab:
            continue
        if any(symbol not in alphabet for symbol in tok):
            continue
        pieces = _apply_merges(list(tok), ranks)
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
    entirely unusable. Special/added tokens are skipped: they bypass the merge table.
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


# =============================================================================
# 1. TARGET-SCRIPT DEFINITIONS
# =============================================================================
# Unicode ranges per script. A token is "residual" for a script if any character
# of its *decoded* surface string falls in one of these ranges. Names are the
# keys accepted by --remove-script (comma-separated).
from script_ranges import SCRIPT_UNICODE_RANGES, resolve_ranges  # noqa: E402,F401


def identify_script_tokens(tokenizer: PreTrainedTokenizerBase, ranges: list[tuple[int, int]]) -> set[str]:
    """Vocab tokens whose decoded surface contains a codepoint in `ranges`.

    Mirrors remove_hindi_tokens.py: byte-level pieces that decode to the target
    script are removed, while the underlying byte alphabet (which does NOT decode
    to the target script) is kept, so the script can be rebuilt from bytes by the
    freshly spliced merges.
    """
    hits: set[str] = set()
    for token in tokenizer.get_vocab():
        decoded = tokenizer.convert_tokens_to_string([token])
        for ch in decoded:
            cp = ord(ch)
            if any(start <= cp <= end for start, end in ranges):
                hits.add(token)
                break
    return hits


# =============================================================================
# 2. PRUNE + DENSE RE-INDEX
# =============================================================================
def prune_backend(
    base_backend: Tokenizer, remove_tokens: set[str]
) -> tuple[Tokenizer, dict[int, int], set[int], list[str]]:
    """Remove `remove_tokens` from a BPE backend and densely re-index survivors.

    Returns
    -------
    pruned_backend : Tokenizer
    old2new        : {old_id -> new_id} for every SURVIVING token id
    removed_ids    : set of original ids that were dropped
    removed_tokens : sorted list of token strings actually removed (present in vocab)
    """
    obj = json.loads(base_backend.to_str())
    model = obj["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"Expected BPE backend, got {model.get('type')!r}")

    vocab: dict[str, int] = model["vocab"]
    n_vocab = len(vocab)

    # ids to remove (only those actually present)
    removed_ids = {old_id for tok, old_id in vocab.items() if tok in remove_tokens}

    # survivors, ordered by original id -> dense new id (keeps relative order,
    # so special/byte tokens with small ids keep their positions where possible)
    survivor_old_ids = [i for i in range(n_vocab) if i not in removed_ids]
    old2new = {old: new for new, old in enumerate(survivor_old_ids)}

    # rebuild vocab
    model["vocab"] = {tok: old2new[old] for tok, old in vocab.items() if old not in removed_ids}

    # filter merges: drop any whose left, right, or merged result is removed.
    # preserve the original element type (list or "a b" string) of each kept rule.
    kept_merges: list[Any] = []
    produced_before: set[str] = set()
    produced_after: set[str] = set()
    for m in model.get("merges", []):
        parts = m.split(" ") if isinstance(m, str) else list(m)
        if len(parts) != 2:
            continue
        left, right = parts
        produced_before.add(left + right)
        if left in remove_tokens or right in remove_tokens or (left + right) in remove_tokens:
            continue
        produced_after.add(left + right)
        kept_merges.append(m)
    model["merges"] = kept_merges

    # A removal set that takes out a merge *ancestor* strands every survivor built
    # on it: the token stays in the vocab with no rule that can produce it, and the
    # splice cannot repair it because it is not a newly learned token. Removing only
    # tokens that decode to the target script cannot trigger this (their outputs
    # decode to the script too), so treat it as a bad removal set, not a repair job.
    orphaned = sorted(
        tok for tok in model["vocab"] if len(tok) > 1 and tok in produced_before and tok not in produced_after
    )
    if orphaned:
        raise ValueError(
            f"Removal set orphans {len(orphaned):,} surviving token(s) that lose every "
            f"producing merge, e.g. {orphaned[:5]}. Remove them too, or narrow the set."
        )

    # re-point added/special token ids (all survive: none are target-script)
    for a in obj.get("added_tokens", []):
        if a["id"] in removed_ids:
            raise ValueError(f"Refusing to remove a special/added token: {a.get('content')!r}")
        a["id"] = old2new[a["id"]]

    pruned = Tokenizer.from_str(json.dumps(obj))
    removed_tokens = sorted(t for t in remove_tokens if t in vocab)
    return pruned, old2new, removed_ids, removed_tokens


def wrap_fast(backend: Tokenizer, base_tok: PreTrainedTokenizerBase) -> PreTrainedTokenizerFast:
    """Wrap a backend as a fast tokenizer, carrying over special tokens + chat template.

    The backend already contains all `added_tokens`, so specials remain functional;
    we additionally set the Python-side attributes and the chat template so the
    saved tokenizer is a drop-in for the base.
    """
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=getattr(base_tok, "unk_token", None),
        bos_token=getattr(base_tok, "bos_token", None),
        eos_token=getattr(base_tok, "eos_token", None),
        pad_token=getattr(base_tok, "pad_token", None),
    )
    chat_template = getattr(base_tok, "chat_template", None)
    if chat_template:
        fast.chat_template = chat_template

    # Carry the rest of the base's tokenizer configuration. Without this the saved
    # tokenizer silently changes behaviour versus the base — model_max_length falls
    # back to the unlimited sentinel and the padding/truncation sides flip — so it
    # is not a drop-in replacement even though the vocabulary is correct.
    for attr in (
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "sep_token",
        "cls_token",
        "mask_token",
    ):
        value = getattr(base_tok, attr, None)
        if value is not None:
            try:
                setattr(fast, attr, value)
            except (AttributeError, ValueError):
                pass
    extra = getattr(base_tok, "additional_special_tokens", None)
    if extra:
        fast.additional_special_tokens = list(extra)
    # NFKC normalizer so inference matches the NFKC decomposition clean_text()
    # applies during BPE training (fixes nukta byte-fragmentation: पड़/मरीज़/...).
    from tokenizers import normalizers

    existing = fast.backend_tokenizer.normalizer
    fast.backend_tokenizer.normalizer = (
        normalizers.NFKC() if existing is None else normalizers.Sequence([normalizers.NFKC(), existing])
    )
    return fast


# =============================================================================
# 3. EMBEDDING REMAP + INIT  (only used in the full, non --tokenizer-only path)
# =============================================================================
def remap_and_init_embeddings(
    model: PreTrainedModel,
    base_tok: PreTrainedTokenizerBase,
    final_tok: PreTrainedTokenizerBase,
    old2new: dict[int, int],
    pruned_size: int,
    init_method: str = "mean_of_constituents",
) -> dict[str, Any]:
    """Reshape model embeddings for the replace-and-extend tokenizer.

    Row layout of the final matrix (size = len(final_tok)):
      [0, pruned_size)      -> surviving base rows, permuted via old2new
      [pruned_size, final)  -> freshly spliced tokens, initialized below

    `mean_of_constituents` encodes each new token's surface with the ORIGINAL base
    tokenizer over the ORIGINAL embeddings -- so a replaced Devanagari token
    warm-starts from the mean of the old residual tokens it is replacing.
    """
    in_layer = model.get_input_embeddings()
    out_layer = model.get_output_embeddings()

    old_n = in_layer.weight.shape[0]
    if old_n != len(base_tok):
        raise ValueError(f"Model rows ({old_n}) != base tokenizer ({len(base_tok)}).")
    final_n = len(final_tok)

    tied = out_layer is None or out_layer.weight.data_ptr() == in_layer.weight.data_ptr()

    # snapshot originals BEFORE resizing
    old_in = in_layer.weight.data.clone()
    old_out = None if tied else out_layer.weight.data.clone()

    model.resize_token_embeddings(final_n)
    in_emb = model.get_input_embeddings().weight
    out_layer2 = model.get_output_embeddings()
    out_emb = None if tied else out_layer2.weight

    gmean_in = old_in[:old_n].mean(dim=0).detach()
    gmean_out = None if old_out is None else old_out[:old_n].mean(dim=0).detach()

    with torch.no_grad():
        # --- survivors: vectorized permutation ---
        src = torch.tensor(list(old2new.keys()), dtype=torch.long)
        dst = torch.tensor([old2new[i] for i in old2new.keys()], dtype=torch.long)
        in_emb[dst.to(in_emb.device)] = old_in[src.to(old_in.device)].to(in_emb.device)
        if out_emb is not None:
            out_emb[dst.to(out_emb.device)] = old_out[src.to(old_out.device)].to(out_emb.device)

        # --- new rows ---
        for tid in tqdm(range(pruned_size, final_n), desc="Init new rows"):
            tok = final_tok.convert_ids_to_tokens(tid)
            vec_in = gmean_in.clone()
            vec_out = gmean_out.clone() if gmean_out is not None else None

            if init_method == "mean_of_constituents":
                surface = final_tok.convert_tokens_to_string([tok])
                enc = base_tok(surface, add_special_tokens=False)
                ids = [i for i in enc.get("input_ids", []) if 0 <= i < old_n]
                if ids:
                    idx = torch.tensor(ids, dtype=torch.long)
                    vec_in = old_in[idx.to(old_in.device)].mean(dim=0)
                    if gmean_out is not None:
                        vec_out = old_out[idx.to(old_out.device)].mean(dim=0)

            in_emb[tid].copy_(vec_in.to(in_emb.device))
            if out_emb is not None and vec_out is not None:
                out_emb[tid].copy_(vec_out.to(out_emb.device))

    return {
        "old_vocab_size": old_n,
        "pruned_size": pruned_size,
        "final_vocab_size": final_n,
        "survivors_remapped": len(old2new),
        "new_rows_initialized": final_n - pruned_size,
        "tied_embeddings": bool(tied),
        "init_method": init_method,
    }


# =============================================================================
# 4. CORPUS
# =============================================================================
def local_jsonl_text_stream(
    paths: list[str],
    text_field: str,
    max_samples: int,
    devanagari_norm: Any,
    pbar: tqdm | None,
) -> Iterator[str]:
    yielded = 0
    checked = False
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not checked:
                    if text_field not in ex:
                        raise ValueError(
                            f"corpus.text_field={text_field!r} is not a column of {path}. "
                            f"Available: {sorted(ex.keys())}."
                        )
                    checked = True
                value = ex.get(text_field)
                raw = value if isinstance(value, str) else ""
                if not raw:
                    continue
                cleaned = clean_text(raw, devanagari_norm)
                if len(cleaned) > 50:
                    yield cleaned
                    yielded += 1
                    if pbar is not None:
                        pbar.update(1)
                    if yielded >= max_samples:
                        return


# =============================================================================
# 5. MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replace a target script's residual tokens with a freshly "
        "trained BPE, then extend (Nemotron replace-and-extend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--remove-script",
        default="devanagari",
        help="Comma-separated script(s) whose residual tokens are removed "
        f"(choices: {sorted(SCRIPT_UNICODE_RANGES)}).",
    )
    p.add_argument(
        "--extension-size",
        type=int,
        default=64000,
        help="Number of fresh tokens to splice in (before DAG-required parents).",
    )
    p.add_argument(
        "--size-neutral",
        action="store_true",
        help="Override --extension-size with the count of removed tokens, so the "
        "final vocab ~= base vocab (pure swap).",
    )

    # corpus: either a local jsonl or Sangraha streaming
    p.add_argument(
        "--corpus-jsonl",
        nargs="*",
        default=None,
        help="Local jsonl file(s) of target-language text. If omitted, streams Sangraha.",
    )
    p.add_argument("--text-field", default="text")
    p.add_argument(
        "--languages",
        default="hin_Deva",
        help="Sangraha split(s) for streaming when --corpus-jsonl is not given.",
    )
    p.add_argument("--dataset-config", default=DATASET_CONFIG)
    p.add_argument("--samples-per-lang", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=1000)

    p.add_argument("--out-dir", default=None)
    p.add_argument("--tokenizer-only", action="store_true")
    p.add_argument("--keep-added-token-positions", action="store_true")
    p.add_argument("--init-method", choices=["mean", "mean_of_constituents"], default="mean_of_constituents")
    p.add_argument("--benchmark", action="store_true", help="Report unreachable BPE-graph tokens.")
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.set_defaults(trust_remote_code=True)
    return p.parse_args()


def load_base_tokenizer(model_id: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    """Load base tokenizer with fix_mistral_regex=True (report.txt fix), robustly."""
    kwargs = dict(use_fast=True, trust_remote_code=trust_remote_code)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True, **kwargs)
        logger.info("Loaded base tokenizer with fix_mistral_regex=True.")
        return tok
    except TypeError:
        logger.warning("fix_mistral_regex not accepted by this tokenizer; loading without it.")
        return AutoTokenizer.from_pretrained(model_id, **kwargs)


def main() -> None:
    args = parse_args()

    if not args.out_dir:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(Path(__file__).resolve().parent / "outputs" / f"replace_bpe_{stamp}")
    os.makedirs(args.out_dir, exist_ok=True)

    devanagari_norm = get_devanagari_normalizer()

    # ------------------------------------------------------------------ base
    logger.info("Loading base tokenizer...")
    base_tok = load_base_tokenizer(args.model_id, args.trust_remote_code)
    base_size = len(base_tok)

    # -------------------------------------------------- Phase 1: identify
    scripts = [s for s in args.remove_script.split(",") if s.strip()]
    ranges = resolve_ranges(scripts)
    logger.info(f"Phase 1: identifying residual tokens for script(s)={scripts} ...")
    remove_tokens = identify_script_tokens(base_tok, ranges)
    logger.info(f"Residual tokens to remove: {len(remove_tokens):,}")
    if not remove_tokens:
        raise SystemExit("No residual tokens matched -- nothing to replace.")

    if args.size_neutral:
        args.extension_size = len(remove_tokens)
        logger.info(f"--size-neutral: extension-size set to {args.extension_size:,} (== removed).")

    # -------------------------------------------------- Phase 2: prune
    logger.info("Phase 2: pruning + dense re-index...")
    pruned_backend, old2new, removed_ids, removed_list = prune_backend(base_tok.backend_tokenizer, remove_tokens)
    pruned_base_tok = wrap_fast(pruned_backend, base_tok)
    pruned_size = len(pruned_base_tok)
    logger.info(
        f"Base {base_size:,} -> pruned {pruned_size:,} "
        f"(removed {len(removed_ids):,}, survivors re-indexed 0..{pruned_size - 1:,})"
    )

    # -------------------------------------------------- Phase 3: train fresh BPE
    langs = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if args.corpus_jsonl:
        total = args.samples_per_lang
        logger.info(f"Phase 3: training fresh BPE on local jsonl {args.corpus_jsonl} ...")
        with tqdm(total=total, desc="Corpus", unit="docs") as pbar:
            stream = local_jsonl_text_stream(args.corpus_jsonl, args.text_field, total, devanagari_norm, pbar)
            trained_tok = base_tok.train_new_from_iterator(
                batch_iterator(stream, args.batch_size),
                vocab_size=base_size + args.extension_size,
            )
    else:
        total = len(langs) * args.samples_per_lang
        logger.info(
            f"Phase 3: training fresh BPE on Sangraha {args.dataset_config} splits={langs} (~{total:,} docs)..."
        )
        doc_id_to_language: dict[str, str] = {}
        with tqdm(total=total, desc="Corpus", unit="docs") as pbar:
            stream = mixed_language_text_stream(
                langs,
                args.samples_per_lang,
                devanagari_norm,
                pbar,
                dataset_config=args.dataset_config,
                doc_id_to_language=doc_id_to_language,
            )
            trained_tok = base_tok.train_new_from_iterator(
                batch_iterator(stream, args.batch_size),
                vocab_size=base_size + args.extension_size,
            )
        with open(os.path.join(args.out_dir, "doc_id_to_language.json"), "w") as f:
            json.dump(doc_id_to_language, f, ensure_ascii=False)

    # -------------------------------------------------- Phase 4: diff + splice
    logger.info("Phase 4: diffing fresh vs pruned base, splicing merges...")
    artifacts = compute_continued_bpe_artifacts(pruned_base_tok.backend_tokenizer, trained_tok.backend_tokenizer)
    merges_pairs = [tuple(x.split(" ")) for x in artifacts.new_merges if len(x.split(" ")) == 2]
    logger.info(f"New candidate tokens: {len(artifacts.new_vocab):,} | new merges: {len(artifacts.new_merges):,}")

    final_tok = extend_tokenizer(
        pruned_base_tok,
        artifacts.new_vocab,
        merges_pairs,
        n_tokens=args.extension_size,
        keep_added_token_positions=args.keep_added_token_positions,
    )
    final_tok = wrap_fast(final_tok.backend_tokenizer, base_tok)
    final_size = len(final_tok)

    # -------------------------------------------------- save tokenizer + artifacts
    final_tok.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(artifacts.new_vocab, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "merges.json"), "w") as f:
        json.dump(artifacts.new_merges, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "removed_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(removed_list))
    with open(os.path.join(args.out_dir, "id_remap.json"), "w") as f:
        json.dump({str(k): v for k, v in old2new.items()}, f)

    report = {
        "mode": "replace_and_extend",
        "model_id": args.model_id,
        "remove_script": scripts,
        "base_vocab_size": base_size,
        "removed": len(removed_ids),
        "pruned_vocab_size": pruned_size,
        "extension_size_requested": args.extension_size,
        "new_tokens_spliced": final_size - pruned_size,
        "final_vocab_size": final_size,
        "net_vs_base": final_size - base_size,
        "size_neutral": bool(args.size_neutral),
        "fix_mistral_regex": True,
        "init_method": args.init_method,
    }
    with open(os.path.join(args.out_dir, "replace_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report: " + json.dumps(report))

    if args.benchmark:
        unreachable = find_unreachable_tokens_merges(final_tok)
        rank_dead = find_rank_dead_tokens(final_tok)
        logger.info(f"Unreachable BPE-graph tokens: {len(unreachable):,} | rank-dead tokens: {len(rank_dead):,}")
        with open(os.path.join(args.out_dir, "unreachable.json"), "w") as f:
            json.dump({"graph_unreachable": unreachable, "rank_dead": rank_dead}, f, ensure_ascii=False)

    if args.tokenizer_only:
        logger.info(f"Tokenizer-only. Saved to {args.out_dir}")
        return

    # -------------------------------------------------- Phase 5: model embeddings
    logger.info(f"Phase 5: loading model & remapping embeddings ('{args.init_method}')...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    emb_report = remap_and_init_embeddings(
        model,
        base_tok=base_tok,
        final_tok=final_tok,
        old2new=old2new,
        pruned_size=pruned_size,
        init_method=args.init_method,
    )
    logger.info("Embedding report: " + json.dumps(emb_report))
    model.save_pretrained(args.out_dir)
    logger.info(f"Success! Replace-and-extend model + tokenizer saved to {args.out_dir}")


if __name__ == "__main__":
    main()
