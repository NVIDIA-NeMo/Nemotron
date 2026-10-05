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

"""Initialize new-token embeddings of an extended tokenizer by subword averaging.

Every token that the extended tokenizer adds is decomposed into subwords of the
original vocabulary, and its input/output embeddings are initialized from a
weighted average of the corresponding original embeddings.  The averaging
strategy and the norm correction are chosen independently for the input and the
output side.

Averaging strategies
    uniform         plain mean of the subword embeddings
    char_weighted   mean weighted by the character length of each subword
    max_char        the longest subword's embedding only (ties: first occurrence)
    bert_weighted   mean weighted by softmax(cos(subword, full token) / temperature),
                    using mean-pooled BERT last hidden states as representations
    gemma_weighted  same weighting, but using Gemma's input embedding layer
                    (embed_tokens) instead of a forward pass

BERT and Gemma may be combined (e.g. gemma_weighted input, bert_weighted output).
They are loaded sequentially and released afterwards, so peak GPU memory is
bounded by the larger of the two.

Norm correction rescales an averaged embedding so that its L2 norm matches the
median norm of the original embeddings; with Hindi norm correction the median is
taken over Devanagari tokens only.  Output norm correction is off by default:
inflating output norms makes the model over-confidently predict the new tokens
and can explode the training loss.

Example
    python extend_model_subword_init_configurable.py \
        --extended-tokenizer /path/to/merged_tokenizer \
        --output-dir /path/to/extended_model \
        --input-averaging bert_weighted \
        --output-averaging char_weighted
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

AVERAGING_METHODS = ("uniform", "char_weighted", "max_char", "bert_weighted", "gemma_weighted")
SEMANTIC_METHODS = ("bert_weighted", "gemma_weighted")

DEFAULT_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
DEFAULT_BERT_MODEL = "google/muril-base-cased"
DEFAULT_GEMMA_MODEL = "google/gemma-2-27b"

DEVANAGARI_RANGE = (0x0900, 0x097F)
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
RULE = "=" * 60


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Model whose embedding matrices are extended.")
    paths.add_argument(
        "--extended-tokenizer", required=True, help="Tokenizer containing the base vocabulary plus the new tokens."
    )
    paths.add_argument("--output-dir", required=True, help="Directory to save the extended model and tokenizer to.")
    paths.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute custom modeling code shipped in the model repo. "
        "Off by default; required by architectures whose code lives there.",
    )
    paths.add_argument(
        "--dtype", choices=sorted(DTYPES), default="bfloat16", help="Precision to load the base model in."
    )

    averaging = parser.add_argument_group("averaging")
    averaging.add_argument("--input-averaging", choices=AVERAGING_METHODS, default="uniform")
    averaging.add_argument("--output-averaging", choices=AVERAGING_METHODS, default="uniform")

    norms = parser.add_argument_group("norm correction")
    norms.add_argument(
        "--input-norm-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rescale averaged input embeddings to the median original norm.",
    )
    norms.add_argument(
        "--output-norm-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rescale averaged output embeddings; risks loss explosion.",
    )
    norms.add_argument(
        "--input-hindi-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Take the target input norm over Devanagari tokens only.",
    )
    norms.add_argument(
        "--output-hindi-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Take the target output norm over Devanagari tokens only.",
    )

    semantic = parser.add_argument_group("semantic weighting")
    semantic.add_argument(
        "--bert-model",
        default=None,
        help="Auxiliary encoder for *_weighted averaging. Default: from --language, else MuRIL (Indic-only).",
    )
    semantic.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    semantic.add_argument(
        "--temperature", type=float, default=0.1, help="Softmax temperature; lower is sharper (0.05 is near-argmax)."
    )
    semantic.add_argument("--bert-batch-size", type=int, default=128)
    semantic.add_argument("--gemma-batch-size", type=int, default=64)
    semantic.add_argument("--semantic-device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--num-samples", type=int, default=10, help="Multi-subword tokens to report weights for.")

    parser.add_argument(
        "--language",
        default=None,
        help="Target language (see languages.py). Selects the script used to find "
        "the base model's existing target-language rows, and supplies defaults "
        "for the auxiliary encoder / fastText vectors. Omit for legacy Hindi.",
    )

    args = parser.parse_args(argv)

    # Apply the language profile before anything reads the target script.
    if getattr(args, "language", None):
        from languages import fasttext_url as _fu
        from languages import profile as _lp
        from script_ranges import set_target_script

        _prof = _lp(args.language)
        set_target_script(_prof.script)
        if getattr(args, "bert_model", None) is None:
            args.bert_model = _prof.encoder
        if hasattr(args, "fasttext_url") and args.fasttext_url is None:
            args.fasttext_url = _fu(args.language)
    if args.bert_model is None:
        args.bert_model = DEFAULT_BERT_MODEL
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.num_samples < 0:
        parser.error("--num-samples must not be negative")
    return args


def semantic_methods_in_use(args: argparse.Namespace) -> set[str]:
    return {args.input_averaging, args.output_averaging} & set(SEMANTIC_METHODS)


def hindi_norm_in_use(args: argparse.Namespace) -> bool:
    return (args.input_norm_correction and args.input_hindi_norm) or (
        args.output_norm_correction and args.output_hindi_norm
    )


def print_configuration(args: argparse.Namespace) -> None:
    def effectiveness(hindi: bool, correction: bool) -> str:
        if not hindi:
            return ""
        return " (effective)" if correction else " (ignored)"

    print(f"  Input  averaging:        {args.input_averaging}")
    print(f"  Output averaging:        {args.output_averaging}")
    print(f"  Input  norm correction:  {args.input_norm_correction}")
    print(f"  Output norm correction:  {args.output_norm_correction}")
    print(
        f"  Input  Hindi norm:       {args.input_hindi_norm}"
        f"{effectiveness(args.input_hindi_norm, args.input_norm_correction)}"
    )
    print(
        f"  Output Hindi norm:       {args.output_hindi_norm}"
        f"{effectiveness(args.output_hindi_norm, args.output_norm_correction)}"
    )
    if args.output_norm_correction:
        print("  WARNING: output norm correction is ON, which can cause loss explosion.")
    methods = semantic_methods_in_use(args)
    if "bert_weighted" in methods:
        print(f"  BERT model:              {args.bert_model}")
    if "gemma_weighted" in methods:
        print(f"  Gemma model:             {args.gemma_model}")
    if methods:
        print(f"  Temperature:             {args.temperature}")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TokenPlan:
    """How a single new token will be initialized."""

    category: str  # "mean" | "copy" | "multi"
    token_str: str
    subword_ids: list[int]


@dataclass
class Decomposition:
    plans: dict[int, TokenPlan]
    multi_token_ids: list[int]
    unique_subword_ids: list[int]
    subword_counts: list[int]


@dataclass
class SemanticInputs:
    """Decoded strings that the semantic models embed."""

    token_ids: list[int]
    token_texts: list[str]
    subword_ids: list[int]
    subword_texts: list[str]


@dataclass
class SemanticEmbeddings:
    label: str
    full_token: np.ndarray
    token_id_to_row: dict[int, int]
    subword: dict[int, np.ndarray]


@dataclass
class SampleRecord:
    token_id: int
    token_str: str
    decoded_text: str
    subword_ids: list[int]
    subword_tokens: list[str]
    subword_texts: list[str]
    char_lengths: list[int]
    input_weights: list[float]
    output_weights: list[float]
    input_sims: list[float] | None = None
    output_sims: list[float] | None = None


@dataclass
class InitResult:
    from_subwords: int = 0
    from_copy: int = 0
    from_mean: int = 0
    target_input_norm: float | None = None
    target_output_norm: float | None = None
    input_norm_source: str = ""
    output_norm_source: str = ""
    input_norms_before: list[float] = field(default_factory=list)
    input_norms_after: list[float] = field(default_factory=list)
    output_norms_before: list[float] = field(default_factory=list)
    output_norms_after: list[float] = field(default_factory=list)
    samples: list[SampleRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


@contextmanager
def phase(title: str) -> Iterator[None]:
    print("\n" + RULE)
    print(title)
    print(RULE)
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.2f}s ({elapsed / 60:.2f} min)")


@contextmanager
def step(message: str) -> Iterator[None]:
    print(f"\n{message}")
    start = time.time()
    yield
    print(f"  Took {time.time() - start:.2f}s")


def describe_tensor(norms: torch.Tensor, extremes: bool = True) -> str:
    parts = [
        f"median: {norms.median().item():.4f}",
        f"mean: {norms.mean().item():.4f}",
        f"std: {norms.std().item():.4f}",
    ]
    if extremes:
        parts += [f"min: {norms.min().item():.4f}", f"max: {norms.max().item():.4f}"]
    return ", ".join(parts)


def describe_values(values: Sequence[float]) -> str:
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"median: {statistics.median(values):.4f}, mean: {statistics.mean(values):.4f}, std: {std:.4f}"


def format_weights(weights: Sequence[float]) -> str:
    return "[" + ", ".join(f"{w:.4f}" for w in weights) + "]"


def free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------


def is_target_script(text: str) -> bool:
    """True if ``text`` lies in the active target script, which defaults to
    Devanagari and is selected by --language / set_target_script()."""
    from script_ranges import is_target

    return is_target(text)


def find_target_script_tokens(tokenizer, vocab_size: int) -> list[tuple[int, str]]:
    found = []
    for token_id in tqdm(range(vocab_size), desc="Scanning vocabulary for target-language tokens"):
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            continue
        if is_target_script(text):
            found.append((token_id, text))
    return found


def decoded_char_lengths(subword_ids: Sequence[int], tokenizer) -> list[int]:
    # Clamped to 1 so that an empty decoding never yields a zero weight.
    return [max(len(tokenizer.decode([sid])), 1) for sid in subword_ids]


def length_based_weights(
    subword_ids: Sequence[int], method: str, tokenizer, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    count = len(subword_ids)
    if method == "uniform":
        return torch.full((count,), 1.0 / count, dtype=dtype, device=device)

    char_lengths = decoded_char_lengths(subword_ids, tokenizer)
    if method == "max_char":
        weights = [0.0] * count
        weights[char_lengths.index(max(char_lengths))] = 1.0
    else:
        total = sum(char_lengths)
        weights = [length / total for length in char_lengths]
    return torch.tensor(weights, dtype=dtype, device=device)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def semantic_similarity_weights(
    subword_ids: Sequence[int],
    full_token_vector: np.ndarray,
    subword_vectors: dict[int, np.ndarray],
    temperature: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    sims = [cosine_similarity(full_token_vector, subword_vectors[sid]) for sid in subword_ids]
    scaled = np.array(sims, dtype=np.float64) / temperature
    exponentiated = np.exp(scaled - scaled.max())
    weights = exponentiated / exponentiated.sum()
    return torch.tensor(weights, dtype=dtype, device=device), sims


def subword_weights(
    method: str,
    token_id: int,
    subword_ids: Sequence[int],
    tokenizer,
    semantic: dict[str, SemanticEmbeddings],
    temperature: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, list[float] | None]:
    if method in SEMANTIC_METHODS:
        embeddings = semantic[method]
        full_token_vector = embeddings.full_token[embeddings.token_id_to_row[token_id]]
        return semantic_similarity_weights(
            subword_ids,
            full_token_vector,
            embeddings.subword,
            temperature,
            dtype,
            device,
        )
    return length_based_weights(subword_ids, method, tokenizer, dtype, device), None


# ---------------------------------------------------------------------------
# Semantic embeddings
# ---------------------------------------------------------------------------


def bert_embeddings(texts: Sequence[str], model, tokenizer, batch_size: int, device: str) -> np.ndarray:
    """Mean-pooled last hidden states, one row per text."""
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"  {len(texts)} texts in {total_batches} batches of {batch_size}")

    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings", total=total_batches, unit="batch"):
            batch = texts[start : start + batch_size]
            try:
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                    device
                )
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                embeddings.append(pooled.float().cpu().numpy())
            except Exception as error:
                # A zero vector here is not a neutral fallback: it silently turns a
                # semantic initialization into a meaningless one, and the run still
                # reports success. Fail instead — the caller can retry with a
                # smaller batch or a different encoder.
                raise RuntimeError(
                    f"encoder failed on the batch at index {start}: {error}. Refusing to "
                    f"substitute zero vectors, which would make this a non-semantic init."
                ) from error

    return np.vstack(embeddings)


def input_embedding_layer(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens
    return model.get_input_embeddings()


def gemma_embeddings(texts: Sequence[str], model, tokenizer, batch_size: int, device: str) -> np.ndarray:
    """Mean-pooled embed_tokens lookups, one row per text (no forward pass)."""
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"  {len(texts)} texts in {total_batches} batches of {batch_size}")

    embed_tokens = input_embedding_layer(model)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), batch_size), desc="Gemma embeddings", total=total_batches, unit="batch"
        ):
            for text in texts[start : start + batch_size]:
                try:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                        device
                    )
                    token_embeddings = embed_tokens(inputs.input_ids)
                    if "attention_mask" in inputs:
                        mask = inputs.attention_mask.unsqueeze(-1)
                        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                    else:
                        pooled = token_embeddings.mean(dim=1)
                    embeddings.append(pooled[0].float().cpu().numpy())
                except Exception as error:
                    raise RuntimeError(
                        f"encoder failed on {text[:50]!r}: {error}. Refusing to substitute a "
                        f"zero vector, which would silently drop this token's semantics."
                    ) from error

    return np.array(embeddings)


def collect_semantic_inputs(extended_tokenizer, original_tokenizer, decomposition: Decomposition) -> SemanticInputs:
    token_texts = []
    for token_id in decomposition.multi_token_ids:
        try:
            token_texts.append(extended_tokenizer.decode([token_id]))
        except Exception:
            token_texts.append(decomposition.plans[token_id].token_str)

    subword_texts = []
    for subword_id in decomposition.unique_subword_ids:
        try:
            subword_texts.append(original_tokenizer.decode([subword_id]))
        except Exception:
            subword_texts.append(original_tokenizer.convert_ids_to_tokens(subword_id))

    print(f"  Full token texts to embed:     {len(token_texts)}")
    print(f"  Unique subword texts to embed: {len(subword_texts)}")
    print(f"  Embedding calls per model:     {len(token_texts) + len(subword_texts)}")

    return SemanticInputs(
        token_ids=list(decomposition.multi_token_ids),
        token_texts=token_texts,
        subword_ids=list(decomposition.unique_subword_ids),
        subword_texts=subword_texts,
    )


def build_bert_semantics(args: argparse.Namespace, inputs: SemanticInputs) -> SemanticEmbeddings:
    with phase("Phase 3a: BERT semantic embeddings"):
        print(f"  Model:       {args.bert_model}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Batch size:  {args.bert_batch_size}")
        print(f"  Device:      {args.semantic_device}")
        print("  Representation: mean-pooled last hidden state of a full forward pass")

        model = AutoModel.from_pretrained(args.bert_model, trust_remote_code=args.trust_remote_code)
        model = model.to(args.semantic_device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=args.trust_remote_code)
        print(f"  Hidden size: {model.config.hidden_size}")

        with step(f"Embedding {len(inputs.token_texts)} full token strings"):
            full_token = bert_embeddings(
                inputs.token_texts, model, tokenizer, args.bert_batch_size, args.semantic_device
            )
            print(f"  Shape: {full_token.shape}")

        with step(f"Embedding {len(inputs.subword_texts)} unique subwords"):
            subwords = bert_embeddings(
                inputs.subword_texts, model, tokenizer, args.bert_batch_size, args.semantic_device
            )
            print(f"  Shape: {subwords.shape}")

        del model, tokenizer
        free_cuda()
        print("\nReleased BERT model")

    return SemanticEmbeddings(
        label="BERT",
        full_token=full_token,
        token_id_to_row={tid: row for row, tid in enumerate(inputs.token_ids)},
        subword=dict(zip(inputs.subword_ids, subwords)),
    )


def build_gemma_semantics(args: argparse.Namespace, inputs: SemanticInputs) -> SemanticEmbeddings:
    on_cuda = args.semantic_device == "cuda"
    dtype = torch.bfloat16 if on_cuda else torch.float32

    with phase("Phase 3b: Gemma semantic embeddings"):
        print(f"  Model:       {args.gemma_model}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Batch size:  {args.gemma_batch_size}")
        print(f"  Device:      {args.semantic_device}")
        print(f"  Dtype:       {dtype}")
        print("  Representation: embed_tokens lookup, no forward pass")

        model = AutoModelForCausalLM.from_pretrained(
            args.gemma_model,
            dtype=dtype,
            device_map="auto" if on_cuda else None,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.gemma_model, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Hidden size: {model.config.hidden_size}")

        with step(f"Embedding {len(inputs.token_texts)} full token strings"):
            full_token = gemma_embeddings(
                inputs.token_texts, model, tokenizer, args.gemma_batch_size, args.semantic_device
            )
            print(f"  Shape: {full_token.shape}")

        with step(f"Embedding {len(inputs.subword_texts)} unique subwords"):
            subwords = gemma_embeddings(
                inputs.subword_texts, model, tokenizer, args.gemma_batch_size, args.semantic_device
            )
            print(f"  Shape: {subwords.shape}")

        del model, tokenizer
        free_cuda()
        print("\nReleased Gemma model")

    return SemanticEmbeddings(
        label="Gemma",
        full_token=full_token,
        token_id_to_row={tid: row for row, tid in enumerate(inputs.token_ids)},
        subword=dict(zip(inputs.subword_ids, subwords)),
    )


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def decompose_new_tokens(
    new_token_ids: Sequence[int], extended_tokenizer, original_tokenizer, original_vocab_size: int
) -> Decomposition:
    plans: dict[int, TokenPlan] = {}
    multi_token_ids: list[int] = []
    unique_subword_ids: set[int] = set()
    subword_counts: list[int] = []

    for token_id in tqdm(new_token_ids, desc="Decomposing tokens"):
        token_str = extended_tokenizer.convert_ids_to_tokens(token_id)

        if token_str in original_tokenizer.all_special_tokens:
            plans[token_id] = TokenPlan("mean", token_str, [])
            continue

        # Decode first so that tokenizer-internal markers (e.g. the SentencePiece
        # word-boundary prefix) do not leak into the re-encoding.
        decoded_text = extended_tokenizer.decode([token_id])
        subword_ids = original_tokenizer.encode(decoded_text, add_special_tokens=False)
        if subword_ids:
            subword_counts.append(len(subword_ids))

        valid_ids = [sid for sid in subword_ids if sid < original_vocab_size]
        if not valid_ids:
            plans[token_id] = TokenPlan("mean", token_str, [])
        elif len(valid_ids) == 1:
            plans[token_id] = TokenPlan("copy", token_str, valid_ids)
        else:
            plans[token_id] = TokenPlan("multi", token_str, valid_ids)
            multi_token_ids.append(token_id)
            unique_subword_ids.update(valid_ids)

    return Decomposition(
        plans=plans,
        multi_token_ids=multi_token_ids,
        unique_subword_ids=sorted(unique_subword_ids),
        subword_counts=subword_counts,
    )


def report_decomposition(decomposition: Decomposition) -> None:
    categories = [plan.category for plan in decomposition.plans.values()]
    print("\nDecomposition summary:")
    print(f"  Mean fallback (empty or special):          {categories.count('mean')}")
    print(f"  Direct copy (single subword):              {categories.count('copy')}")
    print(f"  Multi-subword (weighted averaging):        {categories.count('multi')}")
    print(f"  Unique subword IDs across multi-subwords:  {len(decomposition.unique_subword_ids)}")

    counts = decomposition.subword_counts
    if counts:
        print("\nSubword statistics over all non-empty tokens:")
        print(f"  Average subwords per new token: {sum(counts) / len(counts):.2f}")
        print(f"  Minimum subwords: {min(counts)}")
        print(f"  Maximum subwords: {max(counts)}")
        print(f"  Tokens analyzed:  {len(counts)}")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def resolve_target_norm(
    all_norms: torch.Tensor, hindi_norms: torch.Tensor | None, use_hindi: bool, correction_enabled: bool, side: str
) -> tuple[torch.Tensor, str]:
    if use_hindi and hindi_norms is not None and hindi_norms.numel() > 0:
        return hindi_norms.median(), "Hindi tokens"
    if use_hindi and correction_enabled:
        print(f"WARNING: no Hindi tokens found, using all-token median for {side} norm correction")
    return all_norms.median(), "all tokens"


def report_original_norms(
    input_norms: torch.Tensor,
    output_norms: torch.Tensor,
    hindi_input_norms: torch.Tensor | None,
    hindi_output_norms: torch.Tensor | None,
    result: InitResult,
    args: argparse.Namespace,
) -> None:
    print("\n" + RULE)
    print("Original embedding norm statistics")
    print(RULE)
    print(f"  Input  embeddings  - {describe_tensor(input_norms)}")
    print(f"  Output embeddings  - {describe_tensor(output_norms)}")
    if hindi_input_norms is not None:
        count = hindi_input_norms.numel()
        print(f"\n  Hindi input embeddings  ({count} tokens) - {describe_tensor(hindi_input_norms)}")
        print(f"  Hindi output embeddings ({count} tokens) - {describe_tensor(hindi_output_norms)}")

    if args.input_norm_correction:
        print(
            f"\n  Input norm correction ON, target (median of {result.input_norm_source}): "
            f"{result.target_input_norm:.4f}"
        )
    else:
        print("\n  Input norm correction OFF")
    if args.output_norm_correction:
        print(
            f"  Output norm correction ON, target (median of {result.output_norm_source}): "
            f"{result.target_output_norm:.4f}"
        )
    else:
        print("  Output norm correction OFF")
    print(RULE)


def build_sample_record(
    token_id: int,
    plan: TokenPlan,
    original_tokenizer,
    input_weights: torch.Tensor,
    output_weights: torch.Tensor,
    input_sims: list[float] | None,
    output_sims: list[float] | None,
) -> SampleRecord:
    subword_ids = plan.subword_ids
    try:
        decoded_text = original_tokenizer.decode(subword_ids)
    except Exception:
        decoded_text = "<decode error>"

    subword_texts = []
    for subword_id in subword_ids:
        try:
            subword_texts.append(original_tokenizer.decode([subword_id]))
        except Exception:
            subword_texts.append(f"<decode error: {subword_id}>")

    return SampleRecord(
        token_id=token_id,
        token_str=plan.token_str,
        decoded_text=decoded_text,
        subword_ids=list(subword_ids),
        subword_tokens=original_tokenizer.convert_ids_to_tokens(subword_ids),
        subword_texts=subword_texts,
        char_lengths=decoded_char_lengths(subword_ids, original_tokenizer),
        input_weights=input_weights.float().cpu().tolist(),
        output_weights=output_weights.float().cpu().tolist(),
        input_sims=input_sims,
        output_sims=output_sims,
    )


def initialize_new_embeddings(
    model,
    original_tokenizer,
    new_token_ids: Sequence[int],
    original_vocab_size: int,
    decomposition: Decomposition,
    target_token_ids: Sequence[int],
    semantic: dict[str, SemanticEmbeddings],
    args: argparse.Namespace,
) -> InitResult:
    result = InitResult()

    print("\nResizing token embeddings...")
    model.resize_token_embeddings(original_vocab_size + len(new_token_ids))

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings().weight
        output_embeddings = model.get_output_embeddings().weight

        mean_input = input_embeddings[:original_vocab_size].mean(dim=0)
        mean_output = output_embeddings[:original_vocab_size].mean(dim=0)
        original_input_norms = input_embeddings[:original_vocab_size].norm(dim=1)
        original_output_norms = output_embeddings[:original_vocab_size].norm(dim=1)

        hindi_input_norms = None
        hindi_output_norms = None
        if target_token_ids:
            hindi_ids = list(target_token_ids)
            hindi_input_norms = input_embeddings[hindi_ids].norm(dim=1)
            hindi_output_norms = output_embeddings[hindi_ids].norm(dim=1)

        target_input_norm, result.input_norm_source = resolve_target_norm(
            original_input_norms, hindi_input_norms, args.input_hindi_norm, args.input_norm_correction, "input"
        )
        target_output_norm, result.output_norm_source = resolve_target_norm(
            original_output_norms, hindi_output_norms, args.output_hindi_norm, args.output_norm_correction, "output"
        )
        result.target_input_norm = target_input_norm.item()
        result.target_output_norm = target_output_norm.item()

        report_original_norms(
            original_input_norms, original_output_norms, hindi_input_norms, hindi_output_norms, result, args
        )

        for token_id in tqdm(new_token_ids, desc="Initializing embeddings"):
            plan = decomposition.plans[token_id]
            try:
                if plan.category == "mean":
                    input_embeddings[token_id] = mean_input
                    output_embeddings[token_id] = mean_output
                    result.from_mean += 1
                    continue

                if plan.category == "copy":
                    source_id = plan.subword_ids[0]
                    input_embeddings[token_id] = input_embeddings[source_id].clone()
                    output_embeddings[token_id] = output_embeddings[source_id].clone()
                    result.from_copy += 1
                    continue

                input_weights, input_sims = subword_weights(
                    args.input_averaging,
                    token_id,
                    plan.subword_ids,
                    original_tokenizer,
                    semantic,
                    args.temperature,
                    input_embeddings.dtype,
                    input_embeddings.device,
                )
                output_weights, output_sims = subword_weights(
                    args.output_averaging,
                    token_id,
                    plan.subword_ids,
                    original_tokenizer,
                    semantic,
                    args.temperature,
                    output_embeddings.dtype,
                    output_embeddings.device,
                )

                averaged_input = (input_embeddings[plan.subword_ids] * input_weights.unsqueeze(1)).sum(dim=0)
                averaged_output = (output_embeddings[plan.subword_ids] * output_weights.unsqueeze(1)).sum(dim=0)

                if args.input_norm_correction:
                    result.input_norms_before.append(averaged_input.norm().item())
                    norm = averaged_input.norm()
                    if norm > 1e-8:
                        averaged_input = averaged_input * (target_input_norm / norm)
                    result.input_norms_after.append(averaged_input.norm().item())

                if args.output_norm_correction:
                    result.output_norms_before.append(averaged_output.norm().item())
                    norm = averaged_output.norm()
                    if norm > 1e-8:
                        averaged_output = averaged_output * (target_output_norm / norm)
                    result.output_norms_after.append(averaged_output.norm().item())

                input_embeddings[token_id] = averaged_input
                output_embeddings[token_id] = averaged_output
                result.from_subwords += 1

                if len(result.samples) < args.num_samples:
                    result.samples.append(
                        build_sample_record(
                            token_id,
                            plan,
                            original_tokenizer,
                            input_weights,
                            output_weights,
                            input_sims,
                            output_sims,
                        )
                    )

            except Exception as error:
                print(f"Warning: token {token_id} failed ({error}), falling back to mean")
                input_embeddings[token_id] = mean_input
                output_embeddings[token_id] = mean_output
                result.from_mean += 1

    return result


# ---------------------------------------------------------------------------
# Post-initialization reports
# ---------------------------------------------------------------------------


def report_samples(
    samples: Sequence[SampleRecord], args: argparse.Namespace, semantic: dict[str, SemanticEmbeddings]
) -> None:
    if not samples:
        return

    input_label = semantic[args.input_averaging].label if args.input_averaging in semantic else ""
    output_label = semantic[args.output_averaging].label if args.output_averaging in semantic else ""

    print(f"\nFirst {len(samples)} multi-subword tokens:")
    for index, sample in enumerate(samples, 1):
        count = len(sample.subword_ids)
        total_chars = sum(sample.char_lengths)
        uniform = [1.0 / count] * count
        char_weighted = [length / total_chars for length in sample.char_lengths]
        max_char = [0.0] * count
        max_char[sample.char_lengths.index(max(sample.char_lengths))] = 1.0

        print(f"\n  Sample {index}: {sample.token_str!r} (ID {sample.token_id}, {count} subwords)")
        print(f"    Decoded text:      {sample.decoded_text!r}")
        print(f"    Subword tokens:    {sample.subword_tokens}")
        print(f"    Subword texts:     {sample.subword_texts}")
        print(f"    Subword IDs:       {sample.subword_ids}")
        print(f"    Char lengths:      {sample.char_lengths}")
        print(f"    Uniform weights:   {format_weights(uniform)}")
        print(f"    Char weights:      {format_weights(char_weighted)}")
        print(f"    Max-char weights:  {format_weights(max_char)}")
        if sample.input_sims is not None:
            print(f"    {input_label} cosine sims (input):  {format_weights(sample.input_sims)}")
        if sample.output_sims is not None:
            print(f"    {output_label} cosine sims (output): {format_weights(sample.output_sims)}")
        print(f"    Input  weights used ({args.input_averaging:>14s}): {format_weights(sample.input_weights)}")
        print(f"    Output weights used ({args.output_averaging:>14s}): {format_weights(sample.output_weights)}")


def report_norm_analysis(model, original_vocab_size: int, result: InitResult, args: argparse.Namespace) -> None:
    print("\n" + RULE)
    print("Embedding norm analysis")
    print(RULE)

    with torch.no_grad():
        new_input_norms = model.get_input_embeddings().weight[original_vocab_size:].norm(dim=1)
        new_output_norms = model.get_output_embeddings().weight[original_vocab_size:].norm(dim=1)

    print(f"\n  New token embeddings ({new_input_norms.numel()} tokens):")
    print(f"    Input  - {describe_tensor(new_input_norms)}")
    print(f"    Output - {describe_tensor(new_output_norms)}")

    if args.input_norm_correction and result.input_norms_before:
        print(f"\n  Input norm correction over {len(result.input_norms_before)} averaged tokens:")
        print(f"    Before - {describe_values(result.input_norms_before)}")
        print(f"    After  - {describe_values(result.input_norms_after)}")
        print(f"    Target (median of {result.input_norm_source}): {result.target_input_norm:.4f}")
    else:
        print("\n  Input norm correction OFF, no before/after comparison")

    if args.output_norm_correction and result.output_norms_before:
        print(f"\n  Output norm correction over {len(result.output_norms_before)} averaged tokens:")
        print(f"    Before - {describe_values(result.output_norms_before)}")
        print(f"    After  - {describe_values(result.output_norms_after)}")
        print(f"    Target (median of {result.output_norm_source}): {result.target_output_norm:.4f}")
    else:
        print(f"\n  Output norm correction OFF, plain {args.output_averaging} average preserved")

    print(RULE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    script_start = time.time()

    with phase("Phase 1: Loading model and tokenizers"):
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=DTYPES[args.dtype],
            trust_remote_code=args.trust_remote_code,
        )
        original_tokenizer = AutoTokenizer.from_pretrained(args.base_model, fix_mistral_regex=True)
        extended_tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer, fix_mistral_regex=True)

    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(extended_tokenizer)
    if new_vocab_size <= original_vocab_size:
        raise SystemExit(
            f"Extended tokenizer has {new_vocab_size} tokens, which does not exceed the "
            f"base model's {original_vocab_size} embeddings; nothing to initialize."
        )

    print(f"\nOriginal vocab size: {original_vocab_size}")
    print(f"New vocab size:      {new_vocab_size}")
    print(f"New tokens to add:   {new_vocab_size - original_vocab_size}")
    print(f"Input  embeddings:   {tuple(model.get_input_embeddings().weight.shape)}")
    print(f"Output embeddings:   {tuple(model.get_output_embeddings().weight.shape)}")

    target_token_ids: list[int] = []
    if hindi_norm_in_use(args):
        with phase("Phase 1b: Locating target-script tokens in the original vocabulary"):
            target_tokens = find_target_script_tokens(original_tokenizer, original_vocab_size)
            target_token_ids = sorted(token_id for token_id, _ in target_tokens)
            share = 100.0 * len(target_token_ids) / original_vocab_size
            print(f"Found {len(target_token_ids)} target-script tokens ({share:.2f}% of the vocabulary)")
            for token_id, text in target_tokens[:10]:
                print(f"    ID {token_id}: {text!r}")

    new_token_ids = list(range(original_vocab_size, new_vocab_size))
    with phase("Phase 2: Decomposing new tokens into original subwords"):
        decomposition = decompose_new_tokens(
            new_token_ids, extended_tokenizer, original_tokenizer, original_vocab_size
        )
        report_decomposition(decomposition)

    semantic: dict[str, SemanticEmbeddings] = {}
    methods = semantic_methods_in_use(args)
    if methods:
        with phase("Phase 3: Decoding texts for semantic embedding"):
            semantic_inputs = collect_semantic_inputs(extended_tokenizer, original_tokenizer, decomposition)
        if "bert_weighted" in methods:
            semantic["bert_weighted"] = build_bert_semantics(args, semantic_inputs)
        if "gemma_weighted" in methods:
            semantic["gemma_weighted"] = build_gemma_semantics(args, semantic_inputs)
    else:
        print("\nSkipping semantic embeddings: neither side uses a semantic averaging method")

    with phase("Phase 4: Initializing new token embeddings"):
        print_configuration(args)
        result = initialize_new_embeddings(
            model,
            original_tokenizer,
            new_token_ids,
            original_vocab_size,
            decomposition,
            target_token_ids,
            semantic,
            args,
        )

        print("\nInitialization statistics:")
        print(f"  From subword averaging:   {result.from_subwords}")
        print(f"  From single-subword copy: {result.from_copy}")
        print(f"  From mean fallback:       {result.from_mean}")
        print(f"  Total new tokens:         {len(new_token_ids)}")

        report_samples(result.samples, args, semantic)
        report_norm_analysis(model, original_vocab_size, result, args)

    with phase(f"Phase 5: Saving to {args.output_dir}"):
        from vocab_pad import pad_vocab_to_multiple

        pad_vocab_to_multiple(model, 4)  # TP-safe: divisible by TP (=4); minimal padding across all arms/methods
        model.save_pretrained(args.output_dir)
        extended_tokenizer.save_pretrained(args.output_dir)

    total_elapsed = time.time() - script_start
    print("\n" + RULE)
    print("DONE")
    print(RULE)
    print(f"  Wall-clock time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print_configuration(args)
    print(f"  Base model:              {args.base_model}")
    print(f"  Extended tokenizer:      {args.extended_tokenizer}")
    print(f"  Output:                  {args.output_dir}")
    print(RULE)


if __name__ == "__main__":
    main()
