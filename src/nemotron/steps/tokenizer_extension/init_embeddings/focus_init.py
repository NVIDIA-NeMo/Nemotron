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

"""FOCUS initialization of new-token embeddings from a fastText auxiliary space.

Implements FOCUS (Dobler & de Melo, EMNLP 2023, https://arxiv.org/abs/2305.14481):
each token added by the extended tokenizer is initialized as a Sparsemax-weighted
combination of the pretrained embeddings of the base-vocabulary tokens that are
closest to it in an auxiliary embedding space.

Here the auxiliary space is a fastText binary model (loaded with
fasttext.load_model, not a .vec file), which gives a vector for any string via
character n-grams, so new tokens need not appear in the fastText vocabulary.

For every new token the script computes cosine similarity to each candidate base
token in fastText space, divides by a temperature to sharpen the distribution,
and applies Sparsemax.  Sparsemax projects onto the simplex and drives most
weights to exactly zero, so only a handful of genuinely similar base tokens
contribute.  Raw cosine similarities differ too little for this to work
unscaled: without the temperature the weight mass spreads over thousands of
tokens and the result degenerates into a dense average.

The candidate pool is either the Devanagari tokens of the base vocabulary
(default, a Hindi-specialized variant) or all base tokens (standard FOCUS).

Example
    python extend_model_focus_ft.py \
        --extended-tokenizer /path/to/merged_tokenizer \
        --output-dir /path/to/extended_model \
        --fasttext-model /path/to/cc.hi.300.bin
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
DEVANAGARI_RANGE = (0x0900, 0x097F)
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
RULE = "=" * 60


#: Where the cc.<lang>.300 vectors live when FASTTEXT_CACHE_DIR is unset. The
#: two init paths used to disagree (/tmp vs /tmp/fasttext), so a file staged by
#: one was invisible to the other and got fetched twice.
FASTTEXT_CACHE_DEFAULT = "/tmp/fasttext"


def fasttext_cache_dir() -> Path:
    """The one directory the vectors are cached in."""
    import os

    return Path(os.environ.get("FASTTEXT_CACHE_DIR") or FASTTEXT_CACHE_DEFAULT)


def fasttext_cache_path(url: str) -> Path:
    """Where the asset behind ``url`` is cached, by its own filename."""
    import os

    return fasttext_cache_dir() / os.path.basename(url).replace(".gz", "")


def _announce_fasttext(url: str, dest: Path) -> None:
    """Say what is about to be transferred, and where it lands.

    These files are multi-GB. A download that starts with no size and no
    destination looks like a hang, and on an ephemeral cache it silently
    repeats on every worker.
    """
    import os
    import urllib.request

    size = None
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=30
        ) as resp:
            length = resp.headers.get("Content-Length")
            size = int(length) if length else None
    except Exception:  # noqa: BLE001 - the size is a courtesy, not a gate
        pass
    human = f"{size / 1e9:.2f} GB compressed" if size else "size unknown"
    print(f"  Staging fastText ({human})")
    print(f"    from: {url}")
    print(f"    to:   {dest}")
    if not os.environ.get("FASTTEXT_CACHE_DIR"):
        print(
            f"    NOTE: FASTTEXT_CACHE_DIR is unset, so this caches under "
            f"{FASTTEXT_CACHE_DEFAULT} and is re-fetched on every fresh worker. "
            "Point it at durable shared storage to stage it once."
        )


def ensure_fasttext(path: str, url: str | None) -> str:
    """Config-driven staging: if the fastText .bin isn't at `path`, download it from
    `url` (on the compute node) and .gz-decompress. Idempotent — a second focus cell
    reuses the cached file. Writes atomically so a crashed download can't half-fill it."""
    import os

    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        print(f"  fastText present: {p} ({p.stat().st_size / 1e9:.2f} GB)")
        return str(p)
    if not url:
        raise SystemExit(f"fastText model not at {p} and no --fasttext-url to fetch it.")
    import gzip
    import shutil
    import tempfile
    import urllib.request

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".downloading")
    _announce_fasttext(url, p)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(p.parent)) as raw:
        urllib.request.urlretrieve(url, raw.name)
        src = raw.name
    try:
        if url.endswith(".gz"):
            with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
        else:
            shutil.move(src, tmp)
        os.replace(tmp, p)  # atomic
    finally:
        for f in (src, tmp):
            try:
                os.path.exists(f) and os.unlink(f)
            except OSError:
                pass
    print(f"  fastText ready: {p} ({p.stat().st_size / 1e9:.2f} GB)")
    return str(p)


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
    # Both default to None so --language can supply them below. --fasttext-model
    # used to be required=True, which made argparse reject any language-driven
    # config before the profile was ever applied; --fasttext-url used to default
    # to the *Hindi* cc.hi.300 URL, which meant the `args.fasttext_url is None`
    # override never fired and a Vietnamese run would have silently trained on
    # Hindi vectors. Same failure shape as the MuRIL-for-Vietnamese bug.
    paths.add_argument(
        "--fasttext-model",
        default=None,
        help="fastText binary model (.bin) used as the auxiliary space. "
        "If absent and --fasttext-url is set, it is downloaded here (on the "
        "compute node). Defaults to $FASTTEXT_CACHE_DIR/cc.<code>.300.bin "
        "resolved from --language.",
    )
    paths.add_argument(
        "--fasttext-url",
        default=None,
        help="Source to fetch the fastText .bin from if --fasttext-model is missing. "
        ".gz is auto-decompressed. Defaults to the cc.<code>.300 URL for "
        "--language, or Hindi when neither is given (legacy).",
    )
    paths.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute custom modeling code shipped in the model repo. "
        "Off by default; required by architectures whose code lives there.",
    )
    paths.add_argument(
        "--dtype", choices=sorted(DTYPES), default="bfloat16", help="Precision to load the base model in."
    )

    focus = parser.add_argument_group("FOCUS")
    focus.add_argument(
        "--candidate-pool",
        choices=("hindi", "target", "all"),
        default="target",
        help="Base tokens eligible to contribute: Devanagari only, or all.",
    )
    focus.add_argument(
        "--sparsemax-temperature",
        type=float,
        default=0.05,
        help="Similarities are divided by this before Sparsemax; "
        "smaller is sparser (0.01 to 0.1 is the useful range).",
    )

    report = parser.add_argument_group("reporting")
    report.add_argument(
        "--num-samples", type=int, default=10, help="New tokens to report contributing base tokens for."
    )
    report.add_argument(
        "--top-contributors", type=int, default=5, help="Contributing base tokens listed per reported sample."
    )

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
        if args.fasttext_url is None:
            args.fasttext_url = _fu(args.language)
        if args.fasttext_model is None:
            # One cache shared with the replace path, so a file staged by either
            # is reused by the other. ensure_fasttext is idempotent and writes
            # atomically, so a crashed download never lands here.
            cache = fasttext_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            args.fasttext_model = str(cache / f"cc.{_prof.fasttext}.300.bin")
    # Legacy path: no --language at all keeps the original Hindi default, so
    # every pre-existing Hindi config behaves exactly as before.
    if args.fasttext_url is None:
        args.fasttext_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz"
    if args.fasttext_model is None:
        parser.error("--fasttext-model is required when --language is not given")
    if args.sparsemax_temperature <= 0:
        parser.error("--sparsemax-temperature must be positive")
    if args.num_samples < 0 or args.top_contributors < 1:
        parser.error("--num-samples must not be negative and --top-contributors must be >= 1")
    return args


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CandidatePool:
    """Base tokens eligible to contribute, and their fastText vectors."""

    token_ids: np.ndarray
    unit_vectors: np.ndarray
    label: str


@dataclass
class FocusSample:
    token_id: int
    token_text: str
    nonzero_count: int
    top_token_ids: list[int]
    top_token_texts: list[str]
    top_similarities: list[float]
    top_weights: list[float]


@dataclass
class FocusResult:
    total: int = 0
    successful: int = 0
    failed: int = 0
    nonzero_counts: list[int] = field(default_factory=list)
    samples: list[FocusSample] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Sparsemax
# ---------------------------------------------------------------------------


def sparsemax(scores: np.ndarray) -> np.ndarray:
    """Euclidean projection of `scores` onto the probability simplex.

    Sparsemax (Martins & Astudillo, ICML 2016) returns the point on the simplex
    closest to the input, which unlike softmax is usually sparse:

        support size k = max{ k : 1 + k * s_(k) > sum_{j<=k} s_(j) }  over sorted s
        threshold  tau = (sum_{j<=k} s_(j) - 1) / k
        weights_i      = max(s_i - tau, 0)
    """
    scores = np.asarray(scores, dtype=np.float64)
    descending = np.sort(scores)[::-1]
    cumulative = np.cumsum(descending)
    positions = np.arange(1, len(scores) + 1, dtype=np.float64)

    support = int(np.max(np.where(1.0 + positions * descending > cumulative))) + 1
    threshold = (cumulative[support - 1] - 1.0) / support
    return np.maximum(scores - threshold, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Vocabulary and fastText helpers
# ---------------------------------------------------------------------------


def is_target_script(text: str) -> bool:
    """True if ``text`` lies in the active target script, which defaults to
    Devanagari and is selected by --language / set_target_script()."""
    from script_ranges import is_target

    return is_target(text)


def decode_vocabulary(tokenizer, vocab_size: int, desc: str) -> list[str]:
    texts = []
    for token_id in tqdm(range(vocab_size), desc=desc):
        try:
            texts.append(tokenizer.decode([token_id]))
        except Exception:
            texts.append(tokenizer.convert_ids_to_tokens(token_id))
    return texts


def decode_new_tokens(tokenizer, token_ids: Sequence[int]) -> list[str]:
    texts = []
    for token_id in tqdm(token_ids, desc="Decoding new tokens"):
        try:
            texts.append(tokenizer.decode([token_id]))
        except Exception:
            texts.append(tokenizer.convert_ids_to_tokens(token_id))
    return texts


def clean_token_text(text: str) -> str:
    """Strip tokenizer word-boundary markers before a fastText lookup."""
    return text.replace("\u2581", " ").replace("\u0120", " ").strip()


def fasttext_vectors(texts: Sequence[str], ft_model) -> np.ndarray:
    """One fastText vector per text; character n-grams cover out-of-vocabulary text."""
    dimension = ft_model.get_dimension()
    vectors = []
    for text in tqdm(texts, desc="Getting fastText vectors", unit="token"):
        cleaned = clean_token_text(text)
        vectors.append(ft_model.get_word_vector(cleaned) if cleaned else np.zeros(dimension, dtype=np.float32))
    return np.array(vectors, dtype=np.float32)


def unit_normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def build_candidate_pool(pool: str, base_texts: Sequence[str], ft_model) -> CandidatePool:
    if pool in ("hindi", "target"):
        token_ids = np.array([i for i, text in enumerate(base_texts) if is_target_script(text)], dtype=np.int64)
        if token_ids.size == 0:
            raise SystemExit("No Devanagari tokens in the base vocabulary; rerun with --candidate-pool all")
        share = 100.0 * token_ids.size / len(base_texts)
        label = f"{token_ids.size} Devanagari tokens"
        print(f"  Devanagari candidates: {token_ids.size} of {len(base_texts)} base tokens ({share:.2f}%)")
    else:
        token_ids = np.arange(len(base_texts), dtype=np.int64)
        label = f"all {token_ids.size} base tokens"
        print(f"  Using all {token_ids.size} base tokens as candidates")

    texts = [base_texts[i] for i in token_ids]
    vectors = fasttext_vectors(texts, ft_model)
    print(f"  fastText matrix shape: {vectors.shape}")
    return CandidatePool(token_ids=token_ids, unit_vectors=unit_normalize(vectors), label=label)


def decode_for_display(tokenizer, token_id: int) -> str:
    """Best-effort readable form of a token, for logging only."""
    token_id = int(token_id)
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if text:
            return text
    except Exception:
        pass
    try:
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        return tokenizer.convert_tokens_to_string([token_str]) or token_str
    except Exception:
        return f"<id:{token_id}>"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_focus_embeddings(
    model,
    original_tokenizer,
    extended_tokenizer,
    new_token_ids: Sequence[int],
    new_vectors: np.ndarray,
    pool: CandidatePool,
    original_vocab_size: int,
    args: argparse.Namespace,
) -> tuple[FocusResult, torch.Tensor, torch.Tensor]:
    result = FocusResult(total=len(new_token_ids))

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings().weight
        output_embeddings = model.get_output_embeddings().weight

        original_input = input_embeddings[:original_vocab_size].clone()
        original_output = output_embeddings[:original_vocab_size].clone()
        mean_input = original_input.mean(dim=0)
        mean_output = original_output.mean(dim=0)

        for index, token_id in enumerate(tqdm(new_token_ids, desc="FOCUS initialization", unit="token")):
            try:
                query = new_vectors[index]
                query = query / (np.linalg.norm(query) + 1e-8)
                similarities = pool.unit_vectors @ query
                weights = sparsemax(similarities / args.sparsemax_temperature)

                selected = np.flatnonzero(weights)
                selected_weights = weights[selected]
                selected_token_ids = pool.token_ids[selected]
                result.nonzero_counts.append(selected.size)

                weight_tensor = torch.tensor(
                    selected_weights, dtype=input_embeddings.dtype, device=input_embeddings.device
                ).unsqueeze(1)
                input_embeddings[token_id] = (input_embeddings[selected_token_ids] * weight_tensor).sum(dim=0)
                output_embeddings[token_id] = (output_embeddings[selected_token_ids] * weight_tensor).sum(dim=0)
                result.successful += 1

                if len(result.samples) < args.num_samples:
                    order = np.argsort(selected_weights)[::-1][: args.top_contributors]
                    top_ids = selected_token_ids[order]
                    result.samples.append(
                        FocusSample(
                            token_id=token_id,
                            token_text=decode_for_display(extended_tokenizer, token_id),
                            nonzero_count=int(selected.size),
                            top_token_ids=top_ids.tolist(),
                            top_token_texts=[decode_for_display(original_tokenizer, i) for i in top_ids],
                            top_similarities=similarities[selected[order]].tolist(),
                            top_weights=selected_weights[order].tolist(),
                        )
                    )

            except Exception as error:
                print(f"Warning: token {token_id} failed ({error}), falling back to mean")
                input_embeddings[token_id] = mean_input
                output_embeddings[token_id] = mean_output
                result.failed += 1

    return result, original_input, original_output


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_embeddings(
    model,
    original_input: torch.Tensor,
    original_output: torch.Tensor,
    original_vocab_size: int,
    new_token_ids: Sequence[int],
    samples: Sequence[FocusSample],
) -> None:
    """Report and raise on anything that indicates a broken initialization."""
    print("\n" + RULE)
    print("Validation")
    print(RULE)

    errors: list[str] = []
    warnings: list[str] = []

    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    sides = (
        ("input", input_embeddings[original_vocab_size:], original_input),
        ("output", output_embeddings[original_vocab_size:], original_output),
    )

    print("\n1. Original embeddings unchanged")
    for name, original, current in (
        ("input", original_input, input_embeddings),
        ("output", original_output, output_embeddings),
    ):
        difference = (current[:original_vocab_size] - original).abs().max().item()
        if difference > 1e-6:
            errors.append(f"original {name} embeddings changed, max difference {difference:.2e}")
        else:
            print(f"   original {name} embeddings intact (max difference {difference:.2e})")

    print("\n2. New embeddings are populated")
    for name, new, _ in sides:
        nan_count = torch.isnan(new).sum().item()
        if nan_count:
            errors.append(f"{nan_count} NaN values in the new {name} embeddings")
        else:
            print(f"   no NaN values in the new {name} embeddings")
        zero_count = (new.abs().max(dim=1)[0] < 1e-8).sum().item()
        if zero_count:
            warnings.append(f"{zero_count} all-zero new {name} embeddings")
        else:
            print(f"   no all-zero new {name} embeddings")

    print("\n3. Distribution against the originals")
    for name, new, original in sides:
        print(
            f"   new {name}      - mean: {new.mean().item():.4f}, std: {new.std().item():.4f}, "
            f"range: [{new.min().item():.4f}, {new.max().item():.4f}]"
        )
        print(f"   original {name} - mean: {original.mean().item():.4f}, std: {original.std().item():.4f}")
        if abs(new.mean().item() - original.mean().item()) > 10 * original.std().item():
            warnings.append(
                f"new {name} mean {new.mean().item():.4f} is far from the original mean {original.mean().item():.4f}"
            )

    print("\n4. Shapes")
    expected_dim = original_input.shape[1]
    for name, new, _ in sides:
        if new.shape[1] != expected_dim:
            errors.append(f"new {name} width {new.shape[1]} != {expected_dim}")
        elif new.shape[0] != len(new_token_ids):
            errors.append(f"new {name} row count {new.shape[0]} != {len(new_token_ids)}")
        else:
            print(f"   new {name} embeddings: {tuple(new.shape)}")

    print("\n5. Variance against the originals")
    for name, new, original in sides:
        new_variance = new.var().item()
        original_variance = original.var().item()
        if new_variance < 1e-6:
            errors.append(f"new {name} variance {new_variance:.2e} suggests uninitialized rows")
        elif new_variance < 0.1 * original_variance:
            warnings.append(
                f"new {name} variance {new_variance:.4f} is much lower than the original {original_variance:.4f}"
            )
        else:
            print(f"   new {name} variance {new_variance:.4f} (original {original_variance:.4f})")

    print("\n6. Sampled tokens")
    if not samples:
        print("   no samples collected")
    for position, sample in enumerate(samples[:5], 1):
        input_norm = input_embeddings[sample.token_id].norm().item()
        output_norm = output_embeddings[sample.token_id].norm().item()
        if (
            torch.isnan(input_embeddings[sample.token_id]).any()
            or torch.isnan(output_embeddings[sample.token_id]).any()
        ):
            errors.append(f"sample token {sample.token_id} ({sample.token_text!r}) has NaN values")
        elif min(input_norm, output_norm) < 1e-6:
            warnings.append(
                f"sample token {sample.token_id} ({sample.token_text!r}) has a tiny "
                f"norm (input {input_norm:.2e}, output {output_norm:.2e})"
            )
        else:
            print(
                f"   sample {position} {sample.token_text!r}: input norm {input_norm:.4f}, "
                f"output norm {output_norm:.4f}"
            )

    print("\n" + RULE)
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"  ERROR   {error}")
        raise ValueError("embedding validation failed; see the errors above")
    print("VALIDATION PASSED")
    for warning in warnings:
        print(f"  WARNING {warning}")
    print(RULE)


# ---------------------------------------------------------------------------
# Post-initialization reports
# ---------------------------------------------------------------------------


def report_sparsity(result: FocusResult, pool: CandidatePool) -> None:
    print("\nInitialization statistics:")
    print(f"  Total new tokens:            {result.total}")
    print(f"  Successfully initialized:    {result.successful}")
    print(f"  Failed (mean fallback):      {result.failed}")

    counts = result.nonzero_counts
    if not counts:
        return
    deviation = statistics.stdev(counts) if len(counts) > 1 else 0.0
    print(f"\nSparsemax support over {pool.label}:")
    print(
        f"  Non-zero weights per token - median: {statistics.median(counts):.0f}, "
        f"mean: {statistics.mean(counts):.1f}, std: {deviation:.1f}, "
        f"min: {min(counts)}, max: {max(counts)}"
    )
    sparsity = 100.0 * (1.0 - statistics.mean(counts) / pool.token_ids.size)
    print(f"  Average sparsity: {sparsity:.2f}%")


def report_samples(result: FocusResult, pool: CandidatePool) -> None:
    if not result.samples:
        return
    print(f"\nFirst {len(result.samples)} initialized tokens:")
    for position, sample in enumerate(result.samples, 1):
        print(
            f"\n  Sample {position}: {sample.token_text!r} (ID {sample.token_id}), "
            f"Sparsemax kept {sample.nonzero_count} of {pool.token_ids.size} candidates"
        )
        for rank, (token_id, text, similarity, weight) in enumerate(
            zip(sample.top_token_ids, sample.top_token_texts, sample.top_similarities, sample.top_weights), 1
        ):
            print(f"    {rank}. {text!r} (ID {token_id}) - fastText similarity {similarity:.4f}, weight {weight:.4f}")


def report_norms(model, original_input: torch.Tensor, original_output: torch.Tensor, original_vocab_size: int) -> None:
    print("\n" + RULE)
    print("Embedding norm analysis")
    print(RULE)

    with torch.no_grad():
        new_input = model.get_input_embeddings().weight[original_vocab_size:].norm(dim=1)
        new_output = model.get_output_embeddings().weight[original_vocab_size:].norm(dim=1)

        print("\n  Original embeddings:")
        print(f"    Input  - {describe_tensor(original_input.norm(dim=1), extremes=False)}")
        print(f"    Output - {describe_tensor(original_output.norm(dim=1), extremes=False)}")
        print(f"\n  New token embeddings ({new_input.numel()} tokens):")
        print(f"    Input  - {describe_tensor(new_input)}")
        print(f"    Output - {describe_tensor(new_output)}")
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
    new_token_ids = list(range(original_vocab_size, new_vocab_size))

    print(f"\nOriginal vocab size: {original_vocab_size}")
    print(f"New vocab size:      {new_vocab_size}")
    print(f"New tokens to add:   {len(new_token_ids)}")
    print(f"Input  embeddings:   {tuple(model.get_input_embeddings().weight.shape)}")
    print(f"Output embeddings:   {tuple(model.get_output_embeddings().weight.shape)}")

    # mean_resizing=False leaves the new rows untouched; FOCUS overwrites all of them.
    print("\nResizing token embeddings without automatic initialization...")
    model.resize_token_embeddings(new_vocab_size, mean_resizing=False)

    with phase(f"Phase 2: Building the candidate pool from {args.fasttext_model}"):
        ensure_fasttext(args.fasttext_model, args.fasttext_url)
        try:
            import fasttext  # lazy: only method=focus needs it (see replace_init)
        except ImportError as exc:
            raise SystemExit(
                "method=focus requires fastText. Install it with:  "
                "pip install 'nemotron[tokenizer-extension]'  (or pip install "
                "fasttext-wheel). Other init methods do not need it."
            ) from exc
        ft_model = fasttext.load_model(args.fasttext_model)
        print(f"  fastText dimension: {ft_model.get_dimension()}")
        base_texts = decode_vocabulary(original_tokenizer, original_vocab_size, "Decoding base vocabulary")
        pool = build_candidate_pool(args.candidate_pool, base_texts, ft_model)

    with phase("Phase 3: Embedding the new tokens in fastText space"):
        new_texts = decode_new_tokens(extended_tokenizer, new_token_ids)
        new_vectors = fasttext_vectors(new_texts, ft_model)
        print(f"  fastText matrix shape: {new_vectors.shape}")
        if new_vectors.shape[1] != pool.unit_vectors.shape[1]:
            raise SystemExit(
                f"fastText dimension mismatch: candidates have "
                f"{pool.unit_vectors.shape[1]}, new tokens have "
                f"{new_vectors.shape[1]}"
            )
        del ft_model
        print("  Released the fastText model")

    with phase("Phase 4: Initializing new embeddings with FOCUS"):
        print(f"  Candidate pool:          {pool.label}")
        print(f"  Sparsemax temperature:   {args.sparsemax_temperature}")
        result, original_input, original_output = initialize_focus_embeddings(
            model,
            original_tokenizer,
            extended_tokenizer,
            new_token_ids,
            new_vectors,
            pool,
            original_vocab_size,
            args,
        )
        report_sparsity(result, pool)
        report_samples(result, pool)
        validate_embeddings(model, original_input, original_output, original_vocab_size, new_token_ids, result.samples)
        report_norms(model, original_input, original_output, original_vocab_size)

    with phase(f"Phase 5: Saving to {args.output_dir}"):
        from vocab_pad import pad_vocab_to_multiple

        pad_vocab_to_multiple(model, 4)  # TP-safe: divisible by TP (=4); minimal padding across all arms/methods
        model.save_pretrained(args.output_dir)
        extended_tokenizer.save_pretrained(args.output_dir)

    total_elapsed = time.time() - script_start
    print("\n" + RULE)
    print("DONE")
    print(RULE)
    print(f"  Wall-clock time:         {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print(f"  Candidate pool:          {pool.label}")
    print(f"  Sparsemax temperature:   {args.sparsemax_temperature}")
    print(f"  fastText model:          {args.fasttext_model}")
    print(f"  Base model:              {args.base_model}")
    print(f"  Extended tokenizer:      {args.extended_tokenizer}")
    print(f"  Output:                  {args.output_dir}")
    print(RULE)


if __name__ == "__main__":
    main()
