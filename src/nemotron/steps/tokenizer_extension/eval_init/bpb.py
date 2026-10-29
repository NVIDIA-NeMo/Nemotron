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

"""Evaluate embedding-initialization quality by scoring a validation corpus.

Reports cross-entropy loss, perplexity, and bits-per-byte for one or more
extended models, optionally alongside the unextended base model as a reference.
Bits-per-byte is the metric to compare across models: loss and perplexity are
per-token and therefore not comparable between different vocabularies, while
bits-per-byte normalizes by the UTF-8 size of the text.

Documents are streamed from disk one at a time and scored with a sliding window,
so corpus size is not bounded by memory.  Run once on Hindi data and once on
English data to catch regressions on the original language.

Data formats
    .txt            one document per line
    .jsonl / .json  one JSON object per line; --text-field selects the field

Example
    python eval_tokenizer_init.py \
        --base-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
        --models /path/to/model-subword /path/to/model-focus \
        --data-file /path/to/hindi_val.jsonl \
        --max-tokens 500000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_LABEL = "base_model (reference)"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
IGNORE_INDEX = -100
LN2 = math.log(2)
SPIKE_THRESHOLD_PCT = 5.0
REGRESSION_THRESHOLD_PCT = 15.0


@dataclass
class EvalResult:
    loss: float
    perplexity: float
    bpb: float
    num_tokens: int
    num_bytes: int
    num_docs: int

    @property
    def finite(self) -> bool:
        return math.isfinite(self.bpb)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute custom modeling code shipped in the model repo. Off by default; "
        "required by architectures whose code lives in the repo rather than in "
        "transformers.",
    )

    models = parser.add_argument_group("models")
    models.add_argument("--models", nargs="+", required=True, help="Paths to one or more extended models to evaluate.")
    models.add_argument("--base-model", help="Unextended model to score first as a reference baseline.")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-file",
        default=None,
        help="Local validation corpus (.txt or .jsonl). Path-first: used when "
        "given, otherwise fall back to the --hf-dataset stream.",
    )
    data.add_argument(
        "--hf-dataset",
        default=None,
        help="HF dataset id to stream as the validation corpus when no local "
        "--data-file is given (e.g. ai4bharat/samanantar).",
    )
    data.add_argument("--hf-config", default=None, help="HF dataset config / subset (e.g. hi).")
    data.add_argument("--hf-split", default="train", help="HF dataset split to stream.")
    data.add_argument(
        "--skip-docs",
        type=int,
        default=0,
        help="Skip this many leading docs (held-out slice offset) when streaming HF.",
    )
    data.add_argument("--text-field", default="text", help="JSONL / HF field holding the document text.")
    data.add_argument(
        "--allow-token-cap-comparison",
        action="store_true",
        help="Score several models under --max-tokens anyway. The BPB "
        "values will NOT be comparable across tokenizers; only use "
        "this for within-tokenizer perplexity or to reproduce a "
        "historical run.",
    )
    data.add_argument("--max-docs", type=int, default=-1, help="Stop after this many documents (-1 for all).")
    data.add_argument("--max-tokens", type=int, default=-1, help="Stop after scoring this many tokens (-1 for all).")

    scoring = parser.add_argument_group("scoring")
    scoring.add_argument("--max-length", type=int, default=2048, help="Context window for each forward pass.")
    scoring.add_argument(
        "--stride", type=int, default=512, help="Sliding-window step; overlapping tokens are scored once."
    )

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument(
        "--dtype", choices=sorted(DTYPES), default="bfloat16", help="Precision to load the models in."
    )
    hardware.add_argument(
        "--device", help="Device string such as cuda, cuda:0, or cpu (default: cuda when available)."
    )

    parser.add_argument("--output-json", help="Write the results to this JSON file.")

    args = parser.parse_args(argv)
    if not args.data_file and not args.hf_dataset:
        parser.error("provide --data-file (local corpus) or --hf-dataset (stream)")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if not 1 <= args.stride <= args.max_length:
        parser.error("--stride must be between 1 and --max-length")
    return args


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def report_hardware(device: str, dtype_name: str) -> None:
    print("=" * 70)
    print("Hardware")
    print("=" * 70)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"  CUDA GPUs available: {count}")
        total_memory = 0.0
        for index in range(count):
            memory = torch.cuda.get_device_properties(index).total_memory / (1024**3)
            total_memory += memory
            print(f"    GPU {index}: {torch.cuda.get_device_name(index)} ({memory:.1f} GB)")
        print(f"  Total GPU memory: {total_memory:.1f} GB")
        print("  Models are sharded across all GPUs via device_map='auto'")
    else:
        print("  No CUDA GPUs available, running on CPU")
    print(f"  Selected device: {device} | dtype: {dtype_name}")


def report_gpu_memory() -> None:
    for index in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(index) / (1024**3)
        reserved = torch.cuda.memory_reserved(index) / (1024**3)
        if allocated > 0:
            print(f"  GPU {index} memory: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved")


def fmt(value: float, spec: str) -> str:
    return format(value, spec) if math.isfinite(value) else "N/A"


def safe_exp(value: float) -> float:
    """exp() that saturates instead of overflowing on a badly initialized model."""
    return math.exp(value) if value < 700 else float("inf")


def unique_labels(paths: Sequence[str]) -> list[str]:
    """Directory names, disambiguated when several models share one."""
    labels, seen = [], {}
    for path in paths:
        name = Path(path).name
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return labels


# ---------------------------------------------------------------------------
# Data streaming
# ---------------------------------------------------------------------------


def stream_texts(data_path: str, text_field: str = "text", max_docs: int = -1) -> Iterator[str]:
    """Yield documents from a plain-text or JSONL file without pre-loading."""
    is_json = Path(data_path).suffix.lower() in (".jsonl", ".json")
    yielded = 0

    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            if is_json:
                try:
                    text = json.loads(line).get(text_field, "")
                except json.JSONDecodeError:
                    continue
            else:
                text = line

            if text:
                yield text
                yielded += 1
                if max_docs > 0 and yielded >= max_docs:
                    return


def stream_hf(
    hf_dataset: str,
    hf_config: str | None,
    hf_split: str,
    text_field: str = "text",
    max_docs: int = -1,
    skip_docs: int = 0,
) -> Iterator[str]:
    """Yield documents from a streamed HF dataset (held-out slice via skip_docs)."""
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, hf_config, split=hf_split, streaming=True)
    yielded = 0
    for index, example in enumerate(ds):
        if index < skip_docs:
            continue
        text = example.get(text_field, "") if isinstance(example, dict) else ""
        if text:
            yield text
            yielded += 1
            if max_docs > 0 and yielded >= max_docs:
                return


def make_text_stream(args) -> tuple[callable, str]:
    """Return (factory, source_label): path-first local file, else HF stream."""
    if args.data_file:
        return (lambda: stream_texts(args.data_file, args.text_field, args.max_docs), args.data_file)
    label = f"hf:{args.hf_dataset}:{args.hf_config or '-'}:{args.hf_split}"
    return (
        lambda: stream_hf(
            args.hf_dataset, args.hf_config, args.hf_split, args.text_field, args.max_docs, args.skip_docs
        ),
        label,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    make_stream,
    total: int | None = None,
    max_length: int = 2048,
    stride: int = 512,
    max_tokens: int = -1,
    device: str = "cuda",
    desc: str = "Evaluating",
) -> EvalResult:
    """Score a corpus with a sliding window, counting every token exactly once.

    `make_stream` is a zero-arg factory returning a fresh iterator of document
    strings (local file or streamed HF dataset), so scoring is source-agnostic.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    docs_processed = 0

    progress = tqdm(make_stream(), desc=desc, unit="doc", total=total)

    for text in progress:
        input_ids = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False).input_ids[0]
        sequence_length = input_ids.size(0)
        if sequence_length == 0:
            continue

        previous_end = 0
        for begin in range(0, sequence_length, stride):
            end = min(begin + max_length, sequence_length)
            window = input_ids[begin:end].unsqueeze(0).to(device)

            # Ignore the tokens the previous window already scored, and the very
            # first token, which has no preceding context.
            targets = window.clone()
            overlap = previous_end - begin
            if overlap > 0:
                targets[0, :overlap] = IGNORE_INDEX
            if begin == 0:
                targets[0, 0] = IGNORE_INDEX

            scored = int((targets != IGNORE_INDEX).sum().item())
            if scored > 0:
                outputs = model(window, labels=targets)
                total_loss += outputs.loss.item() * scored
                total_tokens += scored

            previous_end = end

            if total_tokens > 0:
                mean_loss = total_loss / total_tokens
                progress.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    ppl=fmt(safe_exp(mean_loss), ".2f"),
                    bpb=f"{total_loss / (total_bytes * LN2):.4f}" if total_bytes else "0",
                    tokens=f"{total_tokens:,}",
                )

            if end == sequence_length:
                break

        # Count this document's bytes only now that its loss is fully
        # accumulated. Counting them up-front while breaking mid-document
        # inflated the BPB denominator against a partial numerator.
        total_bytes += len(text.encode("utf-8"))
        docs_processed += 1

        # Stop only on document boundaries, so bytes and loss always describe
        # the same text. Note `--max-docs` (which caps the input stream) is
        # tokenizer-independent and is the correct budget for cross-tokenizer
        # BPB; `--max-tokens` is not.
        if max_tokens > 0 and total_tokens >= max_tokens:
            break

    progress.close()

    if total_tokens == 0:
        return EvalResult(float("inf"), float("inf"), float("inf"), 0, total_bytes, docs_processed)

    mean_loss = total_loss / total_tokens
    return EvalResult(
        loss=mean_loss,
        perplexity=safe_exp(mean_loss),
        bpb=(total_loss / LN2) / total_bytes if total_bytes else float("inf"),
        num_tokens=total_tokens,
        num_bytes=total_bytes,
        num_docs=docs_processed,
    )


def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, device: str, trust_remote_code: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=trust_remote_code,
    )
    if device != "cuda":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Result reports
# ---------------------------------------------------------------------------


def report_summary(results: dict[str, EvalResult], args: argparse.Namespace) -> None:
    print("\n\n" + "=" * 90)
    print("SUMMARY - embedding initialization comparison")
    print("=" * 90)
    print(f"Data: {args.data_file}")
    print(f"max_length: {args.max_length} | stride: {args.stride}")
    print()

    header = f"{'Model':<55} {'Loss':>10} {'PPL':>12} {'BPB':>10} {'Tokens':>12} {'Docs':>8}"
    print(header)
    print("-" * len(header))

    comparable = {label: result.bpb for label, result in results.items() if result.finite and label != BASE_LABEL}
    best_label = min(comparable, key=comparable.get) if comparable else None

    for label, result in results.items():
        marker = "  <- best BPB" if label == best_label else ""
        print(
            f"{label:<55} {fmt(result.loss, '.4f'):>10} {fmt(result.perplexity, '.2f'):>12} "
            f"{fmt(result.bpb, '.4f'):>10} {result.num_tokens:>12,} "
            f"{result.num_docs:>8,}{marker}"
        )

    print("-" * len(header))
    if best_label:
        print(f"\nBest BPB: {best_label}")

    base_result = results.get(BASE_LABEL)
    if base_result is not None and base_result.finite:
        print(f"\nRegression check against the base model (BPB {base_result.bpb:.4f}):")
        for label, result in results.items():
            if label == BASE_LABEL:
                continue
            delta = result.bpb - base_result.bpb
            percent = (delta / base_result.bpb) * 100 if base_result.bpb > 0 else 0.0
            if percent < SPIKE_THRESHOLD_PCT:
                status = "OK"
            elif percent < REGRESSION_THRESHOLD_PCT:
                status = "SPIKE"
            else:
                status = "REGRESSION"
            print(f"  {label:<52} delta BPB: {delta:+.4f} ({percent:+.1f}%)  {status}")

    print("=" * 90)


def report_tsv(results: dict[str, EvalResult]) -> None:
    print("\n\n" + "=" * 90)
    print("RESULTS (TSV)")
    print("=" * 90)
    print("\t".join(["Model", "Loss", "PPL", "BPB", "Tokens", "Docs"]))
    for label, result in results.items():
        print(
            "\t".join(
                [
                    label,
                    fmt(result.loss, ".4f"),
                    fmt(result.perplexity, ".2f"),
                    fmt(result.bpb, ".4f"),
                    str(result.num_tokens),
                    str(result.num_docs),
                ]
            )
        )
    print("=" * 90)


def save_results(path: str, results: dict[str, EvalResult], args: argparse.Namespace) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_file": args.data_file,
        "max_docs": args.max_docs,
        "max_tokens": args.max_tokens,
        "max_length": args.max_length,
        "stride": args.stride,
        "dtype": args.dtype,
        "results": {label: asdict(result) for label, result in results.items()},
    }
    destination.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {destination}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]

    make_stream, data_source = make_text_stream(args)
    args.data_file = data_source  # normalize for reporting / save_results

    report_hardware(device, args.dtype)
    print(f"\nData source: {data_source}")
    if args.max_docs > 0:
        print(f"  max-docs: {args.max_docs:,}")
    if args.max_tokens > 0:
        print(f"  max-tokens: {args.max_tokens:,}")
        if args.max_docs <= 0:
            # Comparing >1 model under a token budget is invalid by construction:
            # each tokenizer reaches the budget after a different amount of text,
            # so the BPB denominators describe different corpora. Refuse rather
            # than emit numbers that look comparable and are not.
            # base_model is scored alongside --models, so one extended model plus
            # the base is already a two-tokenizer comparison.
            n_scored = len(args.models) + (1 if args.base_model else 0)
            if n_scored > 1 and not args.allow_token_cap_comparison:
                raise SystemExit(
                    "REFUSING to score multiple models under --max-tokens with no "
                    "--max-docs: a token budget stops each tokenizer after a "
                    f"DIFFERENT amount of source text, so the {n_scored} resulting "
                    "BPB values are not comparable. Set --max-docs "
                    "(tokenizer-independent), or --max-tokens -1 to score the whole "
                    "corpus, or run one model per job if you truly want a token cap. "
                    "Pass --allow-token-cap-comparison to override deliberately."
                )
            print(
                "  WARNING: a token budget stops each tokenizer after a "
                "DIFFERENT amount of source text, so this BPB is NOT comparable "
                "with any other tokenizer's. Use --max-docs (or score the whole "
                "corpus) for cross-tokenizer comparison."
            )
        elif len(args.models) + (1 if args.base_model else 0) > 1:
            # Both caps set: whichever binds first decides. If max_tokens binds, the
            # models still see different document counts, so say so up front.
            print(
                "  NOTE: both --max-tokens and --max-docs are set. If the token "
                "budget binds first, each tokenizer will have scored a different "
                "number of documents and the BPB values are again not comparable; "
                "the per-model 'documents' count below tells you which bound."
            )

    queue: list[tuple[str, str]] = []
    if args.base_model:
        queue.append((BASE_LABEL, args.base_model))
    queue.extend(zip(unique_labels(args.models), args.models))

    results: dict[str, EvalResult] = {}
    for position, (label, model_path) in enumerate(queue, 1):
        print("\n" + "=" * 70)
        print(f"[{position}/{len(queue)}] Evaluating: {label}")
        print(f"  Path: {model_path}")
        print("=" * 70)

        started = time.time()
        print("  Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            model_path, dtype, device, getattr(args, "trust_remote_code", False)
        )
        print(f"  Loaded in {time.time() - started:.1f}s | vocab size: {len(tokenizer):,} | dtype: {dtype}")
        if getattr(model, "hf_device_map", None):
            print(f"  Sharded across: {sorted({str(v) for v in model.hf_device_map.values()})}")

        print(f"\n  Streaming documents from {data_source} ...")
        started = time.time()
        result = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            make_stream=make_stream,
            total=args.max_docs if args.max_docs > 0 else None,
            max_length=args.max_length,
            stride=args.stride,
            max_tokens=args.max_tokens,
            device=device,
            desc=f"  [{position}/{len(queue)}] {label}",
        )
        results[label] = result

        print(
            f"\n  Loss: {fmt(result.loss, '.4f')} | PPL: {fmt(result.perplexity, '.2f')} "
            f"| BPB: {fmt(result.bpb, '.4f')}"
        )
        print(
            f"  Tokens: {result.num_tokens:,} | Bytes: {result.num_bytes:,} "
            f"| Docs: {result.num_docs:,} | Time: {time.time() - started:.1f}s"
        )

        if device == "cuda":
            report_gpu_memory()

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    report_summary(results, args)
    report_tsv(results)
    if args.output_json:
        save_results(args.output_json, results, args)


if __name__ == "__main__":
    main()
