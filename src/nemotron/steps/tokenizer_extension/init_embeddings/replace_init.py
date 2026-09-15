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

"""Embedding init for the REPLACE arm — survivor remap + a choice of fresh-token init.

The replace tokenizer prunes the base residual Devanagari tokens and DENSELY
re-indexes the survivors, so base embedding rows no longer line up with the new
ids. This engine rebuilds the resized matrix explicitly:

  * survivors   -> copied EXACTLY from the base model via id_remap.json
                   (old_id -> new_id), preserving pretrained knowledge;
  * fresh tokens (the spliced Devanagari) -> initialized by `--method`:
        subword    mean-of-constituents (length-weighted mean of base subword rows)
        mean_all   the global mean of every base embedding
        hf_default HuggingFace's multivariate-normal (fit mean+cov of base rows)
        focus      FOCUS: sparsemax blend of the nearest base tokens in fastText space
        bert       MuRIL/BERT-weighted mean of the base subword rows
  * then the vocab is padded to a TP-divisible multiple.

The survivor/padding/validation scaffolding is method-independent; only the
fresh-token step changes. focus/bert reuse focus_init/subword_init helpers so the
math matches the ADD arm exactly. Mirrors the init engines' argparse main(argv).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from subword_init import (
    SEMANTIC_METHODS,
    build_bert_semantics,
    collect_semantic_inputs,
    decompose_new_tokens,
    find_target_script_tokens,
    length_based_weights,
    resolve_target_norm,
    subword_weights,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LENGTH_METHODS = ("uniform", "char_weighted", "max_char")
# bert_weighted is available on either side; gemma_weighted has no engine here.
AVERAGING_METHODS = LENGTH_METHODS + ("bert_weighted",)
METHODS = ("subword", "mean_all", "hf_default", "focus", "bert")
RULE = "=" * 60


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", required=True)
    p.add_argument("--extended-tokenizer", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute custom modeling code shipped in the model repo. "
        "Off by default; required by architectures whose code lives there.",
    )
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument(
        "--id-remap", default=None, help="id_remap.json (old_id->new_id). Default: <extended-tokenizer>/id_remap.json"
    )
    p.add_argument(
        "--language",
        default=None,
        help="Target language (see languages.py). Sets the target script used to "
        "find the base model's existing rows, and the defaults for "
        "--bert-model and --fasttext-url. Omit for the legacy Hindi behaviour.",
    )
    p.add_argument(
        "--method",
        choices=METHODS,
        default="subword",
        help="Fresh-token init: subword | mean_all | hf_default | focus | bert",
    )
    # subword (length-based) options
    p.add_argument("--input-averaging", choices=AVERAGING_METHODS, default="uniform")
    p.add_argument("--output-averaging", choices=AVERAGING_METHODS, default="uniform")
    p.add_argument("--input-norm-correction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-norm-correction", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--input-hindi-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-hindi-norm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument(
        "--pad-vocab-to",
        type=int,
        default=4,
        help="Pad the final row count up to a multiple of this for TP divisibility "
        "(Megatron VocabParallelEmbedding). Padding rows are never indexed. 0 disables.",
    )
    # bert (semantic) options
    p.add_argument(
        "--bert-model",
        default=None,
        help="Auxiliary encoder for --method bert. Default: from --language, "
        "else google/muril-base-cased (Indic-only; wrong for non-Indic targets).",
    )
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--bert-batch-size", type=int, default=128)
    p.add_argument("--gemma-model", default=None)  # parity; unused
    p.add_argument("--semantic-device", default="cuda" if torch.cuda.is_available() else "cpu")
    # focus options
    p.add_argument(
        "--fasttext-model", default=None, help="fastText .bin (e.g. cc.hi.300.bin); required for --method focus"
    )
    p.add_argument(
        "--fasttext-url",
        default=None,
        help="Fetch fastText here (on the compute node) if --fasttext-model is missing (.gz auto-decompressed).",
    )
    p.add_argument(
        "--candidate-pool",
        choices=("hindi", "target", "all"),
        default="target",
        help="FOCUS candidates: the base model's target-language "
        "rows, or all rows. 'hindi' is the old name for 'target'.",
    )
    p.add_argument("--sparsemax-temperature", type=float, default=0.05)
    args = p.parse_args(argv)

    # Resolve language-dependent defaults before validation. Without --language
    # this leaves the historical Hindi values in place.
    from languages import fasttext_url as _ft_url
    from languages import profile as _lang_profile
    from script_ranges import set_target_script

    if args.language:
        prof = _lang_profile(args.language)
        set_target_script(prof.script)
        if args.bert_model is None:
            args.bert_model = prof.encoder
        if args.fasttext_url is None:
            args.fasttext_url = _ft_url(args.language)
    if args.bert_model is None:
        args.bert_model = "google/muril-base-cased"
    if args.fasttext_url is None:
        args.fasttext_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz"

    if args.method == "subword":
        for side in ("input_averaging", "output_averaging"):
            if getattr(args, side) not in LENGTH_METHODS:
                p.error(
                    f"--method subword supports only {LENGTH_METHODS}; use "
                    f"--method bert when either side is bert_weighted"
                )
    if args.method == "focus" and not args.fasttext_model:
        if not args.language:
            p.error(
                "--method focus requires --fasttext-model, or --language so the "
                "cc.<code>.300 vectors can be resolved and fetched"
            )
        # ensure_fasttext() downloads --fasttext-url here if the file is absent.
        # The path comes from the shared resolver so the focus path finds the
        # same file instead of fetching its own copy.
        from focus_init import fasttext_cache_path

        args.fasttext_model = str(fasttext_cache_path(args.fasttext_url))
    return args


# --------------------------------------------------------------------------- #
# Fresh-token initializers. Each fills in_emb[fresh_ids] / out_emb[fresh_ids]
# from the BASE snapshot (base_in/base_out); survivors are already placed.
# --------------------------------------------------------------------------- #
def _norm_correct(vec, do, target):
    if do:
        n = vec.norm()
        if n > 1e-8:
            return vec * (target / n)
    return vec


def _init_mean_all(fresh_ids, in_emb, out_emb, mean_in, mean_out):
    with torch.no_grad():
        for tid in tqdm(fresh_ids, desc="Init fresh (mean_all)"):
            in_emb[tid] = mean_in
            out_emb[tid] = mean_out
    return {"mean_all": len(fresh_ids)}


def _multivariate_samples(base_emb, n):
    """HuggingFace-style multivariate-normal init: fit mean+cov of base rows, sample n."""
    x = base_emb.float()
    mu = x.mean(dim=0)
    centered = x - mu
    cov = (centered.T @ centered) / x.shape[0]
    d = cov.shape[0]
    try:
        cov = 1e-5 * cov + 1e-9 * torch.eye(d, device=cov.device, dtype=cov.dtype)
        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        return dist.sample((n,))
    except Exception:  # fall back to a diagonal approx if Cholesky is unstable
        std = (1e-5 * centered.pow(2).mean(dim=0)).clamp_min(1e-12).sqrt()
        return mu + torch.randn(n, d, device=mu.device, dtype=mu.dtype) * std


def _init_hf_default(fresh_ids, in_emb, out_emb, base_in, base_out):
    with torch.no_grad():
        si = _multivariate_samples(base_in, len(fresh_ids)).to(in_emb.dtype)
        so = _multivariate_samples(base_out, len(fresh_ids)).to(out_emb.dtype)
        for k, tid in enumerate(tqdm(fresh_ids, desc="Init fresh (hf_default mvn)")):
            in_emb[tid] = si[k]
            out_emb[tid] = so[k]
    return {"hf_default": len(fresh_ids)}


def _init_subword(
    fresh_ids, decomp, in_emb, out_emb, base_in, base_out, mean_in, mean_out, base_tok, args, target_in, target_out
):
    stats = {"copy": 0, "mean": 0, "multi": 0}
    with torch.no_grad():
        for tid in tqdm(fresh_ids, desc="Init fresh (mean-of-constituents)"):
            plan = decomp.plans[tid]
            if plan.category == "mean":
                in_emb[tid] = mean_in
                out_emb[tid] = mean_out
                stats["mean"] += 1
                continue
            if plan.category == "copy":
                sid = plan.subword_ids[0]
                in_emb[tid] = base_in[sid]
                out_emb[tid] = base_out[sid]
                stats["copy"] += 1
                continue
            w_in = length_based_weights(plan.subword_ids, args.input_averaging, base_tok, in_emb.dtype, in_emb.device)
            w_out = length_based_weights(
                plan.subword_ids, args.output_averaging, base_tok, out_emb.dtype, out_emb.device
            )
            avg_in = (base_in[plan.subword_ids] * w_in.unsqueeze(1)).sum(dim=0)
            avg_out = (base_out[plan.subword_ids] * w_out.unsqueeze(1)).sum(dim=0)
            in_emb[tid] = _norm_correct(avg_in, args.input_norm_correction, target_in)
            out_emb[tid] = _norm_correct(avg_out, args.output_norm_correction, target_out)
            stats["multi"] += 1
    return stats


def _init_bert(
    fresh_ids,
    decomp,
    in_emb,
    out_emb,
    base_in,
    base_out,
    mean_in,
    mean_out,
    ext_tok,
    base_tok,
    args,
    target_in,
    target_out,
):
    """Encoder-weighted mean of base subword rows (reuses subword_init semantics)."""
    needed = {m for m in (args.input_averaging, args.output_averaging) if m in SEMANTIC_METHODS}
    semantic = {}
    if needed:
        inputs = collect_semantic_inputs(ext_tok, base_tok, decomp)
        semantic = {m: build_bert_semantics(args, inputs) for m in needed}
    stats = {"copy": 0, "mean": 0, "multi": 0}
    with torch.no_grad():
        for tid in tqdm(fresh_ids, desc="Init fresh (bert)"):
            plan = decomp.plans[tid]
            if plan.category == "mean":
                in_emb[tid] = mean_in
                out_emb[tid] = mean_out
                stats["mean"] += 1
                continue
            if plan.category == "copy":
                sid = plan.subword_ids[0]
                in_emb[tid] = base_in[sid]
                out_emb[tid] = base_out[sid]
                stats["copy"] += 1
                continue

            # Input and output are weighted INDEPENDENTLY, matching the Add arm
            # (subword_init) and _init_subword above. Reusing w_in for the output
            # matrix silently ignored --output-averaging.
            def _weights(method, dtype, device):
                if method in SEMANTIC_METHODS:
                    w, _ = subword_weights(
                        method, tid, plan.subword_ids, base_tok, semantic, args.temperature, dtype, device
                    )
                    return w
                return length_based_weights(plan.subword_ids, method, base_tok, dtype, device)

            w_in = _weights(args.input_averaging, in_emb.dtype, in_emb.device)
            w_out = _weights(args.output_averaging, out_emb.dtype, out_emb.device)
            avg_in = (base_in[plan.subword_ids] * w_in.unsqueeze(1)).sum(dim=0)
            avg_out = (base_out[plan.subword_ids] * w_out.unsqueeze(1)).sum(dim=0)
            in_emb[tid] = _norm_correct(avg_in, args.input_norm_correction, target_in)
            out_emb[tid] = _norm_correct(avg_out, args.output_norm_correction, target_out)
            stats["multi"] += 1
    return stats


def _init_focus(fresh_ids, in_emb, out_emb, base_in, base_out, mean_in, mean_out, ext_tok, base_tok, base_vocab, args):
    """FOCUS: sparsemax blend of nearest base tokens in fastText space."""
    import fasttext
    import numpy as np
    from focus_init import (
        build_candidate_pool,
        decode_new_tokens,
        decode_vocabulary,
        ensure_fasttext,
        fasttext_vectors,
        sparsemax,
    )

    ensure_fasttext(args.fasttext_model, getattr(args, "fasttext_url", None))
    ft = fasttext.load_model(args.fasttext_model)
    base_texts = decode_vocabulary(base_tok, base_vocab, "Decoding base vocab")
    pool = build_candidate_pool(args.candidate_pool, base_texts, ft)  # base candidate rows + ft unit vecs
    new_texts = decode_new_tokens(ext_tok, fresh_ids)
    new_vecs = fasttext_vectors(new_texts, ft)
    del ft
    stats = {"focus": 0, "mean": 0}
    with torch.no_grad():
        for k, tid in enumerate(tqdm(fresh_ids, desc="Init fresh (FOCUS)")):
            q = new_vecs[k]
            q = q / (np.linalg.norm(q) + 1e-8)
            sims = pool.unit_vectors @ q
            w = sparsemax(sims / args.sparsemax_temperature)
            sel = np.flatnonzero(w)
            if sel.size == 0:
                in_emb[tid] = mean_in
                out_emb[tid] = mean_out
                stats["mean"] += 1
                continue
            wt = torch.tensor(w[sel], dtype=in_emb.dtype, device=in_emb.device).unsqueeze(1)
            ids = pool.token_ids[sel].tolist()
            in_emb[tid] = (base_in[ids] * wt).sum(dim=0)
            out_emb[tid] = (base_out[ids] * wt).sum(dim=0)
            stats["focus"] += 1
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    dtype = DTYPES[args.dtype]

    print(RULE)
    print(f"REPLACE embedding init (survivor remap + method={args.method})")
    print(RULE)
    print(f"  Base model:         {args.base_model}")
    print(f"  Extended tokenizer: {args.extended_tokenizer}")

    remap_path = Path(args.id_remap) if args.id_remap else Path(args.extended_tokenizer) / "id_remap.json"
    if not remap_path.exists():
        raise SystemExit(f"id_remap.json not found at {remap_path}; a replace tokenizer must ship it.")
    remap = {int(k): int(v) for k, v in json.loads(remap_path.read_text()).items()}
    print(f"  id_remap survivors: {len(remap):,} (from {remap_path})")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, trust_remote_code=args.trust_remote_code
    )
    base_tok = AutoTokenizer.from_pretrained(args.base_model, fix_mistral_regex=True)
    ext_tok = AutoTokenizer.from_pretrained(args.extended_tokenizer, fix_mistral_regex=True)

    base_vocab = model.get_input_embeddings().weight.shape[0]
    new_vocab = len(ext_tok)
    print(
        f"  Base vocab: {base_vocab:,} | new vocab: {new_vocab:,} | survivors: {len(remap):,} | "
        f"fresh: {new_vocab - len(remap):,}"
    )
    if max(remap.values()) >= new_vocab or max(remap.keys()) >= base_vocab:
        raise SystemExit("id_remap indices out of range for the given model/tokenizer.")

    with torch.no_grad():
        base_in = model.get_input_embeddings().weight[:base_vocab].detach().clone()
        base_out = model.get_output_embeddings().weight[:base_vocab].detach().clone()
        mean_in = base_in.mean(dim=0)
        mean_out = base_out.mean(dim=0)

    print("\nLocating base target-language rows for norm correction...")
    hindi_ids = [tid for tid, _ in find_target_script_tokens(base_tok, base_vocab)]
    with torch.no_grad():
        h_in = base_in[hindi_ids].norm(dim=1) if hindi_ids else None
        h_out = base_out[hindi_ids].norm(dim=1) if hindi_ids else None
        target_in, src_in = resolve_target_norm(
            base_in.norm(dim=1), h_in, args.input_hindi_norm, args.input_norm_correction, "input"
        )
        target_out, src_out = resolve_target_norm(
            base_out.norm(dim=1), h_out, args.output_hindi_norm, args.output_norm_correction, "output"
        )
    print(f"  Target-language base tokens: {len(hindi_ids):,} | target input norm ({src_in}): {target_in.item():.4f}")

    print("\nResizing (mean_resizing=False; every row is written explicitly)...")
    model.resize_token_embeddings(new_vocab, mean_resizing=False)
    in_emb = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings().weight

    # 1) Survivors: exact copy old_id -> new_id.
    survivor_new = set()
    with torch.no_grad():
        for old_id, new_id in tqdm(remap.items(), desc="Copying survivor rows"):
            in_emb[new_id] = base_in[old_id]
            out_emb[new_id] = base_out[old_id]
            survivor_new.add(new_id)

    fresh_ids = [i for i in range(new_vocab) if i not in survivor_new]
    print(f"\nFresh tokens to initialize: {len(fresh_ids):,} | method={args.method}")

    # 2) Fresh tokens: dispatch on method.
    if args.method == "mean_all":
        stats = _init_mean_all(fresh_ids, in_emb, out_emb, mean_in, mean_out)
    elif args.method == "hf_default":
        stats = _init_hf_default(fresh_ids, in_emb, out_emb, base_in, base_out)
    elif args.method == "focus":
        stats = _init_focus(
            fresh_ids, in_emb, out_emb, base_in, base_out, mean_in, mean_out, ext_tok, base_tok, base_vocab, args
        )
    else:  # subword or bert -> need the base-subword decomposition
        decomp = decompose_new_tokens(fresh_ids, ext_tok, base_tok, base_vocab)
        if args.method == "bert":
            stats = _init_bert(
                fresh_ids,
                decomp,
                in_emb,
                out_emb,
                base_in,
                base_out,
                mean_in,
                mean_out,
                ext_tok,
                base_tok,
                args,
                target_in,
                target_out,
            )
        else:
            stats = _init_subword(
                fresh_ids,
                decomp,
                in_emb,
                out_emb,
                base_in,
                base_out,
                mean_in,
                mean_out,
                base_tok,
                args,
                target_in,
                target_out,
            )

    # Validation: no NaN, survivor rows byte-identical to base.
    with torch.no_grad():
        assert not torch.isnan(in_emb).any() and not torch.isnan(out_emb).any(), "NaN in new embeddings"
        zero_in = int((in_emb.abs().max(dim=1)[0] < 1e-8).sum().item())
        sample_old, sample_new = next(iter(remap.items()))
        surv_ok = torch.equal(in_emb[sample_new], base_in[sample_old])
    print(f"\nInit stats ({args.method}) — survivors: {len(remap):,} | fresh: {stats}")
    print(f"Validation — NaN: none | all-zero input rows: {zero_in} | survivor-copy exact: {surv_ok}")
    if not surv_ok:
        raise SystemExit("survivor row copy mismatch; aborting rather than saving a corrupt checkpoint.")

    # Pad to a TP-divisible multiple (replace vocab is arbitrary/often odd).
    if args.pad_vocab_to and (new_vocab % args.pad_vocab_to != 0):
        padded = ((new_vocab + args.pad_vocab_to - 1) // args.pad_vocab_to) * args.pad_vocab_to
        print(f"\nPadding vocab {new_vocab:,} -> {padded:,} (multiple of {args.pad_vocab_to}) for TP divisibility")
        model.resize_token_embeddings(padded, mean_resizing=False)
        with torch.no_grad():
            model.get_input_embeddings().weight[new_vocab:] = mean_in
            model.get_output_embeddings().weight[new_vocab:] = mean_out

    print(f"\nSaving resized checkpoint -> {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    ext_tok.save_pretrained(args.output_dir)
    print(RULE)
    print(f"DONE in {time.time() - t0:.1f}s -> {args.output_dir}")
    print(RULE)


if __name__ == "__main__":
    main()
