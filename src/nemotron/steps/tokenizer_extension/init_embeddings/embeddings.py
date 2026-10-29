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

"""Config-driven embedding initialization -> resized HF checkpoint for CPT.

A thin dispatcher over three self-contained initialization engines. Each is
kept intact so its norm-correction and validation logic stays verified:

  * baseline_init.py  (method: baseline)
        hf_default | mean_all | mean_target  [+ optional input norm-correction]
  * subword_init.py   (method: subword)
        decompose every new token into base subwords and average their rows;
        averaging = uniform | char_weighted | max_char | bert_weighted |
        gemma_weighted, chosen independently for the input and output side,
        with per-side (optionally target-script-only) norm correction.
  * focus_init.py     (method: focus)
        FOCUS (Dobler & de Melo 2023): sparsemax-weighted combination of the
        base tokens closest to each new token in a fastText auxiliary space.

Each engine exposes ``main(argv)`` (argparse). ``run_init(cfg)`` translates the
YAML config into that engine's argv and invokes it. The engines are imported
LAZILY inside each branch so that e.g. selecting `subword` never requires
`fasttext` (only `focus` does).

Domain note — ADD vs REPLACE:
  These engines assume an *append-style* extended tokenizer: rows [:base_vocab]
  are the untouched base embeddings and [base_vocab:] are the appended new
  tokens (`new_token_ids = range(base_vocab, new_vocab)`). That is exactly the
  ADD arm. The REPLACE arm densely re-indexes surviving ids via id_remap.json,
  which violates that assumption, so REPLACE needs a remap-aware wrapper (it
  must permute survivor rows before the append-style init runs). `run_init`
  refuses a replace tokenizer rather than silently producing a wrong checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

METHODS = ("baseline", "subword", "focus")


def _flag(name: str, value: bool) -> str:
    """argparse BooleanOptionalAction: True -> --name, False -> --no-name."""
    return f"--{name}" if value else f"--no-{name}"


def _pick(d: dict, new: str, old: str, default):
    """Prefer the language-neutral key, fall back to the historical Hindi one."""
    if new in d:
        return d[new]
    return d.get(old, default)


def _common_argv(cfg: dict) -> list[str]:
    argv = [
        "--base-model",
        str(cfg["base_model"]),
        "--extended-tokenizer",
        str(cfg["extended_tokenizer"]),
        "--output-dir",
        str(cfg["output_dir"]),
        "--dtype",
        str(cfg.get("dtype", "bfloat16")),
    ]
    # Every engine accepts --language; it selects the target script used to find
    # the base model's existing target-language rows, and supplies the auxiliary
    # encoder / fastText defaults. Omitted -> the historical Hindi behaviour.
    if cfg.get("language"):
        argv += ["--language", str(cfg["language"])]
    # Off by default: loading a repo whose modeling code ships with the weights
    # executes that code, so it is opt-in from config rather than implicit.
    if bool(cfg.get("trust_remote_code", False)):
        argv += ["--trust-remote-code"]
    return argv


def _baseline_argv(cfg: dict) -> list[str]:
    b = cfg.get("baseline", {}) or {}
    argv = _common_argv(cfg)
    argv += ["--mode", str(b.get("mode", "mean_target"))]
    argv += [_flag("norm-correction", bool(b.get("norm_correction", True)))]
    argv += ["--num-samples", str(int(b.get("num_samples", 10)))]
    return argv


def _subword_argv(cfg: dict) -> list[str]:
    s = cfg.get("subword", {}) or {}
    argv = _common_argv(cfg)
    argv += ["--input-averaging", str(s.get("input_averaging", "uniform"))]
    argv += ["--output-averaging", str(s.get("output_averaging", "uniform"))]
    argv += [_flag("input-norm-correction", bool(s.get("input_norm_correction", True)))]
    argv += [_flag("output-norm-correction", bool(s.get("output_norm_correction", False)))]
    argv += [_flag("input-hindi-norm", bool(_pick(s, "input_target_norm", "input_hindi_norm", True)))]
    argv += [_flag("output-hindi-norm", bool(_pick(s, "output_target_norm", "output_hindi_norm", False)))]
    argv += ["--temperature", str(float(s.get("temperature", 0.1)))]
    if s.get("bert_model"):
        argv += ["--bert-model", str(s["bert_model"])]
    argv += ["--gemma-model", str(s.get("gemma_model", "google/gemma-2-27b"))]
    argv += ["--num-samples", str(int(s.get("num_samples", 10)))]
    return argv


def _focus_argv(cfg: dict) -> list[str]:
    f = cfg.get("focus", {}) or {}
    if not f.get("fasttext_model") and not cfg.get("language"):
        raise ValueError(
            "method=focus needs focus.fasttext_model (a fastText .bin), "
            "or a top-level `language:` so the cc.<code>.300 vectors "
            "can be resolved and fetched automatically."
        )
    argv = _common_argv(cfg)
    if f.get("fasttext_model"):
        argv += ["--fasttext-model", str(f["fasttext_model"])]
    if f.get("fasttext_url"):
        argv += ["--fasttext-url", str(f["fasttext_url"])]
    argv += ["--candidate-pool", str(f.get("candidate_pool", "target"))]
    argv += ["--sparsemax-temperature", str(float(f.get("sparsemax_temperature", 0.05)))]
    argv += ["--num-samples", str(int(f.get("num_samples", 10)))]
    argv += ["--top-contributors", str(int(f.get("top_contributors", 5)))]
    return argv


def _replace_argv(cfg: dict) -> list[str]:
    """Map the add-style config (method + sub-blocks) to replace_init's --method dispatch.

    baseline(hf_default|mean_all) -> --method hf_default|mean_all
    focus                         -> --method focus (+ fastText args)
    subword(bert_weighted)        -> --method bert  (+ MuRIL args)
    subword(uniform|char|max)     -> --method subword (length-based)
    """
    method = str(cfg.get("method", "subword")).lower()
    argv = _common_argv(cfg)
    if method == "baseline":
        mode = str((cfg.get("baseline") or {}).get("mode", "mean_all"))
        if mode not in ("hf_default", "mean_all"):
            # replace_init has no target-script-mean engine, so mean_target
            # cannot be honoured here. Do NOT quietly substitute
            # mean_all: that would report a different init than was configured.
            extra = (
                " 'mean_target' averages the base model's target-script "
                "rows, which replace_init does not implement — use arm=add for that, "
                "or method=subword with input_averaging=uniform (mean of constituents)."
                if mode in ("mean_target", "mean_hindi")
                else ""
            )
            raise ValueError(f"arm=replace baseline supports mode hf_default|mean_all, got {mode!r}.{extra}")
        argv += ["--method", mode]
    elif method == "focus":
        f = cfg.get("focus") or {}
        if not f.get("fasttext_model") and not cfg.get("language"):
            raise ValueError(
                "arm=replace method=focus needs focus.fasttext_model, "
                "or a top-level `language:` to resolve cc.<code>.300"
            )
        argv += [
            "--method",
            "focus",
            "--candidate-pool",
            str(f.get("candidate_pool", "target")),
            "--sparsemax-temperature",
            str(float(f.get("sparsemax_temperature", 0.05))),
        ]
        if f.get("fasttext_model"):
            argv += ["--fasttext-model", str(f["fasttext_model"])]
        if f.get("fasttext_url"):
            argv += ["--fasttext-url", str(f["fasttext_url"])]
    else:  # subword family
        s = cfg.get("subword") or {}
        ia = str(s.get("input_averaging", "uniform"))
        if ia == "gemma_weighted":
            # replace_init declares --gemma-model for parity but never uses it, and
            # its METHODS has no gemma entry. Silently routing this to --method bert
            # ran a DIFFERENT encoder than the config asked for, so fail instead.
            raise ValueError(
                "arm=replace does not support subword.input_averaging=gemma_weighted "
                "(replace_init has no gemma path). Use bert_weighted, or a length-based "
                "averaging (uniform | char_weighted | max_char), or run arm=add."
            )
        oa = str(s.get("output_averaging", "uniform"))
        if oa == "gemma_weighted":
            raise ValueError(
                "arm=replace does not support subword.output_averaging=gemma_weighted "
                "(replace_init has no gemma path). Use bert_weighted, or a length-based "
                "averaging (uniform | char_weighted | max_char), or run arm=add."
            )
        # The bert engine is required whenever EITHER side is encoder-weighted; the
        # length-based engine rejects semantic methods. Both sides are always
        # forwarded so input and output are weighted independently, as in arm=add.
        if "bert_weighted" in (ia, oa):
            # Do NOT default --bert-model here. Passing it explicitly overrides the
            # language profile, which is how a Vietnamese run silently ended up
            # weighting with MuRIL (Indic-only, no Vietnamese). Omit it and let
            # --language resolve the encoder; only forward an explicit override.
            argv += ["--method", "bert", "--temperature", str(float(s.get("temperature", 0.1)))]
            if s.get("bert_model"):
                argv += ["--bert-model", str(s["bert_model"])]
        else:
            argv += ["--method", "subword"]
        argv += ["--input-averaging", ia, "--output-averaging", oa]
        argv += [_flag("input-norm-correction", bool(s.get("input_norm_correction", True)))]
        argv += [_flag("output-norm-correction", bool(s.get("output_norm_correction", False)))]
        argv += [_flag("input-hindi-norm", bool(_pick(s, "input_target_norm", "input_hindi_norm", True)))]
        argv += [_flag("output-hindi-norm", bool(_pick(s, "output_target_norm", "output_hindi_norm", False)))]
    return argv


def _guard_add(cfg: dict) -> None:
    """On the add (append-style) path, refuse a replace tokenizer (has id_remap.json)."""
    tok = Path(str(cfg["extended_tokenizer"]))
    if (tok / "id_remap.json").exists():
        raise ValueError(
            f"{tok} contains id_remap.json (a replace tokenizer) but arm=add. Append-style "
            "init would mis-place survivor rows. Set arm=replace for this tokenizer."
        )


def run_init(cfg: dict) -> None:
    method = str(cfg.get("method", "subword")).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    for key in ("base_model", "extended_tokenizer", "output_dir"):
        if not cfg.get(key):
            raise ValueError(f"config is missing required key: {key}")

    arm = str(cfg.get("arm", "add")).lower()
    if arm not in ("add", "replace"):
        raise ValueError(f"arm must be 'add' or 'replace', got {arm!r}")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # REPLACE: survivor rows are re-indexed (id_remap.json), so replace_init copies
    # them exactly and inits the fresh tokens by the chosen method (hf_default /
    # mean_all / focus / bert / subword), then pads the vocab for TP divisibility.
    if arm == "replace":
        argv = _replace_argv(cfg)
        log.info("MILESTONE: REPLACE init (survivor remap, method=%s) | argv=%s", method, " ".join(argv))
        import replace_init as engine

        engine.main(argv)
        log.info("MILESTONE: DONE — resized HF checkpoint -> %s", cfg["output_dir"])
        return

    # ADD (append-style): the init engines assume base rows are unchanged.
    _guard_add(cfg)
    if method == "baseline":
        argv = _baseline_argv(cfg)
        log.info("MILESTONE: baseline init | argv=%s", " ".join(argv))
        import baseline_init as engine
    elif method == "subword":
        argv = _subword_argv(cfg)
        log.info("MILESTONE: subword init | argv=%s", " ".join(argv))
        import subword_init as engine
    else:  # focus
        argv = _focus_argv(cfg)
        log.info("MILESTONE: FOCUS init | argv=%s", " ".join(argv))
        import focus_init as engine

    engine.main(argv)
    log.info("MILESTONE: DONE — resized HF checkpoint -> %s", cfg["output_dir"])
