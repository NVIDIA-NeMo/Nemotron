# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tokenization and packing for SFT and CPT data preparation.

Creates Megatron bin/idx (CPT) and packed .npy (SFT) datasets from
local files or HuggingFace Hub, using Megatron-Bridge utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------

_AUTOMODEL = False
try:
    from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import (
        IndexedDatasetBuilder,
        DType,
    )

    _AUTOMODEL = True
except ImportError:
    pass

_MEGATRON_BRIDGE = False
try:
    from megatron.bridge.data.datasets.packing_utils import (
        create_hist,
        create_packing_strategy,
        fill_packing_strategy,
    )
    from megatron.bridge.data.datasets.utils import _chat_preprocess

    _MEGATRON_BRIDGE = True
except ImportError:
    pass

_HF_DATASETS = False
try:
    from datasets import load_dataset as _hf_load

    _HF_DATASETS = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CPTConfig:
    """Configuration for Continued Pre-Training data preparation."""

    output_dir: str = "data/cpt"
    input_path: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_subset: Optional[str] = None
    hf_split: str = "train"
    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    text_field: str = "text"
    num_shards: int = 1
    max_samples: Optional[int] = None
    train_ratio: float = 0.90
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    add_bos: bool = False
    add_eos: bool = True
    min_doc_chars: Optional[int] = None
    max_doc_tokens: Optional[int] = None
    seed: int = 42
    batch_size: int = 1000
    recursive: bool = True

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "CPTConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(CPTConfig)
        merged = OmegaConf.merge(schema, cfg)
        return CPTConfig(**OmegaConf.to_container(merged, resolve=True))


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning data preparation."""

    output_dir: str = "data/sft"
    input_path: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_subset: Optional[str] = None
    hf_split: str = "train"
    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    pack_size: int = 4096
    train_ratio: float = 0.9
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    messages_field: str = "messages"
    conversations_field: Optional[str] = None
    seed: int = 42
    add_generation_prompt: bool = False
    recursive: bool = True
    packing_algorithm: str = "first_fit_decreasing"
    max_samples: Optional[int] = None
    enable_thinking: bool = False
    truncate_history_thinking: bool = True
    thinking_start_token: str = "<think>"
    thinking_end_token: str = "</think>"

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "SFTConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(SFTConfig)
        merged = OmegaConf.merge(schema, cfg)
        return SFTConfig(**OmegaConf.to_container(merged, resolve=True))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    log.info("Loading tokenizer: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _find_files(path: Path, extensions: list[str], recursive: bool = True) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    if not path.is_dir():
        return []
    files: list[Path] = []
    for ext in extensions:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        files.extend(path.glob(pattern))
    return sorted(files)


def _load_text_data(path: Path, text_field: str, recursive: bool = True) -> Iterator[str]:
    if path.is_dir():
        for fp in _find_files(path, [".parquet", ".jsonl", ".txt"], recursive):
            yield from _load_text_data(fp, text_field, recursive=False)
        return

    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            for batch in pq.read_table(path, columns=[text_field]).to_batches():
                for val in batch.column(text_field):
                    if val is not None:
                        yield str(val)
        except ImportError:
            import pandas as pd

            for val in pd.read_parquet(path, columns=[text_field])[text_field]:
                if val is not None:
                    yield str(val)
    elif path.suffix.lower() == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    txt = rec.get(text_field)
                    if txt:
                        yield str(txt)
    elif path.suffix.lower() == ".txt":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def _load_jsonl(path: Path, recursive: bool = True) -> list[dict]:
    if path.is_dir():
        records: list[dict] = []
        for fp in _find_files(path, [".jsonl"], recursive):
            records.extend(_load_jsonl(fp, recursive=False))
        return records
    result: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def _load_hf_messages(
    name: str,
    subset: Optional[str],
    split: str,
    messages_field: str,
    conversations_field: Optional[str],
    max_samples: Optional[int],
) -> list[dict]:
    if not _HF_DATASETS:
        raise ImportError("datasets library required. pip install datasets")
    ds = _hf_load(name, subset, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    records: list[dict] = []
    for item in ds:
        rec: dict = {}
        if messages_field in item:
            rec["messages"] = item[messages_field]
        elif conversations_field and conversations_field in item:
            rec["messages"] = _convert_sharegpt(item[conversations_field])
        else:
            for field in ("messages", "conversation", "conversations", "dialog", "dialogue"):
                if field in item:
                    if field in ("conversation", "conversations", "dialog", "dialogue"):
                        rec["messages"] = _convert_sharegpt(item[field])
                    else:
                        rec["messages"] = item[field]
                    break
        if "tools" in item:
            rec["tools"] = item["tools"]
        if rec.get("messages"):
            records.append(rec)
    return records


def _convert_sharegpt(convos: list[dict]) -> list[dict]:
    msgs = []
    for turn in convos:
        role = turn.get("from", turn.get("role", "user")).lower()
        content = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        msgs.append({"role": role, "content": content})
    return msgs


# ---------------------------------------------------------------------------
# Tokenization and packing
# ---------------------------------------------------------------------------


def _replace_json_args(messages: list[dict]) -> list[dict]:
    import copy

    messages = copy.deepcopy(messages)
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict) and "function" in tc:
                    args = tc["function"].get("arguments")
                    if isinstance(args, str):
                        try:
                            tc["function"]["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            pass
    return messages


def _tokenize_conversation(messages, tokenizer, tools=None):
    if not _MEGATRON_BRIDGE:
        raise ImportError("Megatron-Bridge required for tokenization.")
    messages = _replace_json_args(messages)

    class _Wrapper:
        def __init__(self, hf):
            self._tokenizer = hf
            self.eos_id = hf.eos_token_id

    source = {"messages": messages}
    if tools:
        source["tools"] = tools
    result = _chat_preprocess(source, _Wrapper(tokenizer), tools)
    return result["input_ids"].tolist(), [int(x) for x in result["loss_mask"].tolist()]


def _pack_sequences(sequences, pack_size, pad_id, algorithm="first_fit_decreasing"):
    if not _MEGATRON_BRIDGE:
        raise ImportError("Megatron-Bridge required for packing.")
    if not sequences:
        return [], {"packing_efficiency": 0, "packing_factor": 0}
    dataset = np.array(sequences)
    by_len, hist = create_hist(dataset, pack_size)
    assignments, meta = create_packing_strategy(hist, pack_size, algorithm)
    output = fill_packing_strategy(assignments, by_len, pack_size, pad_id)
    log.info("Packing: efficiency=%.1f%%, factor=%.2f", meta["packing_efficiency"], meta["packing_factor"])
    return output, meta


# ---------------------------------------------------------------------------
# Thinking-aware SFT helpers
# ---------------------------------------------------------------------------


def _has_thinking(messages: list[dict]) -> bool:
    return any(m.get("role") == "assistant" and m.get("reasoning_content") for m in messages)


def _split_for_thinking(messages, truncate_history, start_tok, end_tok):
    import copy
    import re

    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_idxs:
        return []
    seqs = []
    for turn, uidx in enumerate(user_idxs):
        end = user_idxs[turn + 1] if turn + 1 < len(user_idxs) else len(messages)
        seq = copy.deepcopy(messages[:end])
        if truncate_history:
            last_asst = None
            for i in range(len(seq) - 1, -1, -1):
                if seq[i].get("role") == "assistant":
                    last_asst = i
                    break
            if last_asst is not None:
                for i in range(last_asst):
                    if seq[i].get("role") == "assistant":
                        seq[i] = copy.deepcopy(seq[i])
                        seq[i].pop("reasoning_content", None)
                        c = seq[i].get("content", "")
                        if c:
                            pat = re.escape(start_tok) + r".*?" + re.escape(end_tok)
                            seq[i]["content"] = re.sub(pat, "", c, flags=re.DOTALL)
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def prepare_cpt_data(cfg: CPTConfig) -> dict:
    """Prepare data for Continued Pre-Training (Megatron bin/idx format).

    Returns:
        Dict with ``blend_path``, ``output_dir``, and ``stats``.
    """
    import torch

    if not _AUTOMODEL:
        raise ImportError(
            "NeMo-AutoModel required for CPT. Use the correct container."
        )
    if cfg.hf_dataset and cfg.input_path:
        raise ValueError("Specify input_path or hf_dataset, not both.")
    if not cfg.hf_dataset and not cfg.input_path:
        raise ValueError("Must specify input_path or hf_dataset.")

    tokenizer = _load_tokenizer(cfg.tokenizer_model)
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    dtype = DType.optimal_dtype(len(tokenizer))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load texts
    if cfg.hf_dataset:
        if not _HF_DATASETS:
            raise ImportError("datasets library required. pip install datasets")
        texts = list(
            _hf_load(cfg.hf_dataset, cfg.hf_subset, split=cfg.hf_split)
        )
        texts = [t[cfg.text_field] for t in texts if t.get(cfg.text_field)]
        if cfg.max_samples:
            texts = texts[: cfg.max_samples]
        src = f"hf://{cfg.hf_dataset}" + (f"/{cfg.hf_subset}" if cfg.hf_subset else "")
    else:
        texts = list(_load_text_data(Path(cfg.input_path), cfg.text_field, cfg.recursive))
        if cfg.max_samples:
            texts = texts[: cfg.max_samples]
        src = cfg.input_path

    if cfg.min_doc_chars:
        texts = [t for t in texts if len(t) >= cfg.min_doc_chars]

    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]

    te = int(len(texts) * cfg.train_ratio)
    ve = te + int(len(texts) * cfg.valid_ratio)
    splits = {"train": texts[:te], "valid": texts[te:ve], "test": texts[ve:]}

    blend: dict[str, list] = {"train": [], "valid": [], "test": []}
    all_stats: dict = {}

    for name, stexts in splits.items():
        if not stexts:
            continue
        shard_id = f"{name}_shard_000000"
        builder = IndexedDatasetBuilder(str(out / f"{shard_id}.bin"), dtype=dtype)
        total, ndocs = 0, 0
        for i in range(0, len(stexts), cfg.batch_size):
            for text in stexts[i : i + cfg.batch_size]:
                toks = tokenizer.encode(text, add_special_tokens=False)
                if cfg.add_bos and bos_id is not None:
                    toks = [bos_id] + toks
                if cfg.add_eos and eos_id is not None:
                    toks = toks + [eos_id]
                if cfg.max_doc_tokens and len(toks) > cfg.max_doc_tokens:
                    toks = toks[: cfg.max_doc_tokens]
                if toks:
                    builder.add_document(torch.tensor(toks), [len(toks)])
                    total += len(toks)
                    ndocs += 1
        builder.finalize(str(out / f"{shard_id}.idx"))
        all_stats[name] = {"num_documents": ndocs, "total_tokens": total}
        blend[name] = [1.0, str(out / shard_id)]

    blend_path = out / "blend.json"
    with open(blend_path, "w") as f:
        json.dump(blend, f, indent=2)

    meta = {"format": "binidx", "input_source": src, "tokenizer": cfg.tokenizer_model, "splits": all_stats}
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("CPT prep complete: %s", out)
    return {"blend_path": str(blend_path), "output_dir": str(out), "stats": meta}


def prepare_sft_data(cfg: SFTConfig) -> dict:
    """Prepare data for Supervised Fine-Tuning (packed .npy format).

    Returns:
        Dict with training/validation/test paths and stats.
    """
    if not _MEGATRON_BRIDGE:
        raise ImportError("Megatron-Bridge required for SFT. Use the correct container.")
    if cfg.hf_dataset and cfg.input_path:
        raise ValueError("Specify input_path or hf_dataset, not both.")
    if not cfg.hf_dataset and not cfg.input_path:
        raise ValueError("Must specify input_path or hf_dataset.")

    tokenizer = _load_tokenizer(cfg.tokenizer_model)

    if cfg.hf_dataset:
        records = _load_hf_messages(
            cfg.hf_dataset, cfg.hf_subset, cfg.hf_split,
            cfg.messages_field, cfg.conversations_field, cfg.max_samples,
        )
        src = f"hf://{cfg.hf_dataset}"
    else:
        records = _load_jsonl(Path(cfg.input_path), cfg.recursive)
        if cfg.max_samples:
            records = records[: cfg.max_samples]
        src = cfg.input_path

    sequences: list[dict] = []
    thinking_count = 0

    for i, rec in enumerate(records):
        msgs = rec.get(cfg.messages_field, [])
        if not msgs:
            continue
        tools = rec.get("tools")
        try:
            if cfg.enable_thinking and _has_thinking(msgs):
                thinking_count += 1
                for seq_msgs in _split_for_thinking(
                    msgs, cfg.truncate_history_thinking,
                    cfg.thinking_start_token, cfg.thinking_end_token,
                ):
                    ids, mask = _tokenize_conversation(seq_msgs, tokenizer, tools)
                    sequences.append({"input_ids": ids, "loss_mask": mask})
            else:
                ids, mask = _tokenize_conversation(msgs, tokenizer, tools)
                sequences.append({"input_ids": ids, "loss_mask": mask})
        except Exception as exc:
            log.warning("Skipping record %d: %s", i, exc)

    log.info("Tokenized %d sequences from %d records", len(sequences), len(records))

    np.random.seed(cfg.seed)
    packed, pack_meta = _pack_sequences(
        sequences, cfg.pack_size, tokenizer.pad_token_id, cfg.packing_algorithm,
    )

    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(packed))
    rng.shuffle(idx)

    te = int(len(packed) * cfg.train_ratio)
    ve = te + int(len(packed) * cfg.valid_ratio)
    train = [packed[i] for i in idx[:te]]
    valid = [packed[i] for i in idx[te:ve]]
    test = [packed[i] for i in idx[ve:]]

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, split_data in [("training", train), ("validation", valid), ("test", test)]:
        p = out / f"{split_name}_{cfg.pack_size}.npy"
        np.save(p, split_data, allow_pickle=True)
        paths[f"{split_name}_path"] = str(p)
        log.info("Saved %d %s sequences -> %s", len(split_data), split_name, p)

    meta = {
        "format": "packed",
        "pack_size": cfg.pack_size,
        "tokenizer": cfg.tokenizer_model,
        "input_source": src,
        "total_records": len(records),
        "total_packed": len(packed),
    }
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("SFT prep complete: %s", out)
    return {**paths, "output_dir": str(out), "stats": meta}
