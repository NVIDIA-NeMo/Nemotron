#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Convert MMPR-Tiny parquet+images to NeMo Gym JSONL format.

Images are stored as local file paths; nemo-rl's encode_images_in_examples()
converts them to base64 data-URLs automatically at training time.

Usage
-----
    # All questions (default)
    python convert_mmpr_tiny_to_gym_jsonl.py \\
        --cache-dir /shared/data/mmpr_tiny \\
        --output-dir /shared/data/mmpr_tiny_gym

    # MCQ only — single-letter answers, exact-match grading, no model call
    python convert_mmpr_tiny_to_gym_jsonl.py \\
        --cache-dir /shared/data/mmpr_tiny \\
        --output-dir /shared/data/mmpr_tiny_gym \\
        --question-type mcqa

    # MCQ + numeric — both rule-based agents, no LLM judge required (recommended)
    python convert_mmpr_tiny_to_gym_jsonl.py \\
        --cache-dir /shared/data/mmpr_tiny \\
        --output-dir /shared/data/mmpr_tiny_gym \\
        --question-type mcqa_and_numeric

    # Numeric only — math_verify library grading, no LLM judge
    python convert_mmpr_tiny_to_gym_jsonl.py \\
        --cache-dir /shared/data/mmpr_tiny \\
        --output-dir /shared/data/mmpr_tiny_gym \\
        --question-type numeric

    # Free-form only — requires equivalence_llm_judge for grading
    python convert_mmpr_tiny_to_gym_jsonl.py \\
        --cache-dir /shared/data/mmpr_tiny \\
        --output-dir /shared/data/mmpr_tiny_gym \\
        --question-type free_form

Output
------
    --question-type all             → mmpr_tiny_train.jsonl, mmpr_tiny_val.jsonl
    --question-type mcqa            → mmpr_tiny_train_mcqa.jsonl, mmpr_tiny_val_mcqa.jsonl
    --question-type numeric         → mmpr_tiny_train_numeric.jsonl, mmpr_tiny_val_numeric.jsonl
    --question-type mcqa_and_numeric→ mmpr_tiny_train_mcqa_and_numeric.jsonl, mmpr_tiny_val_mcqa_and_numeric.jsonl
    --question-type free_form       → mmpr_tiny_train_free_form.jsonl, mmpr_tiny_val_free_form.jsonl

Agent routing per record:
    mcqa_simple_agent              → single-letter (A–G) answers with visible letter choices
    math_with_judge_simple_agent   → numeric answers (int, decimal, simple fraction)
    equivalence_llm_judge_simple_agent → everything else (expressions, free text)

Each line is a NeMo Gym example with:
    agent_ref                  -> routing key (see above)
    responses_create_params    -> VLM prompt with image + text content
    expected_answer            -> ground-truth answer for the verifier
    question                   -> plain-text question (used by the math/LLM judge)
    dataset                    -> "mmpr_tiny" (for bookkeeping)
"""

import argparse
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Agent routing helpers (mirrors scripts/convert_to_gym_format.py)
# ---------------------------------------------------------------------------
LETTER_OPTION_RE = re.compile(r"(?:^|\n)\s*(?:[A-G][.\):]|\([A-G]\))\s+")
OPTION_EXTRACT_PATTERNS = [
    re.compile(r"(?:^|\n)\s*([A-G])\.\s*(.+?)(?=\n\s*[A-G][.\):]|\n\s*\([A-G]\)|\Z)", re.DOTALL),
    re.compile(r"(?:^|\n)\s*([A-G])\)\s*(.+?)(?=\n\s*[A-G][.\):]|\n\s*\([A-G]\)|\Z)", re.DOTALL),
    re.compile(r"(?:^|\n)\s*\(([A-G])\)\s*(.+?)(?=\n\s*\([A-G]\)|\n\s*[A-G][.\):]|\Z)", re.DOTALL),
    re.compile(r"(?:^|\n)\s*([A-G]):\s*(.+?)(?=\n\s*[A-G][.\):]|\n\s*\([A-G]\)|\Z)", re.DOTALL),
]
GENERIC_OPTIONS = [
    {"A": "Option A"},
    {"B": "Option B"},
    {"C": "Option C"},
    {"D": "Option D"},
]


def _extract_letter_options(question_text: str) -> list[dict] | None:
    for pat in OPTION_EXTRACT_PATTERNS:
        matches = pat.findall(question_text)
        if len(matches) >= 2:
            opts, seen = [], set()
            for letter, txt in matches:
                if letter not in seen:
                    opts.append({letter: txt.strip().split("\n")[0].strip()})
                    seen.add(letter)
            return opts
    return None


def _is_mcqa(question: str, answer: str) -> bool:
    answer_str = str(answer).strip()
    if not re.match(r"^[A-G]$", answer_str):
        return False
    return bool(LETTER_OPTION_RE.search(question))


# Matches integers, decimals, and simple fractions (positive or negative).
# These can be graded deterministically by math_verify without an LLM judge.
# Examples: "42", "-3", "0.6", "3/20", "-1/2"
_NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?(/\d+)?$")


def _is_numeric(answer: str) -> bool:
    """Return True for answers gradable by math_verify without an LLM judge."""
    return bool(_NUMERIC_RE.match(str(answer).strip()))


def _unify_answer_format(question: str) -> str:
    """Append a \boxed{} instruction if not already present."""
    if r"\boxed{" in question:
        return question
    return question + "\n\nPut your final answer (and only the final answer) inside \\boxed{}."


# ---------------------------------------------------------------------------
# MMPR-Tiny download / loading
# ---------------------------------------------------------------------------
def _ensure_downloaded(cache_dir: str) -> None:
    images_dir = os.path.join(cache_dir, "MMPR-Tiny", "images")
    parquet_path = os.path.join(cache_dir, "mmpr_tiny.parquet")
    ready_marker = os.path.join(cache_dir, ".mmpr_ready")
    if os.path.exists(ready_marker):
        return
    if os.path.exists(images_dir) and os.path.exists(parquet_path):
        Path(ready_marker).write_text("ready\n")
        return

    print(f"Downloading MMPR-Tiny to {cache_dir} ...")
    os.makedirs(cache_dir, exist_ok=True)

    zip_path = hf_hub_download("OpenGVLab/MMPR-Tiny", "images.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        tmp = os.path.join(cache_dir, "_tmp_extract")
        zf.extractall(tmp)
        shutil.move(os.path.join(tmp, "images"), images_dir)
        os.rmdir(tmp)

    pq = hf_hub_download("OpenGVLab/MMPR-Tiny", "mmpr_tiny.parquet", repo_type="dataset")
    shutil.copy(pq, parquet_path)

    Path(ready_marker).write_text("ready\n")
    print("Download complete.")


def _load_df(cache_dir: str) -> pd.DataFrame:
    _ensure_downloaded(cache_dir)
    df = pd.read_parquet(os.path.join(cache_dir, "mmpr_tiny.parquet"))

    # image_path: first image of each sample, as an absolute local path
    df["image_path"] = df["images"].str[0].apply(
        lambda x: os.path.join(cache_dir, x["path"]) if isinstance(x, dict) else ""
    )
    # question: user turn of the prompt field (contains "<image>" placeholder)
    df["question_text"] = df["prompt"].apply(
        lambda p: next((m["content"] for m in p if m.get("role") == "user"), "")
    )
    # answer: ground-truth from reward_model
    df["answer_text"] = df["reward_model"].apply(
        lambda r: r.get("ground_truth", "") if isinstance(r, dict) else ""
    )
    return df


# ---------------------------------------------------------------------------
# Per-row conversion
# ---------------------------------------------------------------------------
def _convert_row(image_path: str, question: str, answer: str, dataset_name: str) -> dict:
    question_clean = question.replace("<image>", "").strip()
    mcqa = _is_mcqa(question_clean, answer)

    content = [
        {
            "type": "input_image",
            # Local path is fine; encode_images_in_examples() will base64-encode
            # this at training time inside nemo_rl/environments/nemo_gym.py.
            "image_url": image_path,
            "detail": "auto",
        },
        {
            "type": "input_text",
            "text": _unify_answer_format(question_clean),
        },
    ]

    numeric = not mcqa and _is_numeric(answer)
    if mcqa:
        agent_name = "mcqa_simple_agent"
    elif numeric:
        agent_name = "math_with_judge_simple_agent"
    else:
        agent_name = "equivalence_llm_judge_simple_agent"

    record: dict = {
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
        "responses_create_params": {
            "input": [{"role": "user", "type": "message", "content": content}],
        },
        "expected_answer": str(answer),
        "question": question_clean,
        "dataset": dataset_name,
    }
    if mcqa:
        record["options"] = _extract_letter_options(question_clean) or GENERIC_OPTIONS
        record["grading_mode"] = "strict_single_letter_boxed"
    return record


# ---------------------------------------------------------------------------
# Question-type filter
# ---------------------------------------------------------------------------
# Maps --question-type value → set of allowed agent names (None = keep all).
_AGENT_FILTER: dict[str, set[str] | None] = {
    "all":             None,
    "mcqa":            {"mcqa_simple_agent"},
    "numeric":         {"math_with_judge_simple_agent"},
    "mcqa_and_numeric":{"mcqa_simple_agent", "math_with_judge_simple_agent"},
    "free_form":       {"equivalence_llm_judge_simple_agent"},
}

# Maps --question-type value → filename suffix inserted before ".jsonl".
_FILE_SUFFIX: dict[str, str] = {
    "all":             "",
    "mcqa":            "_mcqa",
    "numeric":         "_numeric",
    "mcqa_and_numeric":"_mcqa_and_numeric",
    "free_form":       "_free_form",
}


# ---------------------------------------------------------------------------
# Split writing
# ---------------------------------------------------------------------------
def _write_split(
    df: pd.DataFrame,
    output_path: str,
    dataset_name: str,
    agent_filter: set[str] | None = None,
) -> Counter:
    """Convert and write one split.

    Args:
        df: DataFrame slice (train or val).
        output_path: Destination JSONL file path.
        dataset_name: Value written to the ``dataset`` field of each record.
        agent_filter: If set, only records whose ``agent_ref.name`` is in this
            set are written. Pass ``None`` to keep all records.
    """
    counts: Counter = Counter()
    skipped = 0
    filtered = 0

    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            img = row["image_path"]
            q = str(row["question_text"])
            a = str(row["answer_text"])

            if not img or not os.path.exists(img) or not q.strip() or not a.strip():
                skipped += 1
                continue

            rec = _convert_row(img, q, a, dataset_name)

            if agent_filter is not None and rec["agent_ref"]["name"] not in agent_filter:
                filtered += 1
                continue

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[rec["agent_ref"]["name"]] += 1

    total = sum(counts.values())
    print(f"  Wrote {total} rows ({skipped} skipped, {filtered} filtered) → {output_path}")
    for agent, n in sorted(counts.items()):
        print(f"    {agent}: {n}")
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert MMPR-Tiny parquet+images to NeMo Gym JSONL"
    )
    ap.add_argument(
        "--cache-dir",
        required=True,
        help="Local directory containing MMPR-Tiny parquet + images "
             "(downloaded automatically if missing)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the output JSONL files",
    )
    ap.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Number of samples reserved for validation (capped at 10%% of total; default 500)",
    )
    ap.add_argument(
        "--question-type",
        choices=["all", "mcqa", "numeric", "mcqa_and_numeric", "free_form"],
        default="all",
        help=(
            "Which question types to include in the output. "
            "'mcqa' keeps only multiple-choice questions (single-letter answers + visible options); "
            "graded by mcqa_simple_agent (exact-match, no model call). "
            "'numeric' keeps only numeric answers (int/decimal/fraction); "
            "graded by math_with_judge_simple_agent (math_verify library, no LLM judge). "
            "'mcqa_and_numeric' keeps both rule-based subsets — recommended for judge-free training. "
            "'free_form' keeps only open-ended answers requiring equivalence_llm_judge. "
            "'all' (default) keeps all three. "
            "Output filenames are suffixed: _mcqa, _numeric, _mcqa_and_numeric, _free_form, or none."
        ),
    )
    args = ap.parse_args()

    agent_filter = _AGENT_FILTER[args.question_type]
    suffix = _FILE_SUFFIX[args.question_type]

    df = _load_df(args.cache_dir)
    total = len(df)
    val_size = min(args.val_size, total // 10)

    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, f"mmpr_tiny_train{suffix}.jsonl")
    val_path = os.path.join(args.output_dir, f"mmpr_tiny_val{suffix}.jsonl")

    print(f"Total samples: {total} → train: {len(df_train)}, val: {len(df_val)}")
    print(f"Question type filter: {args.question_type!r}")
    print("Converting train split ...")
    _write_split(df_train, train_path, "mmpr_tiny", agent_filter)
    print("Converting val split ...")
    _write_split(df_val, val_path, "mmpr_tiny", agent_filter)
    print("Done.")


if __name__ == "__main__":
    main()
