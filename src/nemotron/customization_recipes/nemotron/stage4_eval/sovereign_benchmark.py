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

"""Sovereign benchmark bridge: stage3 BYOB output -> NeMo Evaluator BYOB.

This is a TEMPLATE for defining NeMo Evaluator BYOB benchmarks from
stage3_byob MCQ output. Copy this file and customize the sections
marked with "CUSTOMIZE" comments for your specific sovereign benchmark.

The stage3 BYOB pipeline produces benchmark.jsonl with MCQ questions in
one of two formats:

  4-choice format:
    {"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
     "answer": "B", "metadata": {"topic": "...", "language": "hi"}}

  10-choice format (with distractor expansion):
    {"question": "...", "options": {"A": "...", ..., "J": "..."},
     "answer": "D", "metadata": {"topic": "...", "language": "hi"}}

Usage:
  # Compile and install the benchmark
  nemo-evaluator-byob sovereign_benchmark.py

  # Compile and create a Docker image for the evaluator container
  nemo-evaluator-byob sovereign_benchmark.py --containerize

  # Run evaluation
  nemo-evaluator run_eval \\
    --eval_type byob_sovereign_mcq.sovereign-mcq \\
    --model_url <MODEL_URL> --model_id <MODEL_ID> \\
    --model_type chat \\
    --output_dir ./results/sovereign \\
    --api_key_name API_KEY

Environment variables (optional overrides):
  SOVEREIGN_BENCHMARK_NAME  - Override benchmark name (default: sovereign-mcq)
  SOVEREIGN_DATASET_PATH    - Override dataset path
  SOVEREIGN_LANGUAGE        - Language label for metadata (default: en)
  SOVEREIGN_NUM_CHOICES     - Number of answer choices: 4 or 10 (default: auto-detect)
"""

from __future__ import annotations

import json
import os
import re

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

# ---------------------------------------------------------------------------
# CUSTOMIZE: Benchmark configuration
# ---------------------------------------------------------------------------

# Benchmark name -- used as the evaluation task identifier.
# Override via SOVEREIGN_BENCHMARK_NAME env var.
BENCHMARK_NAME = os.environ.get("SOVEREIGN_BENCHMARK_NAME", "sovereign-mcq")

# Path to the stage3 BYOB output (benchmark.jsonl).
# Override via SOVEREIGN_DATASET_PATH env var.
DATASET_PATH = os.environ.get(
    "SOVEREIGN_DATASET_PATH",
    # Default: relative path assuming standard pipeline directory layout.
    # Replace with absolute path to your benchmark.jsonl.
    os.path.join(os.path.dirname(__file__), "..", "stage3_byob", "output", "benchmark.jsonl"),
)

# Target language (used in prompt and metadata reporting).
# Override via SOVEREIGN_LANGUAGE env var.
LANGUAGE = os.environ.get("SOVEREIGN_LANGUAGE", "en")

# Number of answer choices. Set to "auto" to detect from the first record,
# or "4" / "10" to force a specific format.
# Override via SOVEREIGN_NUM_CHOICES env var.
NUM_CHOICES = os.environ.get("SOVEREIGN_NUM_CHOICES", "auto")

# ---------------------------------------------------------------------------
# CUSTOMIZE: Prompt template
# ---------------------------------------------------------------------------
# The prompt uses {field} placeholders that map to JSONL fields.
# The field_mapping below renames nested "options" fields to flat names.
#
# For 4-choice: placeholders a, b, c, d
# For 10-choice: placeholders a, b, c, d, e, f, g, h, i, j
#
# If your benchmark is in a non-English language, translate the
# instructional text below while keeping the {placeholders} intact.
# ---------------------------------------------------------------------------


def _detect_num_choices(dataset_path: str) -> int:
    """Read the first record from dataset to detect 4 vs 10 choice format."""
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                options = record.get("options", {})
                return len(options)
    except (FileNotFoundError, json.JSONDecodeError, StopIteration):
        pass
    return 4  # default


def _get_num_choices() -> int:
    """Resolve the number of answer choices."""
    if NUM_CHOICES == "auto":
        return _detect_num_choices(DATASET_PATH)
    return int(NUM_CHOICES)


_num_choices = _get_num_choices()
_choice_letters = [chr(ord("A") + i) for i in range(_num_choices)]
_choice_letters_str = "/".join(_choice_letters)

# Build the options block for the prompt
_options_block = "\n".join(
    f"{letter}) {{{letter.lower()}}}" for letter in _choice_letters
)

PROMPT_TEMPLATE = (
    f"The following is a multiple choice question.\n\n"
    f"{{question}}\n\n"
    f"{_options_block}\n\n"
    f"Answer with just the letter ({_choice_letters_str}):"
)

# ---------------------------------------------------------------------------
# CUSTOMIZE: Field mapping
# ---------------------------------------------------------------------------
# The stage3 BYOB output stores options as a nested dict:
#   {"options": {"A": "...", "B": "...", ...}}
#
# NeMo Evaluator BYOB expects flat fields in the prompt template.
# The field_mapping renames "options.A" -> "a", "options.B" -> "b", etc.
#
# Note: The BYOB framework flattens nested JSONL fields using dot notation
# before applying field_mapping. If your JSONL is already flat (e.g.,
# has top-level "choice_a", "choice_b" fields), adjust the mapping keys.
# ---------------------------------------------------------------------------

FIELD_MAPPING = {
    f"options.{letter}": letter.lower() for letter in _choice_letters
}

# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------


@benchmark(
    name=BENCHMARK_NAME,
    dataset=DATASET_PATH,
    prompt=PROMPT_TEMPLATE,
    target_field="answer",
    endpoint_type="chat",
    field_mapping=FIELD_MAPPING,
)
@scorer
def sovereign_mcq_scorer(sample: ScorerInput) -> dict:
    """Score a sovereign MCQ benchmark response.

    Extracts the predicted answer letter from the model response and
    compares it to the ground-truth answer letter from the dataset.

    Handles common response formats:
    - "A"
    - "A)"
    - "The answer is B"
    - "B. Because..."
    - "(C)"

    Returns:
        dict with keys:
        - correct (bool): Whether the predicted answer matches the target
        - parsed (bool): Whether a valid answer letter was extracted
        - correct_<topic> (bool): Per-topic accuracy (if metadata.topic exists)
    """
    response_clean = sample.response.strip()

    # Determine valid answer letters based on number of choices
    num_choices = len(FIELD_MAPPING)
    valid_letters = "".join(chr(ord("A") + i) for i in range(num_choices))
    valid_pattern = f"[{valid_letters}{valid_letters.lower()}]"

    predicted = ""

    # Strategy 1: First character is a valid choice letter
    if response_clean and response_clean[0].upper() in valid_letters:
        predicted = response_clean[0].upper()
    else:
        # Strategy 2: "answer is X" or parenthesized letter
        match = re.search(
            rf"(?:answer\s+is\s+|^\s*\(?)\s*({valid_pattern})\b",
            response_clean,
            re.IGNORECASE,
        )
        if match:
            predicted = match.group(1).upper()
        else:
            # Strategy 3: Any standalone valid letter in first 50 chars
            match = re.search(rf"\b({valid_pattern})\b", response_clean[:50])
            if match:
                predicted = match.group(1).upper()

    target_letter = sample.target.strip().upper()
    is_correct = predicted == target_letter
    is_parsed = bool(predicted)

    scores: dict = {
        "correct": is_correct,
        "parsed": is_parsed,
    }

    # Per-topic breakdown (if metadata.topic is present in the JSONL)
    topic = sample.metadata.get("topic") or sample.metadata.get("metadata", {}).get("topic")
    if topic:
        # Sanitize topic name for use as metric key
        topic_key = re.sub(r"[^a-zA-Z0-9_]", "_", str(topic))
        scores[f"correct_{topic_key}"] = is_correct

    # Per-language breakdown (if metadata.language is present)
    lang = sample.metadata.get("language") or sample.metadata.get("metadata", {}).get("language")
    if lang:
        scores[f"correct_{lang}"] = is_correct

    return scores
