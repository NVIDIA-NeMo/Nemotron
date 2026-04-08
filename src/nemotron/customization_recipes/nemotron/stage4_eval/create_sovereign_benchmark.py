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

"""Auto-generate a NeMo Evaluator BYOB benchmark from stage3 BYOB output.

Reads the stage3_byob benchmark.jsonl to understand its structure (number
of choices, subjects/topics, language), then generates a customized
sovereign_benchmark.py file ready for compilation with nemo-evaluator-byob.

Usage:
  python create_sovereign_benchmark.py \\
    --byob-output /path/to/stage3/benchmark.jsonl \\
    --benchmark-name "hindi-medical-mcq" \\
    --output-dir /path/to/eval/benchmarks/

  # With auto-compilation:
  python create_sovereign_benchmark.py \\
    --byob-output /path/to/stage3/benchmark.jsonl \\
    --benchmark-name "hindi-medical-mcq" \\
    --output-dir /path/to/eval/benchmarks/ \\
    --compile

  # With containerization:
  python create_sovereign_benchmark.py \\
    --byob-output /path/to/stage3/benchmark.jsonl \\
    --benchmark-name "hindi-medical-mcq" \\
    --output-dir /path/to/eval/benchmarks/ \\
    --containerize
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


def analyze_benchmark(jsonl_path: str) -> dict:
    """Read benchmark.jsonl and extract structural metadata.

    Args:
        jsonl_path: Path to the stage3 BYOB benchmark.jsonl file.

    Returns:
        Dict with keys:
        - num_records: Total number of MCQ records
        - num_choices: Number of answer choices (4 or 10)
        - choice_letters: List of choice letters (e.g., ["A","B","C","D"])
        - topics: Counter of topic -> count
        - languages: Counter of language -> count
        - answer_distribution: Counter of answer_letter -> count
        - sample_record: First record for reference
    """
    topics: Counter = Counter()
    languages: Counter = Counter()
    answers: Counter = Counter()
    num_choices = 0
    sample_record = None
    num_records = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            num_records += 1

            if sample_record is None:
                sample_record = record

            # Detect number of choices from options dict
            options = record.get("options", {})
            if len(options) > num_choices:
                num_choices = len(options)

            # Extract metadata
            metadata = record.get("metadata", {})
            topic = metadata.get("topic", record.get("topic", "unknown"))
            lang = metadata.get("language", record.get("language", "unknown"))
            answer = record.get("answer", "")

            topics[topic] += 1
            languages[lang] += 1
            answers[answer] += 1

    if num_choices == 0:
        num_choices = 4  # default

    choice_letters = [chr(ord("A") + i) for i in range(num_choices)]

    return {
        "num_records": num_records,
        "num_choices": num_choices,
        "choice_letters": choice_letters,
        "topics": topics,
        "languages": languages,
        "answer_distribution": answers,
        "sample_record": sample_record,
    }


def generate_benchmark_script(
    benchmark_name: str,
    dataset_path: str,
    analysis: dict,
    language: str | None = None,
) -> str:
    """Generate a sovereign_benchmark.py file customized for the dataset.

    Args:
        benchmark_name: Name for the benchmark (e.g., "hindi-medical-mcq").
        dataset_path: Absolute path to the benchmark.jsonl file.
        analysis: Output from analyze_benchmark().
        language: Override language label. If None, uses the most common
                  language detected in the dataset.

    Returns:
        Python source code for the benchmark definition file.
    """
    num_choices = analysis["num_choices"]
    choice_letters = analysis["choice_letters"]
    choice_letters_str = "/".join(choice_letters)

    # Determine language from analysis if not provided
    if language is None:
        if analysis["languages"]:
            language = analysis["languages"].most_common(1)[0][0]
        else:
            language = "en"

    # Build options block for prompt
    options_lines = "\n".join(
        f'        "{letter}) {{{letter.lower()}}}\\n"' for letter in choice_letters
    )

    # Build field mapping
    mapping_lines = []
    for letter in choice_letters:
        mapping_lines.append(f'    "options.{letter}": "{letter.lower()}"')
    field_mapping_str = ",\n".join(mapping_lines)

    # Build valid letters string
    valid_letters = "".join(choice_letters)

    # Topic summary for docstring
    top_topics = analysis["topics"].most_common(5)
    topics_doc = ", ".join(f"{t} ({c})" for t, c in top_topics)

    return f'''\
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

"""Sovereign benchmark: {benchmark_name}

Auto-generated from stage3 BYOB output by create_sovereign_benchmark.py.

Dataset: {dataset_path}
Records: {analysis["num_records"]}
Choices: {num_choices} ({choice_letters_str})
Language: {language}
Top topics: {topics_doc}

Usage:
  nemo-evaluator-byob {benchmark_name}_benchmark.py
  nemo-evaluator-byob {benchmark_name}_benchmark.py --containerize
"""

import re

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

BENCHMARK_NAME = "{benchmark_name}"
DATASET_PATH = "{dataset_path}"

PROMPT_TEMPLATE = (
    "The following is a multiple choice question.\\n\\n"
    "{{question}}\\n\\n"
{options_lines}
    "\\n"
    "Answer with just the letter ({choice_letters_str}):"
)

FIELD_MAPPING = {{
{field_mapping_str},
}}


@benchmark(
    name=BENCHMARK_NAME,
    dataset=DATASET_PATH,
    prompt=PROMPT_TEMPLATE,
    target_field="answer",
    endpoint_type="chat",
    field_mapping=FIELD_MAPPING,
)
@scorer
def {benchmark_name.replace("-", "_")}_scorer(sample: ScorerInput) -> dict:
    """Score {benchmark_name} MCQ response.

    Extracts the predicted answer letter from the model response and
    compares it to the ground-truth answer letter.
    """
    response_clean = sample.response.strip()
    valid_letters = "{valid_letters}"
    valid_pattern = f"[{{valid_letters}}{{valid_letters.lower()}}]"
    predicted = ""

    # Strategy 1: First character is a valid choice letter
    if response_clean and response_clean[0].upper() in valid_letters:
        predicted = response_clean[0].upper()
    else:
        # Strategy 2: "answer is X" or parenthesized letter
        match = re.search(
            rf"(?:answer\\s+is\\s+|^\\s*\\(?)\\s*({{valid_pattern}})\\b",
            response_clean,
            re.IGNORECASE,
        )
        if match:
            predicted = match.group(1).upper()
        else:
            # Strategy 3: Any standalone valid letter in first 50 chars
            match = re.search(rf"\\b({{valid_pattern}})\\b", response_clean[:50])
            if match:
                predicted = match.group(1).upper()

    target_letter = sample.target.strip().upper()
    is_correct = predicted == target_letter
    is_parsed = bool(predicted)

    scores = {{
        "correct": is_correct,
        "parsed": is_parsed,
    }}

    # Per-topic breakdown
    topic = sample.metadata.get("topic") or sample.metadata.get("metadata", {{}}).get("topic")
    if topic:
        topic_key = re.sub(r"[^a-zA-Z0-9_]", "_", str(topic))
        scores[f"correct_{{topic_key}}"] = is_correct

    return scores
'''


def main():
    parser = argparse.ArgumentParser(
        description="Generate a NeMo Evaluator BYOB benchmark from stage3 BYOB output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python create_sovereign_benchmark.py \\
    --byob-output /data/byob/benchmark.jsonl \\
    --benchmark-name hindi-medical-mcq \\
    --output-dir /data/eval/benchmarks/

  python create_sovereign_benchmark.py \\
    --byob-output /data/byob/benchmark.jsonl \\
    --benchmark-name hindi-medical-mcq \\
    --output-dir /data/eval/benchmarks/ \\
    --compile --containerize
""",
    )

    parser.add_argument(
        "--byob-output",
        required=True,
        help="Path to stage3 BYOB benchmark.jsonl file.",
    )
    parser.add_argument(
        "--benchmark-name",
        required=True,
        help='Name for the benchmark (e.g., "hindi-medical-mcq"). '
        "Used as the eval task identifier.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the generated benchmark definition.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Override language label. Auto-detected from data if omitted.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the benchmark with nemo-evaluator-byob after generation.",
    )
    parser.add_argument(
        "--containerize",
        action="store_true",
        help="Build a Docker image with the benchmark baked in (implies --compile).",
    )

    args = parser.parse_args()

    # Validate input
    byob_path = os.path.abspath(args.byob_output)
    if not os.path.isfile(byob_path):
        print(f"Error: BYOB output file not found: {byob_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Analyze the benchmark dataset
    print(f"Analyzing {byob_path}...")
    analysis = analyze_benchmark(byob_path)

    print(f"  Records: {analysis['num_records']}")
    print(f"  Choices: {analysis['num_choices']} ({'/'.join(analysis['choice_letters'])})")
    print(f"  Topics:  {len(analysis['topics'])} distinct")
    for topic, count in analysis["topics"].most_common(5):
        print(f"           - {topic}: {count}")
    print(f"  Languages: {dict(analysis['languages'])}")
    print(f"  Answer distribution: {dict(analysis['answer_distribution'])}")

    # Step 2: Generate the benchmark script
    safe_name = args.benchmark_name.replace(" ", "-").lower()
    script_filename = f"{safe_name.replace('-', '_')}_benchmark.py"
    script_path = os.path.join(output_dir, script_filename)

    print(f"\nGenerating benchmark definition: {script_path}")
    source = generate_benchmark_script(
        benchmark_name=safe_name,
        dataset_path=byob_path,
        analysis=analysis,
        language=args.language,
    )

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(source)
    print(f"  Written: {script_path}")

    # Step 3: Optionally compile
    if args.compile or args.containerize:
        byob_cli = shutil.which("nemo-evaluator-byob")
        if byob_cli is None:
            print(
                "\nWarning: nemo-evaluator-byob not found on PATH.",
                file=sys.stderr,
            )
            print(
                "Install nemo-evaluator to compile: pip install nemo-evaluator",
                file=sys.stderr,
            )
            print(f"\nTo compile manually:\n  nemo-evaluator-byob {script_path}")
        else:
            print(f"\nCompiling benchmark with: nemo-evaluator-byob {script_path}")
            compile_cmd = [byob_cli, script_path]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("  Compilation successful.")
                if result.stdout.strip():
                    print(f"  {result.stdout.strip()}")
            else:
                print(f"  Compilation failed:\n  {result.stderr}", file=sys.stderr)
                sys.exit(1)

    # Step 4: Optionally containerize
    if args.containerize:
        byob_cli = shutil.which("nemo-evaluator-byob")
        if byob_cli:
            print(f"\nContainerizing benchmark: nemo-evaluator-byob {script_path} --containerize")
            container_cmd = [byob_cli, script_path, "--containerize"]
            result = subprocess.run(container_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("  Containerization successful.")
                if result.stdout.strip():
                    print(f"  {result.stdout.strip()}")
            else:
                print(f"  Containerization failed:\n  {result.stderr}", file=sys.stderr)
                sys.exit(1)

    # Step 5: Print next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)

    if not (args.compile or args.containerize):
        print(f"\n1. Compile the benchmark:")
        print(f"   nemo-evaluator-byob {script_path}")
        print(f"\n2. (Optional) Containerize for evaluator container:")
        print(f"   nemo-evaluator-byob {script_path} --containerize")

    normalized = safe_name.replace("-", "_")
    print(f"\nRun evaluation:")
    print(f"  nemo-evaluator run_eval \\")
    print(f"    --eval_type byob_{normalized}.{safe_name} \\")
    print(f"    --model_url <MODEL_URL> --model_id <MODEL_ID> \\")
    print(f"    --model_type chat \\")
    print(f"    --output_dir ./results/{normalized} \\")
    print(f"    --api_key_name API_KEY")

    print(f"\nOr use with nemotron customize eval:")
    print(f"  nemotron customize eval --run MY-CLUSTER \\")
    print(f"    -t byob_{normalized}.{safe_name}")


if __name__ == "__main__":
    main()
