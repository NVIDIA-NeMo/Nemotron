# SKILL: Stage 4 -- Build Your Own Benchmark (BYOB)

## Purpose

Generate multiple-choice question (MCQ) evaluation benchmarks from domain-specific text corpora. This enables automated evaluation of customized models on domain knowledge that may not be covered by existing public benchmarks.

## When to Use

- You need domain-specific evaluation for a specialized field (medical, legal, financial)
- Existing benchmarks do not cover your target language
- You want to track model quality across customization iterations
- You need to compare multiple model variants on domain knowledge

Skip this stage if:
- Standard benchmarks (MMLU, ARC, HellaSwag) sufficiently cover your evaluation needs
- You already have a domain-specific benchmark dataset
- You are iterating rapidly and can use perplexity as a proxy

## Inputs Required

Before running this stage, confirm these with the user:

| Input | Required? | Default | Notes |
|-------|-----------|---------|-------|
| Source text corpus | Yes | None | Ask: "Where is the domain text corpus for MCQ generation? (local directory of text/JSONL or HuggingFace dataset)" |
| Target subjects/topics | No | All subjects in corpus | Ask: "Any specific subjects or topics to focus on? (e.g., cardiology, neurology)" |
| Source benchmark to adapt | No | `cais/mmlu` | Ask: "Adapt from an existing benchmark (e.g., MMLU), or generate from scratch using your corpus?" |
| Language for benchmarks | Yes | `en-US` | Ask: "What language should the benchmarks be in?" |
| Number of questions | No | 5000 | Ask: "How many MCQ questions to generate? (1000-10000)" |
| LLM endpoint for generation/judging | No | `openai/gpt-oss-120b` via NIM | Ask: "Which LLM for question generation and judging? (NIM API, local NIM, or custom endpoint)" |
| Whether to translate benchmarks | If source corpus is English but target language is not | false | Ask: "Should we translate the generated benchmarks to your target language?" |
| Translation target language | If translating | None | Ask: "What target language for translation? (e.g., hi-IN, fr-FR)" |
| Distractor expansion | No | true (expand to 10 choices) | Ask: "Expand from 4 to 10 answer choices? (harder benchmark, recommended)" |
| Quality thresholds | No | easiness=0.8, hallucination=0.5 | Ask: "Custom quality thresholds, or use defaults? (easiness 0.8, hallucination 0.5)" |

If any required input is missing, ask the user before proceeding.

## Pipeline Architecture

The BYOB pipeline runs 5 sequential sub-stages (with additional planned stages):

```
Input Corpus (HF dataset or local JSONL)
     |
     v
[1] Generate MCQs (DataDesigner LLM) --> raw questions with 4 options
     |
     v
[2] Judge Quality (DataDesigner LLM) --> score each question on validity and category
     |
     v
[3] Expand Distractors (optional) --> expand from 4 to 10 answer choices
     |
     v
[4] Validity Check --> verify distractor correctness (single correct answer)
     |
     v
[5] Filter --> apply easiness/hallucination thresholds
     |
     v
MCQ Benchmark Dataset
```

**Note:** Deduplication (semantic), coverage check, and outlier detection are configured in the YAML but not yet wired into the `generate_byob_benchmark()` pipeline. Their config fields exist in `ByobConfig` (`semantic_deduplication_config`, `do_coverage_check`, `semantic_outlier_detection_config`) for future use.

## Prerequisites

| Prerequisite | Description |
|-------------|-------------|
| Domain text corpus | JSONL or plain text files with domain content |
| OPENAI_API_KEY | OpenAI-compatible API key for NIM endpoint (MCQ generation and judging) |
| NeMo Curator | For deduplication and filtering steps |
| Python 3.10+ | Runtime environment |

No GPU required -- BYOB uses NIM API for generation (cloud-hosted models).

## Input Format

The input corpus should be a directory of text files or JSONL records:

**Plain text:**
```
/data/corpus/
  document_001.txt
  document_002.txt
  ...
```

**JSONL:**
```jsonl
{"text": "...", "source": "textbook_ch1", "topic": "cardiology"}
{"text": "...", "source": "textbook_ch2", "topic": "neurology"}
```

Recommended corpus properties:
- Each document/record: 200-5000 words
- Total corpus: 1000+ documents for meaningful coverage
- Diverse topics within the domain
- Factually accurate reference material

## Config Reference (`config/default.yaml`)

```yaml
expt_name: nemotron_byob
random_seed: 42
ndd_batch_size: 32

# --- Seed data ---
split: test
subset: all
input_dir: ./datasets                     # Local input directory (JSONL)
output_dir: ./output/byob                 # Output directory
hf_dataset: cais/mmlu                     # HuggingFace dataset (used if input_dir is empty)
language: en-US
metadata_file: null
source_subjects: []

target_source_mapping: {}
few_shot_samples_per_query: 1
queries_per_target_subject_document: 1
num_questions_per_query: 2

chunking_config:
  window_size: 4096

# --- Question generation (DataDesigner model config) ---
generation_model_config:
  alias: gpt-oss-120b
  model: openai/gpt-oss-120b
  provider: nvidia
  inference_parameters:
    max_tokens: 16000
    max_parallel_requests: 8
    temperature:
      distribution_type: uniform
      params:
        low: 0.9
        high: 1.0
    top_p: 1.0

judge_model_config:
  alias: gpt-oss-120b
  model: openai/gpt-oss-120b
  provider: nvidia
  inference_parameters:
    max_tokens: 16000
    max_parallel_requests: 8

# --- Semantic deduplication (config exists, not yet wired into pipeline) ---
semantic_deduplication_config:
  model_identifier: sentence-transformers/all-MiniLM-L6-v2
  n_clusters: 1
  eps: 0.07
  remove_duplicates: false

# --- Distractor expansion ---
do_distractor_expansion: true
distractor_expansion_model_config:
  alias: gpt-oss-120b
  model: openai/gpt-oss-120b
  provider: nvidia
  inference_parameters:
    max_tokens: 16000
    max_parallel_requests: 8

# --- Coverage check (config exists, not yet wired into pipeline) ---
do_coverage_check: true
coverage_check_config:
  model_identifier: sentence-transformers/all-MiniLM-L6-v2
  window_size: 1024

# --- Distractor validity ---
distractor_validity_model_config:
  alias: gpt-oss-120b
  model: openai/gpt-oss-120b
  provider: nvidia
  inference_parameters:
    max_tokens: 16000
    max_parallel_requests: 8

# --- Semantic outlier detection (config exists, not yet wired into pipeline) ---
semantic_outlier_detection_config:
  model_identifier: sentence-transformers/all-MiniLM-L6-v2
  n_neighbours_min: 1
  remove_outliers: true

# --- Filtering ---
easiness_threshold: 0.8
hallucination_threshold: 0.5
remove_hallucinated: true
remove_easy: false

# --- Translation (used by run_translate.py) ---
translate:
  dataset_path: null
  source_language: en-US
  target_language: hi-IN
  translation_model_config:
    mode: llm
    params:
      alias: gpt-oss-120b
      model: openai/gpt-oss-120b
      provider: nvidia
      inference_parameters:
        max_tokens: 16000
        max_parallel_requests: 8
  backtranslation_quality_metrics:
    - type: sacrebleu
      threshold: 25
    - type: chrf
      threshold: 50
  remove_low_quality: false
```

The `ByobConfig` dataclass (in `data_prep/byob.py`) maps these fields directly. **Implementation status:** Generate, judge, expand distractors, validity check, and filter are fully implemented. Semantic deduplication, coverage check, and outlier detection have config support but are not yet called in `generate_byob_benchmark()`.

## Execution

```bash
python src/nemotron/customization_recipes/nemotron/stage4_byob/run_generate.py \
  --config src/nemotron/customization_recipes/nemotron/stage4_byob/config/default.yaml \
  input_dir=/data/hindi_medical_texts \
  output_dir=/data/hindi_medical_benchmark \
  language=hi \
  num_questions_per_query=5
```

### Running Sub-Tasks

BYOB also provides separate scripts for seed preparation and translation:

```bash
# Prepare seed dataset only (without running the full pipeline)
python src/nemotron/customization_recipes/nemotron/stage4_byob/run_prepare.py \
  --config src/nemotron/customization_recipes/nemotron/stage4_byob/config/default.yaml \
  input_dir=/data/domain_corpus \
  output_dir=/data/byob_seed

# Translate a generated benchmark to a target language
python src/nemotron/customization_recipes/nemotron/stage4_byob/run_translate.py \
  --config src/nemotron/customization_recipes/nemotron/stage4_byob/config/default.yaml \
  translate.dataset_path=/data/byob_benchmark/benchmark.jsonl \
  translate.target_language=hi-IN
```

## Translation

To create benchmarks in a language where domain corpora are primarily in English, use the `translate` section of the config (see `config/default.yaml`):

```yaml
translate:
  dataset_path: /data/byob_benchmark/benchmark.jsonl   # Path to generated benchmark
  source_language: en-US
  target_language: hi-IN
  translation_model_config:
    mode: llm                            # Backend: google | aws | llm
    params:
      alias: gpt-oss-120b
      model: openai/gpt-oss-120b
      provider: nvidia
      inference_parameters:
        max_tokens: 16000
        max_parallel_requests: 8
  backtranslation_quality_metrics:       # Back-translate and verify consistency
    - type: sacrebleu
      threshold: 25
    - type: chrf
      threshold: 50
  remove_low_quality: false
```

Run translation via the dedicated script:

```bash
python src/nemotron/customization_recipes/nemotron/stage4_byob/run_translate.py \
  --config src/nemotron/customization_recipes/nemotron/stage4_byob/config/default.yaml \
  translate.dataset_path=/data/byob_benchmark/benchmark.jsonl \
  translate.target_language=hi-IN
```

The `TranslationConfig` dataclass in `data_prep/translate.py` uses `source_lang` / `target_lang` (short form). The `translate_byob_benchmark()` facade maps the YAML's `source_language` / `target_language` to these fields, stripping the region suffix (e.g., `en-US` becomes `en`).

## Output Format

The benchmark is saved as JSONL compatible with NeMo Evaluator:

```jsonl
{"question": "<question text>", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "B", "metadata": {"topic": "cardiology", "difficulty": 0.72, "source": "textbook_ch1", "language": "hi"}}
```

Additional output files:
- `benchmark.jsonl` -- the final MCQ dataset
- `metadata.json` -- benchmark statistics (topic distribution, difficulty histogram, language stats)
- `quality_report.json` -- per-sub-stage metrics (pass rates, dedup counts, coverage scores)

## How to Verify Success

1. **Question count**: Final dataset should have >= 80% of `num_questions` target.
   - If significantly fewer: source corpus too small or quality thresholds too strict

2. **Topic coverage**: Check `metadata.json` for topic distribution.
   - Should cover `min_topics` distinct topics
   - No single topic should dominate (>30% of questions)

3. **Difficulty distribution**: Check difficulty histogram in `metadata.json`.
   - Should approximate normal distribution centered around 0.5-0.7
   - Avoid too many trivial (< 0.3) or impossible (> 0.95) questions

4. **Human spot-check**: Randomly sample 50 questions and verify:
   - Question is clear and unambiguous
   - Correct answer is actually correct
   - Distractors are plausible but wrong
   - Language is correct

5. **Baseline model evaluation**: Run the base (uncustomized) model on the benchmark.
   - Should score significantly below the customized model
   - If base model scores >80%: questions may be too easy

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| NIM API errors (429) | Rate limiting | Reduce `batch_size`, add delay between batches |
| Low question yield (<50% of target) | Corpus too small or passages too short | Add more documents, increase `questions_per_passage` |
| Many questions filtered in judging | Generation quality low | Increase generation `temperature` slightly, improve prompt template |
| Poor distractor quality | Distractors too obviously wrong | Set `expand_distractors.strategy: plausible`, use stronger generation model |
| Duplicate questions after dedup | Low corpus diversity | Add more diverse source material, reduce `questions_per_passage` |
| All questions on same topic | Corpus is topic-homogeneous | Set `coverage.rebalance: true`, add documents covering different subtopics |
| Translation quality poor | Machine translation artifacts | Enable `verify_translation`, use human review for critical benchmarks |

## Feeding BYOB Output to Evaluation (Stage 4 -> Stage 5 Bridge)

After generating a benchmark with the BYOB pipeline, use the **NeMo Evaluator BYOB framework** to create a compiled benchmark definition that the evaluator can run directly. This is the "sovereign benchmark bridge" between stage4 and stage5.

### Quick Path: Auto-Generate + Compile

```bash
# 1. Generate the benchmark definition from BYOB output
python src/nemotron/customization_recipes/nemotron/stage5_eval/create_sovereign_benchmark.py \
  --byob-output /data/byob_benchmark/benchmark.jsonl \
  --benchmark-name "hindi-medical-mcq" \
  --output-dir /data/eval/benchmarks/ \
  --compile

# 2. Run evaluation with both standard and sovereign benchmarks
nemotron customize eval --run MY-CLUSTER \
  -t adlr_mmlu \
  -t byob_hindi_medical_mcq.hindi-medical-mcq
```

### Manual Path: Copy and Customize Template

For more control, copy the sovereign benchmark template and customize it:

```bash
# 1. Copy the template
cp src/nemotron/customization_recipes/nemotron/stage5_eval/sovereign_benchmark.py \
   /data/eval/benchmarks/hindi_medical_benchmark.py

# 2. Edit the file: set BENCHMARK_NAME, DATASET_PATH, LANGUAGE, and
#    adjust the prompt template for your specific domain/language.

# 3. Compile with nemo-evaluator-byob
nemo-evaluator-byob /data/eval/benchmarks/hindi_medical_benchmark.py

# 4. (Optional) Containerize for the sovereign evaluator container
nemo-evaluator-byob /data/eval/benchmarks/hindi_medical_benchmark.py --containerize
```

### Using Environment Variables (No Code Changes)

The sovereign_benchmark.py template supports environment variable overrides, so you can use it without editing:

```bash
export SOVEREIGN_BENCHMARK_NAME="hindi-medical-mcq"
export SOVEREIGN_DATASET_PATH="/data/byob_benchmark/benchmark.jsonl"
export SOVEREIGN_LANGUAGE="hi"
export SOVEREIGN_NUM_CHOICES="4"

nemo-evaluator-byob src/nemotron/customization_recipes/nemotron/stage5_eval/sovereign_benchmark.py
```

### Full End-to-End Command Sequence

```bash
# Stage 4: Generate MCQ benchmark
python src/nemotron/customization_recipes/nemotron/stage4_byob/run_generate.py \
  --config src/nemotron/customization_recipes/nemotron/stage4_byob/config/default.yaml \
  input_dir=/data/hindi_medical_texts \
  output_dir=/data/byob_benchmark \
  language=hi \
  num_questions_per_query=5

# Bridge: Create sovereign benchmark definition
python src/nemotron/customization_recipes/nemotron/stage5_eval/create_sovereign_benchmark.py \
  --byob-output /data/byob_benchmark/benchmark.jsonl \
  --benchmark-name "hindi-medical-mcq" \
  --output-dir /data/eval/benchmarks/ \
  --compile

# Stage 5: Evaluate with standard + sovereign benchmarks
nemotron customize eval --run MY-CLUSTER \
  -t adlr_mmlu \
  -t adlr_arc_challenge_llama_25_shot \
  -t hellaswag \
  -t byob_hindi_medical_mcq.hindi-medical-mcq
```

### Utility Reference: create_sovereign_benchmark.py

| Argument | Required | Description |
|----------|----------|-------------|
| `--byob-output` | Yes | Path to stage4 benchmark.jsonl |
| `--benchmark-name` | Yes | Benchmark name (used as eval task ID) |
| `--output-dir` | Yes | Directory for generated benchmark file |
| `--language` | No | Override language label (auto-detected) |
| `--compile` | No | Compile with nemo-evaluator-byob |
| `--containerize` | No | Build Docker image with benchmark (implies --compile) |

## Artifacts Produced

| Artifact | Type | Path | Consumed By |
|----------|------|------|-------------|
| MCQ benchmark | JSONL | `output_dir/benchmark.jsonl` | stage5_eval (via sovereign benchmark bridge) |
| Benchmark metadata | JSON | `output_dir/metadata.json` | Analysis |
| Quality report | JSON | `output_dir/quality_report.json` | Analysis |
| Raw questions (pre-filter) | JSONL | `output_dir/raw_questions.jsonl` | Debugging/re-runs |
