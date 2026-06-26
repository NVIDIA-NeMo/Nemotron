<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Output Files

All paths below are relative to `output_dir` from your YAML and the string `expt_name`.

## Prepare

| File | Description |
| --- | --- |
| `seed.parquet` | Few-shot rows plus domain chunks. |

## Generate

The final generation creates these files:

| File | Description |
| --- | --- |
| `benchmark_raw.parquet` | Snapshot immediately before column renaming for the final schema. |
| `benchmark.parquet` | Final MCQ schema; column meanings and a sample row are in the next section. |

The following intermediate files that are created in the `stage_cache` directory:

| File | Stage |
| --- | --- |
| `generated_questions.parquet` | GENERATION |
| `judged_questions.parquet` | JUDGEMENT |
| `semantic_deduplicated_questions.parquet` | SEMANTIC_DEDUPLICATION |
| `expanded_distractors.parquet` | DISTRACTOR_EXPANSION (only when `do_distractor_expansion` is true) |
| `coverage_check.parquet` | COVERAGE_CHECK (only when `do_coverage_check` is true) |
| `valid_distractors.parquet` | DISTRACTOR_VALIDITY_CHECK |
| `semantic_outlier_detection.parquet` | SEMANTIC_OUTLIER_DETECTION |
| `filtered_questions.parquet` | HALLUCINATION_EASINESS_DETECTION |

(final-mcq-columns)=
## Final MCQ Columns

Generation and translation both export the same eight columns on the final `benchmark.parquet` file.

| Column | Meaning |
| --- | --- |
| `question_id` | Stable identifier for the row, taken from the internal `id_question` field before export. |
| `question` | Stem text for the multiple-choice item. After translation this is the target-language text. |
| `options` | Ordered list of choice strings. The list order matches the letter labels implied by `answer_index`. |
| `answer_index` | Zero-based index into `options` for the correct choice. |
| `answer` | Letter label for the correct choice, derived from `answer_index` (`0` → `A`, `1` → `B`, and so on). |
| `cot_content` | Reserved for chain-of-thought text. The current pipeline sets this column to the literal `-` for every row on export. |
| `src` | Reserved for a source document marker. The current pipeline sets this column to the literal `-` for every row on export. |
| `category` | Target key from `target_source_mapping` in your generation configuration, in other words the domain bucket name for the row, not the Hugging Face few-shot subject name. |

### Sample Row

Values below are illustrative; your identifiers and wording will differ.

```json
{
  "question_id": "mcq-00042",
  "question": "If x^2 = 9, which value can x take?",
  "options": ["-3 only", "-3 or 3", "3 only", "9"],
  "answer_index": 1,
  "answer": "B",
  "cot_content": "-",
  "src": "-",
  "category": "maths"
}
```

## Translate

The translate stage creates the following final output files:

| File | Description |
| --- | --- |
| `benchmark_raw.parquet` | Intermediate snapshot prior to optional quality filtering. |
| `benchmark.parquet` | Final translated MCQ after fields are renamed back to `question`, `options`, `answer_index`, and `answer`. Column semantics match generation; see {ref}`final-mcq-columns`. |

The following intermediate files are created in the `staged_cache` directory.

| File | Stage |
| --- | --- |
| `translated_questions.parquet` | TRANSLATION |
| `backtranslated_questions.parquet` | BACKTRANSLATION |
| `quality_metrics.parquet` | QUALITY_METRICS |

Intermediate translation Parquet files can include additional columns such as `question_translated`, `options_translated`, backtranslation fields, and metric scores.
