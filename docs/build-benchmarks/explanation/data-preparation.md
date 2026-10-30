<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- Explanation: how the BYOB prepare stage pairs Hugging Face few-shots with domain text into seed.parquet. -->

# Data Preparation for Multiple-Choice Question Benchmarks

The prepare stage builds the seed dataset that backs multiple-choice question (MCQ) generation in `nemotron steps run byob/mcq`.

The data preparation stage pairs few-shot rows from your configured Hugging Face benchmark with domain-specific text you place under `input_dir`.

Each seed row joins exemplar MCQs with local passages so generate can copy structure and vocabulary from the benchmark while staying grounded in your corpus.

## Why Few-Shot Examples?

Few-shot rows show answer format, option layout, and tone.
They steer style toward the benchmark you picked and anchor quality on curated source questions instead of inventing layout from scratch.

## Domain-Specific Data Requirements

Domain text is what the model reads when it proposes a new question, options, and a correct answer index.

### Format options

- Text files, one directory per target key in `target_source_mapping`:

  ```text
  input_dir/
  ├── banking/
  │   ├── intro.txt
  │   └── regulations.txt
  └── police/
      └── main.txt
  ```

- One Parquet file per target, for example `input_dir/banking.parquet`.

Each `target_source_mapping` key must match either a directory under `input_dir` or a sibling `*.parquet` basename.

### Parquet Targets

Parquet targets must include `text` and `file_name`.
The loader treats each row as `parquet_path:file_name`, so `file_name` acts as the stable document handle inside that table.

### Content Practices

- Curate plain text for every target.
  One folder such as `medicine` can suffice for a pilot; another team might split into `radiology`, `podiatry`, `clinical_documentation`, and similar labels that mirror internal source taxonomy.
- Avoid shipping personally identifiable information (PII), or replace sensitive spans with synthetic text before you run the step.
- Few-shot tag metadata is separate from domain files.
  When you set `metadata_file`, you can attach tags to benchmark rows and filter few-shots with per-target `tags`, as described in {doc}`get-right-questions`.
- Start with data you can legally host in a development environment, then tighten controls before you connect restricted enterprise corpora.

## Step 1: Load Source Benchmark

The loader reads `hf_dataset` together with `split`, `subset`, and `source_subjects`, letting `ByobConfig.from_yaml` fill defaults where the YAML omits them.

Expect question text, labeled choices, a keyed correct answer, subject labels, and optional tag strings when your metadata pipeline populates them.

### Choosing a Benchmark

Pick a benchmark whose tone matches the style you want the generator to echo, for example:

- `cais/mmlu` for broad academic multiple-choice style.
- `TIGER-Lab/MMLU-Pro` for harder prompts with more reasoning surface.
- `Idavidrein/gpqa` for graduate-level science questions.

Authoritative identifiers and default subsets are listed in {doc}`../reference/benchmarks`.

## Step 2: Filter by Target-Source Mapping

This stage decides which benchmark rows are eligible before sampling.

Each target lists `subjects` and optional `tags` (when `metadata_file` is set) using list or weighted-map syntax.

### Sample Configuration With Weights

```yaml
target_source_mapping:
  banking:
    subjects:
      management: 0.7
      marketing: 0.3
    tags:
      reasoning,unambiguous: 0.8
      knowledge: 0.2
```

The `tags` field always requires `metadata_file` with question-level rows the loader can join.

### How It Works

1. Place domain text under `input_dir/banking/` or supply `input_dir/banking.parquet` with the required columns.
2. Filter few-shot candidates to the listed subjects, then to tag combinations when tags are active.
   Subject strings must exist in the Hugging Face split you loaded; confirm spelling on the dataset card.
3. Weights on `subjects` and `tags` normalize to probabilities.
4. The Cartesian product of subject and tag choices becomes the sampling categories inside `make_samples`.

For the `banking` target above, you get four combined categories:

- `management` + `reasoning,unambiguous`: 0.7 × 0.8 = 0.56
- `management` + `knowledge`: 0.7 × 0.2 = 0.14
- `marketing` + `reasoning,unambiguous`: 0.3 × 0.8 = 0.24
- `marketing` + `knowledge`: 0.3 × 0.2 = 0.06

Strings such as `reasoning,unambiguous` split on commas when matched against metadata, following `mcq/dataset.py`.

## Step 3: Sample Few-Shot Examples

```yaml
few_shot_samples_per_query: 1
```

Each query (domain chunk plus its draw) pulls `few_shot_samples_per_query` rows from the filtered benchmark table using the combined weights from Step 2.

Random draws add diversity, benchmark curation supplies baseline quality, and subject or tag choices keep prompts aligned with each target corpus.
One to three few-shots per query is a common starting point; higher counts improve format mimicry but raise tokens, latency, and cost.

## Step 4: Load Domain Content

### Input formats

- Text files such as `input_dir/banking/intro.txt`.
- Parquet bundles such as `input_dir/banking.parquet` with `text` and `file_name`.

### Chunking

```yaml
chunking_config:
  window_size: 4096  # characters; use null for whole-document context
```

When `window_size` is a positive integer and the document exceeds the window, `chunk_text` samples a random start and returns exactly `window_size` characters for that query.
When `window_size` is `null`, the loader keeps the entire file as one segment (`segment_start` 0, `segment_end` equal to length).

Chunking spreads questions across long manuals, exposes different sections per query, and can shrink per-call payloads compared with sending an entire chapter at once.

## Step 5: Create Seed Dataset

For every domain document under a target, `make_samples` follows this loop:

1. Sample `queries_per_target_subject_document` draws from the subject×tag distribution.
2. For each draw, fetch `few_shot_samples_per_query` benchmark rows that satisfy the subject and tag filters.
3. Attach either the full document or a chunked slice according to `chunking_config`.

```yaml
queries_per_target_subject_document: 10
```

Rows accumulate in `seed.parquet`.
Refer to {doc}`../reference/output-files` for column names.

Generate then reads those paired fields so the model sees canonical MCQ structure from the benchmark plus grounding text from your files.

## Next Steps

- {doc}`question-generation` shows how generate consumes `seed.parquet`.
- {doc}`pipeline-overview` situates prepare inside the wider `mcq` family.
- {doc}`../reference/generate-config` documents every YAML knob, including chunking and query defaults.
