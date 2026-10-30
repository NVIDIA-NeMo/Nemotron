<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- How-to: lay out bring your own benchmark (BYOB) `input_dir` with one UTF-8 `.txt` corpus per `target_source_mapping` key. Use when you supply your own domain text and need the directory layout the prepare step expects. -->

# Using Your Own Domain Data

Use this tutorial to learn how to build your own benchmark with your own domain text.

You will create a parent folder, your `input_dir`, with one subdirectory per key in `target_source_mapping`, each holding UTF-8 `.txt` files the prepare stage can sample.

When you finish the steps below, the prepare stage can build stems from your corpora together with your Hugging Face subject wiring.

- Each directory name directly under `input_dir` must match a key in `target_source_mapping`, for example `banking`, `police`, or `maths`.
- Split long material across several `.txt` files when you want different documents to drive different queries.
- Set `input_dir` in YAML to the parent directory so `uv run nemotron steps run byob/mcq` resolves relative paths from the shell working directory, or use an absolute path for a fixed location.

## Step 1: Create Target Directories

Create one folder per entry you plan under `target_source_mapping`.

```bash
# Example layout next to your workspace; adjust paths to match your machine.
mkdir -p ./data/byob/banking
mkdir -p ./data/byob/police
mkdir -p ./data/byob/maths
```

Path tips:

- Prefer absolute paths in YAML when several people reuse the same file from different working directories.
- Relative paths resolve from the shell working directory where you invoke `uv run nemotron steps run byob/mcq`.

The pipeline resolves `input_dir` relative to that working directory.
If you need a fixed location regardless of where people run the command, set `input_dir` to an absolute path in YAML.

Do not use Hugging Face subject names as directory names; use the same strings you use as keys under `target_source_mapping`.
Few-shot subjects such as `high_school_mathematics` still belong under `source_subjects` and under each target’s `subjects` field.

## Step 2: Add Text Files

Place UTF-8 text under each target directory.
Use several smaller files instead of one huge blob when you want different documents to drive different queries.

```bash
cat > ./data/byob/banking/intro.txt << 'EOF'
Modern banking in India originated in the mid-18th century...
EOF
```

Read the sample `src/nemotron/steps/byob/data/tiny_input/maths/tiny.txt` file for tone and length.

## Step 3: Point YAML at the Parent Directory

Set `input_dir` to the parent of those directories and wire few-shot subjects your Hugging Face split actually contains.

```yaml
input_dir: ./data/byob
hf_dataset: cais/mmlu
subset: all
split: test
source_subjects:
  - high_school_mathematics

target_source_mapping:
  banking:
    subjects:
      - business_ethics
      - econometrics
  police:
    subjects:
      - jurisprudence
      - international_law
  maths:
    subjects:
      - high_school_mathematics
```

Allowed `hf_dataset` values and default `subset` / `split` behavior are listed in {doc}`../reference/benchmarks`.

## Best Practices

### Content Quality

- Aim for a few thousand characters per document when you want diverse stems without exhausting context.
- Prefer complete explanations over fragments so judges and filters see coherent context.
- Strip or replace personally identifiable information before you run in shared environments.

### Organization

- Keep one domain taxonomy per directory; do not mix unrelated subjects under the same target key.
- Split long manuals into several `.txt` files so `queries_per_target_subject_document` can visit different offsets.

### Performance and Chunking

- Pilot with a handful of documents per target before scaling out.
- Use `chunking_config.window_size` when you need sliding windows over very long text; `null` keeps each file as one unit.
  Field detail is in {doc}`../reference/generate-config`.

## Next Steps

- Run prepare alone or the full pipeline after you align YAML: {doc}`prepare-data`.
- Inspect `seed.parquet` and downstream paths in {doc}`../reference/output-files`.
