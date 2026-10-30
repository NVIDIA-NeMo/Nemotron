<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- How-to: lay out `input_dir`, wire `target_source_mapping`, and set sampling counts before running the prepare stage. Use when you already have a BYOB configuration and need a concrete checklist for local inputs. -->

# Prepare Your Own Domain Data

This guide shows how to place domain text under `input_dir`, map each target
key in `target_source_mapping`, and set the positive integers that control how
much work prepare does per document.

Use this guide when your configuration points at local sources and you want to
validate layout before you run prepare or the full pipeline.

For how prepare joins few-shot rows and domain text, see
{doc}`../explanation/data-preparation`.
For a filesystem-first walkthrough of directories and Parquet basenames, see {doc}`domain-data`.
Field-by-field options for the whole step live in {doc}`../reference/generate-config`.

## Directory Layout

The prepare stage reads domain text from `input_dir` and pairs each passage with
few-shot rows from the Hugging Face dataset named in `hf_dataset`.

For each key in `target_source_mapping`, supply one of the following next to
`input_dir`:

- A directory `input_dir/<target>/` that contains at least one `.txt` file.
- A Parquet file `input_dir/<target>.parquet` whose rows include the columns the
  loader expects, described in {doc}`../explanation/data-preparation`.

If both a directory and a file named `<target>.parquet` exist, the loader
prefers the directory and logs a warning.

## Source Subjects And Mapping

In this step, a *target subject* is one key in `target_source_mapping` on the
domain side of the pairing.

The `source_subjects` lists benchmark subjects from the Hugging Face dataset that
you are allowed to sample as few-shot sources.
Every name that appears under a target’s `subjects` field must also appear in
`source_subjects`, or validation fails.

Each target entry in `target_source_mapping` carries a `subjects` value in one
of two shapes:

- A list of source subject names selects those rows with uniform weights.
- An empty list means use every entry in `source_subjects`, again with uniform
  weights.
- A mapping from source subject name to a non-negative weight defines a
  weighted mixture, and weights must sum to a value greater than zero.
  The loader normalizes them internally.

Optional `tags` on a target pair with `metadata_file` when you supply tag
metadata from a file.
If `metadata_file` is absent, do not set `tags`, because validation rejects that
combination.

## Few-Shot and Query Counts

These positive integers live in the same configuration as the rest of the step
and control how much generation work each stored document drives:

- `few_shot_samples_per_query` caps how many few-shot exemplar rows feed each
  query.
- `queries_per_target_subject_document` sets how many queries run per domain
  document.
- `num_questions_per_query` sets how many questions each query produces.

They must all be greater than zero.
Short definitions also appear in the tables under {doc}`../reference/generate-config`.

## Hugging Face Dataset

Set `hf_dataset` to one of the identifiers listed in {doc}`../reference/benchmarks`.
If you omit `subset` or `split`, defaults come from `HF_DATASET_TO_SUBSET` and
the loader logic in `ByobConfig.from_yaml`.

## Run Prepare Alone

Set `stage: prepare` in your configuration file.

Prepare writes the seed artifact that generate consumes.
The file names and roles are listed in {doc}`../reference/output-files`.
