---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference for the curate/ingest step: parameters, artifacts, strategies, and errors."
topics: ["Curation", "Reference", "Ingestion"]
tags: ["Reference", "Curation", "Steps"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/ingest

The `curate/ingest` step reads a raw Parquet or JSONL corpus, mints a content-derived document identifier when the corpus does not carry one, maps source columns onto the names that the other curation steps expect, and writes JSONL shards.
The step runs without Ray or a GPU.

Use this step first when a corpus arrives without a stable document identifier or in Parquet form.
A corpus that is already JSONL with a unique identifier column does not need to pass through it.

## Syntax

```bash
uv run --no-sync nemotron steps run curate/ingest \
    [-c <config-name-or-path>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../../train-models/reference/cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/curate/nemo_curator/ingest/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Reads `./raw/*.parquet`, mints identifiers from `url` and `text`, and carries the `url` column through. |
| `tiny.yaml` | Reads the packaged profile fixture as JSONL, taking identity and source from the `id` and `source` columns. The checked-in input path is a container path; override `input` for local runs. |

## Inputs and Outputs

| Direction | Artifact type | Content |
| --- | --- | --- |
| Consumes | `raw_jsonl` | A raw corpus as Parquet or JSONL, in the column layout it arrived with. |
| Produces | `prepared_jsonl` | `part_<n>.jsonl` shards holding `text`, a unique `id`, optionally `source`, and any carried columns, written beside `ingest_report.json`. |

`ingest_report.json` records how each identifier was derived and lists the columns that were available in the source under `columns_available`.

## Parameters

```{option} input

The raw corpus: a glob, a directory, or a list of either, in Parquet or JSONL.
A directory is searched recursively for corpus files; known report and accounting sidecars are excluded.
```

```{option} output_dir

Directory that receives the `part_<n>.jsonl` shards and `ingest_report.json`.
```

```{option} format

One of `parquet`, `jsonl`, or `auto`.
`auto` infers the format from the file extensions and refuses a mixed set rather than reading the half that matches.

Default: `auto`.
```

```{option} text_field

Column holding the document text in the source corpus.

Default: `text`.
```

```{option} id_from

Column carrying the corpus's own document identifier.
When `null`, an identifier is minted from content so that it survives resharding.

Default: `null`.
```

```{option} id_fields

Columns from which a minted SHA-256 identifier is derived.
The list must include `text_field`, so that changed content cannot retain the same identity.
Values are framed as a canonical JSON array before hashing.
The recipe is recorded in `ingest_report.json`.

Default: `[url, text]`.
```

```{option} id_prefix

Prefix applied to minted identifiers so that documents from different corpora remain distinguishable after mixing.

Default: `""`.
```

```{option} source_from

Column to map onto the `source` field.
```

```{option} source

A single literal source value for a corpus that has only one source.
```

```{option} keep_fields

Columns carried into the output.
Every other column is dropped at ingestion.

Default: `[url]`.
```

```{option} on_duplicate

One of `refuse`, `drop`, or `suffix`.
Documents with the same configured identity fields, or records with the same corpus-provided identifier, collide.
`drop` keeps the first record of each group; `suffix` keeps every copy under a distinguishable identifier.
Both change the corpus, so `refuse` is the default and the run stops with a count.

Default: `refuse`.
```

## Strategies

| When | Then |
| --- | --- |
| The corpus is Parquet | Point `input` at the Parquet glob and leave `format` at `auto`. Parquet is read with `pyarrow`, so ingestion needs no Ray cluster. |
| The corpus has no document identifier | Leave `id_from` at `null`. Set `id_prefix` when the corpus will later be mixed with another. |
| The corpus already has a stable identifier | Set `id_from` to that column. Nothing is minted, and `ingest_report.json` records that identity came from the corpus. |
| The run refuses duplicate identities | Duplicates are a property of the corpus. Decide deliberately: `drop` to deduplicate, or `suffix` to keep every copy. |

## Common Errors

```{option} duplicate_documents

Repeated identity fields or corpus-provided identifiers create a non-unique `id`, which `curate/subset` and `curate/decontamination` refuse.
The message reports the number of occurrences and groups.
Set `on_duplicate` to `drop` or `suffix`.
```

```{option} mixed_formats

The glob matched both Parquet and JSONL.
Set `format` explicitly or narrow the glob.
```

```{option} no_usable_documents

Every record lacked text, or every line was unparsable.
Check `text_field` against the column names listed under `columns_available` in `ingest_report.json`.
```

```{option} input_matched_no_files

The glob matched nothing.
Check the path; a directory is accepted and searched recursively.
```

```{option} invalid_identity_fields

`id_fields` must be a list of column names that includes `text_field`.
Add the text column, or set `id_from` to a stable corpus-provided identifier.
```

## Examples

Ingest a Parquet corpus with a minted identifier and a constant source label:

```bash
uv run --no-sync nemotron steps run curate/ingest -c default \
    input=./raw/vi/*.parquet \
    output_dir=./output/vi/ingested \
    id_prefix=c4vi- \
    source=c4_vi \
    keep_fields='[url]'
```

Ingest JSONL that already carries `id` and `source` columns:

```bash
uv run --no-sync nemotron steps run curate/ingest -c tiny \
    input=${PWD}/src/nemotron/steps/curate/nemo_curator/profile/data/tiny/*.jsonl \
    output_dir=./output/ingest_tiny
```

## Related Pages

- [`src/nemotron/steps/curate/nemo_curator/ingest/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/ingest/README.md)
- {doc}`flow-config` for the `corpus` block that the flow translates into these parameters
- {doc}`../explanation/pipeline-artifacts`
