# Multimodal Data and Generation

Read this reference for Nemotron 3.5 source selection, model roles, bundle
handoffs, or export validation. Use [embedding](embed.md) or [reranking](rerank.md)
for training and evaluation. The procedures here do not require repository docs.

## Choose the Source Input

Use `source_directory.path` for independent text and image files. The producer
discovers UTF-8 `.txt`, `.md`, and `.markdown` files and static PNG, JPEG, and
WebP images recursively. It chunks text to 1,024 tokens by default and creates
one unit per image. Keep outputs outside the source directory.

Use `sources_file` for explicit text/image pairs, page relationships, or custom
unit boundaries. Set exactly one input mode. PDF extraction, optical character
recognition, and page rendering happen before this recipe.

Source records are UTF-8 JSONL objects with these fields:

| Field | Contract |
| --- | --- |
| `unit_id` | Required, nonempty, unique string without tabs or line breaks. |
| `document_id` | Required, nonempty string; pages of one document share an ID. |
| `text` | Original source text; defaults to an empty string. |
| `images` | Zero or one local image path; defaults to an empty list. |
| `page_number` | Optional integer starting at 1; records provenance without sorting. |
| `language` | Optional nonempty string; defaults to `source`. |
| `source_uri` | Optional string or null; records provenance without downloading. |

Every record needs nonblank text, an image, or both. Relative image paths
resolve from the JSONL file. Images must exist locally. Unknown fields,
duplicate IDs, missing content, and multiple images per unit are rejected.
Do not insert generated questions, negatives, or split assignments into sources.

For example, this paired record requires the referenced image:

```json
{"unit_id":"pump-p1","document_id":"pump","text":"Operating range: 10 to 30 degrees Celsius.","images":["pages/pump-001.png"],"page_number":1}
```

From the execution checkout, preview explicit sources and clear any inherited directory:

```bash
uv run nemotron embed sdg -c nemotron-3.5 -d \
  sources_file=/absolute/path/to/sources.jsonl source_directory=null
```

## Configure Generation and Judging

The local embedding or reranking checkpoint does not generate training queries.
Multimodal generation and judging both default to
`nvidia/nemotron-3.5-super-vl-120b-a12b`.

| Role | Model Selection | Request Settings |
| --- | --- | --- |
| Enrichment, summaries, and query generation | `sdg_generator_model` | `sdg_generator_options` |
| Shared summary/query judge and localization | `sdg_judge_model` | `sdg_judge_options` |
| Summary clustering | `sdg_options.summary_embedding_model` | `sdg_options.summary_embedding_*` |

`NEMOTRON_3_5_SDG_GENERATOR_MODEL` and `NEMOTRON_3_5_SDG_JUDGE_MODEL` override
the multimodal defaults. Explicit model fields take precedence over these
environment variables. Text QA keeps `qa_generation_model`,
`quality_judge_model`, and its existing defaults.

Chat roles use `https://integrate.api.nvidia.com/v1` and `NVIDIA_API_KEY` by
default. `NVIDIA_API_BASE_URL` changes both chat endpoints. Per-role options can
set `endpoint`, `credential_env`, `temperature`, `max_tokens`, `timeout`, and
`extra_body`. `credential_env` names an environment variable, never its value.
Keep secrets out of saved configs and request examples.

Summary clustering defaults to `nvidia/nemotron-3-embed-1b` at the NVIDIA
endpoint, with its own credential setting. Changing a chat endpoint does not
change the summary embedding endpoint. For local summary embeddings, supply
a compatible Sentence Transformers model and clear both
`summary_embedding_endpoint` and `summary_embedding_extra_body` with `null`.
An empty mapping merges with inherited options instead of clearing them.

Use image-capable models with structured responses for the multimodal chat
roles. The pinned recipe exposes a shared judge; do not assume support for
stage-specific judges from a newer producer version.

## Preserve the Handoff

The producer owns the query-group split and always exports a portable bundle.
Stage 1 preserves that split and all positive documents. Keep `retrieval_view=mixed`
to retain text-only, image-only, and paired units. `image_and_text` excludes
text-only units; do not substitute a missing view.

Key paths relative to the embedding `artifact_root` are:

| Path | Purpose |
| --- | --- |
| `stage0_sdg/generation_result.json` | Identifies the exact successful output and bundle manifest. |
| `stage0_sdg/multimodal/inputs/sources.jsonl` | Prepared input for directory runs. |
| `stage1_data_prep/train_mined.automodel_unrolled.json` | Training rows and relative corpus references. |
| `stage2_finetune/checkpoints/LATEST/model/consolidated/` | Model for evaluation and direct serving. |
| `stage3_eval/eval_results.json` | Metrics for the selected models. |

Keep the complete bundle and its assets when relocating training data. Portable
evaluation resolves its BEIR data from the Stage 0 manifest; do not substitute
the legacy text `stage1_data_prep/eval_beir` directory.

Query partitions share the document collection. Generated relevance labels are
not independent gold labels. Use independently authored queries and labels for
broader quality claims.

## Audit an Export

Normal handoffs check schema version 2, `grouped_query_disjoint` splits,
`full_collection` corpora, safe paths, required files, view availability, and
manifest identity. They do not repeat a full hash and content audit.

For a full audit, run this command in an environment containing Nemotron and
its Stage 0 dependencies, replacing the manifest path:

```bash
python -m nemotron.recipes.embed.stage0_sdg.validate \
  /absolute/path/to/run_manifest.json --view mixed
```

In a source checkout, the stage lockfile supplies the dependencies:

```bash
PYTHONPATH="$PWD/src" uv run --project src/nemotron/recipes/embed/stage0_sdg --locked \
  python -m nemotron.recipes.embed.stage0_sdg.validate \
  /absolute/path/to/run_manifest.json --view mixed
```

The audit checks inventory hashes, corpus/source consistency, relevance
references, and query assignments, including group and normalized-duplicate
leakage across partitions. A failure exits with an error. Inspect the reported
artifact before reuse; a passing audit does not prove label quality.

## Optional Source Detail

These permalinks document the bundled contract; they are not prerequisites:

- [Data contract at the supported Nemotron revision](https://github.com/NVIDIA-NeMo/Nemotron/blob/e68db1595a9f078a3fb48f8216fa0481d1c862a2/docs/nemotron/embed/data-reference.md).
- [SDG configuration at the supported Nemotron revision](https://github.com/NVIDIA-NeMo/Nemotron/blob/e68db1595a9f078a3fb48f8216fa0481d1c862a2/docs/nemotron/embed/sdg-reference.md).
