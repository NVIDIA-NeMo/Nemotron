---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Configuration field reference for curate/nemo_curator."
topics: ["Curation", "Reference", "Configuration"]
tags: ["Reference", "Configuration", "Curation"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/nemo_curator Configuration

The `curate/nemo_curator` step reads JSONL text with NeMo Curator, optionally materializes a Hugging Face snapshot, applies language, word-count, and domain gates and an optional approved filter policy, and writes JSONL shards.
It is the only curation step that drops rows.

The step reads YAML from `src/nemotron/steps/curate/nemo_curator/config/`.

| File | Purpose |
| --- | --- |
| `tiny.yaml` | Curator-container verification configuration. Optional filters are disabled. Override `input_glob` for local runs because the checked-in path is a container path. |
| `default.yaml` | Example Hugging Face snapshot workflow for FineWeb-Edu-style JSONL with language and word-count filters enabled. |
| `vi_c4_measure.yaml`, `vi_c4_apply.yaml` | Flow configurations, read by the flow driver rather than by this step. Refer to {doc}`flow-config`. |

## Inputs and Outputs

| Direction | Artifact type | Required | Content |
| --- | --- | --- | --- |
| Consumes | `raw_jsonl` | Yes | Raw JSONL records, or a Hugging Face snapshot materialized before Curator reads `input_glob`. `prepared_jsonl` from `curate/ingest` is accepted. |
| Consumes | `filter_policy` | No | An approved policy whose thresholds the step applies. Without one, the step runs its language, length, and domain gates and the manifest records that no thresholds were applied. |
| Produces | `filtered_jsonl` | | Filtered JSONL records. Language and domain annotations are present only when those gates are enabled. |
| Produces | `curation_manifest` | | `run_manifest.json`: per-shard row counts and digests, the policy applied and its approval state, and `completed_at`, written only when the run reaches its write barrier. Written only when `emit_manifest` is set. |
| Produces | `curation_ledger` | | `curation_ledger.json`: how many documents entered, how many left, and which gate rejected each one. Written only when `emit_ledger` is set. |

## Input and Output Fields

```{option} input_glob

JSONL file path or glob passed to NeMo Curator `JsonlReader`.
The step reads `.jsonl`, `.json`, and `.ndjson`; Parquet must pass through `curate/ingest` first.
```

```{option} output_dir

Directory where NeMo Curator writes JSONL output shards.
```

```{option} text_field

Record field containing the text to curate.

Default: `text`.
```

```{option} metadata_fields

Extra record fields to carry through the pipeline.
`JsonlReader` treats fields as a projection, so any field omitted here is discarded at the first stage.
An empty list reproduces the text-only read.

Default: `[]`.
```

```{option} id_field

Field holding the source corpus's own document identifier, recorded in the run manifest.
This is the only document identifier that survives resharding; Curator's `AddId` is positional.

Default: `null`.
```

```{option} source_field

Field naming the corpus a record came from.
When set, the run manifest carries per-source input and output counts.

Default: `null`.
```

```{option} dataset

Optional keyword arguments passed to `huggingface_hub.snapshot_download`, such as `repo_id`, `repo_type`, `local_dir`, and `allow_patterns`.
Set `dataset: null` for local-input-only runs.

Default: `null`.
```

## Policy Fields

```{option} heuristic_filters

Approved filter policy block.

- `approved_policy`: path of a promoted policy file. A candidate policy from `curate/profile` carries `approved: false` and is refused unless `allow_unvalidated_policy` is set.
- `allow_unvalidated_policy`: apply a policy that does not meet the approval contract. The run logs a warning naming the policy and records `override_unvalidated` in the manifest.
- `langpack_dir`: root of the language packs. Required when the policy uses pack-backed signals; no production pack is bundled or selected implicitly.
- `langpack_content_hash`: optional expected content hash of the pack. It pins the pack but does not replace `langpack_dir`.
- `tokenizer`: `{name, revision}` mapping, required when the policy names `token_count`.

Signal names in a policy resolve through the closed registry in `runtime/registry.py`, never through an import path.
Refer to {doc}`policy-file`.

Default: `null`.
```

```{option} mode

How the approved policy's own signals are applied.
One of `filter`, `annotate`, or `both`.
Applies only when `heuristic_filters` names a policy.

| `mode` | Rows the policy removes | Columns added |
| --- | --- | --- |
| `filter` | Rejected rows | None; scores are discarded after use |
| `annotate` | None | `__<signal>` for every row |
| `both` | Rejected rows | `__<signal>` for the survivors |

`mode` does not govern the language, word-count, or domain gates, which drop rows under every mode.
`annotate` is not a non-destructive mode when those gates are configured.

Default: `filter`.
```

## Gate Fields

```{option} language_codes

Uppercase language codes to keep.
Set `language_codes: []` to skip FastText language identification and language filtering.
When the list is non-empty, `models.fasttext_langid` must point at a FastText language identification model.

Default: `[EN]`.
```

```{option} quality_filters

Optional quality settings.
`min_langid_score` applies when language filtering is enabled.
`min_words` and `max_words` enable Curator's `WordCountFilter` and must be provided together.
They are not shipped as defaults because `WordCountFilter` splits on whitespace and therefore removes a large share of a corpus in a language written without spaces.
A run that declares a language pack is refused unless the pack declares `word_segmentation`.
Set `quality_filters: {}` to skip word-count filtering.

Default: `{min_langid_score: 0.3}`.
```

```{option} domains

Domains to keep through NeMo Curator `MultilingualDomainClassifier`.
Set `domains: []` to skip domain classification.

Default: `[]`.
```

```{option} annotate_domains

Run `MultilingualDomainClassifier` without filtering: every document receives a label and none are dropped.
Use it on a first run to obtain the domain distribution before choosing which domains to keep.
Ignored when `domains` is non-empty, which already runs the classifier.

Default: `false`.
```

```{option} domain_score_field

Optional column for Curator's complete class-probability vector, in model label order.
It is not a scalar confidence.
Absent, only the argmax label is recorded.

Default: `null`.
```

## Accounting Fields

```{option} emit_manifest

Path for the run manifest describing what the run read, wrote, and whether it finished.
`null` writes none.

Default: `null`.
```

```{option} emit_ledger

Path for the run's record accounting, consumed by `curate/audit`.
Without it an audit can detect that records are missing but cannot distinguish a record removed by a gate from one lost to a swallowed exception.
Curator reports no per-stage removal counts, so the ledger attributes such removals to `unattributed` and states so in its notes.

Default: `null`.
```

## Runtime Fields

```{option} models

Optional model and cache paths.
`fasttext_langid` is required when `language_codes` is non-empty.
`hf_cache_dir` is the Hugging Face cache directory for classifier assets.

Default: `null`.
```

```{option} ray.num_cpus

Optional Ray CPU count.
If omitted, the Lepton curate profile can provide `NEMOTRON_CURATOR_RAY_NUM_CPUS`.

Default: `null`.
```

## How the Stages Combine

Every enabled stage must accept a record for it to be written: the gates combine with logical AND.
Within a list-valued parameter such as `language_codes` or `domains`, entries combine with OR.
Gate length in one place, either `quality_filters` or the policy; declaring the same gate twice builds two stages for it, and the per-gate breakdown in the ledger no longer reconciles with the removal measured on disk.

## Minimal Local Configuration

```yaml
language_codes: []
domains: []
text_field: text
input_glob: ./data/**/*.jsonl
output_dir: ./output/curated-jsonl
dataset: null
models: {}
quality_filters: {}
```

## Filtered Configuration

```yaml
language_codes:
  - EN
domains: []
text_field: text
input_glob: ./data/**/*.jsonl
output_dir: ./output/curated-jsonl
dataset: null
models:
  fasttext_langid: ./cache/models/fasttext/lid.176.bin
  hf_cache_dir: ./cache/huggingface
quality_filters:
  min_langid_score: 0.3
  min_words: 50
  max_words: 5000
```

## Policy Configuration

```yaml
input_glob: ./output/vi/ingested/*.jsonl
output_dir: ./output/vi/filtered_jsonl
text_field: text
id_field: id
source_field: source
metadata_fields: [id, source, url]
language_codes: []
domains: []
quality_filters: {}
dataset: null
mode: both
heuristic_filters:
  approved_policy: ./output/vi/policy/approved_policy.yaml
  langpack_dir: ./src/nemotron/steps/curate/nemo_curator/data/langpacks
emit_manifest: ./output/vi/filtered_jsonl/run_manifest.json
emit_ledger: ./output/vi/filtered_jsonl/curation_ledger.json
```

## Related Pages

- {doc}`policy-file`
- {doc}`flow-config`
- {doc}`../how-to/apply-a-known-policy`
- [`src/nemotron/steps/curate/nemo_curator/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/README.md)
