# SKILL: Stage 0 -- Data Preparation & Curation

## Purpose

Prepare, curate, and transform raw data before any training stage. This is the upstream stage in the customization pipeline -- it produces clean, translated, or augmented datasets consumed by stages 1-6.

Currently supports:
- **Translation** (sub-stage 0a): Translate text corpora or chat datasets between languages using NeMo Curator's TranslationPipeline with LLM, Google Cloud, AWS, or NMT backends.

Planned (future sub-stages):
- Data acquisition and download
- Language filtering and deduplication
- Quality filtering and scoring
- Format conversion (JSONL, Parquet, chat templates)

## When to Use

Decision tree:

1. Is the target language different from the source language?
   - **YES** --> Run **sub-stage 0a: Translation**
   - **NO** --> Skip to stage 1 (CPT) or stage 2 (SFT)

2. Do you need to translate training data (not just benchmarks)?
   - **YES** --> Use this stage (stage 0 translation)
   - **NO, only benchmarks** --> Use stage 4 BYOB translation instead

3. Is the data already clean, filtered, and in the right format?
   - **YES** --> Skip stage 0 entirely
   - **NO** --> Run the appropriate sub-stage(s)

Skip this stage if:
- All data is already in the target language
- You are working with English-only general-domain data
- Data has already been curated upstream (e.g., from NeMo Curator directly)

## Inputs Required

Before running this stage, confirm these with the user:

| Input | Required? | Default | Notes |
|-------|-----------|---------|-------|
| Source language | Yes | `en` | Ask: "What is the source language of your data? (ISO 639-1 code, e.g., en, zh, ja)" |
| Target language | Yes | `hi` | Ask: "What language should the data be translated to?" |
| Input data path | Yes | `/workspace/data/source` | Ask: "Where is your source data? (directory with JSONL/Parquet files)" |
| Data format | Yes | Plain text (`text` field) | Ask: "Is your data plain text or chat messages? Which field contains the text?" |
| Translation backend | Yes | `llm` | Ask: "Which translation backend? (llm, google, aws, nmt)" |
| API key / credentials | If using LLM or cloud | `NVIDIA_API_KEY` env var | Ask: "Do you have an API key set? (NVIDIA_API_KEY for NIM, GOOGLE_APPLICATION_CREDENTIALS for Google, AWS credentials for AWS)" |
| LLM model | If backend=llm | `mistralai/mistral-small-3.1-24b-instruct` | Ask: "Which LLM model for translation?" |
| Quality evaluation needed? | No | `false` | Ask: "Should we run FAITH quality scoring on translations?" |
| Output directory | Yes | `/workspace/data/translated` | Ask: "Where should translated data be written?" |

If any required input is missing, ask the user before proceeding.

## Sub-Stage 0a: Translation

Translate text corpora using NeMo Curator's TranslationPipeline. The pipeline handles segmentation, translation, reassembly, and optional FAITH quality evaluation.

### Pipeline

1. **Load** source data from JSONL or Parquet files
2. **Skip** already-translated rows (if `skip_translated: true`)
3. **Segment** documents into translatable chunks (coarse or fine mode)
4. **Translate** each segment via the configured backend (LLM, Google, AWS, NMT)
5. **Reassemble** translated segments back into full documents
6. **Evaluate** translation quality with FAITH scores (optional)
7. **Filter** low-quality translations below threshold (optional)
8. **Save** output to the configured directory

### Prerequisites

| Prerequisite | Description |
|-------------|-------------|
| NeMo Curator | Translation pipeline library (`pip install nemo-curator`) |
| Input data | JSONL or Parquet files with a text column |
| API credentials | Depends on backend: `NVIDIA_API_KEY` (LLM/NIM), Google Cloud credentials, AWS credentials, or local NMT server |
| Network access | For cloud/API-based backends |

### Execution

#### Local (Development)

```bash
nemotron customize translate -c default \
  translation.source_lang=en \
  translation.target_lang=hi \
  translation.backend=llm \
  translation.server.model=mistralai/mistral-small-3.1-24b-instruct
```

#### With NIM endpoint

```bash
export NVIDIA_API_KEY=nvapi-...
nemotron customize translate -c default \
  translation.source_lang=en \
  translation.target_lang=hi \
  translation.backend=llm \
  translation.server.url=https://integrate.api.nvidia.com/v1 \
  translation.server.model=mistralai/mistral-small-3.1-24b-instruct
```

#### With Google Cloud Translation

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
nemotron customize translate -c default \
  translation.backend=google \
  translation.google.project_id=my-gcp-project \
  translation.google.api_version=v3
```

#### With AWS Translate

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
nemotron customize translate -c default \
  translation.backend=aws \
  translation.aws.region=us-east-2
```

#### With local NMT server

```bash
nemotron customize translate -c default \
  translation.backend=nmt \
  translation.nmt.server_url=http://localhost:5000 \
  translation.nmt.batch_size=64
```

#### With FAITH quality evaluation

```bash
nemotron customize translate -c default \
  translation.source_lang=en \
  translation.target_lang=hi \
  translation.backend=llm \
  translation.faith_eval.enabled=true \
  translation.faith_eval.threshold=2.5 \
  translation.faith_eval.filter_enabled=true
```

#### Direct script execution

```bash
python src/nemotron/customization_recipes/nemotron/stage0_data_prep/run_translate.py \
  --config src/nemotron/customization_recipes/nemotron/stage0_data_prep/config/translate/default.yaml \
  translation.target_lang=fr \
  translation.backend=llm
```

## Config Reference

### Translation Config (`config/translate/default.yaml`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `translation.source_lang` | str | `en` | ISO 639-1 source language code |
| `translation.target_lang` | str | `hi` | ISO 639-1 target language code |
| `translation.backend` | str | `llm` | Translation backend: `llm`, `google`, `aws`, `nmt` |
| `translation.segmentation_mode` | str | `coarse` | Segmentation: `coarse` (line-level) or `fine` (sentence-level) |
| `translation.text_field` | str | `text` | Input column containing source text |
| `translation.output_field` | str | `translated_text` | Output column for translated text |
| `translation.server.url` | str | `https://integrate.api.nvidia.com/v1` | LLM server URL (for `llm` backend) |
| `translation.server.model` | str | `mistralai/mistral-small-3.1-24b-instruct` | LLM model identifier |
| `translation.server.api_key` | str | `${oc.env:NVIDIA_API_KEY,}` | API key (resolved from env) |
| `translation.max_concurrent_requests` | int | `64` | Max parallel translation requests |
| `translation.skip_translated` | bool | `false` | Skip rows that already have translations |
| `translation.faith_eval.enabled` | bool | `false` | Enable FAITH quality evaluation |
| `translation.faith_eval.threshold` | float | `2.5` | Minimum FAITH average score (1-5 scale) |
| `translation.faith_eval.segment_level` | bool | `false` | Score individual segments (more granular) |
| `translation.faith_eval.filter_enabled` | bool | `true` | Drop rows below threshold |
| `translation.output_mode` | str | `both` | Output format: `replaced`, `raw`, or `both` |
| `translation.preserve_segment_pairs` | bool | `true` | Keep source/target segment pairs in metadata |
| `translation.merge_scores` | bool | `true` | Fold `faith_*` scores into `translation_metadata` JSON |
| `translation.google.project_id` | str | `""` | GCP project ID (for `google` backend) |
| `translation.google.api_version` | str | `v2` | Google Translate API version: `v2` or `v3` |
| `translation.aws.region` | str | `${oc.env:AWS_DEFAULT_REGION,us-east-2}` | AWS region (for `aws` backend) |
| `translation.nmt.server_url` | str | `http://localhost:5000` | NMT server URL (for `nmt` backend) |
| `translation.nmt.batch_size` | int | `32` | NMT batch size |
| `translation.input_path` | str | `/workspace/data/source` | Input data directory |
| `translation.output_dir` | str | `/workspace/data/translated` | Output data directory |

## How to Verify Success

1. **Row counts match**: The translated output should have the same number of rows as the input (minus any filtered by FAITH threshold).
   ```bash
   wc -l /workspace/data/source/*.jsonl
   wc -l /workspace/data/translated/*.jsonl
   ```

2. **Sample translations**: Spot-check a few translations for quality.
   ```bash
   head -5 /workspace/data/translated/output.jsonl | python -m json.tool
   ```
   Check: Is the translated text in the correct target language? Is it coherent? Does it preserve the meaning of the source?

3. **FAITH scores** (if enabled): Check the average FAITH scores across the dataset.
   - Target: `faith_avg >= 2.5` for acceptable quality (scale is 1-5)
   - Scores below 2.0 indicate significant quality issues
   - Review the `translation_metadata` column for per-document details

4. **Segment pairs** (if `preserve_segment_pairs: true`): Verify that source/target pairs are aligned.
   ```python
   import json
   with open("/workspace/data/translated/output.jsonl") as f:
       row = json.loads(f.readline())
       metadata = json.loads(row.get("translation_metadata", "{}"))
       print(json.dumps(metadata.get("segment_pairs", [])[:3], indent=2, ensure_ascii=False))
   ```

5. **No empty translations**: Check for rows where translation failed.
   ```bash
   grep '"translated_text": ""' /workspace/data/translated/output.jsonl | wc -l
   ```

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `NVIDIA_API_KEY not set` or 401 errors | Missing or invalid API key | Set `export NVIDIA_API_KEY=nvapi-...` or pass via `translation.server.api_key` |
| `google-cloud-translate not installed` | Missing Google Cloud SDK | `pip install google-cloud-translate` and set `GOOGLE_APPLICATION_CREDENTIALS` |
| `boto3 not installed` | Missing AWS SDK | `pip install boto3` and configure AWS credentials |
| `nemo_curator not installed` | Missing NeMo Curator library | `pip install nemo-curator` |
| Connection timeout / refused | Server unreachable | Check `translation.server.url`, verify network access, check firewall rules |
| Low FAITH scores (< 2.0) | Poor translation quality | Try a larger/better model, switch backends, or use fine-grained segmentation (`segmentation_mode: fine`) |
| Many empty translations | Backend errors or rate limiting | Reduce `max_concurrent_requests`, check API quotas, review logs for error details |
| OOM on large datasets | Too many rows loaded at once | Process data in smaller batches by splitting input files |
| Slow translation | Low concurrency or large segments | Increase `max_concurrent_requests`, use `segmentation_mode: coarse` for fewer API calls |
| Duplicate translations | Re-running without clearing output | Set `skip_translated: true` to skip already-translated rows, or clear the output directory |
| Chat message structure lost | Using plain text mode on chat data | Set `translation.text_field` to the correct field, consider message reconstruction options |

## Artifacts Produced

| Artifact | Type | Path | Consumed By |
|----------|------|------|-------------|
| Translated data (JSONL/Parquet) | `TranslatedDataArtifact` | `translation.output_dir/` | stage1_cpt, stage2_sft |
| Translation metadata | JSON column | Embedded in output rows | Quality analysis |
| FAITH scores | Float columns | Embedded in output rows | Quality filtering |
| Segment pairs | JSON column | Embedded in output rows | Alignment analysis |
