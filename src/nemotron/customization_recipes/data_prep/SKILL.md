# SKILL: Shared Data Preparation Utilities

## Purpose

Provide data acquisition, filtering, transformation, and formatting utilities shared across all model families in the customization pipeline. This module wraps `nemotron.data_prep` (the core data prep library) and NeMo Curator with customization-specific workflows.

## Capabilities

| Capability | Tool | Input | Output |
|------------|------|-------|--------|
| Data download | `nemotron.data_prep.stages.download` | HF dataset ID, S3 path, or local path | Raw data files |
| Language filtering | NeMo Curator `FastTextLangId` | Raw text | Language-filtered text |
| Quality filtering | NeMo Curator quality classifiers | Raw text | Quality-filtered text |
| Deduplication | NeMo Curator exact/fuzzy/substring dedup | Filtered text | Deduplicated text |
| Translation | NIM Translation API | Source-language text | Target-language text |
| Tokenization (pretrain) | `nemotron.data_prep.api.run_pretrain_pipeline()` | Filtered text | Megatron bin/idx files |
| Tokenization (SFT) | `nemotron.data_prep.api.run_sft_pipeline()` | Chat-format JSONL | Packed Parquet shards |
| JSONL conversion (RL) | `nemotron.data_prep.recipes.rl` | Raw data | JSONL prompt files |
| Data blending | `nemotron.data_prep.blend.DataBlend` | Multiple data sources | Weighted blend specification |

## Core Library: nemotron.data_prep

Location: `src/nemotron/data_prep/`

### Public API

```python
from nemotron.data_prep.api import run_pretrain_pipeline, run_sft_pipeline
from nemotron.data_prep.blend import DataBlend
```

### Pretrain Pipeline (bin/idx)

Tokenizes text data into Megatron-format indexed binary files for continued pretraining.

```python
from nemotron.data_prep import DataBlend, run_pretrain_pipeline

blend = DataBlend.load("blend.json")
result = run_pretrain_pipeline(
    blend=blend,
    output_dir="/data/output",
    tokenizer="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    num_shards=128,
)
print(f"Total tokens: {result.total_tokens:,}")
```

### SFT Pipeline (Packed Parquet)

Converts chat-format data into packed Parquet shards with loss masking for SFT training.

```python
from nemotron.data_prep import DataBlend, run_sft_pipeline

blend = DataBlend.load("sft_blend.json")
result = run_sft_pipeline(
    blend=blend,
    output_dir="/data/output",
    tokenizer="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    num_shards=64,
    chat_template="nano3",
    pack_size=4096,
)
print(f"Total sequences: {result.total_sequences:,}")
```

### Three-Phase Pattern

All data prep scripts follow the same pattern:

```python
# Phase 1: Setup (deterministic planning)
items, context, tokenizer = setup_pretrain_run(blend, output_dir, tokenizer_name, ...)

# Phase 2: Execute (xenna pipeline)
if items:
    ctx = PipelineContext(...)
    stages = [
        StageSpec(PlanStage(...), num_workers=1),
        StageSpec(DownloadStage(...), num_workers_per_node=1),
        StageSpec(BinIdxTokenizationStage(...), slots_per_actor=1),
    ]
    spec = pipelines_v1.PipelineSpec(input_data=items, stages=stages, ...)
    pipelines_v1.run_pipeline(spec)

# Phase 3: Finalize (aggregate results)
result = finalize_pretrain_run(context, blend, output_dir)
```

### Data Blend Specification

```json
{
  "sources": [
    {
      "dataset": "ai4bharat/sangraha",
      "subset": "hi",
      "weight": 0.7,
      "text_field": "text"
    },
    {
      "dataset": "allenai/c4",
      "subset": "en",
      "weight": 0.2
    }
  ]
}
```

Blends are loaded with `DataBlend.load("blend.json")` and control the relative proportion of each data source in the final training data.

## NeMo Curator Integration

### Language Filtering

```python
from nemo_curator.filters import FastTextLangId
from nemo_curator import ScoreFilter

lang_filter = ScoreFilter(FastTextLangId(), threshold=0.7, filter_by="hi")
filtered_dataset = lang_filter(raw_dataset)
```

### Quality Filtering

```python
from nemo_curator.filters import FastTextQualityFilter
from nemo_curator import ScoreFilter

quality_filter = ScoreFilter(FastTextQualityFilter(), threshold=0.5)
filtered_dataset = quality_filter(filtered_dataset)
```

### Deduplication

```python
from nemo_curator import FuzzyDuplicates, FuzzyDuplicatesConfig

fuzzy_config = FuzzyDuplicatesConfig(
    seed=42,
    num_buckets=20,
    hashes_per_bucket=13,
    use_64_bit_hash=False,
    buckets_per_shuffle=5,
    false_positive_check=True,
    num_anchors=2,
    jaccard_threshold=0.8,
)
fuzzy_dedup = FuzzyDuplicates(config=fuzzy_config)
deduplicated = fuzzy_dedup(filtered_dataset)
```

## Translation Workflow

For translating English domain data to a target language:

```python
# Using NIM Translation API
import requests

def translate_batch(texts, source_lang="en", target_lang="hi"):
    """Translate a batch of texts using NIM API."""
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/translate",
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        json={
            "texts": texts,
            "source_language": source_lang,
            "target_language": target_lang,
        },
    )
    return response.json()["translations"]
```

## Output Formats

| Format | Generated By | Used By | Files |
|--------|-------------|---------|-------|
| bin/idx | `run_pretrain_pipeline()` | CPT training (stage0) | `.bin`, `.idx` pairs + `blend.json` |
| Packed Parquet | `run_sft_pipeline()` | SFT training (stage1) | `.parquet` files in `splits/train/`, `splits/valid/` |
| JSONL | `run_rl_pipeline()` / direct | RL training (stage2), BYOB (stage3) | `.jsonl` files |

## Chat Templates

Location: `src/nemotron/data_prep/templates/`

| Template | Model Family | File |
|----------|-------------|------|
| nano3 | Nemotron Nano3 | `nano3.jinja` |
| llama3 | Llama 3.x | (planned) |
| qwen2 | Qwen 2.x | (planned) |

Templates are Jinja2 files that format chat messages into the model-specific prompt format. Specified via the `chat_template` parameter in SFT data prep.

## Data Prep Script Pattern

Each customization stage that needs data prep has a `data_prep.py` script. These scripts:

1. Parse config from `config/data_prep/default.yaml`
2. Call the appropriate `nemotron.data_prep` recipe
3. Produce artifacts tracked by `nemotron.kit`

Example invocation:
```bash
python src/nemotron/customization_recipes/nemotron/stage0_cpt/run_data_prep.py \
  --config src/nemotron/customization_recipes/nemotron/stage0_cpt/config/data_prep/default.yaml \
  output_dir=/data/prepared \
  tokenizer=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
```

## Artifact Tracking

Data prep integrates with `nemotron.kit` for artifact lineage:

```python
import nemotron.kit as kit

# After data prep completes
artifact = kit.PretrainDataArtifact(
    path=Path("/data/prepared"),
    total_tokens=result.total_tokens,
    run_hash=result.run_hash,
)
artifact.save(name="cpt-data")
```

Artifacts are referenced in training configs via `${art:data,path}` URI syntax.

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Download fails (403) | Gated dataset, missing HF_TOKEN | Set `HF_TOKEN` environment variable, accept dataset license on HF |
| Tokenizer not found | Wrong tokenizer name | Use full HF model name (e.g., `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`) |
| Empty output (0 shards) | All data filtered out | Lower quality/language thresholds, check input data format |
| Slow pipeline | I/O bottleneck or too few workers | Increase `num_shards`, use local SSD, check network bandwidth |
| OOM during packing | Pack size too large | Reduce `pack_size` or process in smaller batches |
| Receipt errors on resume | Corrupt intermediate state | Delete receipts directory and re-run from scratch |
| Blend weights don't sum to 1.0 | Weights are relative, not absolute | Weights are normalized automatically; any positive values work |
