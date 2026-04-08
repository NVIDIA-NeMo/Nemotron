# SKILL: Stage 0 -- Continued Pretraining (CPT)

## Purpose

Inject new language and/or domain knowledge into a base Nemotron model by continued pretraining on target-language or domain-specific text corpora. This is the foundational stage of the customization pipeline -- all subsequent stages build on the CPT checkpoint.

## When to Use

- Adapting to a new language not well-represented in the base model's training data
- Specializing for a technical domain (medical, legal, financial, scientific)
- Both language + domain adaptation (e.g., Hindi medical)

Skip this stage if:
- The target language is English and the domain is general
- You only need instruction-following capability (go to stage1_sft)
- The base model already performs well on your target distribution

## Sub-Stages

CPT has two sub-stages that run sequentially:

### Sub-Stage 0a: Data Acquisition and Preparation

Acquire, filter, and tokenize target-language/domain corpora into Megatron bin/idx format.

**Pipeline:**
1. **Download** raw corpora from HuggingFace, S3, or local sources
2. **Language filter** using NeMo Curator language classifiers (fasttext-based)
3. **Quality filter** using NeMo Curator quality classifiers (heuristic + model-based)
4. **Deduplication** using NeMo Curator exact/fuzzy/substring dedup
5. **Optional translation** of high-quality English domain data to target language (NIM Translation API)
6. **Tokenize** to Megatron bin/idx format using `nemotron.data_prep`
7. **Blend** multiple data sources with specified ratios

### Sub-Stage 0b: CPT Training

Continue pretraining the base model on the prepared data using Megatron-Bridge.

## Prerequisites

| Prerequisite | Description |
|-------------|-------------|
| Base model | Nemotron checkpoint (e.g., `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`) |
| Raw corpora | Text data in target language/domain (JSONL, Parquet, or HF dataset) |
| GPU cluster | Minimum 2 nodes x 8 GPUs for Nano; 8+ nodes for Super |
| NeMo Curator | For data filtering and dedup (`pip install nemo-curator`) |
| HF_TOKEN | For downloading gated models/datasets |
| Container | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` or model-specific |

## Data Acquisition

### Using NeMo Curator

```python
# Example: Filter a HuggingFace dataset for Hindi language + quality
from nemo_curator import ScoreFilter, Sequential
from nemo_curator.filters import FastTextLangId, FastTextQualityFilter
from nemo_curator.utils.distributed_utils import get_client

client = get_client(cluster_type="gpu")

pipeline = Sequential([
    FastTextLangId(language="hi", threshold=0.7),
    FastTextQualityFilter(threshold=0.5),
])

dataset = load_dataset("ai4bharat/sangraha")
filtered = pipeline(dataset)
filtered.to_jsonl("/data/hindi_filtered/")
```

### Common Data Sources by Language

| Language | Datasets | Notes |
|----------|----------|-------|
| Hindi | `ai4bharat/sangraha`, `oscar-corpus/OSCAR-2301` | Filter for quality |
| Thai | `oscar-corpus/OSCAR-2301`, CC-100 | |
| Arabic | `oscar-corpus/OSCAR-2301`, `allenai/c4` | |
| Medical (EN) | PubMed, PMC-OA, medical textbooks | May need translation |
| Legal (EN) | Pile of Law, legal corpora | Domain-specific tokenization |

### Data Blend Strategy

| Component | Ratio | Purpose |
|-----------|-------|---------|
| Target language/domain text | 60-70% | Primary knowledge injection |
| English general text | 15-25% | Prevent catastrophic forgetting |
| Code | 5-10% | Maintain reasoning capability |
| English domain text | 5-10% | Cross-lingual domain transfer |

## Tokenization and Data Prep

```bash
# Prepare CPT data using the data_prep pipeline
python src/nemotron/customization_recipes/nemotron/stage0_cpt/run_data_prep.py \
  --config src/nemotron/customization_recipes/nemotron/stage0_cpt/config/data_prep/default.yaml \
  output_dir=/data/cpt_prepared \
  tokenizer=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
  num_shards=128
```

The data_prep script uses the three-phase pattern from `nemotron.data_prep`:
1. `setup_pretrain_run()` -- create work items, plan shard assignments
2. xenna pipeline: PlanStage -> DownloadStage -> BinIdxTokenizationStage
3. `finalize_pretrain_run()` -- scan receipts, generate blend.json

**Output:** Directory containing `.bin`/`.idx` file pairs + `blend.json` manifest.

## Config Reference

### Data Prep Config (`config/data_prep/default.yaml`)

```yaml
# Output directory for filtered JSONL artifacts
output_dir: ./output/cpt_data_prep

# Data source -- download from HF or use a local directory
source:
  hf_dataset: nvidia/Nemotron-Pretraining-Dataset-sample
  hf_subset: Nemotron-CC-High-Quality
  hf_split: train
  local_path: null           # Set to override HF download
  num_records: null           # Limit for quick tests (null = all)

# Language filtering via fastText lid.176.bin
language_filter:
  enabled: true
  language_codes: []          # e.g. [EN, HI] -- empty keeps all languages
  min_score: 0.3

# Domain classification via nvidia/multilingual-domain-classifier
domain_classifier:
  enabled: true
  domains: []                 # e.g. [Science, Technology] -- empty keeps all
  max_chars: 6000
  batch_size: 256

# Optional translation step
translate:
  enabled: false
  source_language: en-US
  target_language: hi-IN
  model: openai/gpt-oss-120b

# Tokenizer (for downstream tokenization reference)
tokenizer:
  model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# Processing
seed: 42
text_field: text
```

The underlying `AcquireConfig` dataclass (in `data_prep/acquire.py`) uses flat fields: `download_dir`, `output_dir`, `record_format`, `url_limit`, `record_limit`, `chat_template_model`, `domain_classifier_model`, `domain_classifier_batch_size`, `domain_text_field`, `lid_model_path`, `lid_text_field`, `sources`. The YAML config above is parsed by the `run_data_prep.py` script and mapped to these fields.

### CPT Training Config (`config/default.yaml`)

```yaml
run:
  data: cpt-data:latest                # Data artifact (bin/idx blends)
  model: null                          # Base model (downloaded from HF if null)
  env:
    container: nvcr.io/nvidia/nemo:25.11.nemotron_3_nano

recipe:
  _target_: megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_pretrain_config
  per_split_data_args_path: ${art:data,path}/blend.json

train:
  train_iters: 10000                   # Adjust based on data volume
  global_batch_size: 256               # Tokens per step = GBS * seq_length
  micro_batch_size: 1

model:
  seq_length: 4096
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 1
  context_parallel_size: 2

optimizer:
  lr: 1e-5                             # Lower than pretrain (avoid forgetting)
  min_lr: 1e-6
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  clip_grad: 1.0

scheduler:
  lr_decay_style: cosine
  lr_warmup_iters: 100

logger:
  log_interval: 10
  wandb_project: ${run.wandb.project}
  wandb_entity: ${run.wandb.entity}

checkpoint:
  save: /results/cpt_checkpoint
  save_interval: 1000
  load: null                           # Set to resume from a checkpoint
```

### Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `train.train_iters` | 10000 | 5000-50000 | ~10B tokens with GBS=256, seq=4096 |
| `train.global_batch_size` | 256 | 64-512 | Higher = smoother gradients, more GPU memory |
| `optimizer.lr` | 1e-5 | 5e-6 to 5e-5 | Lower for less forgetting |
| `optimizer.min_lr` | 1e-6 | 1e-7 to 1e-5 | Cosine decay target |
| `optimizer.weight_decay` | 0.01 | 0.0-0.1 | Regularization |
| `model.seq_length` | 4096 | 2048-8192 | Match base model context length |
| `model.tensor_model_parallel_size` | 4 | 1-8 | Increase for larger models |
| `checkpoint.save_interval` | 1000 | 500-2000 | Save frequently for long runs |

## Execution

### Local (Development)

```bash
nemotron customize cpt -c default
# or directly:
python src/nemotron/customization_recipes/nemotron/stage0_cpt/run_cpt.py \
  --config src/nemotron/customization_recipes/nemotron/stage0_cpt/config/default.yaml
```

### Slurm (Production)

```bash
nemotron customize cpt -c default --run MY-CLUSTER
nemotron customize cpt -c default --batch MY-CLUSTER  # Detached
```

### With Overrides

```bash
nemotron customize cpt -c default --run MY-CLUSTER \
  train.train_iters=20000 \
  optimizer.lr=5e-6 \
  checkpoint.save=/results/my_cpt_run
```

## How to Verify Success

1. **Training loss curve**: Should decrease steadily. Check W&B or logs.
   - If loss plateaus early: LR may be too low, or data may be too easy/repetitive
   - If loss spikes: LR too high, bad data batch, or numerical instability

2. **Validation perplexity**: On a held-out set of target-language text.
   - Target: perplexity should decrease 20-50% from base model on target language
   - Monitor English validation perplexity -- should not increase more than 5-10%

3. **Quick sanity check**: Generate text in target language using the checkpoint.
   ```python
   # After loading checkpoint
   prompt = "<target-language greeting or domain prompt>"
   output = model.generate(prompt, max_length=200)
   # Check: Is output in the correct language? Is it coherent?
   ```

4. **Data quality metrics** from data prep:
   - Token count matches expected volume
   - No corrupt shards (check receipts)
   - Blend ratios match specification

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| OOM on forward pass | Model too large for GPU memory with current parallelism | Increase `tensor_model_parallel_size` or `pipeline_model_parallel_size` |
| OOM on backward pass | Activation memory too high | Enable activation checkpointing, reduce `micro_batch_size` |
| Loss NaN/Inf | Numerical instability, bad data | Reduce LR, check data for special characters/encoding issues, enable gradient clipping |
| Loss not decreasing | LR too low or data not informative | Increase LR to 5e-5, verify data is actually in target language |
| Catastrophic forgetting (English performance drops >10%) | Too aggressive adaptation | Increase English data ratio to 30%, reduce LR, reduce train_iters |
| Slow training | I/O bottleneck or suboptimal parallelism | Use bin/idx format (not JSONL), check data loading workers, verify NVLink topology |
| Tokenizer errors | Wrong tokenizer for model | Ensure tokenizer matches base model exactly |

## Artifacts Produced

| Artifact | Type | Path | Consumed By |
|----------|------|------|-------------|
| CPT data (bin/idx) | `PretrainDataArtifact` | `output_dir/` | This stage (training) |
| CPT checkpoint | `ModelArtifact` | `checkpoint.save/` | stage1_sft |
| blend.json | Manifest | `output_dir/blend.json` | Data lineage |
| Training logs | W&B/TensorBoard | W&B project | Analysis |
