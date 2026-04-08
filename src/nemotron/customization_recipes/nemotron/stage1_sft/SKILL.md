# SKILL: Stage 1 -- Supervised Fine-Tuning (SFT)

## Purpose

Fine-tune the CPT checkpoint (or base model) for instruction following in the target language/domain. This stage includes optional synthetic data generation (SDG) using NVIDIA DataDesigner when real instruction data is scarce.

## When to Use

Always run this stage after CPT (stage0) or as the first stage if skipping CPT. SFT transforms a language model into an instruction-following assistant.

Choose SDG when:
- You have fewer than 10K real instruction examples in the target language/domain
- You need diverse instruction coverage across multiple task types
- You want to bootstrap capability before collecting real user data

Skip SDG when:
- You have >50K high-quality real instruction pairs
- You have an existing SFT dataset in the target language

## Sub-Stages

### Sub-Stage 1a: Data Preparation

Two paths depending on data availability:

**Path A: Synthetic Data Generation (SDG)**
1. Configure DataDesigner with domain/language specifications
2. Generate diverse instruction-response pairs using NIM API
3. Filter generated data for quality (format, coherence, relevance)
4. Convert to chat format (OpenAI messages schema)

**Path B: Real Data Preparation**
1. Load instruction datasets (HuggingFace, local JSONL)
2. Apply chat template (Jinja2 template specific to model family)
3. Filter for quality (length, formatting, deduplication)

**Common (both paths):**
4. Tokenize and pack sequences into Parquet shards using `nemotron.data_prep`
5. Split into train/validation sets

### Sub-Stage 1b: SFT Training

Fine-tune the model using packed sequence training with Megatron-Bridge.

## Prerequisites

| Prerequisite | Description |
|-------------|-------------|
| CPT checkpoint | From stage0_cpt (or base model if skipping CPT) |
| NVIDIA_API_KEY | Required for SDG via NIM API |
| Instruction data | Real data OR SDG config for synthetic generation |
| GPU cluster | Same as CPT (2+ nodes x 8 GPUs for Nano) |
| Container | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` |

## Synthetic Data Generation (SDG)

### Using DataDesigner

```bash
python src/nemotron/customization_recipes/nemotron/stage1_sft/run_sdg.py \
  --config src/nemotron/customization_recipes/nemotron/stage1_sft/config/sdg/default.yaml \
  domain=medical \
  language=hi \
  num_samples=50000 \
  output_dir=/data/sdg_output
```

### SDG Config (`config/sdg/default.yaml`)

```yaml
output_dir: ./output/sdg
output_prefix: synthetic_data
num_records: 100
preview_only: false

# Model for generation
model:
  name: openai/gpt-oss-20b
  alias: gpt-oss
  temperature: 1.0
  top_p: 0.9
  max_tokens: 4096

# Locale for person sampler
locale: en_US

# Columns to export as JSONL
jsonl_columns:
  - generated_conversation
  - rewritten_conversation
```

The underlying `SDGConfig` dataclass (in `data_prep/sdg.py`) uses these fields: `output_dir`, `seed_dataset`, `num_records`, `system_prompt`, `user_prompt`, `column_name`, `column_type`, `output_format`, `model_configs`, `model_alias`. The YAML above is the actual file; the `run_sdg.py` script maps it to the SDG pipeline.

### SDG Output Format

```jsonl
{"messages": [{"role": "system", "content": "You are a medical assistant..."}, {"role": "user", "content": "<question in Hindi>"}, {"role": "assistant", "content": "<answer in Hindi>"}]}
```

## Data Preparation (Tokenize + Pack)

```bash
python src/nemotron/customization_recipes/nemotron/stage1_sft/run_data_prep.py \
  --config src/nemotron/customization_recipes/nemotron/stage1_sft/config/data_prep/default.yaml \
  output_dir=/data/sft_prepared
```

### Data Prep Config (`config/data_prep/default.yaml`)

```yaml
mode: sft

# Output directory for packed .npy files
output_dir: ./output/sft_data_prep

# Input source (choose one)
hf_dataset: HuggingFaceH4/ultrachat_200k
hf_subset: null
hf_split: train_sft
input_path: null                          # Local file or directory -- overrides hf_dataset

# Tokenizer
tokenizer_model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# Packing
pack_size: 8192
packing_algorithm: first_fit_decreasing

# Train / validation / test split
train_ratio: 0.9
valid_ratio: 0.05
test_ratio: 0.05

# Message format
messages_field: messages
conversations_field: null                 # For ShareGPT-format datasets

# Processing
seed: 42
recursive: true
max_samples: null                         # Limit for quick tests (null = all)

# Thinking tokens (optional)
enable_thinking: false
truncate_history_thinking: true
thinking_start_token: "<think>"
thinking_end_token: "</think>"
```

The underlying `SFTConfig` dataclass (in `data_prep/tokenize_pack.py`) has matching fields. Tokenization, chat template application, thinking-token handling, and packing are all delegated to the production `nemotron.data_prep` pipeline (`run_sft_pipeline`). When `enable_thinking` is true, the `nano3` chat template is used, which natively supports `reasoning_content` and history truncation.

**Output:** Packed Parquet shards in `output_dir/runs/<hash>/` compatible with Megatron-Bridge training.

## SFT Training Config (`config/default.yaml`)

```yaml
run:
  data: sft-data:latest
  model: cpt-model:latest               # CPT checkpoint from stage0
  env:
    container: nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
    mounts:
      - ${auto_mount:git+https://github.com/NVIDIA/Megatron-LM.git@<commit>,/opt/megatron-lm}
      - ${auto_mount:git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@<branch>,/opt/Megatron-Bridge}

recipe:
  _target_: megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_finetune_config
  packed_sequence: true
  peft: null                             # null = full SFT; "lora" for LoRA

dataset:
  nano3_packed_sft_dir: ${art:data,path}
  seq_length: ${art:data,pack_size}
  packed_sequence_specs:
    packed_sequence_size: ${art:data,pack_size}

train:
  train_iters: 1700                      # Adjust based on dataset size
  global_batch_size: 4                   # Small GBS for SFT (avoid overfitting)

model:
  seq_length: ${art:data,pack_size}
  pipeline_model_parallel_size: 1
  tensor_model_parallel_size: 4
  context_parallel_size: 2
  calculate_per_token_loss: true

scheduler:
  lr: 5e-6                               # Lower than CPT
  lr_warmup_iters: 4
  lr_decay_style: cosine

logger:
  log_interval: 10
  wandb_project: ${run.wandb.project}
  wandb_entity: ${run.wandb.entity}

checkpoint:
  save: /results/sft_checkpoint
  save_interval: 100
  pretrained_checkpoint: ${art:model,path}
  finetune: true                         # Skip loading optimizer state
```

### Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `train.train_iters` | 1700 | 500-5000 | ~2-3 epochs over dataset |
| `train.global_batch_size` | 4 | 2-16 | Small to prevent overfitting |
| `scheduler.lr` | 5e-6 | 1e-6 to 2e-5 | Lower than CPT LR |
| `dataset.seq_length` | 4096 | 2048-8192 | Must match pack_size |
| `recipe.peft` | null | null/"lora" | null = full SFT, "lora" = parameter-efficient |
| `checkpoint.finetune` | true | true/false | true = skip optimizer state from pretrained |

### LoRA vs Full SFT Decision

| Criterion | Full SFT | LoRA |
|-----------|----------|------|
| Training data | >10K examples | <10K examples |
| GPU budget | 2+ nodes | Single node possible |
| Quality target | Maximum quality | Good quality, faster iteration |
| Forgetting risk | Higher (mitigated by blend) | Lower (fewer params updated) |

To use LoRA, set `recipe.peft: lora` in the config.

## Execution

### Local

```bash
nemotron customize sft -c default
```

### Slurm

```bash
nemotron customize sft -c default --run MY-CLUSTER \
  train.train_iters=2000 \
  checkpoint.pretrained_checkpoint=/results/cpt_checkpoint
```

### Dry Run (Preview Config)

```bash
nemotron customize sft -c default --dry-run
```

## Data Blend Strategy

| Component | Weight | Purpose |
|-----------|--------|---------|
| Domain-specific instruction data (real or SDG) | 50-60% | Primary task capability |
| General instruction data (ChatQA, etc.) | 25-35% | Broad instruction following |
| Safety/alignment data (Aegis, etc.) | 5-10% | Prevent harmful outputs |
| Code instruction data | 0-10% | Maintain code capability |

## How to Verify Success

1. **Training loss**: Should decrease and converge. Final loss typically 0.5-1.5 for SFT.
   - If loss does not decrease below 2.0: check data format, verify chat template applied correctly
   - If loss drops to <0.1: likely overfitting -- reduce iterations or increase data

2. **Validation loss**: Should track training loss without diverging.
   - If val loss increases while train loss decreases: overfitting -- stop training

3. **Qualitative check**: Generate responses to domain-specific prompts.
   ```
   Input: "What are the symptoms of diabetes?" (in target language)
   Expected: Coherent, factual, properly formatted response in target language
   ```

4. **Format compliance**: Verify model follows the chat template correctly.
   - Response should start with assistant turn
   - No system prompt leakage
   - Proper turn boundaries

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Loss NaN after loading CPT checkpoint | Checkpoint format mismatch or corrupt | Verify `finetune: true` is set; check checkpoint integrity |
| Model responds in wrong language | Insufficient target-language data in SFT blend | Increase domain data weight to 70%, add language-specific system prompt |
| Repetitive/generic responses | Overfitting on limited SDG data | Increase SDG diversity (more task types), reduce train_iters |
| Chat format broken (no turn boundaries) | Wrong chat template or packing error | Verify `chat_template` matches model family, check packed Parquet samples |
| SDG API rate limited | Too many concurrent requests | Reduce `batch_size`, add retry logic, use multiple API keys |
| Packed Parquet shards empty | Tokenizer mismatch or all samples filtered | Check `min_response_length`, verify tokenizer produces tokens |

## Artifacts Produced

| Artifact | Type | Path | Consumed By |
|----------|------|------|-------------|
| SDG dataset | JSONL | `sdg_output/` | Data prep (this stage) |
| Packed SFT data | `SFTDataArtifact` | `sft_prepared/splits/` | Training (this stage) |
| SFT checkpoint | `ModelArtifact` | `checkpoint.save/` | stage2_rl |
| Training logs | W&B/TensorBoard | W&B project | Analysis |
