# Fine-Tuning NemotronOmni on CORD-v2 Receipts — End-to-End Guide

**A step-by-step guide for fine-tuning NemotronOmni (33B MoE) to extract structured
receipt data from scanned images using [NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel).
Covers both full SFT and LoRA PEFT.**

---

## What is NemotronOmni?

NemotronOmni (`NemotronH_Nano_Omni_Reasoning_V3`) is a ~33B multimodal MoE model supporting
image, video, and audio inputs.

Key architectural details:
- **LLM backbone**: NemotronV3 hybrid Mamba2 + Attention + MoE, 52 layers, hidden dim 2688
- **Vision encoder**: RADIO v2.5-H (ViT-Huge), 256 vision tokens per tile
- **Audio encoder**: Parakeet FastConformer (1024-dim)
- **MoE**: 128 experts per MoE layer, top-6 routing with sigmoid gating
- **Total parameters**: 33B (31.5B trainable with frozen vision/audio towers)

## Fine-Tune for Receipt Field Extraction

We fine-tune NemotronOmni on the **CORD-v2** (Consolidated Receipt Dataset) to extract
structured fields from scanned receipts:

| Field | Example |
|-------|---------|
| `menu` | Item names, quantities, prices |
| `sub_total` | Subtotal, tax, discount |
| `total` | Total price, cash paid, change |

The **base model** produces free-form descriptions. After fine-tuning, it outputs
**structured XML-like token sequences** matching the receipt fields.

## Guide Overview

| Step | Description |
|------|-------------|
| **Step 0** | Environment setup |
| **Step 1** | Explore the CORD-v2 dataset |
| **Step 2** | Training configuration (SFT and LoRA) |
| **Step 3** | Launch fine-tuning |

## Hardware Requirements

- **8x H100 80 GB** GPUs required (MoE with EP=8)
- **SFT memory**: ~49 GiB per GPU
- **LoRA memory**: ~30 GiB per GPU
- **Estimated training time**: ~10 min on 8x H100 (400 steps, 800 training samples)

---

## Step 0 — Set Up the Environment

```bash
# Inside the NeMo AutoModel container (26.04+):
cd /opt/Automodel

# Or from a source checkout:
git clone -b nemotron-omni ssh://git@gitlab-master.nvidia.com:12051/huiyingl/automodel-omni.git
cd automodel-omni
```

:::{note}
NemotronOmni requires `mamba_ssm`, `causal_conv1d`, and `decord` packages, which are included in the NeMo AutoModel container.
:::

---

## Step 1 — Explore the CORD-v2 Dataset

[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) contains scanned
receipts with structured ground-truth JSON labels.

```python
import json
from datasets import load_dataset

dataset = load_dataset("naver-clova-ix/cord-v2")

print(f"Train      : {len(dataset['train'])} samples")
print(f"Validation : {len(dataset['validation'])} samples")
print(f"Test       : {len(dataset['test'])} samples")

# Inspect a sample
ex = dataset["train"][0]
gt = json.loads(ex["ground_truth"])["gt_parse"]
print(f"\nGround-truth keys: {list(gt.keys())}")
```

Expected output:
```
Train      : 800 samples
Validation : 100 samples
Test       : 100 samples

Ground-truth keys: ['menu', 'sub_total', 'total', 'void_menu']
```

### Target Format: JSON-to-Token Conversion

NeMo Automodel converts structured JSON into an XML-like **token sequence** using
the `json2token()` function. This is the format the model is trained to produce:

```
<s_total><s_total_price>45,500</s_total_price><s_changeprice>4,500</s_changeprice>
<s_cashprice>50,000</s_cashprice></s_total><s_menu><s_price>16,500</s_price>
<s_nm>REAL GANACHE</s_nm><s_cnt>1</s_cnt><sep/><s_price>13,000</s_price>
<s_nm>EGG TART</s_nm><s_cnt>1</s_cnt></s_menu>
```

---

## Step 2 — Training Configuration

### Full SFT Config

**Config file**: `examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2.yaml`

```yaml
recipe: FinetuneRecipeForVLM

step_scheduler:
  global_batch_size: 8
  local_batch_size: 1
  ckpt_every_steps: 100
  val_every_steps: 200
  max_steps: 400

model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: <path_to_nemotron_omni_v2.0>
  trust_remote_code: true
  torch_dtype: torch.bfloat16
  backend:
    _target_: nemo_automodel.components.models.common.BackendConfig
    attn: sdpa
    linear: torch
    rms_norm: torch_fp32
    rope_fusion: false
    enable_deepep: false
    fake_balanced_gate: false
    enable_hf_state_dict_adapter: true

distributed:
  strategy: fsdp2
  ep_size: 8            # 128 MoE experts across 8 GPUs

freeze_config:
  freeze_embeddings: true
  freeze_vision_tower: true
  freeze_audio_tower: true
  freeze_language_model: false

dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  path_or_dataset: naver-clova-ix/cord-v2
  split: train

dataloader:
  collate_fn:
    _target_: nemo_automodel.components.datasets.vlm.collate_fns.nemotron_omni_collate_fn
    max_length: 4096

optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
```

### LoRA PEFT Config

**Config file**: `examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2_peft.yaml`

Adds a `peft:` block to apply LoRA to language model linear layers only:

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: false
  exclude_modules:
    - "*vision_tower*"
    - "*vision_model*"
    - "*audio*"
    - "*sound*"
    - "*lm_head*"
    - "*mlp1*"
  dim: 64
  alpha: 128
  use_triton: true

optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-3
```

### Collate function

NemotronOmni uses InternVL-style image handling where each `<image>` token in the
input is replaced by 256 vision embeddings during the model's forward pass. The
collate function:
1. Extracts images from the conversation
2. Applies the chat template (which adds `<think></think>` prefix for the assistant turn)
3. Processes images through the NemotronOmni processor
4. Builds `image_flags` tensors and creates training labels

---

## Step 3 — Launch Fine-Tuning

### Full SFT

```bash
torchrun --nproc-per-node=8 \
    examples/vlm_finetune/finetune.py \
    -c examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2.yaml
```

### LoRA PEFT

```bash
torchrun --nproc-per-node=8 \
    examples/vlm_finetune/finetune.py \
    -c examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2_peft.yaml
```

### Training log — Full SFT

```
Trainable parameters: 31,570,023,872
Trainable parameters percentage: 95.63%

step    0 | loss 0.6866 | grad_norm  7.57 | lr 1.00e-04 | mem 37.29 GiB | tps/gpu   33
step   10 | loss 0.0705 | grad_norm  1.00 | lr 1.00e-04 | mem 48.95 GiB | tps/gpu 2419
step   50 | loss 0.0173 | grad_norm  0.43 | lr 1.00e-04 | mem 48.72 GiB | tps/gpu 2615
step  100 | loss 0.0115 | grad_norm  0.37 | lr 1.00e-04 | mem 48.84 GiB | tps/gpu 2642
step  200 | loss 0.0099 | grad_norm  0.20 | lr 1.00e-04 | mem 48.76 GiB | tps/gpu 2616
step  300 | loss 0.0056 | grad_norm  0.15 | lr 1.00e-04 | mem 48.72 GiB | tps/gpu 2087
step  399 | loss 0.0039 | grad_norm  0.17 | lr 1.00e-04 | mem 48.79 GiB | tps/gpu 2616

Validation:
  step  99 | val_loss 0.0363
  step 199 | val_loss 0.0342  <-- LOWEST_VAL
  step 299 | val_loss 0.0414
  step 399 | val_loss 0.0425
```

### Training log — LoRA PEFT

```
Trainable parameters: 55,422,976
Trainable parameters percentage: 0.17%

step    0 | loss 0.6866 | grad_norm  1.92 | lr 1.00e-03 | mem 30.26 GiB | tps/gpu   34
step   10 | loss 0.0557 | grad_norm  0.30 | lr 1.00e-03 | mem 30.16 GiB | tps/gpu 2455
step   50 | loss 0.0392 | grad_norm  0.32 | lr 1.00e-03 | mem 30.16 GiB | tps/gpu 3352
step  100 | loss 0.0309 | grad_norm  0.27 | lr 1.00e-03 | mem 30.20 GiB | tps/gpu 2456
step  200 | loss 0.0280 | grad_norm  0.23 | lr 1.00e-03 | mem 30.34 GiB | tps/gpu 2477
step  300 | loss 0.0326 | grad_norm  0.31 | lr 1.00e-03 | mem 30.52 GiB | tps/gpu 2737
step  399 | loss 0.0171 | grad_norm  0.24 | lr 1.00e-03 | mem 30.33 GiB | tps/gpu 3258

Validation:
  step  99 | val_loss 0.0449  <-- LOWEST_VAL
  step 199 | val_loss 0.0524
  step 299 | val_loss 0.0482
  step 399 | val_loss 0.0566
```

### Checkpoints saved

```
checkpoint_dir/
  epoch_0_step_99/
  epoch_1_step_199/
  epoch_2_step_299/
  epoch_3_step_399/
    model/
      consolidated/          <-- HF-compatible checkpoint for inference
        config.json
        model.safetensors.index.json
        model-00001-of-00017.safetensors
        ...
    optim/
    rng/
    dataloader/
  LATEST -> epoch_3_step_399
  LOWEST_VAL -> epoch_1_step_199
  training.jsonl
  validation.jsonl
```

For LoRA, the checkpoint saves adapter weights instead:
```
  model/
    adapter_model.safetensors   (~27 MB)
    adapter_config.json
```

> **Tip**: `LOWEST_VAL` symlink points to the checkpoint with the best validation loss.
