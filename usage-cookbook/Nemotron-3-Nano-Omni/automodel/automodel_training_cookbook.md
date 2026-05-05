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
| **Step 4** | Run inference on the base model and the fine-tuned model |
| **Step 5** | Compare SFT vs LoRA results |

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

### Live training visualization

Stdout-only loss is hard to follow when warnings interleave the log lines. The
recipe writes per-step metrics to `<checkpoint_dir>/training.jsonl` and
`<checkpoint_dir>/validation.jsonl`, which makes live visualization
straightforward:

- **TensorBoard** — stream the JSONL files into a `SummaryWriter`, then run
  `tensorboard --logdir <checkpoint_dir>/tb` and open the URL. Re-run the
  consumer (or wrap it in a `tail -f`-style watcher) to keep the dashboard
  updated mid-run.
- **`tqdm` + `matplotlib`** — for a quick spot-check, read the JSONL files and
  plot `step` vs `loss` for train and val on the same axis. `tqdm` wraps the
  step loop for a live progress bar without a long-running process.

Example loss curves from the SFT and LoRA runs above:

![Loss curves: SFT and LoRA train + val on CORD-v2](TODO-loss-curves.png)

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

---

## Step 4 — Run Inference

### Baseline inference — before fine-tuning

It helps to see what the **base model** produces on the validation samples
before training, so the fine-tuning effect is visible side-by-side. Without
fine-tuning, NemotronOmni answers with free-form prose rather than the
structured XML-like token format the recipe trains it to emit.

```python
import json
import torch
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset

BASE = "<path_to_nemotron_omni_v3>"
PROCESSOR_METADATA_KEYS = ("num_patches", "num_tokens", "imgs_sizes")

processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
model = AutoModel.from_pretrained(
    BASE, trust_remote_code=True, dtype=torch.bfloat16,
    device_map={"": torch.cuda.current_device()},
)
if hasattr(model, "vision_model") and hasattr(model.vision_model, "radio_model"):
    model.vision_model.radio_model.summary_idxs = None
model.eval()

dataset = load_dataset("naver-clova-ix/cord-v2")
tokenizer = processor.tokenizer

for i in [0, 20, 40, 60, 80]:
    sample = dataset["validation"][i]
    image = sample["image"].convert("RGB")
    messages = [{"role": "user", "content": "<image>\nDescribe this image."}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=text, images=[image], return_tensors="pt")
    for k in PROCESSOR_METADATA_KEYS:
        inputs.pop(k, None)
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    answer = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    print(f"\n=== Sample {i} ===\nBase output: {answer}")
```

Example outputs from the base v3 dump (truncated):

| Sample | Base model output |
|--------|-------------------|
| 0 (REAL GANACHE / EGG TART) | "This image shows a close-up of a receipt, which appears to be from a restaurant or food establishment. The receipt is printed on white paper and is partially visible…" |
| 20 (JAMUR / TAHU) | "…items listed are '2 JAMUR' and '1 TAHU,' with prices of 10,000 and 5,000 respectively. The subtotal is 15,000, with a 10% PB1 tax of 1,500, resulting in a total of 16,500…" |
| 40 (Gojek Chicken Chilli Sauce) | "This image shows a close-up of a restaurant receipt, which is partially obscured by a brown, textured surface, likely a table or counter…" |
| 60 (VANILLA CHOCO HEART CAKE) | "…the top portion showing the item '1 VANILLA CHOCO HEART CAKE' priced at 180,000… paid by 'VISA CARD'…" |
| 80 (Sate Padang) | "This image shows a close-up of a white paper receipt placed on a light brown wooden surface with a visible grain pattern…" |

The base model can read the receipts, but it returns natural-language
descriptions — not the `<s_total>...<s_nm>...</s_total>` token sequence the
downstream consumer expects. That gap is what fine-tuning closes.

### Full SFT inference

Load the consolidated checkpoint and run inference on the same validation
samples to spot-check structured output.

```python
import torch
import json
from transformers import AutoConfig, AutoModel, AutoProcessor
from datasets import load_dataset
from nemo_automodel.components.datasets.vlm.utils import json2token

CKPT = "<checkpoint_dir>/LOWEST_VAL/model/consolidated"

# Load processor
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
tokenizer = processor.tokenizer

# Resolve the trust_remote_code model class via from_config, then load weights.
config = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
model_class = type(AutoModel.from_config(config, trust_remote_code=True))
if not hasattr(model_class, "all_tied_weights_keys"):
    model_class.all_tied_weights_keys = {}
model = model_class.from_pretrained(CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16)

if hasattr(model, "vision_model") and hasattr(model.vision_model, "radio_model"):
    model.vision_model.radio_model.summary_idxs = None

model = model.cuda().eval()

dataset = load_dataset("naver-clova-ix/cord-v2")
PROCESSOR_METADATA_KEYS = ("num_patches", "num_tokens", "imgs_sizes")

for i in [0, 20, 40, 60, 80]:
    sample = dataset["validation"][i]
    image = sample["image"].convert("RGB")
    gt = json.loads(sample["ground_truth"])["gt_parse"]
    gt_text = json2token(gt, sort_json_key=True)

    messages = [{"role": "user", "content": "<image>\nDescribe this image."}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=text, images=[image], return_tensors="pt")
    for k in PROCESSOR_METADATA_KEYS:
        inputs.pop(k, None)
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()

    print(f"\n=== Sample {i} ===")
    print(f"Ground truth: {gt_text}")
    print(f"Prediction:   {generated}")
```

### LoRA PEFT inference

NeMo Automodel saves LoRA adapters under its internal wrapper FQNs
(e.g. `language_model.model.layers.X.mixer.in_proj`), which differ from the HF
base model namespace (`language_model.backbone.layers.X.mixer.in_proj`).
To apply the adapter, merge the delta weights directly into the base model with
a small FQN translation:

```python
import json, re
import torch
from pathlib import Path
from safetensors import safe_open
from transformers import AutoModel, AutoProcessor

BASE    = "<path_to_nemotron_omni_v3>"
ADAPTER = "<ckpt_dir>/LOWEST_VAL/model"

processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
model = AutoModel.from_pretrained(
    BASE, trust_remote_code=True, dtype=torch.bfloat16,
    device_map={"": torch.cuda.current_device()},
)
if hasattr(model, "vision_model") and hasattr(model.vision_model, "radio_model"):
    model.vision_model.radio_model.summary_idxs = None

# Wrapper -> HF base FQN translation. vision_projector.* targets are listed in
# adapter_config.json but no tensors are saved for them, so we just skip those.
def translate(fqn):
    if fqn.startswith("language_model.model."):
        return "language_model.backbone." + fqn[len("language_model.model."):]
    return None

cfg   = json.loads((Path(ADAPTER) / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]

pairs = {}
with safe_open(str(Path(ADAPTER) / "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        m = re.match(r"^base_model\.model\.(.+)\.lora_(A|B)\.weight$", k)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)

modules = dict(model.named_modules())
for wrapper_fqn, ab in pairs.items():
    hf_fqn = translate(wrapper_fqn)
    if hf_fqn is None or hf_fqn not in modules:
        continue
    W = modules[hf_fqn].weight
    A = ab["A"].to(device=W.device, dtype=torch.float32)
    B = ab["B"].to(device=W.device, dtype=torch.float32)
    with torch.no_grad():
        W.add_(((B @ A) * scale).to(W.dtype))

model.eval()
# ... then run the same generate() loop as in the SFT example above.
```

**Resources** — single GPU; ~60 GB GPU RAM for the bf16 30B base.
**Runtime** — ~75 s base load + ~1 s LoRA merge + ~5–15 s per sample.

---

## Step 5 — Results Comparison

### Evaluation on 5 CORD-v2 Validation Samples

#### Full SFT (lr=1e-4, 400 steps, epoch_3_step_399)

| Sample | Ground Truth | Prediction | Match |
|--------|-------------|------------|-------|
| 1 | `<s_total>...<s_nm>REAL GANACHE</s_nm>...<s_nm>EGG TART</s_nm>...<s_nm>PIZZA TOAST</s_nm>...` | Exact match | 100% |
| 2 | `<s_total>...<s_nm>JAMUR</s_nm>...<s_nm>TAHU</s_nm>...` | Exact match | 100% |
| 3 | `<s_total>...<s_nm>Gojek Chicken Chilli Sauce H</s_nm>...` | Correct values, slight name segmentation diff | 33% |
| 4 | `<s_total>...<s_nm>VANILLA CHOCO HEART CAKE</s_nm>...` | Exact match | 100% |
| 5 | `<s_total>...<s_nm>Sate Padang</s_nm>...` | Correct, extra `<s_unitprice>` field | ~0% |

**3/5 exact matches. All samples produce correct structured output.**

#### LoRA PEFT (rank=64, lr=1e-3, 400 steps, epoch_0_step_99)

| Sample | Ground Truth | Prediction | Match |
|--------|-------------|------------|-------|
| 1 | `<s_total>...<s_nm>REAL GANACHE</s_nm>...` | Exact match | 100% |
| 2 | `<s_total>...<s_nm>JAMUR</s_nm>...<s_nm>TAHU</s_nm>...` | Exact match | 100% |
| 3 | `<s_total>...<s_nm>Gojek Chicken Chilli Sauce H</s_nm>...` | Correct values, slight name segmentation diff | 33% |
| 4 | `<s_total>...<s_nm>VANILLA CHOCO HEART CAKE</s_nm>...` | Exact match | 100% |
| 5 | `<s_total>...<s_nm>Sate Padang</s_nm>...` | Exact match | 100% |

**4/5 exact matches. All samples produce correct structured output.**

### Summary

| | Full SFT | LoRA PEFT |
|---|---|---|
| Trainable params | 31.5B (95.63%) | 55M (0.17%) |
| Learning rate | 1e-4 | 1e-3 |
| GPU memory | ~49 GiB | ~30 GiB |
| Training time (8x H100) | ~10 min | ~6 min |
| Best val loss | 0.034 (step 199) | 0.045 (step 99) |
| Final train loss | 0.004 | 0.017 |
| Checkpoint size | ~64 GB | ~27 MB |
| Exact matches (5 val) | 3/5 | 4/5 |
