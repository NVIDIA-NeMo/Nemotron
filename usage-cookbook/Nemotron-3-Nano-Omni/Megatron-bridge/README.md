# Nemotron-3-Nano-Omni LoRA Fine-Tuning on CORD-v2 — Megatron Bridge

LoRA PEFT fine-tuning of **Nemotron-3-Nano-Omni 30B-A3B Reasoning BF16** on the
[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) receipt-parsing dataset
using a single 8×H100 80 GiB node and the Megatron Bridge backend.

The main walkthrough is in **[`mbridge_lora_cord_v2_cookbook.ipynb`](mbridge_lora_cord_v2_cookbook.ipynb)**.

---

## Quick Start

### 1 — Set host-side variables

```bash
export WORKSPACE="/path/to/your/workspace"   # must contain Nemotron directory and will hold all outputs
export HF_HOME="${HOME}/.cache/huggingface"
```

Expected workspace layout:

```
$WORKSPACE/
├── Megatron-Bridge/          ← https://github.com/NVIDIA-NeMo/Megatron-Bridge (branch: nemotron-3-omni)
├── Nemotron/                 ← this repo (https://github.com/NVIDIA-NeMo/Nemotron)
├── models/
│   ├── Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/   ← HF model dir (config + tokenizer)
│   └── megatron_base/                                  ← converted Megatron checkpoint (Step 1 in notebook)
└── results/cord_v2/          ← training outputs
```

### 2 — Start the container

```bash
CONTAINER="nvcr.io/nvidia/nemo:26.04.00"

docker container run \
  --gpus all -it --rm \
  --shm-size=16g --net=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${WORKSPACE}:/workspace" \
  -w /workspace \
  "${CONTAINER}" bash
```

### 3 — Launch Jupyter

```bash
cd /workspace/Nemotron

jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser \
  --NotebookApp.token='' --NotebookApp.password='' \
  usage-cookbook/Nemotron-3-Nano-Omni/Megatron-bridge/
```

Open `http://<host-ip>:8888` and run **`mbridge_lora_cord_v2_cookbook.ipynb`** top to bottom.

---

## What the notebook covers

| Step | Description |
|---|---|
| Setup | Clone Megatron-Bridge, download or symlink HF model |
| Step 0 | Verify environment |
| Step 1 | Convert HF checkpoint → Megatron format |
| Step 2 | Inspect CORD-v2 dataset |
| Step 3 | Recipe overview (LoRA config, parallelism) |
| Step 4 | Launch LoRA training (200 iters, 8×H100) |
| Step 5 | Inspect checkpoints |
| Step 6 | Export LoRA adapter to HF PEFT format |
| Step 7 | Merge LoRA into base Megatron checkpoint |
