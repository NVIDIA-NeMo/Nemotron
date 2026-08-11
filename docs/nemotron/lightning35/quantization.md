# Quantization (PTQ + QAD)

This stage quantizes Nemotron 3.5 Lightning to NVFP4, then recovers the accuracy lost to
quantization with Quantization-Aware Distillation (QAD). Every command below runs from
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)'s
[`examples/megatron_bridge/`](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge),
which builds on [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge): `quantize.py`
loads the model through `AutoBridge`, and `distill.py` drives Megatron-Bridge's
`megatron.bridge.training.distill` training loop.

Compressed down to 22 GB from the 66 GB BF16 checkpoint by quantizing many of its weights to 4 bits,
the NVFP4 checkpoint preserves accuracy while unlocking up to 4x faster throughput.

---

## What is QAD?

Quantization-aware distillation uses the original full-precision model (teacher) to teach the
quantized model (student):

1. **Stage 1 — PTQ.** Run post-training quantization on the BF16 model to produce a quantized
   student checkpoint (W4A16 NVFP4).
2. **Stage 2 — QAD.** Distill the frozen BF16 teacher into that student. Every forward pass of the
   student runs through simulated quantization, so the model learns to account for the quantization
   noise it will see at inference, while a KL-divergence loss on the logits aligns it with the
   teacher.

Because QAD recovers accuracy afterward, PTQ can be pushed harder than it could be as a final step.
A PTQ checkpoint that shows a small but meaningful drop confirms the size and latency gains were
banked, while leaving room for QAD to close the gap.

---

## Quantization Configurations

Several PTQ recipes were tried for the student, which differ in how the weights are calibrated and
how aggressively the Mamba projections and KV cache are quantized. A few settings are shared across
every recipe: all of them quantize the `lm_head` to W4A16, a choice we call a faithful `lm_head`,
while attention projection layers stay in BF16. Calibration uses 1000 samples and runs on a single
B300.

| Recipe | MoE / shared / lm_head weights | Calibration | Mamba in/out_proj | KV cache |
|---|---|---|---|---|
| `max` | W4A16 dynamic NVFP4 | max | W4A16 NVFP4 | FP8 |
| `mamba_fp8_max` | W4A16 dynamic NVFP4 | max | FP8 (W+A) | FP8 |
| `MSE` | W4A16 static NVFP4 | MSE (mean squared error) | W4A16 NVFP4 | FP8 |
| **`four_over_six`** | W4A16 static NVFP4 | 4/6 (MSE over M=6 vs M=4, [arXiv:2512.02010](https://arxiv.org/abs/2512.02010)) | W4A16 NVFP4 | FP8 |
| `four_over_six + NVFP4 KV` | W4A16 static NVFP4 | 4/6 | W4A16 NVFP4 | NVFP4 |

All five use W4A16 NVFP4 weights and range from max-calibrated dynamic recipes to MSE-based static
recipes. The last recipe, `four_over_six + NVFP4 KV`, is the most aggressive: it pushes only K and V
to NVFP4 (W4A4) and leaves the Q·Kᵀ and attn·V batched matrix multiplications in BF16, with Q left
unquantized.

**After evaluating different PTQ recipes, `four_over_six` with W4A16 Mamba linears provided the best
tradeoff** of accuracy degradation to boost in inference performance, and is the recipe used for the
published checkpoint (`Nemotron-3.5-Lightning-30B-A3B/lightning_w4a16_nvfp4_4o6`).

This choice carries into training: max-calibrated recipes feed **dynamic-scale** QAD, where the
scales are recomputed on the fly during training and both the weights and the scales adapt as the
model learns. MSE-based recipes — `MSE`, `four_over_six`, `four_over_six + NVFP4 KV` — arrive at
their scales through a search that is far too expensive to repeat at every step, so they feed
**frozen-scale** QAD: the scales and amax values found during PTQ are frozen for the duration of
training and only the weights are updated. The scale strategy is chosen before training time and
follows directly from the PTQ recipe picked here (see Step 2).

---

## Data

Both stages below use **public** data only.

| Stage | Data | Notes |
|---|---|---|
| PTQ calibration | `cnn_nemotron_v2_mix` (cnn_dailymail + Nemotron-Post-Training v2) | The default when `--calib_dataset_name` is unset; size set by `--calib_num_samples` |
| QAD training | Public Nemotron post-training blend | Pre-tokenized to Megatron `.bin`/`.idx`; see [Preparing the QAD data](#preparing-the-qad-data) |

For reproducing this work, NVIDIA's released open datasets — Nemotron-Post-Training v1 and
Nemotron-Post-Training v2 — cover a similar distribution to the internal mixes used during
development.

---

## Recipe Execution

### Setup

```bash
git clone https://github.com/NVIDIA/Model-Optimizer.git
cd Model-Optimizer/examples/megatron_bridge

export HF_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
export PTQ_CKPT=/path/to/lightning-nvfp4-ptq        # Megatron-format student
export QAD_CKPT=/path/to/lightning-nvfp4-qad        # Megatron-format QAD output
export EXPORT_DIR=/path/to/lightning-nvfp4-hf       # deployable HF checkpoint
```

### Step 1 — PTQ

Produces the quantized student that seeds QAD. Calibration uses 1000 samples of the default public
mix (`cnn_nemotron_v2_mix` — cnn_dailymail plus Nemotron-Post-Training v2), at the sequence length
the model will be distilled at.

```bash
python quantize.py \
    --hf_model_name_or_path $HF_MODEL \
    --trust_remote_code \
    --tp_size 1 --ep_size 1 --pp_size 1 \
    --recipe models/Nemotron-3.5-Lightning-30B-A3B/lightning_w4a16_nvfp4_4o6 \
    --calib_batch_size 1 \
    --calib_num_samples 1000 \
    --seq_length 32768 \
    --export_megatron_path $PTQ_CKPT \
    --skip_generate
```

Run across 4 ranks on one node. `--tp_size 1 --ep_size 1 --pp_size 1` gives pure data parallelism
(DP=4), so each rank calibrates on its own shard of the samples; the static-block NVFP4 recipe is
TP=1 only. Drop `--skip_generate` to emit sample generations at the end as a sanity check.

`--recipe` is authoritative: when set, `--quant_cfg`, `--kv_cache_quant`, `--weight_only` and
`--moe_calib_experts_ratio` are ignored, and the YAML supplies `quant_cfg`, the calibration
algorithm, and the KV-cache config.

> The `lightning_w4a16_nvfp4_4o6` recipe ships with the model release. Until then, the four-over-six
> numerics are public — see `modelopt_recipes/configs/numerics/nvfp4_four_over_six.yaml` and the
> `Nemotron-3-Ultra-550B-A55B/ptq/nvfp4-4o6.yaml` recipe for a working example of the same
> calibration method on a different model size.

### Step 2 — QAD

QAD runs Megatron-Bridge's distillation stack:
`megatron.bridge.training.distill.distill()` is the training loop, and the KD loss and ModelOpt
integration live in `megatron.bridge.training.post_training.distillation`
(`ModelOptDistillConfig`, `loss_func_kd`).

`distill.py` is a thin driver over it: it assembles the Megatron-Bridge `ConfigContainer` from the
flags below and calls `distill(config)`.

#### Preparing the QAD data

The distillation script consumes pre-tokenized Megatron `.bin`/`.idx` data. Tokenize the public
Nemotron datasets once with Model Optimizer's preprocessing utility — full commands for every
dataset below are in
[`examples/dataset/MEGATRON_DATA_PREP.md`](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/dataset/MEGATRON_DATA_PREP.md):

```bash
export TOKENIZER=$HF_MODEL
export OUTPUT_DIR=/path/to/tokenized_nemotron

python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Pretraining-SFT-v1 \
    --hf_split train \
    --hf_streaming \
    --hf_max_samples_per_split 10_000_000 \
    --json_keys text \
    --tokenizer ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --workers 96 \
    --max_sequence_length 256_000 \
    --append_eod \
    --strip_newlines
```

Use `--append_eod` for raw-text datasets (`text` key) and omit it for chat datasets (`messages`
key), whose chat template already terminates each conversation.

The blend used here weights the tokenized outputs as follows:

| Weight | Dataset |
|---|---|
| 20 | `Nemotron-Pretraining-SFT-v1` / Nemotron-SFT-General |
| 17 | `Nemotron-SFT-Math-v3` / train |
| 15 | `Nemotron-SFT-Competitive-Programming-v2` / python |
| 10 | `Nemotron-Math-v2` / high_part00 |
| 8 | `Nemotron-Post-Training-Dataset-v1` / stem |
| 5 | `Nemotron-Pretraining-SFT-v1` / Nemotron-SFT-Code |
| 5 | `Nemotron-Pretraining-SFT-v1` / Nemotron-SFT-MATH |
| 5 | `Nemotron-SFT-Competitive-Programming-v2` / cpp |
| 5 | `Nemotron-Agentic-v1` / tool_calling |
| 3 | `Nemotron-Science-v1` / MCQ |
| 3 | `Nemotron-SFT-Instruction-Following-Chat-v2` / reasoning_on |
| 2 | `Nemotron-Science-v1` / RQA |
| 2 | `Nemotron-SFT-Instruction-Following-Chat-v2` / reasoning_off |

#### Running QAD

```bash
export DATA_BLEND="20 ${OUTPUT_DIR}/nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-General_train_text_max10000000 \
17 ${OUTPUT_DIR}/nvidia--Nemotron-SFT-Math-v3_default_train_messages \
15 ${OUTPUT_DIR}/competitive_programming_python_00_messages"   # ... etc, per the table above

python -u distill.py \
    --teacher_hf_path $HF_MODEL \
    --student_hf_path $HF_MODEL \
    --student_megatron_path $PTQ_CKPT \
    --trust_remote_code \
    --tp_size 1 --ep_size 16 --pp_size 1 --cp_size 4 \
    --data_paths ${DATA_BLEND} \
    --data_path_to_cache /path/to/blend_idx_cache \
    --seq_length 32768 --mbs 1 --gbs 64 \
    --lr 2e-5 --min_lr 5e-6 --lr_warmup_iters 30 \
    --eval_interval 50 --eval_iters 8 --log_interval 10 \
    --train_iters 200 \
    --checkpoint_keep_last 2 \
    --output_dir $QAD_CKPT
```

Run this across 8 nodes × 4 GPUs. With TP=1, PP=1, CP=4 that leaves DP=8; at micro-batch-size 1 and
global-batch-size 64 each step is 8 gradient-accumulation microbatches, and 200 iterations covers
roughly 419M training tokens.

Sequence length turned out to be critical for certain benchmarks, especially the longer-context
ones, where training too short leaves accuracy on the table. Post-training SFT used roughly 522K
tokens, and the ablations showed that a 522K sequence length was necessary for preserving long
context performance. To train at a higher sequence length, raise `--seq_length`.

### Step 3 — Export to HuggingFace

Convert the distilled (still quantized) Megatron checkpoint into a deployable unified-HF checkpoint:

```bash
python -u export_quantized_megatron_to_hf.py \
    --hf_model_name_or_path $HF_MODEL \
    --megatron_path $QAD_CKPT/checkpoints \
    --trust_remote_code \
    --pp_size 1 \
    --export_unified_hf_path $EXPORT_DIR
```

Tensor parallelism must be 1 for export — the HF writer does not gather TP shards. `--pp_size 1`
performs a single-rank export, which fits the 30B model on one large-memory GPU and avoids a
pipeline gather; increase `--pp_size` if the model does not fit.

The same step exports a PTQ-only checkpoint: point `--megatron_path` at `$PTQ_CKPT` instead.

---

## Infrastructure

| Component | Role |
|---|---|
| [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) | Distributed training primitives (TP, PP, EP, CP) |
| [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | Model loading (`AutoBridge`), the distillation training loop, checkpointing |
| [Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | Quantization algorithms and recipes, the PTQ/QAD/export driver scripts |

### Parallelism configuration

| Stage | TP | PP | EP | CP | Resources |
|---|---|---|---|---|---|
| PTQ | 1 | 1 | 1 | — | 1 node × 4 GPUs (DP=4) |
| QAD | 1 | 1 | 16 | 4 | 8 nodes × 4 GPUs (DP=8) |
| Export | 1 | 1 | — | — | 1 GPU |

---

## Reference

- [Model Optimizer Megatron-Bridge examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge) — the PTQ, QAD and export scripts used here
- [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — the underlying training and conversion framework
- [Megatron data preparation](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/dataset/MEGATRON_DATA_PREP.md) — tokenizing datasets for Megatron
- [Model Optimizer QAD documentation](https://github.com/NVIDIA/Model-Optimizer)
