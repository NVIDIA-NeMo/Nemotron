# Quantization (PTQ + QAD)

The released Nemotron 3.5 Super NVFP4 checkpoint ([`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)) is produced with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer): post-training quantization (PTQ) with the predefined **aggressive** recipe, followed by **200 iterations** of SFT-based Quantization-Aware Distillation (QAD). It reaches **99.52% median accuracy recovery** relative to BF16 across 35 benchmarks, with a mean per-benchmark verbosity increase of 0.49%.

This page summarizes §4–5 of the tech report. Quantization is not a `nemotron super35` stage: to quantize your own Super 3.5 checkpoint (for example your own fine-tune), see [Recipe Execution](#recipe-execution), which runs the released precision recipe through PTQ → QAD → export with Megatron-Bridge, using Model Optimizer's [NemotronH Omni VL W4A4 tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD).

---

## Precision Recipes

The released checkpoint uses the **Aggressive** column (Table 12 of the report). Bits per element (BPE) counts stored payload including scales over all original parameters, including vision, MTP, and unquantized modules: 74.987 GB over 124.453B parameters gives 4.82 BPE.

| Layer | BF16 | FP8 | Conservative | **Aggressive (released)** | Very aggressive (not released) |
|-------|------|-----|--------------|---------------------------|--------------------------------|
| C-RADIO encoder | BF16 | BF16 | BF16 | BF16 | BF16 |
| VL connector | BF16 | BF16 | BF16 | BF16 | BF16 |
| MoE + shared-expert GEMMs | BF16 | FP8 | NVFP4 | **NVFP4** | NVFP4 |
| Latent down/up projection | BF16 | FP8 | BF16 | **BF16** | NVFP4 |
| Router / gate (compute) | FP32 | FP32 | FP32 | **FP32** | FP32 |
| Attention QKV / out projection | BF16 | FP8 | BF16 | **FP8** | NVFP4 |
| KV cache | BF16 | FP8 | FP8 | **FP8** | FP4 |
| BMM1 (QKᵀ) | BF16 | FP8 | FP8 | **FP8** | FP4 |
| BMM2 (PV) | BF16 | FP8 | FP8 | **FP8** | FP8 |
| Mamba linear | BF16 | FP8 | BF16 | **FP8** | NVFP4 |
| Mamba SSM cache | FP32 | FP32 | FP32 | **FP16 + stochastic rounding** | FP16 + stochastic rounding |
| Mamba convolution | BF16 | BF16 | BF16 | **BF16** | BF16 |
| Embedding | BF16 | BF16 | BF16 | **BF16** | BF16 |
| lm_head | BF16 | FP8 | BF16 | **FP8** | W4A16 |
| Avg. BPE | 16.00 | 8.10 | 5.15 | **4.82** | 4.64 |

The released checkpoint also contains the quantized, boosted MTP head. It was evaluated with an FP8 KV cache; ablations showed comparable accuracy with an NVFP4 KV cache (99.36% vs. 99.52% median recovery).

### Acceptance Criteria

Recipes are developed between conservative and aggressive bookends and must meet:

- **Median accuracy recovery ≥ 99%** — the median over benchmarks of the NVFP4 / BF16 score ratio, averaged over repeats.
- **Verbosity increase ≤ 5%** relative to BF16.

## PTQ

| Aspect | Choice |
|--------|--------|
| Calibration data | 1,024 image–text samples from the post-training SFT data; images go through the vision encoder so the language model sees projected visual tokens |
| Calibration parallelism | Data parallel × expert parallel across 4 ranks (256 samples each), routed and MTP experts split evenly; vision, attention, and Mamba layers replicated |
| Weight-scale selection | **MSE** (per-block scale search over FP8 values) — better recovery than max-based or MSE 4-over-6 scales |

ModelOpt **AutoQuantize** was also explored as a sensitivity-driven alternative: it assigns NVFP4, FP8, or BF16 per module to minimize total sensitivity under a target bitwidth. The sensitivity profile is highly non-uniform — the `lm_head` accounts for nearly half the NVFP4 sensitivity, attention and Mamba projections come next, and MoE experts are comparatively robust.

## QAD

QAD starts from the calibrated PTQ student and distills the BF16 model into it with a KL-divergence objective.

| Parameter | Value |
|-----------|-------|
| Trainable | Weights of language-model GEMMs with enabled quantizers only (embeddings, norms, vision encoder, projector, MTP, router bias frozen) |
| Quantizer state | Restored from PTQ, not recalibrated |
| Losses | KL only (no hard-target CE, MTP CE, load-balancing, or feature distillation) |
| Optimizer | Distributed AdamW, β₁ = 0.9, β₂ = 0.95, ε = 1e-8, weight decay 0.1, clip 1.0 |
| Learning rate | Linear warm-up to 5e-6 over ~9,100 source records, then constant |
| Data | 100% SFT blend (a 75/25 SFT/RL-rollout mix gave only modest benefits) |
| Scale | 128 nodes × 4 GB200 (512 GPUs); about 2 hours per 100 iterations |
| Selected | 4.82-BPE aggressive recipe at **200 iterations** |

Additional QAD iterations were not uniformly helpful: the best iteration count depended on the recipe, and gains largely saturated by 300 iterations.

## Results

Selected rows from the BF16 vs. released NVFP4 comparison (Table 22 of the report):

| Benchmark | BF16 | NVFP4 | Recovery |
|-----------|-----:|------:|---------:|
| SWE-bench Verified | 71.10 | 71.07 | 99.95% |
| SWE-bench Pro | 51.78 | 51.30 | 99.08% |
| Terminal-Bench 2.1 | 60.67 | 62.78 | 103.47% |
| LiveCodeBench v6 | 88.22 | 87.41 | 99.08% |
| GPQA-Diamond (no tools) | 89.52 | 89.90 | 100.42% |
| HLE (no tools, text) | 31.79 | 31.44 | 98.91% |
| CharXiv (RQ) | 80.82 | 79.88 | 98.85% |
| MathVision | 78.16 | 77.79 | 99.52% |
| Video-MME-v2 | 32.85 | 32.54 | 99.05% |
| MMLongBench-Doc | 62.82 | 62.82 | 100.00% |
| AA-LCR | 68.62¹ | 67.69 | 98.63% |
| IFBench | 70.86 | 71.50 | 100.90% |
| MMLU-ProX | 76.91 | 73.76 | 95.90% |
| **Median recovery (35 benchmarks)** | | | **99.52%** |
| **Verbosity change (ratio of totals)** | | | **+0.49%** |

¹ Table 22 lists the BF16 AA-LCR score as 68.62; Table 10 lists 68.63.

Verbosity excludes non-completed Terminal-Bench rollouts: about 4.5% runaway trajectories of roughly 700K tokens each dominate that benchmark's mean, and including them raises the ratio-of-totals change to 2.99%.

### MTP Drafter Precision

The MTP module (one attention layer and one MoE layer) shares the checkpoint's single `lm_head`. Profiling shows the MTP experts and the `lm_head` dominate drafting latency, so the released checkpoint quantizes the MTP experts to NVFP4 W4A4 (MTP attention weights stay BF16) and the shared `lm_head` to FP8. This barely changes acceptance length while speeding up drafting (Table 20; SPEED-Bench, temperature 1; speedup projected at draft length 5 on one GB200 relative to the NVFP4 backbone with unquantized MTP and `lm_head`):

| Backbone / LM head | MTP experts / MTP LM head | Acceptance length, draft 5 | Acceptance length, draft 7 | Projected speedup |
|--------------------|---------------------------|-------------------:|-------------------:|------------------:|
| BF16 / BF16 | BF16 / BF16 | 4.115 | 4.612 | — |
| NVFP4 / BF16 | BF16 / BF16 | 4.109 | 4.592 | 1.000 |
| NVFP4 / BF16 | NVFP4 / BF16 | 4.066 | 4.518 | 1.043 |
| NVFP4 / BF16 | NVFP4 / FP8 | 4.069 | 4.514 | 1.114 |
| NVFP4 / BF16 | NVFP4 / W4A16 | 4.031 | 4.469 | 1.136 |
| **NVFP4 / FP8** | **NVFP4 / FP8 (released)** | **4.063** | **4.520** | **1.127** |
| NVFP4 / FP8 | NVFP4 / W4A16 | 4.025 | 4.471 | 1.150 |

The MTP KV cache must use the backbone's precision. The released checkpoint was built by quantizing the boosted MTP separately and splicing it into the quantized backbone; the report recommends co-quantizing both in future releases.

## Deployment and Serving

The NVFP4 checkpoint runs in one of two modes, depending on the GPU:

| Platform | Mode | Notes |
|----------|------|-------|
| Blackwell (B200, GB200, GB300) | **W4A4** | NVFP4 weights and activations |
| Non-Blackwell (for example H100) | **W4A16** | NVFP4 weights, 16-bit activations |

Figure 10 measures serving with vLLM on a 20-turn agentic proxy workload (2K input / 1K output tokens per turn). Peak throughput is output tokens/s/GPU at ≥ 50 tokens/s/user; speed at minimum latency is generation tokens/s/user.

| Platform | Peak throughput, BF16 → NVFP4 | Speed at minimum latency, BF16 → NVFP4 |
|----------|-------------------------------|----------------------------------------|
| H100 80GB (W4A16) | 522 → 970 (1.86×) | 509 → 576 (1.13×) |
| GB200 (W4A4) | 1,778 → 4,324 (2.43×) | 587 → 765 (1.30×) |

![NVFP4 vs. BF16 serving with vLLM on H100 (W4A16) and GB200 (W4A4): maximum throughput at ≥ 50 tokens/s/user and generation speed at minimum latency (tech report Figure 10).](../../assets/super35/nvfp4-serving.png)

The report's inference work (§5) targets interactive, long-context agentic serving in vLLM:

- **Prefix caching + speculative decoding**: an extra Mamba cache checkpoint near the speculative-decoding replay boundary avoids replaying a whole Mamba cache block to recover the drafter's hidden state.
- **Per-quantization GEMM backend selection** for mixed-precision checkpoints at intermediate batch sizes.
- **Internal prefill checkpoints** write reusable Mamba states without splitting the prefill forward pass.
- **ReplaySSM** (checkpointing replay for the Mamba state) stores a checkpointed state plus a small buffer of recent inputs instead of writing every speculative state. On one GB200 with a 16-step buffer it raises output throughput by 4.7–6.4% and per-user generation speed by 6.4–12.5% (Table 23).

Headline numbers:

- **Peak throughput**: up to **4,632 output tokens/s/GPU** on GB200 at ≥ 50 tokens/s/user (abstract and conclusion; Table 23: one GB200, concurrency 64, 5 speculative tokens, ReplaySSM off; 4,856 with ReplaySSM on).
- **Against Qwen3.8 Flash-Next**: the abstract and conclusion quote **2.88×** under the evaluated interactive agentic workload. Figure 11, at the same ≥ 50 tokens/s/user threshold, shows 4,331.88 output tokens/s/GPU on GB200 against 2,282.73 for Qwen3.8 Flash-Next (1.90×) and 1,854.35 for Qwen3.5-122B-A10B (2.34×). The report does not reconcile the two.
- **Disaggregated serving** of the NVFP4 checkpoint with NVIDIA Dynamo and vLLM on 11 Vera Rubin (VR200) GPUs, replaying SWE-bench Pro agentic sessions (Table 24):

| Deployment | Concurrency | Output tokens/s/GPU | Tokens/s/user (median) | TTFT (median) |
|------------|------------:|--------------------:|-----------------------:|--------------:|
| Aggregated (11 instances) | 616 | 2,969 | 96.6 | 0.49 s |
| Disaggregated (7 prefill + 4 decode GPUs) | 528 | 3,276 (+10.3%) | 155.9 (+61%) | 1.12 s |

## Recipe Execution

Model Optimizer's [NemotronH Omni VL W4A4 PTQ + QAD tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD) quantizes this architecture (`NemotronH_Omni_Reasoning_V3`, which Megatron-Bridge maps to its Super-VL bridge) with Megatron-Bridge: PTQ → QAD → Hugging Face export → vLLM serving. Its PTQ recipe is the report's released **Aggressive** tier; its calibration, QAD data, and QAD scale are smaller and public. The commands below are summarized from the tutorial; follow it for the full notes.

### Where the Tutorial Differs from the Report

| Aspect | Tech report (released checkpoint) | Model Optimizer tutorial |
|--------|-----------------------------------|--------------------------|
| Precision recipe | Aggressive tier (Table 12) | `model_type/nemotron_h/ptq/nvfp4-aggressive-mse`: NVFP4 W4A4 experts (MSE weight scales), FP8 attention, Mamba `in_proj`/`out_proj` and `lm_head`, FP8 KV cache, NVFP4 MTP experts; vision tower, projector, latent MoE, routers and embeddings BF16 |
| Calibration | 1,024 image–text samples from the SFT data | 512 text-only samples (`cnn_nemotron_v2_mix`); the vision tower is not run |
| QAD data | SFT blend | 35,800 conversations from 11 public Nemotron chat datasets ([`data_blend.yaml`](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD/data_blend.yaml)), loss on assistant tokens |
| QAD scale | 512 GB200 GPUs, 200 iterations, LR 5e-6 constant after warm-up | 32 GB300 GPUs, 600 iterations, GBS 64 at 32K, LR 2e-5 → 5e-6 |
| MTP head | Boosted MTP quantized separately, then spliced in | Quantized in PTQ, frozen during QAD (`--freeze_mtp`) |
| Checkpoint size | 74.987 GB | 70 GB |

### Setup

The tutorial runs in the NeMo 26.08 container (`nvcr.io/nvidia/nemo:26.08`) on GB300 (4 GPUs per node), with Model Optimizer mounted at `/opt/Model-Optimizer` and Megatron-Bridge `main` at `/opt/Megatron-Bridge`. Serving NVFP4 W4A4 requires a Blackwell GPU.

### Step 1 — Data

```bash
# Set output_dir in data_blend.yaml first; writes train.jsonl (35,800) and validation.jsonl (1,000)
python -m modelopt.torch.utils.plugins.prepare_megatron_data_blend --config data_blend.yaml
```

### Step 2 — PTQ (one 4-GPU node)

```bash
torchrun --nproc_per_node 4 /opt/Model-Optimizer/examples/megatron_bridge/quantize.py \
    --hf_model_name_or_path nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --trust_remote_code \
    --recipe model_type/nemotron_h/ptq/nvfp4-aggressive-mse \
    --tp_size 1 --ep_size 4 --pp_size 1 \
    --calib_dataset_name cnn_nemotron_v2_mix --calib_num_samples 512 --calib_batch_size 1 \
    --export_megatron_path /path/to/super35_w4a4_megatron \
    --skip_generate
```

`--tp_size 1` is required: static NVFP4 weight scales are not supported with tensor parallelism, so expert parallelism spreads the experts instead.

### Step 3 — QAD (8 nodes × 4 GB300)

QAD distills the frozen BF16 teacher into the PTQ student with a logit KL loss only. Launch one task per GPU (32 in all) with `srun`, mapping the Slurm rank variables to `RANK` / `WORLD_SIZE` / `LOCAL_RANK`; the tutorial has the full `sbatch` script. Each task runs:

```bash
python -u /opt/Model-Optimizer/examples/megatron_bridge/distill.py \
    --teacher_hf_path nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --student_hf_path nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --student_megatron_path /path/to/super35_w4a4_megatron \
    --trust_remote_code \
    --tp_size 1 --pp_size 1 --cp_size 4 --ep_size 32 \
    --sft --sft_hf_dataset /path/to/qad_blend/train.jsonl \
    --sft_hf_validation /path/to/qad_blend/validation.jsonl \
    --sft_loss_mode assistant \
    --freeze_mtp \
    --seq_length 32768 --mbs 1 --gbs 64 --train_iters 600 \
    --lr 2e-5 --min_lr 5e-6 --lr_warmup_iters 30 \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --no_async_save --save_interval 100 --checkpoint_keep_last -1 \
    --eval_iters 4 --eval_interval 100 --log_interval 10 \
    --output_dir /path/to/qad_output
```

Each saved iteration is about 1.6 TB; size `--save_interval` and `--checkpoint_keep_last` to your storage.

### Step 4 — Export to Hugging Face (one 4-GPU node)

```bash
torchrun --nproc_per_node 4 /opt/Model-Optimizer/examples/megatron_bridge/export_quantized_megatron_to_hf.py \
    --hf_model_name_or_path nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --trust_remote_code \
    --megatron_path /path/to/qad_output/checkpoints \
    --export_unified_hf_path /path/to/super35_w4a4_hf \
    --pp_size 4
```

The exporter writes the quantized language model (including the MTP head) and copies the vision tower and projector from the BF16 checkpoint. It exports the latest `iter_*` under `--megatron_path`.

### Tutorial Results

Served with vLLM on one 4-GPU GB300 node (TP4 with expert parallelism; FP8 KV cache for the quantized checkpoints, FP16 Mamba SSM cache with stochastic rounding, `nemotron_v3` reasoning parser):

| Benchmark | BF16 | W4A4 PTQ | W4A4 QAD (600 iters) |
|-----------|-----:|---------:|---------------------:|
| AA-LCR | 68.63 | 68.62 | 67.94 |
| AA-Omniscience | 26.21 | 25.63 | 25.87 |
| GPQA Diamond | 89.52 | 89.84 | 89.90 |
| HLE (no tools) | 31.79 | 30.49 | 30.72 |
| MMLU-ProX | 76.91 | 74.41 | 81.15 |
| SciCode (subtask) | 47.13 | 46.48 | 46.58 |
| τ³-bench banking | 18.88 | 17.40 | 20.10 |

These are the tutorial's numbers for its own checkpoint, not the released one (see [Results](#results)). Most of the MMLU-ProX gain after QAD comes from fewer responses missing the required answer phrase.

## Reference

- Nemotron 3.5 tech report, §4 (Quantization), §5 (Inference)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) — PTQ, AutoQuantize, QAD
- [NemotronH Omni VL W4A4 PTQ + QAD tutorial](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge/tutorials/NemotronH-Omni-W4A4-QAD) — the commands in [Recipe Execution](#recipe-execution)
- [Nemotron 3.5 Lightning quantization](../lightning35/quantization.md) — PTQ + QAD workflow with the same tooling
- [Nemotron 3 Super quantization](../super3/quantization.md) — Mamba-MoE quantization configs and Megatron-Bridge scripts
- [Back to Overview](./README.md)
