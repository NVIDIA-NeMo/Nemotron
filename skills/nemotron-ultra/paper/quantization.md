---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "quantization"
paper_sections: ["4", "4.0.1", "4.0.2", "4.0.3", "4.1", "4.2", "4.3"]
title: "Nemotron 3 Ultra Quantization: NVFP4 PTQ recipe, BPE selection, FP4 algorithms, SSM cache, single checkpoint"
summary: |
  Covers §4: post-training quantization (PTQ) of Nemotron 3 Ultra to NVFP4 for Blackwell inference via
  Model-Optimizer. Documents the per-operator precision recipe (Table 12), the bits-per-element (BPE)
  sweep selecting 5.03 BPE (Table 13), FP4 weight scale-selection ablations selecting Four-Over-Six
  (Table 14), the Model-Optimizer software support (Megatron-LM vs HuggingFace, Table 15), Mamba SSM
  cache quantization (Table 16), and the single NVFP4 checkpoint deployed as W4A16/W4A4 (Table 17).
key_facts:
  - "PTQ target is NVFP4 for efficient inference on NVIDIA Blackwell GPUs, applied with Model-Optimizer."
  - "Initial recipe seeded from a heuristic mixed per-layer precision recipe informed by Model-Optimizer AutoQuantize sensitivity analysis on Nemotron 3 Super."
  - "Final operating point is 5.03 bits-per-element (BPE): NVFP4 with mixed-FP8."
  - "Per-operator recipe (Table 12): Embedding/Output classification/MTP layers = BF16; MoE routed experts = NVFP4; MoE shared experts = FP8 per-tensor; Mamba mixer linears = FP8 per-tensor; Attention linears = BF16; Latent MoE = BF16; Mamba conv1d = BF16; KV cache = FP8; Mamba SSM cache = FP16 with stochastic rounding (from FP32 baseline)."
  - "BPE sweep evaluated via Nemo Evaluator SDK served on vLLM v0.20.0 over seven benchmarks: SciCode, GPQA Diamond, HLE, CritPt, IFBench, AA-Omniscience, AA-LCR."
  - "BPE sweep range was 4.85–7.19 BPE; most capabilities saturated at the lowest BPE tried."
  - "The single discriminating axis is long-context: AA-LCR improves +2.4 points at the 4.85 -> 5.03 step then plateaus (64.2–65.0) through 7.19 BPE."
  - "Going from 5.03 to 7.19 BPE is a 43% increase in bits with no benchmark gain beyond noise."
  - "AA-LCR step (4.85->5.03) coincides with introducing mixed-FP8 layers on top of the NVFP4-amax recipe."
  - "CritPt sits near the benchmark floor (~3–5%) and is treated as noise-dominated; AA-Omniscience non-hallucination mildly favors lowest BPE (54.13 at 4.85 vs 51.59 at 5.03), attributed to variance."
  - "FP4 input activations use the default NVFP4 PTQ recipe with max-based scaling from calibration statistics; only the FP4 weight scale-selection rule is varied."
  - "Weight scale-selection algorithms compared: max-based (block absolute maximum), MSE-based (minimizes reconstruction error), and Four-Over-Six (per-block choice of M=4 vs M=6 grids)."
  - "Four-Over-Six increases the global per-tensor weight scale by 1.75x; M=4 uses a 1.5x larger block scale than M=6."
  - "Max-calibrated Four-Over-Six reduced median relative MSE of quantized weight reconstruction by 16.4% vs standard max calibration, improving across all 49,152 projection weights from 48 MoE expert layers."
  - "MSE calibration gave an additional 27.1% weight-MSE reduction but no consistent downstream accuracy gain; Four-Over-Six was selected for FP4 routed-expert weight scales in the mixed-precision 5.03 BPE setting."
  - "Final GEMM/MoE PTQ recipe: NVFP4 routed-expert GEMMs (dynamic max-based activation scaling + max-calibrated 4/6 weight scaling); FP8 per-tensor GEMMs for shared experts and Mamba linears (static max-calibrated per-tensor scales); BF16 for attention linears and MoE latent projection layers."
  - "Quantization used Megatron-LM (chosen over HuggingFace for Ultra's size) for multi-node distributed parallelism, expert/data/context parallelism."
  - "Ultra's 550B params -> BF16 model is ~1.1TB, cannot fit in one node; HuggingFace path quantizes layer-by-layer with CPU offloading on a single node."
  - "PTQ time: HuggingFace layerwise PTQ ~2 hours vs Megatron-LM 42 minutes (45 minutes total in Table 15) on Ultra."
  - "Table 15 compute: HuggingFace = 4x B300; Megatron-LM = 16x B300s with Expert Parallelism = Data Parallelism = 16."
  - "In Nemotron 3 Ultra, the 32-bit Mamba cache is larger than the FP8 KV cache at sequence lengths up to 64K (Figure 14)."
  - "Following Nemotron 3 Super, SSM cache uses 16-bit FP16 with stochastic rounding, preserving FP32-cache accuracy and verbosity; 8-bit formats explored on Super and validated to hold on Ultra."
  - "Mamba cache results (Table 16, on Nemotron 3 Super): FP16 SR = -0.32% accuracy drop / 1.13% verbosity increase; INT8 SR (block size 128) = 0.00% / 2.03%; FP8 SR (block 128) = 1.96% / 3.95%; FP16 RTN = 2.32% / 10.84%; INT8 RTN (block 128) = 2.87% / 10.72%; FP8 RTN (block 128) = 12.68% / 26.07%."
  - "Optimized INT8 Mamba cache kernels are still under development; current release uses FP16 SSM cache storage with stochastic rounding."
  - "One NVFP4 checkpoint released: runs as native FP4 (W4A4) on Blackwell and as W4A16 (NVFP4 weights, BF16 activations) on Hopper (which lacks native FP4 tensor cores)."
  - "On an 8-GPU H100 node (TP=8, 640 GiB aggregate HBM): FP8 checkpoint ~540 GiB leaves ~10 GiB/GPU for activations/KV/Mamba; NVFP4 checkpoint ~330 GiB leaves ~40 GiB; measured throughput-vs-latency Pareto places W4A16 on or above W8A8."
  - "NVFP4 = E2M1 elements with E4M3 block scales (effective E6M4); naive W4->FP8 downcast saturates, so W4A8 needs a W4->BF16->FP8 round-trip; W4A8 was not shipped (strictly worse than W4A16, no accuracy upside)."
  - "Single-checkpoint W4A16/W4A4 accuracy (Table 17) vs BF16; W4A16 outperforms NVFP4 on four of five tasks (all but HLE) and stays within 1 point of or above BF16 on four of five (all but HLE)."
related_steps: []
currency: "frozen"
---

# Scope

Answers:
- What format does Nemotron 3 Ultra quantize to, on what hardware, and with what tool?
- What is the per-operator quantization recipe (GEMM, KV cache, Mamba cache)?
- How was the bits-per-element budget chosen, and why 5.03 BPE?
- Which FP4 weight scale-selection algorithm was chosen and why?
- What is the final weight + GEMM recipe?
- How does Model-Optimizer software support (Megatron-LM vs HuggingFace) compare?
- How is the Mamba SSM cache quantized?
- Why a single NVFP4 checkpoint, and how does it deploy on Blackwell vs Hopper?

# Per-operator quantization recipe (Table 12)

NVFP4 PTQ from Model-Optimizer. All baselines are BF16 except SSM cache (FP32 baseline).

| Layer / Operator | BF16 Baseline | Quantized Checkpoint Precision |
|---|---|---|
| Embedding, Output classification layer, MTP layers | BF16 | BF16 |
| MoE routed experts | BF16 | NVFP4 |
| MoE shared experts | BF16 | FP8 per-tensor |
| Mamba mixer linears | BF16 | FP8 per-tensor |
| Attention linears | BF16 | BF16 |
| Latent MoE | BF16 | BF16 |
| Mamba conv1d | BF16 | BF16 |
| KV cache | BF16 | FP8 |
| Mamba SSM cache | FP32 | FP16 with stochastic rounding |

# Bits-per-element (BPE) sweep (Table 13)

Fixed intermediate checkpoint; Nemo Evaluator SDK on vLLM v0.20.0; averaged pass@1. `†` 5.03 = NVFP4 with mixed-FP8 = selected operating point. Best per row in bold per source.

| Task | Metric | 4.85 | 5.03† | 5.25 | 5.43 | 7.19 |
|---|---|---|---|---|---|---|
| SciCode | pass@1 (avg-16), subtask acc | 43.82 | 43.88 | 43.45 | 43.27 | 43.44 |
| GPQA Diamond | pass@1 (avg-32), sym. correct | 84.66 | 84.33 | 84.75 | 84.12 | 84.52 |
| HLE | pass@1, judge correct | 24.24 | 24.84 | 25.00 | 24.98 | 25.44 |
| CritPt | pass@1 (avg-8), accuracy | 3.04 | 3.93 | 5.18 | 4.82 | 4.46 |
| AA-Omniscience | pass@1 (avg-20), judge correct | 29.21 | 29.75 | 29.18 | 29.29 | 29.00 |
| AA-Omniscience | pass@1 (avg-20), non-hallucination | 54.13 | 51.59 | 51.84 | 51.70 | 52.81 |
| IFBench | pass@1 (avg-8), avg. score | 79.34 | 79.26 | 79.83 | 79.53 | 79.83 |
| AA-LCR | pass@1 (avg-16), judge correct | 62.25 | 64.69 | 64.19 | 64.94 | 65.00 |

Higher-BPE points are qualitatively different strategies (router-only quantization; keeping last MoE layer in BF16; skipping top-8 most sensitive experts). 5.03 BPE selected as smallest budget recovering long-context (AA-LCR) with no measurable gain at higher precision (5.03 -> 7.19 = +43% bits, no change beyond noise).

# FP4 weight scale-selection ablations (Table 14)

Median accuracy recovery on 6 AA benchmarks (GPQA, SciCode, HLE AA, IFBench, CritPT, Omniscience) relative to BF16, intermediate checkpoint. Activation scales use max calibration in all columns; SSM cache used FP32.

| BPE | Max per-block | MSE per-block | Four-Over-Six per-block |
|---|---|---|---|
| 5.43 (NVFP4 Routed Experts only) | 97.44 | 98.27 | n/a |
| 5.03 (NVFP4 Routed Experts + mixed FP8) | 96.78 | 98.40 | 98.50 |
| 4.85 (NVFP4 experts + Mamba) | 98.32 | 97.57 | 84.71 |

Four-Over-Six wins at balanced 5.03 BPE but degrades sharply at 4.85 BPE (likely Mamba linear layers sensitive to outliers, preferring max scales). Selected: Four-Over-Six for FP4 routed-expert weight scales at 5.03 BPE.

# Model-Optimizer software support (Table 15)

| Metric | HuggingFace transformers | Megatron-LM |
|---|---|---|
| Compute | 4 x B300 | 16 x B300s; Expert Parallelism = Data Parallelism = 16 |
| Model loading time | 40 minutes | < 2 minutes |
| Model loading & Calibration time | 85 minutes | 9 minutes |
| Export | 42 min | 33 min |
| Total Time | 2 hours | 45 minutes |

Megatron-LM chosen for Ultra due to multi-node n-D parallelism. BF16 Ultra (~1.1TB) cannot fit one node; HuggingFace path uses single-node CPU offloading, quantizing layer-by-layer.

# Mamba SSM cache quantization (Table 16, on Nemotron 3 Super)

Lower is better for both metrics. Conclusions validated on Ultra.

| Mamba cache precision | Median accuracy drop from FP32 cache | Average verbosity increase from FP32 cache |
|---|---|---|
| FP16 RTN | 2.32% | 10.84% |
| FP16 SR | -0.32% | 1.13% |
| INT8 RTN (block size = 128) | 2.87% | 10.72% |
| INT8 SR (block size = 128) | 0.00% | 2.03% |
| FP8 RTN (block size = 128) | 12.68% | 26.07% |
| FP8 SR (block size = 128) | 1.96% | 3.95% |

Benchmarks: MMLU Pro, GPQA, HLE, LiveCodeBench, IFBench, OmniScience, AA-LCR, Ruler 128K, Ruler 256K. INT8 SR best preserves accuracy but optimized INT8 kernels are still under development; current release ships FP16 SR.

# Single NVFP4 checkpoint deployment accuracy (Table 17)

Score = pass@1; Tok. = average completion length in tokens. Higher score per task bold in source.

| Task | BF16 Score | BF16 Tok. | W4A16 Score | W4A16 Tok. | NVFP4 (W4A4) Score | NVFP4 (W4A4) Tok. |
|---|---|---|---|---|---|---|
| GPQA | 86.67 | 14408 | 86.67 | 14841 | 86.36 | 15134 |
| HLE | 26.92 | 37479 | 25.12 | 39103 | 25.67 | 39103 |
| IFBench | 82.12 | 5272 | 82.75 | 5466 | 82.42 | 5764 |
| AA-Omniscience | 24.38 | 1140 | 25.50 | 1396 | 24.55 | 1226 |
| AA-LCR | 63.67 | 4110 | 65.33 | 4286 | 64.00 | 4346 |

One checkpoint: native FP4 (W4A4) on Blackwell; W4A16 on Hopper (no native FP4 tensor cores). KV cache FP8, Mamba cache FP16 with stochastic rounding. W4A8 not shipped: NVFP4 (effective E6M4) downcast to FP8 saturates, requiring a W4->BF16->FP8 round-trip, making it strictly worse than W4A16.

# Caveats

- Table 14 numbers are median accuracy recovery vs BF16 on an intermediate checkpoint, not final Ultra eval scores.
- Table 16 (Mamba cache) was measured on Nemotron 3 Super; the report states conclusions were validated to hold on Ultra but does not give Ultra-specific numbers.
- BPE is "a summary axis over a family of quantization recipes," not a single tunable knob; do not interpret BPE columns as pure bit-count changes.
- CritPt and AA-Omniscience non-hallucination differences across the BPE sweep are treated by the authors as noise, not signal.
- The 8-GPU H100 / GiB-budget figures use "≈"/"roughly" qualifiers in the source; treat as approximate.
- Several Table 18-style eval cells in §5 carry "[TODO: Carlo]" markers in the source slice; the Table 17 numbers above do not.
