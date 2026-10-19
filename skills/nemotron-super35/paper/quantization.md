---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "quantization"
paper_sections: ["4", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.9"]
title: "Nemotron 3.5 Super VL Quantization: aggressive NVFP4 recipe, MSE PTQ, AutoQuant, 200-iteration QAD, KV cache, MTP, W4A4/W4A16"
summary: |
  Covers §4 of the Nemotron 3.5 Super and Lightning tech report as it applies to Super 3.5 VL (120B-A12B):
  acceptance criteria and bookend recipe design, the predefined precision tiers (Table 12; released =
  Aggressive, 4.82 BPE), PTQ calibration and weight-scale selection (Table 13; MSE chosen), AutoQuantize
  sensitivity search (§4.3), quantization-aware distillation (QAD; Tables 14-17; released = aggressive
  4.82-BPE recipe at 200 iterations, 100% SFT blend), FP8 vs NVFP4 KV cache (Table 18), attention /
  Mamba / lm_head precision findings (§4.6, Table 19), MTP drafter quantization (Table 20), the full
  BF16-vs-NVFP4 results (Table 22: 99.52% median recovery, +0.49% verbosity) and serving speedups
  (Figure 10; W4A4 on Blackwell, W4A16 elsewhere). Lightning-only material (§4.8, Table 21) is excluded.
  Mechanistic ablations behind §4.6 live in quantization-ablations.md (Appendix C).
key_facts:
  - "Tooling: post-training quantization (PTQ) plus quantization-aware distillation (QAD) with NVIDIA Model Optimizer (ModelOpt) (§4)."
  - "Deployment: one NVFP4 checkpoint deploys W4A4 on Blackwell and W4A16 on non-Blackwell platforms (§4). Lightning 3.5 is W4A16 on every SKU; do not conflate."
  - "Acceptance criteria: median accuracy recovery >= 99% (median over benchmarks of NVFP4/BF16 score ratios across repeats) and verbosity increase <= 5% vs BF16 (§4)."
  - "Recipe bookends (conservative, aggressive, optionally very aggressive) are used to project speedups at two SLAs: throughput (tok/s/gpu) at 50 tok/s/usr, and max throughput; default 51200/2048/1024 pre-cached ISL / new ISL / OSL (§4). Projected bookend speedup numbers are not reported."
  - "Released checkpoint = predefined Aggressive PTQ recipe + SFT-based QAD at 200 iterations (§4.1)."
  - "Aggressive recipe (Table 12): MoE + shared-expert GEMMs NVFP4; latent down/up proj BF16; router compute FP32; attention QKV/out proj FP8; KV cache FP8; BMM1/BMM2 FP8; Mamba linear FP8; Mamba SSM cache FP16+SR; Mamba conv BF16; embedding BF16; lm_head FP8; vision encoder and VL connector BF16."
  - "Whole-model BPE = 8S/N: S = 74.987 GB stored payload incl. scales, N = 124.453B original params incl. vision, MTP and unquantized modules -> 4.82 BPE; runtime caches/activations excluded (§4.1). Other tiers are policy estimates: BF16 16.00, FP8 8.10, Conservative 5.15, Very aggressive 4.64 (Table 12)."
  - "Very aggressive tier (NVFP4 latent proj / attention / Mamba linear, FP4 KV cache and BMM1, W4A16 lm_head) was explored but not released (Table 12)."
  - "PTQ calibration: 1,024 image-text samples from the post-training SFT data, images passed through the vision encoder; DP x EP over 4 ranks (256 samples each), 512 routed experts and MTP experts split evenly, vision/attention/Mamba replicated (§4.2)."
  - "Weight-scale selection (Table 13, aggressive tier): MSE 100.06% score recovery vs Max 98.58% vs MSE 4-over-6 98.15%; MSE chosen for PTQ and AutoQuant recipes (§4.2)."
  - "AutoQuant sensitivity is highly nonuniform: lm_head accounts for nearly half of NVFP4 sensitivity, attention and Mamba projections next, MoE experts comparatively robust (§4.3)."
  - "QAD: KL-divergence-only objective; only LM GEMM weights with enabled quantizers train; quantizer stats restored from PTQ, not recalibrated; AdamW beta1 0.9, beta2 0.95, eps 1e-8, wd 0.1, clip 1.0; LR linear warmup to 5e-6 over 9,100 source records then constant (§4.4)."
  - "QAD students use NVFP4 KV cache (the only intentional change vs the quantization-reference checkpoints, which use FP8 KV) (§4.4)."
  - "QAD cost: PTQ init ~50 min on 32 GB200 (~27 GPU-hours per candidate); QAD 100 iterations ~2 h on 512 GB200 (1,024 GPU-hours) per candidate (§4.4)."
  - "Exploratory sweep (earlier BF16 baseline, 4.8-BPE recipe, Table 14/15): PTQ 97.68% recovery, -0.22% tokens; best QAD = 200 iters 100% SFT at 99.18% / +1.11% tokens; 75/25 SFT/RL rollout mix gave only modest benefit (§4.4)."
  - "Final downselection (Table 16/17): 4.82-BPE Q200 = 100.12% median recovery, -0.06% verbosity, the strongest candidate; QAD benefits largely saturate by 300 iterations (§4.4)."
  - "Released config headline: 99.52% median accuracy recovery across 35 benchmarks vs BF16 weights with FP8 KV cache; verbosity +0.49% (§4.1, Table 22)."
  - "KV cache (Table 18): NVFP4 KV (4.5 bits/element payload) 99.36% vs FP8 KV (8.0) 99.52% median recovery; NVFP4 KV saves 43.75% ideal KV payload (storage calculation only); release defaults to FP8 KV for runtime-efficiency and software-readiness reasons (§4.5)."
  - "Precision guidance (§4.6): expert weights drive runaway-loop verbosity; q/k/v do not need BF16 but per-tensor FP8 hurts disambiguation retrieval (MRCR-128k), MXFP8 per-block helps; attention out-proj tolerates FP8; Mamba out-proj FP4 weights inflate length; lm_head should stay FP8 or above."
  - "MTP (Super only; Lightning MTP is not quantized): MTP experts NVFP4 W4A4, MTP attention weights BF16 with K/V BMM quantizers FP8; a single FP8 lm_head serves both target verification and MTP drafting (§4.7)."
  - "Selected MTP config (shaded row, Table 20): NVFP4 backbone, FP8 lm_head, NVFP4 MTP experts -> AL 4.063 (DL=5) / 4.520 (DL=7), projected drafting speedup 1.127x at DL=5."
  - "Figure 10 (20-turn agentic proxy, 2K in / 1K out per turn, vLLM): max throughput at >= 50 tok/s/user BF16 -> NVFP4: H100 W4A16 522 -> 970 (1.86x), GB200 W4A4 1,778 -> 4,324 (2.43x); min-latency tokens/s/user: H100 509.131 -> 575.667 (1.13x), GB200 587.453 -> 764.863 (1.30x)."
  - "Contradiction: §4.7 prose says NVFP4 backbone + BF16 MTP has AL 4.612 and quantizing MTP experts drops it to 4.592 (DL=7); Table 20 gives 4.612 for the BF16 backbone + BF16 MTP, 4.592 for NVFP4 backbone + BF16 MTP, and 4.518 after quantizing MTP experts."
related_steps: []
currency: "frozen"
---

# Scope

Answers:
- What precision does the released Nemotron 3.5 Super VL NVFP4 checkpoint use per module, and what is its BPE?
- What were the acceptance criteria, and does the release meet them?
- How was PTQ calibrated, and which weight-scale algorithm was chosen?
- What did AutoQuantize find about sensitivity?
- How was QAD run (objective, data, optimizer, cost), and why 200 iterations on the 4.82-BPE recipe?
- FP8 vs NVFP4 KV cache: what was measured and what ships?
- Which projections are sensitive (attention q/k/v/o, Mamba out-proj, lm_head)?
- How is the MTP drafter quantized, and what does it cost in acceptance length?
- Full BF16 vs NVFP4 benchmark table and serving speedups (W4A4 vs W4A16).

Out of scope: Lightning 3.5 results (§4.8, Table 21; W4A16-only, 99.24% recovery, -3.63% verbosity, all Lightning numbers). Mechanistic ablations (runaway loops, stop-vs-continue margin, MRCR stratification, Mamba out-proj, unembedding): see `quantization-ablations.md`.

# Goals, bookends and acceptance criteria (§4)

- Methods: PTQ and QAD with ModelOpt, on Lightning 3.5 and Super 3.5 VL, relative to the post-trained BF16 checkpoint.
- Recipe development starts from **bookends**: conservative and aggressive (some models also a very aggressive variant). Bookends project runtime gains across the throughput-interactivity Pareto. Default traffic shape: **51200 / 2048 / 1024** pre-cached ISL / new ISL / OSL. Two SLAs:
  - Throughput (tok/s/gpu) at 50 tok/s/usr.
  - Max throughput (usually highest concurrency).
  - Numerical bookend projections are not reported.
- Deployment drives the recipe: **Super 3.5 VL NVFP4 checkpoint deploys W4A4 on Blackwell and W4A16 on non-Blackwell platforms.** (Lightning 3.5 deploys W4A16 on every GPU SKU, optimizing for local/agentic weight-fetch-bound use.)
- KV cache grows linearly with sequence length; Mamba SSM state is constant size. New in Super 3.5 VL: NVFP4 KV cache ablations (§4.5).
- **Acceptance criteria** (aim: highest efficiency / lowest HBM footprint subject to):
  - Median accuracy recovery **>= 99%** — median of NVFP4/BF16 eval-score ratios across several repeats.
  - Verbosity increase **<= 5%** relative to BF16.
- Released checkpoint: 99.52% median recovery, +0.49% verbosity (§4.1, Table 22) — meets both. The NVFP4-KV variant reaches 99.36% (Table 18); its verbosity is not reported.

# Per-module precision tiers (Table 12)

NVFP4 = e2m1-based 4-bit; FP8 = e4m3; W4A16 = NVFP4 weights + 16-bit activations; +SR = stochastic rounding. Reference columns are precision policies, not serving configs. **Aggressive = released.**

| Layer / axis | BF16 (ref) | FP8 (ref) | Conservative | **Aggressive (released)** | Very aggressive (not released) |
|---|---|---|---|---|---|
| C-RADIO encoder | BF16 | BF16 | BF16 | **BF16** | BF16 |
| VL connector | BF16 | BF16 | BF16 | **BF16** | BF16 |
| MoE + shared GEMM | BF16 | FP8 | NVFP4 (a) | **NVFP4** | NVFP4 |
| Latent down/up proj | BF16 | FP8 | BF16 | **BF16** | NVFP4 |
| Router / gate (compute) (b) | FP32 | FP32 | FP32 | **FP32** | FP32 |
| Attention QKV / out proj | BF16 | FP8 | BF16 | **FP8** | NVFP4 |
| KV cache | BF16 | FP8 | FP8 | **FP8** | FP4 |
| BMM1 (QK^T) | BF16 | FP8 | FP8 | **FP8** | FP4 |
| BMM2 (PV) | BF16 | FP8 | FP8 | **FP8** | FP8 |
| Mamba linear | BF16 | FP8 | BF16 | **FP8** | NVFP4 |
| Mamba SSM cache | FP32 | FP32 | FP32 | **FP16+SR** | FP16+SR |
| Mamba convolution | BF16 | BF16 | BF16 | **BF16** | BF16 |
| Embedding | BF16 | BF16 | BF16 | **BF16** | BF16 |
| lm_head | BF16 | FP8 | BF16 | **FP8** | W4A16 |
| Avg. BPE | 16.00 (c) | 8.10 (c) | 5.15 (c) | **4.82** (measured) | 4.64 (c) |

Footnotes (verbatim in substance):
- (a) FP8 was considered for the shared expert during exploration; the Conservative BPE estimate assumes NVFP4 for all shared experts.
- (b) FP32 = router computation, not storage; BPE uses BF16-stored router weights in every column.
- (c) Estimated by applying each column's weight policy to backbone and MTP modules; MTP-specific fusion projection and normalization stay BF16. The released MTP also keeps BF16 attention weights (counted directly in the measured BPE). **Released runtime uses an FP32 convolution cache and FP16+SR SSM cache.**

BPE definition (§4.1): whole-model BPE = 8S/N, S = stored tensor payload in bytes incl. quantization scales, N = original parameter count incl. vision, MTP and unquantized modules. Released: S = 74.987 GB, N = 124.453 B -> **4.82 BPE**. Runtime caches and activations excluded. Whether the released attention FP8 is per-tensor or per-block (MXFP8) is not stated in Table 12 (see §4.6 / Appendix C.3 recommendation for per-block).

# Post-training quantization (§4.2)

Model described as 120B total / 12B active MoE VLM: C-RADIO vision encoder, hybrid Mamba/attention LM backbone with **512 routed experts per MoE layer**, plus an MTP head.

| Aspect | Choice |
|---|---|
| Calibration data | Image-text pairs (not text-only); images go through the vision encoder so the LM sees projected visual tokens. Corpus = post-training SFT data, **1,024 samples** |
| Calibration parallelism | Explored pipeline parallel, DP x EP, FSDP2. PP left only one GPU active at a time -> chose **DP x EP**: 4 ranks x 256 samples; 512 routed experts (and MTP experts) split evenly; vision encoder/projector, dense attention, Mamba layers replicated |
| Weight-scale selection | **MSE** (per-block scale chosen by sweeping FP8 scale values to minimize quantization error), used for PTQ and AutoQuant recipes |

## Weight-scale selection (Table 13, aggressive tier, deltas vs BF16)

| Benchmark | BF16 | MSE | Max | MSE 4-over-6 |
|---|---|---|---|---|
| MMLU-ProX | 76.92 | 73.92 (-3.00) | 72.77 (-4.15) | 74.77 (-2.15) |
| GPQA-D w/o tools | 89.52 | 90.03 (+0.51) | 89.90 (+0.38) | 89.52 (±0.00) |
| APEX Shortlist | 88.56 | 86.97 (-1.60) | 87.37 (-1.20) | 88.70 (+0.13) |
| SciCode Subtask | 47.52 | 48.37 (+0.86) | 46.81 (-0.70) | 46.79 (-0.72) |
| SciCode Problem | 15.77 | 16.54 (+0.77) | 15.77 (±0.00) | 16.35 (+0.58) |
| AA-LCR | 69.00 | 68.06 (-0.94) | 66.50 (-2.50) | 67.00 (-2.00) |
| AA-Omni Acc. | 26.21 | 25.25 (-0.96) | 25.52 (-0.69) | 25.25 (-0.96) |
| AA-Omni Non-Halluci. | 70.09 | 71.19 (+1.10) | 71.98 (+1.89) | 71.50 (+1.41) |
| HLE w/o tools | 31.84 | 31.70 (-0.14) | 30.21 (-1.62) | 31.14 (-0.70) |
| Tau3 Banking | 19.46 | 20.10 (+0.64) | 18.94 (-0.52) | 18.94 (-0.52) |
| Baby Vision | 29.81 | 29.47 (-0.34) | 29.81 (±0.00) | 28.69 (-1.12) |
| V* w/o CI | 85.69 | 86.21 (+0.52) | 86.91 (+1.22) | 85.51 (-0.17) |
| **Score Recovery** | – | **100.06%** | 98.58% | 98.15% |

Cross-model note (from the Nemotron 3 Ultra report, not stated in this report): Nemotron 3 Ultra chose max-calibrated Four-Over-Six for routed-expert weight scales; Super 3.5 VL chose plain MSE. These are different selections on different models — do not transfer.

# Sensitivity-driven mixed precision: AutoQuant (§4.3)

- ModelOpt AutoQuantize replaces hand-written layer rules. For each quantizable module i and format q in {NVFP4, FP8, BF16}, sensitivity S_i(q) = quantization-induced output error weighted by the squared task-loss gradient.
- Solves: minimize sum_i S_i(q_i) subject to sum_i n_i b(q_i) <= b̄ · sum_i n_i (Eq. 3), where n_i = #weights, b(q) includes the format's modeled storage overhead, b̄ = target effective bitwidth.
- Grouped decisions: operators that must share a deployment format (Q/K/V projections; all routed experts within an MoE layer) are one decision. Held outside the search: vision encoder, cache quantizers, fixed MTP recipe.
- **Sensitivity profile:** lm_head ≈ nearly half of total NVFP4 sensitivity; attention and Mamba projections next; MoE experts comparatively robust. Assignments therefore put NVFP4 on expert weights, FP8/BF16 on attention and Mamba projections, higher precision on lm_head.
- AutoQuant candidates were carried into QAD (§4.4) but the released checkpoint uses the fixed Aggressive recipe. Appendix C.3 notes the AutoQuant search left q/k/v BF16 on 3 of 8 attention layers under both 4.85 and 4.90 bit-per-parameter budgets (see Contradictions re: 4.8/4.9 vs 4.85/4.90 labels).

# Quantization-aware distillation (§4.4)

## Setup

| Aspect | Value |
|---|---|
| Student init | Calibrated PTQ checkpoint; quant config derived from a reference checkpoint (AutoQuant or fixed-recipe PTQ). Only intentional change: **NVFP4 KV cache** in the student (references use FP8 KV). All §4.4 results are NVFP4-KV deployments |
| Student calibration | 512 packed multimodal samples; variable-length records packed up to 524,288 tokens; packing buffer 5,000 source samples |
| PTQ init cost | 8 nodes x 4 GB200 (32 GPUs), ~50 min per candidate (~27 GB200 GPU-hours) |
| Data blends | 100% SFT (post-training SFT blend) vs 75% SFT / 25% teacher-generated RL rollouts (16 trajectories per source record via vLLM, original RL task workflows incl. environment/tool interactions; prompt, history, tool-output and prefilled prefix tokens masked). Rollout mix gave only modest benefits |
| Trainable params | Only LM GEMM weights with enabled quantizers. Frozen: token embeddings, unquantized LM params, norms, vision encoder + projector, MTP module. Router-bias updates disabled |
| Quantizer state | Config and calibrated statistics restored from PTQ, not recalibrated |
| Loss | KL divergence only. Disabled: hard-target CE, MTP CE, MoE load-balancing aux loss; no intermediate-feature distillation |
| Optimizer | Distributed AdamW, beta1 0.9, beta2 0.95, eps 1e-8, weight decay 0.1, grad clip global norm 1.0; fresh optimizer per experiment |
| LR | Linear 0 -> 5e-6 over 9,100 original source records, then constant |
| Compute | 128 nodes x 4 GB200 (512 GPUs); 100 iterations ≈ 2 h (1,024 GB200 GPU-hours) per candidate |

Recipes evaluated: (a) 4.8 BPE NVFP4 (from an AutoQuant run's config), (b) 4.9 BPE NVFP4 ("similarly, based on a 4.8 BPE target AutoQuant" — wording as in source), (c) fixed 4.82 BPE NVFP4 recipe.

## Exploratory sweep (Tables 14/15; earlier, pre-final BF16 baseline; 4.8-BPE recipe)

| Checkpoint | Median score recovery (Table 14) | Verbosity change, total tokens (Table 15) |
|---|---|---|
| BF16 | 100.00% | 0.00% |
| PTQ | 97.68% | -0.22% |
| QAD 200, 100/0 SFT/RL | **99.18%** | **+1.11%** |
| QAD 200, 75/25 | 98.36% | +6.24% |
| QAD 400, 100/0 | 98.36% | +4.44% |
| QAD 400, 75/25 | 99.12% | +2.39% |
| QAD 600, 100/0 | 99.16% | +4.67% |
| QAD 600, 75/25 | 98.61% | +2.93% |

Benchmarks (13): MMLU-ProX, GPQA-D w/o tools, APEX Shortlist, SciCode subtask/problem, AA-LCR, AA-Omni accuracy / non-halluc., HLE w/o tools, Tau3 Banking, BabyVision, CharXiv-RQ, V* w/o CI. Conclusion: best tradeoff = 200 iterations on 100% SFT (99.18%, +1.50 pp over PTQ, +1.11% tokens); more iterations not uniformly beneficial; 75/25 best at the intermediate (400) checkpoint; favor shorter QAD with 100% SFT; optimal duration depends on data mix.

## Final downselection (Tables 16/17; final BF16 checkpoint; 100% SFT only; checkpoints every 100 iters)

| Recipe | Checkpoint | Median score recovery (Table 16) | Verbosity change (Table 17) |
|---|---|---|---|
| BF16 base | – | 100.00% | 0.00% |
| 4.8 BPE | PTQ / Q200 / Q300 | 100.00% / 99.14% / 99.57% | -1.10% / +2.30% / +0.75% |
| **4.82 BPE** | PTQ / **Q200** / Q300 | 98.98% / **100.12%** / 99.71% | -0.53% / **-0.06%** / -1.21% |
| 4.9 BPE | PTQ / Q200 / Q300 | 99.75% / 98.56% / 99.73% | -0.62% / +1.79% / +0.06% |

- 4.8 and 4.9 BPE improve from 200 -> 300 iterations; 4.82 BPE is best at 200 (300 is less accurate, slightly less verbose).
- **4.82 BPE at 200 iterations = strongest candidate** (selected). The best checkpoint of each other recipe also went to a full-benchmark sweep (results for those not reported).
- 400-iteration checkpoints were evaluated but omitted from Tables 16/17; they do not alter downselection. QAD benefits largely saturate by 300 iterations.
- Selected Table 16 rows (BF16 / 4.82 Q200): MMLU-ProX 76.92 / 74.49; GPQA-D 89.52 / 89.71; AA-LCR 69.00 / 69.38; HLE 31.84 / 30.91; Tau3 Banking 19.46 / 18.56; V* 85.69 / 86.91.

Note: Table 16/17 numbers are on the NVFP4-KV QAD student; the released checkpoint is served with FP8 KV (see next section) and its final numbers are Table 22.

# KV cache quantization (§4.5, Table 18)

- Both K and V caches across attention layers in FP8 or NVFP4. NVFP4 = 4-bit E2M1 + one FP8 E4M3 scale per 16 elements = **4.5 bits/element** (excluding tensor-level scale); vs FP8 8 bits -> ideal payload reduction 1 - 4.5/8 = **43.75%**. Storage calculation only, not measured runtime traffic/latency. KV BPE is separate from whole-model BPE.
- Reference: BF16 weights + FP8 KV cache. Both test configs: aggressive NVFP4 recipe + QAD 200 iters. Both use FP32 Mamba conv cache and FP16+SR SSM cache.
- NVFP4-KV run uses the original QAD checkpoint with fixed backbone KV scales targeting NVFP4 (1/6). FP8-KV run uses the released checkpoint, which contains **no KV scales** (vLLM defaults to KV scale 1.0 for FP8 KV).
- Median recovery = median of 35 config-to-reference ratios; reflects full weight + cache config, not KV alone.

| Metric | NVFP4 KV | FP8 KV (released) |
|---|---|---|
| KV cache payload BPE | 4.5 | 8.0 |
| Median accuracy recovery | 99.36% | 99.52% |
| MMLU-ProX | 74.47 | 73.76 |
| APEX Shortlist w/o tools | 87.30 | 89.36 |
| AA-LCR | 68.16 | 67.69 |
| GraphWalks (<=1M) | 89.54 | 90.34 |
| HLE w/o tools | 30.63 | 31.44 |
| HLE Vision | 27.68 | 29.00 |
| Tau2 Telecom | 69.35 | 73.19 |
| Tau3 Banking | 18.88 | 17.78 |
| Terminal-Bench v2.1 | 60.67 | 62.78 |
| SWE-Bench Verified | 70.53 | 71.07 |
| LCB v6 | 87.37 | 87.41 |

(Selected rows; Table 18 has 35 benchmarks. The FP8-KV column equals the NVFP4 column of Table 22.)

- Ships with **FP8 KV** "for runtime-efficiency and software-readiness considerations despite NVFP4 KV's storage advantage". Mixed FP8-key/NVFP4-value configurations not evaluated. §4 intro claims NVFP4 KV + quantized SSM meets acceptance criteria (99.36% >= 99%; NVFP4-KV verbosity not reported).

# Attention and projection precision (§4.6)

Ablations assume W4A4 routed/shared experts + FP8 KV cache as default and vary one projection family. Findings (mechanisms in `quantization-ablations.md`):

- **MoE expert weights** are the primary accuracy-impacting component, drifting token distribution in FP4 latent space; critical for **runaway loops** when an abstention decision boundary is fragile. No attention-projection precision recovers the verbosity inflation. Promoting expert precision and fine-tuning on RL rollouts can mitigate (App. C.2).
- **Attention q/k/v:** long-context retrieval is dominated by KV cache precision, so projections matter little in most cases; but attention-score dilution (many keys sharing one representable quantized score level) hurts **disambiguation** (MRCR-128k) at 32k+ context. q/k/v do not need BF16, but per-tensor FP8 dynamic range limits logit granularity; **MXFP8 per-block scaling mitigates**. Attention out-proj is structurally exposed but tolerant to FP8 (App. C.3).
- **Mamba output projection:** minimal accuracy impact, but FP4 **weights** inflate generation length. Input activations separately exposed: the SSM gate's near-zero-but-nonzero values flush to zero under FP4 activation quant; rotation or higher precision helps activations but not the verbosity inflation without weight realignment (App. C.4).
- **lm_head:** FP4 does not change accuracy but amplifies perturbation of the stop-vs-continue margin; **keep lm_head at FP8 weights or above** (App. C.5).

## Table 19 — long-context capability vs q/k/v precision (Δ pp vs BF16 reference; all quantized models FP8 KV + W4A4 experts)

C = distinction (content), P = disambiguation (position), S = aggregation (summary). t = mean(Δ)/SE(Δ), SE over tasks.

| Benchmark | Rollouts | q/k/v BF16 | q/k/v MXFP8 | q/k/v FP8 |
|---|---|---|---|---|
| GraphWalks (<=1M) C,S | 950 x 1 | -0.52 (t=-0.70) | +0.15 (t=+0.21) | -0.83 (t=-1.03) |
| MMLongBench-Doc C,S | 1091 x 3 | -0.76 (t=-1.42) | -0.90 (t=-1.81) | -0.34 (t=-0.61) |
| AA-LCR S | 100 x 16 | +1.12 (t=+0.89) | -0.12 (t=-0.10) | +1.56 (t=+1.19) |
| **MRCR-128k P** | 1470 x 3 | **-2.81** (t=-3.90) | **-2.65** (t=-3.61) | **-3.63** (t=-4.97) |

# MTP quantization (§4.7, Table 20)

- Applies **only to Super 3.5**; Lightning's MTP is not quantized.
- MTP = one attention layer + one MoE layer; no own output projection — the checkpoint stores a single lm_head shared by target verification and MTP drafting.
- Profiling: MoE experts and lm_head dominate drafting latency -> MTP experts quantized to **NVFP4 W4A4**; MTP attention weights stay **BF16**, only K/V BMM quantizers at **FP8**.
- Simulated latency (draft length 5, one GB200, quantized backbone), MTP and lm_head unquantized: drafting = **23.5%** of total latency, split **49.6% MTP / 51.4% lm_head** (sums to 101.0% as printed). Projected reductions: quantizing MTP layer -47% of its latency; lm_head FP8 -49%, W4A16 -71%.
- Target lm_head quantization was initially seen as higher risk; fallback = separate target / MTP lm_head precisions, implemented in **vLLM 0.30.0** (target lm_head BF16, draft lm_head lower precision). Final choice: **FP8 target lm_head**, so one FP8 lm_head serves both; the separate-precision support remains for models where quantizing the target lm_head hurts quality.
- **KV-cache coupling:** MTP cannot use a KV precision different from the backbone; empirically validated not to degrade AL (structural constraint, not a quality trade-off).
- **Construction:** boosted MTP (§3.7) quantized separately and **spliced** into the selected quantized backbone (weights + scales; MTP tensor names, shapes, dtypes must match exactly). Co-quantizing MTP with the backbone is the preferred direction for future releases.

Table 20 — SPEED-Bench qualitative split, temperature 1; speedup projected at DL=5 relative to NVFP4 backbone with unquantized MTP and lm_head. Shaded (selected) row confirmed from the PDF rendering.

| Backbone | Target LM head | MTP experts | MTP LM head | AL DL=5 | AL DL=7 | Projected speedup (DL=5) |
|---|---|---|---|---|---|---|
| BF16 | BF16 | BF16 | BF16 | 4.115 | 4.612 | - |
| NVFP4 | BF16 | BF16 | BF16 | 4.109 | 4.592 | 1 |
| NVFP4 | BF16 | NVFP4 | BF16 | 4.066 | 4.518 | 1.043 |
| NVFP4 | BF16 | NVFP4 | FP8 | 4.069 | 4.514 | 1.114 |
| NVFP4 | BF16 | NVFP4 | W4A16 | 4.031 | 4.469 | 1.136 |
| **NVFP4** | **FP8** | **NVFP4** | **FP8** | **4.063** | **4.520** | **1.127** (selected) |
| NVFP4 | FP8 | NVFP4 | W4A16 | 4.025 | 4.471 | 1.150 |

**4.612 vs 4.592:** 4.612 is the BF16 backbone with the BF16 boosted MTP at DL=7, temperature 1 — the same number as Table 8 (§3.7, "+25k MTP Boosting iterations", temp=1). 4.592 is the NVFP4 backbone with that BF16 MTP (Table 20 row 2). So moving the backbone to NVFP4 costs 0.020 AL at DL=7; quantizing the MTP experts then drops it to 4.518, and the released config (adds FP8 lm_head on both sides) is 4.520. The §4.7 prose labels these differently — see Contradictions. §5.6 Vera Rubin serving uses a fixed AL of 4.06 at DL=5, consistent with the selected row's 4.063.

# Results: BF16 vs released NVFP4 (§4.9, Table 22)

Same evaluation/serving params for both; both served with FP8 KV cache; evaluated via NeMo Gym; each benchmark run twice and averaged (in addition to per-benchmark repeats).
- **Median Accuracy Recovery:** average accuracy over completed runs per checkpoint, ratio NVFP4/BF16 per benchmark, median across benchmarks.
- **Ratio of Totals Verbosity:** average output tokens per trajectory, summed across benchmarks per checkpoint, % change of NVFP4 total vs BF16 total.
- Terminal-Bench: +0.2% mean output tokens/trajectory over agent-completed rollouts, +0.8% median over all rollouts, but **+15.7%** mean over all rollouts because ~4.5% non-completing runaway trajectories (~700k tokens each) dominate. Non-completed TB rollouts are excluded; including them raises the verbosity change to **2.99%**.

| Category | Benchmark | BF16 tok/traj | NVFP4 tok/traj | BF16 acc | NVFP4 acc | Δ (pp) | Recovery (%) |
|---|---|---|---|---|---|---|---|
| Agentic | SWE-Bench Verified | 22,353 | 21,166 | 71.10 | 71.07 | -0.03 | 99.95 |
| Agentic | SWE-Bench Multilingual | 29,710 | 30,610 | 67.72 | 67.00 | -0.72 | 98.93 |
| Agentic | SWE-Bench Pro | 25,630 | 25,036 | 51.78 | 51.30 | -0.48 | 99.08 |
| Agentic | Tau2 Telecom | 5,784 | 5,659 | 73.19 | 73.19 | 0.00 | 100.00 |
| Agentic | Tau2 Retail | 4,028 | 3,983 | 69.24 | 69.30 | +0.05 | 100.08 |
| Agentic | Tau2-AA (Airline) | 5,426 | 5,224 | 76.50 | 75.75 | -0.75 | 99.02 |
| Agentic | Tau3 Banking | 12,173 | 12,065 | 18.88 | 17.78 | -1.10 | 94.20 |
| Agentic | Terminal-Bench v2.1 | 65,560 | 65,717 | 60.67 | 62.78 | +2.11 | 103.47 |
| Agentic | GDPVal | 49,705 | 52,688 | 49.98 | 55.74 | +5.76 | 111.53 |
| Coding | SciCode (subtask) | 6,621 | 6,796 | 47.13 | 46.61 | -0.52 | 98.89 |
| Coding | SciCode (problem) | 6,621 | 6,796 | 15.10 | 14.13 | -0.96 | 93.63 |
| Coding | LiveCodeBench v6 | 15,449 | 15,733 | 88.22 | 87.41 | -0.81 | 99.08 |
| Coding | BIRD (text2sql, no tools) | 637 | 573 | 59.58 | 60.60 | 1.02 | 101.71 |
| Math & Science | GPQA Diamond (no tools) | 14,383 | 14,738 | 89.52 | 89.90 | +0.38 | 100.42 |
| Math & Science | APEX Shortlist (no tools) | 68,050 | 66,926 | 88.36 | 89.36 | +1.00 | 101.13 |
| Math & Science | HLE (no tools, text-only) | 39,253 | 40,170 | 31.79 | 31.44 | -0.35 | 98.91 |
| Math & Science | AA-Omniscience Accuracy | 4,373 | 4,549 | 26.20 | 25.86 | -0.34 | 98.70 |
| Math & Science | AA-Omni Non-Hallucination | 4,373 | 4,549 | 69.96 | 72.33 | +2.36 | 103.38 |
| Multimodal | Baby Vision | 11,839 | 11,634 | 29.60 | 29.51 | -0.09 | 99.71 |
| Multimodal | CharXiv (RQ) | 4,157 | 3,930 | 80.82 | 79.88 | -0.93 | 98.85 |
| Multimodal | MathVision | 7,434 | 7,585 | 78.16 | 77.79 | -0.37 | 99.52 |
| Multimodal | OCR Reasoning | 5,362 | 5,300 | 65.08 | 65.09 | +0.02 | 100.02 |
| Multimodal | OCRBenchV2 EN | 1,860 | 1,919 | 68.12 | 68.34 | +0.22 | 100.32 |
| Multimodal | OCRBenchV2 ZH | 2,688 | 2,867 | 59.15 | 59.03 | -0.11 | 99.81 |
| Multimodal | ScreenSpot Pro | 236 | 324 | 43.53 | 42.25 | -1.28 | 97.07 |
| Multimodal | Video-MME-v2 (w/o sub.) | 4,579 | 5,046 | 32.85 | 32.54 | -0.31 | 99.05 |
| Multimodal | V* | 855 | 732 | 85.60 | 85.95 | +0.35 | 100.41 |
| Multimodal | HLE (full) | 36,564 | 37,365 | 29.20 | 29.00 | -0.20 | 99.32 |
| Multimodal | MMLongBench-Doc | 3,365 | 3,484 | 62.82 | 62.82 | 0.00 | 100.00 |
| Long context | AA-LCR | 4,601 | 4,228 | 68.62 | 67.69 | -0.94 | 98.63 |
| Long context | GraphWalks (<=1M) | 10,635 | 10,135 | 89.71 | 90.34 | +0.63 | 100.70 |
| IF & Chat | IFBench | 10,016 | 9,027 | 70.86 | 71.50 | +0.64 | 100.90 |
| IF & Chat | NV Chat Benchmark | 3,835 | 3,977 | 52.67 | 51.93 | -0.74 | 98.60 |
| Multilingual | MMLU-ProX | 3,478 | 3,460 | 76.91 | 73.76 | -3.15 | 95.90 |
| Multilingual | WMT24++ (en->xx) | 1,175 | 1,194 | 52.32 | 51.94 | -0.38 | 99.27 |
| Multilingual | WMT24++ COMET | 1,175 | 1,194 | 84.80 | 84.76 | -0.04 | 99.95 |
| | **Median recovery** | | | | | | **99.52%** |
| | **Verbosity change (ratio of totals)** | | | | | | **+0.49%** |

Largest regressions: MMLU-ProX -3.15 pp (95.90%), Tau3 Banking -1.10 pp (94.20%), SciCode problem -0.96 pp (93.63%), ScreenSpot Pro -1.28 pp. Largest gains: GDPVal +5.76 pp, AA-Omni non-hallucination +2.36 pp, Terminal-Bench +2.11 pp.

# Serving speedup: W4A4 vs W4A16 (Figure 10)

![Nemotron Super 3.5 NVFP4 vs BF16 serving with vLLM: max throughput at >= 50 TPS/user and generation TPS/user at minimum latency, H100 (W4A16) and GB200 (W4A4) (Figure 10).](../../../docs/assets/super35/nvfp4-serving.png)

vLLM, 20-turn agentic proxy workload, 2K input + 1K output tokens per turn.

| Platform (NVFP4 mode) | Max throughput at >= 50 tok/s/user, output tok/s/GPU, BF16 -> NVFP4 | Generation tok/s/user at min latency, BF16 -> NVFP4 |
|---|---|---|
| H100 80GB (W4A16) | 522 -> 970 (1.86x) | 509.131 -> 575.667 (1.13x) |
| GB200 (B200 GPU) (W4A4) | 1,778 -> 4,324 (2.43x) | 587.453 -> 764.863 (1.30x) |

Ratios recomputed from the printed values match the printed multipliers. The report says only "W4A16 on non-Blackwell platforms"; it does not state the hardware reason in §4 (the Ultra report attributes it to Hopper lacking native FP4 tensor cores). H100 is the only non-Blackwell platform measured here.

# Contradictions and inconsistencies in the source

1. **MTP AL labels (§4.7 vs Table 20).** Prose: "Against the NVFP4 backbone with a BF16 MTP (AL = 4.612 at draft length 7), quantizing the MTP experts reduces AL to AL = 4.592." Table 20: 4.612 = BF16 backbone + BF16 MTP; 4.592 = NVFP4 backbone + BF16 MTP; NVFP4 MTP experts = 4.518. The prose appears shifted by one row; trust the table (which also matches §3.7 Table 8's 4.612).
2. **Drafting-latency split (§4.7):** 49.6% MTP + 51.4% lm_head = 101.0%.
3. **QAD "100 iterations" (§4.4):** "For the 100% SFT experiments, the 100 iterations checkpoint provides the strongest overall accuracy-verbosity tradeoff", but the exploratory sweep (Tables 14/15) only has 200/400/600-iteration checkpoints, and the same paragraph names 200 iterations as best.
4. **Recipe BPE labels:** §4.4 calls the AutoQuant recipes 4.8 and 4.9 BPE (with (b) "based on a 4.8 BPE target AutoQuant"); Appendix C.3 calls the AutoQuant budgets 4.85 and 4.90 bits-per-parameter. Not reconciled.
5. **Verbosity metric wording:** §4.1 says "mean per-benchmark verbosity increases by 0.49%"; Table 22 and §4.9 label +0.49% as Ratio of Totals. Recomputing from Table 22 token columns: ratio of totals ≈ +0.49% (matches the label), whereas the mean of per-benchmark changes ≈ +0.78%. The ratio-of-totals definition is the one consistent with the numbers.
6. **Benchmark count:** §4.1 and Table 18 say 35 benchmarks; Table 22 lists 36 rows. Recomputing, the median over the 35 rows excluding BIRD (absent from Table 18) is exactly 99.52%; including BIRD gives 99.615%. So the headline 99.52% is over the Table 18 set and BIRD is not in it — the report does not say so.
7. **AA-LCR BF16:** Table 22 lists 68.62; the main evaluation suite (Table 10) lists 68.63 for Super 3.5.
8. Observation (not proven inconsistent): Table 22's BF16 values for Tau3 Banking (18.88) and Terminal-Bench v2.1 (60.67) are identical to the NVFP4-KV column of Table 18. May be coincidence; flag if citing either BF16 value.

# Caveats

- Tables 14/15 use an earlier (pre-final) BF16 baseline; Tables 16/17 use the final BF16 checkpoint; both quantized sides there use NVFP4 KV. Only Table 22 (and Table 18's FP8 column) describe the released FP8-KV configuration.
- Table 12 BPE values other than Aggressive are policy-based estimates, not measured checkpoints.
- KV-cache 43.75% is an ideal storage calculation, not a measured memory-traffic or latency saving.
- Table 20 speedups are simulated/projected drafting speedups at DL=5 on one GB200, not end-to-end measured throughput.
- Bookend Pareto projections (51200/2048/1024 traffic) are described but not reported numerically.
- Eval results are subject to run-to-run variation from stochastic sampling and judge variability; scores may differ from model cards (footnote to §4.8/4.9).
- No Mamba SSM cache precision ablation is reported for Super 3.5 in §4; FP16+SR appears only as the Table 12 policy, the Table 12 runtime footnote, and the §4.5 serving setup. (Lightning keeps an FP32 SSM cache, §4.8.)
