---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "quantization"
paper_sections: ["4", "4.7", "4.8", "Table 21", "Appendix C.3"]
title: "Nemotron 3.5 Lightning NVFP4: W4A16 on Every SKU, 99.24% Median Recovery, Report vs Repo Recipe"
summary: |
  Lightning's NVFP4 quantization. Report (§4, §4.8, Table 21): a single NVFP4 checkpoint that deploys as
  W4A16 on every GPU SKU, chosen for local and agentic use (minimize weight fetch, keep activations at
  higher precision). Routed experts, shared experts and lm_head are NVFP4 weights with 16-bit
  activations, Mamba in/out projections are FP8 W+A, attention projections stay BF16, and the MTP module
  is not quantized. Evaluated with FP8 KV and FP32 SSM cache: 99.24% median recovery, -3.63% verbosity
  across Table 21's benchmarks. The NVFP4 model card confirms FP8 Mamba in/out projections and names
  four-over-six NVFP4 calibration (1000 samples of a Nemotron Ultra validation subset at 32K). The repo
  docs' ModelOpt recipe (four-over-six on 1,000 public samples, then 200 QAD iterations) instead puts the
  Mamba projections in W4A16 NVFP4; the docs flag this as unreconciled. Checkpoint 22 GB vs 66 GB BF16.
key_facts:
  - "§4: PTQ and QAD with ModelOpt are applied to Lightning 3.5 and Super 3.5 VL. Acceptance criteria for both: median accuracy recovery >= 99% (median of NVFP4/BF16 score ratios across repeats) and verbosity increase <= 5% vs BF16."
  - "§4: 'Lightning 3.5 is designed to excel in local, agentic use cases as well as datacenter deployments. We therefore quantize Lightning 3.5 to minimize weight fetch and raise activation precision to maintain accuracy. Lightning 3.5 deploys to W4A16 regardless of GPU SKU' (Super: W4A4 on Blackwell, W4A16 elsewhere)."
  - "§4.8 recipe: routed experts, shared experts and language-model output head = 4-bit NVFP4 weights with 16-bit activations; Mamba input and output projections = FP8 weights and activations; attention projections remain BF16. Selected 'using the same PTQ/QAD procedure' (as Super); no Lightning-specific PTQ/QAD hyperparameters, calibration method, BPE or checkpoint size are reported."
  - "§4.7: the MTP module of Lightning is not quantized (only Super's MTP is)."
  - "§4.8 eval serving: both BF16 and NVFP4 served with FP8 KV cache and FP32 Mamba SSM cache; 'SSM cache is kept in FP32 as it was not a limiter on DGX Spark for low batch (< 4) use cases.'"
  - "Table 21 headline: median accuracy recovery 99.24% (CritPt excluded because BF16 = 0); verbosity change (ratio of totals) -3.63%. Both meet the §4 criteria. Table re-checked against the PDF (page 41)."
  - "Table 21 extremes: lowest recovery Terminal-Bench v2.1 90.22% (24.44 -> 22.05), LM Arena Proxy 92.01%, AA-Omniscience Accuracy 93.61%; highest GDPval 111.01% (17.19 -> 19.09), IFBench 102.84%, Tau2 Telecom 102.21%. SWE-Bench Verified and BrowseComp unchanged (100.00%)."
  - "Appendix C.3 (Lightning): quantization's MRCR-128k disambiguation loss also appears on Lightning; moving the KV cache FP8 -> BF16 (weights and projections fixed) recovers +1.43 pp (2.4 sigma), moving projections FP8 -> BF16 recovers +2.97 pp (2.8 sigma). C limitation 5: the KV cache is identified as the damage carrier in the Lightning study."
  - "Repo docs (docs/nemotron/lightning35/quantization.md, model card / repo docs): PTQ with ModelOpt recipe `lightning_w4a16_nvfp4_4o6`; five PTQ recipes tried (max, mamba_fp8_max, MSE, four_over_six, four_over_six + NVFP4 KV); all quantize lm_head to W4A16 and keep attention projections BF16; four_over_six (static NVFP4, 4/6 calibration) 'with W4A16 Mamba linears' chosen for the published checkpoint."
  - "Repo docs QAD: frozen-scale QAD (scales from the MSE-family PTQ kept fixed), KL loss on logits against the frozen BF16 teacher; seq length 32,768, MBS 1, GBS 64, LR 2e-5 -> min 5e-6, 30 warm-up iters, 200 iterations (~419M tokens), TP=1 / EP=16 / CP=4 on 8 nodes x 4 GPUs; calibration 1,000 samples of cnn_dailymail + Nemotron-Post-Training v2 at seq length 32,768; public 13-dataset QAD blend."
  - "Size: 22 GB NVFP4 vs 66 GB BF16 (repo docs; matches the HF safetensors sizes). No throughput or latency number for Lightning in the report or the model cards."
  - "NVFP4 model card PTQ recipe: 'Four Over Six NVFP4 (a variant of static MSE calibration) W4A16 on routed and shared experts, FP8 per-tensor dynamic scales on mamba in_proj/out_proj and KV cache', calibrated on 1000 samples of a Nemotron Ultra validation subset at 32k tokens. This confirms the report's FP8 Mamba projections and names the calibration the report leaves out."
  - "Contradiction (Mamba projections): report §4.8 and NVFP4 model card = FP8; repo docs' chosen four_over_six recipe = W4A16 NVFP4 (FP8 Mamba only in the non-chosen `mamba_fp8_max`). The docs state this is not reconciled."
  - "Calibration: report says only 'the same PTQ/QAD procedure' as Super (Super's weight-scale method is MSE, Table 13); NVFP4 model card and repo docs both say four-over-six. The docs run it on 1,000 samples of the public cnn_nemotron_v2_mix rather than the card's Nemotron Ultra validation subset."
  - "Repo docs: development ablations indicated QAD should run at the post-training SFT sequence length (roughly 522K tokens) to preserve long context; the documented QAD command uses 32,768 and the docs say it will not reproduce that setting. The report gives no Lightning SFT or QAD sequence length."
  - "Serving (usage cookbook): vLLM on H100 runs the Mamba SSM cache in FP16 with stochastic rounding (5 Philox rounds) for memory; DGX Spark uses FP8 KV cache. Both differ from the report's FP32 SSM evaluation setting."
related_steps: []
currency: "frozen"
---

# Scope

- How is the released Lightning NVFP4 checkpoint quantized, module by module?
- How much accuracy does it keep, per benchmark?
- How does the repo's PTQ/QAD recipe compare with the report?

# Design (report §4)

| Item | Lightning 3.5 | Super 3.5 VL (contrast only) |
| --- | --- | --- |
| Deployment | W4A16 on every GPU SKU | W4A4 on Blackwell, W4A16 elsewhere |
| Rationale | Local + agentic + datacenter; minimize weight fetch, keep activations higher precision | Throughput at 50 tok/s/user and max throughput |
| Acceptance criteria | Median recovery >= 99%, verbosity <= +5% | same |
| MTP module | Not quantized (§4.7) | NVFP4 W4A4 experts |
| Results section | §4.8, Table 21 | §4.9, Table 22 |

# Released W4A16 recipe (report §4.8)

| Module | Precision |
| --- | --- |
| Routed experts | NVFP4 weights, 16-bit activations |
| Shared experts | NVFP4 weights, 16-bit activations |
| lm_head (output head) | NVFP4 weights, 16-bit activations |
| Mamba input / output projections | FP8 weights and activations |
| Attention projections | BF16 |
| MTP module | not quantized (§4.7) |
| KV cache (as evaluated) | FP8 (BF16 baseline also FP8 KV) |
| Mamba SSM cache (as evaluated) | FP32 |

Not reported for Lightning (in the report): calibration algorithm and sample count, QAD iterations/data/LR, BPE, checkpoint
size, router/embedding/conv precision, any throughput or latency number. The NVFP4 model card supplies the
calibration (four-over-six, 1000 samples, Nemotron Ultra validation subset, 32K).

Note that Super's guidance (§4.6, keep lm_head at FP8 or above to protect verbosity) does not apply here:
Lightning's lm_head is NVFP4 W4A16, and its verbosity went down (-3.63%).

# Table 21: BF16 vs W4A16 NVFP4 (re-checked against the PDF)

| Benchmark | BF16 tok/traj | NVFP4 tok/traj | BF16 | NVFP4 | Delta (pp) | Recovery (%) |
| --- | --- | --- | --- | --- | --- | --- |
| SWE-Bench Verified | 24,569 | 23,414 | 52.20 | 52.20 | 0.00 | 100.00 |
| SWE-Bench Multilingual | 31,071 | 33,924 | 37.13 | 37.00 | -0.13 | 99.65 |
| Tau2 Telecom | 7,076 | 7,075 | 59.65 | 60.96 | +1.32 | 102.21 |
| Tau3 Banking | 19,107 | 17,998 | 9.90 | 9.48 | -0.41 | 95.83 |
| Terminal-Bench v2.1 | 76,219 | 69,523 | 24.44 | 22.05 | -2.39 | 90.22 |
| GDPval | 52,043 | 52,563 | 17.19 | 19.09 | +1.89 | 111.01 |
| PinchBench | 7,206 | 7,115 | 83.86 | 84.78 | +0.92 | 101.09 |
| BrowseComp | 142,589 | 138,151 | 39.50 | 39.50 | 0.00 | 100.00 |
| SciCode (subtask) | 17,190 | 17,428 | 32.74 | 31.53 | -1.21 | 96.30 |
| GPQA Diamond (no tools) | 15,552 | 14,394 | 76.45 | 75.38 | -1.07 | 98.60 |
| HLE (no tools) | 30,352 | 27,270 | 10.80 | 10.70 | -0.09 | 99.14 |
| AA-Omniscience Accuracy | 2,505 | 2,588 | 17.47 | 16.35 | -1.12 | 93.61 |
| AA-Omni Non-Hallucination | 2,505 | 2,588 | 68.88 | 69.20 | +0.32 | 100.46 |
| CritPt | 4,379 | 3,622 | 0.00 | 0.00 | 0.00 | - |
| AA-LCR | 7,679 | 7,388 | 52.38 | 50.00 | -2.38 | 95.47 |
| RULER-v1 (1M) | 22 | 21 | 91.52 | 90.83 | -0.69 | 99.24 |
| IFBench (loose) | 13,069 | 11,422 | 72.00 | 74.04 | +2.04 | 102.84 |
| LM Arena Proxy | 3,401 | 3,496 | 12.06 | 11.10 | -0.96 | 92.01 |
| **Median recovery** | | | | | | **99.24%** |
| **Verbosity change (ratio of totals)** | | | | | | **-3.63%** |

Table notes (from the report): scores, deltas and recoveries rounded to two decimals, calculations use
unrounded scores; CritPt excluded from the median because BF16 = 0. Token columns are average tokens per
trajectory. Footnote 19: scores may differ from the model cards because of sampling and judge variability.

Check: the 17 defined recoveries have median 99.24% (the 9th value, RULER-v1). Five benchmarks
recover below 96% (Terminal-Bench, LM Arena Proxy, AA-Omniscience Accuracy, AA-LCR, Tau3 Banking). The
largest token reduction is Terminal-Bench (76,219 -> 69,523).

# Long-context mechanism (Appendix C.3, Lightning data points)

- Quantization costs disambiguation-style retrieval (MRCR-128k) on Super, and "the same effect is also
  observed on Nemotron 3.5 Lightning".
- Lightning: KV cache FP8 -> BF16 (weights and projections fixed) recovers +1.43 pp (2.4 sigma) of
  MRCR-128k; projections FP8 -> BF16 recover +2.97 pp (2.8 sigma).
- Limitation 5: the KV cache is identified as the damage carrier in the Lightning study (Super's
  ablation does not test BF16 KV).
- Which "projections" were FP8 in the Lightning ablation is not specified. The released recipe keeps
  attention projections in BF16 (§4.8).

# Repo recipe (model card / repo docs)

Source: `docs/nemotron/lightning35/quantization.md` (runs from Model Optimizer `examples/megatron_bridge`).

## PTQ recipes tried

| Recipe | MoE / shared / lm_head weights | Calibration | Mamba in/out_proj | KV cache |
| --- | --- | --- | --- | --- |
| max | W4A16 dynamic NVFP4 | max | W4A16 NVFP4 | FP8 |
| mamba_fp8_max | W4A16 dynamic NVFP4 | max | FP8 (W+A) | FP8 |
| MSE | W4A16 static NVFP4 | MSE | W4A16 NVFP4 | FP8 |
| **four_over_six** (chosen in the docs) | W4A16 static NVFP4 | 4/6 | W4A16 NVFP4 | FP8 |
| four_over_six + NVFP4 KV | W4A16 static NVFP4 | 4/6 | W4A16 NVFP4 | NVFP4 (K and V only; BMMs BF16, Q unquantized) |

Shared: lm_head W4A16 ("faithful lm_head"), attention projections BF16, 1,000 calibration samples.

## PTQ and QAD settings (docs)

| Setting | Value |
| --- | --- |
| Calibration data | `cnn_nemotron_v2_mix` (cnn_dailymail + Nemotron-Post-Training v2), 1,000 samples, seq length 32,768 |
| PTQ parallelism | TP=1, EP=1, PP=1, DP=4 on one node (the docs also say "runs on a single B300") |
| QAD type | frozen-scale (MSE-family PTQ scales and amax fixed; only weights train) |
| QAD loss | KL divergence on logits vs frozen BF16 teacher |
| QAD hyperparameters | seq 32,768, MBS 1, GBS 64, LR 2e-5 -> min 5e-6, 30 warm-up iters, 200 iters (~419M tokens) |
| QAD parallelism | TP=1, PP=1, EP=16, CP=4; 8 nodes x 4 GPUs (DP=8) |
| QAD data | public blend of 13 tokenized Nemotron datasets (top weights: Pretraining-SFT-v1 General 20, SFT-Math-v3 17, SFT-Competitive-Programming-v2 python 15) |
| Size | 22 GB NVFP4 vs 66 GB BF16 (no throughput claim) |

# Contradictions and gaps

| Topic | Report | NVFP4 model card | Repo docs recipe |
| --- | --- | --- | --- |
| Mamba in/out projections | FP8 W+A (§4.8) | FP8 per-tensor dynamic | W4A16 NVFP4 in the chosen four_over_six recipe (flagged in the docs) |
| Calibration | "same PTQ/QAD procedure" as Super (Super = MSE weight scales, Table 13) | four-over-six, 1000 samples, Nemotron Ultra validation subset, 32K | four-over-six, 1,000 samples of public `cnn_nemotron_v2_mix` |
| lm_head | NVFP4 W4A16 | not stated | W4A16 (consistent) |
| Attention projections | BF16 | not stated | BF16 (consistent) |
| KV cache | FP8 (eval) | FP8 | FP8 (consistent) |
| Mamba SSM cache | FP32 in eval | not stated | not stated in quantization.md; cookbook serves FP16 + stochastic rounding on H100 |
| QAD sequence length | not reported | not stated | command uses 32,768; docs say ablations favoured ~522K (the SFT length) and the command will not reproduce it |
| Size / throughput | not reported | not stated | 22 GB vs 66 GB; no throughput claim |

# Caveats

- No catalog step in `src/nemotron/steps` (e.g. `optimize/modelopt/quantize`, `optimize/modelopt/distill`) lists Lightning as a model, so `related_steps` is empty.
- Do not quote Super's 99.52% / 4.82 BPE / Table 12 tiers for Lightning.
- "W4A16 regardless of GPU SKU" means Blackwell GPUs also run Lightning NVFP4 with 16-bit activations.
