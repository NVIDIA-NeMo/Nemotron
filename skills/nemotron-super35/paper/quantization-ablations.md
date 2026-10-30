---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "quantization-ablations"
paper_sections: ["C", "C.1", "C.2", "C.3", "C.4", "C.5", "C.6"]
title: "Nemotron 3.5 Super VL Quantization Ablations: runaway loops, FP4 stop-vs-continue margin, attention-score dilution, Mamba out-proj, unembedding"
summary: |
  Covers Appendix C of the Nemotron 3.5 Super and Lightning tech report: the mechanistic ablation study behind
  the §4.6 precision guidance for Super 3.5 VL. C.1 defines runaway loops (termination failure) and the
  truncation/output-length early-warning signal (Table 27). C.2 attributes FP4-induced verbosity to expert
  weights shifting a weak stop-vs-continue margin, and shows MOPD (RL post-training) makes the model robust
  to it (Tables 28-29). C.3 explains long-context disambiguation damage (MRCR-128k) via attention-score
  dilution under FP8 KV and recommends per-block FP8 on q/k/v (Tables 19, 30-33). C.4 shows FP4 weights on
  the Mamba output projection inflate length (Table 34). C.5 shows NVFP4 lm_head flips stop/continue
  decisions 3.14x more than FP8 (Table 35). C.6 states the structural recommendation and 8 limitations.
key_facts:
  - "Runaway loop = termination failure from reasoning collapse ('failed negative proof': model cannot certify a negative result and reopens settled conclusions); distinct from knowledge collapse (hallucination) (C.1)."
  - "Early warning: truncation rate (share of rollouts with >= 1 model call hitting an output limit) plus output length per rollout (C.1, Table 27)."
  - "Tau3 Banking BF16 pre-MOPD: 20.9% truncation, 35,173 output tokens/rollout, 19.7% runaway (153/776 rollouts); after MOPD: 0.3% truncation, 13,669 tokens/rollout, one runaway (C.1)."
  - "Table 27 (BF16, pre-MOPD -> MOPD): RL post-training removed the non-terminating tail, not baseline verbosity; e.g. Baby Vision 42.3k -> 11.2k OSL (0.27x), truncation 39.0% -> 1.9%; AA-LCR 4.7k -> 4.7k (1.00x)."
  - "Pre-MOPD Tau3 Banking: experts NVFP4 -> FP8 drops runaway 28.2% -> 17.9% (below BF16's 19.7%), verbosity -21.5% at 2.7 sigma (C.2)."
  - "Expert activations -> BF16: runaway ~0 change (verbosity falls on 5/5 focused benchmarks); lm_head -> BF16: no termination recovery; so expert weights are the culprit (C.2)."
  - "W4A4 expert GEMM output relative L2 error 7.5-22.9% vs BF16, >10% in 39 of 40 MoE layers, 15% on average (C.2)."
  - "FP4 expert weights relative error 0.087 vs FP8 expert quantization 0.027, across all layers (C.2)."
  - "Routed/shared crossover at layer 59: shared experts dominate nearly every layer L59-L83; weak late-band routed contribution (alpha) is a signature of the looping state (C.2, Table 28)."
  - "Stop decision: P(stop) where </think> is emitted ~0.99 vs <= 1e-9 elsewhere; FP4 experts never cost a termination (23/23 pre-MOPD, 24/24 MOPD); FP4 shifts the decision away from stopping in 17/23 pre-MOPD rollouts (p=0.035) but toward stopping in 20/24 MOPD rollouts (p=0.002) (Table 29)."
  - "Runaway rollouts: peak P(stop) median 0.043 pre-MOPD vs 0.729 MOPD; </think> at peak 2/10 vs 6/10 (Table 29)."
  - "Knowledge abstention unaffected: AA-Omniscience non-hallucination spans 0.991x-1.022x across five ablations (67.8%-69.9% vs BF16 68.4%) (C.2)."
  - "Long-context taxonomy: distinction (content), disambiguation (position), aggregation; quantization costs 2.65-3.6 pp on disambiguation (MRCR-128k) and nothing measurable elsewhere (C.3, Table 19)."
  - "Attention-score dilution under FP8 KV (e4m3 P): 4,730-7,981 keys in window compete for 35-80 distinct e4m3 levels, 60-120 keys per level, mass preserved 0.994-1.001 (Table 30)."
  - "MRCR-128k: q/k/v per-tensor FP8 retains 95.3%/95.5% below 32k but 91.4%/92.1% above (3.6 pp break); q/k/v MXFP8 recovers 95.3%/94.3% above 32k, similar to BF16 (Table 31)."
  - "MRCR-128k BF16 accuracy falls 0.800 -> 0.378 as candidates rise 2 -> 8; damage worst at 4 candidates (91.7% recovery) (Table 32)."
  - "AA-LCR attention-projection ablation: all-NVFP4 0.5713 (-2.25 pp vs BF16 baseline); all-FP8 0.5962 (+0.25); q/k/v FP8 + o BF16 0.6050 (+1.12) (Table 33)."
  - "Recommendation: FP8 on q/k/v at per-block scale granularity; output projection tolerant of per-tensor FP8 (C.3)."
  - "Mamba out-proj input: SiLU gate near-zero-never-zero -> ~16 survivors per FP4 block, 27.6% of elements below amax_block/24 flush limit; vs expert fc2 (squared-ReLU) 90-99% exact zeros, 1.05-3.32 survivors/block; rules out W4A4 for Mamba out-proj (C.4)."
  - "Mamba out-proj W4A16 vs W8A8 (post-MOPD): longer outputs on all 6 benchmarks, OSWorld 1.27x, SciCode 1.16x; task accuracy unchanged (Table 34)."
  - "lm_head (Table 35, fake quant, identical hidden input): FP8 vs NVFP4 weight error 0.0265 vs 0.0952 (3.6x); top-1 agreement 98.6% vs 95.8%; KL mean 0.0013 vs 0.0121 bits; stop/continue flip rate 0.781% vs 2.450% (3.14x)."
  - "lm_head mechanism: </think> row cosine 0.9961 after quantization, but decision is the difference of two near-orthogonal rows (cos ~0.22); per-element grid (E2M1 50% step vs E4M3 12.5%) is the remaining hypothesis; two-pass FP4 residual emulation recovers KL 0.00117 vs FP8 0.00134 and top-1 0.9951 (C.5)."
  - "C.6: stop-vs-continue geometry invariant within 1% across three pre-/post-RL checkpoints (cosine 0.997, margin 0.996, noise-to-margin 1.001); routed contribution invariant within 0.007; RL changed where the hidden state points, so precision recommendations are structural but margin protection is checkpoint-dependent."
related_steps: []
currency: "frozen"
---

# Scope

Answers:
- What is a runaway loop, and how is it detected cheaply?
- Why does FP4 quantization inflate verbosity without hurting accuracy, and which component is responsible?
- Why did the released (post-MOPD) checkpoint tolerate FP4 experts when earlier checkpoints did not?
- Why does quantization hurt MRCR-128k but not other long-context benchmarks, and what q/k/v precision fixes it?
- Why not W4A4 (or FP4 weights) on the Mamba output projection?
- Why keep lm_head at FP8 or above?
- What are the stated limitations of these ablations?

Context: these are the mechanisms behind §4.6 of the report (see `quantization.md`). Unless noted, measurements are on Nemotron 3.5 Super VL. "pre-MOPD" = checkpoint before multi-teacher on-policy distillation (RL post-training); "MOPD" = after. Every configuration in this study, including the BF16 reference, serves an FP8 KV cache (C.6 limitation 4).

# C.1 Runaway loop: termination failure in truncated rollouts

- Long reasoning in agentic work is mostly probing (running code, calling tools, reading results) and is not itself a defect.
- Abstention fails two ways: **knowledge collapse** -> hallucination; **reasoning collapse** -> termination failure, seen as semantic runaway looping.
- Runaway loop signature: "failed negative proof" — the model cannot certify a negative result from exhaustive investigation, reopens a settled conclusion, and cycles. A non-terminating agent emits *more* calls of ordinary length (per-call statistics steady), not longer calls; eventually the last reasoning explodes in length or truncation rises.
- Identified by reading reasoning passages post-mortem while tracking failure to terminate as the model hits the context wall.
- Early-warning pair: **truncation** (fraction of rollouts with >= 1 model call hitting an output limit) + **output length per rollout**.
- Tau3 Banking: pre-MOPD 20.9% truncation, 35,173 output tokens/rollout, **19.7% runaway (153/776)**; after MOPD 0.3% truncation, 13,669 tokens/rollout, one runaway.

## Table 27 — output length per rollout and truncation, pre-MOPD -> MOPD, both BF16

OSL = total over a rollout's policy-model calls, excluding the user simulator. "Post-training RL removed the non-terminating tail, not baseline verbosity."

| Benchmark | Modality | Rollouts | ISL (k) | OSL (k)/rollout | Trunc. % |
|---|---|---|---|---|---|
| Baby Vision | vision | 388 x 3 | 0.5 | 42.3 -> 11.2 (0.27x) | 39.0 -> 1.9 |
| CharXiv (RQ) | vision | 1,000 x 3 | 3.1 | 11.8 -> 3.3 (0.28x) | 9.2 -> 0.6 |
| MMLongBench-Doc | vision | 1,091 x 3 | 75.0 | 10.8 -> 3.0 (0.28x) | 9.9 -> 0.4 |
| OCR Reasoning | vision | 1,069 x 3 | 1.5-1.6 (a) | 16.8 -> 5.2 (0.31x) | 13.7 -> 0.6 |
| OSWorld | vision/agentic | 361 x 3 | — | 13.6 -> 5.1 (0.38x) | 1.2 -> 0.0 |
| Tau3 Banking | text/agentic | 97 x 8 | — | 35.2 -> 13.7 (0.39x) | 20.9 -> 0.3 |
| AA-Omniscience | text | 600 x 10 | 0.1 | 10.0 -> 5.6 (0.56x) | 1.9 -> 0.3 |
| SciCode | text | 65 x 8 | 3.0-3.1 (a) | 7.2 -> 6.5 (0.90x) | 0.0 -> 0.0 |
| AA-LCR | text | 100 x 16 | 108.0 | 4.7 -> 4.7 (1.00x) | 0.0 -> 0.2 |
| Tau2-AA | text/agentic | 278 x 8 | — | 5.0 -> 5.1 (1.03x) | 0.0 -> 0.0 |

(a) Evaluation-harness update caused the ISL difference.

# C.2 FP4 weight quantization swings the stop-versus-continue margin

Quantized models inherit runaway behavior from the base model, and quantization can worsen it. Ablations of non-attention components on **pre-MOPD** models:

| Intervention | Result | Conclusion (as stated) |
|---|---|---|
| Routed/shared experts NVFP4 -> FP8 | Tau3 Banking runaway 28.2% -> 17.9% (below BF16 reference's 19.7%); verbosity -21.5% at 2.7 sigma | Expert weights are not innocent |
| Expert activations -> BF16 | Runaway moves ≈ 0; verbosity falls on 5/5 focused benchmarks (mmlongbench-doc, scicode, tau3-banking, tau2-AA, AA-LCR) | Activation path is not the carrier |
| lm_head -> BF16 | No termination recovery | Final token generation is not the source |
| W4A4 expert GEMM output error | Relative L2 error 7.5-22.9% vs BF16; >10% in 39 of 40 MoE layers; 15% average | Perturbation is not negligible |
| Mamba out-proj -> W4A16 | Tau3 Banking verbosity +18.5% at 2.4 sigma; runaway rate and knowledge abstention unaffected | Separate verbosity effect (C.4) |

- Hypothesis: the culprit is **quantization of expert weights**. FP8 expert weights mitigate runaway on pre-MOPD models; on post-MOPD models FP4 expert weights recover verbosity **without runaway regression**.
- Mamba result carries over post-MOPD: vs a recipe with out-proj at W8A8, W4A16 is more verbose on all 6 benchmarks (Table 34).
- FP4 expert perturbation does not destroy information: AA-Omniscience non-hallucination spans 0.991x-1.022x across the five ablations (67.8% and 69.9% vs BF16 68.4%).

## Where in the stack (Table 28)

alpha = ||routed|| / ||layer||, beta = ||shared|| / ||layer|| per MoE layer output. Crossover at **layer 59**: routed contribution roughly halves, shared rises; shared dominates nearly every layer L59-L83. All MoE layers before L59 agree within 0.031 across pre-MOPD/MOPD and BF16/FP4.

alpha/beta averaged over 12 MoE layers in L59-L83, Tau3 Banking prompts split by whether each pre-MOPD rollout looped:

| pre-MOPD runaway? BF16 | FP4 | pre-MOPD BF16 | pre-MOPD FP4 | MOPD BF16 | MOPD FP4 |
|---|---|---|---|---|---|
| No | No | 0.100/0.137 | 0.097/0.146 | 0.100/0.142 | 0.097/0.140 |
| No | Yes | 0.091/0.144 | 0.061/0.162 | 0.092/0.149 | 0.092/0.147 |
| Yes | No | 0.055/0.161 | 0.093/0.139 | 0.081/0.164 | 0.081/0.161 |
| Yes | Yes | 0.059/0.151 | 0.063/0.155 | 0.077/0.160 | 0.078/0.155 |

MOPD rollouts: 1 runaway in BF16, 3 in FP4.

- Weaker late-band routed contribution accompanies looping in both BF16 and FP4 — "the weak routed contribution is a signature of the looping state".
- FP4 weight error is uniform: relative error 0.087 (FP4 experts) vs 0.027 (FP8 experts), across all layers.
- Under low-amplitude signals FP4 flips are **margin-confined**: they occur where continuations are interchangeable, redirecting output without changing knowledge. A small shift in stop probability at ordinary positions attenuates termination over a long generation — hence flat accuracy but moving verbosity.
- MOPD fortifies latent support for stopping: on the same prompts alpha recovers to 0.081 / 0.077 with beta elevated at ~0.16; FP4 leaves them at 0.081 / 0.078.

## Table 29 — Tau3 Banking P(stop) at lm_head (lm_head kept BF16)

First three rows on rollouts where </think> was emitted; last two on rollouts that looped pre-MOPD. Binomial two-sided direction test.

| Quantity | pre-MOPD | MOPD |
|---|---|---|
| P(stop) at the stop vs elsewhere (median) | 0.994 vs 2.0e-9 | 0.996 vs 6.5e-11 |
| Terminations under FP4 experts | 23/23 rollouts | 24/24 rollouts |
| FP4 experts shift the decision | 17/23 away (p=0.035) | 20/24 toward (p=0.002) |
| Peak P(stop), runaway rollouts (median) | 0.043 | 0.729 |
| </think> at the peak | 2/10 (a) | 6/10 |

(a) lm_head put </think> on top (0.384 and 0.871) but sampling did not take it and the loop continued.

GEMM output error cannot capture this difference because it measures magnitude only, not distribution and bias.

# C.3 Long-context retrieval damage on attention projections

Three retrieval decision types: **distinction** (target identifiable by content), **disambiguation** (near-identical candidates, must index by ordinal position), **aggregation** (combine evidence across many candidates). Quantization costs **2.65-3.6 pp on disambiguation** and nothing measurable elsewhere (Table 19: MRCR-128k -2.81 / -2.65 / -3.63 pp for q/k/v BF16 / MXFP8 / FP8). The same effect is observed on Lightning 3.5; this section studies Super 3.5 VL.

## Attention-score dilution (Table 30)

Tau3 Banking under FP8 KV cache, per head, 12 contexts of 10,532-63,641 prefix tokens; softmax output P cast to e4m3 for the PV GEMM (logits are not).

| Quantity | Measured | Reading |
|---|---|---|
| Keys in representable window | 4,730-7,981 | population competing |
| Distinct e4m3 P levels | 35-80 | levels available |
| Keys sharing a level | 60-120 | the dilution |
| Ties at the top e4m3 level | 1-3 | peak stays separated |
| Attn. score mass preserved | 0.994-1.001 | mass intact, resolution lost |
| Spearman rho(context len, keys/level) | +0.73 to +0.94 | more keys per level when longer |

The peak score stays cleanly separated, so distinction and aggregation (e.g. GraphWalks up to 361k tokens of observed context) work with FP8 or NVFP4 KV; when candidates differ only by position, precision becomes the limit (MRCR-128k).

## MRCR-128k stratification (Tables 31, 32)

- Table 19 contrasts are individually small: q/k/v BF16-over-FP8 = 0.83 pp (SE 0.71 pp); MXFP8 gains +0.99 pp over per-tensor FP8, landing 0.16 pp beyond BF16. Neither clears 2 sigma (1.4 pp).
- Stratified by context, damage is present at every length but breaks at **32k+**.

Table 31 — recovery % vs BF16 reference, 1,470 tasks x 3 reps; o projection is FP8 in every recovery column; context bins use n_tokens (undercounts served prompt by ~3%).

| Context | n | BF16 acc | q/k/v BF16 | q/k/v MXFP8 | q/k/v FP8 |
|---|---|---|---|---|---|
| 0-8k | 290 | 0.768 | 97.9 | 97.5 | 95.3 |
| 8-32k | 578 | 0.552 | 94.6 | 95.1 | 95.5 |
| 32-64k | 295 | 0.559 | 95.0 | 95.3 | 91.4 |
| 64-131k | 307 | 0.596 | 93.8 | 94.3 | 92.1 |

MXFP8 differs from per-tensor FP8 only in scale granularity (one E8M0 exponent per 32 elements vs one float per tensor), which is enough to close the >32k gap.

Table 32 — BF16 accuracy by number of near-identical candidates x served context (each context band is a 33% mix of the three candidate counts; candidate strata length-matched at 36.4-37.8k mean prompt tokens). Recovery = q/k/v/o FP8 vs BF16 reference.

| Candidates | 0-8k | 8-32k | 32-64k | 64-131k | Pooled BF16 | Recovery % |
|---|---|---|---|---|---|---|
| 2 | 0.883 | 0.749 | 0.793 | 0.823 | 0.800 | 94.9 |
| 4 | 0.806 | 0.585 | 0.522 | 0.694 | 0.637 | 91.7 |
| 8 | 0.616 | 0.320 | 0.353 | 0.290 | 0.378 | 96.0 |

- Baseline accuracy steps down 13.4-29.5 pp between 0-8k and 8-32k, then saturates; above 32k candidate count dominates — "the signature of an exhausted indexing budget rather than a strained context window".
- Damage is worst in the middle (4 candidates); smaller at 2 (wide margins) and at 8 (reference already collapsed). Like the unembedding (C.5), quantization moves outcomes where the decision was already close.
- Methodological lesson: a single benchmark number hides long-context damage; stratify along capability partitions.

## Projection precision and scale granularity (Table 33)

- **Lightning 3.5 (not Super):** KV cache FP8 -> BF16 with weights/projections fixed recovers MRCR-128k +1.43 pp (2.4 sigma); projections FP8 -> BF16 recovers +2.97 pp (2.8 sigma).
- Super 3.5 VL: KV cache is the damage carrier; high-precision q/k/v does not prevent dilution but buys back part of the margin destroyed by the cast.

Table 33 — attention {q,k,v,o} precision on AA-LCR (aggregation; ~108k tokens context average; 100 tasks x 16 reps; does not exercise disambiguation):

| Attention {q,k,v,o} precision | Score | Δ vs BF16 baseline |
|---|---|---|
| All four NVFP4 | 0.5713 | -2.25 pp |
| All four BF16 | 0.5900 | -0.38 pp |
| All four FP8 | 0.5962 | +0.25 pp |
| o FP8, q/k/v BF16 | 0.6031 | +0.94 pp |
| q/k/v FP8, o BF16 | 0.6050 | +1.12 pp |

- All-FP8 -> all-NVFP4 projections costs 2.50 pp (SE 1.11).
- Dynamic range is not the issue: across the 24 q/k/v weight tensors, per-16-element block-max spread is 2.51 binades median / 8.78 worst; activations entering them span 1.67; an E4M3 block scale encodes 17.81 binades before underflow; no block loses its scale (9.03 binades unused on the worst tensor). Per-block granularity recovers **precision headroom** a per-tensor scale had exhausted.
- Fused q/k/v FP8 with one shared scale: the key projection sets the scale on 6 of 8 attention layers, value on the other 2; fusion adds ~32% relative weight error to q and v vs 4% for k.
- **Recommendation: FP8 on q/k/v at per-block scale granularity.** If a deployment does no disambiguation-style retrieval, no other workload shows a significant difference.
- **Output projection:** weight spread 4.21 binades median / 8.15 worst across 8 attention layers, not exposed to FP8 underflow; calibrated per-tensor input scale sits at 49-76% of the largest head-output norm at every attention layer across a 68x range of magnitudes. Concatenating head outputs under one per-tensor scale shows no benchmark-level penalty at FP8: "structurally exposed and measurably tolerant in per-tensor FP8".
- Consistency with AutoQuant (§4.3): the gradient-based search leaves q/k/v BF16 on 3 of 8 attention layers (same 3) under both 4.85 and 4.90 bit-per-parameter budgets; o stays FP8 on all layers at 4.85 and reaches BF16 on 3 layers only at 4.90.

# C.4 Mamba output projection: FP4 weights inflate generation length

- Input-shape difference vs expert fc2: fc2 is fed by squared-ReLU, with 90-99% of inputs exactly zero and only 1.05-3.32 survivors per FP4 block. The Mamba out-proj receives a normalized SiLU gate — near-zero but never exactly zero — so ~16 survivors per FP4 block, and **27.6%** of elements fall below the amax_block/24 rounding limit under FP4 activation quantization (flushed). This rules out **W4A4** for the Mamba out-proj; rotation or higher activation precision can mitigate the flush.
- Weight effect isolated as W4A16 vs FP8 W8A8 (high-precision activations):

Table 34 — output length per rollout, recipes differing only in Mamba out-proj (policy-model output, excludes user simulator):

| Benchmark | FP8 (W8A8) | NVFP4 (W4A16) | Ratio |
|---|---|---|---|
| OSWorld | 8.3k | 10.6k | 1.27x |
| SciCode | 12.0k | 13.9k | 1.16x |
| Tau3 Banking | 14.3k | 15.5k | 1.08x |
| AA-Omniscience | 6.4k | 6.7k | 1.06x |
| AA-LCR | 5.7k | 5.9k | 1.02x |
| Tau2-AA | 5.2k | 5.2k | 1.01x |

- All six move the same direction; task accuracy unchanged across the two recipes. Fix preferred: parameter alignment via QAD (§4.4). The released Aggressive recipe keeps Mamba linear layers at FP8 (Table 12).

# C.5 The unembedding: FP8 weights; sensitivity is on termination

- **lm_head should stay at FP8 or above.** NVFP4 lm_head keeps high top-1 agreement with BF16; its problem is the stop-vs-continue decision -> verbosity inflation.
- Decision modeled as one logit comparison: margin = h · (v_stop - v_cont). After quantization the </think> row has cosine 0.9961 to the original, yet 2.45% of decisions flip, because the decision lives in the difference of two nearly orthogonal rows (cos ≈ 0.22), where small rotations become a large perturbation of the difference.

Table 35 — per-tensor FP8 vs NVFP4 lm_head, identical hidden-state input, fake quantization:

| Metric | FP8 (W8A16) | NVFP4 (W4A16) | Ratio |
|---|---|---|---|
| Weight error | 0.0265 | 0.0952 | 3.6x |
| Top-1 agreement with BF16 | 98.6% | 95.8% | — |
| KL mean / p99 (bits) | 0.0013 / 0.0080 | 0.0121 / 0.0797 | 9.3x / 10x |
| Stop/continue flip rate | 0.781% | 2.450% | 3.14x |

Rare-token hypothesis rejected four ways:
1. Row-amax spread 2.84 binades (2.12 at p99.9) vs FP8's ~17.8 usable — nothing for per-block scaling to reclaim.
2. Group sizes 1 to 64 give pooled error identical to four decimals per format (0.0265 FP8, 0.0952 FP4).
3. Per-row median spread by vocabulary-frequency quartile 5.222 / 5.246 / 5.252 / 5.274 binades (0.05-binade drift over 131,072 tokens); zero dead rows; lowest-norm decile error 0.0941 vs highest 0.0952 — error uniform across vocabulary.
4. NVFP4 crosses its flush line on 66.9% of blocks, flushing 6.8% of elements, with no vocabulary structure.

Remaining hypothesis: **per-element grid** — E4M3's 3 mantissa bits give a 12.5% step (1 -> 1.125); E2M1's 1 bit gives 50% (1 -> 1.5). Emulated two-pass FP4 weights (second channel correcting the residual below the amax/24 flush line) recover the gap: with a linear E2M1 residual, KL 0.00117 vs per-tensor FP8 0.00134, top-1 agreement 0.9951. Reducing lm_head quantization error remains open; the sensitivity is "confined in FP8 precision".

# C.6 Precision recommendation and limitations

- The same recipe that aggravated termination failure before RL post-training reduced it afterwards. Across three checkpoints spanning pre-RL and post-RL, stop-vs-continue geometry is invariant within 1% (cosine 0.997, margin 0.996, noise-to-margin 1.001) while termination behavior changed significantly; routed-expert contribution profile invariant within 0.007.
- Conclusion: **RL post-training did not move the lm_head or the expert contribution structure; it changed where the hidden state points.** Precision recommendations can be stated structurally, but margin protection is checkpoint-dependent.

Distilled recommendation (C.2-C.5 with §4.6):

| Component | Recommendation | Evidence |
|---|---|---|
| Routed/shared expert weights | FP4 acceptable on a post-MOPD checkpoint; FP8 mitigates runaway on weaker-margin (pre-MOPD) checkpoints | C.2, Tables 28-29 |
| Attention q/k/v | FP8 with per-block scales (e.g. MXFP8); per-tensor FP8 costs disambiguation at 32k+ | C.3, Tables 19, 31 |
| Attention o | Per-tensor FP8 tolerated | C.3 |
| Mamba out-proj | Not W4A4 (activation flush); FP4 weights inflate length -> FP8 (W8A8) | C.4, Table 34 |
| lm_head | FP8 or above | C.5, Table 35 |

Limitations (as listed):
1. Weight-vs-activation attribution for termination failure is inferred from a bound; W16A4 was not tested.
2. Expert GEMM output error and benchmark verbosity are separate measurements, consistent in sign and magnitude; the relation is coherent, not demonstrated causal.
3. Attention score-path dilution is measured on clean weights as structure, not as a perturbation response.
4. Every configuration, including the BF16 reference, serves an FP8 KV cache, so dilution is already in the baseline; projection deltas are on top of an already-diluted score.
5. MRCR-128k loss is measured against an unquantized reference with FP8 KV, but the quantized side is a full configuration (FP8 KV, FP4 routed experts, FP4 Mamba out-proj). KV as damage carrier was identified on Lightning 3.5; the Super 3.5 VL ablation does not test BF16 KV.
6. MRCR-128k is the only disambiguation instrument; 3 reps per task suffice for the decoding-noise floor but not per-stratum claims beyond the candidate-count split.
7. Unembedding measurements use a proxy input, norm_f(hidden + residual) at the last layer (true input minus that layer's MoE contribution); comparison isolates lm_head quantization exactly, but absolute rates are not the model's true outputs.
8. Contribution analysis uses two fixed token windows (one per stage); within-stage BF16-vs-FP4 contrasts are clean, cross-stage comparisons change both weights and window, so they highlight but do not pinpoint the root cause.

# Contradictions and inconsistencies in the source

1. **AutoQuant budget labels:** C.3 says 4.85 and 4.90 bits-per-parameter; §4.4 names the AutoQuant-derived QAD recipes 4.8 and 4.9 BPE. Not reconciled.
2. **MRCR-128k ablation config:** C.6 limitation 5 says the quantized side used **FP4 Mamba output projection**, whereas C.4 rules out W4A4 for that layer and the released Aggressive recipe keeps Mamba linear at FP8 (Table 12). This is an ablation configuration, not the release, but the report does not say whether "FP4" there means W4A4 or W4A16.
3. **Rounding:** C.3 quotes "2.65 to 3.6 pp" while Table 19 gives -3.63; C.3 quotes BF16-over-FP8 as 0.83 pp while Table 19 differences give 0.82 (-2.81 vs -3.63); the 0-8k -> 8-32k drop for 8 candidates computes to 29.6 pp from Table 32 vs "29.5" in text (likely unrounded source values). Minor.
4. **Table 29 caption** refers to "the opposite intervention to Table 35" — Table 35 quantizes the lm_head while Table 29 keeps it BF16 and quantizes experts; consistent once read that way, but the cross-reference is easy to misread.

# Caveats

- The tau3-banking runaway numbers in C.2 (28.2% FP4 vs 17.9% FP8 vs 19.7% BF16) are on **pre-MOPD** checkpoints, not the released model. The released (post-MOPD) checkpoint loops on 1/776 (BF16) and 3 (FP4) Tau3 Banking rollouts (Table 28 caption).
- The +18.5% Mamba out-proj verbosity in C.2 is pre-MOPD Tau3 Banking; Table 34 (1.08x on Tau3 Banking) is post-MOPD — different checkpoints, do not mix.
- Lightning 3.5 numbers in C.3 (+1.43 pp KV, +2.97 pp projections) are Lightning, not Super.
- Table 33's "BF16 baseline" (implied 0.5938 from the all-BF16 row's -0.38 pp) is not printed directly.
- t-statistics and sigma figures use SE over tasks (Table 19) or the study's stated variance; many contrasts (e.g. Table 19 non-MRCR rows) are within noise by the authors' own criterion.
