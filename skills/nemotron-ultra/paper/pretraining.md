---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "pretraining"
paper_sections: ["2.2", "2.4", "2.5", "2.7"]
title: "Nemotron 3 Ultra Pretraining: NVFP4 Recipe, Hyperparameters, Long-Context Extension, Stability"
summary: |
  Covers Nemotron 3 Ultra's NVFP4 pretraining recipe (E2M1, with select layers kept in higher precision),
  the WSD hyperparameter schedule over a 20T-token horizon, the long-context extension phase to 1M tokens,
  and the two training-divergence instabilities observed during pretraining (output-layer gradient
  precision at ~8T tokens, and an undetermined divergence at ~16T tokens) along with the diagnostic metrics
  (MaxVio, residual norms) used to study them.
key_facts:
  - "Same NVFP4 pretraining recipe as Nemotron 3 Super; uses Transformer Engine open-source cuBLAS NVFP4 GEMM kernels for fprop, dgrad, wgrad."
  - "NVFP4 layers use the E2M1 datatype with two-dimensional block quantization on weights, Random Hadamard Transforms on inputs to wgrad, and stochastic rounding on gradients."
  - "Kept in higher precision: final 15% of the network (16 layers), Mamba output projections, latent projections, QKV and attention projections, MTP layers, and embedding layers."
  - "Claimed largest-scale demonstration of stable and accurate NVFP4 training to date."
  - "BF16 ablations branched at 5T, 10T, 16T checkpoints, continued 60B tokens; average relative train loss gap < 0.4%."
  - "Relative train loss gap within first 5B tokens of BF16 training: 0.27% (5T), 0.28% (10T), 0.25% (16T)."
  - "After 60B tokens BF16: gap rises to 0.33% (5T) and 0.34% (10T), falls to 0.30% (16T)."
  - "Warmup-Stable-Decay (WSD) LR schedule over a 20T-token horizon; warmup 200B tokens to peak LR 2.5e-4."
  - "Final 5T tokens: minus-sqrt decay to minimum LR 2.5e-6."
  - "Offline checkpoint merging used; sliding merge window 500B tokens, checkpoint interval 25B tokens; final selection over merges with windows from 125B to 1T tokens (sequential/random/reversed)."
  - "MTP loss scaling factor 0.1 (0.05 per MTP block)."
  - "Long-context phase: constant LR 2.5e-6; 32-way context parallel, 8-way tensor parallel, 128-way expert parallel, 2-way pipeline parallel on GB200."
  - "LC blend: 46% long-context data, 54% Phase 2 data; no RULER-style data; trained 92% iterations at 1M (1,048,576) context, 8% at 4K (4,096)."
  - "LC each iteration trained 25,165,824 tokens; only math/code SFT-style data in 4K iterations; LC-Phase trained for 33B tokens total."
  - "Divergence 1 at ~8T tokens: caused by output-layer local gradient accumulation precision dropped FP32->BF16; fixed by rollback + full FP32 gradient reduction."
  - "Divergence 2 at ~16T tokens: undetermined; mitigated by early LR annealing after rolling back to ~15T checkpoint; total horizon cut to 20T tokens."
  - "MaxVio_max = E/k = 23.27 for Ultra and Super; 21.33 for Nano."
  - "Ultra MoE routing: initial median MaxVio 1.2, max 4.8 (first MoE layer); median stayed ~1.2 but max rose to ~12 by 12T tokens."
  - "Residual norms differ by 4 orders of magnitude across depth for Ultra (3 for Super); Ultra early-layer norms started rising ~7.5T tokens, large spikes ~11T tokens."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- What is the NVFP4 pretraining recipe and which layers stay higher-precision?
- What hyperparameters (LR schedule, peak/min LR, warmup, merging, MTP loss) were used?
- How was context extended to 1M, with what parallelism and data blend?
- What training instabilities occurred and how were they handled?

# §2.2 NVFP4 Pretraining

| Aspect | Setting |
|---|---|
| Recipe source | Same as Nemotron 3 Super (NVIDIA, 2026) |
| Kernels | Transformer Engine open-source cuBLAS NVFP4 GEMM (fprop, dgrad, wgrad) |
| NVFP4 datatype | E2M1 |
| Weight quantization | Two-dimensional block quantization |
| wgrad inputs | Random Hadamard Transforms |
| Gradients | Stochastic rounding |

Kept in higher precision: final 15% of the network (16 layers), Mamba output projections, latent projections, QKV and attention projections, MTP layers, embedding layers.

BF16 health-check ablations (branch -> switch all tensors to BF16 -> continue 60B tokens):

| Branch checkpoint | Rel. train-loss gap, first 5B BF16 tokens | Rel. gap after 60B BF16 tokens |
|---|---|---|
| 5T | 0.27% | 0.33% |
| 10T | 0.28% | 0.34% |
| 16T | 0.25% | 0.30% |

Average relative train loss gap across the three ablations: < 0.4%. Switching all tensors to BF16 did NOT resolve the §2.7 training divergence.

# §2.4 Hyperparameters

| Hyperparameter | Value |
|---|---|
| LR schedule | Warmup-Stable-Decay (WSD) |
| Total horizon | 20 trillion tokens |
| Warmup | 200 billion tokens to peak LR |
| Peak LR | 2.5 x 10^-4 |
| Decay phase | Final 5 trillion tokens, minus-sqrt decay |
| Minimum LR | 2.5 x 10^-6 |
| Checkpoint merging | Offline (Tian et al., 2025), sliding window 500B tokens |
| Checkpoint interval | 25B tokens (weighted to emulate LR decay) |
| Final selection merge windows | 125B to 1T tokens; sequential, random, reversed orderings |
| Selected for long-context | 500B-token merge window checkpoint (balanced knowledge/math/code) |
| MTP loss scaling factor | 0.1 |

All other hyperparameters same as Nemotron 3 Super.

# §2.5 Long-Context Extension (LC-Phase)

| Aspect | Setting |
|---|---|
| Method | Continuous pretraining (CPT) at end of pretraining |
| Learning rate | Constant 2.5 x 10^-6 |
| Context parallelism | 32-way |
| Tensor parallelism | 8-way |
| Expert parallelism | 128-way |
| Pipeline parallelism | 2-way |
| Hardware | GB200 GPUs |
| Data blend | 46% long-context data, 54% Phase 2 data |
| RULER-style data | None used in blend |
| 1M-context iterations | 92% (1,048,576 tokens context) |
| 4K-context iterations | 8% (4,096 tokens context) |
| Tokens per iteration | 25,165,824 |
| 4K-iteration data | Only math and code SFT-style data |
| LC-Phase total tokens | 33B tokens |

Each iteration uses either 1M or 4K length; sequence lengths were not mixed within an iteration. Added long-context SFT-style data on top of the long-context document QA data used in Super & Nano.

# §2.7 Model Stability — Two Divergences
Both divergences showed simultaneous increases in training cross-entropy loss and wgrad L2 norm.

**Divergence 1 (~8T tokens) — Output-layer gradient precision.** Caused by reducing local gradient accumulation precision for the output layer from FP32 to BF16 (a throughput optimization to do data-parallel gradient reductions in BF16 over the wire). With 2 MTP blocks at MTP loss scaling 0.1 (0.05 each), the MTP blocks' wgrad contribution to the shared output layer is essentially lost under BF16's 7-bit mantissa. MTP-2 loss diverged first with frequent large spikes (Figure 6). Fix: roll back to an earlier checkpoint and revert to full FP32 gradient reduction.

**Divergence 2 (~16T tokens) — Undetermined.** Ablations showed that starting LR annealing (both a 5T and a 10T decay) immediately after rolling back to the 15T-token checkpoint mitigates the divergence. Practical decision: cut total pretraining horizon to 20T tokens. No single root cause identified; two diagnostic metrics were useful:

| Diagnostic | Findings |
|---|---|
| MaxVio (expert load imbalance) | MaxVio_max = E/k = 23.27 (Ultra and Super), 21.33 (Nano). Computed over 20 iterations / 500M tokens per checkpoint. Train MaxVio always < validation. Nano ~1.3 train / ~5 val; Super ~2 / ~6. Ultra: started balanced (median 1.2, max 4.8 first MoE layer); median stayed ~1.2 but max rose to ~12 by 12T tokens (first layer). Correlated with, not proven causal of, instability. |
| Residual stream activation norms | Differ by 3 orders of magnitude across depth for Super, 4 for Ultra. Ultra early-layer residual norms started rising ~7.5T tokens with large spikes ~11T tokens, unlike the rise-then-stabilize pattern in Nano/Super early layers. |

# Caveats
- The < 0.4% loss gap and per-checkpoint gaps are from BF16 health-check ablations (proxy for high-precision training), not the production NVFP4 run's absolute loss.
- "16 layers" = the final 15% of the network kept higher precision; do not conflate with total layer count (108, see architecture.md).
- Divergence 2 root cause is explicitly undetermined; do not attribute it to a specific mechanism.
- MaxVio correlation with instability is stated as non-causal by the authors.
- LR values are 2.5e-4 (peak) and 2.5e-6 (min / LC constant); do not transpose.
