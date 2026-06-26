---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "inference"
paper_sections: ["5", "5.1", "5.2", "5.3", "6"]
title: "Nemotron 3 Ultra Inference: serving regimes, throughput vs frontier MoEs, Ultra-scale parallelism, disaggregation, looking forward"
summary: |
  Covers §5 (and inference-relevant points of §6 conclusion): how LatentMoE + hybrid Mamba-2/Attention
  + MTP play out across prefill- vs decode-heavy and small- vs large-batch serving. Documents relative
  throughput vs GLM-5.1, Kimi-K2.6, and Qwen-3.5 (Figure 15), MTP speculative decoding for hybrids
  (Figure 16), Ultra-scale parallelism (TP/EP, prefill-decode disaggregation, all-to-all backend,
  MoE chunking, weight padding), and forward-looking optimizations.
key_facts:
  - "Ultra inherits Super's inference-aware architecture: LatentMoE, hybrid Mamba-2 stack with sparse global Attention anchors, and Multi-Token Prediction (MTP) for native speculative decoding."
  - "Conclusion headline: ~5x higher inference throughput than other state-of-the-art open LLMs at on-par accuracy."
  - "Figure 15 compares max-throughput serving on GB200 NVL72 at NVFP4 with speculative decoding disabled, for decode-heavy 8K/64K and prefill-heavy 50K/2K ISL/OSL, normalized to GLM-5.1 in both settings."
  - "Decode-heavy 8K/64K relative throughput (output tokens/s/GPU): Nemotron-3-Ultra 550B-A55B-NVFP4 = 5.9; Qwen-3.5 397B-17B = 3.7; Kimi-K2.6 1T-A32B = 1.2; GLM-5.1 754B-A40B = 1.0."
  - "Prefill-heavy 50K/2K relative throughput: Qwen-3.5 = 4.6; Nemotron-3-Ultra-NVFP4 = 3.9; Kimi-K2.6 = 0.9; GLM-5.1 = 1.0."
  - "Nemotron 3 Ultra leads decode-heavy (stated as 1.6x over Qwen-3.5) but trails Qwen-3.5 on prefill-heavy."
  - "Prefill is compute-bound; per-token cost tracks FLOPs set by active parameters; Ultra pays ~3.2x FLOPs penalty vs Qwen-3.5-397B-17B (55B vs 17B active params)."
  - "In large-batch decode, routing activates essentially all experts and per-step cost is set by total weight I/O; the gap shrinks to ~1.39x (550B vs 397B total params)."
  - "Mamba-2 state-space layers have per-step decode cost constant in sequence length, letting Ultra lead decode-heavy despite trailing prefill."
  - "MTP draft length is exposed as a deployment-time knob; small batches favor high draft length (weight-read bound, cheap verification), large batches favor lower draft length or disabling MTP."
  - "Figure 16: decode throughput vs MTP draft length on single-user ISL/OSL/BS = 10K/16K/1, single GB200 node TP=4, acceptance lengths from SPEED-Bench; throughput peaks at DL=6 giving a 2.89x speedup before rolling off."
  - "Mamba SSM state rollback on draft-token rejection handled by snapshotting SSM state at every draft step; coarser-cadence snapshotting also enables prefix caching across requests for Mamba."
  - "Parallelism axes: tensor parallelism (TP, shards weight matrices), expert parallelism (EP, distributes full routed experts; Attention/Mamba data-parallel), and TP+EP combinations."
  - "Small batches are weight-read bandwidth bound -> wide TP wins; large batches are activation-communication bound -> wide EP wins; wide EP for high-throughput, wide TP for low-latency serving."
  - "Wide-EP serving requires load balancing across DP ranks (synchronous all-to-all stalls on under-loaded rank); requests routed to least-loaded worker; hot experts replicated across EP ranks (EPLB)."
  - "GB200 NVL72's single NVLink domain across all 72 GPUs lets wide EP span the full system without cross-domain interconnect cost in the all-to-all path."
  - "Prefill-decode disaggregation adopted for Ultra; for hybrid Mamba-Attention both KV cache and Mamba SSM state must transfer correctly; landed semantic KV-event metadata path and NIXL side-channel host-resolution fix for multi-node Ray; works out-of-the-box in vLLM."
  - "Disaggregation measured up to ~10% throughput improvement on Ultra on prefill-heavy workloads."
  - "Routed-expert all-to-all accounts for ~15% to 20% of total runtime on prefill-heavy 50K/2K workloads."
  - "Adopted FlashInfer NVLinkOneSided all-to-all backend (vs default vLLM AllGather+ReduceScatter) for ~5% throughput improvement on GB200 NVL72."
  - "MoE-side chunking landed upstream in vLLM to let prefill chunk size grow without hitting routed-expert kernel resource limits under wide EP."
  - "Weight padding at load time fixes GEMM dimensions that violate kernel alignment (MoE NVFP4 kernels; Marlin NVFP4 linear/MoE kernels on Hopper)."
  - "Looking forward: overlap routed-expert all-to-all with computation (e.g. DWDP) to hide inter-rank cost; continue co-designing future models with inference efficiency as a first-class constraint."
related_steps: []
currency: "frozen"
---

# Scope

Answers:
- How does Nemotron 3 Ultra perform across prefill- vs decode-heavy and small- vs large-batch serving?
- What are the throughput multipliers vs GLM-5.1, Kimi-K2.6, and Qwen-3.5?
- How does MTP speculative decoding work for a hybrid Mamba model, and what speedup?
- What parallelism choices (TP/EP) apply at Ultra scale, and why?
- How are prefill-decode disaggregation, all-to-all, MoE chunking, and weight padding handled?
- What inference headroom remains (looking forward / conclusion)?

# Relative throughput (Figure 15, normalized to GLM-5.1, GB200 NVL72, NVFP4, spec decoding off)

Output tokens/s/GPU. Decode-heavy = 8K input / 64K output; prefill-heavy = 50K input / 2K output.

| Model | Decode-heavy 8K/64K | Prefill-heavy 50K/2K |
|---|---|---|
| Nemotron-3-Ultra 550B-A55B-NVFP4 | 5.9 | 3.9 |
| Qwen-3.5 397B-17B | 3.7 | 4.6 |
| Kimi-K2.6 1T-A32B | 1.2 | 0.9 |
| GLM-5.1 754B-A40B | 1.0 | 1.0 |

Ultra leads decode-heavy (1.6x over Qwen-3.5 per text) and trails Qwen-3.5 on prefill-heavy. Decode-heavy methodology matches Figure 1.

# Prefill vs decode bottleneck analysis

| Regime | Bound by | Cost driver | Ultra vs Qwen-3.5 |
|---|---|---|---|
| Prefill (compute-bound) | FLOPs | active parameters (55B vs 17B) | ~3.2x FLOPs penalty; MoE GEMMs dominate; Ultra trails |
| Large-batch decode | total weight I/O | total parameters (550B vs 397B) | gap shrinks to ~1.39x; Mamba-2 constant per-step cost lets Ultra lead |

# MTP speculative decoding (Figure 16)

- Operating point: NVFP4 checkpoint, single-user ISL/OSL/BS = 10K/16K/1, single GB200 node, TP=4.
- Acceptance lengths measured on SPEED-Bench.
- Throughput peaks at draft length DL=6 -> 2.89x speedup over no-MTP baseline, then rolls off as verification overhead outweighs marginal acceptance gain.
- Draft length is a deployment-time knob: high DL wins at small batches (latency); low DL or no-MTP wins at large batches.
- Hybrid rollback: snapshot Mamba SSM state at every draft step (single fixed-size entry overwritten every token); coarser-cadence snapshotting also yields prefix caching across requests for Mamba.

# Ultra-scale parallelism and serving

| Topic | Key point |
|---|---|
| TP vs EP | Small batch (weight-read bound) -> wide TP; large batch (activation-comm bound) -> wide EP; TP+EP (with DP for Attention/Mamba) can beat either in some low-latency settings |
| Load balancing | Synchronous routed-expert all-to-all means under-loaded rank stalls the group; route to least-loaded worker; replicate hot experts across EP ranks (EPLB) |
| GB200 NVL72 fit | Single NVLink domain across 72 GPUs; wide EP spans full system without cross-domain interconnect in all-to-all |
| Prefill-decode disaggregation | Adopted; transfers both KV cache and Mamba SSM state; landed semantic KV-event metadata + NIXL host-resolution fix; works out-of-the-box in vLLM; up to ~10% throughput gain on prefill-heavy |
| All-to-all backend | Routed-expert all-to-all = ~15–20% of runtime on prefill-heavy 50K/2K; adopted FlashInfer NVLinkOneSided for ~5% throughput gain over default vLLM AllGather+ReduceScatter |
| MoE chunking | MoE-side chunking landed in vLLM so prefill chunk size grows without hitting routed-expert kernel resource limits under wide EP |
| Weight padding | Pad weight matrices at load time when (TP, quant, hardware) tuples violate kernel alignment (MoE NVFP4 kernels; Marlin NVFP4 linear/MoE on Hopper) |

# Looking forward (§5.3) and conclusion (§6)

- Overlap routed-expert all-to-all with computation (e.g. DWDP) to mask inter-rank cost on EP deployments; all-to-all kernel still has optimization room.
- Continue co-designing future models with inference efficiency as a first-class constraint.
- Conclusion: Ultra = 550B total / 55B active, MoE Hybrid Mamba-Attention + LatentMoE + MTP; pre-trained on 20 trillion text tokens; post-trained with SFT, RL, MOPD; ~5x higher inference throughput than other SOTA open LLMs at on-par accuracy; pre-trained/post-trained/quantized checkpoints + training data open-sourced on HuggingFace.

# Caveats

- Figure 15 numbers are relative throughput normalized to GLM-5.1 = 1.0, not absolute tokens/s; competitor labels are GLM-5.1 754B-A40B, Kimi-K2.6 1T-A32B, Qwen-3.5 397B-17B.
- The "1.6x over Qwen-3.5" decode-heavy figure is stated in the Figure 15 caption; the bar values give 5.9 vs 3.7 (≈1.59x).
- 3.2x (prefill FLOPs), 1.39x (decode weight I/O), 10% (disaggregation), 5% (all-to-all), 15–20% (all-to-all runtime share) all carry "roughly"/"up to" qualifiers in the source.
- 2.89x MTP speedup is specific to the DL=6 peak at the single-user 10K/16K/1, TP=4, single-GB200-node operating point with SPEED-Bench acceptance lengths; not a general claim.
- Quantization recipe and SSM-cache treatment are in §4 (see quantization.md), not here.
