---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "inference"
paper_sections: ["Abstract", "1", "5", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "6"]
title: "Nemotron 3.5 Super Inference: 20-turn proxy workload, NVFP4 vs BF16, cross-platform throughput vs Qwen, prefix reuse with MTP, ReplaySSM, Mamba checkpoints, disaggregated serving on Vera Rubin"
summary: |
  Covers §5 (Inference) for Nemotron 3.5 Super (120B-A12B) plus the inference headline claims in the
  Abstract, §1, and §6. Defines the 20-turn agentic proxy workload and the "usable" criterion
  (>= 50 output tokens/s/user). Documents NVFP4 vs BF16 serving on H100 and GB200 (Figure 10),
  max throughput vs Qwen3.5-122B-A10B and Qwen3.8-Flash-Next (Figure 11, mirrored in the Figure 1
  right panel), the report's performance-model explanation, prefix reuse combined with MTP speculative
  decoding for a hybrid Mamba model (§5.3), per-quantization GEMM backend selection and ReplaySSM
  checkpointing replay (§5.4, Table 23), internal Mamba prefill checkpoints (§5.5), and aggregated vs
  disaggregated serving with Dynamo + vLLM on 11 VR200 GPUs (§5.6, Table 24). Records the unreconciled
  gap between the Abstract/§6 "2.88x vs Qwen3.8 Flash-Next" claim and Figure 11 (1.90x).
key_facts:
  - "Proxy workload (§5 intro): 20 turns, 2,048 input tokens and 1,024 output tokens per turn; shared prompt and prior conversation are eligible for prefix reuse on later turns. §5.1 adds a 50k prefix; Table 23 states a 51,200-token shared prefix; §4 default is 51200/2048/1024 (pre-cached ISL / new ISL / OSL)."
  - "Usable serving point (§5 intro): reported per-user generation rate >= 50 tokens/s; among usable points, maximize aggregate output throughput. §5.6 (Vera Rubin) adds median TTFT < 5 s and uses median output speed."
  - "Figure 10 (vLLM, max throughput at >= 50 TPS/user, output tokens/s/GPU): H100 80GB BF16 522 vs NVFP4 (W4A16) 970 = 1.86x; GB200 BF16 1,778 vs NVFP4 (W4A4) 4,324 = 2.43x."
  - "Figure 10 (generation tokens/s/user at minimum latency): H100 BF16 509.131 vs NVFP4 575.667 = 1.13x; GB200 BF16 587.453 vs NVFP4 764.863 = 1.30x."
  - "Platforms validated (§5.2): A100, H100, B200, GB200, GB300, DGX Spark, DGX Station, RTX-class GPUs; quantization and speculative decoding tuned per platform."
  - "Figure 11 (vLLM, best TPS/GPU at TPS/user >= 50), H100: Qwen3.5-122B-A10B 424.94 (1.00x), Qwen3.8-Flash-Next 752.95 (1.77x), Nemotron Super 3.5 988.87 (2.33x)."
  - "Figure 11, GB200: Qwen3.5-122B-A10B 1,854.35 (1.00x), Qwen3.8-Flash-Next 2,282.73 (1.23x), Nemotron Super 3.5 4,331.88 (2.34x); Super / Flash-Next = 4,331.88 / 2,282.73 = 1.90x (derived). Same GB200 bars appear in the Figure 1 right panel."
  - "CONTRADICTION: Abstract claims 'up to 4,632 output tokens/s/GPU on GB200 at >= 50 tokens/s/user'; §1 and §6 claim 2.88x the throughput of Qwen3.8 Flash-Next on GB200. Figure 11 / Figure 1 show 4,331.88 vs 2,282.73 (1.90x). The report does not reconcile these."
  - "4,632 appears in Table 23 as the ReplaySSM-OFF output TPS/GPU at concurrency 64, MTP 5 (83.0 TPS/user) on one GB200; the report does not explicitly tie the Abstract number to that row."
  - "Performance model (§5.2): advantage attributed to smaller per-request state (fewer attention layers, narrower KV heads, Mamba-2 state constant in context), higher MTP acceptance length under probabilistic draft sampling (Table 9), and LatentMoE exchanging smaller latent vectors in expert-parallel communication."
  - "Prefix reuse + speculative decoding (§5.3): one Mamba cache page >= ~4K tokens of Attention KV in bytes; vLLM optionally retains an extra Mamba checkpoint near the speculative replay boundary (e.g. T-16 instead of T-L_page) so only ~16 tokens are replayed to recover the drafter's hidden state."
  - "Kernel control (§5.4): global linear backend stays default; a kernel config can override it per quantization scheme (e.g. specialized backend for NVFP4 W4A16 layers)."
  - "ReplaySSM / checkpointing replay (§5.4): store checkpoint state plus a small buffer of (k, v, alpha); reduces per-step state memory traffic from T+1 to roughly 1 + 1/L; shifts work to Tensor Cores; quantizes less often; enables larger batch x draft length."
  - "Table 23 (one GB200, vLLM, 16-step buffer, default KV-cache dtype): ReplaySSM improves output throughput 4.7-6.4% and per-user generation throughput 6.4-12.5% across matched configs; best ON value 5,728 TPS/GPU at conc 128 / MTP 4 at 49.1 TPS/user (OFF N/A)."
  - "Internal prefill checkpoints (§5.5): Mamba-2 adaptation of a checkpoint builder/exporter first written in vLLM for Kimi K3; writes the reusable Mamba state at a cache-block boundary inside one prefill forward instead of splitting into two forwards."
  - "Table 24 (Nemotron 3.5 Super VL NVFP4, 11 VR200 GPUs, Dynamo + vLLM, SWE-bench Pro replay): aggregated conc 616, 2,969 TPS/GPU, 96.6 TPS/user, TTFT 0.49 s; disaggregated (7 prefill + 4 decode) conc 528, 3,276 TPS/GPU (+10.3%), 155.9 TPS/user (+61%), TTFT 1.12 s."
  - "Vera Rubin engine settings (§5.6): NVFP4 weights, FP8 KV cache, FP16 Mamba state cache with stochastic rounding, MTP draft length 5 with fixed acceptance length 4.06, 240 GiB host-memory KV tier, KV-event-aware routing with session affinity, NIXL KV transfer, multi-wave KV split in decode attention."
related_steps: []
currency: "frozen"
---

# Scope

Answers:
- What workload and "usable" criterion does the report use to tune and compare serving?
- How much does NVFP4 help over BF16 on H100 and GB200 (Figure 10)?
- What is Super's max throughput vs Qwen3.5-122B-A10B and Qwen3.8-Flash-Next, and how does that square with the 2.88x / 4,632 headline claims?
- Why does the report think Super wins (performance model)?
- How are prefix caching and MTP speculative decoding made composable for a hybrid Mamba model?
- What kernel-control, ReplaySSM, and Mamba-checkpoint changes were made, and what do they buy (Table 23)?
- What does disaggregated serving on Vera Rubin give (Table 24)?

All serving results in §5 use vLLM (Figures 10-11, Table 23); §5.6 adds NVIDIA Dynamo on top of vLLM.

# Workload and the "usable" criterion (§5 intro, §5.1, §4)

| Item | Value | Source |
|---|---|---|
| Turns | 20 | §5 intro; Fig 10/11 captions |
| Input tokens per turn | 2,048 ("2K" in captions) | §5 intro; Fig 10/11 captions |
| Output tokens per turn | 1,024 ("1K" in captions) | §5 intro; Fig 10/11 captions |
| Prefix reuse | Shared prompt and prior conversation eligible on subsequent turns | §5 intro |
| Shared prefix | "50k prefix" (§5.1, Figure 10); 51,200 tokens (Table 23 workload text) | §5.1; §5.4 |
| Default quantization-eval assumption | 51200/2048/1024 as pre-cached ISL, new ISL, OSL | §4 |
| Usable point | Reported per-user generation rate >= 50 tokens/s | §5 intro |
| Objective | Highest aggregate output throughput among usable points; operating point depends on platform and serving config | §5 intro, §5.2 |

Points below 50 output tokens/s/user are excluded from the usable region even when they give higher aggregate throughput (§5.2). The Vera Rubin study (§5.6) uses a different workload (SWE-bench Pro replays) and a stricter usable definition: median output speed >= 50 tokens/s/user AND median TTFT < 5 s.

The report names three stack requirements this regime stresses (§5 intro): (1) prefix reuse must stay effective for a hybrid Mamba-Attention model when speculative decoding changes the resume position; (2) kernel selection must be controllable finer than one global policy, since the best kernels at a balanced operating point need not win at either end of the latency-throughput curve; (3) one model must map to accelerators with different memory capacity, bandwidth, and quantization support.

# Efficiency: NVFP4 vs BF16 (§5.1, Figure 10)

Figure 10: vLLM, 20-turn proxy workload, 2K in / 1K out per turn (§5.1 text adds the 50k prefix). NVFP4 runs as W4A16 on H100 and W4A4 on GB200 (figure axis labels; consistent with §4: Super 3.5 VL's NVFP4 checkpoint deploys W4A4 on Blackwell and W4A16 on non-Blackwell).

| Platform | NVFP4 mode | Max throughput at >= 50 TPS/user, BF16 (tok/s/GPU) | NVFP4 (tok/s/GPU) | Gain | Gen TPS/user at min latency, BF16 | NVFP4 | Gain |
|---|---|---|---|---|---|---|---|
| H100 80GB | W4A16 | 522 | 970 | 1.86x | 509.131 | 575.667 | 1.13x |
| GB200 (B200 GPU) | W4A4 | 1,778 | 4,324 | 2.43x | 587.453 | 764.863 | 1.30x |

Speculative-decoding settings, concurrency, and parallelism for Figure 10 are not reported.

# Performance across deployment platforms (§5.2, Figure 11, Figure 1)

**Platforms validated:** A100, H100, B200, GB200, GB300, DGX Spark, DGX Station, and RTX-class GPUs (§5.2). Quantization and speculative decoding are tuned per platform rather than fixed; the objective stays the same (max aggregate throughput on the 20-turn proxy at >= 50 output tokens/s/user). Per-platform results are reported only for H100 and GB200 (Figures 10-11) and VR200 (Table 24); other platforms: not reported.

**Figure 11** — "Best TPS/GPU by Model (TPS/user >= 50)", vLLM, 20-turn proxy, 2K/1K per turn. Multipliers are as printed in the figure, normalized to Qwen3.5-122B-A10B.

| Model | H100 (tok/s/GPU) | H100 rel. | GB200 (tok/s/GPU) | GB200 rel. |
|---|---|---|---|---|
| qwen3.5-122b-a10b | 424.94 | 1.00x | 1,854.35 | 1.00x |
| qwen3.8-flash-next | 752.95 | 1.77x | 2,282.73 | 1.23x |
| nemotron-super-3.5 | 988.87 | 2.33x | 4,331.88 | 2.34x |

Derived (not printed in the report): Super vs Qwen3.8-Flash-Next = 988.87 / 752.95 = 1.31x on H100 and 4,331.88 / 2,282.73 = 1.90x on GB200.

The GB200 panel is reproduced as the right panel of **Figure 1** ("Maximum Output Throughput (NVIDIA GB200)", subtitle "20-turn agentic proxy · TPS/user >= 50"): Qwen3.5 122B-A10B 1,854.35 (1.00x), Qwen3.8 Flash-Next 2,282.73 (1.23x), N3.5 Super 120B-A12B 4,331.88 (2.34x). The Figure 1 caption does not describe this panel.

Precision, checkpoint, speculative-decoding, and system configuration for each model in Figure 11 are not reported: the §5.2 text says they are "reported in the final caption", but the Figure 11 caption gives only the workload and the >= 50 TPS/user threshold.

# Headline claims vs Figure 11 (unreconciled)

| Claim | Location | Value |
|---|---|---|
| "up to 4,632 output tokens/s/GPU on GB200 at an interactive serving rate of at least 50 tokens/s/user" | Abstract | 4,632 tok/s/GPU |
| "On GB200, Nemotron 3.5 Super achieves 4,632 output tokens/s/GPU, delivering 2.88x the throughput of Qwen3.8 Flash-Next ... while sustaining at least 50 tokens/s/user" | §1 Introduction | 4,632; 2.88x |
| "On GB200, Super achieves 2.88x the serving throughput of Qwen3.8 Flash-Next under the evaluated interactive agentic workload" | §6 Conclusion | 2.88x |
| Figure 11 / Figure 1 GB200 bars | §5.2, Figure 11; Figure 1 right panel | Super 4,331.88; Qwen3.8-Flash-Next 2,282.73 (Super/Flash-Next = 1.90x derived); Qwen3.5-122B-A10B 1,854.35 (2.34x printed) |
| Figure 10 GB200 NVFP4 bar | §5.1, Figure 10 | 4,324 |
| Table 23 ReplaySSM-OFF, conc 64, MTP 5 | §5.4, Table 23 | 4,632 tok/s/GPU at 83.0 TPS/user (ON: 4,856 at 89.9) |

Notes, without reconciling beyond the report:
- 2.88x does not match any multiplier printed in Figure 11 or Figure 1 (1.23x for Flash-Next, 2.34x for Super, both vs Qwen3.5-122B-A10B). It also does not equal 4,632 / 2,282.73 (= 2.03x, derived). The report gives no Flash-Next number from which 2.88x would follow.
- 4,632 is not the Figure 11 value (4,331.88) nor the Figure 10 value (4,324). It matches the Table 23 ReplaySSM-OFF row (conc 64, MTP 5), but the report does not say that is the source; Table 23 also contains a higher usable value (4,856 with ReplaySSM ON at 89.9 TPS/user), so "up to 4,632" is not the maximum usable number in Table 23 either.
- Figure 10 (4,324) and Figure 11 (4,331.88) both claim max GB200 throughput at >= 50 TPS/user on the same nominal workload but differ slightly; the report does not explain the difference.

# Performance model (§5.2, "Performance Modeling and Analysis")

The report's internal performance model attributes the Figure 11 advantage to three factors:

| Factor | Report's argument |
|---|---|
| Per-request state | Each decode step reads weights once (shared by the batch), but each request reads/updates its own KV cache and recurrent state, which must stay in GPU memory. Super has fewer attention layers with narrower KV heads; Mamba-2 state does not grow with context, so per-request memory grows much more slowly as agentic sessions accumulate history. More requests fit per GPU, so more requests share each weight read. |
| vs Qwen3.8 | Each Qwen3.8 request needs more GPU memory at these lengths, limiting batch; its sparse attention reduces KV read per step but not KV stored. |
| vs Qwen3.5 | Stores about as much KV as Qwen3.8 and reads all of it every step, costing more memory and bandwidth per request. |
| Tokens per step | MTP head reaches higher average acceptance length than both Qwen models under probabilistic draft sampling (Table 9). |
| Expert communication | LatentMoE exchanges latent vectors smaller than Qwen hidden states, reducing expert-parallel communication per token. |
| Remaining cost | Updating Mamba state is a large per-request cost because speculative decoding writes it for every verified token; ReplaySSM (§5.4) reduces this traffic and memory. |

Table 9 averages (SPEED-Bench qualitative split, draft length 7; greedy temp=0, parenthetical temp=1 probabilistic draft sampling): Super boosted MTP 4.718 (4.612); Super native 4.288 (3.787); Qwen3.5-122B-A10B 4.799 (4.331); Qwen3.8-Flash-Next low 4.425 (4.213), medium 4.375 (4.095), xhigh 3.764 (3.408); Gemma-4 31B 4.837 (4.565). Under greedy decoding Qwen3.5-122B-A10B (4.799) exceeds Super boosted (4.718); the §5.2 claim is specifically for probabilistic draft sampling.

# Prefix reuse with speculative decoding (§5.3)

- **Coarse hybrid checkpoints.** Prefix reuse for Attention is per-token KV in small pages; a hybrid must also recover the Mamba recurrent state at the resume position, which is much larger per cached position. vLLM matches Attention and Mamba cache-page sizes by byte size; on the target Mamba-2 model one Mamba page occupies at least as many bytes as approximately 4K tokens of Attention KV, so Mamba states are kept on a much coarser token grid.
- **Problem.** With a prefix-cache hit through token T, drafting T+1 needs the target model's intermediate hidden state at T, which the cache does not store. vLLM reconstructs it by resuming from the latest Mamba checkpoint, which may be at T - L_page, forcing a replay of L_page tokens.
- **Fix.** Optionally retain an additional Mamba cache page near the speculative-decoding replay boundary. Example: with a 16-token fine-grained prefix-match unit, the extra checkpoint sits near T - 16, so a later request replays only the last 16 target-model tokens. Scheduler and cache manager jointly select and publish a valid checkpoint at the shared prefix boundary (avoiding stale recurrent state), trading one extra Mamba page for less recomputation. vLLM PR #53945 (footnote 22); merge status not stated.
- **Why it matters more for Super.** Non-SSM models have small pages so replay is cheap; Mamba-2's larger page footprint makes L_page larger than for hybrids with smaller recurrent states.
- **Result.** Prefix caching (removes repeated prefill) and speculative decoding (fewer target decode steps) become composable. No quantitative speedup is reported for this change alone.

# Kernel control at the target operating point (§5.4)

- **Per-quantization GEMM backend selection.** The throughput-optimal kernel at saturation is not necessarily best at the intermediate batch sizes needed to sustain >= 50 tokens/s/user, and a mixed-precision checkpoint has linear layers with different quantization schemes, shapes, and best backends. The global linear backend remains the default; a kernel configuration can override it per quantization scheme (e.g. a specialized backend for NVFP4 W4A16 layers while other layers stay on the platform default). Lets Hopper, Blackwell, and compact systems use different backend maps with the same model artifact and workload objective. vLLM PR #51204 (footnote 23); merge status not stated. No quantitative gain reported.

# ReplaySSM: checkpointing replay for the SSM cache (§5.4, Table 23)

Mechanism (described as growing out of a suggestion in the Nemotron v3 Ultra tech report):
- Mamba-2 decode: S_new = alpha * S_old + k^T v; output o = q * S_new. Speculative decoding computes this for T incoming tokens and writes every possible new state because the accepted count is unknown, making decode extremely memory bound and capping batch_size x draft_length by memory capacity.
- Checkpointing replay stores a checkpoint S_ckpt and a small buffer of (k_i, v_i, alpha_i) since the checkpoint; S is reconstructed on demand with no quantization error beyond that in S_ckpt. Rejected tokens: decrement a counter. Outputs can be computed directly as q S_ckpt + sum_i (q k_i^T) v_i (ignoring alpha terms, per the report). When the buffer fills it is flushed/folded by reconstructing and writing the state; for speculative decoding only accepted tokens are kept, and that step's tokens seed the new buffer.
- Per-step state-related memory traffic drops from T + 1 to roughly 1 + 1/L (L = buffer length, as printed). Also amortizes per-write overhead (randomness generation, stochastic rounding, scaling). FLOPs go up, but matvecs/outer products move from CUDA cores to Tensor Cores so compute time goes down.
- Report's summary: reduces memory traffic on an IO-bound op; reduces compute time via Tensor Cores; reduces quantization error by quantizing less often; enables larger batch sizes and speculation lengths.
- Implemented in TensorRT-LLM; independently discovered by the Dao lab ("ReplaySSM"), with whom NVIDIA collaborated to bring it to vLLM.

Table 23 — ReplaySSM OFF/ON, Nemotron Super 3.5, one GB200, vLLM, 16-step buffer, default KV-cache dtype. Workload: 20 turns, 51,200-token shared prefix, 2,048 input / 1,024 output per turn. "MTP" = number of speculative tokens. Gains computed before rounding; N/A = non-optimal values; individual measurements.

| Conc. | MTP | Output TPS/GPU OFF | ON | Gain | Gen TPS/user OFF | ON | Gain |
|---|---|---|---|---|---|---|---|
| 64 | 3 | 4,374 | 4,628 | +5.8% | 80.2 | 86.0 | +7.2% |
| 64 | 4 | 4,458 | 4,729 | +6.1% | 81.6 | 86.9 | +6.4% |
| 64 | 5 | 4,632 | 4,856 | +4.8% | 83.0 | 89.9 | +8.4% |
| 64 | 7 | 4,629 | 4,846 | +4.7% | 81.3 | 91.5 | +12.5% |
| 128 | 1 | 4,566 | 4,859 | +6.4% | 38.1 | 41.9 | +10.0% |
| 128 | 4 | N/A | 5,728 | N/A | N/A | 49.1 | N/A |

Text summary (§5.4): across matched configurations, +4.7-6.4% output throughput and +6.4-12.5% per-user generation throughput. Note (derived from the table): both concurrency-128 rows fall below the 50 tokens/s/user usable threshold.

# Reducing the cost of Mamba checkpoints (§5.5)

- **Problem.** In aligned Mamba cache mode, a reusable recurrent state must be materialized at a cache-block boundary. Enforcing that by ending a prefill step at the boundary and issuing another forward for the rest of the prompt repeats fixed forward-pass overhead and re-traverses Attention, MoE, routing, and TP communication.
- **Fix.** Internal prefill checkpointing, first implemented in vLLM for Kimi K3, adapted to Mamba-2 (vLLM PR #57329, footnote 24). The Mamba scan already materializes intermediate states at logical chunk boundaries; the adapted path makes a logical chunk end at the desired cache boundary, extracts the state and convolution window there, and writes it into the paged cache, all within one model forward.
- No quantitative gain reported.

The report's §5.5 wrap-up: prefix-aware speculative decoding cuts repeated prefill and decode work; per-quantization backend selection exposes kernel choices for balanced operating points; internal Mamba checkpoints cut the cost of creating reusable state.

# Disaggregated serving on NVIDIA Vera Rubin (§5.6, Table 24)

Setup: Nemotron 3.5 Super VL NVFP4 checkpoint, NVIDIA Dynamo + vLLM, VR200 GPUs, one GPU per model instance (no TP). Workload replays multi-turn agentic coding sessions from SWE-bench Pro traces (long contexts, high prefix reuse). MTP with fixed acceptance length 4.06 and draft length 5. Usable = median output >= 50 tokens/s/user and median TTFT < 5 s; report output tokens/s/GPU at the highest-throughput usable load. 11 GPUs per deployment.

| Deployment | Layout | Conc. | TPS/GPU | TPS/user (median) | TTFT (s, median) |
|---|---|---|---|---|---|
| Aggregated | 11 independent prefill+decode instances, KV-cache-aware router | 616 | 2,969 | 96.6 | 0.49 |
| Disaggregated | 7 prefill-only + 4 decode-only GPUs, KV moved prefill -> decode | 528 | 3,276 | 155.9 | 1.12 |
| Gain | | | +10.3% | +61% | |

Explanation (§5.6): prefill and decode no longer share a GPU, so decode GPUs run larger decode batches without long prefills between steps. TTFT rises 0.49 s -> 1.12 s but stays well under the 5 s target.

Engine settings (both deployments, §5.6):

| Area | Setting |
|---|---|
| Weights / KV cache | NVFP4 weights; FP8 KV cache |
| Mamba state cache | FP16 with stochastic rounding (as in §4) |
| Compilation | Attention fused with quantization of its output; CUDA graphs for full and piecewise steps |
| Decode attention | Multi-wave KV split (splits KV further than one wave allows and runs several waves); added to FlashInfer (PR #6331) and vLLM (PR #61077) |
| Speculative decoding | MTP head drafts 5 tokens per step |
| Prefix caching | Mamba states at KV-block boundaries plus extra checkpoints at prefixes shared by several requests; vLLM option `disable_eagle_block_drop` lets a cache hit keep its last full block with spec decoding on |
| Host KV tier | 240 GiB host (CPU) memory KV tier per aggregated instance and per prefill GPU, retaining earlier-turn prefixes after GPU eviction |
| Routing | Workers publish KV-cache events; session affinity keeps all turns of a session on one worker; NIXL transfers KV prefill -> decode in disaggregated mode |

# Caveats

- **Unreconciled headline:** 2.88x vs Qwen3.8 Flash-Next (§1, §6) and "up to 4,632 tok/s/GPU" (Abstract, §1) are not shown in Figure 11 / Figure 1, which give 4,331.88 vs 2,282.73 (1.90x derived) and 2.34x vs Qwen3.5-122B-A10B. The report does not explain the gap. Prefer quoting the figure values with their source when precision matters.
- Figure 11's configuration (precision, checkpoint, MTP settings, system configuration) is not reported despite §5.2 saying it is in the caption; §5.2 also describes Figure 11 as comparing "a Qwen model of comparable size on a representative platform", while the figure shows two Qwen models on two platforms.
- Workload prefix length is stated inconsistently in wording ("50k prefix" in §5.1 vs 51,200 tokens in Table 23 / §4 default); Figure 10 and 11 captions mention only 2K/1K per turn and 20 turns.
- Figure 10 (4,324) and Figure 11 (4,331.88) GB200 Super values differ slightly with no explanation.
- Table 23 values are individual measurements on one GB200; concurrency-128 rows are below the 50 tokens/s/user usable threshold.
- Table 24 uses a different workload (SWE-bench Pro replay), platform (VR200), serving stack (Dynamo + vLLM), and usable definition (adds median TTFT < 5 s) from Figures 10-11; its numbers are not comparable to them.
- §5.3 (extra Mamba checkpoint), §5.4 per-quantization backend selection, and §5.5 internal prefill checkpoints have no standalone quantitative results.
- The ReplaySSM state-reconstruction formula is reported as printed (the extracted product subscript reads j > i+1); consult the PDF for exact notation.
- Lightning inference results are out of scope for this file; quantization recipes and SSM-cache quantization are in §4 (see quantization.md).
