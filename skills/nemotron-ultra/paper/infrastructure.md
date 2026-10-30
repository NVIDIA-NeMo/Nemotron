---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "infrastructure"
paper_sections: ["3.6", "3.6.1", "3.6.2", "3.6.3"]
title: "Post-training Infrastructure: RL rollout acceleration, scaling RL infra, future work"
summary: |
  Covers §3.6 post-training infrastructure. §3.6.1 accelerates RL rollout
  generation with MTP speculative decoding (k=5 gives 1.46x speedup). §3.6.2
  details scaling RL infrastructure on GB200/Slurm: failure attribution, Ray GCS
  and Slurm launch overheads, topology-aware NVLink placement (+20%), NUMA
  binding (+10%), checkpoint-save and JIT-cache optimizations, multi-node vLLM
  stability, and container/storage I/O. §3.6.3 outlines future fault-isolation work.
key_facts:
  - "RL and MOPD use a one-step off-policy asynchronous RL setup: rollout generation is overlapped with the policy update; step time is bounded by the slower stage (typically rollout generation, dominated by straggler generations)."
  - "Rollout generation accelerated with MTP speculative decoding: MTP head applied recurrently to propose k candidate tokens, verified by the base model in a single forward pass."
  - "Sweeping k in {0, 3, 5, 7}: k=5 gives the largest gain at 1.46x speedup over the k=0 (no-MTP) baseline (Figure 11a). MTP benefit concentrated in the long-tail (slowest) generations (Figure 11b)."
  - "Production cluster: NVIDIA GB200 nodes, Slurm orchestration, co-located CPUs for sandbox execution."
  - "RL software failure breakdown (Table 7): Generation engine failures/timeouts 56%, Sandbox/tool calling 36%, Other software issues 8% (generation + sandbox/tool ~92% of failures)."
  - "Infra optimization impacts (Table 8): Ray GCS startup 30+ min -> 10 min; Checkpoint blocking 60 sec -> <1 sec; Cold init (JIT) 38.8 min -> 0.4 min; Multi-node vLLM startup 25 min -> 9.5 min; Container extraction 2-3 min (with cascading failures) -> ~0s (warm)."
  - "Slurm launch: many serial srun RPCs scaled O(n) with node count; restructuring to a single multi-node srun reduced controller interactions to O(1); startup dropped from 30+ min to 10 min."
  - "Ray GCS: at 3K+ GPU scale the single-threaded GCS was overwhelmed by actor registrations (startups 25-49 min, thundering-herd failures); eliminated 40% of actor registrations by converting short-lived actors to tasks and pooling init actors per node; Anyscale fixes shipped in Ray 2.55 public release."
  - "GB200 NVL72 NVLink domain spans an entire rack (72 GPUs across 18 nodes); topology-aware NVLink domain placement keeps EP all-to-all on NVLink, delivering 20% end-to-end throughput improvement."
  - "NUMA binding: GPUs 0,1 -> NUMA node 0; GPUs 2,3 -> NUMA node 1; binding policy and vLLM workers to the socket-local CPU delivered 10% end-to-end throughput improvement."
  - "Checkpoint save: synchronous saves blocked training ~60s/save; NVRx async checkpointing cut exposed blocking to ~6-8s; with Megatron Core Distributed Optimizer exposed save time reduces to <1 second."
  - "JIT cold-start at 1K+ GPU scale ~49 min total, ~38.8 min dominated by JIT compilation; warm caches cut init from 38.8 min to 0.4 min (99% reduction)."
  - "Cold-vs-warm JIT (Table 9, 1K GPU): FlashInfer cubin 28.0 min -> 0; Inductor/torch.compile 5.5 min -> 0; Triton autotuning 2.0 min -> 0; vLLM CUDA graph capture 2.5 min -> 0; Model load 0.4 min -> 0.4 min; Total init 38.8 min -> 0.4 min."
  - "Container image is ~44 GB squashfs; concurrent reads at scale produced tens of TB; slow nodes stalled 12+ min during extraction (vs ~2-3 min normal); fixed via Enroot local squashfs cache and asymmetric read/write cache paths."
  - "Future work (§3.6.3) targets the two dominant failure categories: fail-fast fault isolation, component-level recovery, disaggregated sandbox/tool-calling infra, and fine-grained checkpointing of in-flight rollouts/KV cache/conversation state."
related_steps: []
currency: "frozen"
---

# Scope
Answers:
- How is RL rollout generation accelerated, and what speedup does MTP give?
- What hardware/orchestration does the production RL cluster use?
- What were the dominant RL software failure categories?
- What infrastructure optimizations were made and what was their measured impact?
- What is the planned future work for RL infrastructure?

# §3.6.1 Accelerating Rollout Generation with MTP
- One-step off-policy asynchronous RL setup; rollout generation overlapped with policy update; step time bounded by the slower stage (usually rollout generation, dominated by straggler generations).
- MTP speculative decoding proposes k candidate tokens per iteration, verified by the base model in one forward pass; accepted tokens committed without extra sequential decode steps.
- k swept over {0, 3, 5, 7}; k=5 best at 1.46x over no-MTP baseline (Figure 11a). Benefit concentrated in long-tail / slowest generations (Figure 11b): these emit many more tokens and decode at lower concurrency near batch end, where speculative decoding is most effective.

# §3.6.2 Scaling RL Infrastructure
Production cluster: NVIDIA GB200 nodes, Slurm orchestration, co-located CPUs for sandbox execution.

## Failure attribution (Table 7)
| Failure Category | % |
|---|---|
| Generation engine failures / timeouts | 56 |
| Sandbox / tool calling | 36 |
| Other Software issues | 8 |

Generation + sandbox/tool calling = ~92% of failures.

## Key optimizations and impact (Table 8)
| Issue | Before | After |
|---|---|---|
| Ray GCS startup | 30+ min | 10 min |
| Checkpoint blocking | 60 sec | <1 sec |
| Cold init (JIT) | 38.8 min | 0.4 min |
| Multi-node vLLM startup | 25 min | 9.5 min |
| Container extraction | 2-3 min (with cascading failures) | ~0s (warm) |

## Ray GCS scalability and Slurm launch overheads
- Heterogeneous NeMo-RL job launches training, vLLM generation, gym, and judge workers. Original launch issued multiple separate srun invocations per node; each srun is a serial RPC to slurmctld, scaling O(n) with node count -> bottleneck. Restructured to a single multi-node srun -> O(1) controller interactions; startup 30+ min -> 10 min.
- At 3K+ GPU scale, Ray's single-threaded GCS overwhelmed by actor registrations (startups 25-49 min, thundering-herd failures). Eliminated 40% of actor registrations (short-lived actors -> tasks, pooled init actors per node) plus aggressive GCS tuning. Anyscale resolved GCS regressions in Ray 2.55 public release.

## Topology-aware NVLink domain placement
- GB200 NVL72 NVLink domain = an entire rack (72 GPUs across 18 nodes). Without topology awareness, Megatron EP groups can span racks, forcing MoE all-to-all over InfiniBand instead of NVLink.
- Fix (threefold): probe script parses NVLink fabric ClusterUUID from `nvidia-smi -q` and registers a Ray custom resource (nvlink_domain_<ClusterUUID>) plus a topo_rank from SLURM_TOPOLOGY_ADDR; RayVirtualCluster sorts bundle indices by (domain_min_topo_rank, topo_rank, gpu_id); Megatron runs with external_gpu_device_mapping=True trusting Ray's GPU pinning.
- Impact: 20% end-to-end throughput improvement on GB200.

## NUMA binding for policy and vLLM workers
- Each GB200 NVL72 compute tray has a Grace CPU with two sockets / multiple NUMA nodes. GPUs 0,1 -> NUMA node 0; GPUs 2,3 -> NUMA node 1. NVLink-C2C is socket-local; cross-socket placement degrades GPU memory bandwidth.
- Fix: bind policy and vLLM workers to the socket-local CPU. Impact: 10% end-to-end throughput improvement on GB200.

## Checkpoint save blocking
- Synchronous saves blocked training ~60s/save. NVRx async checkpointing (params copied to CPU, persisted in background) cut exposed blocking to ~6-8s. Further: overlapped NCCL transfers with D2H copies, persistent checkpoint worker processes, background finalization, cached distributed save plan. With Megatron Core Distributed Optimizer (optimizer state sharded across DP ranks), exposed save time reduces to <1 second.

## JIT cache and initialization (Table 9, 1K GPU scale)
| Component | Cold Start | Warm Start |
|---|---|---|
| FlashInfer cubin compilation | 28.0 min | 0 (cached) |
| Inductor / torch.compile | 5.5 min | 0 (cached) |
| Triton kernel autotuning | 2.0 min | 0 (cached) |
| vLLM CUDA graph capture | 2.5 min | 0 (cached) |
| Model load | 0.4 min | 0.4 min |
| Total init/total | 38.8 min | 0.4 min |

- Cold start ~49 min total, ~38.8 min JIT-dominated. Fix: persistent warm cache on shared storage (tarballs written at job completion); node-local seeding into /tmp before Ray init; container-baked FlashInfer cubins via `flashinfer download-cubin`. Warm caches dropped init 38.8 min -> 0.4 min (99% reduction).

## Multi-node vLLM operational stability
- Issues at scale: package/kernel ABI mismatches across RL stack components (silent crashes); subprocess environment divergence (vars not propagated to multiprocessing.spawn subprocesses); collective-communication incompatibilities (certain JIT kernels incompatible with NCCL multi-node NVLink memory registration on GB200).
- Fixes: dependency unification to a single shared GPU-kernel-library version; explicit env/library-path forwarding to subprocesses; disable multi-node NVLink memory registration for affected kernel paths pending FlashInfer upstream fix; vLLM health checks, RPC timeouts, graceful shutdown + orphan cleanup.

## Container and storage I/O
- ~44 GB squashfs container image read concurrently by every node at startup -> tens of TB of reads; some nodes hit I/O error or stalled 12+ min (vs ~2-3 min normal). Job-completion JIT cache write-back created a second I/O storm.
- Fixes: Enroot local squashfs cache (warm nodes reuse cached copy); asymmetric read/write cache paths (JIT writes to node-local storage during training; single sidecar archives caches to shared storage at completion; one node's cache persisted since all nodes compile identical kernels; startup uses one sequential read per cache type).

# §3.6.3 Future Work
Targets the two dominant failure categories: fail-fast fault isolation to prevent retry-and-cascade modes (component-level recovery so individual generation workers / sandbox instances restart without full job restart); disaggregating sandbox and tool-calling infrastructure to eliminate cascading failures and allow independent scaling; fine-grained checkpointing of in-flight rollouts, KV cache, and conversation state to enable replay from the last consistent snapshot.

# Caveats
- The 1.46x rollout speedup is specific to k=5 MTP speculative decoding during RLVR training; do not generalize to other k or to non-RL inference.
- The +20% (NVLink topology) and +10% (NUMA binding) figures are end-to-end throughput improvements on GB200 specifically.
- Table 8/Table 9 before/after numbers are environment-specific (GB200, 1K+ GPU scale); do not treat as universal.
- Failure percentages (Table 7) are the observed RL software failure breakdown, not hardware failures.
