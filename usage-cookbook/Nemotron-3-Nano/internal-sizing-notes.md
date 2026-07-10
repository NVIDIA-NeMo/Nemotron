# Internal Sizing Notes: Nemotron 3 Nano 30B-A3B

These are internal development notes split out from the main sizing document.
They cover known gaps and future work items.

---

## Open Items (updated 2026-07-03 review)

1. ~~**4-GPU Automodel run**~~: Superseded — Automodel now ships an official
   **single-GPU** LoRA config (`nemotron_nano_v3_singlegpu_lora.yaml`, EP=1,
   activation checkpointing, memory-efficient LoRA). Empirical validation of
   the 4-GPU EP=4 config is still nice-to-have but no longer the floor.
2. **Megatron-Bridge EP=4**: Still not in existing recipes — EP override via
   CLI remains untested below 8 GPUs.
3. **`out_proj` LoRA target in Megatron-Bridge — CONFIRMED still a latent bug
   (re-verified 2026-07-05)**: The recipes' target list still includes
   `out_proj` (`nemotron_3_nano.py:515`; same in super/ultra recipes) with no
   Mamba-2 filtering. Important nuance: `exclude_modules` **cannot** be
   combined with `target_modules` (module_matcher asserts it's empty), so the
   remedy is to REPLACE the target list with the conservative default
   `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` — which is the `LoRA`
   class's own default; only the recipes extend it unsafely.
   **Action: file a Megatron-Bridge bug/PR to trim the recipe target lists.**
4. **Throughput benchmarks**: Still open. Automodel's perf page gained a LoRA
   fine-tuning section (2026-07) but it covers Llama/Qwen only — no Nemotron
   Nano row. Pretraining remains 328 TFLOPs/sec/GPU on 8x H100.
5. **FP8 KV cache validation**: Still open. Note the model now has a Day-0 FP8
   checkpoint variant (`NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`), and the
   Megatron-Bridge recipes expose commented-out FP8/MXFP8 knobs
   (`fp8_recipe`, `fp8_param_gather`, `moe_router_padding_for_fp8`).
6. **Long-context LoRA recipes**: Partially improved — the Megatron-Bridge
   SLURM sweep now includes a CP=2 config (`TP=2,EP=8,CP=2` on 16 GPUs), and
   Automodel has a `nemotron_nano_v3_cp_test.yaml`. Still no tested recipe
   for seq_len > 2048.
7. **NEW — sizing-relevant additions since April 2026**: NeMo RL added
   Megatron-backend GRPO (`grpo-...-megatron-lora.yaml`, EP=8),
   context-parallel + sequence-packing GRPO, single-node DPO recipes
   (1x4 = 4 GPUs min), and full SFT at 2x4 = 8 GPUs. Megatron-Bridge nano
   examples moved to `examples/models/nemotron/nemotron_3/nano/` and use the
   flex/DeepEP token dispatcher by default.

---

## Existing Memory Calculator Limitations (re-verified 2026-07-03)

The Megatron-Bridge utility at `training/utils/theoretical_memory_utils.py`:

| Gap | Impact | Severity | Status |
|-----|--------|----------|--------|
| ~~No Expert Parallelism (EP)~~ | ~~Overestimates by ~8x for EP=8~~ | — | **FIXED** — now divides routed-expert params by `PP × EP × ETP` (incl. dedicated expert optimizer shard sizing) |
| No LoRA/PEFT awareness | Assumes all 31.6B params are trainable — overestimates optimizer memory by ~49x at rank 32 | Critical | Still present (no peft/lora/adapter references in file) |
| No FSDP2 sharding | Cannot model the Automodel/NeMo RL memory distribution | Critical | Still present |
| No FP8 KV cache | Cannot model activation savings from FP8 quantization | Medium | Still present |

### What a corrected calculator would need

- Split parameters into `expert_params` and `non_expert_params`
- Divide expert params by `expert_model_parallel_size`
- For LoRA: allocate 2 bytes/param (BF16) for frozen base, 12 bytes/param
  (BF16 weight + FP32 optimizer + BF16 gradient) only for adapter parameters
- Account for FSDP sharding of frozen weights across DP (Automodel/NeMo RL path)
- Model activation memory as a function of sequence length, MBS, and precision

### How the current calculator works

Three functions in `theoretical_memory_utils.py`:

1. **`compute_weight_and_optimizer_memory()`**: Counts total transformer params
   (attention + MLP + layernorms) + embedding params. Computes the most-loaded
   shard as `(transformer_params/PP + embedding) / TP`. Multiplies by
   `18 bytes/param` (no dist opt) or `6 + 12/DP` (dist opt).
   **Update 2026-07:** now EP-aware — reads `expert_model_parallel_size` and
   `expert_tensor_parallel_size` and divides routed-expert params by
   `PP × EP × ETP`, with a dedicated expert optimizer shard size. The April
   2026 "all 128 experts replicated" bug is fixed.

2. **`compute_activation_memory()`**: Uses the Megatron-LM paper formula
   (Table 2): `seq * mbs * hidden * (18 + 4*ffn/h)` per layer, divided by TP.
   **Update 2026-07:** activation estimation now accounts for MoE top-k (the
   earlier "no MoE awareness" note is stale). Remaining gaps: treats every
   non-MoE layer as attention/dense (no Mamba-2 modeling), models MTP as
   copies of the final layer type — over-counts this hybrid checkpoint by
   ~0.5B params even for full SFT. Coarse planning tool only.

3. **`report_theoretical_memory()`**: Calls the above two functions and prints.
   Only computes activations if sequence parallelism AND selective recomputation
   are both enabled.

### Estimated overestimates for Nemotron 3 Nano LoRA (EP=8, LoRA rank=32)

| What it computes | Calculator result (2026-07) | Actual (estimated) | Overestimate |
|-----------------|------------------:|-------------------:|:------------:|
| Weight memory/GPU (EP=8, TP=1) | ~11 GiB (EP-aware now) | ~11 GiB | **fixed** |
| Optimizer memory/GPU (all params trainable) | ~15-59 GiB (depends on DP sharding) | ~1.2 GiB (LoRA only) | ~12-49x |
| Total weight+optimizer/GPU | ~26-70 GiB | ~12 GiB | ~2-6x |

> With the EP fix, the remaining error is entirely the missing LoRA awareness:
> optimizer + gradient memory is computed for all 31.6B params instead of the
> ~0.9B trainable adapter params.
