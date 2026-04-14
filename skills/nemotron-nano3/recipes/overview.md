# Nano3 Public Recipe Map

This directory explains how the public Nano3 recipe stack in `src/nemotron/recipes/nano3/` relates to the Nano3 paper.

## The Short Version

The paper describes the **full NVIDIA training/evaluation program** for Nemotron 3 Nano.
The public repo provides a **reference implementation** of the same stage structure:

1. pretraining
2. supervised fine-tuning
3. reinforcement learning
4. evaluation

But the repo repeatedly states that the public recipes use **open-source subsets** of the data and should be treated as **reference implementations**, not exact reproductions of the published benchmark runs.

## Stage Map

| Stage | Repo path | Main framework | Public artifact shape |
|---|---|---|---|
| 0 | `src/nemotron/recipes/nano3/stage0_pretrain/` | Megatron-Bridge | Megatron checkpoint |
| 1 | `src/nemotron/recipes/nano3/stage1_sft/` | Megatron-Bridge | packed SFT Parquet + Megatron checkpoint |
| 2 | `src/nemotron/recipes/nano3/stage2_rl/` | NeMo-RL + NeMo-Gym + vLLM | JSONL RL data + aligned Megatron checkpoint |
| 3 | `src/nemotron/recipes/nano3/stage3_eval/` | NeMo Evaluator | benchmark metrics / W&B results |

## Paper → Public Recipe Alignment

| Paper topic | Public stage | Notes |
|---|---|---|
| §2.1 architecture | stage0 pretrain | recipe invokes Nano3 Megatron-Bridge recipe target |
| §2.2 pretraining data | stage0 pretrain data prep | public data blend is an open subset, not the full internal mix |
| §2.3 two-phase curriculum | stage0 pretrain | represented indirectly via configs and data blends, not a single paper-faithful manifest |
| §2.5 long-context CPT | stage0 pretrain | public recipe is the closest pretraining scaffold, but long-context specifics are not surfaced as a one-command paper clone |
| §3.1 SFT | stage1 sft | public data prep applies chat template + role masking + packing |
| §3.2 RLVR | stage2 rl | public GRPO config mirrors major paper settings closely |
| §3.3 RLHF / DPO | stage2 rl + paper-only notes | RLHF details are mostly documented in the paper, not exposed as a separate public stage |
| §3.4 evaluations | stage3 eval | public default benchmark set is smaller than the full paper matrix |
| §4 quantization | model-card deployment path | public FP8 checkpoint is released, but quantization is not a separate recipe stage here |

## What the Public Recipes Reproduce Well

They are strongest when you want to reproduce the **shape** of the system:

- Megatron-format pretraining and SFT
- chat-template-based SFT data prep
- GRPO-based RL on JSONL task data
- evaluator-driven benchmark runs with W&B artifacts
- artifact lineage between stages
- cluster execution via NeMo-Run

## What They Do *Not* Promise

They do **not** promise:

- the full proprietary/private training data used for the paper
- exact published data mixtures and sampling schedules at every point
- exact cluster scale and internal infra used for all reported results
- exact published scores from a public clean-room rerun

That boundary should be stated clearly whenever the user asks about “reproducing the paper.”

## Where the Public Repo Is Especially Close to the Paper

Stage 2 RL is the closest public mirror of the paper’s training design because the repo config exposes many paper-aligned values directly:

- `grpo.num_prompts_per_step = 128`
- `grpo.num_generations_per_prompt = 16`
- `policy.train_global_batch_size = 2048`
- frozen MoE router
- aux-loss-free expert-bias updates
- NeMo-Gym environment list
- vLLM rollout backend

## Where the Public Repo Is Intentionally Lighter

The public SFT recipe is the clearest example.
The paper describes:

- 18M total samples
- 13k steps
- sequence packing to 256k
- many proprietary or not-fully-open post-training mixtures

The public stage1 recipe instead provides a compact operational path:

- open-source SFT blend
- packed Parquet data prep
- 4k-scale pack settings by default in the public pipeline
- cluster-friendly Megatron-Bridge configs

So the right framing is:

> “The repo exposes the methodology and wiring, not a literal paper benchmark rerun.”

## Reproduce with nemotron-customize

Use this map when the user shifts from “what is Nano3?” to “help me build something like Nano3.”

| Goal | `nemotron-customize` path |
|---|---|
| curate text corpora | `curate/nemo_curator` |
| pack SFT JSONL for Megatron | `prep/sft_packing` |
| run Nano3-style Megatron SFT | `sft/megatron_bridge` |
| run smaller-GPU SFT | `sft/automodel` |
| run GRPO alignment | `rl/nemo_rl_grpo` *(listed in the step catalog as planned; ground it on the stage2 recipe)* |
| benchmark a checkpoint | `eval/model_eval` |
| convert released HF weights to Megatron | `convert/hf_to_megatron` |
| export Megatron checkpoint back to HF | `convert/megatron_to_hf` |

## Important Gaps and Maturity Notes

`src/nemotron/steps/STEPS.md` does **not** currently expose a catalog `pretrain/*` step.
So when a user wants to build Nano3-like pretraining with `/nemotron-customize`:

- treat stage0 as an **Explorer-mode** or direct recipe task
- ground on `src/nemotron/recipes/nano3/stage0_pretrain/`
- optionally combine with `curate/nemo_curator` for upstream corpus work

For RL, `rl/nemo_rl_grpo` does appear in the step catalog, but its manifest is marked **planned**.
So RL requests should still be handled with recipe-grounded caution:

- mention `rl/nemo_rl_grpo` as the nearest step surface
- ground details on `src/nemotron/recipes/nano3/stage2_rl/`
- avoid implying that the RL catalog path is as mature as the SFT/eval paths

## Recommended Answer Pattern

When the user asks a reproduction question, answer in this order:

1. **Paper answer** — what the report says
2. **Public recipe answer** — what the repo exposes
3. **Customization answer** — which `/nemotron-customize` step(s) to use
4. **Gap note** — what is not public or not yet cataloged

## Read Next

- `stage0_pretrain.md`
- `stage1_sft.md`
- `stage2_rl.md`
- `stage3_eval.md`
