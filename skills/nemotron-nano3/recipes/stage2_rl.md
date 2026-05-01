# Stage 2 Recipe Bridge: RL

This file connects the paper’s RL sections to `src/nemotron/recipes/nano3/stage2_rl/`.

## What Exists Publicly

Public files:

- `src/nemotron/recipes/nano3/stage2_rl/data_prep.py`
- `src/nemotron/recipes/nano3/stage2_rl/train.py`
- `src/nemotron/recipes/nano3/stage2_rl/config/default.yaml`
- `src/nemotron/recipes/nano3/stage2_rl/config/tiny.yaml`
- `src/nemotron/recipes/nano3/stage2_rl/README.md`

The public RL stage is real and substantial.
It covers:

1. RL data prep into NeMo-RL-friendly JSONL
2. placeholder resolution for external HF-linked examples
3. GRPO training with NeMo-RL
4. NeMo-Gym reward environments
5. vLLM generation during rollouts

## What It Maps To In The Paper

| Paper section | Public recipe element |
|---|---|
| §3.2.1 environments | `env.nemo_gym.config_paths` |
| §3.2.2 curriculum | dataset blend + RL data prep |
| §3.2.3 RLVR surpassing SFT | not a recipe knob; paper result only |
| §3.2.4 infrastructure | NeMo-RL + Megatron backend + Ray + vLLM config |
| §3.2.5 algorithm | GRPO config values in `default.yaml` |
| §3.3 RLHF | paper-driven understanding; not a separate public recipe stage |
| Appendix C DPO | paper-driven understanding; not a standalone public stage here |

## Paper-Aligned Public Defaults

The public `config/default.yaml` exposes several paper-aligned numbers directly:

| Setting | Value |
|---|---:|
| `grpo.num_prompts_per_step` | 128 |
| `grpo.num_generations_per_prompt` | 16 |
| `policy.train_global_batch_size` | 2048 |
| `policy.max_total_sequence_length` | 49152 |
| optimizer LR | 3e-6 |
| tensor parallel | 2 |
| pipeline parallel | 2 |
| context parallel | 4 |
| expert parallel | 8 |
| cluster nodes | 32 |
| GPUs per node | 8 |

That makes stage2 the public Nano3 stage that most directly reflects the paper’s exposed algorithmic settings.

## Important Training Details In The Public Config

The default RL config also shows the paper’s intended stability choices in operational form:

- `freeze_moe_router: true`
- `moe_router_load_balancing_type: "none"`
- `moe_router_enable_expert_bias: true`
- `use_on_policy_kl_approximation: True`
- `use_importance_sampling_correction: True`
- `token_level_loss: True`
- sequence packing enabled
- colocated vLLM rollouts enabled

These are not generic RL defaults; they are Nano3-specific alignment choices.

## Environment Coverage

The public stage wires these NeMo-Gym environments:

- `math_with_judge`
- `code_gen`
- `workplace_assistant`
- `mcqa`
- `instruction_following`
- `structured_outputs_json`

That list mirrors the six-environment RLVR framing from the paper.

## Public Data Prep Specifics

`data_prep.py` also reveals an important public-reproduction detail:

- some training records are placeholders that resolve to external datasets like DAPO and Skywork
- the stage reconstructs usable JSONL samples from those linked records
- that makes stage2 a concrete example of how public RL data is stitched together for Nano3-like training

## Tiny Config Notes

The public `tiny.yaml` keeps the same basic GRPO structure but collapses resources to a debug-friendly shape:

| Setting | Value |
|---|---|
| cluster nodes | `1` |
| base HF model handle | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` |
| convert initial checkpoint to HF | `false` by default |
| data path | artifact-driven |

This is useful for debugging the stack, not for reproducing the paper’s RL scale.

## What The Public Stage Does Not Fully Expose

The paper discusses:

- RLHF with a large GenRM
- circular comparison strategy
- group-relative length control
- DPO experiments for hallucinated tool use

The repo gives you the GRPO/RLVR backbone directly, but not a separate one-click public stage for every appendix-level alignment component.

## Reproduce with nemotron-customize

The closest catalog step surface is:

- `rl/nemo_rl_grpo`

Important maturity note:

- the step exists in `src/nemotron/steps/` and is listed in `STEPS.md`
- its manifest marks it as **planned**
- so generation/build work should still be grounded on `src/nemotron/recipes/nano3/stage2_rl/`

Common surrounding steps:

- upstream `sft/megatron_bridge`
- downstream `eval/model_eval`
- `convert/megatron_to_hf` if the user wants HF export after RL

## Good Handoff Pattern

> “For the public Nano3 RL path, the nearest `nemotron-customize` surface is `rl/nemo_rl_grpo`, but it is still marked planned. I would ground the build on `src/nemotron/recipes/nano3/stage2_rl/` and use the step only as the conceptual catalog bridge.”
