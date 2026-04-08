# SKILL: Stage 2 -- Reinforcement Learning (RL)

## Purpose

Align the SFT model with human preferences and improve reasoning quality using reinforcement learning. Supports two approaches: DPO (Direct Preference Optimization) for offline preference data, and GRPO (Group Relative Policy Optimization) for online reward-based training.

## When to Use

Run this stage after stage1_sft when you need:
- Preference alignment (model chooses better responses)
- Improved reasoning and chain-of-thought quality
- Safety alignment (reduce harmful outputs)
- Task-specific optimization with verifiable rewards (math, code)

Skip this stage if:
- SFT model already meets quality requirements
- No preference data or reward signal available
- Rapid iteration is needed (RL is the most compute-intensive alignment stage)

## Inputs Required

Before running this stage, confirm these with the user:

| Input | Required? | Default | Notes |
|-------|-----------|---------|-------|
| DPO or GRPO | Yes | None | Ask: "Which RL method? DPO (if you have preference pairs) or GRPO (if you have a reward signal)?" |
| Preference data path (DPO) | If DPO | None | Ask: "Where is your preference data? (JSONL with chosen/rejected pairs)" |
| Preference data source (DPO) | If DPO and no data | None | Ask: "Do you have preference pairs, or should we generate them using an LLM judge?" |
| Reward environment (GRPO) | If GRPO | `math_with_judge` | Ask: "What reward environment? (math_with_judge, code_gen, instruction_following, mcqa, or custom)" |
| Prompt data path (GRPO) | If GRPO | None | Ask: "Where are the training prompts? (JSONL with messages)" |
| SFT checkpoint path | Yes | None | Ask: "Path to SFT checkpoint from stage 1? (HuggingFace format for GRPO, Megatron for DPO)" |
| KL penalty | No | 0.01 (GRPO) / 0.05 (DPO) | Ask: "KL penalty for divergence control? (higher = more conservative, 0.0-0.1)" |
| Compute: number of nodes | Yes | 4 (GRPO) / 2 (DPO) | Ask: "How many nodes? (GRPO needs 4+ nodes, DPO can use 2+)" |
| Compute: GPUs per node | Yes | 8 | Ask: "How many GPUs per node?" |
| Executor type | Yes | Slurm | Ask: "Where will this run? (Slurm recommended; GRPO is not suited for local)" |
| Max training steps | No | 100 (GRPO) / 150 (DPO) | Ask: "How many training steps?" |

If any required input is missing, ask the user before proceeding.

## DPO vs GRPO Decision

| Criterion | DPO | GRPO |
|-----------|-----|------|
| Data required | Chosen/rejected response pairs | Prompts + reward function |
| Compute cost | Lower (offline, no generation) | Higher (online generation + training) |
| Quality ceiling | Limited by preference data quality | Can exceed data quality via exploration |
| Best for | Safety alignment, style preferences | Math, code, verifiable tasks |
| Infrastructure | Standard Megatron-Bridge training | Ray + vLLM + Megatron |

**Decision rule:**
- If you have preference pairs (human-annotated or AI-judged) -> use DPO
- If you have a reward function (automated judge, code execution, math verification) -> use GRPO
- If you have both -> run DPO first, then GRPO

## Prerequisites

| Prerequisite | Description |
|-------------|-------------|
| SFT checkpoint | From stage1_sft (HuggingFace format for GRPO, Megatron for DPO) |
| Preference data (DPO) | JSONL with chosen/rejected pairs |
| Prompt data (GRPO) | JSONL with prompts + reward config |
| GPU cluster | 4+ nodes x 8 GPUs for Nano GRPO; 2+ nodes for DPO |
| Container (GRPO) | `nvcr.io/nvidia/nemo:25.11.nemotron` |
| Container (DPO) | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` |

## Data Format

### DPO Data Format

```jsonl
{"prompt": "<user prompt>", "chosen": "<preferred response>", "rejected": "<dispreferred response>"}
```

### GRPO Data Format

```jsonl
{"messages": [{"role": "user", "content": "<prompt>"}]}
```

GRPO generates responses online and scores them with the configured reward function (NeMo Gym environments).

## Config Reference

### Unified Config (`config/default.yaml`)

Both DPO and GRPO are configured via a single `config/default.yaml` file. The `training_type` field selects the mode. Override it on the command line as needed.

```yaml
# Set training_type to "dpo" or "grpo"
training_type: grpo

# --- GRPO settings (used when training_type: grpo) ---
grpo:
  num_prompts_per_step: 12
  num_generations_per_prompt: 8
  max_rollout_turns: 1
  max_num_epochs: 1
  max_num_steps: 100
  normalize_rewards: true
  use_leave_one_out_baseline: true
  val_period: 10
  val_at_start: false
  seed: 42

loss_fn:
  reference_policy_kl_penalty: 0.01
  reference_policy_kl_type: k3
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  token_level_loss: true

# --- DPO settings (used when training_type: dpo) ---
dpo:
  max_num_epochs: 1
  max_num_steps: 150
  val_period: 25
  reference_policy_kl_penalty: 0.05
  preference_loss_weight: 1
  sft_loss_weight: 0
  seed: 42

# --- Policy (shared between DPO and GRPO) ---
policy:
  model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  tokenizer:
    name: ${policy.model_name}
    chat_template_kwargs: null
  train_global_batch_size: 96
  train_micro_batch_size: 1
  max_total_sequence_length: 8192
  precision: bfloat16
  max_grad_norm: 1.0

  megatron_cfg:
    enabled: true
    tensor_model_parallel_size: 4
    expert_tensor_parallel_size: 1
    expert_model_parallel_size: 8
    pipeline_model_parallel_size: 1
    sequence_parallel: true

  optimizer:
    name: torch.optim.AdamW
    kwargs:
      lr: 5.0e-6
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1e-8

  generation:
    backend: vllm
    max_new_tokens: ${policy.max_total_sequence_length}
    temperature: 1.0
    top_p: 1.0

# --- Data ---
data:
  dataset_name: OpenMathInstruct-2
  shuffle: false
  num_workers: 1
  max_input_seq_length: ${policy.max_total_sequence_length}

# --- Checkpointing ---
checkpointing:
  enabled: true
  checkpoint_dir: ./output/rl_checkpoints
  metric_name: "val:accuracy"
  higher_is_better: true
  keep_top_k: 3
  save_period: 10

# --- Cluster ---
cluster:
  gpus_per_node: 8
  num_nodes: 4
```

### Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `grpo.num_prompts_per_step` | 12 | 32-512 | More = better gradient estimates, more compute |
| `grpo.num_generations_per_prompt` | 8 | 4-32 | More = better advantage estimates |
| `loss_fn.reference_policy_kl_penalty` | 0.01 | 0-0.1 | Higher = more conservative (less forgetting) |
| `loss_fn.ratio_clip_min` | 0.2 | 0.1-0.3 | PPO-style clipping |
| `loss_fn.ratio_clip_max` | 0.2 | 0.2-0.4 | Symmetric clipping (same as clip_min) |
| `policy.optimizer.kwargs.lr` | 5e-6 | 1e-6 to 1e-5 | RL learning rate |
| `dpo.reference_policy_kl_penalty` | 0.05 | 0.01-0.5 | Higher = more conservative DPO |
| `cluster.num_nodes` | 4 | 4-64 | GRPO is compute-intensive |

## Execution

### DPO (Megatron-Bridge, Slurm)

```bash
nemotron customize rl \
  -c src/nemotron/customization_recipes/nemotron/stage2_rl/config/default.yaml \
  --run MY-CLUSTER \
  training_type=dpo \
  policy.model_name=/results/sft_checkpoint
```

### GRPO (Ray, Slurm)

```bash
nemotron customize rl \
  -c src/nemotron/customization_recipes/nemotron/stage2_rl/config/default.yaml \
  --run MY-CLUSTER \
  training_type=grpo \
  policy.model_name=/results/sft_checkpoint_hf
```

GRPO uses Ray for distributed execution. The CLI automatically selects RayJob execution when the config specifies Ray-based training.

### Local (DPO Only, Single Node)

```bash
nemotron customize rl \
  -c src/nemotron/customization_recipes/nemotron/stage2_rl/config/default.yaml \
  training_type=dpo
```

GRPO requires multiple nodes and is not recommended for local execution.

## Reward Configuration for GRPO

GRPO uses NeMo Gym for reward computation. Common reward environments:

| Environment | Use Case | Config Path |
|-------------|----------|-------------|
| `math_with_judge` | Math problem verification | `resources_servers/math_with_judge/configs/math_with_judge.yaml` |
| `code_gen` | Code execution + unit tests | `resources_servers/code_gen/configs/code_gen.yaml` |
| `instruction_following` | Instruction compliance | `resources_servers/instruction_following/configs/instruction_following.yaml` |
| `mcqa` | Multiple choice QA | `resources_servers/mcqa/configs/mcqa.yaml` |

For custom domain rewards, implement a reward server compatible with NeMo Gym's API and add its config path to `env.nemo_gym.config_paths`.

## How to Verify Success

1. **Reward curve (GRPO)**: Mean reward should increase over training steps.
   - If reward plateaus early: increase `num_generations_per_prompt` or adjust temperature
   - If reward oscillates: reduce learning rate

2. **DPO accuracy**: Fraction of preference pairs where model agrees with "chosen".
   - Target: >70% accuracy on held-out preference data
   - If <60%: data quality issue or beta too high

3. **KL divergence from reference**: Monitor to ensure model doesn't drift too far.
   - If KL > 10 nats: increase `reference_policy_kl_penalty`
   - If KL ~ 0: model barely changed (lr too low or too few steps)

4. **Qualitative check**: Compare SFT vs RL model on the same prompts.
   - RL model should give more nuanced, well-reasoned responses
   - Should not refuse valid queries (over-alignment)

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| GRPO reward collapse (all generations get same score) | Reward model not discriminative enough | Check reward environment config, add diversity penalty |
| DPO loss stays flat | Beta too high or data too noisy | Reduce beta to 0.05, clean preference data |
| OOM during GRPO generation | vLLM memory allocation too high | Reduce `gpu_memory_utilization` to 0.3, reduce `max_model_len` |
| Ray cluster fails to start | Container/mount issues | Verify container image has nemo-rl installed, check mounts |
| Model becomes sycophantic | Over-optimization on preference signal | Increase KL penalty, add diversity in training prompts |
| vLLM generation errors | Model format mismatch | Ensure checkpoint is in HuggingFace format for vLLM |
| Slow GRPO training | Generation bottleneck | Increase `vllm_cfg.tensor_parallel_size`, reduce `max_new_tokens` |

## Artifacts Produced

| Artifact | Type | Path | Consumed By |
|----------|------|------|-------------|
| RL checkpoint | `ModelArtifact` | `checkpoint.save/` or `checkpointing.checkpoint_dir/` | stage4_eval, stage5_quantization |
| Training logs | W&B/TensorBoard | W&B project | Analysis |
| Generation samples | JSONL | Logged to W&B | Qualitative analysis |
