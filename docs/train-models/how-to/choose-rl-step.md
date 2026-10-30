# Choose an RL Alignment Step

Post-training alignment with NeMo RL is split into three steps under `rl/nemo_rl/`. The table uses short names: direct preference optimization (DPO), reinforcement learning with verifiable rewards (RLVR) paired with group relative policy optimization (GRPO), and reinforcement learning from human feedback (RLHF). Choose a step based on how the reward signal enters training, not based on model family alone.

## Options

| Step id | Reward source | Typical data shape | Output |
|---------|---------------|-------------------|--------|
| `rl/nemo_rl/dpo` | Static preference pairs | Prompt with chosen and rejected completions | `checkpoint_megatron` |
| `rl/nemo_rl/rlvr` | Verifiable or programmatic checks | Prompt with answers, tests, or environment metadata | `checkpoint_megatron` |
| `rl/nemo_rl/rlhf` | Learned reward or judge model | Prompts plus a reward model checkpoint | `checkpoint_megatron` |

All three steps consume a warm-start policy in `checkpoint_megatron` format produced by Megatron-style supervised fine tuning (SFT). They do not train a policy from scratch.

## Decision Flow

1. If you only have pairwise preferences and no online reward, use `rl/nemo_rl/dpo`.
2. If reward is deterministic, for example unit tests, answer match, or tool success, use `rl/nemo_rl/rlvr`.
3. If a separate reward model or judge produces scores, use `rl/nemo_rl/rlhf`.
4. For resource-server rewards or NeMo Gym style rewards, use the RLVR or RLHF configuration paths documented in each step `SKILL.md` file and YAML file. Some flows use `config/nemo_gym.yaml`.

## Data Preparation

When preference JSONL still contains Hugging Face placeholders or needs sharding resolution, run the RL prep step upstream. Inspect `data_prep/rl_prep` in the step tree. Read the manifest for your chosen `rl/nemo_rl/...` step for required `consumes` types.

## Sample Commands

```console
$ uv run nemotron steps run rl/nemo_rl/dpo -c tiny
$ uv run nemotron steps run rl/nemo_rl/rlvr -c tiny
$ uv run nemotron steps run rl/nemo_rl/rlhf -c tiny
```

## Success Criteria

- You validate reward design on a small batch before you scale rollout count.
- You track Kullback–Leibler (KL) drift, reward variance, response length, and held-out task metrics. Average reward alone is not sufficient.

## Related Reading

- [Execution through NeMo Run](../../nemo_runspec/nemo-run.md) describes Ray-backed RL workloads on supported executors.
- [Training Libraries](../explanation/training-libraries.md) places NeMo RL in the wider library ecosystem.
