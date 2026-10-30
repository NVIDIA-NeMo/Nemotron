# Environment Profiles and Executors

Training steps use the same NeMo Run integration as the rest of Nemotron. Configuration is compiled into a job. Then a *backend* submits that job locally, to Slurm, to Lepton, or to other supported targets that your environment profile defines.

## What to Read First

[Execution through NeMo Run](../../nemo_runspec/nemo-run.md) is the canonical guide for the following topics:

- Environment profile files and how they attach clusters, images, and mounts.
- Attached runs with `--run` compared with detached runs with `--batch`.
- How dotlist overrides merge into YAML training configuration.

## How This Maps to Steps

The command `nemotron steps run` resolves a step identifier to `step.py`, loads the step `config/` directory, and parses the same `env.toml` file that other Nemotron jobs use. Your workflow can select a different environment file if you configure that consistently. The command then hands the built job to the selected backend. You do not need a separate executor YAML file per step beyond what NeMo Run already consumes through the profile.

## Practical Checklist

1. Confirm `uv run nemotron steps show <step_id>` prints the run specification you expect.
2. Run with `--dry-run` once per new profile to catch mount and image issues early.
3. For reinforcement learning (RL) steps that require Ray, align node count, GPU count, and Ray settings with what NeMo RL expects for your cluster. Long-form details stay in NeMo Run and NeMo RL documentation linked from each step `step.toml` reference block.

## Related Reading

- [Getting Started](../getting-started.md)
- [Nemotron CLI Overview](../../nemotron/cli.md)
