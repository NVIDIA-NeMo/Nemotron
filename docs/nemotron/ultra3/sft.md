# Stage 1: SFT

This stage fine-tunes Nemotron 3 Ultra on OpenMathInstruct-2 using [Megatron-Bridge](../nvidia-stack.md#megatron-bridge)'s `finetune()` entry point and the existing Ultra recipe:

```text
megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_sft_openmathinstruct2_packed_config
```

## Reference configuration

| Setting | Value |
|---------|-------|
| CLI command | `nemotron ultra3 sft` |
| Runspec name | `ultra3/sft` |
| Launch | `torchrun` |
| Container | `nvcr.io/nvidia/nemo:26.04.01` |
| HF model id | `nvidia/nemotron-ultra-rl-052726` |
| Default resources | 24 nodes × 8 GPUs |
| Default SFT parallelism | TP=2, PP=6, EP=32, ETP=1, CP=1 |

## Data

No Ray chat-packing pipeline is included for Ultra3 SFT. The Megatron-Bridge recipe configures OpenMathInstruct-2 through `default_openmathinstruct2_config(seq_length=4096, packed_sequence=True)`, including packed sequence setup.

## Run

Preview the compiled job config:

```bash
uv run nemotron ultra3 sft -c tiny --run YOUR-CLUSTER --dry-run
```

Submit an attached Slurm job through NeMo-Run:

```bash
uv run nemotron ultra3 sft --run YOUR-CLUSTER
```

If you have an imported/pretrained checkpoint path, provide it with either `PRETRAINED_CHECKPOINT` or a config override:

```bash
PRETRAINED_CHECKPOINT=/path/to/checkpoint \
uv run nemotron ultra3 sft --run YOUR-CLUSTER

uv run nemotron ultra3 sft --run YOUR-CLUSTER \
  checkpoint.pretrained_checkpoint=/path/to/checkpoint
```

## Slurm wiring

SLURM execution uses the same `nemo_runspec` / NeMo-Run command path as pretraining. No separate `sbatch` script is required in this repo.

## Source

- Recipe: `src/nemotron/recipes/ultra3/stage1_sft/`
- CLI: `src/nemotron/cli/commands/ultra3/sft.py`
- Back to [Ultra3 overview](./README.md)
