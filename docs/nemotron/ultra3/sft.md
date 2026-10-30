# Stage 1: SFT

This stage fine-tunes Nemotron 3 Ultra with [Megatron-Bridge](../nvidia-stack.md#megatron-bridge)'s `finetune()` entry point and the existing Ultra recipe:

```text
megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_sft_openmathinstruct2_packed_config
```

The default Nemotron-side config overrides the Megatron-Bridge OpenMath starter into a paper-style packed-Parquet SFT run that consumes an externally prepared packed-Parquet artifact.

## Reference configuration

| Setting | Value |
|---------|-------|
| CLI command | `nemotron ultra3 sft` |
| Runspec name | `ultra3/sft` |
| Launch | `torchrun` |
| Container | `~/.cache/nemotron/containers/ultra3-sft.sqsh` (built from `nvcr.io/nvidia/nemo:26.04.01`) |
| HF model id | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` |
| Resources | 384 nodes × 8 GPUs |
| Parallelism | TP=2, PP=6, EP=32, ETP=1, CP=8 |
| Sequence length | 294,912 packed tokens |
| Global batch | 64 |

## Container build

Ultra3 ships no released training container — the stage owns a `Dockerfile` that
builds the `nemotron_3_ultra` Megatron-Bridge branch on `nvcr.io/nvidia/nemo:26.04.01`.
Build it before training.

On Slurm (our CLI builds only on Slurm):

```bash
uv run nemotron kit slurm build <profile> --recipe ultra3 --stage sft
```

Or build the Dockerfile directly with Docker on any host:

```bash
docker build -t ultra3-sft src/nemotron/recipes/ultra3/stage1_sft
```

The train configs/runspec expect the squashfs at
`${build_cache_dir:-~/.cache/nemotron}/containers/ultra3-sft.sqsh`.

## Data

The default config consumes an externally prepared packed-Parquet artifact:

```yaml
run:
  data: ultra3-sft-data:latest

dataset:
  ultra3_packed_sft_dir: ${art:data,path}
  seq_length: 294912
  packed_sequence_specs:
    packed_sequence_size: 294912
```

The corresponding data-prep stage lives under `src/nemotron/recipes/ultra3/stage1_sft/data_prep.py` and packs the open Ultra/Super-style SFT blend into `train/` and optional `valid/` Parquet shards. The default requires the data artifact to be packed at 294,912 tokens to match the paper.

## Run

Preview the compiled default job config:

```bash
uv run nemotron ultra3 sft -c tiny --run YOUR-CLUSTER --dry-run
uv run nemotron ultra3 sft --run YOUR-CLUSTER --dry-run
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

SLURM execution uses the same `nemo_runspec` / NeMo-Run command path as pretraining.

## Direct execution (Megatron-Bridge)

This recipe wraps the Megatron-Bridge Ultra SFT recipe
(`megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_sft_openmathinstruct2_packed_config`).
To run it directly outside this CLI, use the example scripts in the
[Megatron-Bridge `nemotron_3_ultra` branch](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_ultra/examples/models/nemotron/nemotron_3/ultra):

```bash
# Clone the repository and checkout the nemotron_3_ultra branch
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout nemotron_3_ultra

# Pre-pack OpenMath data, then run full SFT (TP=2 PP=6 EP=32)
sbatch examples/models/nemotron/nemotron_3/ultra/pack_data_job.sh
sbatch examples/models/nemotron/nemotron_3/ultra/slurm_sft.sh
```

See the
[Ultra examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron_3_ultra/examples/models/nemotron/nemotron_3/ultra/README.md)
for PEFT, packing, conversion, and the full set of Slurm scripts.

## Source

- Recipe: `src/nemotron/recipes/ultra3/stage1_sft/`
- CLI: `src/nemotron/cli/commands/ultra3/sft.py`
- Megatron-Bridge: [Ultra examples (`nemotron_3_ultra` branch)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_ultra/examples/models/nemotron/nemotron_3/ultra)
- Back to [Ultra3 overview](./README.md)
