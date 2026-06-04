# Stage 0: Pretraining

This stage trains Nemotron 3 Ultra using [Megatron-Bridge](../nvidia-stack.md#megatron-bridge)'s `pretrain()` entry point and the existing Ultra recipe:

```text
megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_pretrain_config
```

## Reference configuration

| Setting | Value |
|---------|-------|
| CLI command | `nemotron ultra3 pretrain` |
| Runspec name | `ultra3/pretrain` |
| Launch | `torchrun` |
| Container | `nvcr.io/nvidia/nemo:26.04.01` |
| HF model id | `nvidia/nemotron-ultra-rl-052726` |
| Default resources | 48 nodes × 8 GPUs |
| Default parallelism | TP=2, PP=12, EP=32, ETP=1, CP=1 |

## Data

Pretraining consumes the tokenized output of `nemotron ultra3 data prep pretrain` (a Ray pipeline
that tokenizes the open pretrain mixture to Megatron `bin/idx`), the same flow super3/nano3 use:

```bash
uv run nemotron ultra3 data prep pretrain -c phase1 --run YOUR-RAY-CLUSTER   # diversity (~15T)
uv run nemotron ultra3 data prep pretrain -c phase2 --run YOUR-RAY-CLUSTER   # quality (~5T)
```

The blends (`config/data_prep/data_blend_raw_{phase1,phase2}.json`) encode the two-phase Figure 4
mixture; per-`category` weights match the report's phase shares. New-for-Ultra data uses the
released repos `Nemotron-Pretraining-Specialized-v1.2` (Multiple-Choice / Generative / Fact-Seeking /
Moral-Scenarios subsets) and `Nemotron-Pretraining-Legal-v1` (Case-Law-Summary). Categories with no
open data — `code`, `nemotron-cc-code`, `academic`, `crawl++` — are listed in `_missing_categories`
to backfill (notably `Nemotron-Pretraining-Code-v3` is released as a repo manifest, not tokenized
text). Training then consumes the result via `data.per_split_data_args_path: ${art:data,path}/blend.json`.

## Run

```bash
uv run nemotron ultra3 pretrain -c tiny --run YOUR-CLUSTER --dry-run
uv run nemotron ultra3 pretrain --run YOUR-CLUSTER
```

## Slurm wiring

SLURM execution comes from the shared `nemo_runspec` CLI machinery. `src/nemotron/cli/commands/ultra3/pretrain.py` parses the PEP-723 runspec header in `train.py`, merges YAML config with the `env.toml` profile, packages the script/config with `SelfContainedPackager`, and submits through NeMo-Run's Slurm executor for `--run` / `--batch`.

No separate `sbatch` script is required in this repo.

## Long-context phase (not included)

The report's Long-Context (LC) phase extends Ultra to 1M-token context, but **this recipe does not
include it because the data is not open-source.** Per §2.5 the LC blend is 46% long-context data +
54% Phase 2 data, where the long-context half is (a) document-QA data reused from Nemotron 3 Super &
Nano and (b) synthetic long-context SFT-style data — neither is part of the Ultra dataset release.
No RULER-style data is used.

To replicate it with your own long-context corpus, run a short continued-pretraining (CPT) phase from
the Phase 2 checkpoint:

- Blend ~46% long-context data + ~54% Phase 2 data (reuse `data_blend_raw_phase2.json`).
- Long-context data = long documents with document-QA pairs + synthetic long-context SFT-style
  samples (multi-document reasoning / retrieval / synthesis at 128K–1M tokens).
- Constant LR `2.5e-6`; ~33B tokens total; 92% of iterations at 1M (1,048,576) context, 8% at 4K
  (math/code SFT-style data only).
- Parallelism (GB200): CP=32, TP=8, EP=128, PP=2.
- Do not include RULER-style data.

## Source

- Recipe: `src/nemotron/recipes/ultra3/stage0_pretrain/`
- CLI: `src/nemotron/cli/commands/ultra3/pretrain.py`
- Back to [Ultra3 overview](./README.md)
