# Stage 0 Recipe Summary — Pretraining and Long Context

This file maps the paper’s base-model training story to the released recipe under:

- `src/nemotron/recipes/super3/stage0_pretrain/`

---

## What stage 0 covers

The paper’s pretraining program appears here as four released configs:

| Paper phase | Released config |
|---|---|
| Phase 1 (20T) | `config/phase1.yaml` |
| Phase 2 (5T) | `config/phase2.yaml` |
| Long-context stage 1 (34B) | `config/long_context_1m.yaml` |
| Long-context stage 2 (17B) | `config/long_context_mixed.yaml` |

This is one of the cleanest paper → repo mappings in the whole skill.

---

## Main source files

| Path | Role |
|---|---|
| `src/nemotron/recipes/super3/stage0_pretrain/README.md` | human overview and CLI examples |
| `.../data_prep.py` | Ray tokenization pipeline |
| `.../train.py` | Megatron-Bridge training entrypoint |
| `.../config/phase1.yaml` | phase 1 schedule and artifact wiring |
| `.../config/phase2.yaml` | phase 2 resume and schedule continuation |
| `.../config/long_context_1m.yaml` | 1M continuation config |
| `.../config/long_context_mixed.yaml` | mixed 1M/4K continuation config |

---

## Data preparation path

The data-prep script uses a 3-stage Ray pipeline:

```text
PlanStage → DownloadStage → BinIdxTokenizationStage
```

The output is Megatron-compatible tokenized data plus a split manifest:

```text
output/super3/stage0_pretrain/
  train/*.bin, *.idx
  valid/*.bin, *.idx
  test/*.bin, *.idx
  blend.json
```

That `blend.json` is the object passed into Megatron-Bridge through `per_split_data_args_path`.

---

## Training entrypoint

The training entrypoint is:

- `src/nemotron/recipes/super3/stage0_pretrain/train.py`

It is a runspec-style Python script that:

- loads a YAML config,
- resolves artifacts,
- imports the recipe target dynamically,
- merges OmegaConf overrides,
- calls Megatron-Bridge `pretrain()`.

The default recipe target is:

- `megatron.bridge.recipes.nemotronh.nemotron_3_super.nemotron_3_super_pretrain_config`

---

## Phase 1 details

From `config/phase1.yaml`:

| Setting | Value |
|---|---|
| Data artifact | `super3-pretrain-data-phase1:latest` |
| Container | `nvcr.io/nvidian/nemo:26.02.super.rc4` |
| Train iterations | 993404 |
| Scheduler | WSD |
| WSD decay style | `minus_sqrt` |
| Warmup iters | 7949 |
| Decay iters | 992411 |
| WSD decay iters | 198682 |
| Save path | `/nemo_run/super3-pretrain-model-phase1` |

The important implementation detail is that phase 1 already encodes the **full 25T schedule**, so phase 2 can resume with the correct scheduler state.

---

## Phase 2 details

From `config/phase2.yaml`:

| Setting | Value |
|---|---|
| Data artifact | `super3-pretrain-data-phase2:latest` |
| Train iterations | 993404 |
| Scheduler | same full 25T WSD schedule |
| Save path | `/nemo_run/super3-pretrain-model-phase2` |
| Load path | `/nemo_run/super3-pretrain-model-phase1` |

This matches the paper’s description of phase 2 as a continuation of the same base run with a refined data mix.

---

## Long-context stage 1 details

From `config/long_context_1m.yaml`:

| Setting | Value |
|---|---|
| Data artifact | `super3-pretrain-data-long-context:latest` |
| Sequence length | 1,048,576 |
| Global batch size | 16 |
| Context parallelism | 64 |
| Tensor parallelism | 2 |
| Expert parallelism | 64 |
| Train iterations | 2027 |
| Scheduler | constant |
| Learning rate | 4.5e-6 |
| Save path | `/nemo_run/super3-pretrain-model-lc1` |
| Load path | `/nemo_run/super3-pretrain-model-phase2` |

This is the clearest repo-level manifestation of the paper’s 1M-context continuation stage.

---

## Long-context stage 2 details

From `config/long_context_mixed.yaml`:

| Setting | Value |
|---|---|
| Sequence length | 1,048,576 |
| Global batch size | 16 |
| Context parallelism | 64 |
| Tensor parallelism | 2 |
| Expert parallelism | 64 |
| Train iterations | 1014 |
| Scheduler | constant |
| Learning rate | 4.5e-6 |
| Save path | `/nemo_run/super3-pretrain-model-lc2` |
| Load path | `/nemo_run/super3-pretrain-model-lc1` |

The config also contains an important caveat: alternating 1M and 4K sequences may require custom dataloader support because Megatron-Bridge does not natively alternate sequence lengths inside one run.

---

## Commands exposed by the repo

```bash
uv run nemotron super3 data prep pretrain -c phase1 --run <profile>
uv run nemotron super3 data prep pretrain -c phase2 --run <profile>
uv run nemotron super3 data prep pretrain -c long_context --run <profile>

uv run nemotron super3 pretrain -c phase1 --run <profile>
uv run nemotron super3 pretrain -c phase2 --run <profile>
uv run nemotron super3 pretrain -c long_context_1m --run <profile>
uv run nemotron super3 pretrain -c long_context_mixed --run <profile>
```

This is the best answer when the user asks for a runnable order of operations.

---

## Artifact flow

```text
raw text
  → data_prep.py (phase-specific blend)
  → tokenized bin/idx + blend.json
  → phase1 checkpoint
  → phase2 checkpoint
  → lc1 checkpoint
  → lc2 checkpoint / base model artifact for SFT
```

---

## Paper-vs-recipe caveats

1. **Open data is partial.** The README notes the open data likely covers only ~8–10T of the internal 25T blend.
2. **LC mixed is conceptually faithful but operationally approximate.** The config itself warns about alternating-sequence support.
3. **Checkpoint merging is a paper concept, not a standalone recipe stage.**
4. **Training precision details from the paper are not fully spelled out in the YAML.** For the NVFP4 rationale, use `../paper/pretraining.md`.

---

## Best next file

- `stage1_sft.md` if the user wants the next pipeline stage.
- `../paper/pretraining.md` if the user wants the research rationale instead of the runnable config.
