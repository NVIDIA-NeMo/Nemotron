# Reproducibility — Continued Pretraining

The data, pipeline and hyperparameters behind the results in this guidebook.

This is a **settings reference**, not a turnkey replication bundle. It gives
the datasets, the pipeline and every hyperparameter that shapes the result —
enough to re-run the method and reproduce the reported *trends*. It does not
ship dataset revision pins, run or checkpoint identifiers, or the scripts
behind the figures, so individual scores will not reproduce digit-for-digit.

Cluster-specific values — scheduler profiles, mount points, output paths — are
site configuration; substitute your own.

**Contents:** [Quick start](#quick-start) · [Hardware](#hardware) ·
[Data](#data) · [Blending](#blending) · [Pipeline](#pipeline) ·
[Hyperparameters](#hyperparameters) · [Cost](#cost-and-runtime) ·
[Evaluation](#evaluation) · [Adapting to a new language](#adapting-to-a-new-language)

---

## Quick start

One arm, end to end — Punjabi at 1:4 replay (762 iterations, ~6 h on 32 H100):

```bash
# 1. tokenize target + replay corpora
nemotron steps run data_prep/pretrain_prep  -c <target-prep> -b <profile>
nemotron steps run data_prep/pretrain_prep  -c <replay-prep> -b <profile>

# 2. build the ratio mixture, then train
nemotron steps run pretrain/megatron_bridge -c <cpt-config>  -b <profile>

# 3. serve the final checkpoint and evaluate
```

Expect ~30 s per iteration and ~140k tokens/s at the 32-GPU shape below.

---

## Hardware

**Training (all CPT arms)**

| Item | Value |
|---|---|
| Nodes | 4 |
| GPUs per node | 8 × NVIDIA H100 80GB SXM |
| Total GPUs | 32 |
| Intra-node interconnect | NVLink / NVSwitch |
| Inter-node interconnect | InfiniBand |
| Precision | BF16 |
| Storage | shared POSIX filesystem, readable from every node |
| Runtime | `nvcr.io/nvidia/nemo:26.02` |

The 32-GPU layout is fixed by the parallelism: TP 4 × EP 8 = 32 ranks, one per
GPU (PP 1, CP 1, ETP 1).

**Evaluation / serving**

| Item | Value |
|---|---|
| Nodes | 1 |
| GPUs | 8 × H100 80GB SXM |
| Parallelism | TP 4 × EP 8 — must match training |

In training the parallel degrees multiply out to the rank count, so TP 4 ×
EP 8 requires exactly 32 GPUs. Inference is different: the serving stack
overlays expert parallelism on the same 8 GPUs, which is why the evaluation
shape below reads TP 4 × EP 8 on a single node.

**Smaller-scale reproduction (1 node, 8 GPUs).** Use **TP 2 × EP 4 = 8 ranks**
(PP 1, CP 1, ETP 1) and keep `micro_batch_size: 2`. Hold the *token* budget
fixed, not the step count — when `global_batch_size` falls, `train_iters` must
**rise** by the same factor:

| | 32-GPU reference | 8-GPU reproduction |
|---|--:|--:|
| TP × EP | 4 × 8 | 2 × 4 |
| `global_batch_size` | 512 | 128 |
| Tokens per iteration | 4,194,304 | 1,048,576 |
| `train_iters` (Punjabi 1:4) | 762 | 3,048 |

Scale warmup and the WSD decay window by the same factor so the schedule keeps
its shape. Optimizer state and gradient accumulation differ at a smaller batch,
so treat the result as a method check rather than a numeric match.

---

## Data

| Language | Dataset | Subset / split | Target tokens |
|---|---|---|--:|
| Hindi | `ai4bharat/sangraha` | `verified` / `hin` | 5 B, 14.41 B |
| Punjabi | `ai4bharat/sangraha` | `verified` / `pan` | 2.558 B |
| Malayalam | `ai4bharat/sangraha` | `verified` / `mal` | 5.030 B |
| English replay | `nvidia/Nemotron-CC-v2.1` | `High-Quality-DQA` | varies by ratio |

The Vietnamese CPT arms shown in the guidebook figures used an internal mixed
blend whose composition is not published here; those arms are **not
reproducible from this document**. Every other arm is.

Tokenization uses the base model's own tokenizer (no extension in this track),
with `add_bos: false`, `add_eos: true`, 64 train / 1 valid / 1 test shards,
`split_seed: 42`, `int32` indices.

---

## Blending

Target and replay corpora are tokenized separately and mixed by weight at
training time, so one tokenization serves every replay ratio.

Each corpus's share is spread evenly across its own shards:

```
shard_weight = corpus_share / shards_in_that_corpus
```

With 62 target and 62 replay shards:

| Arm | Target share | Replay share | Per-shard weight (target) |
|---|--:|--:|--:|
| 1:1 | 0.500 | 0.500 | 0.008065 |
| 1:4 | 0.800 | 0.200 | 0.012903 |

Sampling is interleaved, so replay is distributed across the run rather than
arriving in a block. Changing the ratio changes only the weights — the shards
are identical across arms. Validation and test are target-only; English
retention is measured with downstream benchmarks.

The mixture is read at run time (shard paths carry a per-prep hash) and wired
into Megatron's `blend_per_split`.

**Extensions of the same scheme:** split the replay share across multiple
domains (code, math, web); train several target languages in one run; vary the
mixture across phases for a curriculum; or weight a curated subset above bulk
crawl. Each is a different weight vector over the same shards — no
re-tokenization needed.

---

## Pipeline

```bash
# tokenize target and replay corpora -> bin/idx shards
nemotron steps run data_prep/pretrain_prep  -c <target-prep-config> -b <profile>
nemotron steps run data_prep/pretrain_prep  -c <replay-prep-config> -b <profile>

# continued pretraining
nemotron steps run pretrain/megatron_bridge -c <cpt-config>         -b <profile>
```

`<profile>` is a scheduler profile from your `env.toml`. The mixture is a
`blend.json` per ratio, referenced as `dataset.per_split_data_args_path`.

For the tokenizer-extension track see its
[reproducibility notes](../../tokenizer_extension/guidebook/REPRODUCIBILITY.md).

---

## Hyperparameters

Shared by every arm:

| Group | Setting | Value |
|---|---|---|
| Base model | | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` |
| | recipe | `megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_pretrain_config` |
| Parallelism | tensor / pipeline / context | 4 / 1 / 1 |
| | expert / expert-tensor | 8 / 1 |
| | sequence parallel, MoE dispatcher | on, `alltoall` |
| | recompute granularity | `selective` |
| Batch | sequence length | 8192 |
| | global / micro batch | 512 / 2 |
| | tokens per iteration | 4,194,304 |
| Optimizer | lr / min lr / weight decay | 3.0e-5 / 3.0e-6 / 0.1 |
| Schedule | decay style | WSD, cosine decay phase |

Per arm — replay notation is **English:target** (1:4 = 20% English):

| Arm | Target | Replay | `train_iters` | warmup | WSD decay |
|---|---|---|--:|--:|--:|
| Punjabi 1:1 | 2.558 B | 50% | 1220 | 24 | 244 |
| Punjabi 1:2 | 2.558 B | 33% | 915 | 18 | 183 |
| Punjabi 1:4 | 2.558 B | 20% | 762 | 15 | 152 |
| Punjabi 1:8 | 2.558 B | 11% | 686 | 14 | 137 |
| Malayalam 1:1 | 5 B | 50% | 2384 | 48 | 477 |
| Malayalam 1:2 | 5 B | 33% | 1788 | 36 | 358 |
| Malayalam 1:4 | 5 B | 20% | 1490 | 30 | 298 |
| Malayalam 1:8 | 5 B | 11% | 1341 | 27 | 268 |
| Hindi 1:1 | 5 B | 50% | 2384 | 48 | 477 |
| Hindi 1:2 | 5 B | 33% | 1788 | 48 | 358 |
| Hindi 1:4 | 5 B | 20% | 1490 | 30 | 298 |
| Hindi 1:8 | 5 B | 11% | 1341 | 30 | 268 |
| Hindi 15B 1:4 (WSD) | 14.41 B | 20% | 4294 | 30 | 859 |
| Hindi 15B 1:4 (cosine) | 14.41 B | 20% | 4294 | 30 | — |

```
total_tokens = target_tokens × (1 + english_share / (1 − english_share))
train_iters  = total_tokens / (global_batch_size × sequence_length)
```

The two `Hindi 15B` arms are named for their nominal 15 B budget; the 4,294
iterations they actually ran are 14.41 B target tokens, and that is the figure
quoted everywhere in the guidebook.

Warmup is ~2% of `train_iters` for most arms. Two Hindi arms depart from that
and are reported as run rather than silently normalised: `Hindi 1:2` used 48
where the 2% rule gives 36, and `Hindi 1:8` used 30 where it gives 27.

Checkpoint cadence: 610 iters for the ~2.5B arms, 1630 for the 5B arms, 200 for
the longest curve. Keep it identical across arms being compared.

---

## Cost and runtime

Measured at the 32 × H100 shape: **~30 s per iteration**, ≈140k tokens/s
including startup. Multiply by 32 for GPU-hours.

| Arm | `train_iters` | Wall-clock | GPU-hours |
|---|--:|--:|--:|
| Punjabi 1:8 | 686 | ~5.7 h | ~185 |
| Punjabi 1:4 | 762 | ~6.4 h | ~205 |
| Punjabi 1:1 | 1220 | ~10.2 h | ~325 |
| 5B target, 1:8 | 1341 | ~11.2 h | ~360 |
| 5B target, 1:4 | 1490 | ~12.4 h | ~400 |
| 5B target, 1:1 | 2384 | ~19.9 h | ~635 |
| 14.41B target, 1:4 | 4294 | ~35.8 h | ~1145 |

A full four-ratio grid at a 5B target budget is roughly **1,900 GPU-hours**.
Data prep adds a few CPU-hours per corpus; evaluation adds ~1 GPU-hour per
checkpoint per benchmark suite.

---

## Evaluation

Checkpoints are Megatron distributed format (`iter_{N:07d}`). Serve with the
same parallelism used for training (TP 4 × EP 8, PP 1, ETP 1) and evaluate over
HTTP.

| Benchmark | Harness | Shots | Measures |
|---|---|---|---|
| MILU (hi / pa / ml / en) | lm-evaluation-harness | 5 | target-language quality |
| VMLU (vi) | lm-evaluation-harness | — | target-language quality |
| MMLU-ProX (en) | NeMo-Evaluator | — | retained capability |
| ARC-Challenge | NeMo-Evaluator | 25 | retained capability |
| HellaSwag | NeMo-Evaluator | — | retained capability |

**Scores are not comparable across harnesses.** The same checkpoint can differ
by tens of points between two harnesses on one benchmark, because prompt
construction, normalisation and scoring all differ. Record which harness
produced every number and compare only within one.

---

## Adapting to a new language

The pipeline is language-agnostic; four things change.

1. **Corpus** — point the prep config at your target dataset. Everything else
   in prep (tokenizer, `add_eos`, shard counts, seed) stays as given.
2. **Token budget** — measure your corpus size first. If it is below ~5B, the
   budget is the corpus and `train_iters` follows from it; above that, choose a
   budget and subset.
3. **Iterations** — recompute per ratio with the formula above. Warmup ≈ 2% of
   `train_iters`, WSD decay ≈ 20%. The 2% figure is the recommendation; two
   arms in the table above ran hand-set warmups.
4. **Benchmarks** — substitute a target-language benchmark for MILU/VMLU. Keep
   the English retention set unchanged so retention stays comparable.

Everything else — parallelism, batch, learning rate, schedule — transfers
unchanged and is the recommended starting point.

---

## What to record

Config name, code revision, target and replay token counts, replay ratio,
`train_iters`, checkpoint path, and evaluation harness.

---

[← Back to the guidebook](./README.md)
