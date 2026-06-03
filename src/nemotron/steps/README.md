# Nemotron Steps

Use this README as the developer entry point for the Nemotron step library under
`src/nemotron/steps/`. A step is a small, versioned module with a manifest
(`step.toml`), runner (`step.py`), starter configs, tests, and a local README.
The README files explain how to use a step; `step.toml` is the contract that
tools and developers should trust when choosing, wiring, or extending a step.

## Developer Journey

1. Start with the kind of artifact you have: raw text, chat JSONL, packed
   Parquet, bin/idx, a checkpoint, an adapter, or benchmark documents.
2. Use the route table to choose the category README.
3. Follow that category's data and artifact flow until the next consumer is
   obvious.
4. Open the leaf step README for concrete config knobs, smoke commands, and
   guardrails.
5. Read the leaf `step.toml` only when you need the exact manifest: artifact
   types, parameters, strategies, errors, and upstream references.
6. Keep generated configs, local data, checkpoints, and eval outputs under one
   project root so the run can be reproduced or transferred.

## What Is A Step?

Each runnable step lives in a directory such as `sft/megatron_bridge/` or
`data_prep/sft_packing/`.

- `step.toml` defines the step ID, category, consumed and produced artifact
  types, important parameters, decision strategies, expected errors, and
  upstream references.
- `step.py` adapts the generic step runner to the upstream library or recipe.
- `config/default.yaml` shows the production-shaped config.
- `config/tiny.yaml` is for runner and wiring smoke tests, not quality claims.
- `README.md` gives human-facing workflow, guardrails, and local commands.

Read the most specific README first, then read that step's `step.toml`
end-to-end before editing configs or code.

## Try A Step In 10 Minutes

1. Choose the stage from the route table below.
2. Open the stage README, then the target step's `step.toml`.
3. Check `[consumes]` and `[produces]` before adding converters or data prep.
4. Start with `config/tiny.yaml` when it exists.
5. Compile or run a dry run before launching real work:

```bash
uv run nemotron steps run <category>/<step> -c tiny --dry-run
```

Examples:

```bash
uv run nemotron steps run data_prep/sft_packing -c tiny --dry-run
uv run nemotron steps run sft/automodel -c tiny --dry-run
uv run nemotron steps run eval/model_eval -c tiny_chat dry_run=true
```

For remote runs, create or verify the repository-root env profile first. The
default lookup is `env.toml`; generated examples may use `env.lepton.toml`,
`env.slurm.toml`, or `env.dgxcloud.toml` and must be selected with
`NEMOTRON_ENV_FILE`.

## Use `step.toml` For Decisions

The manifest is intentionally more than metadata. Treat these sections as the
developer rulebook:

- `[step]`: stable ID, category, description, tags.
- `[[consumes]]` and `[[produces]]`: artifact wiring contract. Insert a convert
  step only when producer and consumer artifact types disagree.
- `[[parameters]]`: the knobs that developers should understand before changing
  configs.
- `[[strategies]]`: decision rules such as when to choose AutoModel,
  Megatron-Bridge, NeMo-Gym, LoRA, or hosted endpoint evaluation.
- `[[errors]]`: known failure modes and expected recovery path.
- `[reference]`: upstream docs, recipe directories, scripts, and context files.

## Decision Guide

- Use `data_prep/sft_packing` only for consumers of `packed_parquet`, especially
  `sft/megatron_bridge` and `peft/megatron_bridge`. Skip it for AutoModel,
  which reads `training_jsonl` directly.
- Use AutoModel SFT or PEFT when you want HF-native checkpoints, JSONL input,
  smaller GPU counts, or fast iteration. Its manifests explicitly reject packed
  Parquet as input.
- Use Megatron-Bridge SFT, PEFT, or pretraining for large distributed jobs,
  TP/PP/CP/EP parallelism, Nano3/Super3 recipe parity, or packed-sequence
  throughput. Keep `data_prep/sft_packing pack_size`, packed sequence size, and
  training `seq_length` identical.
- Use LoRA / PEFT when memory, checkpoint size, iteration speed, or swappable
  domain adapters matter. Merge only into the exact base checkpoint used for
  adapter training.
- Use pretraining from `binidx` data only after the token budget, tokenizer,
  validation slices, checkpoint cadence, and CPT vs from-scratch plan are
  written down. CPT typically needs lower LR and stricter forgetting checks.
- Choose RL by reward source: DPO for static preference pairs, RLVR for
  deterministic or verifiable rewards, and RLHF for learned judge / GenRM-style
  rewards. All RL paths need a validated `checkpoint_megatron` SFT policy.
- Use `convert/*` only at real artifact boundaries: Megatron to HF, HF to
  Megatron, or LoRA adapter to merged checkpoint. Do not add converters "just in
  case."
- Use `eval/model_eval` before and after quality-changing stages. Hosted smoke
  tests start from `config/tiny_chat.yaml`; Megatron checkpoint eval starts from
  `config/default.yaml` and should point to a concrete `iter_*` checkpoint.
- Use ModelOpt optimization after the source checkpoint has a baseline eval.
  Quantize for numeric format changes, prune for architecture changes, and
  distill to recover quality after compression.

## Developer Workflow

When modifying an existing step:

1. Read the local README and the full `step.toml`.
2. Confirm the artifact contract still matches the runner and configs.
3. Keep shipped `default.yaml` and `tiny.yaml` as catalog examples. Put user or
   experiment overlays under a separate project root, not inside the step
   directory.
4. Update tests when behavior, parameters, artifacts, or error recovery change.
5. Regenerate indexes after adding or renaming steps or patterns:

```bash
python src/nemotron/steps/index.py
```

6. Run focused tests before scaling or opening a PR:

```bash
python -m pytest tests/steps/test_index.py tests/steps/test_patterns.py
```

When adding a new step:

1. Create `src/nemotron/steps/<category>/<step>/`.
2. Add `step.toml` with `id`, `category`, `description`, artifacts,
   parameters, strategies, errors, and references.
3. Add `step.py` and keep shared logic in `_runners/` when another step can
   reuse it.
4. Add `config/default.yaml` and, where practical, `config/tiny.yaml`.
5. Add a local `README.md` with inputs, outputs, workflow, smoke command,
   patterns to cite, and guardrails.
6. Add or update tests, then regenerate `STEPS.md` and `PATTERNS.md`.

## Project Layout For Generated Work

For user-specific runs, keep generated overlays and local inputs under a single
portable project root:

- `<project>/config/` for generated YAML overlays.
- `<project>/data/` for JSONL, Parquet, bin/idx, blends, and sampled fixtures.
- `<project>/checkpoints/` or `<project>/outputs/` for produced artifacts.
- Project-root scripts only when catalog runners cannot serve the request.

Do not write generated overlays into `src/nemotron/steps/<category>/<step>/config/`.
Those checked-in configs are examples and test fixtures.

## Route

| Need | Start With | Primary Artifacts |
| --- | --- | --- |
| Lightweight text curation | `curate/README.md` | `raw_jsonl`, `filtered_jsonl` |
| SFT packing, pretrain bin/idx, RL sharding | `data_prep/README.md` | `training_jsonl`, `packed_parquet`, `binidx` |
| Corpus translation and FAITH scoring | `translate/README.md` | `filtered_jsonl`, `translated_jsonl` |
| Pretraining or continued pretraining | `pretrain/README.md` | `binidx`, `checkpoint_hf`, `checkpoint_megatron` |
| Supervised fine-tuning | `sft/README.md` | `training_jsonl`, `packed_parquet`, checkpoints |
| LoRA or adapter tuning | `peft/README.md` | `checkpoint_lora` |
| DPO, RLVR, or RLHF alignment | `rl/README.md` | prompt or preference JSONL, Megatron checkpoints |
| SFT SDG or RL preference SDG | `sdg/README.md` | `synthetic_jsonl` |
| BYOB benchmark generation or translation | `byob/README.md` | benchmark parquet artifacts |
| Checkpoint format conversion or LoRA merge | `convert/README.md` | `checkpoint_hf`, `checkpoint_megatron`, `checkpoint_lora` |
| Quantization, distillation, pruning | `optimize/README.md` | optimized HF or Megatron checkpoints |
| Evaluation | `eval/README.md` | `eval_results` |
| Execution profiles and Lepton/Ray env setup | Repository-root `env*.toml` plus `NEMOTRON_ENV_FILE` | execution profile |

## Decision Patterns

- Use `patterns/prep-data-is-tokenizer-locked.md` for packed Parquet, bin/idx, tokenizer, chat-template, and sequence-length compatibility.
- Use `patterns/pretrain-token-budget-before-scale.md` before scaling pretraining or continued pretraining.
- Use `patterns/sft-small-dataset-prefer-lora.md` and `patterns/peft-adapter-merge-discipline.md` for PEFT and LoRA decisions.
- Use `patterns/rl-validate-rewards-before-scale.md` before scaling DPO, RLVR, or RLHF jobs.
- Use `patterns/sdg-pipeline-versioning.md` for SFT SDG and RL preference SDG.
- Use `patterns/eval-before-and-after-training.md` and `patterns/byob-benchmark-design.md` for ModelOpt quality decisions.
- Use each step's `step.toml [reference]` section for upstream repos and documentation.

## Guardrails

- Treat data, checkpoints, configs, and eval results as versioned artifacts.
- Keep tokenizer, chat template, sequence length, checkpoint format, and split names aligned across stages.
- Keep env profile files at the repository root for profile lookup, and never blindly regenerate over a user's existing `env.toml`, `env.lepton.toml`, or `env.slurm.toml`.
- Do not use tiny configs as quality evidence; they only prove that plumbing starts.
- Prefer existing runners and configs over inventing a new wrapper.
- Validate the smallest realistic path before scaling to cluster resources.
- Inspect formatted prompts, packed records, loss masks, and a few generations
  before trusting training loss or aggregate benchmark metrics.
