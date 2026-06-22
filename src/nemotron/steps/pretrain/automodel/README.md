# AutoModel Pretrain

Use `pretrain/automodel` when Hugging Face-native checkpoint output and fast iteration matter.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `binidx` data from `data_prep/pretrain_prep`.
- Produce `checkpoint_hf`.
- Validate data loading and checkpoint output with a short run before scaling token budget.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for launch validation and `config/default.yaml`
for the production-shaped example. In a project overlay, developers usually
change:

- `model.pretrained_model_name_or_path`: HF base for CPT, or architecture source
  for from-scratch runs.
- `load_weights`: `true` for CPT, `false` for from-scratch pretraining.
- `dataset.paths` and `validation_dataset.paths`: prep-emitted `blend.json`.
- `dataset.seq_length` and `validation_dataset.seq_length`: keep aligned with
  model context and the token-budget calculation.
- Learning-rate schedule, train iterations, checkpoint cadence, and output dirs.

Example shape:

```bash
uv run nemotron steps run pretrain/automodel \
  -c <project>/config/pretrain_automodel.yaml \
  dataset.paths=<prep-output>/blend.json \
  validation_dataset.paths=<prep-output>/blend.json
```

Related patterns:

- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before changing pretraining strategy.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing bin/idx data.

## Config Nuances

- Use `_target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining` for bin/idx blends; do not pass `blend_path`, `split`, or `sequence_length` to raw `IndexedDataset`.
- Set `dataset.paths` and `validation_dataset.paths` to the prep-emitted `blend.json`.
- Set `dataset.seq_length`, `validation_dataset.seq_length`, and downstream model context consistently.
- Keep `dataloader.shuffle: false`; `MegatronPretraining` injects a `batch_sampler`.
- Use `lr_scheduler.lr_decay_style` and `lr_scheduler.min_lr`; avoid `lr_scheduler.warmup_steps` with containers whose `OptimizerParamScheduler` does not accept it.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run pretrain/automodel -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run pretrain/automodel \
  -c <project>/config/pretrain_automodel.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/pretrain/automodel/step.toml`
- Runner: `src/nemotron/steps/pretrain/automodel/step.py`
- Configs: `src/nemotron/steps/pretrain/automodel/config/default.yaml`, `src/nemotron/steps/pretrain/automodel/config/tiny.yaml`
- Shared runner: `src/nemotron/steps/_runners/automodel.py`

## Guardrails

- Use lower learning rates for continued pretraining than for training from scratch.
- Validate checkpoint save and restore before long runs.
- Keep output format expectations clear for downstream SFT, PEFT, eval, or conversion.
