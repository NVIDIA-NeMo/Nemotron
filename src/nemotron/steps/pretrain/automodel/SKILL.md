---
name: nemotron-pretrain-automodel
description: Configure Nemotron pretrain/automodel for pretraining or continued pretraining with NeMo AutoModel. Use for HF-native checkpoints, smaller GPU counts, direct Hugging Face ecosystem workflows, bin/idx input validation, and checkpoint_hf output.
---

# AutoModel Pretrain

Use `pretrain/automodel` when Hugging Face-native checkpoint output and fast iteration matter.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `binidx` data from `prep/pretrain_prep`.
- Produce `checkpoint_hf`.
- Validate data loading and checkpoint output with a short run before scaling token budget.

## Configure

- Set `model.pretrained_model_name_or_path` for continued pretraining from an HF base.
- Set `load_weights=false` only when intentionally training from scratch.
- Keep tokenizer and vocab settings aligned with the bin/idx artifact.
- Use launcher and executor settings from the AutoModel runner for cluster moves.
- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before changing pretraining strategy.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing bin/idx data.

## Config Nuances

- Use `_target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining` for bin/idx blends; do not pass `blend_path`, `split`, or `sequence_length` to raw `IndexedDataset`.
- Set `dataset.paths` and `validation_dataset.paths` to the prep-emitted `blend.json`.
- Set `dataset.seq_length`, `validation_dataset.seq_length`, and downstream model context consistently.
- Keep `dataloader.shuffle: false`; `MegatronPretraining` injects a `batch_sampler`.
- Use `lr_scheduler.lr_decay_style` and `lr_scheduler.min_lr`; avoid `lr_scheduler.warmup_steps` with containers whose `OptimizerParamScheduler` does not accept it.

## Local Files

- Contract: `src/nemotron/steps/pretrain/automodel/step.toml`
- Runner: `src/nemotron/steps/pretrain/automodel/step.py`
- Configs: `src/nemotron/steps/pretrain/automodel/config/default.yaml`, `src/nemotron/steps/pretrain/automodel/config/tiny.yaml`
- Shared runner: `src/nemotron/steps/_runners/automodel.py`

## Guardrails

- Use lower learning rates for continued pretraining than for training from scratch.
- Validate checkpoint save and restore before long runs.
- Keep output format expectations clear for downstream SFT, PEFT, eval, or conversion.
