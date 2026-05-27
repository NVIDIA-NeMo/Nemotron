---
name: nemotron-optimizer-distillation
description: Configure Nemotron optimize/modelopt/distill for teacher-student distillation with NVIDIA ModelOpt and Megatron-Bridge. Use to train a smaller or optimized student, recover quality after pruning or quantization, select teacher and student checkpoints, or smoke-test distillation plumbing with mock data.
---

# ModelOpt Distillation

Use `optimize/modelopt/distill` when a student checkpoint should learn from a teacher checkpoint.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume teacher and student `checkpoint_hf` paths.
- Optionally consume `binidx` data from `data_prep/pretrain_prep`.
- Produce `checkpoint_megatron`.
- Validate launch and checkpoint writing before using real distillation data for quality evaluation.

## Configure

- Set `args.teacher_hf_path` and `args.student_hf_path`.
- Set `args.data_paths` for real Megatron bin/idx distillation data.
- Set `args.output_dir` away from teacher and student checkpoint roots.
- Use `args.use_mock_data=true` only for launch validation.
- Set `args.hf_export_path` when a Hugging Face export is needed directly.
- Use `extra_args` for newly exposed upstream flags.

## Config Nuances

- Keep `args.teacher_hf_path`, `args.student_hf_path`, and tokenizer expectations explicit; do not infer them from previous optimize outputs.
- Use `args.use_mock_data=true` only for launch validation, then switch to real `args.data_paths` for any quality signal.
- Keep `args.output_dir` distinct from teacher and student checkpoint roots.
- Distill has native upstream W&B flags; keep `wandb_project`, `wandb_entity`, and experiment name wired from the run environment instead of wrapping the process.
- After pruning, point `args.student_hf_path` at the pruned HF output and keep the original BF16 checkpoint as `args.teacher_hf_path` for quality recovery.
- Use the same tokenizer/chat-template assumptions for teacher, student, and distillation data, especially after structural pruning.

## Local Files

- Contract: `src/nemotron/steps/optimize/modelopt/distill/step.toml`
- Runner: `src/nemotron/steps/optimize/modelopt/distill/step.py`
- Configs: `src/nemotron/steps/optimize/modelopt/distill/config/default.yaml`, `src/nemotron/steps/optimize/modelopt/distill/config/tiny.yaml`

## Guardrails

- Use the original BF16 model as teacher when recovering from pruning or quantization.
- Do not treat mock-data runs as quality validation.
- Keep teacher, student, tokenizer, and distillation-data assumptions explicit.
- Choose distillation data that matches the deployment domain.
