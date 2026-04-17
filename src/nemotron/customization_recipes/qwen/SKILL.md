# SKILL: Qwen Model Customization Pipeline

## Purpose

Customize Alibaba Qwen models (Qwen2, Qwen2.5, Qwen3) for new languages, domains, and use cases. Follows the same 6-stage pipeline as Nemotron customization with Qwen-specific model configs, tokenizers, and parallelism settings.

## Pipeline Structure

This family uses the same stage structure as Nemotron. See `src/nemotron/customization_recipes/nemotron/SKILL.md` for full pipeline documentation.

| Stage | Directory | Status |
|-------|-----------|--------|
| 1 - CPT | `stage1_cpt/` | Planned |
| 2 - SFT | `stage2_sft/` | Planned |
| 3 - RL | `stage3_rl/` | Planned |
| 4 - BYOB | `stage4_byob/` | Shared with Nemotron |
| 5 - Eval | `stage5_eval/` | Shared with Nemotron |
| 6 - Quantization | `stage6_quantization/` | Shared with Nemotron |

Stages 4-6 are model-agnostic and reuse the Nemotron implementations. Stages 1-3 require Qwen-specific configs.

## Key Differences from Nemotron

| Aspect | Nemotron | Qwen |
|--------|----------|------|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | `Qwen/Qwen2.5-7B` (or 14B, 32B, 72B) |
| Architecture | MoE (Mixture of Experts) | Dense or MoE (Qwen2.5-MoE) |
| Tokenizer | Nemotron tokenizer | Qwen tokenizer (BPE, 151K vocab) |
| Chat template | `nano3.jinja` | Qwen chat template (`<|im_start|>` format) |
| Recipe target | `megatron.bridge.recipes.nemotronh.*` | `megatron.bridge.recipes.qwen.*` |
| Parallelism (7B) | TP=4, PP=1, CP=2 | TP=1, PP=1 |
| Parallelism (72B) | TP=4, PP=2, CP=2 | TP=8, PP=1 |
| Container | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` | `nvcr.io/nvidia/nemo:25.11` |
| CJK support | Limited | Strong (pre-trained on CJK data) |

## Usage

Once recipe scripts are implemented, usage will follow the same pattern:

```bash
# CPT (Qwen-specific configs -- planned)
nemotron customize cpt -c default --run MY-CLUSTER \
  policy.model_name=Qwen/Qwen2.5-7B

# SFT (Qwen-specific configs -- planned)
nemotron customize sft -c default --run MY-CLUSTER \
  policy.model_name=Qwen/Qwen2.5-7B

# RL (Qwen-specific configs -- planned)
nemotron customize rl -c default --run MY-CLUSTER \
  policy.model_name=Qwen/Qwen2.5-7B

# Eval (shared)
nemotron customize eval -c default --run MY-CLUSTER \
  deployment.checkpoint_path=/results/qwen_checkpoint

# Quantize (shared)
python src/nemotron/customization_recipes/nemotron/stage6_quantization/run_quantize.py \
  --config default.yaml \
  model.name_or_path=/results/qwen_checkpoint
```

## Notes

- Qwen models have strong CJK (Chinese, Japanese, Korean) coverage. For CJK language adaptation, CPT data volume can be reduced compared to Nemotron/Llama.
- Qwen2.5-MoE variants use the same MoE infrastructure as Nemotron -- parallelism settings (expert_model_parallel_size) apply similarly.
- Qwen tokenizer vocabulary (151K tokens) is larger than Llama -- this affects training throughput and memory.

## Reference

- Full pipeline documentation: `src/nemotron/customization_recipes/nemotron/SKILL.md`
- Per-stage details: `src/nemotron/customization_recipes/nemotron/stage*/SKILL.md`
- Shared data prep: `src/nemotron/customization_recipes/data_prep/SKILL.md`
