# SKILL: Llama Model Customization Pipeline

## Purpose

Customize Meta Llama models (Llama 3.1, Llama 3.2, Llama 4) for new languages, domains, and use cases. Follows the same 6-stage pipeline as Nemotron customization with Llama-specific model configs, tokenizers, and parallelism settings.

## Pipeline Structure

This family uses the same stage structure as Nemotron. See `src/nemotron/customization_recipes/nemotron/SKILL.md` for full pipeline documentation.

| Stage | Directory | Status |
|-------|-----------|--------|
| 0 - CPT | `stage0_cpt/` | Planned |
| 1 - SFT | `stage1_sft/` | Planned |
| 2 - RL | `stage2_rl/` | Planned |
| 3 - BYOB | `stage3_byob/` | Shared with Nemotron |
| 4 - Eval | `stage4_eval/` | Shared with Nemotron |
| 5 - Quantization | `stage5_quantization/` | Shared with Nemotron |

Stages 3-5 are model-agnostic and reuse the Nemotron implementations. Stages 0-2 require Llama-specific configs.

## Key Differences from Nemotron

| Aspect | Nemotron | Llama |
|--------|----------|-------|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | `meta-llama/Llama-3.1-8B` (or 70B, 405B) |
| Architecture | MoE (Mixture of Experts) | Dense transformer |
| Tokenizer | Nemotron tokenizer | Llama tokenizer (tiktoken-based) |
| Chat template | `nano3.jinja` | Llama chat template |
| Recipe target | `megatron.bridge.recipes.nemotronh.*` | `megatron.bridge.recipes.llama.*` |
| Parallelism (8B) | TP=4, PP=1, CP=2 | TP=1, PP=1 |
| Parallelism (70B) | TP=4, PP=2, CP=2 | TP=8, PP=1 |
| Container | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` | `nvcr.io/nvidia/nemo:25.11` |

## Usage

Once recipe scripts are implemented, usage will follow the same pattern:

```bash
# CPT (Llama-specific configs -- planned)
nemotron customize cpt -c default --run MY-CLUSTER \
  policy.model_name=meta-llama/Llama-3.1-8B

# SFT (Llama-specific configs -- planned)
nemotron customize sft -c default --run MY-CLUSTER \
  policy.model_name=meta-llama/Llama-3.1-8B

# RL (Llama-specific configs -- planned)
nemotron customize rl -c default --run MY-CLUSTER \
  policy.model_name=meta-llama/Llama-3.1-8B

# Eval (shared)
nemotron customize eval -c default --run MY-CLUSTER \
  deployment.checkpoint_path=/results/llama_checkpoint

# Quantize (shared)
python src/nemotron/customization_recipes/nemotron/stage5_quantization/run_quantize.py \
  --config default.yaml \
  model.name_or_path=/results/llama_checkpoint
```

## Prerequisites

- HF_TOKEN with access to Meta Llama models (gated)
- Accept Meta Llama license on HuggingFace
- Same infrastructure requirements as Nemotron (scale with model size)

## Reference

- Full pipeline documentation: `src/nemotron/customization_recipes/nemotron/SKILL.md`
- Per-stage details: `src/nemotron/customization_recipes/nemotron/stage*/SKILL.md`
- Shared data prep: `src/nemotron/customization_recipes/data_prep/SKILL.md`
