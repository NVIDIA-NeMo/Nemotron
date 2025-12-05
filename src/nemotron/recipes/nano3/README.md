# Nemotron Nano 3 Training Recipe

Training recipe for Nemotron Nano 3 - a 2B parameter model with pretraining, instruction tuning, and alignment stages.

## Paper

[Nemotron-Mini-4B-Instruct: A Small Yet Powerful Language Model for Conversational AI](https://arxiv.org/pdf/2508.14444)

## Model Overview

- **Size**: 2B parameters
- **Architecture**: Transformer-based
- **Training Stages**: 3-stage pipeline
  - Stage 0: Pretraining
  - Stage 1: SFT (Supervised Fine-Tuning)
  - Stage 2: RL (Reinforcement Learning)

## TODO

- [ ] Implement Stage 0: Pretraining
  - [ ] Data curation pipeline using Curator
  - [ ] Training script using Megatron-Bridge or Automodel
  - [ ] Evaluation benchmarks using Evaluator
- [ ] Implement Stage 1: SFT
  - [ ] Instruction dataset curation
  - [ ] SFT training script
  - [ ] Evaluation on instruction-following benchmarks
- [ ] Implement Stage 2: RL
  - [ ] Preference dataset preparation
  - [ ] DPO/RLHF training using NeMo-RL
  - [ ] Final evaluation suite
- [ ] Add stage orchestrators (`__main__.py`)
- [ ] Document hyperparameters and training configs
- [ ] Add hardware requirements and training time estimates

## Quick Start (Coming Soon)

```bash
# Run complete pipeline
uv run python -m nemotron.recipes.nano3.stage0_pretrain --scale tiny
uv run python -m nemotron.recipes.nano3.stage1_sft --scale tiny
uv run python -m nemotron.recipes.nano3.stage2_rl --scale tiny
```

## Expected Results (To Be Determined)

After completing all stages, the model should achieve:
- TBD: Benchmark scores
- TBD: Training time and resources
