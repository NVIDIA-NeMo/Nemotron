# ChipNeMo/ScaleRTL Training Recipe

Training recipe for ChipNeMo/ScaleRTL - a domain-adapted LLM for RTL (Register Transfer Level) code generation with reasoning data and test-time compute.

## Paper

[ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation](https://arxiv.org/abs/2506.05566)

## Model Overview

- **Base Model**: Nemotron (domain-adapted for chip design)
- **Domain**: RTL code generation and hardware design
- **Training Stages**: 3-stage pipeline
  - Stage 0: Domain Pretraining (chip design corpus)
  - Stage 1: Supervised Fine-tuning (RTL code pairs)
  - Stage 2: Reasoning Enhancement (test-time compute)

## TODO

- [ ] Implement Stage 0: Domain Pretraining
  - [ ] Curate chip design corpus (Verilog, SystemVerilog, documentation)
  - [ ] Domain adaptation training using Megatron-Bridge or Automodel
  - [ ] Evaluation on RTL understanding benchmarks
- [ ] Implement Stage 1: Supervised Fine-tuning
  - [ ] RTL code pair dataset preparation
  - [ ] SFT training for code generation
  - [ ] Evaluation on RTL generation tasks
- [ ] Implement Stage 2: Reasoning Enhancement
  - [ ] Reasoning data preparation
  - [ ] Test-time compute integration
  - [ ] Final evaluation on ScaleRTL benchmarks
- [ ] Add stage orchestrators (`__main__.py`)
- [ ] Document domain-specific hyperparameters
- [ ] Add hardware requirements and training time estimates

## Quick Start (Coming Soon)

```bash
# Run complete pipeline
uv run python -m nemotron.recipes.chipnemo.stage0_pretrain --scale tiny
uv run python -m nemotron.recipes.chipnemo.stage1_sft --scale tiny
uv run python -m nemotron.recipes.chipnemo.stage2_reasoning --scale tiny
```

## Expected Results (To Be Determined)

After completing all stages, the model should achieve:
- TBD: RTL code generation accuracy
- TBD: Hardware design benchmark scores
- TBD: Training time and resources
