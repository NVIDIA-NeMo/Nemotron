# Stage 1: Supervised Fine-tuning

Fine-tuning on RTL code generation pairs.

## TODO

### Data Curation (`data_curation.py`)
- [ ] Collect RTL code generation datasets
- [ ] Create (specification, code) pairs
- [ ] Include diverse RTL design patterns
- [ ] Apply quality filtering and validation
- [ ] Save as Dataset artifact

### Training (`training.py`)
- [ ] Load domain-adapted checkpoint from Stage 0
- [ ] Configure SFT for code generation
- [ ] Implement RTL-specific training objectives
- [ ] Set up code validation during training
- [ ] Save as Checkpoint artifact

### Evaluation (`evaluation.py`)
- [ ] Evaluate on RTL generation benchmarks
- [ ] Measure code correctness and functionality
- [ ] Test synthesis and simulation success rates
- [ ] Save evaluation results as Results artifact

### Stage Orchestrator (`__main__.py`)
- [ ] Implement end-to-end stage execution
- [ ] Load Stage 0 checkpoint automatically
- [ ] Handle artifact passing between steps
