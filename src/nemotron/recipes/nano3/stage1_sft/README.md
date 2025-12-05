# Stage 1: Instruction Tuning

Supervised fine-tuning stage to teach the model to follow instructions.

## TODO

### Data Curation (`data_curation.py`)
- [ ] Collect instruction-following datasets
- [ ] Format data for instruction tuning (prompt/response pairs)
- [ ] Apply quality filtering and deduplication
- [ ] Create train/validation splits
- [ ] Save as Dataset artifact

### Training (`training.py`)
- [ ] Load pretrained checkpoint from Stage 0
- [ ] Configure SFT training hyperparameters
- [ ] Implement instruction tuning training loop
- [ ] Set up evaluation during training
- [ ] Save as Checkpoint artifact

### Evaluation (`evaluation.py`)
- [ ] Evaluate on instruction-following benchmarks
- [ ] Measure instruction adherence and quality
- [ ] Compare with base model performance
- [ ] Save evaluation results as Results artifact

### Stage Orchestrator (`__main__.py`)
- [ ] Implement end-to-end stage execution
- [ ] Load Stage 0 checkpoint automatically
- [ ] Handle artifact passing between steps
