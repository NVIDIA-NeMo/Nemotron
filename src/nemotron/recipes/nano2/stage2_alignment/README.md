# Stage 2: Alignment

Alignment stage using DPO or RLHF to align model behavior with human preferences.

## TODO

### Data Curation (`data_curation.py`)
- [ ] Collect preference datasets (chosen/rejected pairs)
- [ ] Format data for DPO/RLHF training
- [ ] Apply quality filtering
- [ ] Create train/validation splits
- [ ] Save as Dataset artifact

### Training (`training.py`)
- [ ] Load instruction-tuned checkpoint from Stage 1
- [ ] Implement DPO or RLHF training using NeMo-RL
- [ ] Configure alignment hyperparameters
- [ ] Set up reward model (if using RLHF)
- [ ] Save as Checkpoint artifact

### Evaluation (`evaluation.py`)
- [ ] Evaluate on safety and alignment benchmarks
- [ ] Measure helpfulness, harmlessness, honesty
- [ ] Compare with Stage 1 model
- [ ] Save evaluation results as Results artifact

### Stage Orchestrator (`__main__.py`)
- [ ] Implement end-to-end stage execution
- [ ] Load Stage 1 checkpoint automatically
- [ ] Handle artifact passing between steps
