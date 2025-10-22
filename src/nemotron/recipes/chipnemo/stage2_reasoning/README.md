# Stage 2: Reasoning Enhancement

Enhance model with reasoning data and test-time compute strategies.

## TODO

### Data Curation (`data_curation.py`)
- [ ] Create reasoning traces for RTL design
- [ ] Generate step-by-step design solutions
- [ ] Include verification and debugging examples
- [ ] Apply quality filtering
- [ ] Save as Dataset artifact

### Training (`training.py`)
- [ ] Load SFT checkpoint from Stage 1
- [ ] Implement reasoning enhancement training
- [ ] Configure test-time compute strategies
- [ ] Train on reasoning traces
- [ ] Save as Checkpoint artifact

### Evaluation (`evaluation.py`)
- [ ] Evaluate on ScaleRTL benchmarks
- [ ] Measure reasoning quality and correctness
- [ ] Test test-time compute effectiveness
- [ ] Compare with baseline models
- [ ] Save evaluation results as Results artifact

### Stage Orchestrator (`__main__.py`)
- [ ] Implement end-to-end stage execution
- [ ] Load Stage 1 checkpoint automatically
- [ ] Handle artifact passing between steps
- [ ] Enable test-time compute configuration
