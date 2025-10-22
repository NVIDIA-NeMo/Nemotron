# Stage 0: Domain Pretraining

Domain adaptation pretraining on chip design and RTL corpus.

## TODO

### Data Curation (`data_curation.py`)
- [ ] Collect chip design corpus (Verilog, SystemVerilog, VHDL)
- [ ] Include technical documentation and specifications
- [ ] Integrate with Curator for domain-specific filtering
- [ ] Apply deduplication and quality filtering
- [ ] Save as Dataset artifact

### Training (`training.py`)
- [ ] Load base Nemotron model
- [ ] Configure domain adaptation training
- [ ] Implement continued pretraining on chip design data
- [ ] Set up distributed training
- [ ] Save as Checkpoint artifact

### Evaluation (`evaluation.py`)
- [ ] Evaluate on RTL understanding benchmarks
- [ ] Measure code completion accuracy
- [ ] Test domain knowledge retention
- [ ] Save evaluation results as Results artifact

### Stage Orchestrator (`__main__.py`)
- [ ] Implement end-to-end stage execution
- [ ] Handle artifact passing between steps
