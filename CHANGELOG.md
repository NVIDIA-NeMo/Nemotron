# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Python 3.11 is now the minimum supported runtime.
- `curate/nemo_curator` now uses a `min_langid_score` of `0.3` when the key is
  omitted, matching both the shipped configuration and NeMo Curator. Set it
  explicitly to `0.0` only when retaining low-confidence language predictions
  is intentional.
- Curation runtimes now use NeMo Curator 1.3.0 and a fixed NeMo Run revision
  instead of moving development branches.
- Approved policies are now bound to the input corpus, signal implementation,
  thresholds, and language-pack content recorded in run manifests.
- Curation manifests, ledgers, input discovery, and artifact handling now reject
  invalid accounting and stale-output states before they can appear successful.
- Language-pack use now requires an explicit reviewed pack directory; the
  language-specific validation data lives only under test fixtures and is not
  shipped as a user default. Pack resources are normalized to NFC when loaded.
- Release wheels exclude the curation `__local__` workspace so downloaded
  corpora, models, and prior run outputs cannot enter packages.

### Added

#### Core Framework
- Artifact system with Pydantic validation for reproducible pipelines
- Support for artifact lineage tracking and metadata
- Scale factors (tiny/small/medium/full) for fast iteration
- Unix piping support between pipeline steps
- Atomic file writes for artifact metadata
- WandbTracker for optional W&B integration
- CLI generation with typer and OmegaConf

#### Training Recipe Structure
- Recipe package structure at `src/nemotron/recipes/`
- Placeholder structure for Nemotron Nano 2 recipe:
  - Stage 0: Pretraining
  - Stage 1: Instruction Tuning
  - Stage 2: Alignment
- Placeholder structure for ChipNeMo/ScaleRTL recipe:
  - Stage 0: Domain Pretraining
  - Stage 1: Supervised Fine-tuning
  - Stage 2: Reasoning Enhancement

#### Examples
- Training pipeline pattern examples in `examples/training/`:
  - `hello.py` - Minimal pipeline step example
  - `tutorial_data_prep.py` - Data preparation patterns
  - `tutorial_training.py` - Training loop patterns
  - `tutorial_evaluation.py` - Evaluation patterns

#### Documentation
- Comprehensive README with:
  - NVIDIA AI stack integration details (Curator, Megatron-Bridge, Automodel, NeMo-RL, Evaluator)
  - Recipe overview with paper links
  - Usage examples and quick start
- Recipe-specific READMEs with detailed TODO lists
- Contribution guidelines inspired by NVIDIA-NeMo/Megatron-Bridge

#### Testing
- Unit tests for Artifact system
- Test suite with pytest
- 100% test coverage for core framework

### Technical Details

**Dependencies:**
- pydantic >= 2.0.0
- typer >= 0.12.0
- omegaconf >= 2.3.0
- rich >= 13.0.0
- wandb (optional)

**Python Support:** 3.11+

**Package Management:** uv

---

## Future Releases

### Planned for 0.1.0 (First Release)

#### Nemotron Nano 2 Recipe
- [ ] Stage 0 implementation (pretraining)
- [ ] Stage 1 implementation (instruction tuning)
- [ ] Stage 2 implementation (alignment)
- [ ] Benchmark results and validation
- [ ] Hardware requirements documentation

#### ChipNeMo/ScaleRTL Recipe
- [ ] Stage 0 implementation (domain pretraining)
- [ ] Stage 1 implementation (SFT)
- [ ] Stage 2 implementation (reasoning enhancement)
- [ ] RTL generation benchmarks
- [ ] Integration with test-time compute

#### Infrastructure
- [ ] CI/CD pipeline
- [ ] Automated testing
- [ ] Docker containers for reproducibility
- [ ] Pre-commit hooks

#### Documentation
- [ ] Usage guides in `docs/usage/`
- [ ] Deployment guides in `docs/deployment/`
- [ ] Video tutorials
- [ ] API documentation

---

## Version History

This is the initial version of the project. Version history will be updated as releases are published.

---

**Note**: This project is in active development. Recipe implementations are coming soon. Contributions are welcome!
