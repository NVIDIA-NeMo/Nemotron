# Embedding Recipes

This directory contains the `nemotron embed` implementation and stage configs.
The [official embedding guide](../../../../docs/nemotron/embed/index.md)
is maintained in `docs/` and covers model selection, prerequisites, execution,
evaluation, and deployment.

- [Nemotron 3.5 multimodal workflow](../../../../docs/nemotron/embed/multimodal.md)
- [Text workflows](../../../../docs/nemotron/embed/README.md)
- [Shared data contracts](../../../../docs/nemotron/embed/data-reference.md)

Choose a profile before running a stage. Keep that profile and `artifact_root`
consistent throughout the workflow. Each `stage*/config/` directory contains
its supported profiles; the [CLI implementation](../../cli/commands/embed/)
defines stage and execution-mode support.
