# Reranking Recipes

This directory contains the `nemotron rerank` implementation and stage configs.
The [official reranking guide](../../../../docs/nemotron/rerank/index.md)
is maintained in `docs/` and covers model selection, prerequisites, execution,
evaluation, and deployment.

- [Nemotron 3.5 multimodal workflow](../../../../docs/nemotron/rerank/multimodal.md)
- [Text workflows](../../../../docs/nemotron/rerank/README.md)
- [Shared data contracts](../../../../docs/nemotron/embed/data-reference.md)

Choose a profile before running a stage. Keep that profile and `artifact_root`
consistent throughout the workflow. Each `stage*/config/` directory contains
its supported profiles; the [CLI implementation](../../cli/commands/rerank/)
defines stage and execution-mode support.
