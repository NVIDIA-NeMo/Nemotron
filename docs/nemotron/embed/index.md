# Embedding Fine-Tuning

Adapt an embedding model to retrieve relevant documents from your domain.
Choose a model below, prepare its training data, and compare the base and
fine-tuned checkpoints on a fixed evaluation set.

Start here when relevant documents are missing from the candidate set. If
candidate recall is already acceptable but the top results are ordered poorly,
refer to [Reranking Fine-Tuning](../rerank/index.md).

## Choose Your Workflow

The profile determines the data format, training environment, and deployment path.

| Model | Profile | Input | Start Here |
| --- | --- | --- | --- |
| Nemotron 3.5 Embed 1B | `nemotron-3.5` | Text, images, or paired documents | [Fine-tune multimodal embeddings](multimodal.md) |
| Nemotron 3 Embed 1B | `default` | Text | [Fine-tune text embeddings](README.md) |
| Llama Nemotron Embed 1B v2 | `llama` | Text | [Text profile selection](README.md#choose-a-model-profile) |

Keep the selected profile and `artifact_root` consistent across stages.
Nemotron 3.5 preparation, training, and checkpoint evaluation run locally.
Its consolidated checkpoint deploys directly through vLLM; it has no Stage 4
export profile. The text guide documents its profiles' execution and export paths.

## Prepare Data and Configure Generation

Use these references when you need to change the walkthrough's inputs or defaults:

- [Data contracts](data-reference.md): directory input, explicit source records,
  retrieval views, portable bundles, and artifact handoffs.
- [Multimodal synthetic data generation](sdg-reference.md): generator and judge
  models, credentials, context planning, and execution limits.

## Evaluate and Use the Result

Complete the base-versus-fine-tuned comparison before serving a checkpoint.
Synthetic holdout labels measure the generated task; independently authored
queries and labels support broader quality claims.

- [Evaluate independent data](multimodal.md#evaluate-independent-data).
- [Serve a multimodal checkpoint](multimodal.md#serve-the-consolidated-checkpoint).
- [Verify the workflow](multimodal.md#run-a-basic-verification) with a small fixture.
- [Troubleshoot a multimodal run](multimodal.md#troubleshoot-a-run).
- [Train a reranker from prepared embedding data](../rerank/multimodal.md).

```{toctree}
:hidden:

Multimodal Workflow <multimodal>
Text Workflows <README>
Data Reference <data-reference>
SDG Reference <sdg-reference>
```
