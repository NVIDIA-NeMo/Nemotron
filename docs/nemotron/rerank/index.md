# Reranking Fine-Tuning

Adapt a reranker to order the documents returned by a first-stage retriever.
Use this workflow when relevant documents reach the candidate set but rank
below less relevant results. If candidate recall is low, start with
[Embedding Fine-Tuning](../embed/index.md).

## Choose Your Workflow

The two profiles use different data preparation and deployment paths:

| Model | Profile | Input | Start Here |
| --- | --- | --- | --- |
| Nemotron 3.5 Rerank 8B | `nemotron-3.5` | Text, images, or paired documents prepared by the embedding recipe | [Fine-tune multimodal rerankers](multimodal.md) |
| Llama Nemotron Rerank 1B v2 | `default` | Text | [Fine-tune text rerankers](README.md) |

For Nemotron 3.5, prepare data with the embedding workflow, then run reranker
training, evaluation, and deployment explicitly. Use a separate reranker
`artifact_root` and pass the prepared data paths. The consolidated checkpoint
deploys through vLLM without Stage 4 export.

The default Llama workflow includes text generation, preparation, training,
evaluation, ONNX/TensorRT export, and NIM deployment.

## Plan and Verify a Run

Follow the task links relevant to your stage:

- [Plan checkpoint resources](multimodal.md#checkpoint-resources) before full
  8B fine-tuning.
- [Prepare multimodal data](multimodal.md#1-prepare-training-data), retaining
  the complete source bundle and corpus references.
- [Evaluate base and fine-tuned checkpoints](multimodal.md#3-evaluate-both-checkpoints)
  on a fixed candidate set.
- [Deploy and compare served scores](multimodal.md#4-deploy-and-check-served-scores)
  using the saved candidates and raw relevance scores.
- [Inspect shared data contracts](../embed/data-reference.md) and
  [configure multimodal generation](../embed/sdg-reference.md).

```{toctree}
:hidden:

Multimodal Workflow <multimodal>
Text Workflow <README>
```
