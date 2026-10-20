# Fine-Tune Multimodal Rerankers

Use the `nemotron-3.5` profile with `rerank finetune`, `rerank eval`, and
`rerank deploy` for
[Nemotron 3.5 Rerank 8B BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16).
Prepare data through the embedding workflow described below. Run the rerank
stages explicitly and deploy the consolidated checkpoint directly with vLLM.
Skip Stage 4 export for this model.

## Prerequisites

Complete the [CLI installation](../embed/multimodal.md#prerequisites)
and use a Linux machine with Python 3.12 and NVIDIA GPUs. Start with local
execution and inspect memory use with your image sizes, sequence lengths, and
batch settings before scaling. Full fine-tuning updates all model weights.

For a small save-and-resume verification using the production checkpoint writer,
refer to [Run a Basic Verification](../embed/multimodal.md#run-a-basic-verification).

### Checkpoint Resources

Budget disk space for optimizer state, sharded weights, and the consolidated
model. With `optimizer_backend=flash_adamw` and
`flash_adamw_master_weight_bits=32`, one full 8B checkpoint is estimated to
require about 80 GiB, excluding model caches and data. Reserve additional space
for the next checkpoint while the previous one remains on disk.
`checkpoint.max_recent_checkpoints=1` limits completed checkpoint history;
checkpoints referenced by pointers such as `LATEST` or `LOWEST_VAL` can also remain.
The profiles set `dist_env.timeout_minutes=30` to accommodate checkpoint writes
that finish at different times across ranks.

Allow host memory for checkpoint saving as well as GPU memory for training.
The default PyTorch 2.10 writer can retain about 47 GiB of this optimizer's
tensor data across the training ranks, plus allocator overhead and the rest of
the workload. Check available host memory and monitor the first checkpoint save.
On systems with unified memory, GPU allocations and checkpoint staging share
the same memory pool. Reserve capacity for both, plus runtime overhead.

Before loading weights, training reports available disk space and host RAM,
checkpoint size, and estimated space needed through the next save. For the
pinned model, it stops if the destination cannot hold even one model payload.
Larger checkpoint and memory estimates are advisory and exclude caches, data,
and concurrent jobs. A dry run with `-d` only previews configuration; it does not
perform these resource checks.

## 1. Prepare Training Data

Provide an AutoModel training JSON with a corpus reference, one positive per
unrolled query row, and at least three negatives. To generate this input from
a directory of text and image files or explicit source records, follow the
[multimodal embedding walkthrough](../embed/multimodal.md)
through Stage 1. The embedding model mines the negatives; the reranker then
learns to score the resulting query–document pairs. Keep the complete prepared
corpus with the training JSON so its relative corpus references resolve.

## 2. Fine-Tune the Reranker

Select the prepared embedding run and a separate directory for reranker
artifacts. With `-c nemotron-3.5`, pass the same `artifact_root` to training,
evaluation, and deployment. It selects the checkpoint and evaluation output paths;
input data paths remain explicit.

For a prepared run at `./output/embed/multimodal-first-run`, inspect the
configuration before training:

```bash
export EMBED_RUN=./output/embed/multimodal-first-run
export RERANK_RUN=./output/rerank/multimodal-first-run

uv run nemotron rerank finetune -c nemotron-3.5 -d \
  artifact_root="$RERANK_RUN" \
  train_data_path="$EMBED_RUN/stage1_data_prep/train_mined.automodel_unrolled.json"
```

Confirm the input and output paths, then start training and check the final
checkpoint:

```bash
uv run nemotron rerank finetune -c nemotron-3.5 \
  artifact_root="$RERANK_RUN" \
  train_data_path="$EMBED_RUN/stage1_data_prep/train_mined.automodel_unrolled.json"
test -f "$RERANK_RUN/stage2_finetune/checkpoints/LATEST/model/consolidated/config.json"
```

The default artifact root is `./output/rerank/nemotron-3.5`. Explicit
`checkpoint_dir`, `finetuned_model_path`, `output_dir`, and `model_dir` overrides
take priority over paths derived from `artifact_root`.

The profile pins the model revision and uses a multimodal processor,
bidirectional attention, average pooling, and model score temperature `0.05`.
It preserves the released scoring head and uses native NeMo Automodel
cross-encoder training. The processor uses a 4,096-token limit and a 1,120-pixel
image longest edge. Keep the configured scaled dot product attention (`sdpa`)
backend for image inputs. The `0.05` temperature scales logits during training.
Consolidated checkpoints export the learned head without temperature scaling,
so local evaluation and vLLM return raw relevance logits.

Checkpoints include a consolidated model under `LATEST/model/consolidated`.
A completed training run establishes checkpoint creation. Evaluate the trained
reranker with representative, independently labeled query–document pairs before
using it in a retrieval application.

## 3. Evaluate Both Checkpoints

Compare the base and fine-tuned checkpoints on a fixed evaluation set. For the
prepared embedding run above, select the same retrieval view used for training:

```bash
uv run nemotron rerank eval -c nemotron-3.5 \
  artifact_root="$RERANK_RUN" \
  sdg_input_path="$EMBED_RUN/stage0_sdg/generation_result.json" \
  retrieval_view=mixed
```

Evaluation resolves the selected bundle's BEIR directory and image root. It
retrieves candidates with Nemotron 3.5 Embed, then scores them with each reranker.
The `$RERANK_RUN/stage3_eval` directory contains `eval_results.json`,
`first_stage_results.json`, `base_rankings.json`, and `finetuned_rankings.json`.
Higher reranker scores indicate greater predicted relevance.

For independently labeled evaluation data, omit `sdg_input_path` and
`retrieval_view`. Set `eval_data_path` to a BEIR directory containing
`corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. Set `image_root` to the root
for relative `image_path` values in the corpus. Synthetic holdout metrics describe
the generated query task; use independent labels for broader quality claims.

## 4. Deploy and Check Served Scores

After local evaluation finishes, start the server with Docker. Select an available
host GPU with `gpus`; the example uses device `0`. `CUDA_VISIBLE_DEVICES` in your
shell does not select the container's GPU.

```bash
uv run nemotron rerank deploy -c nemotron-3.5 \
  artifact_root="$RERANK_RUN" \
  gpus='device=0' detach=true
curl --fail http://localhost:8000/health
```

Reuse the saved candidates to evaluate the served model on exactly the same
query–document pairs:

```bash
uv run nemotron rerank eval -c nemotron-3.5 \
  artifact_root="$RERANK_RUN" \
  sdg_input_path="$EMBED_RUN/stage0_sdg/generation_result.json" \
  retrieval_view=mixed \
  first_stage_results_path="$RERANK_RUN/stage3_eval/first_stage_results.json" \
  eval_base=false eval_finetuned=false eval_vllm=true \
  vllm_url=http://localhost:8000 \
  output_dir="$RERANK_RUN/stage3_eval_served"
```

The endpoint evaluation uses `/classify` with structured text and image messages
and requests raw logits. Compare `stage3_eval_served/eval_results.json` and
`stage3_eval_served/vllm_rankings.json` under `$RERANK_RUN` with the local evaluation
outputs. Keep the candidate depth, token limits, and image settings fixed. The profiles use a
4,096-token limit; shorten longer inputs before comparing local and served scores.

Stop the example server when you finish:

```bash
docker stop nemotron-3.5-rerank
```
