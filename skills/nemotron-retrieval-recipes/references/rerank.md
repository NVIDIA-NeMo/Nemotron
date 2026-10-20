# Rerank Recipe Reference

Load this reference for `nemotron rerank ...` work or for questions about cross-encoder reranking, second-stage retrieval, top-rank precision, low nDCG with acceptable Recall, ranking NIMs, or reranking evaluation.

## Contents

- Grounding Paths
- When To Use Rerank
- Nemotron 3.5 Workflow
- Default Text-Profile Commands
- Stage Map
- Stage Contracts
- Important Defaults
- Operating Patterns
- NIM Smoke Test
- Tests And Checks

## Grounding Paths

These paths describe the execution checkout, not files bundled with this skill.
Planning uses the guidance below without a checkout. Run CLI checks from the
checkout root when preparing to execute.

- CLI group: `src/nemotron/cli/commands/rerank/_typer_group.py`
- Pipeline command: `src/nemotron/cli/commands/rerank/run.py`
- Stage configs: `src/nemotron/recipes/rerank/stage*/config/default.yaml`
- Main outputs: `output/rerank/`

## When To Use Rerank

Use reranker fine-tuning when relevant documents are already in the candidate set but the top ranks are wrong, nDCG@k is low while Recall@k is acceptable, or users say the right answer appears below worse answers. A reranker re-scores query-document pairs; it cannot recover documents that first-stage retrieval did not return.

## Nemotron 3.5 Workflow

For `nvidia/Nemotron-3.5-Rerank-8B-BF16`, use `nemotron-3.5` with `finetune`,
`eval`, and `deploy`. Read the bundled [multimodal data reference](multimodal-data.md)
when preparing inputs. Use the Nemotron 3.5 [embedding](embed.md) SDG and preparation stages for
a text/image directory or explicit source records. Supply the mined, unrolled
training data through `train_data_path` and retain the corpus at the relative
location recorded in that JSON. Select a separate rerank `artifact_root`.
Carry that root through `finetune`, `eval`, and `deploy`. It defaults to
`output/rerank/nemotron-3.5` and derives the Stage 2 checkpoint, Stage 3 output,
and Stage 5 model paths. Input paths remain explicit. Per-stage
`checkpoint_dir`, `finetuned_model_path`, `output_dir`, and `model_dir`
overrides take priority.
Run the stages explicitly; skip Stage 4 and deploy the consolidated model with
vLLM. The default Llama ONNX/NIM path does not apply to this model.

Preview the training handoff from the execution checkout. Replace the example run
directories with the prepared embedding run and a separate reranker output:

```bash
uv run nemotron rerank finetune -c nemotron-3.5 -d \
  artifact_root=./output/rerank/my-run \
  train_data_path=./output/embed/my-run/stage1_data_prep/train_mined.automodel_unrolled.json
```

Remove `-d` only for an authorized training run. Full 8B fine-tuning requires
capacity for optimizer state and checkpoint staging as well as model weights.
With FlashAdamW and 32-bit master weights, budget approximately
80 GiB per full checkpoint and space for the next save before retention runs.
The PyTorch 2.10 writer can retain about 47 GiB of optimizer tensor data across
ranks, in addition to the rest of the workload. Unified-memory systems share
that capacity with GPU allocations.
`checkpoint.max_recent_checkpoints=1` limits recent history; pointer-protected
checkpoints can remain. A configuration preview does not check resource capacity.

After training, check `stage2_finetune/checkpoints/LATEST/model/consolidated/config.json`
under the reranker root, then evaluate using the original data handoff:

```bash
uv run nemotron rerank eval -c nemotron-3.5 \
  artifact_root=./output/rerank/my-run \
  sdg_input_path=./output/embed/my-run/stage0_sdg/generation_result.json \
  retrieval_view=mixed
```

For multimodal evaluation:

- Run locally with `-c nemotron-3.5`; the CLI selects the pinned AutoModel environment.
- Select portable input with `sdg_input_path=<generation_result.json>` and `retrieval_view=mixed`, or provide independent `eval_data_path` and `image_root` paths.
- Keep `top_k`, `k_values`, image settings, and the held-out data fixed. Set `first_stage_results_path` to the local evaluation's `first_stage_results.json` to reuse candidates for endpoint comparisons.
- Read `base_rankings.json` and `finetuned_rankings.json` alongside `eval_results.json`. Scores are raw logits; higher values indicate greater predicted relevance.
- Deploy with `rerank deploy -c nemotron-3.5 artifact_root=<rerank-run> gpus='device=0' detach=true`, replacing `<rerank-run>` with the training run directory and `0` with the selected host GPU. The shell's `CUDA_VISIBLE_DEVICES` does not apply to Docker. Evaluate that server with the same `artifact_root` and input paths, plus `eval_base=false eval_finetuned=false eval_vllm=true vllm_url=http://localhost:8000`. Set a separate `output_dir` to preserve the local evaluation outputs.
- Use `/classify` with query/document messages for multimodal requests. Keep raw logits by setting `use_activation=false` and preserve the checkpoint template with `add_special_tokens=false`.
- Compare `vllm_rankings.json` and metrics against the local checkpoint. Check processor formatting, score-head identity, image resizing, and sequence limits before attributing drift to training.

These profiles provide the execution path; report the actual completed GPU and
endpoint checks for each run rather than inferring successful validation from a
configuration preview.

Optional background is available in the
[reranker guide at the supported revision](https://github.com/NVIDIA-NeMo/Nemotron/blob/e68db1595a9f078a3fb48f8216fa0481d1c862a2/docs/nemotron/rerank/README.md#fine-tune-nemotron-35-rerank-8b-on-image-documents).

## Default Text-Profile Commands

The remaining sections describe the `default` Llama reranker profile. Read them
for text-profile tasks or comparisons with the Nemotron 3.5 workflow.

Use `uv run` when `nemotron` is not already available.

```bash
uv run nemotron rerank info
uv run nemotron rerank --help
uv run nemotron rerank run -c default -d --from prep --to eval
```

Stage commands:

```bash
uv run nemotron rerank sdg -c default corpus_dir=/path/to/docs
uv run nemotron rerank prep -c default
uv run nemotron rerank finetune -c default
uv run nemotron rerank eval -c default
uv run nemotron rerank export -c default
uv run nemotron rerank deploy -c default
```

Remote execution uses root `env.toml` profiles:

```bash
uv run nemotron rerank finetune -c default --run my-cluster
uv run nemotron rerank finetune -c default --batch my-cluster
```

## Stage Map

| Stage | Command | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| 0 SDG | `rerank sdg` | Text corpus or HF URI | `output/rerank/stage0_sdg` | Requires `NVIDIA_API_KEY`; uses the same SDG pipeline shape as embed. |
| 1 prep | `rerank prep` | Stage 0 output or existing QA data | `output/rerank/stage1_prep` | Converts to train/eval data, mines hard negatives, creates BEIR eval data. |
| 2 finetune | `rerank finetune` | `train_mined.automodel_unrolled.json` | `output/rerank/stage2_finetune/checkpoints` | AutoModel cross-encoder classification training. |
| 3 eval | `rerank eval` | BEIR eval data and checkpoint | `output/rerank/stage3_eval/eval_results.json` | Dense retrieval, rerank top candidates, compare base vs fine-tuned nDCG. |
| 4 export | `rerank export` | Fine-tuned HF checkpoint | `output/rerank/stage4_export` | Default config exports ONNX only; set `export_to_trt=true` for TensorRT. |
| 5 deploy | `rerank deploy` | ONNX/TensorRT model dir | NIM on `host_port` | Requires Docker/NGC setup and `NGC_API_KEY`. |

The default text pipeline order is `sdg`, `prep`, `finetune`, `eval`, `export`, `deploy`; `rerank run` defaults to `--to eval`. The stage tables describe that text workflow; use the multimodal instructions above for `nemotron-3.5`.


## Stage Contracts

| Stage | Required Inputs | Creates | Cheapest Check | Expensive Resource | Common Overrides |
| --- | --- | --- | --- | --- | --- |
| 0 SDG | Text corpus or HF URI, `NVIDIA_API_KEY` | `output/rerank/stage0_sdg` | `uv run nemotron rerank run -c default -d --from sdg --to prep` | Provider API calls | `corpus_dir`, `num_pairs`, `sentences_per_chunk`, `file_extensions`, `preview=true` |
| 1 prep | Stage 0 output or `sdg_input_path` | `output/rerank/stage1_prep`, `eval_beir/` | `uv run nemotron rerank prep -c default -d` | Hard-negative mining on larger sets | `sdg_input_path`, `quality_threshold`, `hard_negatives_to_mine`, `mining_batch_size` |
| 2 finetune | `train_mined.automodel_unrolled.json` | `output/rerank/stage2_finetune/checkpoints` | `uv run nemotron rerank finetune -c default -d` | GPU training | `num_epochs`, `learning_rate`, `global_batch_size`, `local_batch_size`, `train_n_passages`, `prompt_template` |
| 3 eval | Fixed `eval_beir/`, checkpoint, first-stage retriever | `output/rerank/stage3_eval/eval_results.json` | `uv run nemotron rerank eval -c default -d` | Retrieval plus rerank inference | `finetuned_model_path`, `eval_data_path`, `retrieval_model`, `top_k`, `k_values`, `eval_nim` |
| 4 export | Fine-tuned checkpoint | `output/rerank/stage4_export/onnx` or `tensorrt` | `uv run nemotron rerank export -c default -d` | Export container/GPU for TensorRT | `model_path`, `export_to_trt`, `attn_implementation`, TensorRT profile settings |
| 5 deploy | ONNX/TensorRT model dir, Docker, NGC access | Rerank NIM on `host_port` | `uv run nemotron rerank deploy -c default -d` | Docker, GPU, NGC image pull | `model_dir`, `use_onnx`, `host_port`, container/image fields |

## Important Defaults

Stage 0:

- Sample corpus: `hf://nvidia/Retrieval-Synthetic-NVDocs-v1@1c0d1856f3fb595b2dda98d4b61061fa6d782d51/sample_corpus/nv_pp_random`; confirm access and license before recommending it, or use the user's `corpus_dir`.
- Output: `./output/rerank/stage0_sdg`
- Generation model: `nvidia/nemotron-3-nano-30b-a3b`
- SDG embedding model: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- Useful overrides: `corpus_dir`, `num_pairs`, `sentences_per_chunk`, `file_extensions`, `max_parallel_requests_for_gen`, `preview=true`

Stage 1:

- Input: `./output/rerank/stage0_sdg`
- Output: `./output/rerank/stage1_prep`
- Base model for hard-negative mining: `nvidia/llama-nemotron-embed-1b-v2`
- Quality threshold: `7.0`
- Split: `train_ratio=0.8`, `val_ratio=0`, `test_ratio=0.2`
- Hard negatives: `hard_negatives_to_mine=5`, `hard_neg_margin=0.95`, `mining_batch_size=128`

Stage 2:

- Base model: `nvidia/llama-nemotron-rerank-1b-v2`
- Train data: `./output/rerank/stage1_prep/train_mined.automodel_unrolled.json`
- Checkpoints: `./output/rerank/stage2_finetune/checkpoints`
- Defaults: `num_epochs=3`, `global_batch_size=128`, `local_batch_size=4`, `learning_rate=3.0e-6`, `train_n_passages=5`
- Optimizer backend: `auto`, using Transformer Engine FusedAdam when available and FlashAdamW otherwise.
- Sequence limit: `rerank_max_length=512`.
- Prompt template: `prompt_template="question:{query} \n \n passage:{passage}"`.
- For real corpora, start with 1-2 epochs unless Stage 3 metrics still improve; the 3 epoch default is for small examples.

Stage 3:

- Eval data: `./output/rerank/stage1_prep/eval_beir`
- Fine-tuned model: `./output/rerank/stage2_finetune/checkpoints/LATEST/model/consolidated`
- First-stage retrieval model: `nvidia/llama-nemotron-embed-1b-v2`
- Candidate depth: `top_k=100`
- Metrics: `k_values=[1,5,10,100]`
- Modes: `eval_base=true`, `eval_finetuned=true`, `eval_nim=false`
- NIM verification: `uv run nemotron rerank eval -c default eval_nim=true eval_base=false`

Stage 4:

- Model path: `./output/rerank/stage2_finetune/checkpoints/LATEST/model/consolidated`
- ONNX output: `./output/rerank/stage4_export/onnx`
- TensorRT output: `./output/rerank/stage4_export/tensorrt`
- `attn_implementation=eager` is the export-safe default.
- TensorRT sequence profile defaults: min 3, opt 256, max 512.

Stage 5:

- NIM image: `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0`
- Container: `nemotron-rerank-nim`
- Default API: `http://localhost:8000/v1/ranking`
- Default deploy runs in the foreground; for service handoff, add `detach=true` plus explicit container name and port overrides when needed.

## Operating Patterns

- Keep Stage 3's first-stage retrieval model and `top_k` fixed across base vs fine-tuned comparisons.
- Track candidate depth carefully. If Recall is low before reranking, tune the embedder or retrieval index first.
- Mine at least as many hard negatives as Stage 2 will consume: `hard_negatives_to_mine >= train_n_passages - 1`.
- Hold the Stage 1 `eval_beir/` split fixed across sweeps so metric changes are not caused by new splits.
- Start learning-rate sweeps near `1e-6`, `3e-6`, and `1e-5`.
- Keep the Stage 2 `prompt_template` and Stage 3 eval `prompt_template` identical.
- Inspect existing `output/rerank/` artifacts before rerunning a stage. Ask before deleting checkpoints, cached embeddings, or generated data.
- For deploy handoff, include the exact deploy command, `detach=true` when background service ownership is expected, container name, host port, smoke test, and container stop or replacement steps.

## Rerank NIM Eval Drift Checklist

When served rerank metrics are worse than checkpoint metrics, find the first boundary where quality changes: checkpoint eval, ONNX export, TensorRT export, then served NIM. Keep the Stage 3 `eval_data_path`, retrieval model, `top_k`, prefixes, `prompt_template`, and `max_length` fixed across comparisons. Verify Stage 4 exports the exact checkpoint path that Stage 3 evaluated, usually `output/rerank/stage2_finetune/checkpoints/LATEST/model/consolidated`. Start with ONNX parity before TensorRT; if ONNX matches but TensorRT drops, inspect `export_to_trt`, `quant_cfg`, TensorRT sequence profiles, and layernorm FP32 settings. For deploy, confirm `model_dir` and `use_onnx` match the intended `stage4_export/onnx` or `stage4_export/tensorrt` artifact, not a stale or base-model mount.

## NIM Smoke Test

```bash
curl -X POST http://localhost:8000/v1/ranking \
  -H 'Content-Type: application/json' \
  -d '{"model": "nvidia/llama-nemotron-rerank-1b-v2", "query": {"text": "what is AI?"}, "passages": [{"text": "AI is artificial intelligence"}]}'
```

## Tests And Checks

Run these only in a prepared source checkout; they are not prerequisites for planning.

```bash
uv run nemotron rerank --help
uv run nemotron rerank finetune -c default -d
uv run pytest src/nemotron/recipes/rerank/stage2_finetune/tests tests/nemo_runspec/test_execution_uv_spec.py -q
```
