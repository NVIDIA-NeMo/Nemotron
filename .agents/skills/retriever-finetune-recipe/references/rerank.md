# Rerank Recipe Reference

Load this reference for `nemotron rerank ...` work or for questions about cross-encoder reranking, second-stage retrieval, top-rank precision, low nDCG with acceptable Recall, ranking NIMs, or reranking evaluation.

## Contents

- Grounding Paths
- When To Use Rerank
- Commands
- Stage Map
- Important Defaults
- Operating Patterns
- NIM Smoke Test
- Tests And Checks

## Grounding Paths

- Recipe README: `src/nemotron/recipes/rerank/README.md`
- CLI group: `src/nemotron/cli/commands/rerank/_typer_group.py`
- Pipeline command: `src/nemotron/cli/commands/rerank/run.py`
- Stage configs: `src/nemotron/recipes/rerank/stage*/config/default.yaml`
- Main outputs: `output/rerank/`

## When To Use Rerank

Use reranker fine-tuning when relevant documents are already in the candidate set but the top ranks are wrong, nDCG@k is low while Recall@k is acceptable, or users say the right answer appears below worse answers. A reranker re-scores query-document pairs; it cannot recover documents that first-stage retrieval did not return.

## Commands

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
| 2 finetune | `rerank finetune` | `train_mined.automodel_unrolled.json` | `output/rerank/stage2_finetune/checkpoints` | Automodel cross-encoder classification training. |
| 3 eval | `rerank eval` | BEIR eval data and checkpoint | `output/rerank/stage3_eval/eval_results.json` | Dense retrieval, rerank top candidates, compare base vs fine-tuned nDCG. |
| 4 export | `rerank export` | Fine-tuned HF checkpoint | `output/rerank/stage4_export` | Default config exports ONNX only; set `export_to_trt=true` for TensorRT. |
| 5 deploy | `rerank deploy` | ONNX/TensorRT model dir | NIM on `host_port` | Requires Docker/NGC setup and `NGC_API_KEY`. |

The pipeline order is `sdg`, `prep`, `finetune`, `eval`, `export`, `deploy`; `rerank run` defaults to `--to eval`.

## Important Defaults

Stage 0:

- Sample corpus: `hf://nvidia/Retrieval-Synthetic-NVDocs-v1@1c0d1856f3fb595b2dda98d4b61061fa6d782d51/sample_corpus/nv_pp_random`
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
- Tokenization: `rerank_max_length=512`, `prompt_template="question:{query} \n \n passage:{passage}"`
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

## Operating Patterns

- Keep Stage 3's first-stage retrieval model and `top_k` fixed across base vs fine-tuned comparisons.
- Track candidate depth carefully. If Recall is low before reranking, tune the embedder or retrieval index first.
- Mine at least as many hard negatives as Stage 2 will consume: `hard_negatives_to_mine >= train_n_passages - 1`.
- Hold the Stage 1 `eval_beir/` split fixed across sweeps so metric changes are not caused by new splits.
- Start learning-rate sweeps near `1e-6`, `3e-6`, and `1e-5`.
- Keep the Stage 2 `prompt_template` and Stage 3 eval `prompt_template` identical.
- Inspect existing `output/rerank/` artifacts before rerunning a stage. Ask before deleting checkpoints, cached embeddings, or generated data.

## NIM Smoke Test

```bash
curl -X POST http://localhost:8000/v1/ranking \
  -H 'Content-Type: application/json' \
  -d '{"model": "nvidia/llama-nemotron-rerank-1b-v2", "query": {"text": "what is AI?"}, "passages": [{"text": "AI is artificial intelligence"}]}'
```

## Tests And Checks

```bash
uv run nemotron rerank --help
uv run nemotron rerank finetune -c default -d
uv run pytest src/nemotron/recipes/rerank/stage2_finetune/tests tests/nemo_runspec/test_execution_uv_spec.py -q
```
