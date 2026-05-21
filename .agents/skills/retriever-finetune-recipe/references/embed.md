# Embedding Recipe Reference

Load this reference for `nemotron embed ...` work or for questions about first-stage retrieval, bi-encoder training, low Recall@k, missing relevant documents, embedding NIMs, or re-indexing after model changes.

## Contents

- Grounding Paths
- When To Use Embed
- Commands
- Stage Map
- Important Defaults
- Operating Patterns
- NIM Smoke Test
- Tests And Checks

## Grounding Paths

- Recipe README: `src/nemotron/recipes/embed/README.md`
- CLI group: `src/nemotron/cli/commands/embed/_typer_group.py`
- Pipeline command: `src/nemotron/cli/commands/embed/run.py`
- Stage configs: `src/nemotron/recipes/embed/stage*/config/default.yaml`
- Main outputs: `output/embed/`

## When To Use Embed

Use embedding fine-tuning when relevant documents are not retrieved into the candidate set, Recall@k is low, domain terms are poorly matched, or the user needs a better first-stage retrieval model. Embedding changes usually require re-embedding and re-indexing the deployment corpus.

## Commands

Use `uv run` when `nemotron` is not already available.

```bash
uv run nemotron embed info
uv run nemotron embed --help
uv run nemotron embed run -c default -d --from prep --to eval
```

Stage commands:

```bash
uv run nemotron embed sdg -c default corpus_dir=/path/to/docs
uv run nemotron embed prep -c default
uv run nemotron embed finetune -c default
uv run nemotron embed eval -c default
uv run nemotron embed export -c default
uv run nemotron embed deploy -c default
```

Remote execution uses root `env.toml` profiles:

```bash
uv run nemotron embed finetune -c default --run my-cluster
uv run nemotron embed finetune -c default --batch my-cluster
```

## Stage Map

| Stage | Command | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| 0 SDG | `embed sdg` | Text corpus or HF URI | `output/embed/stage0_sdg` | Requires `NVIDIA_API_KEY`; generates synthetic retrieval QA data. |
| 1 prep | `embed prep` | Stage 0 output or existing QA data | `output/embed/stage1_data_prep` | Converts to train/eval data, mines hard negatives, creates BEIR eval data. |
| 2 finetune | `embed finetune` | `train_mined.automodel_unrolled.json` | `output/embed/stage2_finetune/checkpoints` | Automodel contrastive training. |
| 3 eval | `embed eval` | BEIR eval data and checkpoint | `output/embed/stage3_eval/eval_results.json` | Compare base vs fine-tuned on nDCG, Recall, Precision, and MAP. |
| 4 export | `embed export` | Fine-tuned HF checkpoint | `output/embed/stage4_export` | Default config exports ONNX only; set `export_to_trt=true` for TensorRT. |
| 5 deploy | `embed deploy` | ONNX/TensorRT model dir | NIM on `host_port` | Requires Docker/NGC setup and `NGC_API_KEY`. |

The pipeline order is `sdg`, `prep`, `finetune`, `eval`, `export`, `deploy`; `embed run` defaults to `--to eval`.

## Important Defaults

Stage 0:

- Sample corpus: `hf://nvidia/Retrieval-Synthetic-NVDocs-v1@1c0d1856f3fb595b2dda98d4b61061fa6d782d51/sample_corpus/nv_pp_random`
- Output: `./output/embed/stage0_sdg`
- Generation model: `nvidia/nemotron-3-nano-30b-a3b`
- SDG embedding model: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- Useful overrides: `corpus_dir`, `num_pairs`, `sentences_per_chunk`, `file_extensions`, `max_parallel_requests_for_gen`, `preview=true`

Stage 1:

- Input: `./output/embed/stage0_sdg`
- Output: `./output/embed/stage1_data_prep`
- Base model for mining: `nvidia/llama-nemotron-embed-1b-v2`
- Quality threshold: `7.0`
- Split: `train_ratio=0.8`, `val_ratio=0`, `test_ratio=0.2`
- Hard negatives: `hard_negatives_to_mine=5`, `hard_neg_margin=0.95`, `mining_batch_size=128`

Stage 2:

- Base model: `nvidia/llama-nemotron-embed-1b-v2`
- Train data: `./output/embed/stage1_data_prep/train_mined.automodel_unrolled.json`
- Checkpoints: `./output/embed/stage2_finetune/checkpoints`
- Defaults: `num_epochs=3`, `global_batch_size=128`, `local_batch_size=4`, `learning_rate=1.0e-5`, `temperature=0.02`, `train_n_passages=5`
- Prefixes: `query_prefix="query:"`, `passage_prefix="passage:"`
- For real corpora, start with 1-2 epochs unless Stage 3 metrics still improve; the 3 epoch default is for small examples.

Stage 3:

- Eval data: `./output/embed/stage1_data_prep/eval_beir`
- Fine-tuned model: `./output/embed/stage2_finetune/checkpoints/LATEST/model/consolidated`
- Metrics: `k_values=[1,5,10,100]`
- Modes: `eval_base=true`, `eval_finetuned=true`, `eval_nim=false`
- NIM verification: `uv run nemotron embed eval -c default eval_nim=true eval_base=false`

Stage 4:

- Model path: `./output/embed/stage2_finetune/checkpoints/LATEST/model/consolidated`
- ONNX output: `./output/embed/stage4_export/onnx`
- TensorRT output: `./output/embed/stage4_export/tensorrt`
- `attn_implementation=eager` is the export-safe default.

Stage 5:

- NIM image: `nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.10.1`
- Container: `nemotron-embed-nim`
- Default API: `http://localhost:8000/v1/embeddings`

## Operating Patterns

- Skip SDG when the user already has generated QA pairs or wants NVIDIA's pre-generated dataset; start Stage 1 with `sdg_input_path`.
- For production-like chunks, align `sentences_per_chunk`, `passage_max_length`, and eval `max_length` with expected retrieval chunks.
- If increasing sequence length, reduce batch sizes before attempting to recover from OOM.
- Mine at least as many hard negatives as Stage 2 will consume: `hard_negatives_to_mine >= train_n_passages - 1`.
- Preserve `output/embed/stage1_data_prep/eval_beir/` across comparisons so metrics are not shifted by new splits.
- Use `val_ratio=0` only for small datasets where preserving test size matters; use a validation split for larger datasets.
- Inspect existing `output/embed/` artifacts before rerunning a stage. Ask before deleting checkpoints, cached embeddings, or generated data.

## NIM Smoke Test

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["hello"], "model": "nvidia/llama-3.2-nv-embedqa-1b-v2", "input_type": "query"}'
```

## Tests And Checks

```bash
uv run nemotron embed --help
uv run nemotron embed finetune -c default -d
uv run pytest tests/recipes/embed tests/nemo_runspec/test_execution_uv_spec.py -q
```
