# Retrieval Recipe Pitfalls

Load this file when a recipe command fails, metrics look wrong, or the user asks for debugging help.

## Setup And CLI

- CLI help or dry-run fails before reaching `embed` or `rerank` with a missing optional dependency: run the repo's documented sync path, usually `uv sync --all-extras`. If the error names Data Designer, the smaller recovery may be `uv sync --extra data-sdg`.
- `uv run` rebuilds or installs packages unexpectedly: report that the environment is being prepared, then continue with help/dry-run before launching work.
- In an already synced checkout, use `uv run --no-sync ...` for help and dry-runs when you want to avoid dependency changes; if dependencies are missing, fall back to the documented sync path.
- CUDA symbol, `nvJitLink`, or library mismatch errors: clear inherited CUDA library paths with `LD_LIBRARY_PATH=""` for the command, then rerun the cheapest failing validation.
- Unknown override field: inspect the stage config model or `uv run nemotron <family> <stage> --help`; Pydantic configs usually reject extra fields.
- Hugging Face `429 Too Many Requests` or gated-model access errors: set `HF_TOKEN`, run `huggingface-cli login`, or reduce parallel work before retrying.

## Stage 0 SDG

- Missing `NVIDIA_API_KEY`: Stage 0 requires it. Ask the user to configure the environment, but do not ask them to paste the key.
- API rate limits or flaky generation: reduce `max_parallel_requests_for_gen`, lower `batch_size`, or run a smaller pilot with fewer files.
- No or low-quality generated QA: inspect a sample of generated JSON before lowering `quality_threshold`; improve corpus quality, chunking, or SDG model settings first.
- Large corpus takes too long: use `num_files`, batch index ranges, or a representative pilot corpus before full generation.

## Stage 1 Prep

- GPU OOM during hard-negative mining: reduce `mining_batch_size`, sequence lengths, or visible GPUs workload.
- Few valid training rows: check Stage 0 quality scores and Stage 1 `quality_threshold`; confirm SDG output path points to the intended family output directory.
- Train/eval comparisons shift unexpectedly: preserve the same `eval_beir/` split across runs.
- Hard negatives are insufficient: ensure `hard_negatives_to_mine >= train_n_passages - 1`.

## Stage 2 Finetune

- OOM: reduce `local_batch_size`, `global_batch_size`, sequence length, or `train_n_passages`.
- NaN or unstable loss: reduce learning rate, inspect corrupted data, and check positives/negatives in the unrolled training file.
- Loss not decreasing: try a lower learning rate, inspect data quality, and confirm positives and hard negatives are sensible.
- Overfitting: start real corpora at 1-2 epochs; the default 3 epochs is mainly for small example datasets.
- Small datasets may trigger training-code auto-scaling of batch size or checkpoint/validation frequency. Preserve those log messages when reporting what happened.
- Checkpoint expectations are wrong: Stage 3 and Stage 4 default to `checkpoints/LATEST/model/consolidated`; pass explicit paths when using older or custom checkpoints.
- Rerank optimizer confusion: `optimizer_backend=auto` should use Transformer Engine FusedAdam in the container and FlashAdamW otherwise.

## Stage 3 Eval

- Fine-tuned model looks worse: confirm eval data, prefixes, sequence lengths, prompt template, pooling/normalization, and checkpoint path match training.
- Reranker cannot improve recall: a reranker only reorders retrieved candidates. If relevant documents are missing from `top_k`, tune the embedder or retrieval index.
- Metrics look noisy: increase held-out eval queries where possible and compare on a fixed `eval_beir/` split.
- NIM eval mismatch: compare checkpoint vs ONNX vs TensorRT, then inspect quantization, pooling/normalization, prefixes, prompt template, and sequence lengths.
- If rerank NIM eval regresses, freeze the same Stage 3 `eval_data_path`, retrieval model, `top_k`, prefixes, `prompt_template`, and `max_length`; then verify the Stage 4 `model_path` is the exact checkpoint evaluated and Stage 5 `model_dir`/`use_onnx` points at the intended ONNX or TensorRT export.

## Stage 4 Export

- ONNX export fails with attention kernels: keep `attn_implementation=eager` for export.
- TensorRT export fails: first validate ONNX-only export with `export_to_trt=false`, then check the NeMo Export-Deploy container and TensorRT profile settings.
- Rerank TensorRT instability: keep the layernorm FP32 overrides unless there is a tested reason to change them.

## Stage 5 Deploy

- Docker or NGC errors: confirm Docker runtime, GPU access, NGC login/access, and `NGC_API_KEY`.
- Port conflicts: override `host_port` or stop the existing container.
- Service starts but eval fails: run the family-specific smoke test from the reference, then run Stage 3 NIM eval with `eval_nim=true eval_base=false`.

## Artifact Hygiene

- Before rerunning stages, inspect the family output directory: `output/embed/` or `output/rerank/`.
- Do not delete generated data, cached embeddings, checkpoints, exports, or running containers unless the user explicitly asks.
- If stale artifacts may be causing shape or resume problems, explain the specific path and ask before cleanup.
