# Evaluation Practices

Use Stage 3 metrics as the source of truth for recipe quality. Training loss is useful for diagnosing learning dynamics, but it is not retrieval accuracy.

## Minimum Practice

- Compare base vs fine-tuned on the same held-out eval set.
- Keep the Stage 1 `eval_beir/` split fixed across hyperparameter, SDG, and data-volume comparisons.
- Inspect `output/embed/stage3_eval/eval_results.json` or `output/rerank/stage3_eval/eval_results.json`.
- Prioritize nDCG@10 and Recall@10, then check the rest of the k values for consistency.
- Use at least 100 eval queries when possible; 200-500 is better for detecting small changes.
- Treat less than roughly 5 absolute points of nDCG@10 improvement as a reason to inspect data quality, SDG coverage, hard negatives, and hyperparameters before deployment.
- For rerank, treat high Recall with low nDCG as a ranking problem; treat low Recall as a first-stage retrieval or embedding problem.
- Public benchmarks can be useful for broad sanity checks, but recipe personalization should be judged on the recipe's domain-specific held-out eval split.

## Experiment Hygiene

- Save the exact command, dotlist overrides, git commit, config files, and output directory for each run.
- Change one major variable at a time.
- Start embedding LR sweeps near `5e-6`, `1e-5`, and `2e-5`.
- Start rerank LR sweeps near `1e-6`, `3e-6`, and `1e-5`.
- Start real datasets at 1-2 epochs unless validation and Stage 3 metrics continue improving.
- Evaluate data saturation by running 25%, 50%, and 100% corpus sizes with the same held-out eval set.

## Deployment Checks

- Evaluate the exported or served model against the same eval set.
- For embedding NIM, use `uv run nemotron embed eval -c default eval_nim=true eval_base=false`.
- For rerank NIM, use `uv run nemotron rerank eval -c default eval_nim=true eval_base=false`.
- If metrics drift after export or deploy, check ONNX vs TensorRT, quantization, pooling, normalization, prefixes, prompt templates, and sequence length.
