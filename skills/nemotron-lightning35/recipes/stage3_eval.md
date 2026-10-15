---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-lightning35"
section: "recipes-stage3-eval"
title: "Lightning 3.5 stage 3: NeMo Gym evaluation"
source_path: "src/nemotron/recipes/lightning35/stage3_eval/"
currency: "evolving"
---

# Stage 3 recipe summary — Evaluation

Maps Lightning evaluation to `src/nemotron/recipes/lightning35/stage3_eval/`.

---

## What the repo exposes

`lightning35 eval` runs `eval.py`: it serves the HF checkpoint with vLLM inside the job (default `lightning35-rl-model:latest`), waits for the endpoint, then runs `gym eval prepare` and `gym eval run` per benchmark and collects each benchmark's aggregate metrics into `summary.json`. This follows the NeMo Gym reproducibility flow used for the published Lightning results.

Serving settings: TP 4 with expert parallelism, `nemotron_v3` reasoning parser, `qwen3_coder` tool parser, FP32 Mamba SSM cache, MTP speculative decoding with 5 tokens.

Default benchmark: `gpqa` only (needs `HF_TOKEN`). The rest of the published suite is listed in the config but needs extra assets or services: SciCode (test data), HLE, AA-LCR, AA-Omniscience, CritPt, GDPval, PinchBench (judge models), BrowseComp (search API key), tau2 (tool servers).

---

## What it does not reproduce

Scores depend on the benchmarks you enable and on the checkpoint; a recipe-trained model on open data will not match the model-card numbers.

---

## Reproduce with nemotron-customize

`eval/model_eval` uses NeMo Evaluator Launcher, not NeMo Gym, and does not list Lightning; use it only for a quick check against a served endpoint. For report- or model-card-comparable numbers, use `src/nemotron/recipes/lightning35/stage3_eval/`.
