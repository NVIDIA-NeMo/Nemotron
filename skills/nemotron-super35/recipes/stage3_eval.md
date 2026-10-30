---
paper: "Nemotron 3.5 Super and Lightning Tech Report (2026)"
model: "nemotron-super35"
section: "recipes-stage3-eval"
title: "Super 3.5 stage 3: NeMo Evaluator subset of Table 10"
source_path: "src/nemotron/recipes/super35/stage3_eval/"
currency: "evolving"
---

# Stage 3 recipe summary — Evaluation

Maps the paper's post-trained evaluation (§3.8) to `src/nemotron/recipes/super35/stage3_eval/`.

---

## What the repo exposes

`nemotron super35 eval` compiles `config/default.yaml` and runs `nemo-evaluator-launcher`: it serves the checkpoint with vLLM (default model `super35-rl-mopd-phase2-model:latest`) and runs NeMo Evaluator tasks. vLLM serves Hugging Face checkpoints only, so export the selected RL step to Hugging Face, register it with `super35 model import rl`, and pass it as `run.model`. Task settings follow NeMo Evaluator's Nemotron 3 Super example.

Enabled by default: GPQA-Diamond, SciCode, LiveCodeBench v6 (`test_v6_2408_2505`, 3 repeats), IFBench, MMLU-ProX. Available but off by default because they need a judge, an API key, a user simulator, or a local scorer: HLE (text), AA-LCR, AA-Omniscience, CritPt, Tau2 Retail/Airline/Telecom, WMT24++ COMET.

---

## What it does not reproduce

- The paper collected Table 10 with **NeMo Gym**, not NeMo Evaluator. To reproduce Table 10, use NeMo Gym's Super 3.5 reproducibility guide (`nemotron_recipes/super-3.5/reproducibility.md`, on Gym's `jakub/marlin` branch until merged to `main`): one recipe per benchmark, run with `nemotron_recipes/super-3.5/run.sh <benchmark>` against a vLLM endpoint (published runs: vLLM 0.30.0, TP4 + EP on 4× GB200). It has no recipe for the multimodal benchmarks, NV Chat Benchmark, HLE with tools, the OpenCode BrowseComp harness or Text Arena. Scores from this stage are not expected to match Table 10 exactly; see `docs/nemotron/super35/evaluate.md`.
- Multimodal benchmarks (BabyVision, CharXiv-RQ, MathVision, OCRBench-v2, Video-MME-v2, ...) and agentic benchmarks (SWE-bench, Terminal-Bench, BrowseComp, ...) are not covered; several use internal scaffolding the report says will be open-sourced later.
- The vLLM container is `vllm/vllm-openai:v0.30.0`, the vLLM release used for the published NeMo Gym runs.

---

## Reproduce with nemotron-customize

`eval/model_eval` (NeMo Evaluator Launcher) is the matching catalog step; it can evaluate a hosted OpenAI-compatible endpoint, which is the simplest path for the released Hugging Face checkpoint. Use the task list in `src/nemotron/recipes/super35/stage3_eval/config/default.yaml` as the Super 3.5 starting point.
