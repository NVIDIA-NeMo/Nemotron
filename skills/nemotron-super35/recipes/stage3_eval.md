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

`nemotron super35 eval` compiles `config/default.yaml` and runs `nemo-evaluator-launcher`: it serves the checkpoint with vLLM (default model `super35-rl-mopd-phase2-model:latest`) and runs NeMo Evaluator tasks. Task settings follow NeMo Evaluator's Nemotron 3 Super example.

Enabled by default: GPQA-Diamond, SciCode, LiveCodeBench v6 (`test_v6_2408_2505`, 3 repeats), IFBench, MMLU-ProX. Available but off by default because they need a judge, an API key, a user simulator, or a local scorer: HLE (text), AA-LCR, AA-Omniscience, CritPt, Tau2 Retail/Airline/Telecom, WMT24++ COMET.

---

## What it does not reproduce

- The paper collected Table 10 with **NeMo Gym**, not NeMo Evaluator, and points to a Super 3.5 reproducibility guide in the Gym repo that is not published yet. Scores from this stage are not expected to match Table 10 exactly.
- Multimodal benchmarks (BabyVision, CharXiv-RQ, MathVision, OCRBench-v2, Video-MME-v2, ...) and agentic benchmarks (SWE-bench, Terminal-Bench, BrowseComp, ...) are not covered; several use internal scaffolding the report says will be open-sourced later.
- The vLLM container is `vllm/vllm-openai:latest` until a release that serves Super 3.5 VL is pinned.

---

## Reproduce with nemotron-customize

`eval/model_eval` (NeMo Evaluator Launcher) is the matching catalog step; it can evaluate a hosted OpenAI-compatible endpoint, which is the simplest path for the released Hugging Face checkpoint. Use the task list in `src/nemotron/recipes/super35/stage3_eval/config/default.yaml` as the Super 3.5 starting point.
