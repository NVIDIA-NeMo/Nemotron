# Nemotron Step Catalog

Use this as the first-line routing reference for `/nemotron-customize`.
After selecting a likely step here, verify the exact live contract with the
CLI and source files only when you need current fields, checked-in config names,
or runner behavior.

## Table Of Contents

- [Selection Rules](#selection-rules)
- [Step Summary](#step-summary)
- [Category Notes](#category-notes)
- [Fallbacks](#fallbacks)

## Selection Rules

- Pick an existing catalog step before considering new code.
- Route by artifact contract first: downstream `consumes` decides upstream `produces`.
- Compose multi-step pipelines by artifact matching, not by fixed recipes:
  start from the requested end goal, then walk backward through `ARTIFACTS.md`,
  inserting whichever step produces the input type the next step consumes. Add
  prerequisite steps (data cleaning, packing/prep, conversion, eval) only when a
  downstream `consumes` type is not already available upstream. Do not hardcode
  named step combinations; derive every chain from the goal and the artifact graph.
- Each step is independent and selected on its own merits; the agent stitches
  steps together. A given step never implies a fixed predecessor or successor.
- Use AutoModel for HF-native JSONL, small GPU counts, quick LoRA, and direct HF output.
- Use Megatron-Bridge for packed Parquet, bin/idx, multi-node parallelism, Nano3/Super3 recipe parity, and Megatron checkpoints.
- LoRA/PEFT on a HuggingFace base with a small GPU count (about 1-8 GPUs) routes
  to `peft/automodel`. Use `peft/megatron_bridge` only when the base is a
  Megatron checkpoint or the run needs packed Parquet plus multi-node
  parallelism. When the user says LoRA/PEFT + HF model + few GPUs and gives no
  Megatron signal, the answer is `peft/automodel` (do not offer Megatron-Bridge
  as the default).
- Use `data_prep/sft_packing` before `sft/megatron_bridge` or `peft/megatron_bridge`; skip it for AutoModel SFT/PEFT.
- Use `data_prep/pretrain_prep` before either pretraining backend.
- Use `data_prep/rl_prep` when RL data starts as HF references, blends, or needs sharding/materialization.
- Route light Curator smoke tests, cleaned local JSONL output, permissive
  filtering, and first-pass IO/schema validation to `curate/nemo_curator`.
  Require concrete `input_glob` and `output_dir` before a runnable command.
- Route raw Parquet, column normalization, or missing stable document IDs to
  `curate/ingest` before filtering. For defensible thresholding, profile the
  unfiltered corpus with `curate/profile`, obtain an explicit policy approval,
  apply it with `curate/nemo_curator`, and verify the producer manifest/ledger
  with `curate/audit`. Use `curate/decontamination` before final subset/training
  when a held-out set must be protected, and `curate/subset` for nested,
  fixed-token-budget ablations.
- Route direct corpus translation to `translate/nemo_curator`. It consumes
  `filtered_jsonl`, so any upstream producing translation-ready JSONL (curation,
  SDG, or a user corpus) satisfies it; insert an upstream step only when the
  input is not yet translation-ready.
- HARD GUARD (overrides artifact composition): MCQ, multiple-choice, or any
  benchmark/evaluation dataset routes to `byob/mcq` for BOTH creation and
  translation — never `translate/nemo_curator`, even when the user says
  "translate". `translate/nemo_curator` is for plain training corpora only; it
  flattens MCQ structure (question/options/answer_index) and breaks the
  benchmark. Trigger on: "MCQ", "multiple choice", "benchmark", "eval set",
  "questions and options", or any `answer`/`answer_index` schema. When unsure
  whether data is a benchmark, ask before routing.
- Persona-grounded MCQ-shaped **SFT data** routes to `sdg/persona_mcq`, not
  `byob/mcq`; the latter is held-out benchmark data. Function-calling benchmark
  generation, localization, or executable evaluation routes to `byob/bfcl`.
- Tokenizer extension routes through `tokenizer_extension/extend`, optional
  `evaluate`, `init_embeddings`, and optional `eval_init`, followed by CPT and
  downstream `eval/model_eval` when model quality is in scope.
- Insert conversion only when adjacent stages disagree on checkpoint type.
- Bookend quality-changing stages with `eval/model_eval`.

## Step Summary

| Step | Use When | Consumes | Produces | Configs | Key Knobs / Notes |
|---|---|---|---|---|---|
| `byob/mcq` | Generate or translate domain MCQ benchmarks while preserving answer indexes and row identity. | `benchmark_source_corpus`; optional `benchmark_parquet` | `mcq_benchmark_parquet`; optional `translated_mcq_benchmark_parquet` | `default`, `tiny`, `translate` | `family=mcq`, `stage=prepare/generate/translate/all`, `target_source_mapping`, translation settings. Final rows keep `question_id`, `question`, `options`, `answer_index`, `answer`, `cot_content`, `src`, `category`. |
| `byob/bfcl` | Build, localize, or evaluate an executable function-calling benchmark from an allowlisted oracle pack. | optional `oracle_pack` | BFCL parquet/manifests/caches; localized benchmark; eval artifacts | `default`, `tiny`, `translate`, `eval.*` | `stage=prepare/generate/translate/eval/all`. Verify/freeze the source before evaluation; direct and Launcher eval have separate orchestration configs. Do not treat compatibility exports as independently authored truth. |
| `curate/ingest` | Normalize raw Parquet/JSONL, project columns, and mint stable content-derived document IDs. | `raw_jsonl` | `prepared_jsonl` | `default`, `tiny` | Choose `id_from` or explicit `id_fields`; duplicate handling changes corpus semantics and defaults to refusal. CPU-only. |
| `curate/profile` | Measure signal distributions and threshold retention before filtering. | `raw_jsonl` or `prepared_jsonl` | `profile_report`, unapproved `filter_policy` | `default`, `en` | Profile unfiltered input. Pin language pack and tokenizer revision; candidate policies are never implicitly approved. |
| `curate/nemo_curator` | Apply light filters or an approved policy and emit downstream JSONL plus accounting evidence. | `raw_jsonl` or `prepared_jsonl`; optional `filter_policy` | `filtered_jsonl`, `curation_manifest`, `curation_ledger` | `default`, `tiny` | Preserve needed columns with `metadata_fields`. `annotate`/`both` keep scores; word-count bounds are opt-in because whitespace counting is not language-neutral. |
| `curate/audit` | Independently verify shard readability, counts, digests, and optional containment. | `filtered_jsonl`; optional manifest/ledger | `curation_report` | `default`, `tiny` | Completeness requires a producer manifest; cause attribution requires a ledger. A damaged unreadable shard makes row counts a floor. |
| `curate/decontamination` | Remove whole training documents that duplicate/near-duplicate a held-out split. | `filtered_jsonl` | `decontaminated_jsonl`, `decontamination_report` | `default`, `tiny` | Holdout is read-only. Similarity uses GPU MinHash/LSH plus exact Jaccard; `skip_similarity=true` runs source-identity only on CPU. Does not detect a short benchmark item embedded in a long document. |
| `curate/subset` | Produce nested, stratified fixed-token-budget tiers for controlled ablations. | `filtered_jsonl` or `decontaminated_jsonl` | `filtered_jsonl`, `subset_plan`, `subset_report` | `default`, `tiny` | Stable unique IDs are mandatory. Plan all tiers together and pin tokenizer revision; shortfall may be required to preserve nesting. |
| `translate/nemo_curator` | Translate plain JSONL/Parquet training corpora or chat messages. NOT for MCQ/benchmark/eval datasets -> those go to `byob/mcq`. | `filtered_jsonl` | `translated_jsonl` | `default` | Require source/target language, input/output paths, format, `text_field`, backend, and auth env-var names. Preserve user-provided globs exactly. Use `messages.*.content` with `reconstruct_messages=true` for chat. |
| `sdg/data_designer` | Generate synthetic SFT, tool-call SFT, or DPO preference data from seeds and declarative columns. | optional `training_jsonl` | `synthetic_jsonl` | `default`, `customer_support_tools`, `rl_pref`, `tiny` | Use preview/tiny before scale. `default` emits OpenAI messages, `customer_support_tools` emits tool-call records, `rl_pref` emits DPO preference rows. |
| `sdg/persona_mcq` | Generate multilingual persona-grounded MCQ-shaped SFT training data, not a held-out benchmark. | - | `training_jsonl` | `default`, `tiny` | Resumable staged pipeline; use a fresh `pipeline.experiment_name`, at least three answer teachers, explicit language/script contracts, and inspect agreement/purity summaries. Production semantic dedup uses one GPU. |
| `data_prep/sft_packing` | Pack chat JSONL for Megatron-Bridge SFT/PEFT. | `training_jsonl` | `packed_parquet` | `default`, `tiny` | `tokenizer`, `pack_size`, `chat_template`, split ratios, shard counts. `pack_size` must match downstream seq length. |
| `data_prep/pretrain_prep` | Tokenize text blends into Megatron bin/idx shards and `blend.json`. | `filtered_jsonl` | `binidx` | `default`, `tiny` | `blend_path`, tokenizer, shards, splits, `text_field`. Rebuild if tokenizer changes. |
| `data_prep/rl_prep` | Resolve HF references and shard prompt/preference data for RL. | `training_jsonl` | `training_jsonl` | `default`, `tiny` | Validate DPO chosen/rejected ordering and RLVR verifier fields before training. |
| `sft/automodel` | HF-format SFT on OpenAI-style chat JSONL, smaller GPU counts, direct HF output. | `training_jsonl` | `checkpoint_hf` | `default`, `tiny` | `model.pretrained_model_name_or_path`, `dataset.path_or_dataset_id`, `peft=null/lora`. Do not feed packed Parquet. |
| `sft/megatron_bridge` | Distributed SFT with packed Parquet and Megatron checkpoints. | `packed_parquet`; optional `checkpoint_megatron` | `checkpoint_megatron` | `default`, `tiny`; topology-only `super3_128k`, `super3_256k` | Nano3 default min 8 GPUs; Super3 min 32. Keep all four sequence-length fields and prepared pack size identical. Long-context Super3 configs are dry-run references and need alignment-aware external packing before launch. |
| `peft/automodel` | LoRA adapter tuning with HF base and direct JSONL, especially 1-4 GPUs. | `training_jsonl` | `checkpoint_lora` | `default`, `tiny` | Keep base model/tokenizer/rank/alpha provenance for later merge. |
| `peft/megatron_bridge` | LoRA over a Megatron base with packed Parquet and distributed parallelism. | `packed_parquet`, `checkpoint_megatron` | `checkpoint_lora` | `default`, `tiny` | Plan merge/export path up front; keep base, adapter, merged outputs separate. |
| `pretrain/automodel` | HF-native pretraining/CPT over bin/idx data. | `binidx` | `checkpoint_hf` | `default`, `tiny` | `load_weights=true` for CPT with lower LR; set dataset paths to emitted `blend.json`. |
| `pretrain/megatron_bridge` | Large distributed pretraining/CPT with TP/PP/CP/EP and Megatron output. | `binidx`; optional `checkpoint_megatron` | `checkpoint_megatron` | `default`, `tiny` | Use for large token budgets and recipe parity; keep token budget, seq length, and blend fixed. |
| `rl/nemo_rl/dpo` | Static preference-pair alignment. | `training_jsonl`, `checkpoint_megatron` | `checkpoint_megatron` | `default`, `tiny` | Data requires `prompt`, `chosen`, `rejected`; validate pair ordering. |
| `rl/nemo_rl/rlvr` | GRPO/RLVR with deterministic/verifiable rewards. | `training_jsonl`, `checkpoint_megatron` | `checkpoint_megatron` | `default`, `nemo_gym`, `tiny` | Data needs verifier fields such as answer/tests/env metadata. Use `nemo_gym` for resource-server rewards. |
| `rl/nemo_rl/rlhf` | RLHF with learned judge/GenRM reward model. | `training_jsonl`, `checkpoint_megatron`, `checkpoint_hf` | `checkpoint_megatron` | `default`, `tiny` | Keep policy, reference, reward model, NeMo-Gym server config, and prompt data separate. |
| `convert/hf_to_megatron` | A Megatron consumer needs an HF checkpoint. | `checkpoint_hf` | `checkpoint_megatron` | `default` | Convert clean model dirs, not logs/optimizer/adapters. Merge LoRA first when needed. |
| `convert/megatron_to_hf` | HF-native eval/deploy/optimize needs a Megatron checkpoint. | `checkpoint_megatron` | `checkpoint_hf` | `default` | Point at a concrete `iter_*` checkpoint, not the parent run directory. |
| `convert/merge_lora` | Produce a standalone checkpoint from a LoRA adapter and exact base. | `checkpoint_lora`, `checkpoint_hf`; optional `checkpoint_megatron` | `checkpoint_hf`; optional `checkpoint_megatron` | `default` | Merge only into the exact base used for adapter training. Evaluate adapter and merged outputs separately. |
| `optimize/modelopt/quantize` | FP8/NVFP4/PTQ for deployment footprint. | `checkpoint_hf` | `checkpoint_megatron` | `default`, `fp8`, `nvfp4`, `tiny` | H100/Hopper -> `fp8`; B200/Blackwell -> `nvfp4`; representative calibration is required for quality. |
| `optimize/modelopt/prune` | Structured architecture pruning or target-parameter search. | `checkpoint_hf` | `checkpoint_hf` | `default`, `tiny` | Use target params or exact export config, not both. Distill afterward if quality matters. |
| `optimize/modelopt/distill` | Teacher-student recovery or standalone distillation. | `checkpoint_hf`; optional `binidx` | `checkpoint_megatron` | `default`, `tiny` | Mock data is launch validation only. Teacher is usually the original BF16 checkpoint. |
| `tokenizer_extension/extend` | Train and splice target-language subwords into a base tokenizer. | `checkpoint_hf` | `tokenizer` | `default` | CPU-only. `method=add`, `replace`, or naive `expand`; one arm per job. Fix language, corpus, normalization, and extension budget across comparisons; confirm `tokens_spliced`. |
| `tokenizer_extension/evaluate` | Compare tokenizer fertility on a held-out corpus. | `tokenizer` | `eval_results` | `default` | CPU streaming. Run each tokenizer on the exact same corpus/slice; lower fertility is better but is not model-quality evidence. |
| `tokenizer_extension/init_embeddings` | Resize an HF model and initialize rows for an extended tokenizer. | `tokenizer`, `checkpoint_hf` | `checkpoint_hf` | `default` | `arm` must match extend (`add` also covers `expand`; `replace` needs `id_remap.json`). Default to `subword/uniform`; validate FOCUS/encoder-weighted methods per language. |
| `tokenizer_extension/eval_init` | Compare resized/CPT checkpoints across vocabularies. | `checkpoint_hf` | `eval_results` | `default` | GPU. Use BPB on the same bytes and `max_docs`; per-token loss/PPL and `max_tokens` are not cross-vocabulary comparable. |
| `eval/model_eval` | Evaluate a Megatron checkpoint or an existing OpenAI-compatible endpoint. | optional `checkpoint_megatron` | `eval_results` | `default`, `tiny_chat`, `direct`, `base_en`, `instruct_en`, `milu`, `mmlu_prox*` | `launcher` submits then must be polled; `direct` runs the harness in the Nemotron job and needs a separately hosted endpoint, harness image, matching tokenizer, and durable results path. Exact task names come from the selected registry/image. |
| `env/env_toml` | Generate Lepton, Slurm, or DGX Cloud env profile TOML. | - | `env_toml` | `lepton`, `slurm`, `dgxcloud` | Keep site logistics in env TOML and step runtime flags in YAML. Export `NEMOTRON_ENV_FILE` for non-default env files. |

## Category Notes

### Curation, Translation, And Data Generation

- Curation may be a lightweight standalone filter or the governed flow:
  ingest -> profile -> human approval -> filter -> audit -> decontamination ->
  subset. Do not apply candidate thresholds directly, profile already-filtered
  output, or claim completeness/attribution without the corresponding manifest
  and ledger.
- Translation is a data step, not benchmark translation for MCQ artifacts. For chat/tool/code data prefer the `llm` backend; for large plain text and local service prefer `nmt`; for high-value data enable FAITH and keep scores.
- SDG must project to the downstream schema: OpenAI messages for SFT, structured messages for tool-call SFT, DPO preference rows for DPO.
- `sdg/persona_mcq` emits training JSONL; `byob/mcq` and `byob/bfcl` emit
  held-out benchmark artifacts. Never route solely from the word "MCQ" without
  deciding whether the requested output trains or evaluates the model.

### Tokenizer Extension

- The standard chain is `extend -> evaluate` (optional tokenizer metric) and
  `extend -> init_embeddings -> pretrain/* -> eval_init/eval/model_eval`.
- Keep the base model, corpus slice, language profile, normalization, and
  extension size fixed across add/replace/expand comparisons.
- A resized checkpoint is ready for CPT, not finished adaptation. Rebuild
  tokenizer-locked `binidx`/packed data with the extended tokenizer.
- Token fertility is a tokenizer metric; BPB is the cross-vocabulary model
  metric; downstream benchmarks remain separate.

### SFT And PEFT

- AutoModel paths consume JSONL directly and produce HF-format outputs or HF adapters.
- Megatron-Bridge paths consume packed Parquet and produce Megatron checkpoints or Megatron adapters.
- For small datasets, tight memory, or narrow changes, try LoRA before full SFT.
- Deterministic LoRA backend choice: HuggingFace base + LoRA/PEFT + about 1-8
  GPUs -> `peft/automodel`. Megatron base, packed Parquet, or multi-node scale
  -> `peft/megatron_bridge`. Do not present Megatron-Bridge as the default for
  the small-GPU HuggingFace LoRA case.
- Preserve tokenizer, chat template, base checkpoint, LoRA rank/alpha, and data blend provenance through merge/eval.

### Pretraining And CPT

- Data prep is mandatory: both backends consume bin/idx plus `blend.json`.
- CPT is a lower-LR, blend-sensitive run from existing weights; from-scratch pretraining uses a full token-budget schedule.
- Record target tokens, seq length, global batch size, train iters, LR schedule, checkpoint cadence, and validation slices before launch.

### RL

- DPO: static preference pairs only.
- RLVR: deterministic/programmatic verifier, tests, answers, or resource-server reward.
- RLHF: learned reward/judge model or GenRM path.
- All RL stages warm-start from a validated SFT `checkpoint_megatron`.

### Conversion, Optimization, Evaluation

- Convert only at real format boundaries.
- Optimization happens after source checkpoint eval, never before the customization is proven.
- Evaluation should surround SFT, RL, conversion, and optimization whenever quality is being claimed.
- In launcher mode, a successful step exit means the invocation was accepted,
  not that tasks passed. In direct mode the step blocks to completion. Both chat
  and completions harnesses need the evaluated tokenizer when the harness loads
  one client-side, and extended checkpoints must use their own tokenizer.

## Fallbacks

Use bundled references first:

1. This catalog for routing and step fit.
2. `ARTIFACTS.md` for type compatibility.
3. `COMMANDS.md` for run shapes, profile rules, and source tiers.
4. `PATTERNS.md` for cross-step guardrails.
5. `HARDWARE.md` for GPU/backend heuristics.

Fall back to source files only when:

- The bundled reference is missing a needed field or looks stale.
- You need exact current parameter names, config fields, smoke config names, or runner imports.
- You are about to write YAML or emit a command that must match the checked-in repo.

Source fallback order for a selected step: CLI `steps show/list` when available,
then `src/nemotron/steps/<step>/step.toml`, checked-in config YAML, step
README, `step.py`, and shared runner code.
