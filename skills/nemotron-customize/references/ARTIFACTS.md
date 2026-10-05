# Artifact Compatibility

Use this reference before planning DAGs or inserting conversion stages. It is a
compact copy of the catalog artifact graph; verify with `src/nemotron/steps/types.toml`
only when exact live metadata is required.

## Table Of Contents

- [Type Graph](#type-graph)
- [Common Pipelines](#common-pipelines)
- [Compatibility Checks](#compatibility-checks)

## Type Graph

| Artifact | Meaning | Compatible As | Explicit Conversion |
|---|---|---|---|
| `raw_jsonl` | Raw downloaded/local JSONL records. | `training_jsonl` | - |
| `prepared_jsonl` | Raw JSONL normalized to canonical text/id/source fields with a stable document ID. | `raw_jsonl`, therefore `training_jsonl` | Produced by `curate/ingest`. |
| `filtered_jsonl` | JSONL accepted for downstream data steps, often after curation/language/domain filters. Existing clean corpora may enter here without a new curation run. | `training_jsonl` | - |
| `decontaminated_jsonl` | Filtered training JSONL with whole-document holdout overlaps removed. | `filtered_jsonl`, therefore `training_jsonl` | Produced by `curate/decontamination`. |
| `translated_jsonl` | Translated JSONL plus optional quality metadata. | `training_jsonl` | - |
| `synthetic_jsonl` | Data Designer generated JSONL. | `training_jsonl` | - |
| `training_jsonl` | OpenAI-style chat JSONL or RL prompt/preference JSONL. | - | - |
| `packed_parquet` | Packed Megatron-Bridge SFT shards with `input_ids` and `loss_mask`. | - | Produced by `data_prep/sft_packing`. |
| `binidx` | Megatron pretraining bin/idx shards plus `blend.json`. | - | Produced by `data_prep/pretrain_prep`. |
| `tokenizer` | Extended tokenizer directory with vocabulary/merge files. | - | Produced by `tokenizer_extension/extend`; consumed by fertility and embedding-init steps. |
| `checkpoint_megatron` | Megatron distributed checkpoint. | - | `convert/megatron_to_hf` -> `checkpoint_hf`. |
| `checkpoint_hf` | Hugging Face safetensors checkpoint. | - | `convert/hf_to_megatron` -> `checkpoint_megatron`. |
| `checkpoint_lora` | LoRA adapter weights. | - | `convert/merge_lora` -> `checkpoint_hf`; optional Megatron output. |
| `eval_results` | Evaluation metrics and output artifacts. | - | - |
| `profile_report` | Signal distributions and retention surfaces measured on a corpus. | - | Produced by `curate/profile`. |
| `filter_policy` | Corpus-bound filtering thresholds plus approval state. | - | Candidate from `curate/profile`; only an explicitly approved policy may feed `curate/nemo_curator`. |
| `curation_manifest` | Producer-declared shard list, counts, digests, policy state, and completion marker. | - | Consumed by `curate/audit` for completeness claims. |
| `curation_ledger` | Per-stage input/success/filter/failure accounting. | - | Consumed by `curate/audit` for cause attribution. |
| `curation_report` | Independent readability/count/digest/containment evidence. | - | Produced by `curate/audit`. |
| `decontamination_report` | Removed training/holdout near-duplicate pairs plus exact similarity and method. | - | Produced by `curate/decontamination`. |
| `subset_plan`, `subset_report` | Planned nested quotas and achieved token-budget evidence. | - | Produced by `curate/subset`. |
| `env_toml` | Environment profile TOML for remote/local execution. | - | Produced by `env/env_toml`. |
| `benchmark_source_corpus` | Domain documents grouped by benchmark target subject. | - | - |
| `benchmark_parquet` | BYOB benchmark dataset. | - | - |
| `mcq_benchmark_parquet` | Multiple-choice BYOB benchmark parquet. | `benchmark_parquet` | - |
| `translated_mcq_benchmark_parquet` | Translated multiple-choice BYOB benchmark parquet. | `mcq_benchmark_parquet` | - |
| `oracle_pack` | Allowlisted executable function-calling oracle plus tools, fixtures, templates, assertions, and validation cases. | - | Optional input to `byob/bfcl`. |
| `bfcl_benchmark_parquet` | Published function-calling benchmark with expected calls and provenance. | `benchmark_parquet` | Produced by `byob/bfcl`. |
| `bfcl_localized_benchmark` | Localized BFCL benchmark preserving executable truth and source task identity. | `bfcl_benchmark_parquet` | Produced by `byob/bfcl stage=translate`. |
| `bfcl_run_manifest`, `bfcl_translation_manifest` | Content-addressed BFCL publication/localization lineage. | - | Required evidence for downstream BFCL source verification. |
| `bfcl_stage_cache`, `bfcl_candidate_io_cache`, `bfcl_tool_trace_cache` | Resumable generation and replay evidence. | - | Reuse only after hash/config verification. |
| `bfcl_compatibility_exports`, `bfcl_export_validation_report` | Derived BFCL JSONL/Evaluator bundles and proof they match canonical parquet. | - | Compatibility artifacts, not separate sources of truth. |
| `bfcl_source_verification_report`, `bfcl_contamination_report` | Proof of immutable source identity and candidate authorization. | - | Produced before BFCL candidate evaluation. |
| `bfcl_eval_artifacts` | BFCL eval report, task results, and binding manifest. | `eval_results` | Produced by `byob/bfcl stage=eval`. |
| `bfcl_bias_audit_reports` | Recomputed content-addressed bias audit over frozen BFCL release/eval evidence. | - | Produced by the explicit read-only BFCL bias-audit workflow. |

## Common Pipelines

### Data-To-Training (compose by artifact type)

Each data step is independent. `raw_jsonl`, `filtered_jsonl`,
`translated_jsonl`, and `synthetic_jsonl` all satisfy `training_jsonl`, so the
agent inserts a data step only when the goal requires that transform (cleaning,
translation, generation). The chain below shows the maximal path; drop any hop
the request does not need.

```text
raw_jsonl
  -> [curate/nemo_curator]      # only if cleaning/filtering is requested
  -> [translate/nemo_curator]   # only if translation is requested
  -> training_jsonl
  -> sft/automodel              # JSONL-native AutoModel path
  -> checkpoint_hf
```

```text
training_jsonl
  -> data_prep/sft_packing      # required because Megatron-Bridge consumes packed_parquet
  -> packed_parquet
  -> sft/megatron_bridge
  -> checkpoint_megatron
```

### Governed Curation

```text
raw_jsonl or raw parquet
  -> curate/ingest -> prepared_jsonl + stable ids
  -> curate/profile -> profile_report + unapproved filter_policy
  -> explicit human approval
  -> curate/nemo_curator -> filtered_jsonl + curation_manifest + curation_ledger
  -> curate/audit -> curation_report
  -> curate/decontamination -> decontaminated_jsonl + decontamination_report
  -> curate/subset -> nested filtered_jsonl tiers + subset_plan/subset_report
```

Ingest is optional only when the input already is JSONL with a stable unique
document ID. Profile measures the unfiltered corpus. Decontamination is optional
when no holdout protection is requested; if enabled, subset must consume its
output rather than the pre-decontamination corpus.

### SFT / PEFT Backend Split

```text
training_jsonl -> sft/automodel -> checkpoint_hf
training_jsonl -> peft/automodel -> checkpoint_lora -> convert/merge_lora -> checkpoint_hf
```

```text
training_jsonl
  -> data_prep/sft_packing
  -> packed_parquet
  -> sft/megatron_bridge or peft/megatron_bridge
  -> checkpoint_megatron or checkpoint_lora
```

### Pretraining / CPT

```text
filtered_jsonl
  -> data_prep/pretrain_prep
  -> binidx + blend.json
  -> pretrain/automodel        -> checkpoint_hf
  -> pretrain/megatron_bridge  -> checkpoint_megatron
```

### RL Alignment

```text
sft/megatron_bridge -> checkpoint_megatron
training_jsonl or data_prep/rl_prep output
  -> rl/nemo_rl/dpo | rl/nemo_rl/rlvr | rl/nemo_rl/rlhf
  -> checkpoint_megatron
```

### Checkpoint Bridges

```text
checkpoint_hf       -> convert/hf_to_megatron -> checkpoint_megatron
checkpoint_megatron -> convert/megatron_to_hf -> checkpoint_hf
checkpoint_lora + exact base -> convert/merge_lora -> checkpoint_hf
```

### BYOB Benchmarks

```text
benchmark_source_corpus
  -> byob/mcq stage=prepare
  -> byob/mcq stage=generate
  -> mcq_benchmark_parquet
  -> byob/mcq stage=translate
  -> translated_mcq_benchmark_parquet
```

```text
oracle_pack
  -> byob/bfcl stage=prepare/generate
  -> bfcl_benchmark_parquet + bfcl_run_manifest
  -> byob/bfcl stage=translate (optional) -> bfcl_localized_benchmark
  -> byob/bfcl stage=eval -> bfcl_eval_artifacts
```

### Persona MCQ Training Data

```text
sdg/persona_mcq
  -> training_jsonl + blend manifests
  -> sft/automodel
  OR -> data_prep/sft_packing -> sft/megatron_bridge
```

Do not join this training path to BYOB benchmark artifacts.

### Tokenizer Extension

```text
checkpoint_hf + corpus
  -> tokenizer_extension/extend -> tokenizer
  -> tokenizer_extension/evaluate (optional fertility comparison)
  -> tokenizer_extension/init_embeddings -> resized checkpoint_hf
  -> data_prep/pretrain_prep with the extended tokenizer
  -> pretrain/automodel or pretrain/megatron_bridge (CPT)
  -> tokenizer_extension/eval_init (BPB) and eval/model_eval (downstream quality)
```

## Compatibility Checks

- `is_a` compatibility is implicit; conversion is not needed for those edges.
- `convert_to` edges require an explicit converter step; do not rely on downstream steps to read another checkpoint layout.
- Prepared data is tokenizer-locked. Rebuild `packed_parquet` or `binidx` after tokenizer, chat template, sequence length, split, or blend changes.
- `prepared_jsonl` and `decontaminated_jsonl` are stronger subtypes, not
  alternate file formats. Preserve their stable IDs and post-decontamination
  ordering when feeding downstream curation steps.
- A candidate `filter_policy` is deliberately non-executable until approval is
  recorded. An audit without a manifest may report counts but cannot claim
  completeness; without a ledger it cannot attribute loss.
- `curate/decontamination` protects the holdout by shrinking training only and
  covers whole-document near-duplicates, not substring contamination.
- `packed_parquet` is only for Megatron-Bridge SFT/PEFT paths.
- AutoModel SFT/PEFT reads `training_jsonl` directly.
- `checkpoint_lora` is not a deployable full model until merged with the exact base.
- For Megatron exports, point conversion/eval at a concrete `iter_*` checkpoint.
- Keep benchmark artifacts separate from training artifacts; BYOB output is held-out eval data.
- `sdg/persona_mcq` is the exception to the word “MCQ”: its output is
  `training_jsonl`, not a benchmark artifact.
- For tokenizer comparisons, fertility runs must share one corpus/slice and BPB
  runs must score the same bytes using a document budget (`max_docs`).
