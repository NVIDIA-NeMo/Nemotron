<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-persona-mcq-data)=
# Generate Persona-Grounded MCQ Data for SFT

Use this guide when you need multilingual, multiple-choice supervised fine-tuning (SFT) data whose questions are grounded in regional personas and whose answers have been cross-checked by several teacher models.

`sdg/persona_mcq` is a separate step from `sdg/data_designer`. It embeds a Data Designer column for question authoring inside a resumable eight-stage pipeline that deduplicates the questions, collects answers from a *teacher panel* (three or more models that answer every question independently), keeps only records on which the panel agrees, and writes aligned `train.jsonl` files per teacher and language mix.

Use `byob/mcq` instead when the output is a held-out benchmark or evaluation set rather than training data.

## Outcomes

- Run the pipeline on the shipped `tiny` configuration to validate model endpoints and persona assets.
- Understand what each stage consumes and produces, and how to resume or rerun selected stages.
- Select a teacher and language-mix variant and pass it to `sft/automodel` or `data_prep/sft_packing`.

## How It Works

The pipeline runs these stages in a fixed order; a selected stage requires the outputs of the stages before it:

| Stage | Consumes | Produces |
|---|---|---|
| `personas` | `languages.*.locale` | Cached Data Designer persona assets for every configured locale |
| `questions` | Personas, `question_models` | One MCQ per persona sample, per question model and language |
| `lexical_dedup` | Question records | Structurally valid, script-filtered, exact- and MinHash-deduplicated questions pooled across question models |
| `semantic_dedup` | Lexically deduplicated questions | Near-duplicate removal by cosine similarity of `intfloat/multilingual-e5-large` embeddings |
| `answer_seed` | Deduplicated questions | Stable `query_id` values and deterministic option shuffles |
| `answers` | Answer seed, `answer_models` | Per-teacher answers, parsed answer letters, and retryable failures |
| `build_sft` | All teacher answers | Vote per question, then quality-gated `{messages, metadata}` records per response teacher |
| `sample` | Gated records | Aligned `train.jsonl` and `blend.json` files per teacher, all languages and English/target views |

Each question is authored from a deterministic sample of one persona *facet* (for example geography, profession, or cuisine) and one difficulty level; the sampling weights are defined in the plugin's `taxonomy.py`. The question, options, and final answer marker are written in the configured target language, while `sft.reasoning` independently requires the retained reasoning to be in English in the shipped configurations.

Voting is a consistency gate, not factual verification. With `sft.agreement: unanimous`, every teacher's parsed letter must match; with `majority`, strictly more than half must. A record is also rejected when the exporting teacher dissents from the vote, when its response was truncated, or when the answer or reasoning falls outside the configured script-fraction ranges. `summary.json` records the count for each rejection reason.

## Prerequisites

- Install the SDG extra: `uv sync --extra data-sdg`. It provides `data-designer`, `sentence-transformers`, and `torch`.
- The [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) on `PATH` and `NGC_API_KEY` exported. The `personas` stage downloads missing persona assets with it; cached runs do not need the key.
- Three OpenAI-compatible endpoints and one API key, exported as `QWEN_API_BASE`, `OSS_API_BASE`, `GEMMA_API_BASE`, and `NVIDIA_API_KEY`. The shipped configurations resolve endpoints from these variables and read the key from the environment at request time; neither is written to run metadata.
- `HF_TOKEN` when the embedding model is gated or anonymous Hugging Face Hub rate limits are a concern.
- A GPU for production semantic deduplication (`semantic_dedup.device: cuda` in `default.yaml`). The `tiny` configuration uses `cpu`.

## Procedure

1. Run the `tiny` configuration with a fresh experiment name:

   ```console
   $ export NGC_API_KEY='<your-ngc-api-key>'
   $ export NVIDIA_API_KEY='<your-api-key>'
   $ export QWEN_API_BASE='<endpoint>' OSS_API_BASE='<endpoint>' GEMMA_API_BASE='<endpoint>'
   $ uv run nemotron steps run sdg/persona_mcq -c tiny pipeline.experiment_name=my-verification
   ```

   `tiny` authors 8 questions per language from one question model, answers with three teachers, and samples one record per language. Artifacts are written under `${NEMOTRON_RUN_DIR:-./outputs}/persona_mcq/my-verification/`.

2. Inspect `summary.json` in the experiment directory. Confirm that the `answers` entries report `unparsed: 0` for every teacher and language, and review the rejection reasons under `build_sft` (`no_agreement`, `teacher_dissents`, `truncated`, `empty_answer`, `answer_language_impurity`, `empty_reasoning`, `reasoning_language_impurity`) before scaling.

3. Run the production-shaped defaults:

   ```console
   $ uv run nemotron steps run sdg/persona_mcq -c default pipeline.experiment_name=my-run
   ```

   `default.yaml` requests 100,000 questions per question model and language and samples 50,000 records per language. For cluster execution, use the `*_sdg_persona_mcq` environment profile for your backend as described in {doc}`dispatch-to-cluster`; the profiles request one GPU for semantic deduplication and forward `NVIDIA_API_KEY`, `NGC_API_KEY`, and the three `*_API_BASE` variables from the submitting shell.

   ```console
   $ export NEMOTRON_ENV_FILE=env.slurm.toml
   $ uv run nemotron steps run sdg/persona_mcq -c default pipeline.experiment_name=my-run --batch slurm_sdg_persona_mcq
   ```

4. Resume or rerun selected stages with the same experiment name. Failed answer rows in `answers/<model>/<language>/failures.jsonl` are retried on resume; completed rows are skipped:

   ```console
   $ uv run nemotron steps run sdg/persona_mcq -c default \
       pipeline.experiment_name=my-run 'pipeline.stages=[answers,build_sft,sample]'
   ```

   Reusing an experiment name with a different configuration is rejected. Choose a new name, or set `pipeline.overwrite=true` to delete the experiment directory and start again.

5. Select a training file. Each response teacher has a root view containing all configured languages and one `english_<target>` view per non-English language containing equal counts of English and target-language records:

   ```text
   <output_root>/<experiment_name>/training/<teacher>/train.jsonl
   <output_root>/<experiment_name>/training/<teacher>/english_hindi/train.jsonl
   <output_root>/<experiment_name>/training/<teacher>/english_malayalam/train.jsonl
   ```

## Adapt the Configuration

- **Add a language.** Add an entry under `languages` with `display_name`, `locale`, `answer_label`, `script_pattern`, and the two script-fraction ranges; set `strip_latin_glosses: true` for non-Latin scripts where parenthetical Latin glosses should be removed. Data Designer supplies the persona locales; the shipped Malayalam entry reuses `en_IN` personas because no managed Malayalam locale exists.
- **Change the region.** Override a locale, for example `languages.english.locale=en_US`. Geography comes from the sampled persona, not from the configuration key names.
- **Change the teacher panel.** Add a model alias under `models` and list it in `answer_models` and `sft.response_teachers`. At least three answer models are required, and every response teacher must be an answer model. Pointing several aliases at one endpoint is suitable only for verification runs.
- **Relax agreement.** Set `sft.agreement=majority` when unanimous agreement leaves too few records.
- **Adjust the reasoning mix.** `sampling.reasoning_off_fraction` (default `0.10`) is applied per language; `sampling.answer_variant=stripped` replaces the assistant `content` with only `<answer_label>: <letter>`.

## Downstream Use

Pass a selected `train.jsonl` to AutoModel SFT directly:

```console
$ uv run nemotron steps run sft/automodel \
    -c <project>/config/sft_automodel.yaml \
    dataset.path_or_dataset_id=<output_root>/<experiment_name>/training/gemma/english_malayalam/train.jsonl
```

For Megatron-Bridge, pack the same view through its `blend.json` first:

```console
$ uv run nemotron steps run data_prep/sft_packing \
    -c <project>/config/sft_packing.yaml \
    blend_path=<output_root>/<experiment_name>/training/gemma/english_malayalam/blend.json
```

The `sample` entry of `summary.json` lists every teacher/view combination with per-language record and reasoning-off counts, so orchestration does not need to infer filenames.

## Next Steps

- Field-level reference, outputs, and errors: {doc}`../reference/persona-mcq-config`
- Step contract: [`src/nemotron/steps/sdg/persona_mcq/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/sdg/persona_mcq/README.md)
