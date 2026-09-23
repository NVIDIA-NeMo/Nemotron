# Persona MCQ SDG

`sdg/persona_mcq` is a config-driven, resumable Nemotron step for generating
persona-grounded multiple-choice SFT data. It generates regionally grounded
questions from configured persona locales, removes lexical and semantic
duplicates, asks the configured teacher panel to answer each question, applies
agreement and quality gates, and writes aligned SFT JSONL for downstream
training comparisons.

The reusable Data Designer column lives in `sdg/plugins/persona_mcq`; future SDG
steps can consume it without copying the pipeline.

Add model aliases when a larger teacher panel is useful. `sft.agreement` accepts
`unanimous` (every parsed answer must match) or `majority` (strictly more than
half of the configured teachers), and an exported teacher response must match
the resulting vote.

## Install

```bash
uv sync --extra data-sdg
export NGC_API_KEY='<your-ngc-api-key>'
uv run nemotron steps run sdg/persona_mcq -c tiny
```

Keep the [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) on `PATH`
for local, Slurm, and DGX Cloud/Run:ai execution. The shipped Lepton Persona
MCQ profiles install the official CLI in each worker during startup. The first
`personas` stage derives its locales from `languages`, downloads only missing
managed assets using `NGC_API_KEY`, and caches them under
`DATA_DESIGNER_MANAGED_ASSETS_PATH`. The key remains in the environment; it is
not written into the pipeline config or an NGC config file. Cached runs do not
require it.

For Lepton, Slurm, or DGX Cloud/Run:ai, export the key in the submission shell.
The Persona MCQ environment profiles forward it, so the download happens on the
remote worker and the assets remain in its configured shared storage:

```bash
export NGC_API_KEY='<your-ngc-api-key>'
uv run nemotron steps run sdg/persona_mcq -c tiny --batch lepton_sdg_persona_mcq_tiny
```

The shipped configs generate English, Hindi, and Malayalam. They use the
`en_IN` and `hi_Deva_IN` persona assets; Malayalam currently uses `en_IN`
personas because Data Designer does not provide a managed Malayalam persona
locale. The output language is selected independently by `display_name`.
Language handling is not dispatched from those three key names: every entry
under `languages` configures its own `display_name`, localized `answer_label`,
optional `answer_label_aliases`, `script_pattern`, question/answer
script-fraction ranges, and optional Latin gloss removal. Add another language
by adding one such entry and staging its configured persona `locale`.

Reasoning is configured separately under `sft.reasoning`. The shipped configs
prompt and retain English `reasoning_content` for every question language, while
the question, options, and final answer marker use the configured target
language. Geography is also not fixed to India: question grounding comes from
the sampled persona. Download another supported locale and override the
corresponding value, for example:

```bash
export NGC_API_KEY='<your-ngc-api-key>'
uv run nemotron steps run sdg/persona_mcq -c tiny \
  languages.english.locale=en_US pipeline.experiment_name=us-smoke
```

Set `QWEN_API_BASE`, `OSS_API_BASE`, `GEMMA_API_BASE`, and `NVIDIA_API_KEY`.
Also export `HF_TOKEN` when the embedding model is gated or to avoid anonymous
Hugging Face Hub rate limits. Resolved credentials are never written to
committed configs or run metadata; endpoint URLs are retained in the redacted
run configuration for provenance.

## Run

The generic Nemotron step CLI discovers Persona MCQ from its `step.toml` manifest:

```bash
uv run nemotron steps list --category sdg
uv run nemotron steps show sdg/persona_mcq
```

Start with the smoke profile:

```bash
uv run nemotron steps run sdg/persona_mcq -c tiny \
  pipeline.experiment_name=my-smoke
```

Run production-shaped defaults only after inspecting the smoke artifacts:

```bash
uv run nemotron steps run sdg/persona_mcq -c default \
  pipeline.experiment_name=my-run
```

Run remotely through the same Nemotron environment-profile interface as other
steps. The generated environment templates include profiles for Lepton, Slurm,
and DGX Cloud (Run:ai); select attached execution with `--run` or detached
execution with `--batch`:

```bash
export NEMOTRON_ENV_FILE=env.lepton.toml
uv run nemotron steps run sdg/persona_mcq -c tiny \
  --run lepton_sdg_persona_mcq_tiny

export NEMOTRON_ENV_FILE=env.slurm.toml
uv run nemotron steps run sdg/persona_mcq -c default \
  --batch slurm_sdg_persona_mcq

export NEMOTRON_ENV_FILE=env.dgxcloud.toml
uv run nemotron steps run sdg/persona_mcq -c default \
  --batch dgxcloud_sdg_persona_mcq
```

Before Slurm or DGX Cloud/Run:ai submission, keep the NGC CLI on the remote
image's `PATH`; the Lepton profile installs it during worker startup. Export
`NGC_API_KEY`, the three endpoint variables, and `NVIDIA_API_KEY`. Export
`HF_TOKEN` when Hub authentication is needed; every generated backend profile
forwards it from the submitting environment.
`NEMOTRON_RUN_DIR` also points to shared storage so detached runs can resume and
their outputs persist after the worker exits.

Run or resume selected stages with an OmegaConf list override:

```bash
uv run nemotron steps run sdg/persona_mcq -c default \
  pipeline.experiment_name=my-run 'pipeline.stages=[answers,build_sft,sample]'
```

Stages always follow this order: `personas`, `questions`, `lexical_dedup`,
`semantic_dedup`, `answer_seed`, `answers`, `build_sft`, `sample`. Inputs for a
selected stage must already exist. Reusing an experiment name with a different
configuration is rejected unless `pipeline.overwrite=true` is explicit.

Use `sdg/persona_mcq` for persona-grounded MCQ-shaped **SFT training data**. Use
`byob/mcq` instead when the output is a held-out benchmark or evaluation set.

## Artifacts

Artifacts are rooted at `<output_root>/<experiment_name>/`:

- `questions/`, `lexical/`, `semantic/`, and `answer_seed/` preserve generation
  and deduplication provenance.
- `answers/<model>/<language>/` contains append-safe successes and retryable
  failures.
- `sft/<teacher>/<language>.jsonl` contains quality-gated records.
- `training/<teacher>/train.jsonl` contains aligned English/Hindi/Malayalam
  samples in the `training_jsonl` contract.
- `training/<teacher>/blend.json` points SFT packing at that teacher's JSONL.
- `training/<teacher>/english_hindi/` and `english_malayalam/` contain equal-size
  English/target `train.jsonl` and `blend.json` views for downstream comparisons.
- `run.json` records the redacted configuration and dependency/repository versions;
  `summary.json` records stage yields and rejection reasons.

The final schema is `{messages, metadata}`. Reasoning-on records include English
`assistant.reasoning_content`; the deterministic reasoning-off subset omits it.
The assistant `content` uses the configured target-language answer label.

## Downstream handoff

Choose one teacher and language-mix variant. The root teacher file contains all
configured languages; the `english_<target>` views contain the 50/50
English/target mix used by target-language comparisons. The shipped 0.10
`reasoning_off_fraction` is applied exactly per language, producing a 90:10
reasoning-on/off mix when the requested sample count is divisible by ten.

Pass the selected stable `train.jsonl` filename directly to AutoModel SFT:

```bash
uv run nemotron steps run sft/automodel \
  -c <project>/config/sft_automodel.yaml \
  dataset.path_or_dataset_id=<output_root>/<experiment_name>/training/gemma/english_malayalam/train.jsonl
```

Megatron-Bridge workflows first pack the same data using the emitted blend
manifest, then train from the resulting Parquet splits:

```bash
uv run nemotron steps run data_prep/sft_packing \
  -c <project>/config/sft_packing.yaml \
  blend_path=<output_root>/<experiment_name>/training/gemma/english_malayalam/blend.json
```

The `summary.json` sample-stage entry records every teacher/view combination,
including per-language sample and reasoning-off counts, so orchestration does
not need to infer filenames. English instruction-following replay or translation
data are separate inputs that can be composed downstream; this step emits the
persona-MCQ portion of a training-data mix.

## Guardrails

- Never commit generated data, resolved secrets, or endpoint-specific configs.
- Preserve the same model list across answering, voting, and aligned sampling.
- Keep the reasoning and answer script contracts explicit. The shipped config
  validates English reasoning independently from the target-language answer.
- Configure teacher aliases to use distinct model endpoints in production;
  pointing several aliases at one endpoint is suitable only for smoke testing.
- Treat teacher agreement as a consistency gate, not factual verification;
  fact-check or separately judge generated examples before training on them.
- Treat a low shared-teacher intersection as a quality signal, not something to
  bypass silently.
- Use a GPU for production semantic deduplication; the tiny profile uses CPU for
  portability only.
