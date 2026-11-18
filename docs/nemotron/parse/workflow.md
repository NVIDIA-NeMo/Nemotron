# Run the Parse Adaptation Workflow

Use this recipe as a starting point to adapt
[Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) to your
document domain through supervised fine-tuning (SFT), reinforcement learning
with verifiable rewards (RLVR), and held-out evaluation. It provides stage
commands, a model adapter, an isolated runtime and explicit dataset/checkpoint
handoffs.

**Data preparation, rewards and evaluation are dental reference examples.**
The example generates fictional records, renders page images and complete Parse
targets, and defines dental correctness criteria for monitoring, RLVR rewards
and held-out evaluation. Customers should develop their own data preparation,
reward functions and evaluation for their use cases. Reuse the shared SFT/RLVR
training modules and [adapt the task components](adaptation.md)
to your documents, labels and quality requirements.

| Command | Input | Output |
| --- | --- | --- |
| `nemotron parse data` | Dental example record/rendering rules and seed | Images, full-page Parse targets, train/monitor/test manifests and dataset identity |
| `nemotron parse sft` | Dataset artifact, base model and optional task monitor | Full-model SFT checkpoint and training/monitor history |
| `nemotron parse rlvr` | Dataset and SFT model artifacts, task adapter and reward environment | Candidate mining, decoder GRPO checkpoint and task-defined evaluation before/after |
| `nemotron parse eval` | Dental example dataset and RLVR artifact with model lineage | Matched base → SFT → RLVR answers, dental metrics and uplift report |

The [shared training workflow](adaptation.md) supports the Parse architecture.
They provide model integration, training orchestration and artifact flow; a
configured task supplies domain-specific monitoring, rewards and evaluation.
Optional [document component scorers](rewards.md) provide building blocks for
your domain's evaluation and reward functions.

## Runtime

Use the dedicated Parse runtime to keep training and evaluation dependencies
separate from the main Nemotron environment. Follow the
[runtime setup guide](runtime.md) to install it from a Nemotron source
checkout on Linux with your GPU visible, then run the commands below.

## Run the dental worked example

After completing runtime setup, run these commands from the repository root.
They execute the dental example with its default data and scoring.

```bash
export PYTHONPATH="$PWD/src"
PARSE_RUNTIME="$PWD/.nemotron/parse-runtime"
PARSE_PYTHON="$PARSE_RUNTIME/venv/bin/python"
PARSE_RUN="$PWD/.nemotron/parse"

"$PARSE_PYTHON" -m nemotron parse data \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data"

"$PARSE_PYTHON" -m nemotron parse sft \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data" \
  output_dir="$PARSE_RUN/sft"

"$PARSE_PYTHON" -m nemotron parse rlvr \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data" \
  sft_output="$PARSE_RUN/sft" output_dir="$PARSE_RUN/rlvr"

"$PARSE_PYTHON" -m nemotron parse eval \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data" \
  model_artifact="$PARSE_RUN/rlvr/artifact.json" output_dir="$PARSE_RUN/eval"
```

An editable Nemotron installation from this source checkout can use
`nemotron parse ...` with the same arguments. Each command runs directly in a
separate runtime process and releases GPU resources when it exits. The v1
commands do not support the NeMo-Run `--run` or `--batch` options.

Default SFT downloads the pinned base-model revision
`62cb1e4ba153d113ff64aaf2be6e068c377cf53c`. Supply `model=/absolute/checkpoint`
to use an existing local snapshot, or `model=organization/parse-model` and optional
`model_revision=revision` for another compatible Hub model. SFT records the
resolved source, revision and model/processor file hashes. Evaluation uses this
recorded base, including a custom base, rather than resolving a default again.
RLVR consumes `sft_output/artifact.json`; any explicit model or processor override
must match that artifact's lineage.

Use a new output directory for each stage attempt. Completed or partial runs
are preserved, and an existing output is rejected instead of silently retraining
or resuming. This version does not implement optimizer-state resume.

Preview a configuration without installing the GPU runtime:

```bash
nemotron parse sft --dry-run
nemotron parse rlvr -c tiny --dry-run
```

`-c` accepts a bundled profile or a YAML/JSON file. Dot overrides such as
`data_dir=/path/to/data` apply afterward. `training.*` overrides merge into the
native framework configuration. The `tiny` SFT/RLVR profiles exercise two updates
and checkpoint handoff; their results are not quality benchmarks. Default
profiles use the training settings described below.

## Data preparation example

The dental data stage takes a seed and the example's record/rendering rules.
It generates fictional structured records and renders each page image together
with its complete Parse extraction target. The structured record and geometry
provide references for checking the generated output. No seed documents or
images are required: the example generates fictional records and renders their
page layouts and labels together. The numeric random seed defaults to `42`;
override it with `seed=123` and choose a new `data_dir` for a different dataset.

The example generates 256 training pages from 32 contexts, 32 independent
monitor pages and 256 independent held-out pages. Each JSONL record includes
an image reference, prompt, complete Parse target, structured dental record,
context ID and geometry. Patient context exclusions prevent split leakage.
Dataset identity includes manifest and image hashes and is checked before use.

The page manifests illustrate the information needed for adaptation:

| Field | Purpose | How to use it for your domain |
| --- | --- | --- |
| `id` | Stable page identifier | Assign unique page IDs and retain them through evaluation. |
| `image` | Page image, relative to its manifest | Keep the image inside the dataset bundle so identity checks can hash it. |
| `prompt` | Extraction instruction | Use an instruction consistent with your desired Parse output. |
| `target` | Complete supervised Parse target | Supply the visible content and relevant Parse markup, including coordinates and structured blocks. |
| `context_id` | Groups related pages or variants | Group by source document, template or seed family before creating disjoint splits. |
| `record` and geometry | Example-specific scoring references | Supply the structured references your own metrics require and adapt their consumers. |

For your domain, paired images and verified targets can come from existing
labeled documents, a renderer, or a synthetic-data pipeline. The dental renderer
demonstrates how to produce images and references together. The
[domain adaptation guide](adaptation.md) explains how to connect another source
and define its monitoring, rewards and evaluation.

## SFT, RLVR and artifact flow

With the dental example defaults, SFT runs 2,852 optimizer updates with global/local
batch size 4 and seed 42. `max_steps: 2852` sets the training budget;
`num_epochs: null` lets the scheduler derive enough epochs to reach it. The 256
training pages give 64 updates per epoch, so training stops after 44 full epochs
and 36 updates in epoch 45. The cosine learning-rate schedule uses
`lr_decay_steps: 2852`, decaying from `3e-6` to `3e-7` with no warmup. When adapting
the recipe to another dataset, tune the update budget and learning-rate schedule
together; a fixed update count does not preserve the number of dataset passes.

RLVR generates 512
independent candidate contexts, samples four answers per candidate and selects 256 contexts. Candidate images
and selected manifests live in the RLVR output's `curriculum/` directory, leaving
the fixed dataset unchanged. GRPO runs 128 updates, with 16 prompts and four
responses per update. To change native framework settings, use `training.*`
CLI overrides or the `training` section of a stage YAML configuration supplied
with `-c`. For example,
`training.step_scheduler.max_steps=128` changes the SFT update budget; keep
`training.lr_scheduler.lr_decay_steps` aligned with that budget.

Each stage writes `config.json`, `stage.json` and an explicit `artifact.json`
handoff. Nemotron Artifact metadata is saved in the stage output bundle, with
`payload_path` pointing to the dataset, checkpoint or evaluation output. Checkpoint
contents remain unchanged by artifact publication. Training also writes `training-config.json`, `schedule.json`,
`history.jsonl`, predictions and checkpoints. RLVR retains every sampled answer
under `mining/`, `before/` and `after/`, plus `evaluation.json`. No external
tracking service is required.

SFT artifacts include explicit base/SFT model lineage; RLVR adds its final
checkpoint and a reference to the parent SFT artifact. Keep the referenced model
and processor directories available and unchanged for comparison. The evaluator
verifies their file hashes and the fixed dataset identity before generation.
Artifacts without this lineage require a new training run; ancestry is not
reconstructed from previous outputs.

## Evaluation and dental example results

These reward and evaluation definitions are specific to the dental example.
Develop your own definitions for other document types and use cases.

The bundled example's reward is binary: exact tooth/procedure/surface content
and zero normalized visible-text CER. Reported sampled success also requires natural EOS.
This is content correctness, not a guarantee of box, block-class or markup
correctness. Report pass@1/pass@k alongside CER, chart, patient-field and history
metrics. A weighted general reward changes the objective and is not enabled.

The matched comparison uses the same 256 held-out pages and eight answers per
page, temperature 1.0, top-p 1.0 and seed 1231031. Sampled pass@1 estimates
first-attempt success from those answers; it is not greedy monitor accuracy.
The evaluation command compares the recorded base, SFT and RLVR checkpoints by
default, using a separate generation worker for each model on one GPU:

```bash
"$PARSE_PYTHON" -m nemotron parse eval \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data" \
  model_artifact="$PARSE_RUN/rlvr/artifact.json" output_dir="$PARSE_RUN/eval"
```

Each checkpoint has its own `base/`, `sft/` or `rlvr/` subdirectory containing
`predictions/answers.json` and `evaluation.json`. The top-level `evaluation.json`
records all metrics, lineage, the shared generation/scoring protocol and uplift
for base→SFT, SFT→RLVR and base→RLVR. `comparison.md` displays the scores and
percentage-point changes. Positive pass@1 gain and positive CER reduction mean
improvement; relative CER reduction is undefined when the starting CER is zero.
No extra comparison flag is needed.

For a single-checkpoint diagnostic, explicitly pass `model=/absolute/checkpoint`
and, if needed, `processor_source=/absolute/processor`. That path evaluates only
the selected model; the default artifact-based command produces the full report.

The reference run used one H200 per stage and 256 fixed training pages. SFT used
global/local batch size 4, seed 42 and 2,852 optimizer updates with the
learning-rate schedule described above. RLVR continued the SFT checkpoint for
128 updates with the default settings and seed 1231031. The base, SFT and RLVR
models were evaluated on the same 256 held-out pages with eight sampled answers
per page.

| Checkpoint | Sampled pass@1 ↑ | Mean text CER ↓ |
| --- | ---: | ---: |
| Base | 0.00% | 45.5291% |
| SFT | 95.85% | 0.0385% |
| RLVR | 96.58% | 0.0278% |

In this dental reference run, SFT improved sampled pass@1 by 95.85 percentage
points over the base model. RLVR added 0.73 percentage points and further reduced
mean text CER. Quality and run-to-run variation depend on the data, training
settings and runtime; measure adaptation quality on your domain's held-out data.
Preserve runtime records and report both accuracy and error metrics when
comparing checkpoints.
The initial SFT preview uses greedy monitoring. The final evaluation command
provides the matched base/SFT/RLVR comparison on the held-out set.

## Verification and attribution

Run the CPU contract tests from the repository root:

```bash
PYTHONPATH=src python -m pytest tests/recipes/parse
```

The CPU tests cover reward/EOS behavior, split exclusions, configuration,
artifact handoff, CLI dispatch and runtime isolation. The dental example's
end-to-end validation also checked checkpoint reload, dataset identity,
training histories and recomputed evaluation metrics.

See [third-party attribution](attribution.md) for the bundled fonts and
document scoring components.
