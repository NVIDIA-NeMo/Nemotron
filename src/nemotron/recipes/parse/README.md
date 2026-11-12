# Nemotron Parse model adaptation and evaluation

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
training modules and [adapt the task components](#adapt-the-example-to-your-domain)
to your documents, labels and quality requirements.

| Command | Input | Output |
| --- | --- | --- |
| `nemotron parse data` | Dental example record/rendering rules and seed | Images, full-page Parse targets, train/monitor/test manifests and dataset identity |
| `nemotron parse sft` | Dataset artifact, base model and optional task monitor | Full-model SFT checkpoint and training/monitor history |
| `nemotron parse rlvr` | Dataset and SFT model artifacts, task adapter and reward environment | Candidate mining, decoder GRPO checkpoint and task-defined evaluation before/after |
| `nemotron parse eval` | Dental example dataset and a model artifact or checkpoint | Held-out answers and dental reference metrics |

The [shared training modules](training/README.md) support the Parse architecture.
They provide model integration, training orchestration and artifact flow; a
configured task supplies domain-specific monitoring, rewards and evaluation. Optional [document component scorers](document_rewards/README.md)
provide building blocks for your domain's evaluation and reward functions.

## Runtime

Use the dedicated Parse runtime to keep training and evaluation dependencies
separate from the main Nemotron environment. Follow the
[runtime setup guide](runtime/README.md) to install it from a Nemotron source
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
```

An editable Nemotron installation from this source checkout can use
`nemotron parse ...` with the same arguments. Each command runs directly in a
separate runtime process and releases GPU resources when it exits. The v1
commands do not support the NeMo-Run `--run` or `--batch` options.

Default SFT downloads the pinned base-model revision
`62cb1e4ba153d113ff64aaf2be6e068c377cf53c`. Supply `model=/absolute/checkpoint`
to use an existing local snapshot. RLVR normally consumes `sft_output/artifact.json`;
an explicit `model=/absolute/sft/checkpoint` overrides that handoff.

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
provide references for checking the generated output. Study
[record and split preparation](dental/data.py) and the
[page renderer](dental/render.py) when building your own labeled data.

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
demonstrates how to produce images and references together; the next section
explains how to connect another source to this implementation.

## Adapt the example to your domain

Start by defining the document types, complete extraction targets and success
criteria for your task. The SFT and RLVR stages import reusable modules from
`training/`; their task-specific behavior comes from the configured `task`.
The bundled profiles select `nemotron.recipes.parse.dental.task.DentalTask`.
Changing `data_dir` alone retains that dental task.

Implement your own importable task class or factory and select it with
`task=my_domain.task.DocumentTask`. The [training guide](training/README.md)
describes the hooks, input/output contracts and reward environment wiring.
SFT can use `task=null` for native validation loss without additional task
monitoring. RLVR requires a task and its reward environment.

| Component | Reference implementation | What to adapt |
| --- | --- | --- |
| Data preparation | [Data stage](stage0_data/data_prep.py), [records/splits](dental/data.py), [renderer](dental/render.py) | Develop or load your domain's images, prompts, full targets and scoring references. Define context groups and independent splits. |
| Dataset handoff | [Identity and artifact helpers](common.py), [manifest reader](training/data.py) | Write `train.jsonl`, `monitor.jsonl` and `test.jsonl`; preserve dataset identity checks when replacing the data producer. |
| SFT monitoring | [Shared trainer](training/sft.py), [collator](training/sft_data.py), [dental task](dental/task.py) | Reuse Parse image/prompt/target training. Implement your task's `monitor` callback with its own metrics, or use `task=null` for loss-only validation. |
| RL candidates and curriculum | [RLVR stage](stage2_rlvr/train.py), [dental task](dental/task.py), [selection](dental/curriculum.py) | Implement candidate creation and selection using your scoring criteria; exclude all fixed dataset contexts. |
| RLVR reward | [Dental reward environment](dental/environment.py), [shared data processor](training/rl_data.py), [shared trainer](training/rlvr.py) | Implement your domain's reward environment and select it through `task.environment_class`. The shared processor passes reference metadata without interpreting it. |
| Evaluation | [Dental prediction/scoring/summary](dental/evaluate.py), [metrics](dental/metrics.py), [eval stage](stage3_eval/evaluate.py) | Develop your own correctness criteria, component metrics and reports. Use them in task evaluation during RLVR and adapt the standalone evaluation example consistently. |

`nemotron parse data` and `nemotron parse eval` remain dental reference stages;
they do not accept the training stages' `task` override. Adapt these stages for
your domain alongside the task's monitoring, reward and evaluation hooks.

Prepare the three fixed manifests and dataset artifact through your adapted data
stage. The identity checks expect unique page IDs, disjoint context IDs across
fixed splits, and unchanged manifest/image bytes. Keep variants from the same
source family in one split. The shared training modules do not require a dental
`record`: define the labels consumed by your task and reward environment.
The dental curriculum recomputes context IDs from dental record fields; your
selection hook must enforce the corresponding exclusions for your own schema.

The SFT collator trains on `image`, `prompt` and `target`. Its token alignment
and masking are specific to this Parse architecture. The shared greedy generation
helper requires one prompt per monitor batch, and the RL processor checks each
row's prompt against `task.prompt_file`. Update prompts and scoring references
together when defining a new task.

Choose evaluation metrics before defining the RLVR reward. The optional
[document scorers](document_rewards/README.md) cover transcription, boxes,
tables, charts and completion. They are reference building blocks: customers
must choose extraction/matching rules, tolerances, thresholds and reward weights
for their use cases. Connect these decisions to both saved-answer evaluation
and the reward environment, and report individual metrics alongside any scalar
reward. The dental reward does not establish correctness for other domains.

After implementing the data and task components, use the shared SFT/RLVR stages
and artifact handoffs. Check complete target lengths, image dimensions and
generation limits for your documents. Start with a small run and saved-checkpoint
reload, then compare SFT and RLVR on the same independent held-out pages using
your domain's evaluation. The dental settings and results below illustrate the
reference example.

## SFT, RLVR and artifact flow

With the dental example defaults, SFT runs 2,852 updates with global/local batch
size 1, seed 42 and the configured learning-rate schedule. RLVR generates 512
independent candidate contexts, samples four answers per candidate and selects 256 contexts. Candidate images
and selected manifests live in the RLVR output's `curriculum/` directory, leaving
the fixed dataset unchanged. GRPO runs 128 updates, with 16 prompts and four
responses per update. Training configurations are in `stage1_sft/config/automodel.yaml`
and `stage2_rlvr/config/grpo.yaml`.

Each stage writes `config.json`, `stage.json` and an explicit `artifact.json`
handoff. Nemotron Artifact metadata is saved in the stage output bundle, with
`payload_path` pointing to the dataset, checkpoint or evaluation output. Checkpoint
contents remain unchanged by artifact publication. Training also writes `training-config.json`, `schedule.json`,
`history.jsonl`, predictions and checkpoints. RLVR retains every sampled answer
under `mining/`, `before/` and `after/`, plus `evaluation.json`. No external
tracking service is required.

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
To evaluate a saved checkpoint separately:

```bash
"$PARSE_PYTHON" -m nemotron parse eval \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data" \
  model_artifact="$PARSE_RUN/rlvr/artifact.json" output_dir="$PARSE_RUN/eval"
```

The complete dental example was validated on one H200, including SFT, RLVR and
evaluation of the saved checkpoint. Standalone evaluation produced the same
results as evaluation within the RLVR stage.

| Checkpoint | Sampled pass@1 | Mean text CER ↓ |
| --- | ---: | ---: |
| SFT | 59.03% | 0.1954% |
| RLVR | 76.76% | 0.1298% |

Both checkpoints were evaluated on the same 256 held-out pages with eight
sampled answers per page. RLVR improved both metrics in this run. Quality and
run-to-run variation depend on the data, training settings and runtime; measure
adaptation quality on your domain's held-out data. Preserve runtime records and
report both accuracy and error metrics when comparing checkpoints.
The baseline preview uses greedy monitoring; it is not a matched base/SFT/RLVR
benchmark on the held-out set.

## Verification and attribution

Run the CPU contract tests from the repository root:

```bash
PYTHONPATH=src python -m pytest tests/recipes/parse
```

The CPU tests cover reward/EOS behavior, split exclusions, configuration,
artifact handoff, CLI dispatch and runtime isolation. The dental example's
end-to-end validation also checked checkpoint reload, dataset identity,
training histories and recomputed evaluation metrics.

Third-party attribution is retained in the
[font license](dental/fonts/LICENSE.txt) and
[document scorer notice](document_rewards/NOTICE).
