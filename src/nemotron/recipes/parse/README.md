# Nemotron Parse model adaptation and evaluation

Use this recipe as a starting point to adapt
[Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) to your
document domain through supervised fine-tuning (SFT), reinforcement learning
with verifiable rewards (RLVR), and held-out evaluation. It provides stage
commands, a model adapter, an isolated runtime and explicit dataset/checkpoint
handoffs.

**Dental forms are the worked data preparation example.** The example shows how
to generate fictional records, render page images and complete Parse targets,
and create separate training, monitoring and test sets. Run it to understand
the workflow, then [adapt the example to your domain](#adapt-the-example-to-your-domain)
using your own documents, labels and correctness criteria. The packaged defaults
also include dental monitoring, rewards and evaluation; the adaptation guide
identifies those components alongside data preparation.

| Command | Input | Output |
| --- | --- | --- |
| `nemotron parse data` | Dental example record/rendering rules and seed | Images, full-page Parse targets, train/monitor/test manifests and dataset identity |
| `nemotron parse sft` | Dataset artifact and pinned base model | Full-model SFT checkpoint and greedy monitor history |
| `nemotron parse rlvr` | Dataset and SFT model artifacts | Candidate mining, decoder GRPO checkpoint, matched sampled evaluation before/after |
| `nemotron parse eval` | Dataset and a model artifact or checkpoint | Held-out answers and component metrics |

The model adapter supports the Parse architecture. Reuse its model integration,
training orchestration and artifact flow while adapting the example's data and
task logic. Optional [document component scorers](document_rewards/README.md)
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
profiles retain the reference scientific settings.

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
criteria for your task. Reuse the Parse model/runtime and the staged
data → SFT → RLVR → evaluation workflow. In this version, connecting a new
domain requires adapting example code as well as supplying data: changing
`data_dir` alone leaves dental monitors, candidate generation and scoring active.
There is no built-in domain selector or automatic reference conversion.

| Component | Reference implementation | What to adapt |
| --- | --- | --- |
| Data preparation | [Data stage](stage0_data/data_prep.py), [records/splits](dental/data.py), [renderer](dental/render.py) | Load or generate your own images, prompts, full targets and scoring references. Replace dental record generation and context grouping. |
| Dataset handoff | [Identity and artifact helpers](common.py) | Write `train.jsonl`, `monitor.jsonl` and `test.jsonl`; preserve the dataset artifact and hash validation when replacing the data producer. |
| SFT | [Training stage](stage1_sft/train.py), [collator](dental/sft_data.py), [monitoring](dental/sft.py) | Reuse image/prompt/target training for this Parse architecture. Replace dental monitor metrics and any reference fields they read. |
| RL candidates and curriculum | [RLVR stage](stage2_rlvr/train.py), [selection](dental/curriculum.py) | Supply candidates from your domain; adapt context grouping, split exclusion and difficulty selection to your scoring criteria. The default stage generates fresh dental candidates internally. |
| RLVR reward | [Reward environment](dental/environment.py), [data processor](dental/rl_data.py), [trainer wiring](dental/rlvr.py) | Pass task references in scoring metadata, implement your reward, and connect the environment/data processor to training. |
| Evaluation | [Prediction and scoring](dental/evaluate.py), [metrics](dental/metrics.py), [summary](common.py), [eval stage](stage3_eval/evaluate.py) | Retain checkpoint loading, saved answers and matched evaluation; replace dental correctness/component metrics and their summary fields. |

Prepare the three fixed manifests and dataset artifact through your adapted data
stage, then pass its directory to SFT, RLVR and evaluation. The identity checks
expect unique page IDs, disjoint context IDs across fixed splits, and unchanged
manifest/image bytes. Keep variants from the same source family in one split.
The dental `record` schema is a reference implementation; new domains should
define their own labels and update the code that consumes them together.
Curriculum selection also recomputes context IDs from dental record fields;
adapt that grouping function along with the IDs stored in your manifests.

The SFT collator trains on `image`, `prompt` and `target`. Its token alignment
and masking are specific to this Parse architecture and can be reused without
retaining dental field semantics. The surrounding SFT monitoring path still
reads dental references, so adapt that path even if you only intend to run SFT.
Keep prompts consistent with those references: the current SFT monitor requires
one shared prompt within each batch, and the RL data processor checks prompts
against the configured [prompt file](dental/prompt.txt). If your task changes
the prompt, update these consumers and the RL configuration together.

Choose evaluation metrics before defining the RLVR reward. The optional
[document scorers](document_rewards/README.md) cover transcription, boxes,
tables, charts and completion. Extract and match the relevant components from
predictions and references before using those scorers. Connect your chosen
metrics to both saved-answer evaluation and the reward environment, and keep
individual metrics visible when combining them into a scalar reward.

After adapting the data and task components, use the same SFT/RLVR/eval stage
workflow and artifact handoffs. Check complete target lengths, image dimensions
and generation limits for your documents. First validate a small run and saved
checkpoint reload, then evaluate SFT and RLVR on the same independently held-out
pages. The dental defaults, sample counts and results below are a reference
example; validate training settings and quality for your own domain.

## SFT, RLVR and artifact flow

With the dental example defaults, SFT runs 2,852 updates with global/local batch
size 1, seed 42 and the reference learning-rate schedule. RLVR generates 512
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

| Workflow | SFT pass@1 | RLVR pass@1 | SFT text CER | RLVR text CER |
| --- | ---: | ---: | ---: | ---: |
| Original notebooks | 82.67% | 86.67% | 0.1167% | 0.1418% |
| Nemotron CLI | 59.03% | 76.76% | 0.1954% | 0.1298% |

Each dental comparison uses 256 held-out pages and eight sampled answers per page.
These results validate the worked example; adaptation quality in another domain
must be measured on that domain's held-out data.
The CLI run improved both metrics within the run, but its absolute pass@1 was
lower than the notebook result. The cause remains unresolved. Execution success
does not establish numerical reproduction; fixed seeds alone do not establish it
either. Preserve the runtime records and report both accuracy and error metrics.
The baseline preview uses greedy monitoring; it is not a matched base/SFT/RLVR
benchmark on the held-out set.

## Source and verification

The scientific code and framework patches come from Parse revision
`2c2be873c05d904de38c80e55ef5ac754aea180d`. File hashes and relocation changes
are recorded in `dental/PROVENANCE.json`. Font and optional scorer notices are
retained in their respective directories.

```bash
PYTHONPATH=src python -m pytest tests/recipes/parse
```

The CPU tests cover reward/EOS behavior, split exclusions, configuration,
artifact handoff, CLI dispatch and runtime isolation. The dental example's
end-to-end validation also checked checkpoint reload, dataset identity,
training histories and recomputed evaluation metrics.
