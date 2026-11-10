# Nemotron Parse dental fine-tuning

Adapt [Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse)
to fictional dental forms with synthetic data, supervised fine-tuning and
reinforcement learning with verifiable rewards. This v1 recipe preserves the
reference dental experiment and exposes its stages through the Nemotron CLI.

| Command | Input | Output |
| --- | --- | --- |
| `nemotron parse data` | Fixed record/rendering rules and seed | Images, full-page Parse targets, train/monitor/test manifests and dataset identity |
| `nemotron parse sft` | Dataset artifact and pinned base model | Full-model SFT checkpoint and greedy monitor history |
| `nemotron parse rlvr` | Dataset and SFT model artifacts | Candidate mining, decoder GRPO checkpoint, matched sampled evaluation before/after |
| `nemotron parse eval` | Dataset and a model artifact or checkpoint | Held-out answers and component metrics |

The model adapter supports this Parse architecture; dental semantics live in
the renderer, records and reward. General document generation and adaptation to
other domains are follow-on work. Optional [general document scorers](document_rewards/README.md)
are examples, not the dental training objective.

## Runtime

Use Linux and one H200 with a driver compatible with the selected CUDA runtime.
The reference workflow passed on H200 with driver 580.126.20, Python 3.13.16,
Torch 2.14.0+cu132, Transformers 5.16.1 and vLLM 0.33.0. A full reference run
took approximately 86 minutes. Its sampled GPU-memory peaks were 23.12 GiB
for SFT and 36.32 GiB for RLVR; these are observations, not minimum requirements.
H100/A100 compatibility has not been established by that run.

Parse uses an isolated environment because the main Nemotron environment has
different Torch/Transformers requirements. The installer resolves the complete
dependency set, checks it, applies bundled patches to pinned NeMo Automodel and
NeMo RL sources, and records the resolved packages. It does not modify the
main Nemotron environment. The default CUDA backend is the tested `cu132`;
`--torch-backend auto` is available but may select another build. The current
vLLM wheel requires CUDA 13, so selecting CUDA 12 Torch alone is insufficient.

From the repository root, on a compute host:

```bash
python3 -m pip install uv==0.13.1
export PYTHONPATH="$PWD/src"
PARSE_RUNTIME="$PWD/.nemotron/parse-runtime"
python3 -m nemotron.recipes.parse.runtime.setup --runtime-dir "$PARSE_RUNTIME"
```

On a Slurm cluster, run installation and all stages inside an allocation.
The observed full run allocated one H200, eight CPUs and 128 GiB host memory.
Use a short node-local `TMPDIR`/`RAY_TMPDIR` when the project path would exceed
Ray's Unix-socket path limit. A system CUDA compiler is not needed for sampling:
the recipe supplies `VLLM_USE_FLASHINFER_SAMPLER=0` to stage and Ray processes.

`$PARSE_RUNTIME/runtime.json` and `$PARSE_RUNTIME/environment.txt` record the
installation. Reusing that directory verifies its configuration and package
freeze; changed inputs require a new runtime directory. Broad dependency ranges
still allow different fresh resolutions, so retain these records with results.

## Run the complete recipe

The runtime includes the lightweight source CLI. These commands use the same
`nemotron parse` command implementation as an installed Nemotron CLI, without
installing conflicting root dependencies into the training environment.

```bash
PARSE_RUN="$PWD/.nemotron/parse"
PARSE_PYTHON="$PARSE_RUNTIME/venv/bin/python"

"$PARSE_PYTHON" -m nemotron parse data \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data"

"$PARSE_PYTHON" -m nemotron parse sft \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data" \
  output_dir="$PARSE_RUN/sft"

"$PARSE_PYTHON" -m nemotron parse rlvr \
  runtime_dir="$PARSE_RUNTIME" data_dir="$PARSE_RUN/data" \
  sft_output="$PARSE_RUN/sft" output_dir="$PARSE_RUN/rlvr"
```

An existing Nemotron installation can use `nemotron parse ...` with the same
arguments. Every command launches a separate runtime process, releasing GPU
resources between stages. Native NeMo-Run `--run`/`--batch` submission is not
part of this v1; submit the local commands through your cluster scheduler.

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

## Data, training and artifacts

The data stage generates 256 training pages from 32 contexts, 32 independent
monitor pages and 256 independent held-out pages. Each JSONL record includes
an image reference, prompt, complete Parse target, structured dental record,
context ID and geometry. Patient context exclusions prevent split leakage.
Dataset identity includes manifest and image hashes and is checked before use.

SFT runs 2,852 updates with global/local batch size 1, seed 42 and the reference
learning-rate schedule. RLVR generates 512 independent candidate contexts,
samples four answers per candidate and selects 256 contexts. Candidate images
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

The dental reward is binary: exact tooth/procedure/surface content and zero
normalized visible-text CER. Reported sampled success also requires natural EOS.
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

Exact quality values vary across runs. The unrefactored reference H200 run
improved pass@1 from 82.67% to 86.67%, while CER changed from 0.1167% to 0.1418%.
These are source-workflow observations, not yet results from these entry points.
Fixed seeds do not establish numerical reproducibility across environments.
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

CPU checks cover reward/EOS behavior, split exclusions, configuration, artifact
handoff, CLI dispatch and runtime isolation. GPU acceptance must exercise the
complete default sequence, actual checkpoint reload and matched evaluation.
