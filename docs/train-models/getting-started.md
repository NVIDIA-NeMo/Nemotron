# Getting Started with Training Steps

This page walks through one supervised fine tuning (SFT) run on DGX Cloud Lepton using the *tiny* configuration.
The tiny configuration lives in `src/nemotron/steps/sft/automodel/config/tiny.yaml` and is meant for short validation before you scale work.
The goal is to validate end-to-end execution, NeMo Run, and your environment profile on real multi-node hardware.

## Prerequisites

- You need access to DGX Cloud Lepton with GPU nodes.
  This path assumes two nodes with eight A100 80 GB GPUs per node, matching the `run.env` block in `src/nemotron/steps/sft/automodel/config/tiny.yaml`.
- You set the following environment variables:
  - `HF_TOKEN`
  - `WANDB_API_KEY`
  - `NVIDIA_API_KEY`
- You ran `lep login` after synchronizing dependencies and are logged into Lepton.

The preceding list applies to the steps on this page.
Refer to [](./index.md#limitations-and-restrictions) for information about supported environments.

## Procedure

1. Clone the repository, if you haven't already:

   ```console
   $ git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. Set the dependencies:

   ```console
   $ uv sync
   ```

1. Create an `env.toml` at the root of the repository like the following example:

   ```{literalinclude} _snippets/input/env.tmpl
   :language: text
   ```

   ```{dropdown} Summary of the Config File

   The `[lepton_base]` table defines cluster fundamentals that every profile inherits: the executor, the base container image, the node group, shared-storage paths, the Ray runtime version, the shared-memory size, the Python package extras the Nemotron CLI needs, the cluster mount, and an `env_vars` block whose `${oc.env:VAR,''}` entries pull credentials from your shell at submit time.
   The `[lepton_sft_automodel]` table extends the base and adds the AutoModel container image, the resource shape needed for full-parameter SFT, the node count, and the additional Python package extras the AutoModel runtime expects.

   Contact your cluster administrator for the values that replace the placeholders.

   - `<lepton-node-group>`: The Lepton node group identifier for the cluster you have access to.
   - `<your-username>`: A directory you own on the shared mount where NeMo Run records each experiment.
   - `<lepton-fileset-alias>`: The alias of the Lepton storage fileset that each container mounts.
   - `<lepton-mount-source-path>`: The host path the fileset exposes.
   - `<project>`: The Weights & Biases project name the run reports to.
   ```

   Export `HF_TOKEN`, `WANDB_API_KEY`, and `NVIDIA_API_KEY` in your shell before submitting; the env file pulls them in without writing the values to disk.

   If you would rather generate a complete env file with every Nemotron training profile pre-wired, run the bundled environment profile generator instead of writing the file by hand.

   ```console
   $ uv run nemotron steps run env/env_toml -c lepton output_path=env.toml force=true
   ```

   The generator emits every canonical profile the training steps expect, including data-prep, SFT, PEFT, RL, and pretrain variants.

1. View the step manifest and run specification:

   ```console
   $ uv run nemotron steps show sft/automodel
   ```

   ````{dropdown} Example Output
   :icon: code-square

   ```{literalinclude} _snippets/output/gs-show.txt
   ```
   ````

1. Compile the job against your Lepton profile without submitting it.
   The profile name `lepton_sft_automodel` must match a table in your root `env.toml`.

   ```console
   $ uv run nemotron steps run sft/automodel --config tiny --run lepton_sft_automodel --dry-run
   ```

   ````{dropdown} Partial Output
   ```text
   Compiled Configuration

   ╭─────────────────────────────────────────── run ───────────────────────────────────────────╮
   │ env:                                                                                      │
   │   nodes: 2                                                                                │
   │   gpus_per_node: 8                                                                        │
   │   nprocs_per_node: 8                                                                      │
   │   executor: lepton                                                                        │
   │   container_image: nvcr.io/nvidia/nemo-automodel:26.04                                    │
   │   node_group: az-sat-lepton-001                                                           │
   │   resource_shape: gpu.8xa100-80gb                                                         │
   │   remote_job_dir: /mnt/lustre-shared/user/nemotron/.nemotron-jobs
   ...
   ```
   ````

1. Submit the sample SFT job:

   ```console
   $ uv run nemotron steps run sft/automodel -c tiny -r lepton_sft_automodel
   ```

The sample `tiny` config sets small training and validation splits.
To specify the output path for checkpoints, set `SFT_OUTPUT_DIR` before running or specify the `checkpoint.checkpoint_dir` CLI override.

## Success Checks

- The command `nemotron steps show <step_id>` lists `consumes` and `produces` artifact types. Those types must line up with your pipeline when you chain steps.
- A finished sample run leaves logs and job metadata where NeMo Run is configured to write them. See [Execution through NeMo Run](../nemo_runspec/nemo-run.md) for experiment layout.
- If you change tokenizer, template, or sequence length, keep them consistent across every step that touches the same model line. The [Artifact Graph](explanation/artifact-graph.md) page explains why consistency matters.

## Next Steps

- Follow [Run SFT with AutoModel on Custom Data](how-to/run-sft-automodel.md) when you need to point `tiny.yaml` at your own data or change the base model.
- Read [Choose an SFT Backend](how-to/choose-sft-backend.md) when you need Megatron Bridge instead of AutoModel.
