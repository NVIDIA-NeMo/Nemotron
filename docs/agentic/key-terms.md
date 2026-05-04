# Key Terms and Concepts

Building agentic workflows using this software requires understanding these terms and concepts.

## A

(artifact-type)=
Artifact type
: An artifact type names a class of data or checkpoint passed between _steps_, such as `training_jsonl`, `checkpoint_hf`, `packed_parquet`, and `translated_jsonl`.
By standardizing the artifacts, you can maintain the interface contract between the steps.
Types are centralized in `src/nemotron/steps/types.toml` and summarized under [artifact types](../customize/steps/types.md).

## P

(pattern)=
Pattern
: A pattern is cross-cutting guidance for composing _steps_ such as when to run eval, how to choose translation backends, packing and tokenizer checks, and so on.
On disk, patterns are in the `src/nemotron/steps/patterns/` directory and are published under [patterns](../customize/patterns/index.md).

(pipeline)=
Pipeline
: A pipeline is the agentic workflow that is implemented as an ordered composition of stages.
A pipeline defines the multi-step workflow from inputs, such as raw or filtered JSONL, to outputs like checkpoints, eval artifacts, or exports.
Generated scaffolds describe this in `README`, wire stages from a root **`pipeline.py`**, and may record a canonical graph in **`.generated/pipeline.toml`**.

  This is **not** the same as:

  - **Pipeline parallelism** in Megatron (`pipeline_model_parallel_size`, and related knobs), or
  - The **NeMo RunSpec** module `nemo_runspec.pipeline`, which schedules jobs for recipe commands.

## S

(stage)=
Stage
: A stage is one unit of work in a project.
On disk, a stage is a directory in the `stages/` directory with a thin `run.py`, configuration files, and an entry for the project CLI.
Each stage is instantiated from a step ID, such as `sft/megatron_bridge`, plus user-specific settings.
This meaning of stage is not the same as the training recipies stages such as the Nano3 SFT stage.

(step)=
Step
: A step is a reusable building block.
A step has typed inputs and outputs with a `step.toml` manifest and an optional implementation in `step.py`.
Steps are grouped by concern, such as convert, curate, prep, sft, translate, eval.
The canonical list is in the [step library](../customize/steps/index.md).
Some steps are exposed as `nemotron steps` subcommands of the CLI.
