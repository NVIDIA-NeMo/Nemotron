# Nemotron Parse Model Adaptation and Evaluation

Adapt Nemotron 3.5 Parse to document extraction tasks with supervised fine-tuning
(SFT), reinforcement learning with verifiable rewards (RLVR), and checkpoint
evaluation. The recipe provides these stages through the Nemotron CLI, with
explicit data and model artifacts and a dedicated training runtime.
Default evaluation follows the training artifacts to compare the configured base
model, SFT checkpoint and RLVR checkpoint on the same held-out pages, and reports
the uplift at each stage.

Dental forms provide reference implementations for data preparation, monitoring,
rewards and evaluation. Customers should develop each of these for their own
use cases, including the correctness criteria used for rewards and evaluation.
The shared SFT and RLVR modules handle Parse model integration and training; a
configurable task adapter supplies the domain-specific behavior.

## Run the Reference Workflow

1. [Set up the Parse runtime](runtime.md) on Linux with your GPU visible.
2. [Run data preparation, SFT, RLVR and evaluation](workflow.md#run-the-dental-worked-example).
3. [Compare the base, SFT and RLVR models](workflow.md#evaluation-and-dental-example-results)
   on the same held-out pages and inspect the saved answers and uplift report.

The workflow guide covers the [page manifest contract](workflow.md#data-preparation-example),
[training settings and artifact handoffs](workflow.md#sft-rlvr-and-artifact-flow),
and the dental reward and evaluation definitions.

The dental reference run used one H200 per stage, with SFT batch size 4 and 2,852
optimizer updates followed by 128 RLVR updates. See the
[results and evaluation protocol](workflow.md#evaluation-and-dental-example-results).
Additional domain examples and other GPU profiles remain future extensions.

## Adapt the Recipe to Your Domain

Follow the [domain adaptation guide](adaptation.md) to prepare your own dataset,
define correctness criteria and implement the [training task hooks](adaptation.md#task-contract).
The executable defaults include dental generation, monitoring, rewards and
evaluation. Configure your own task for SFT/RLVR and adapt the reference data
and standalone evaluation stages for another domain.

Use the optional [document evaluation and reward building blocks](rewards.md)
for transcription, layout, tables, charts and completion. Choose matching rules,
thresholds and reward composition for your own use case.

```{toctree}
:hidden:

Runtime Setup <runtime>
Training and Evaluation Workflow <workflow>
Domain Adaptation <adaptation>
Evaluation and Reward Building Blocks <rewards>
Third-Party Attribution <attribution>
```
