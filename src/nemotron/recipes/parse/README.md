# Nemotron Parse model adaptation and evaluation

This directory contains the `nemotron parse` implementation and stage configs.
The [Parse model adaptation and evaluation guide](../../../../docs/nemotron/parse/index.md)
is maintained in `docs/` and covers local uv and container setup, data preparation, SFT, RLVR,
evaluation, and domain adaptation.

- [Local uv and container setup](../../../../docs/nemotron/parse/runtime.md)
- [Stage environment reference](../../../../docs/nemotron/parse/environments.md)
- [Training and evaluation workflow](../../../../docs/nemotron/parse/workflow.md)
- [Domain adaptation and task contract](../../../../docs/nemotron/parse/adaptation.md)
- [Document evaluation and reward building blocks](../../../../docs/nemotron/parse/rewards.md)

The shared SFT and RLVR implementation lives in [training/](training/), with
domain behavior supplied by a configurable task adapter. The [dental/](dental/)
data preparation, monitoring, rewards, and evaluation are reference examples;
develop these components and their correctness criteria for your own domain.

Follow the setup guide before running stages, and keep dataset and checkpoint
paths consistent throughout the workflow. Each `stage*/config/` directory
contains its supported profiles; the [CLI implementation](../../cli/commands/parse/)
defines stage and execution-mode support.
