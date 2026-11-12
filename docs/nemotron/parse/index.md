# Nemotron Parse Model Adaptation and Evaluation

Adapt Nemotron 3.5 Parse to document extraction tasks with supervised fine-tuning
(SFT), reinforcement learning with verifiable rewards (RLVR), and checkpoint
evaluation. The recipe provides these stages through the Nemotron CLI, with
explicit data and model artifacts and a dedicated training runtime.

Dental forms provide reference implementations for data preparation, monitoring,
rewards and evaluation. Customers should develop each of these for their own
use cases, including the correctness criteria used for rewards and evaluation.
The shared SFT and RLVR modules handle Parse model integration and training; a
configurable task adapter supplies the domain-specific behavior.

Follow the [recipe guide](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/parse/README.md) for
setup, commands, data contracts, artifact handoffs, hardware and metric definitions.
The [domain adaptation guide](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/parse/README.md#adapt-the-example-to-your-domain)
identifies the example components to adapt. The executable v1 defaults include
dental generation, monitoring, rewards and evaluation. Configure your own task
for SFT/RLVR and adapt the reference data and standalone evaluation stages for
another domain. See the [training task contract](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/parse/training/README.md).

The complete dental example has been validated on one H200. Additional domain
examples and other GPU profiles remain future extensions.
