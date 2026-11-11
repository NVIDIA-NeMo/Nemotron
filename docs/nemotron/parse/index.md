# Nemotron Parse Model Adaptation and Evaluation

Adapt Nemotron 3.5 Parse to document extraction tasks with supervised fine-tuning
(SFT), reinforcement learning with verifiable rewards (RLVR), and checkpoint
evaluation. The recipe provides these stages through the Nemotron CLI, with
explicit data and model artifacts and a dedicated training runtime.

Dental forms are the worked data preparation example. Follow the example to
create paired page images and full-page Parse targets, define training and
held-out splits, and connect task references to rewards and evaluation. Use it
as a starting point for developing data and scoring for your own domain.

Follow the [recipe guide](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/parse/README.md) for
setup, commands, data contracts, artifact handoffs, hardware and metric definitions.
The [domain adaptation guide](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/parse/README.md#adapt-the-example-to-your-domain)
identifies the example components to adapt. The executable v1 defaults include
dental generation, monitoring, rewards and evaluation; another domain requires
corresponding data and task code changes.
On a cluster, execute those commands inside a Slurm allocation.

The complete dental example has been validated on one H200. Additional domain
examples, configurable task interfaces and other GPU profiles remain future
extensions.
