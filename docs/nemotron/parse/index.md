# Nemotron Parse Fine-Tuning

Fine-tune Nemotron 3.5 Parse on synthetic dental forms, then train with verifiable
rewards for document content. The v1 workflow preserves the dental
reference experiment and provides local CLI stages for data generation, SFT,
RLVR and evaluation, with a dedicated training runtime.

Follow the [recipe guide](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/parse/README.md) for
setup, commands, data contracts, artifact handoffs, hardware and metric definitions.
On a cluster, execute those commands inside a Slurm allocation.

The initial hardware target is one H200. General document SDG, generic reward
mixtures and other GPU profiles remain future extensions.
