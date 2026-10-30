# Nemotron 3.5 Lightning Recipes for DGX Station

These end-to-end recipes customize
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on DGX Station GB300 systems. Each guide uses one B300 GPU per Station and
covers system preparation, persistent storage, containers, data, training,
monitoring, and troubleshooting.

## Available recipes

| Recipe | Framework | Hardware | Training workflow | Configuration |
| --- | --- | --- | --- | --- |
| [Supervised fine-tuning (SFT)](sft.md) | NeMo AutoModel | Two Stations | Full-weight SFT on multi-turn agentic tool-use data | [`sft_lightning35_station.yaml`](sft_lightning35_station.yaml) |
| [Low-rank adaptation (LoRA)](lora.md) | NeMo AutoModel | One Station | Parameter-efficient tuning on multi-turn agentic tool-use data | [`lora_lightning35_station.yaml`](lora_lightning35_station.yaml) |
| [Group Relative Policy Optimization (GRPO)](grpo.md) | NeMo RL and NeMo Gym | Two Stations | Full-weight reinforcement learning with an ARC-AGI verifier | [`grpo_lightning35_station.yaml`](grpo_lightning35_station.yaml) |

## Hardware and storage

All recipes assume current DGX Station software and use one B300 GPU per
Station. They mount persistent host storage at `/shared` inside their training
containers so that downloads, data, configuration, and checkpoints survive
container removal.

The [LoRA recipe](lora.md) runs entirely on one Station and can use a local
persistent directory. The [SFT](sft.md) and [GRPO](grpo.md) recipes use two
Stations, require shared storage visible at the same host path on both systems,
and designate fixed **head** and **worker** roles. Complete the two-Station
networking and RDMA validation before starting either distributed recipe.

The individual guides are self-contained. Follow one from its prerequisites
through checkpoint creation; you do not need to complete another recipe first.

## Choose a recipe

Use [SFT](sft.md) when you have examples of the responses and tool behavior
you want the model to learn. The included recipe trains on the
`interactive_agent` subset of
[Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
with NeMo AutoModel.

Use [LoRA](lora.md) when you have supervised examples but want to update a
small adapter instead of the full model, or when only one DGX Station is
available. It uses the same agentic dataset as SFT and produces an adapter that
is loaded alongside the original base model.

Use [GRPO](grpo.md) when behavior can be evaluated by a reward or verifier.
The included recipe generates responses with NeMo RL, scores them through a
NeMo Gym ARC-AGI environment, and updates the policy from those rewards.

These recipes showcase ways to customize Nemotron models on DGX Stations.
They do not reproduce the model's complete post-training program or published
evaluation results.

## Related resources

- [Nemotron 3.5 Lightning usage cookbook](../README.md)
- [Nemotron 3.5 Lightning training guide](../../../docs/nemotron/lightning35/README.md)
- [Connect Two DGX Stations for Distributed Workloads](https://build.nvidia.com/station/connect-two-stations/overview)
- [DGX Station software stack](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
