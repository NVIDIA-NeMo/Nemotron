# Nemotron 3.5 Lightning Recipes for DGX Station

These end-to-end recipes customize
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on DGX Station GB300 systems. Each guide uses one B300 GPU per Station and
covers system preparation, shared storage, containers, data, distributed
launch, monitoring, and troubleshooting.

## Available recipes

| Recipe | Framework | Training workflow | Configuration |
| --- | --- | --- | --- |
| [Supervised fine-tuning (SFT)](sft.md) | NeMo AutoModel | Full-weight SFT on multi-turn agentic tool-use data | [`sft_lightning35_station.yaml`](sft_lightning35_station.yaml) |
| [Group Relative Policy Optimization (GRPO)](grpo.md) | NeMo RL and NeMo Gym | Full-weight reinforcement learning with an ARC-AGI verifier | [`grpo_lightning35_station.yaml`](grpo_lightning35_station.yaml) |

## Shared topology

Both recipes assume:

- Two DGX Station GB300 systems connected and validated for distributed GPU
  workloads.
- One Station designated as the **head** and the other as the **worker** for
  the entire run.
- One B300 GPU used on each Station.
- A high-speed node-to-node network with RDMA available to the training
  containers.
- Shared storage mounted at the same host path on both systems and exposed as
  `/shared` inside each container.

The individual guides are self-contained. Follow one from its prerequisites
through checkpoint creation; you do not need to complete one recipe before
using the other.

## Choose a recipe

Use [SFT](sft.md) when you have examples of the responses and tool behavior
you want the model to learn. The included recipe trains on the
`interactive_agent` subset of
[Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
with NeMo AutoModel.

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
