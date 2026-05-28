# Hardware — What to Ask, How It Affects Config

## Questions to ask the user
1. What GPUs? (H100, A100, H200, B200)
2. How much memory per GPU? (40GB, 80GB, 96GB, 192GB)
3. How many nodes? How many GPUs per node?
4. Interconnect? (NVLink, NVSwitch, InfiniBand, RoCE)
5. Slurm? Local? Cloud?

## How hardware maps to config

### GPU memory → parallelism + batch size
| GPU | Memory | Typical config (Nano3 SFT) | Typical config (Super3 SFT) |
|-----|--------|---------------------------|----------------------------|
| A100 40GB | 40GB | tp=4, pp=2, cp=1, micro_bs=1, activation_ckpt=ON | Not recommended |
| A100 80GB | 80GB | tp=4, pp=1, cp=2, micro_bs=1 | tp=8, pp=4, cp=1, 4+ nodes |
| H100 80GB | 80GB | tp=4, pp=1, cp=2, micro_bs=2 | tp=8, pp=4, cp=1, 4+ nodes |
| H200 141GB | 141GB | tp=4, pp=1, cp=2, micro_bs=4 | tp=8, pp=2, cp=2 |

### Number of GPUs → what's feasible
| GPUs | Can train |
|------|-----------|
| 1–4 | AutoModel SFT only. LoRA recommended. |
| 8 (1 node) | Nano3 SFT via Megatron-Bridge. |
| 16–32 (2–4 nodes) | Super3 SFT, Nano3 RL (GRPO). |
| 64+ | Super3 RL, pretraining. |

### Interconnect → communication strategy
- **NVLink/NVSwitch**: TP within node is fast. Use TP up to gpus_per_node.
- **InfiniBand**: PP and DP across nodes. Enable communication overlap.
- **Ethernet/RoCE**: Avoid large TP across nodes. Prefer PP + DP.

## The agent should NOT memorize all of this
Read this file once, use it to ask the right questions, then let the
step manifest's `[[strategies]]` route to specific perf-technique skills
based on the answers.
