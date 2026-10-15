# RLVR (Stage 2.1)

RLVR improves the general capabilities of the SFT model and provides the shared trunk for every specialist teacher. Samples from each environment are routed to the matching verifier or reward model, so RLVR and RLHF-style rewards are applied jointly in one run.

:::{admonition} Pending open-source release
:class: warning

The RLVR data used for Super 3.5 is not yet published, and neither is the report's length-penalized GenRM variant (NeMo-RL's public recipe uses [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM)). The training stack is public: NeMo-RL `super-v3.5-posttraining` @ `f299847f1` (NeMo Gym `e24db4e6`).
:::

---

## Data and Environments

Compared with Nemotron 3 Ultra, the report refreshes the RL data with more recent collections and runs reward profiling before training. The reference-run blend (47 NeMo Gym environment configs, listed in the stage README) spans math, code, STEM, instruction following, structured outputs, safety, chat (GenRM), long context, Reasoning Gym, and tool use. The report highlights two new environments:

| Environment | Description |
|-------------|-------------|
| **Competitive programming** | 2,847 training and 372 validation problems from past Olympiad-level contests, each with a validated executable evaluator. Generated C++ solutions are compiled and run against problem-specific tests; reward is 1 for full credit, 0 otherwise. |
| **Finance agent** | A NeMo Gym environment ([`finance_sec_search`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/finance_sec_search)) with SEC filing search, document parsing, evidence retrieval, and source-supported submission tools. The same tools are used for trajectory generation, SFT, RL, and evaluation. Questions come from two synthetic-data tracks: document-grounded questions over 10-K / 10-Q / 8-K filings of S&P 500 companies, and an agentic track that targets coverage gaps against the Vals.ai Finance Agent taxonomy. |

### Medium-effort reasoning

Per the report, about 4% of RL training prompts are in the **medium-effort** reasoning mode and cover math, STEM, and coding, with length-based reward adjustments. The effect generalizes beyond those domains: averaged over the Artificial Analysis Intelligence Index benchmarks, medium-effort uses about 2.2× fewer tokens than regular mode for about a 9% accuracy drop.

In the RLVR config, `env.nemo_gym.effort_levels.low_weight: 0.1` marks 10% of sampled prompts with `{reasoning effort: efficient}` and penalizes those responses past 15,000 tokens. This is the reference-run setting; the report does not break its 4% figure down by stage.

---

## Algorithm

RLVR uses the asynchronous GRPO algorithm of Nemotron 3 Ultra. The report adds two optimizations against train–inference mismatch — **router replay** (replaying MoE routing decisions recorded during generation) and an **FP32 LM head** — which together cut the KL error between the training and inference engines by more than an order of magnitude.

| Parameter | Report | Recipe (`config/default.yaml`) |
|-----------|--------|--------------------------------|
| Prompts × generations per step | 512 × 16 (batch 8,192) | 512 × 16 |
| Max generation length | 80K, later 128K | 80K (81,920) throughout |
| Learning rate | — | 4e-6 |
| Max trajectory age | 1 | 1 |
| Router replay | On | **Off** (simplification) |
| LM head precision | FP32 | FP32 LM head in vLLM generation (`NRL_VLLM_FP32_LM_HEAD=1`) |
| MTP | — | 5 repeated detached heads, coefficient 0.3; no speculative decoding |
| GenRM length penalties | — | 0.12 / 0.12 |
| Train / generate parallelism | — | TP4 / CP2 / EP16 training; vLLM TP4 / EP4 generation |

The config follows the reference run that produced the selected **step-120** checkpoint, with three simplifications: it uses the final dataset version from step 0 (the reference run's first 34 steps used an earlier blend), it disables router replay throughout (step 120 came from a router-replay fork created at step 101), and it leaves out that run's later context-length and length-penalty changes. It is a port of the training inputs, not a replay of the historical run.

---

## Configuration

| Setting | Value |
|---------|-------|
| Starting checkpoint | `run.model: super35-sft-model:latest` |
| Data | `run.data: super35-rl-rlvr-data:latest` |
| Output | `super35-rl-rlvr-model` |
| Nodes | 144 GB200 nodes allocated (32 training + 112 generation); the reference run used 26 more for external judges (170 total) |
| Selected checkpoint | Step 120 |
| External servers | GenRM (chat reward; `GENRM_BASE_URL`, `GENRM_MODEL`), safety judge (`SAFETY_BASE_URL`, default model `nvidia/Nemotron-Content-Safety-Reasoning-4B`), nl2bash judge (`NL2BASH_BASE_URL`, default model `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`) |

The config is a flattened NeMo-RL master config. Paths that point at cluster-local resources are read from environment variables documented in `src/nemotron/recipes/super35/stage2_rl/stage1_rlvr/README.md`.

## Usage

```bash
# Import data (default artifact name: super35-rl-rlvr-data) and run
uv run nemotron super35 data import rl /data/rlvr/train.jsonl --val /data/rlvr/val.jsonl
uv run nemotron super35 rl rlvr --run YOUR-CLUSTER

# Start from a specific SFT artifact version
uv run nemotron super35 rl rlvr --run YOUR-CLUSTER run.model=super35-sft-model:v2

# Stop at the selected step (the config has no step limit)
uv run nemotron super35 rl rlvr --run YOUR-CLUSTER grpo.max_num_steps=120

# Preview the resolved config
uv run nemotron super35 rl rlvr --dry-run
```

The step-120 checkpoint is the starting point for every teacher except RLHF teacher 1, and the student (and self-anchor teacher) of MOPD phase 1.

## Next Steps

Train the [specialist teachers](teachers.md).

## Reference

- Nemotron 3.5 tech report, §3.2 (Reinforcement Learning), §3.9 (Reasoning Efficiency and Control)
- [RL overview](index.md)
- **Recipe Source**: `src/nemotron/recipes/super35/stage2_rl/stage1_rlvr/`
- [Back to Overview](../README.md)
