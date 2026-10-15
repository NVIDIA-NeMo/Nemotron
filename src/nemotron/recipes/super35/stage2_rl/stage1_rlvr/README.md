# Stage 1: Text-Only RLVR

Text-only multi-environment RLVR is the first RL stage for Nemotron 3.5 Super VL. It starts from the SFT checkpoint and trains on a broad NeMo-Gym environment blend. The blend mixes verifiable-reward environments with GenRM pairwise comparison (RLHF), so RLVR and RLHF run jointly in one stage.

The selected checkpoint is **step 120**. It feeds most of the rest of the RL pipeline:

- **MOPD phase 1** uses it as the student start and as the "general" teacher.
- **Every specialist teacher** starts from it, except RLHF teacher 1, which starts from SFT ([stage2_teachers](../stage2_teachers/README.md)).

## Data

The RLVR blend contains 47 Gym config files (see `env.nemo_gym.config_paths`), grouped roughly as follows:

| Domain | Environments |
|--------|--------------|
| **Math** | Math with LLM-judge fallback, Lean4 formal proofs (multi-turn), NVARC inductive/transductive puzzles |
| **Code** | Unit-test code generation, competitive-coding challenges (CCC, sandboxed), SciCode |
| **STEM / science** | MCQA, equivalence LLM judge, ether0 chemistry, RDKit chemistry agent |
| **Instruction following** | IF, inverse IF, multi-challenge, structured outputs (JSON/YAML/XML, v3, v4), free-form formatting, citation format / citation IF |
| **Chat / RLHF** | GenRM pairwise comparison (reasoning on and off), abstention |
| **Safety** | Jailbreak detection, over-refusal detection, indirect prompt injection |
| **Long context** | Long-context judge and rule-based equivalence |
| **Tool use / agentic** | Single-step tool use (incl. search / schema / SWE / Droid pivots), conversational tool-use simulation, calendar, workplace assistant, finance SEC search, NeMo-Skills tools, terminal multi-harness (several harnesses), SWE pivot, NL2Bash, Terminus judge |
| **Reasoning Gym** | Assorted reasoning tasks |

Medium-effort ("efficient") reasoning: the config sets `env.nemo_gym.effort_levels.low_weight: 0.1`, so the environment marks 10% of sampled prompts with `{reasoning effort: efficient}` and penalizes those responses past 15,000 tokens. The tech report (§3.9) describes the overall RL mix as approximately 4% medium-effort prompts covering math, STEM and coding. The 10% weight is the reference-run RLVR setting; the report does not break the 4% figure down by stage.

The `super35-rl-rlvr-data` artifact must point at NeMo-Gym JSONL with train/val splits. The reference run used its training file as the validation set; validation is effectively off (`val_period: 1000000`).

## Algorithm

This stage uses **asynchronous GRPO** with training and generation on separate nodes:

- Policy lag is at most one step (`max_trajectory_age_steps: 1`).
- Weights are updated in flight, and the KV cache is not recomputed after an update.
- 512 prompts × 16 generations per step (8,192 samples), with leave-one-out baseline, normalized rewards and advantages clipped to ±20.
- Token-level loss with truncated importance sampling (TIS, ratio range [0.2, 5]) and no KL penalty.
- Reward penalties for duplicated reasoning, empty final answers, malformed `<think>` tags and unwanted tokens.
- Invalid tool calls get advantage −5.
- Five repeated, detached MTP heads are trained with loss coefficient 0.3. Policy MTP speculative decoding is off.

## Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 144 GB200 nodes, 4 GPUs each (576 GPUs) |
| Training / generation nodes | 32 / 112 (`colocated.enabled: false`) |
| External judge nodes (reference) | 26 more, outside the Ray cluster |
| Prompts/step × generations | 512 × 16 |
| Global batch | 8,192 |
| Max sequence length | 81,920 |
| Training parallelism | TP4 / CP2 / EP16, sequence packing |
| Generation | vLLM TP4 / EP4, 256 seqs, 32,768 batched tokens |
| Learning rate | 4e-6 constant (10 warmup steps from 4e-7) |
| KL penalty | 0 |
| Router replay | off |

The config is a flattened port of the maintained reference-run recipe. It keeps every key and value except the substitutions listed in its header (artifacts, paths, judge endpoints, logger names). It differs deliberately from the historical run in three ways:

- It uses the same data blend from step 0. The historical run's first 34 steps used an earlier blend.
- Router replay is off throughout. Historically, step 120 came from a router-replay fork created at step 101.
- It keeps the 81,920 context, CP2 and 0.12/0.12 GenRM length penalties. The source recipe's later context-length and penalty changes are not included.

## External judge servers

Three model servers run **outside** the Ray cluster. Start them yourself (for example, vLLM with the OpenAI API) and point the config at them with environment variables. The config resolves these variables when the job starts, so set them in the job environment or pass the equivalent Hydra overrides.

| Server (config key) | Env vars | Model (default) | Reference deployment | Used by |
|---------------------|----------|-----------------|----------------------|---------|
| GenRM (`genrm_model`) | `GENRM_BASE_URL`, `GENRM_MODEL`, `GENRM_API_KEY` | `/path/to/genrm_model` | 8 replicas × TP8, 131,072 context, 384 seqs | GenRM pairwise comparison |
| General judge (`nl2bash_judge_model`) | `NL2BASH_BASE_URL`, `NL2BASH_JUDGE_MODEL` | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | 8 replicas × TP4 | math, equivalence, LC, multi-challenge, inverse IF, abstention, finance, NL2Bash |
| Safety judge (`safety_judge_model`) | `SAFETY_BASE_URL`, `SAFETY_JUDGE_MODEL` | `nvidia/Nemotron-Content-Safety-Reasoning-4B` | 2 replicas × TP4 | jailbreak, over-refusal |

> The GenRM variant the report trained with a reasoning-length penalty (§3.5) is not public.
> NeMo-RL's public Super 3.5 recipe uses [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM) for this server.

All `base_url`s default to `http://127.0.0.1:8000/v1`. The `vllm_serve_kwargs` blocks record how the reference servers were configured.

## Other runtime inputs

| Env var | Purpose |
|---------|---------|
| `CCC_TEST_FILE` | Competitive-coding-challenges metadata JSONL (forwarded via `run.env.env_vars`) |
| `CCC_SHARED_TEMP_DIR` | Compile directory shared by the policy and sandbox containers; it must be mounted in both |
| `FINANCE_SEC_CACHE_DIR`, `FINANCE_SEC_DUMP_PATH` | SEC filing cache and dump for the finance search environment |
| `SCICODE_TEST_DATA`, `SCICODE_TRAIN_DATA` | SciCode test data (`.h5`) and agent transitions JSONL |

`run.env.env_vars` also carries the runtime settings the reference launcher exported. These include the NeMo-RL/vLLM timeouts, `NRL_USE_FASTOKENS=1`, the FlashInfer MoE FP8 backend and the Gym venv directory. `VLLM_ATTENTION_BACKEND` is deliberately left unset.

## Prerequisites

- **NeMo-RL**: the `super-v3.5-posttraining` branch with its pinned Gym, Megatron-Bridge and Megatron-LM submodules. The Super3.5 RL container bakes these in; [`../Dockerfile`](../Dockerfile) pins commit `f299847f1` (Gym `e24db4e6`, Megatron-Bridge `3961f399`).
- **Sandbox container**: NeMo-Skills sandbox for code execution, Lean4 and CCC (`run.env.sandbox.container`). See the [NeMo-Skills sandbox Dockerfile](https://github.com/NVIDIA-NeMo/Skills/blob/main/dockerfiles/Dockerfile.sandbox).
- **Judge servers**: running and reachable from the training nodes (see above).

## Usage

```bash
nemotron super35 rl rlvr \
    --run <profile> \
    run.env.sandbox.container=<sandbox-image> \
    run.env.persistent_cache=/path/to/cache
```

By default the stage starts from `super35-sft-model:latest` and reads `super35-rl-rlvr-data:latest`. It registers checkpoints as `super35-rl-rlvr-model`. Checkpoints are saved every 5 steps. The run stops only at `grpo.max_num_steps`, so append `grpo.max_num_steps=120` to stop at the selected step.

> **`--run <profile>`** refers to a profile defined in your `env.toml` file,
> which configures SLURM account, partition, mounts, and other cluster settings.
> See the [env.toml setup guide](../../../super3/README.md#envtoml-setup) for details.

## References

- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL)
- [Nemotron 3 Super RLVR recipe](../../../super3/stage2_rl/stage1_rlvr/README.md) (the predecessor stage this one follows)
