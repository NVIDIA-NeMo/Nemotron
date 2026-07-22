# Build a LangGraph CLI Agent with Synthetic Data and GRPO

This tutorial builds a small Nemotron agent that turns natural-language requests into validated [LangGraph CLI](https://docs.langchain.com/langgraph-platform/cli) tool calls. It covers synthetic data generation, reinforcement learning with verifiable rewards (RLVR), and a human-confirmed runtime.

The updated workflow uses:

1. **NeMo Data Designer + OpenRouter** to generate varied requests while deriving every target deterministically.
2. **NeMo Gym** to score structured JSON outputs through a real resource server.
3. **Unsloth + TRL GRPO** to adapt Nemotron 3 Nano 4B.
4. A runtime that converts structured model output to immutable process arguments, validates it, asks for confirmation, and executes without a shell.

## Pinned compatibility matrix

The generation pins in [`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock) and the Linux CUDA pins in [`training/pyproject.toml`](training/pyproject.toml) / [`training/uv.lock`](training/uv.lock) were refreshed on July 1, 2026.

| Component | Version |
|---|---:|
| uv | 0.11.25 |
| Python | 3.12 |
| NeMo Data Designer | 0.7.0 |
| NeMo Gym | 0.3.0 |
| LangGraph CLI | 0.4.30 |
| PyTorch / torchvision | 2.10.0 / 0.25.0 |
| causal-conv1d / Mamba SSM | 1.6.1 / 2.3.1 |
| Unsloth / Unsloth Zoo | 2026.6.9 / 2026.6.7 |
| Transformers / TRL | 4.56.2 / 0.24.0 |
| OpenAI Python SDK | 2.7.2 |

Unsloth 2026.6.9 currently caps TRL at 0.24.0. TRL 0.24 supports Transformers 4.56.2, which is pinned because NVIDIA's reviewed 4B repository code uses that generation API; Transformers 5.5 calls it with an incompatible cache contract. Data Designer 0.7 requires Hugging Face Hub 1.x while Transformers 4.56 requires Hub <1, so generation and training have separate lockfiles and virtual environments rather than forcing an invalid shared environment. NeMo Gym 0.3 currently caps the OpenAI SDK at 2.7.2, and the Mamba pins match the Torch 2.10 wheels used by Unsloth's compatible stack. These are tested compatibility choices, not stale independent pins.

## Requirements

- [`uv` 0.11.25](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12. Both projects enforce this uv version so lockfile behavior is reproducible.
- An `OPENROUTER_API_KEY` for hosted data generation.
- Linux x86_64, a CUDA installation compatible with PyTorch 2.10, and a BF16-capable NVIDIA GPU for training.
- Docker for LangGraph `up` and `build`; `dev` runs without Docker.
- A `LANGSMITH_API_KEY` with LangSmith Deployment access for `langgraph up` (or the production license credentials required by your LangGraph deployment).

The training recipe targets [`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16). The notebook is a short pipeline smoke test; memory use still depends on sequence length, number of generations, and GPU architecture. Other Nemotron checkpoints can have different module structures and need checkpoint-specific LoRA targets and memory tuning.

## Setup

From the repository root:

```bash
cd use-case-examples/Build_a_Bash_Agent_with_Synthetic_Data
uv sync --extra generation --dev
```

The generation notebook prompts for `OPENROUTER_API_KEY` with hidden input when it is not already supplied by a credential manager. Do not paste a secret into a command that will be saved in shell history; shut down the notebook kernel or unset the variable when finished.

Add the CUDA training stack only on the training machine:

```bash
uv sync --project training
uv run --project training python install_ssm_kernels.py
uv run --project training python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The Linux-only training project creates `training/.venv`; it does not modify the generation environment at `.venv`.

`install_ssm_kernels.py` detects the active Torch, CUDA, Python, C++ ABI, and CPU architecture, then installs the matching official `causal-conv1d` and `mamba-ssm` wheels. If no matching wheel exists, it builds the pinned release against the active environment, which requires a CUDA toolkit and compiler. It verifies the exact kernel symbols in a fresh interpreter. Run the helper after every training `uv sync`, and again after changing Torch or Python; these ABI-specific wheels are deliberately installed after the portable dependency sync.

## 1. Generate synthetic data

Open [`01_synthetic_data_generation.ipynb`](01_synthetic_data_generation.ipynb):

```bash
uv run jupyter lab 01_synthetic_data_generation.ipynb
```

The notebook samples the command and arguments, derives the exact JSON label in Python, then asks `nvidia/nemotron-3-nano-30b-a3b` through OpenRouter only to phrase a matching user request. A conservative validator requires exact template IDs, paths, ports, tags, and enabled boolean options; incomplete or embellished phrasing is replaced with an unambiguous deterministic request before review. The OpenRouter endpoint, model, reasoning-off behavior, and no-fallback routing are explicit in the notebook. It writes NeMo Gym-native records to:

```text
data/langgraph_cli/train.jsonl
data/langgraph_cli/validation.jsonl
```

Review the five-record preview before generating the full dataset. The notebook then validates every target against the runtime grammar and requires explicit approval of a reproducible 25-row semantic audit before export. Natural wording can still omit or invent a requested argument even when the deterministic label is correct, so use a larger independent audit for production data.

## 2. Start and test the verifier

The Gym 0.3 resource server lives in [`resources_servers/langgraph_cli`](resources_servers/langgraph_cli). Validate it first:

```bash
uv run --project training ng_dump_config "+config_paths=[resources_servers/langgraph_cli/configs/langgraph_cli.yaml]"
uv run --project training ng_test +entrypoint=resources_servers/langgraph_cli
```

Then start it in a separate terminal:

```bash
uv run --project training ng_run "+config_paths=[resources_servers/langgraph_cli/configs/langgraph_cli.yaml]"
```

`ng_run` supplies Gym's head-server configuration and serves the verifier at `http://127.0.0.1:8000/verify`. Do not start this server directly with Uvicorn.

The verifier accepts plain JSON, fenced JSON, or a JSON object after a closed reasoning block. It validates the schema and scores command and argument accuracy in `[-1, 1]`.

## 3. Train with GRPO

With the verifier still running, open [`02_grpo_training.ipynb`](02_grpo_training.ipynb):

```bash
uv run --project training jupyter lab 02_grpo_training.ipynb
```

The notebook first sends a known-good record to Gym, runs 50 GRPO steps, computes TRL's held-out loss, then greedily generates every held-out response and scores it through the real Gym verifier. It persists the per-record completions and task rewards alongside the summary. Nemotron's reasoning mode is disabled at the tokenizer boundary so TRL 0.24 uses the same short-output template during sizing, rollouts, and inference. It does not use vLLM, bitsandbytes, FlashAttention, or the old Nemotron-specific Unsloth model monkey patch. The merged checkpoint is saved to:

```text
outputs/grpo_langgraph_cli/merged_model
```

The save step copies the pinned Nemotron-H modeling code and license into the merged directory, then asserts the files required for a clean `trust_remote_code=True` load. Increase the training budget only after inspecting reward logs, held-out predictions, and sampled JSON. The official 4B checkpoint currently declares custom Nemotron-H classes, so the notebook pins the reviewed NVIDIA revision `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` rather than executing mutable `main`. Review the repository diff before deliberately updating that revision.

## 4. Run the agent

[`03_run_agent.ipynb`](03_run_agent.ipynb) demonstrates validation and runtime setup. For the interactive local-model runtime:

```bash
uv run --project training python -m bash_agent.main_hf \
  --model-path outputs/grpo_langgraph_cli/merged_model \
  --device cuda \
  --root-dir /path/to/trusted/langgraph-project
```

To use the merged Nemotron model from a vLLM OpenAI-compatible server instead:

```bash
uv sync --extra api --dev
uv run python -m bash_agent.main_hf \
  --use-api \
  --api-url http://127.0.0.1:8001/v1 \
  --api-model my-served-model \
  --root-dir /path/to/trusted/langgraph-project
```

Supply `OPENAI_API_KEY` through a credential manager or a hidden shell prompt before starting the API client; avoid literal secret values in command history.

The API client sends vLLM's `chat_template_kwargs` reasoning-off extension by default so serving matches training. If a strict OpenAI-compatible endpoint rejects that extension, add `--omit-api-thinking-override` and configure reasoning-off in the server's chat template instead.

The fixed `--root-dir` is both the command working directory and the path-confinement boundary. Use an existing LangGraph project for `dev`, `up`, `build`, and `dockerfile`; use a trusted parent directory when creating a new child project.

Finite operations have a 600-second default timeout, configurable with `--command-timeout SECONDS`. Each stdout/stderr stream and background log retains at most 1 MiB. `dev` and `up` without `--wait` run as managed background process groups, write combined output to the displayed temporary log, and are stopped—with their logs deleted—when the launcher closes normally or handles an interactive interruption. Abrupt termination such as `SIGKILL` cannot run cleanup.

Current LangGraph CLI makes `up --wait` detached after Docker reports readiness. Its containers therefore outlive this launcher and must be stopped through the normal Docker/LangGraph workflow.

Child commands automatically omit common credential variables, including `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, and `LANGGRAPH_CLOUD_LICENSE_KEY`. If a reviewed project or `up` genuinely needs one, opt in by name without placing its value on the command line:

```bash
uv run --project training python -m bash_agent.main_hf \
  --model-path outputs/grpo_langgraph_cli/merged_model \
  --device cuda \
  --root-dir /path/to/trusted/langgraph-project \
  --pass-env LANGSMITH_API_KEY
```

Set `LANGSMITH_API_KEY` through a credential manager or hidden prompt before using `--pass-env`, and unset it when the reviewed child process no longer needs it.

The runtime supports this deliberately narrow LangGraph CLI grammar:

| Command | Supported arguments |
|---|---|
| `new` | child project path plus a current template ID |
| `dev` | `--port`, `--no-browser` |
| `up` | `--port`, `--watch`, `--wait` |
| `build` | `-t IMAGE_TAG` |
| `dockerfile` | positional `SAVE_PATH` |

Current template IDs are `agent-python`, `deep-agent-python`, `deep-agent-js`, `new-langgraph-project-python`, and `new-langgraph-project-js`.

### Safety boundary

This runtime is safer than executing model-generated shell text: it rejects raw command strings, validates a fixed grammar, confines generated paths to the configured working directory, passes an argument list to `subprocess` with `shell=False`, revalidates at the execution boundary, filters common secrets from the child environment, bounds retained output, terminates whole managed process groups, and requires approval of the exact arguments.

It is **not a sandbox**. `new` and `dockerfile` write files; `dev` can execute project code; and `up`/`build` invoke Docker, which is a powerful host capability. `langgraph new` also downloads templates from mutable upstream branches, so inspect the generated project before executing it. Run the agent only in a trusted project with least-privilege credentials and review every proposed invocation.

## Tests

CPU-only tests cover deterministic labels, Gym request/response wiring, verifier scoring, command injection resistance, path confinement, confirmation behavior, inference configuration, and tool-message ordering:

```bash
uv run pytest
```

## Project layout

```text
01_synthetic_data_generation.ipynb  Data Designer 0.7 pipeline
02_grpo_training.ipynb              Gym-backed GRPO smoke training
03_run_agent.ipynb                  Validation and inference walkthrough
install_ssm_kernels.py              ABI-matched Mamba CUDA kernel installer
training/                           Separate Linux CUDA project and lockfile
bash_agent/                         Validated interactive runtime
resources_servers/langgraph_cli/    NeMo Gym 0.3 verifier
tests/                              CPU-only tutorial and runtime tests
tutorial_utils.py                   Shared dataset and reward helpers
```

## References

- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Unsloth reinforcement-learning guide](https://docs.unsloth.ai/basics/reinforcement-learning-guide)
- [LangGraph CLI reference](https://docs.langchain.com/langgraph-platform/cli)
