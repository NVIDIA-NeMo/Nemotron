# How to Train an AI Agent for Command-Line Tasks with Synthetic Data and Reinforcement Learning

What if your computer-use agent could learn a new Command Line Interface (CLI)—and operate it safely without ever writing files or free-typing shell commands?

This tutorial teaches a large reasoning model with no prior knowledge to safely operate the **LangGraph Platform CLI** using **Synthetic Data Generation (SDG)** and **Reinforcement Learning with Verifiable Rewards (RLVR)**, optimized via **Group Relative Policy Optimization (GRPO)**.

## What You'll Build

A specialized AI agent that can:

- **Propose valid LangGraph CLI commands** (e.g., `langgraph dev --port 8123 --no-browser`)
- **Ask for explicit human confirmation** before executing
- **Learn new subcommands** from synthetic seed data
- **Train efficiently on a single GPU** using RLVR

Here's what a typical interaction looks like once the model is trained:

```
[User] Bring the LangGraph server online.

[Agent] I can execute:
["langgraph", "up", "--wait"]

Execute this command? [y/N]: y

[Agent] Server started successfully on port 8000.
```

**This pattern generalizes**: The same workflow can be extended to support new CLI tools and environments.

## Why Use Synthetic Data Generation and Reinforcement Learning?

Teaching an AI agent to operate a specialized CLI tool presents unique challenges:

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Data Scarcity** | Most specialized CLI tools lack massive usage logs needed for training | SDG bootstraps high-quality examples from a handful of seeds |
| **Safety-Accuracy Tradeoff** | Models must be creative in understanding intent but precise in command generation | RLVR teaches consistent, syntactically correct commands |
| **Cold-Start Problem** | Waiting for real-world data could take months | Generate comprehensive datasets in hours |

**Together, SDG + RLVR create a virtuous cycle**: SDG provides diverse training scenarios, while RLVR ensures the model learns to handle them correctly.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 1                      Step 2                      Step 3            │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐   │
│  │  NeMo Data       │      │  Unsloth GRPO    │      │  Human-in-the-   │   │
│  │  Designer        │─────>│  + NeMo Gym      │─────>│  Loop Execution  │   │
│  │                  │      │                  │      │                  │   │
│  │  • Seed Examples │      │  • RLVR Training │      │  • Confirmation  │   │
│  │  • Samplers      │      │  • Verifiable    │      │  • Safe Execution│   │
│  │  • Validation    │      │    Rewards       │      │  • Allowlist     │   │
│  └──────────────────┘      └────────┬─────────┘      └──────────────────┘   │
│                                     │                                       │
│                                     v                                       │
│                            ┌──────────────────┐                             │
│                            │  NeMo Gym        │                             │
│                            │  Resource Server │                             │
│                            │                  │                             │
│                            │  • verify()      │                             │
│                            │  • Reward: [-1,1]│                             │
│                            │  • Flag Accuracy │                             │
│                            └──────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements

> **Important**: This tutorial requires an **80GB GPU** (NVIDIA H100 or A100 80GB) for full training. The GRPO training with multiple generations per prompt requires significant VRAM.

| Mode | VRAM | Recommended GPU |
|------|------|-----------------|
| **BF16 LoRA** | **~60 GB** | **H100 80GB / A100 80GB** |

> **Note**: 4-bit quantization is NOT supported because the Nemotron-Nano-9B-v2 model uses Mamba2 CUDA kernels that are incompatible with bitsandbytes quantization.

**Additional Requirements:**
- Minimum **64 GB system RAM** (128 GB recommended)
- **200 GB free disk space** for model weights, checkpoints, and datasets
- Fast SSD storage recommended for model loading

### Operating System Requirements

- **Ubuntu 22.04 LTS** (tested and recommended)
- Other Linux distributions may work but are not officially supported

### Software Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Ubuntu** | 22.04 LTS | Required |
| **Python** | 3.12+ | 3.13 recommended |
| **CUDA** | 12.0+ | 12.6 tested |
| **NVIDIA Driver** | 535+ | Compatible with CUDA 12.x |
| **uv** | Latest | [Install guide](https://github.com/astral-sh/uv) |

### Core Components

| Component | Role | Documentation |
|-----------|------|---------------|
| **NeMo Data Designer** | Synthetic data generation | [GitHub](https://github.com/NVIDIA-NeMo/DataDesigner) |
| **NeMo Gym** | RL training environments with verifiable rewards | [Docs](https://docs.nvidia.com/nemo/gym/latest/index.html) |
| **Unsloth** | Efficient GRPO training (80% less VRAM) | [Docs](https://docs.unsloth.ai/) |
| **Nemotron-Nano-9B-v2** | Base reasoning model (Transformer-Mamba2 hybrid) | [HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |

---

## Learning Path

Follow these steps in order:

| Step | What You'll Do | Time | Output |
|------|----------------|------|--------|
| **1. Setup** | Install dependencies and configure environment | 15 min | Working environment |
| **2. Generate Data** | Create synthetic training data with NeMo Data Designer | 30 min | `train.jsonl`, `val.jsonl` |
| **3. Train Model** | Fine-tune with GRPO using NeMo Gym rewards | 30-60 min | Trained model checkpoint |
| **4. Run Agent** | Deploy the trained agent for interactive use | 5 min | Working CLI agent |

---

## Step 0: Environment Setup

### Clone and Enter the Project

```bash
git clone <repository-url>
cd Build_a_Computer_Use_Agent_with_Synthetic_Data
```

### Install Base Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync project dependencies
uv sync
```

### Install Training Dependencies

These packages require special installation flags due to CUDA compilation:

```bash
# Install Unsloth and vLLM
uv pip install unsloth vllm

# Install Mamba dependencies (required for Nemotron-Nano-9B-v2)
uv pip install --no-build-isolation "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.2"
uv pip install --no-build-isolation "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.5"
```

> **Note**: The `--no-build-isolation` flag is required because these packages need access to your system's CUDA installation during compilation.

### Set Up API Keys

```bash
# For NeMo Data Designer (synthetic data generation)
export NVIDIA_API_KEY="nvapi-..."

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Quick Start (TL;DR)

Once setup is complete, here's the full workflow:

```bash
# 1. Start the NeMo Gym resource server (in a separate terminal)
cd nemo_gym_resources/langgraph_cli
uv run uvicorn app:app --host 0.0.0.0 --port 8000

# 2. Generate synthetic data (or use provided examples)
uv run jupyter notebook 01_synthetic_data_generation.ipynb

# 3. Train with GRPO (run the notebook)
uv run jupyter notebook 02_grpo_training.ipynb

# 4. Run the trained agent
python bash_agent/main_hf.py
```

---

## Step 1: Design a Synthetic Dataset with NeMo Data Designer

> **Notebook**: [`01_synthetic_data_generation.ipynb`](01_synthetic_data_generation.ipynb)
>
> **Output**: `data/langgraph_cli/train.jsonl`, `data/langgraph_cli/val.jsonl`

Before training, we need data: pairs of natural-language requests mapped to LangGraph CLI invocations.

### Why Use Synthetic Data Generation?

Think of SDG like teaching someone a new language by showing them a pattern, then having them create variations:

1. **Provide a few high-quality "seed" examples**
2. **Use an AI model to generate diverse variations**
3. **Validate each generated example against strict rules**
4. **Build a comprehensive dataset in hours instead of months**

### The Dataset Structure

| User Request | CLI Command | Expected JSON |
|--------------|-------------|---------------|
| "Start a local dev server on port 8123." | `langgraph dev --port 8123` | `{"command": "dev", "port": 8123}` |
| "Build the project image with tag v1." | `langgraph build -t myapp:v1` | `{"command": "build", "tag": "myapp:v1"}` |
| "Create a new react-agent project." | `langgraph new --template react-agent` | `{"command": "new", "template": "react-agent"}` |

### Install NeMo Data Designer

```bash
pip install data-designer
export NVIDIA_API_KEY="nvapi-..."
```

### Generate the Dataset

Open the notebook and follow along:

```bash
uv run jupyter notebook 01_synthetic_data_generation.ipynb
```

The implementation uses NeMo Data Designer's builder pattern with Pydantic validation:

```python
from data_designer.essentials import (
    DataDesigner,
    DataDesignerConfigBuilder,
    SamplerColumnConfig,
    LLMTextColumnConfig,
    LLMStructuredColumnConfig,
    SamplerType,
    CategorySamplerParams,
    ModelConfig,
)
from pydantic import BaseModel, Field

# Define structured output schema
class CLIToolCall(BaseModel):
    command: str = Field(..., description="CLI command: new, dev, up, build, or dockerfile")
    template: Optional[str] = Field(None, description="Template name for 'new' command")
    port: Optional[int] = Field(None, description="Port for 'dev' or 'up' command")
    # ... additional fields

designer = DataDesigner()
builder = DataDesignerConfigBuilder(model_configs=[
    ModelConfig(
        alias="command-generator",
        provider="nvidia",
        model="nvidia/nemotron-3-nano-30b-a3b",  # For generation
    )
])

# 1. Define seed distributions
builder.add_column(SamplerColumnConfig(
    name="command",
    sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(values=["new", "dev", "build", "up", "dockerfile"])
))

# 2. Generate natural language requests from seeds
builder.add_column(LLMTextColumnConfig(
    name="input",
    model_alias="command-generator",
    prompt="Generate a natural user request for {{ command }}..."
))

# 3. Generate structured JSON tool-calls
builder.add_column(LLMStructuredColumnConfig(
    name="output",
    model_alias="command-generator",
    prompt="Convert this user request to a CLI tool-call...",
    output_format=CLIToolCall,  # Pydantic validation!
))

# Preview and generate
preview = designer.preview(config_builder=builder, num_records=5)
dataset = designer.create(config_builder=builder, num_records=250)
```

**Output**: `train.jsonl` (225 examples) and `val.jsonl` (25 examples) files containing validated command pairs.

---

## Step 2: Fine-Tune with RLVR (using GRPO)

> **Notebook**: [`02_grpo_training.ipynb`](02_grpo_training.ipynb)
>
> **NeMo Gym Server**: [`nemo_gym_resources/langgraph_cli/app.py`](nemo_gym_resources/langgraph_cli/app.py)
>
> **Output**: `outputs/grpo_langgraph_cli/merged_model/`

With clean, verified data in hand, we move to fine-tuning using **Unsloth** and **NeMo Gym**.

### Before Training: Start the NeMo Gym Resource Server

The training process requires a running NeMo Gym server to compute rewards. **Start this in a separate terminal before running the training notebook:**

```bash
cd nemo_gym_resources/langgraph_cli
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Keep this terminal running throughout training.

### Reinforcement Learning with Verifiable Rewards (RLVR)

Traditional RLHF uses human judges—subjective, expensive, and inconsistent. **RLVR replaces human judges with deterministic code-based verification.**

Instead of asking humans *"Does this command look good?"*, we ask code *"Does this command pass our validation rules?"*

**The reward system:**

| Output | Reward | Effect |
|--------|--------|--------|
| Valid command (exact match) | +1.0 | Strongly reinforced |
| Correct command, partial flags | 0.0 to +1.0 | Partially reinforced |
| Wrong command | -1.0 | Discouraged |
| Invalid JSON | -1.0 | Discouraged |

This consistency is crucial: **The same output always yields the same reward**, making training stable and predictable.

### Building the Training Environment with NeMo Gym

The CLI agent environment is implemented as a **NeMo Gym resource server** (`nemo_gym_resources/langgraph_cli/app.py`), which encapsulates:

- **Tool definitions**: The CLI commands the agent can propose
- **Verification logic**: Rules that check command validity and correctness
- **Reward computation**: Scores returned to the RL training loop

```python
# The "Verifiable Reward" Function (simplified)
def score_cli_output(predicted, reference):
    # Normalize both inputs for fair comparison
    predicted = normalize_cli_output(predicted)
    reference = normalize_cli_output(reference)

    # If command is wrong, full penalty
    if predicted.get("command") != reference.get("command"):
        return -1.0

    # Compute flag accuracy
    correct_count = sum(1 for k, v in reference.items()
                       if predicted.get(k) == v)
    return correct_count / len(reference)
```

### Optimization via Group Relative Policy Optimization (GRPO)

**GRPO is a simpler, more memory-efficient alternative to PPO.** Instead of training a separate "critic" model, GRPO samples multiple outputs for the same prompt and uses their average reward as the baseline.

Here's how it handles sparse rewards:

1. Model generates 4 command variations for the same prompt
2. Three are invalid (reward = -1), one is valid (reward = +1)
3. GRPO computes **relative advantages within the group**
4. Strongly reinforces that one success, making it stand out from failures

### Run GRPO Training

Open the training notebook:

```bash
uv run jupyter notebook 02_grpo_training.ipynb
```

**Key training configuration:**

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Load Nemotron-Nano-9B-v2 with Unsloth (BF16, NOT 4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    max_seq_length=1024,
    load_in_4bit=False,  # Mamba2 kernels incompatible with 4-bit
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Apply Nemotron patch for Unsloth GRPO compatibility
from nemotron_unsloth_patch import patch_nemotron_for_unsloth_grpo
patch_nemotron_for_unsloth_grpo(model)

# Apply LoRA adapters for parameter-efficient training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# Configure GRPO training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],  # Connected to NeMo Gym server
    args=GRPOConfig(
        max_steps=50,
        num_generations=4,  # Completions per prompt for GRPO
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
    ),
    train_dataset=dataset,
)

trainer.train()
```

**Important notes:**
- Nemotron-Nano-9B-v2 is a **hybrid Transformer-Mamba2** model (~9B parameters)
- The `nemotron_unsloth_patch.py` monkey-patch is required for Unsloth GRPO compatibility
- Training takes ~30 minutes for 50 steps on H100 80GB

---

## Step 3: Human-in-the-Loop Execution

> **Main Agent**: [`bash_agent/main_hf.py`](bash_agent/main_hf.py) (HuggingFace inference)
>
> **Configuration**: [`bash_agent/config.py`](bash_agent/config.py)
>
> **Bash Tool**: [`bash_agent/bash.py`](bash_agent/bash.py) (with security features)
>
> **Interactive Notebook**: [`03_run_agent.ipynb`](03_run_agent.ipynb)

Once fine-tuned, we embed the model into a runtime loop that **always requests human confirmation before execution**.

### The Safety Architecture

The agent implements multiple layers of protection:

| Layer | Protection | Implementation |
|-------|------------|----------------|
| **Training-time** | RLVR ensures valid command generation | NeMo Gym rewards |
| **Command allowlist** | Only approved commands can execute | Pattern matching |
| **Injection prevention** | Blocks `$` and backtick patterns | Regex validation |
| **Human confirmation** | Explicit approval required | Interactive prompt |

```python
# From bash_agent/bash.py
class Bash:
    def exec_bash_command(self, cmd: str):
        # Prevent command injection via backticks or $
        if re.search(r"[`$]", cmd):
            return {"error": "Command injection patterns are not allowed."}

        # Check the allowlist
        for cmd_part in self._split_commands(cmd):
            if cmd_part not in self.config.allowed_commands:
                return {"error": f"Command '{cmd_part}' is not in the allowlist."}

        return self._run_bash_command(cmd)
```

### Run the Agent

The agent uses the trained model checkpoint from Step 2 for local HuggingFace inference:

```bash
python bash_agent/main_hf.py
```

**Command-line options:**

```bash
# Use a different model path
python bash_agent/main_hf.py --model-path /path/to/your/model

# Use OpenAI-compatible API (e.g., vLLM server)
python bash_agent/main_hf.py --use-api --api-url http://localhost:8000/v1

# Adjust generation parameters
python bash_agent/main_hf.py --temperature 0.2 --device cuda
```

**Example interaction:**

```
============================================================
Bash Computer Use Agent
============================================================
Model: outputs/grpo_langgraph_cli/merged_model
Working directory: /home/user/project
Type 'quit' or 'exit' to stop.
============================================================

['/home/user/project'] > Create a new project with the react-agent template

Thinking...

Proposed command: langgraph new --template react-agent
    Execute 'langgraph new --template react-agent'? [y/N]: y

Output:
Project created successfully!

------------------------------------------------------------
```

**Alternative: Run in a Jupyter notebook:**

```bash
uv run jupyter notebook 03_run_agent.ipynb
```

---

## Project Structure

```
Build_a_Computer_Use_Agent_with_Synthetic_Data/
│
├── README.md                              # This walkthrough
├── pyproject.toml                         # Dependencies (uv)
│
├── 01_synthetic_data_generation.ipynb  # Step 1: SDG notebook
├── 02_grpo_training.ipynb              # Step 2: Training notebook
├── 03_run_agent.ipynb                  # Step 3: Interactive agent notebook
│
├── nemotron_unsloth_patch.py           # Unsloth compatibility patch
├── langgraph_cli_synthetic_preview.jsonl  # Sample synthetic data
│
├── data/                                  # Training data
│   └── langgraph_cli/
│       ├── train.jsonl                    # Training dataset (225 examples)
│       └── val.jsonl                      # Validation dataset (25 examples)
│
├── outputs/                               # Training outputs
│   └── grpo_langgraph_cli/
│       └── merged_model/                  # Trained model checkpoint
│
├── nemo_gym_resources/                    # NeMo Gym integration
│   └── langgraph_cli/
│       ├── app.py                         # Resource server + verify()
│       ├── README.md                      # Server documentation
│       ├── configs/langgraph_cli.yaml     # Environment config
│       ├── data/example.jsonl             # Example tasks
│       └── tests/test_app.py              # Unit tests
│
└── bash_agent/                            # Step 3: CLI agent
    ├── main_hf.py                         # Main agent (HuggingFace inference)
    ├── config.py                          # Configuration and system prompts
    ├── bash.py                            # Bash tool with security features
    ├── helpers.py                         # LLM wrappers (HuggingFace + OpenAI)
    └── prompts.py                         # System prompts
```

---

## Why RLVR + Synthetic Data Work for Customizing Agentic AI

| Component | Role | Why It Matters |
|-----------|------|----------------|
| **NeMo Data Designer** | Generates diverse, validated training data | Solves the cold-start problem |
| **NeMo Gym** | Provides training environment with verifiable rewards | Defines success measurement |
| **Unsloth + GRPO** | Efficient RL training (80% less VRAM) | Makes RL accessible on single GPU |
| **Human approval loop** | Final safety gate | Maintains user trust and control |

**The result**: We can teach Nemotron-Nano-9B-v2 to precisely and safely operate a new CLI tool—all without full retraining or compromising on safety.

---

## Extending to Your Own CLI Tools

The workflow generalizes cleanly to **any CLI tool**:

1. **Use NeMo Data Designer** to define structured, verifiable examples
2. **Build a NeMo Gym environment** with your CLI tools and verification logic
3. **Fine-tune efficiently** with Unsloth's GRPO
4. **Maintain human-in-the-loop execution** for safety

This pattern lets you turn any capable LLM into a **domain-specific, verifiably safe computer-use agent**—from LangGraph today to your proprietary internal tools tomorrow.

---

## References

### Documentation
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)
- [NeMo Gym](https://docs.nvidia.com/nemo/gym/latest/index.html)
- [Unsloth GRPO Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)

### Models
- [Nemotron-Nano-9B-v2 (HuggingFace)](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) - Training model
- [Nemotron-3-Nano-30B-A3B (HuggingFace)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B) - Data generation model

### Related Resources
- [NVIDIA Bash Computer Use Agent Reference](https://github.com/NVIDIA/GenerativeAIExamples/tree/main/nemotron/LLM/bash_computer_use_agent) - The reference implementation this project aligns with
- [NVIDIA Nemotron Developer Page](https://developer.nvidia.com/nemotron)
- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)

---

## License

This project is provided for educational and research purposes.
