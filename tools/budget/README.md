# Custom Logit Processors for NVIDIA Nemotron-3-Nano

A vLLM V1 custom logit processor for runtime thinking budget control on reasoning models like `nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3`.

## Overview

This package provides `ThinkingBudgetLogitsProcessor` - a logit processor that allows dynamic control over the "thinking" phase of reasoning models. When a model is in thinking mode (indicated by `<think>` tags), this processor can:

- Enforce a maximum token budget for the thinking phase
- Gracefully truncate thinking with customizable end tokens
- Allow per-request budget overrides from the client

This is useful for balancing inference cost against reasoning depth, or enforcing latency constraints in production deployments.

## Installation

```bash
pip install -e .
```

## Server Setup

Start a vLLM server with the custom logit processor:

```bash
./serve_v3.sh
```

Or manually with environment configuration:

```bash
export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 500, "thinking_budget_grace_period": 50, "end_token_ids": [2259, 74045, 1062], "prompt_think_ids": [198, 27, 27963, 397], "end_think_ids": [[524, 27963, 397]]}'

vllm serve nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3 \
    --port 8881 \
    --trust-remote-code \
    --logits-processors custom_logit_processors.v1.ThinkingBudgetLogitsProcessor
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `thinking_budget` | Max tokens before attempting to end thinking. Set to `-1` for unlimited. | `-1` |
| `thinking_budget_grace_period` | Additional tokens allowed after budget to find a natural breakpoint (newline). | `-1` |
| `end_token_ids` | Token IDs to inject when truncating thinking (e.g., `</think>` tokens). | `[]` |
| `prompt_think_ids` | Token sequence indicating the model is in thinking mode (e.g., `\n<think>\n`). | `[]` |
| `end_think_ids` | Token sequences that indicate thinking has ended naturally. | `[]` |

### Recommended Values for Nemotron-Nano-v3

For `nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3`, we recommend the following token ID configurations:

```json
{
  "end_token_ids": [2259, 74045, 1062],
  "prompt_think_ids": [198, 27, 27963, 397],
  "end_think_ids": [[524, 27963, 397]]
}
```

These correspond to:
- `end_token_ids`: `</think>` (injected when truncating)
- `prompt_think_ids`: `\n<think>\n` (detects thinking mode)
- `end_think_ids`: `</think>\n` (natural thinking termination)

## Client Usage

### Default Behavior (Server-Side Budget)

When no per-request overrides are provided, the server uses the parameters configured in the `THINKING_BUDGET_LOGITS_PROCESSOR_ARGS` environment variable at startup.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8881/v1",
    api_key="EMPTY"
)

result = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 5.9 plus 6.1?"}
    ],
    temperature=1.0,
    max_tokens=12200,
)

# Parse thinking and answer
thinking_part, delim, answer_part = result.choices[0].message.content.partition("</think>")
print("Thinking:", thinking_part + delim)
print("Answer:", answer_part)
```

### Custom Per-Request Budget

Override the server defaults for individual requests:

```python
import json

# Custom truncation message tokens: "Reached thinking limit set by client\n\n</think>"
custom_think_truncation = [1871, 5565, 11483, 6139, 2016, 1536, 6934, 1338, 13]

result = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve this complex problem..."}
    ],
    temperature=1.0,
    max_tokens=12200,
    extra_body={
        "vllm_xargs": {
            "thinking_budget": 100,
            "thinking_budget_grace_period": 20,
            "end_token_ids": json.dumps(custom_think_truncation),
        }
    }
)
```

## How It Works

1. **Detection**: When a request arrives, the processor checks if the prompt ends with `prompt_think_ids` (e.g., `\n<think>\n`), indicating the model should reason before answering.

2. **Monitoring**: As tokens are generated, the processor tracks output length and watches for natural thinking endings (`end_think_ids`).

3. **Truncation**: When `thinking_budget` is exceeded:
   - If within the grace period, it waits for a newline token as a natural breakpoint
   - Once triggered, it forces the `end_token_ids` sequence by setting all other logits to `-inf`

4. **Completion**: After injecting the end sequence, normal generation resumes for the answer portion.

## File Structure

```
custom_logit_processors/
├── v1/
│   ├── __init__.py
│   └── nano_v3_logit_processors.py   # ThinkingBudgetLogitsProcessor
├── client.py                          # Example client usage
├── pyproject.toml
└── README.md
```

## Token ID Reference

The `NEWLINE_TOKENS` set in the processor contains all token IDs that represent newline characters across the tokenizer vocabulary. These are used to find natural breakpoints when truncating thinking.

Common end sequences for Nemotron models:
- `[2259, 74045, 1062]` → `</think>`
- `[1871, 5565, 11483, 6139, 1046, 2259, 74045, 1062]` → `Reached thinking limit. </think>`

## Requirements

- vLLM >= 0.10.1 (with V1 engine support)
- PyTorch
- transformers
