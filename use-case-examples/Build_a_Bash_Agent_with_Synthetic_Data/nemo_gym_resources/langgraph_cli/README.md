# LangGraph CLI Resource Server

NeMo Gym resource server for training AI agents to translate natural language instructions into correct LangGraph CLI commands.

## Overview

This resource server implements RLVR (Reinforcement Learning with Verifiable Rewards) verification logic for CLI tool-call generation tasks.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/verify` | POST | Verify model output and compute reward signal |
| `/parse_cli_command` | POST | Tool endpoint for CLI parsing (optional) |
| `/health` | GET | Health check |

## Reward Logic

```
Exact Match       → +1.0
Correct Command   → 0.0 to +1.0 (based on flag accuracy)
Wrong Command     → -1.0
Invalid JSON      → -1.0
```

## Usage

### Run Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Test Verification

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-1",
    "task_input": {
      "input": "Build a Docker image with tag v1",
      "output": {"command": "build", "tag": "v1"}
    },
    "model_response": "{\"command\": \"build\", \"tag\": \"v1\"}"
  }'
```

## Integration with Unsloth

The reward functions can be used directly with Unsloth's GRPOTrainer:

```python
from nemo_gym_resources.langgraph_cli import cli_correctness_reward

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[cli_correctness_reward],
    ...
)
```

## License

Apache 2.0
