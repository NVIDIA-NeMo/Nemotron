# RLVR

```python
# 1. The "Verifiable Reward" Function
# This code runs for every generated command during training
def compute_reward(agent_output, expected):
    try:
        cmd = json.loads(agent_output)
        
        # Hard Rule: Command must match expectation
        if cmd.name != expected.name:
            return -1.0  # Penalize hallucinations
            
        # Soft Rule: Flags must be accurate
        # +1.0 for perfect match, partial credit for some correct flags
        accuracy = calculate_flag_accuracy(cmd.flags, expected.flags)
        return accuracy
        
    except JSONDecodeError:
        return -1.0 # Penalize broken syntax

# 2. Start GRPO Training
# Group Relative Policy Optimization efficiently explores the solution space
grpo.train(
    model="nemotron-nano-9b",
    algorithm="GRPO",
    env=compute_reward,      # The verifier above
    dataset=synthetic_data   # From Data Designer
)
```
