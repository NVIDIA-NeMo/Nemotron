# Data Designer

```python
# 1. Define the Distribution (The "Seeds")
# explicitly define the problem space with typed samplers
command  = Sampler(["new", "dev", "build"])
port     = Sampler(range(3000, 9000))
template = Sampler(["react-agent", "rag-agent"])

# 2. Synthetic Input (The "Problem")
# Use an LLM to generate natural language requests from the seeds
# e.g. "Start a dev server on port 8080"
user_request = LLM(
    prompt=f"Ask to {command} using {template} on port {port}...",
    model="nemotron-4-340b"
)

# 3. Synthetic Output (The "Solution")
# Generate the exact JSON tool-call the agent *should* produce
# e.g. {"command": "dev", "port": 8080, ...}
tool_call = LLM(
    prompt=f"Convert '{user_request}' to a valid CLI JSON object.",
    schema=CLIToolCall, # Pydantic model ensures valid JSON structure
    model="nemotron-4-340b"
)
```
