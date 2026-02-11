import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """
    Configuration class for the Bash Computer Use Agent.

    Aligned with NVIDIA GenerativeAIExamples bash_computer_use_agent.
    """

    # -------------------------------------
    # Model configuration
    # -------------------------------------

    # Path to the trained model checkpoint (from unsloth_grpo_training.ipynb)
    model_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs/grpo_langgraph_cli/merged_model"
    )

    # Alternatively use a HuggingFace model ID
    # model_path: str = "unsloth/Qwen2.5-1.5B-Instruct"

    # Maximum sequence length for generation
    max_seq_length: int = 512
    max_new_tokens: int = 256

    # Sampling parameters (reduced temperature for deterministic outputs)
    temperature: float = 0.1
    top_p: float = 0.95

    # Device configuration
    device: str = "cuda"  # or "cpu"

    # Optional: Use OpenAI-compatible API instead of local inference
    use_api: bool = False
    api_base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed-for-local"
    api_model_name: str = "local-model"

    # -------------------------------------
    # Agent configuration
    # -------------------------------------

    # The directory path that the agent can access and operate in
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # The list of commands that the agent can execute
    #
    # WARNING: Be very careful about which commands you allow here.
    #          By running this code you assume all responsibility for
    #          unintended consequences of command execution.
    allowed_commands: List[str] = field(default_factory=lambda: [
        "cd", "cp", "ls", "cat", "find", "touch", "echo", "grep", "pwd",
        "mkdir", "wget", "sort", "head", "tail", "du", "wc", "file",
        # LangGraph CLI commands
        "langgraph",
    ])

    @property
    def system_prompt(self) -> str:
        """Generate the system prompt for the LLM based on allowed commands."""
        return f"""You are a helpful and very concise Bash assistant with the ability to execute commands in the shell.
You engage with users to help answer questions about bash commands, or execute their intent.
If user intent is unclear, keep engaging with them to figure out what they need and how to best help
them. If they ask questions that are not relevant to bash or computer use, decline to answer.

When a command is executed, you will be given the output from that command and any errors. Based on
that, either take further actions or yield control to the user.

The bash interpreter's output and current working directory will be given to you every time a
command is executed. Take that into account for the next conversation.
If there was an error during execution, tell the user what that error was exactly.

You are only allowed to execute the following commands. Break complex tasks into shorter commands from this list:

```
{self.allowed_commands}
```

**Never** attempt to execute a command not in this list. **Never** attempt to execute dangerous commands
like `rm`, `mv`, `rmdir`, `sudo`, etc. If the user asks you to do so, politely refuse.

When you want to execute a bash command, output it in a tool call format like:
{{"tool": "exec_bash_command", "cmd": "your command here"}}

If you need to run a LangGraph CLI command, use the same format:
{{"tool": "exec_bash_command", "cmd": "langgraph build -t myapp:v1"}}
"""

    @property
    def json_system_prompt(self) -> str:
        """System prompt for JSON-structured tool calling (matches training format)."""
        return """You are an expert CLI assistant for the LangGraph Platform CLI.

Translate user requests into structured JSON tool calls.

Available commands:
- new: Create project (flags: template, path)
- dev: Start dev server (flags: port, no_browser)
- up: Launch container (flags: port, watch)
- build: Build image (flags: tag)
- dockerfile: Generate Dockerfile (flags: output_path)

Example: {"command": "new", "template": "react-agent", "path": null, "port": null, "no_browser": null, "watch": null, "tag": null, "output_path": null}

Respond with ONLY a JSON object. Set unused flags to null.
"""
