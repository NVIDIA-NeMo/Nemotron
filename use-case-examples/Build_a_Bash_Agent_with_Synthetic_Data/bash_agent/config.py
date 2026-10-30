"""Configuration for the LangGraph CLI agent runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .prompts import JSON_SYSTEM_PROMPT


_EXAMPLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    """Runtime settings shared by local and OpenAI-compatible inference."""

    model_path: str = os.path.join(_EXAMPLE_ROOT, "outputs/grpo_langgraph_cli/merged_model")
    max_seq_length: int = 1024
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    device: str = "cuda"

    use_api: bool = False
    api_base_url: str = "http://localhost:8000/v1"
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "not-needed-for-local")
    )
    api_model_name: str = "local-model"
    api_send_thinking_override: bool = True

    root_dir: str = _EXAMPLE_ROOT
    command_timeout_seconds: float = 600.0
    background_startup_grace_seconds: float = 1.0
    background_shutdown_grace_seconds: float = 5.0
    # Retained bytes per finite stdout/stderr stream and per background log.
    # Background logs are deleted by Bash.close().
    output_limit_bytes: int = 1024 * 1024
    pass_environment: set[str] = field(default_factory=set)

    @property
    def json_system_prompt(self) -> str:
        """Prompt matching the structured command format used for training."""
        return JSON_SYSTEM_PROMPT

    @property
    def system_prompt(self) -> str:
        """Backward-compatible alias for the structured runtime prompt."""
        return self.json_system_prompt
