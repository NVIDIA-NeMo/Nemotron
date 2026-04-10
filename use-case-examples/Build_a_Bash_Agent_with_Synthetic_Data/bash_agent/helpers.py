"""
Helper classes for message handling and LLM inference.

Aligned with NVIDIA GenerativeAIExamples bash_computer_use_agent/helpers.py
with added HuggingFace local inference support.
"""

from typing import Any, Dict, List, Tuple, Optional
import json
import re

from config import Config


class Messages:
    """
    An abstraction for a list of system/user/assistant/tool messages.

    Compatible with both OpenAI API format and HuggingFace chat templates.
    """

    def __init__(self, system_message: str = ""):
        self.system_message = None
        self.messages = []
        self.set_system_message(system_message)

    def set_system_message(self, message: str):
        """Set the system message."""
        self.system_message = {"role": "system", "content": message}

    def add_user_message(self, message: str):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": message})

    def add_tool_message(self, message: Any, id: str):
        """Add a tool response message."""
        self.messages.append({
            "role": "tool",
            "content": str(message) if not isinstance(message, str) else message,
            "tool_call_id": id
        })

    def to_list(self) -> List[Dict[str, str]]:
        """Convert to a list of messages for API calls."""
        return [self.system_message] + self.messages

    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert to HuggingFace chat format (without tool_call_id)."""
        result = [self.system_message]
        for msg in self.messages:
            # For tool messages, convert to user message format for HF
            if msg.get("role") == "tool":
                result.append({
                    "role": "user",
                    "content": f"Tool result: {msg['content']}"
                })
            else:
                result.append({"role": msg["role"], "content": msg["content"]})
        return result


class HuggingFaceLLM:
    """
    LLM wrapper for local HuggingFace model inference.

    Loads the trained model from the checkpoint and provides
    a query interface compatible with the agent loop.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer from the checkpoint."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from: {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 (matches training)
            device_map=self.config.device,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Model loaded on {self.config.device}")

    def query(
        self,
        messages: Messages,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Query the model with the given messages.

        Args:
            messages: The conversation messages
            tools: Optional list of tool schemas (for future tool calling support)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, tool_calls)
        """
        import torch

        # Convert messages to chat format
        chat_messages = messages.to_chat_format()

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response (only the new tokens)
        response = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response)

        return response, tool_calls

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from the model response.

        The trained model outputs JSON objects for tool calls.
        This parser extracts them from the response.

        Args:
            response: The model's text response

        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        # Strip out thinking tags if present
        clean_response = response
        if "</think>" in clean_response:
            clean_response = clean_response.split("</think>")[-1].strip()

        # Try to parse as direct JSON first
        try:
            parsed = json.loads(clean_response.strip())
            if isinstance(parsed, dict):
                if "tool" in parsed and "cmd" in parsed:
                    # Format: {"tool": "exec_bash_command", "cmd": "..."}
                    tool_calls.append({
                        "id": "call_0",
                        "function": {
                            "name": parsed["tool"],
                            "arguments": json.dumps({"cmd": parsed["cmd"]})
                        }
                    })
                    return tool_calls
                elif "command" in parsed:
                    # Format from training: {"command": "build", "tag": "..."}
                    # Convert to bash command
                    cmd = self._json_to_bash_command(parsed)
                    if cmd:
                        tool_calls.append({
                            "id": "call_0",
                            "function": {
                                "name": "exec_bash_command",
                                "arguments": json.dumps({"cmd": cmd})
                            }
                        })
                        return tool_calls
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from the response by finding balanced braces
        # This handles JSON with nested objects
        start_idx = clean_response.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(clean_response[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_str = clean_response[start_idx:end_idx]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        if "tool" in parsed and "cmd" in parsed:
                            tool_calls.append({
                                "id": "call_0",
                                "function": {
                                    "name": parsed["tool"],
                                    "arguments": json.dumps({"cmd": parsed["cmd"]})
                                }
                            })
                        elif "command" in parsed:
                            cmd = self._json_to_bash_command(parsed)
                            if cmd:
                                tool_calls.append({
                                    "id": "call_0",
                                    "function": {
                                        "name": "exec_bash_command",
                                        "arguments": json.dumps({"cmd": cmd})
                                    }
                                })
                except json.JSONDecodeError:
                    pass

        return tool_calls

    def _json_to_bash_command(self, parsed: Dict[str, Any]) -> Optional[str]:
        """
        Convert a parsed JSON command to a bash command string.

        Args:
            parsed: The parsed JSON object from model output

        Returns:
            The bash command string, or None if invalid
        """
        command = parsed.get("command")
        if not command:
            return None

        # Build the langgraph command
        cmd_parts = ["langgraph", command]

        # Add flags based on command type
        if command == "new":
            if parsed.get("template"):
                cmd_parts.extend(["--template", parsed["template"]])
            if parsed.get("path"):
                cmd_parts.append(parsed["path"])

        elif command == "dev":
            if parsed.get("port"):
                cmd_parts.extend(["--port", str(parsed["port"])])
            if parsed.get("no_browser"):
                cmd_parts.append("--no-browser")

        elif command == "up":
            if parsed.get("port"):
                cmd_parts.extend(["--port", str(parsed["port"])])
            if parsed.get("watch"):
                cmd_parts.append("--watch")

        elif command == "build":
            if parsed.get("tag"):
                cmd_parts.extend(["-t", parsed["tag"]])

        elif command == "dockerfile":
            if parsed.get("output_path"):
                cmd_parts.extend(["-o", parsed["output_path"]])

        return " ".join(cmd_parts)


class OpenAILLM:
    """
    LLM wrapper for OpenAI-compatible API (e.g., vLLM, TGI).

    Use this when running the model as a server.
    """

    def __init__(self, config: Config):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key,
        )
        print(f"Using API at: {config.api_base_url}")

    def query(
        self,
        messages: Messages,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Query the model via OpenAI-compatible API.

        Args:
            messages: The conversation messages
            tools: Optional list of tool schemas
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, tool_calls)
        """
        kwargs = {
            "model": self.config.api_model_name,
            "messages": messages.to_list(),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if tools:
            kwargs["tools"] = tools

        completion = self.client.chat.completions.create(**kwargs)

        return (
            completion.choices[0].message.content or "",
            completion.choices[0].message.tool_calls or [],
        )


def get_llm(config: Config):
    """
    Factory function to get the appropriate LLM based on config.

    Args:
        config: The application configuration

    Returns:
        Either HuggingFaceLLM or OpenAILLM based on config.use_api
    """
    if config.use_api:
        return OpenAILLM(config)
    else:
        return HuggingFaceLLM(config)
