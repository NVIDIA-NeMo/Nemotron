"""Conversation and inference helpers for the LangGraph CLI agent."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from .commands import (
    CommandValidationError,
    LangGraphInvocation,
    invocation_from_argv,
    invocation_from_payload,
)
from .config import Config


def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    if hasattr(tool_call, "model_dump"):
        serialized = tool_call.model_dump(exclude_none=True)
    elif isinstance(tool_call, Mapping):
        serialized = dict(tool_call)
    else:
        raise TypeError("Tool calls must be mappings or OpenAI SDK model objects.")

    function = serialized.get("function")
    if hasattr(function, "model_dump"):
        serialized["function"] = function.model_dump(exclude_none=True)
    if "type" not in serialized and "function" in serialized:
        serialized["type"] = "function"
    return serialized


class Messages:
    """Conversation history compatible with Chat Completions and local chat templates."""

    def __init__(self, system_message: str = ""):
        self.system_message: dict[str, Any] = {
            "role": "system",
            "content": system_message,
        }
        self.messages: list[dict[str, Any]] = []

    def set_system_message(self, message: str) -> None:
        self.system_message = {"role": "system", "content": message}

    def clear(self) -> None:
        """Clear the turn history while retaining the system prompt."""
        self.messages.clear()

    def add_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self.messages.append({"role": "assistant", "content": message})

    def add_assistant_tool_calls(self, content: Optional[str], tool_calls: list[Any]) -> None:
        """Record the assistant call before its tool-result messages."""
        self.messages.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [_serialize_tool_call(call) for call in tool_calls],
            }
        )

    def add_tool_message(self, message: Any, id: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "content": message if isinstance(message, str) else str(message),
                "tool_call_id": id,
            }
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [self.system_message, *self.messages]

    def to_chat_format(self) -> list[dict[str, str]]:
        """Convert API tool messages to text turns understood by the fine-tuned model."""
        result: list[dict[str, str]] = [
            {"role": "system", "content": str(self.system_message["content"])}
        ]
        for message in self.messages:
            role = message.get("role")
            if role == "tool":
                result.append({"role": "user", "content": f"Tool result: {message['content']}"})
            elif role == "assistant" and message.get("tool_calls"):
                content = message.get("content")
                if not content:
                    first_call = message["tool_calls"][0]
                    content = first_call.get("function", {}).get("arguments", "")
                result.append({"role": "assistant", "content": str(content)})
            else:
                result.append({"role": str(role), "content": str(message.get("content", ""))})
        return result


def _extract_json_object(response: str) -> Optional[dict[str, Any]]:
    clean_response = response.split("</think>")[-1].strip()
    try:
        parsed = json.loads(clean_response)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed = None
        for index, character in enumerate(clean_response):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(clean_response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
    return parsed if isinstance(parsed, dict) else None


def _tool_call_for(invocation: LangGraphInvocation) -> dict[str, Any]:
    return {
        "id": "call_local_0",
        "type": "function",
        "function": {
            "name": "run_langgraph",
            "arguments": json.dumps({"argv": list(invocation.argv)}),
        },
    }


def parse_model_tool_calls(response: str, root_dir: str) -> list[dict[str, Any]]:
    """Parse either trained structured output or an explicit argv object."""
    parsed = _extract_json_object(response)
    if parsed is None:
        return []
    try:
        if "argv" in parsed:
            invocation = invocation_from_argv(parsed["argv"], root_dir)
        else:
            invocation = invocation_from_payload(parsed, root_dir)
    except CommandValidationError:
        return []
    return [_tool_call_for(invocation)]


def _model_load_settings(config: Config, torch: Any) -> tuple[Any, Any]:
    """Select a supported inference dtype and an explicit device map."""
    if config.device == "auto":
        return "auto", "auto"

    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but no MPS device is available.")
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dtype, {"": str(device)}


class HuggingFaceLLM:
    """Local Transformers inference wrapper for the trained checkpoint."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=False,
        )
        dtype, device_map = _model_load_settings(self.config, torch)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        print(f"Model loaded with device map: {device_map}")

    def _input_device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, NotImplementedError):
            return self.model.device

    def query(
        self,
        messages: Messages,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        del tools  # The fine-tuned local model emits the trained JSON format directly.
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages.to_chat_format(),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
            enable_thinking=False,
        ).to(self._input_device())

        do_sample = self.config.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or self.config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        response = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return response, parse_model_tool_calls(response, self.config.root_dir)

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Backward-compatible entry point used by the tutorial notebook."""
        return parse_model_tool_calls(response, self.config.root_dir)


class OpenAILLM:
    """Chat Completions wrapper for OpenAI-protocol model servers."""

    def __init__(self, config: Config):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(base_url=config.api_base_url, api_key=config.api_key)
        print(f"Using API at: {config.api_base_url}")

    def query(
        self,
        messages: Messages,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, list[Any]]:
        kwargs: dict[str, Any] = {
            "model": self.config.api_model_name,
            "messages": messages.to_list(),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_completion_tokens": max_tokens or self.config.max_new_tokens,
            "stream": False,
        }
        if self.config.api_send_thinking_override:
            # vLLM/Nemotron extension; omit it for strict protocol-compatible servers.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = False

        completion = self.client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        content = message.content or ""
        tool_calls = list(message.tool_calls or [])
        if not tool_calls and content:
            tool_calls = parse_model_tool_calls(content, self.config.root_dir)
        return content, tool_calls


def get_llm(config: Config) -> HuggingFaceLLM | OpenAILLM:
    return OpenAILLM(config) if config.use_api else HuggingFaceLLM(config)
