"""Nemotron-3 Nano Omni prompt, history, and response parser.

This agent observes a desktop environment via screenshots and generates
executable pyautogui actions to complete automation tasks through an
OpenAI-compatible vLLM server.

The prompt format matches the Nemotron CUA training specification:
- System prompt with password injection
- Multi-turn history with screenshots (limited sliding window)
- Older steps summarized as text-only history
- Response parsed for ## Action / ## Code blocks
- Coordinates projected from relative [0,1] to absolute pixels
"""

from __future__ import annotations

import asyncio
import ast
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from openai import AsyncOpenAI
from loguru import logger

# Type for the optional streaming delta callback
DeltaCallback = Callable[[str, str], Awaitable[None]]

# ── Prompt templates (from Nemotron CUA training spec) ───────────────────────

INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, task "
    "instruction and previous steps (if provided).\n"
)
STEP_TEMPLATE = "# Step {step_num}:\n"

SYSTEM_PROMPT_THINKING = """\
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}\
"""

SYSTEM_PROMPT_NON_THINKING = """\
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
## Thought
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}\
"""

TEXT_HISTORY_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n"
ASSISTANT_HISTORY_TEMPLATE_THINKING = "<think>\n{thought}\n</think>\n## Action:\n{action}\n"
ASSISTANT_HISTORY_TEMPLATE_NON_THINKING = "## Thought:\n{thought}\n\n## Action:\n{action}\n"


# ── Coordinate projection ────────────────────────────────────────────────────

_PYAUTOGUI_PARAM_NAMES: dict[str, list[str]] = {
    "click": ["x", "y", "clicks", "interval", "button", "duration", "pause"],
    "rightClick": ["x", "y", "duration", "tween", "pause"],
    "middleClick": ["x", "y", "duration", "tween", "pause"],
    "doubleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "tripleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "moveTo": ["x", "y", "duration", "tween", "pause"],
    "dragTo": ["x", "y", "duration", "button", "mouseDownUp", "pause"],
}


def project_pyautogui_coords(code: str, screen_w: int, screen_h: int) -> str:
    """Replace relative (0..1) coords in pyautogui.* calls with absolute px."""
    pattern = re.compile(r"(pyautogui\.\w+\([^\)]*\))")
    out = code
    for full_call in pattern.findall(code):
        m = re.match(r"(pyautogui\.\w+)\((.*)\)", full_call, re.DOTALL)
        if not m:
            continue
        func_name, args_str = m.group(1), m.group(2)
        try:
            parsed = ast.parse(f"f({args_str})").body[0].value
        except SyntaxError:
            continue

        param_names = _PYAUTOGUI_PARAM_NAMES.get(func_name.split(".")[-1], [])
        args: dict[str, Any] = {}
        for idx, arg in enumerate(parsed.args):
            if idx < len(param_names):
                try:
                    args[param_names[idx]] = ast.literal_eval(arg)
                except (ValueError, SyntaxError):
                    pass
        for kw in parsed.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                pass

        if "x" not in args or "y" not in args:
            continue
        try:
            x_rel = float(args["x"])
            y_rel = float(args["y"])
        except (TypeError, ValueError):
            continue
        if x_rel <= 1.0 and y_rel <= 1.0:
            args["x"] = int(round(x_rel * screen_w))
            args["y"] = int(round(y_rel * screen_h))
        else:
            args["x"] = int(round(x_rel))
            args["y"] = int(round(y_rel))

        positional: list[str] = []
        for name in param_names:
            if name in args:
                v = args.pop(name)
                positional.append(repr(v) if isinstance(v, str) else str(v))
            else:
                break
        keyword = [
            f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}"
            for k, v in args.items()
        ]
        out = out.replace(full_call, f"{func_name}({', '.join(positional + keyword)})")
    return out


# ── Response parsing ─────────────────────────────────────────────────────────


@dataclass
class ParsedStep:
    thought: str = ""
    action: str = ""
    code: str = ""           # absolute-coord pyautogui (or "WAIT" / "DONE" / "FAIL")
    original_code: str = ""  # raw fenced code block, pre-projection
    status: str = "continue"  # one of: continue | wait | done | fail | error
    error: Optional[str] = None
    tool_name: str = ""      # populated by HolotronAgent ("click", "write", ...)


def parse_response(
    content: str,
    reasoning_content: str,
    screen_w: int,
    screen_h: int,
    *,
    thinking: bool,
) -> ParsedStep:
    """Parse model output into a ParsedStep."""

    def _strip_think_block(text: str) -> str:
        if "</think>" in text:
            return text.rsplit("</think>", 1)[-1]
        return text

    def _clean_code_section(section: str) -> str:
        section = section.strip()
        fenced = re.findall(
            r"```[A-Za-z0-9_-]*\s*(.*?)\s*```",
            section,
            re.DOTALL,
        )
        if fenced:
            return fenced[-1].strip()
        section = section.strip("`").strip()
        section = re.sub(r"\s*```\s*$", "", section).strip()
        return section

    def _terminal_status(block: str) -> tuple[str, str] | None:
        lower = block.lower()
        if "computer.wait" in lower:
            return "WAIT", "wait"
        if "computer.terminate" in lower:
            if "failure" in lower or "fail" in lower:
                return "FAIL", "fail"
            if "success" in lower:
                return "DONE", "done"
            return "", "error"
        return None

    def _parse_content(candidate: str, thought: str) -> ParsedStep:
        out = ParsedStep(thought=thought)
        candidate = _strip_think_block(candidate).lstrip()

        if not thinking:
            thought_m = re.search(
                r"^##\s*Thought\s*:?[\n\r]+(.*?)(?=^##\s*Action:|^##|\Z)",
                candidate,
                re.DOTALL | re.MULTILINE,
            )
            out.thought = thought_m.group(1).strip() if thought_m else ""

        action_heading = re.search(
            r"^\s*##\s*Action\s*:?",
            candidate,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if not action_heading:
            out.status = "error"
            out.error = "missing action after parsing"
            return out

        candidate = candidate[action_heading.start():]
        next_action = re.search(
            r"\n\s*##\s*Action\s*:?",
            candidate[len(action_heading.group(0)) :],
            flags=re.IGNORECASE,
        )
        if next_action:
            span_end = len(action_heading.group(0)) + next_action.start()
            candidate = candidate[:span_end]

        action_m = re.search(
            r"##\s*Action\s*:?\s*(.*?)(?=\s*##\s*Code\b|\Z)",
            candidate,
            re.DOTALL | re.IGNORECASE,
        )
        if action_m:
            out.action = action_m.group(1).strip()

        code_sections = re.findall(
            r"##\s*Code\s*:?\s*(.*?)(?=\s*##\s*Code\b|\s*##\s*Action\b|\Z)",
            candidate,
            re.DOTALL | re.IGNORECASE,
        )
        if not code_sections:
            fenced_blocks = re.findall(
                r"```(?:code|python|py)?\s*(.*?)\s*```",
                candidate,
                re.DOTALL | re.IGNORECASE,
            )
            code_sections = fenced_blocks

        if not code_sections:
            out.status = "error"
            out.error = "no code block found"
            return out

        executable_blocks: list[str] = []
        original_blocks: list[str] = []
        for raw_section in code_sections:
            block = _clean_code_section(raw_section)
            if not block:
                continue
            original_blocks.append(block)
            terminal = _terminal_status(block)
            if terminal is not None:
                if executable_blocks:
                    # Some responses can append repeated terminate calls
                    # after a valid pyautogui action. The first executable
                    # action is the safe one to run for this step.
                    continue
                code, status = terminal
                if status == "error":
                    out.status = "error"
                    out.error = "computer.terminate without explicit status"
                else:
                    out.code, out.status = code, status
                out.original_code = block
                return out
            executable_blocks.append(block)

        if not executable_blocks:
            out.status = "error"
            out.error = "no executable code found"
            return out

        block = "\n".join(executable_blocks).strip()
        out.original_code = "\n".join(original_blocks).strip()
        out.code = project_pyautogui_coords(block, screen_w, screen_h)
        if not out.action or not out.code:
            out.status = "error"
            out.error = "missing action or code after parsing"
        return out

    thought = reasoning_content.strip() if thinking else ""
    parsed = _parse_content(content, thought)
    if parsed.status != "error":
        return parsed

    # Some OpenAI-compatible vLLM responses may place the formatted action/code
    # block in the reasoning stream instead of the final content field. Only
    # fall back when reasoning contains explicit response headings, so random
    # scratch-pad examples are not executed.
    if thinking:
        action_heading = re.search(r"^##\s*Action\b", reasoning_content, re.MULTILINE)
    else:
        action_heading = None
    if action_heading:
        fallback_thought = reasoning_content[: action_heading.start()].strip() or thought
        fallback = _parse_content(reasoning_content[action_heading.start():], fallback_thought)
        if fallback.status != "error":
            return fallback

    return parsed


# ── The agent ────────────────────────────────────────────────────────────────


@dataclass
class _Turn:
    """One historical agent turn."""
    screenshot_png: bytes
    thought: str
    action: str


@dataclass
class NemotronAgent:
    """Desktop automation agent using Nemotron-3 Nano Omni prompt conventions."""

    api_key: str
    api_base: str = "http://host.docker.internal:8001/v1"
    model: str = "vllm_local"
    max_tokens: int = 20480
    top_p: float = 0.95
    temperature: float = 0.6
    max_image_history_length: int = 3
    password: str = "password"
    thinking: bool = True
    truncate_history_thinking: bool = False
    reasoning_budget: int = 16384
    reasoning_grace_tokens: int = 1024
    model_attempt_timeout: float = 120.0
    max_retry: int = 3
    retry_sleep: float = 5.0
    history: list[_Turn] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def record_tool_result(self, tool_name: str, output: str) -> None:
        """No-op for Nemotron — kept on the base class so the runner can
        unconditionally call it. Holotron uses this to feed
        ``<tool_output>`` back as the next user turn."""
        return None

    @property
    def system_prompt(self) -> str:
        tmpl = SYSTEM_PROMPT_THINKING if self.thinking else SYSTEM_PROMPT_NON_THINKING
        return tmpl.replace("{password}", self.password)

    @property
    def assistant_template(self) -> str:
        return (
            ASSISTANT_HISTORY_TEMPLATE_THINKING
            if self.thinking
            else ASSISTANT_HISTORY_TEMPLATE_NON_THINKING
        )

    def _b64(self, data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    def build_messages(self, instruction: str, current_png: bytes) -> list[dict]:
        """Construct the chat messages array for the API call."""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        instr_block = INSTRUCTION_TEMPLATE.format(instruction=instruction)

        n_with_images = min(len(self.history), max(1, self.max_image_history_length) - 1)
        image_window_start = len(self.history) - n_with_images

        # Text-only history for older steps
        text_history = ""
        if image_window_start > 0:
            parts = []
            for i in range(image_window_start):
                parts.append(
                    STEP_TEMPLATE.format(step_num=i + 1)
                    + TEXT_HISTORY_TEMPLATE.format(
                        thought=self.history[i].thought,
                        action=self.history[i].action,
                    )
                )
            text_history = "# Previous History Actions:\n" + "\n".join(parts)

        # Image-included history (recent steps)
        for i in range(image_window_start, len(self.history)):
            user_text = instr_block
            if i == image_window_start and text_history:
                user_text += text_history + "\n"
            user_text += f"You are currently on Step {i + 1}.\n"
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self._b64(self.history[i].screenshot_png)}"},
                    },
                    {"type": "text", "text": user_text},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": self.assistant_template.format(
                    thought=self.history[i].thought,
                    action=self.history[i].action,
                ),
            })

        # Current step
        current_text = instr_block
        if n_with_images == 0 and text_history:
            current_text += text_history + "\n"
        current_text += f"You are currently on Step {len(self.history) + 1}.\n"

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self._b64(current_png)}"},
                },
                {"type": "text", "text": current_text},
            ],
        })
        return messages

    def build_extra_body(self) -> dict[str, Any]:
        """Build reasoning controls for the local or remote vLLM path.

        Nemotron-3 Nano Omni uses thinking mode for CUA quality. The model docs
        and vLLM launch path both describe the same contract: enable thinking
        in chat template kwargs, give the model a reasoning budget, and reserve
        a small grace window so generation can transition from reasoning into
        the final `## Action` / `## Code` answer instead of ending at length.
        """

        if not self.thinking:
            return {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "truncate_history_thinking": self.truncate_history_thinking,
                }
            }

        return {
            "thinking_token_budget": self.reasoning_budget + self.reasoning_grace_tokens,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_budget": self.reasoning_budget,
                "truncate_history_thinking": self.truncate_history_thinking,
            },
        }

    async def step(
        self,
        instruction: str,
        screenshot_png: bytes,
        screen_size: tuple[int, int],
        *,
        delta_callback: Optional[DeltaCallback] = None,
    ) -> ParsedStep:
        """Send one OpenAI-compatible inference request and parse the result.

        Retry logic:
        - Retry request, timeout, finish-reason, and parse failures
        - Each model attempt has a bounded wall-clock timeout
        - Temperature floor bumped to 0.2 on subsequent attempts
        - finish_reason must be "stop" for a valid response
        """

        messages = self.build_messages(instruction, screenshot_png)

        client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

        max_retry = max(1, self.max_retry)
        last_error = "unknown"
        parsed: Optional[ParsedStep] = None

        for attempt in range(max_retry):
            # Match cua_demo: bump temperature floor on retries so the model
            # isn't deterministically stuck producing the same malformed output
            temperature = self.temperature if attempt == 0 else max(0.2, self.temperature)

            if delta_callback is not None and attempt > 0:
                # Surface a visible marker into the live stream so the UI
                # shows that we're retrying and why.
                try:
                    await delta_callback(
                        f"\n\n[retry {attempt + 1}/{max_retry}: {last_error}]\n\n",
                        "",
                    )
                except Exception:
                    pass

            if delta_callback is not None:
                try:
                    await delta_callback(
                        (
                            f"\n\n[model attempt {attempt + 1}/{max_retry}; "
                            f"timeout {self.model_attempt_timeout:g}s]\n\n"
                        ),
                        "",
                    )
                except Exception:
                    pass

            try:
                extra_body = self.build_extra_body()

                logger.info(
                    "model attempt {}/{} (timeout={}s, temperature={})",
                    attempt + 1,
                    max_retry,
                    self.model_attempt_timeout,
                    temperature,
                )

                content, reasoning, finish_reason = await asyncio.wait_for(
                    self._stream_completion(
                        client,
                        messages,
                        temperature,
                        extra_body,
                        delta_callback,
                    ),
                    timeout=self.model_attempt_timeout,
                )

                # Match cua_demo: check finish_reason is "stop"
                if finish_reason not in (None, "stop"):
                    last_error = f"unexpected finish_reason={finish_reason}"
                    logger.warning(
                        "attempt {}/{}: {}; retrying",
                        attempt + 1,
                        max_retry,
                        last_error,
                    )
                    if attempt + 1 < max_retry:
                        await asyncio.sleep(self.retry_sleep)
                    continue

                parsed = parse_response(
                    content, reasoning, screen_size[0], screen_size[1],
                    thinking=self.thinking,
                )

                if parsed.status != "error":
                    break
                last_error = parsed.error or "parse error"
                logger.warning(
                    "attempt {}/{}: parse error: {}; retrying",
                    attempt + 1,
                    max_retry,
                    last_error,
                )
                if attempt + 1 < max_retry:
                    await asyncio.sleep(self.retry_sleep)

            except asyncio.TimeoutError:
                last_error = f"model timed out after {self.model_attempt_timeout:g}s"
                logger.warning(
                    "attempt {}/{}: {}; retrying",
                    attempt + 1,
                    max_retry,
                    last_error,
                )
                if attempt + 1 < max_retry:
                    await asyncio.sleep(self.retry_sleep)

            except Exception as e:
                last_error = f"request failed: {e}"
                logger.error(
                    "attempt {}/{}: {}; retrying",
                    attempt + 1,
                    max_retry,
                    last_error,
                )
                if attempt + 1 < max_retry:
                    await asyncio.sleep(self.retry_sleep)
                continue

        if parsed is None or parsed.status == "error":
            return ParsedStep(
                status="error",
                error=f"all {max_retry} attempts failed: {last_error}",
            )

        # Record history for successful steps
        if parsed.status in {"continue", "wait"}:
            self.history.append(
                _Turn(screenshot_png=screenshot_png, thought=parsed.thought, action=parsed.action)
            )
        return parsed

    async def _stream_completion(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
        temperature: float,
        extra_body: dict[str, Any],
        delta_callback: Optional[DeltaCallback],
    ) -> tuple[str, str, Optional[str]]:
        """Run one streaming model request and return content, reasoning, finish."""

        reasoning_buf: list[str] = []
        content_buf: list[str] = []
        finish_reason: Optional[str] = None

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
            extra_body=extra_body if extra_body else None,
        )

        async with stream:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # Handle reasoning content (thinking tokens). vLLM variants
                # have used both field names.
                r_delta = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or ""
                )
                c_delta = delta.content or ""

                if r_delta:
                    reasoning_buf.append(r_delta)
                if c_delta:
                    content_buf.append(c_delta)

                if (r_delta or c_delta) and delta_callback:
                    try:
                        await delta_callback(r_delta, c_delta)
                    except Exception:
                        logger.warning("delta_callback raised; continuing stream")

                # Capture finish_reason from the final chunk
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        return "".join(content_buf), "".join(reasoning_buf), finish_reason
