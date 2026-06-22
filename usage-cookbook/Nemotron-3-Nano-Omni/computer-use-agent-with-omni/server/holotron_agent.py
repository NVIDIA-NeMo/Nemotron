"""Holotron-3-Nano agent — H Company agent-loop edition.

Implements the official H Company agent contract documented in the
``holo-nano`` reference harness:

- 12-tool union (``update_plan``, ``write_desktop``, ``click_desktop``,
  ``double_click_desktop``, ``drag_to_desktop``, ``scroll_desktop``,
  ``move_to_desktop``, ``key_down_desktop``, ``key_up_desktop``,
  ``hotkey_desktop``, ``hold_and_tap_key_desktop``, ``answer``).
- Single JSON output per step: ``{note, thought, tool_call: {...}}``.
- Tool union enforced via vLLM ``structured_outputs`` extra_body.
- Coordinates emitted as integers in ``[0, 1000]``, scaled to absolute
  pixels here before pyautogui code is sent to the desktop container.
- User observations wrapped in ``<observation>...</observation>``.
- Tool results injected back as user messages wrapped in
  ``<tool_output tool="name">...</tool_output>`` (or ``<error tool=...>``
  on failure).
- Image budget: keep at most the last N screenshots; older ``image_url``
  chunks are demoted to ``[Image omitted by context cleaning]`` text.
- Only the parsed JSON (``Step.model_dump_json()``) is pushed back into
  history — never raw model output, per H Company docs.

Public surface mirrors :class:`server.agent.NemotronAgent` so the same
:class:`server.agent_runner.AgentRunner` drives both:

- ``reset()``
- ``step(instruction, screenshot_png, screen_size, *, delta_callback)``
- ``record_tool_result(tool_name, output)`` (stores a ``<tool_output>``
  for the next step).
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.agent import DeltaCallback, ParsedStep


# ── Pydantic Step union (matches H Company holo-nano harness tools.py) ──────

MouseButton = Literal["left", "right", "middle"]
ScrollDirection = Literal["up", "down", "left", "right"]
GoalStatus = Literal["todo", "running", "done", "failed"]


class _Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Goal(_Tool):
    """One step of the agent's task plan."""

    title: str = Field(
        description="Action-oriented title beginning with a verb."
    )
    status: GoalStatus = Field(description="Current status of this goal.")


class UpdatePlan(_Tool):
    """Create or update the agent's task plan."""

    tool_name: Literal["update_plan"]
    goals: list[Goal] = Field(
        min_length=1,
        max_length=20,
        description="The full list of goals; only one may be 'running' at a time.",
    )


class WriteDesktop(_Tool):
    """Type text at the current cursor position (no click first)."""

    tool_name: Literal["write_desktop"]
    content: str = Field(description="The text to type.")
    press_enter: bool = Field(
        default=False, description="Press Enter immediately after typing."
    )
    overwrite: bool = Field(
        default=False,
        description="Select-all and delete the field's contents before typing.",
    )


class ClickDesktop(_Tool):
    """Click at (x, y) on a UI element."""

    tool_name: Literal["click_desktop"]
    element: str = Field(description="Detailed description of the target element.")
    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000].")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000].")
    button: MouseButton = Field(default="left")


class DoubleClickDesktop(_Tool):
    """Double-click at (x, y) on a UI element."""

    tool_name: Literal["double_click_desktop"]
    element: str = Field(description="Detailed description of the target element.")
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class DragToDesktop(_Tool):
    """Drag from the current cursor position to (x, y).

    Useful for selecting text, moving objects, or drawing. Move the
    cursor with ``move_to_desktop`` first to set the drag origin.
    """

    tool_name: Literal["drag_to_desktop"]
    element: str = Field(
        description=(
            "Description of the destination element / area the drag ends on."
        )
    )
    x: int = Field(ge=0, le=1000, description="Destination X in [0, 1000].")
    y: int = Field(ge=0, le=1000, description="Destination Y in [0, 1000].")


class ScrollDesktop(_Tool):
    """Move to (x, y) and scroll in a direction."""

    tool_name: Literal["scroll_desktop"]
    element: str = Field(description="Description of the region to scroll.")
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    direction: ScrollDirection
    scroll_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of mouse wheel clicks to scroll.",
    )


class MoveToDesktop(_Tool):
    """Move the mouse cursor to (x, y) without clicking."""

    tool_name: Literal["move_to_desktop"]
    element: str = Field(description="Detailed description of the target element.")
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class KeyDownDesktop(_Tool):
    """Press a key down without releasing it (must be paired with key_up_desktop)."""

    tool_name: Literal["key_down_desktop"]
    key: str = Field(description="Key to press, e.g. 'shift', 'ctrl', 'a'.")


class KeyUpDesktop(_Tool):
    """Release a previously pressed key."""

    tool_name: Literal["key_up_desktop"]
    key: str = Field(description="Key to release.")


class HotkeyDesktop(_Tool):
    """Press multiple keys in order, release in reverse (e.g. ['ctrl','c'])."""

    tool_name: Literal["hotkey_desktop"]
    keys: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Keys to chord together, e.g. ['ctrl','shift','t'].",
    )
    repeat_count: int = Field(
        default=1, ge=1, le=20, description="Repeat the hotkey this many times."
    )


class HoldAndTapKeyDesktop(_Tool):
    """Hold modifier keys while tapping a sequence of keys."""

    tool_name: Literal["hold_and_tap_key_desktop"]
    hold_keys: list[str] = Field(
        min_length=1, max_length=3, description="Keys to hold down."
    )
    tap_keys: list[str] = Field(
        min_length=1, max_length=5, description="Keys to tap once each, in order."
    )


class Answer(_Tool):
    """Terminate the task and return the final answer to the user."""

    tool_name: Literal["answer"]
    content: str = Field(description="The final answer (Markdown-formatted).")


ToolCall = (
    UpdatePlan
    | WriteDesktop
    | ClickDesktop
    | DoubleClickDesktop
    | DragToDesktop
    | ScrollDesktop
    | MoveToDesktop
    | KeyDownDesktop
    | KeyUpDesktop
    | HotkeyDesktop
    | HoldAndTapKeyDesktop
    | Answer
)


class Step(BaseModel):
    """The structured output the model emits each turn."""

    model_config = ConfigDict(extra="forbid")
    note: Optional[str] = Field(
        default=None,
        description=(
            "Persistent notes — extract task-relevant facts from the current "
            "screen. Set to null when nothing new is worth recording."
        ),
    )
    thought: str = Field(
        description="Brief reasoning about progress and the next action."
    )
    tool_call: ToolCall = Field(description="Exactly one tool to invoke.")


# ── System prompt (port of holo-nano/prompts/system.j2) ─────────────────────

# Schema is computed once at import time and embedded in the system prompt
# so the model sees the same shape it is constrained against.
_SCHEMA: dict[str, Any] = Step.model_json_schema()


_SYSTEM_PROMPT_TEMPLATE = """\
You are HoloNano, a navigation agent designed by H Company.

# Core principles

You operate autonomously within a local desktop environment where you interact with users.

1. **Thoroughness over speed**: Accuracy matters; speed does not. You have no time pressure to complete the task. Move deliberately: observe carefully, think strategically, act precisely. Do not shortcut your way to the answer. Verify all critical steps thoroughly.
2. **Evidence over inference**: Base every statement or action on what you can see or verify. When uncertain, gather more data.
3. **Persistence through adaptation**: When blocked, analyze why. Refine your approach, try alternative methods, or pivot to new strategies until momentum returns.
4. **Methodical progress**: Each action should clarify, advance, or eliminate possibilities. Avoid aimless steps. Plan clearly, execute cleanly, verify always.
5. **Structured awareness**: Track what's known, pending, and missing. Record essential evidence before changing windows or applications; your notes are your only lasting memory.
6. **Closure with care**: Conclude only when the targeted result is verifiably reached or all information is gathered beyond doubt.
7. **Complete retention**: Make your answer as complete and verifiable as possible. Everything else (notes, thoughts, intermediate steps) will be lost forever.

# Workflow

You are an agent using the ReAct pattern to iteratively (1) observe the environment, (2) reason about next steps and (3) act in a series of steps.
You must emit at each step a single JSON object with keys `note`, `thought` and `tool_call`.

## `note`

GOAL: Persist information from the previous observation.
- Only the last few screenshots are kept in memory; notes persist. Before changing screens or windows, extract ALL task-relevant information into your note (you cannot return to review old screenshots).
- Extract visible information and evidence from the current environment state in relation to your task.
- Capture ALL relevant data in detail: values, short text excerpts, tables, context, application names, window titles, file paths, dialog messages, button states, timestamps, etc.
- Notes build on top of previous notes. New notes must be distinct from previous notes. Never restate old info. Set to null if nothing new.

## `thought`

GOAL: Reason strategically about next steps.
- Assess whether your past tool call was successful or not.
- Detect loops: Have you performed this same action before? If yes and it failed previously, you MUST pivot to a different approach.
- Assess progress: advancing toward completion, temporarily blocked, or fundamentally stuck.
- Identify remaining gaps: information needed, actions to perform, etc.
- Reason about next steps: continue, shortcut, backtrack, pivot, change strategy, etc.
- Select the optimal next tool to call.
- Reason about the arguments to pass to the tool.

## `tool_call`

GOAL: Emit a single tool call in JSON format strictly following the provided schema.
- Select one of the available tools as chosen in your reasoning.
- Each tool's schema specifies required vs optional arguments and their types (consult it before calling).
- Call the tool and its arguments in JSON format strictly following the provided schema.
- Invalid tool calls will not be executed.
- Emit only a single tool call per step.

# Tools

## Desktop Control Tools

You control a live desktop environment via dedicated tools (mouse, keyboard, hotkeys).
At each step, you will be provided with a fresh screenshot of the current desktop screen.

Coordinates `x` and `y` are integers in `[0, 1000]`, normalized against the
screenshot. The origin is the top-left corner. Coordinates will be scaled
to absolute pixels by the host driver before execution.

Available desktop tools:
- `click_desktop(element, x, y, button="left")` — single click.
- `double_click_desktop(element, x, y)` — double click.
- `move_to_desktop(element, x, y)` — move cursor only, no click.
- `drag_to_desktop(element, x, y)` — drag from CURRENT cursor position to (x, y); call `move_to_desktop` first to set the drag origin (use this to select text, move files, draw, etc.).
- `scroll_desktop(element, x, y, direction, scroll_size=10)` — move to (x, y) then scroll up/down/left/right by `scroll_size` mouse wheel clicks (max 100).
- `write_desktop(content, press_enter=False, overwrite=False)` — type text at the current focus; `overwrite=True` selects-all then deletes first; `press_enter=True` presses Enter when done.
- `hotkey_desktop(keys, repeat_count=1)` — press multiple keys in order then release them in reverse, e.g. `keys=["ctrl","c"]`. Use this for keyboard shortcuts.
- `hold_and_tap_key_desktop(hold_keys, tap_keys)` — hold modifiers while tapping a sequence (e.g. hold `["ctrl"]`, tap `["a","c"]`).
- `key_down_desktop(key)` / `key_up_desktop(key)` — fine-grained press/release; pair them.

### Critical blockers

Critical blockers are obstacles that prevent progress on the current path. Handle them immediately, then pivot to alternative approaches. Only report a blocker as insurmountable after exhausting all viable alternatives: alternative methods, different navigation paths, simplified approaches, or workarounds.

1. **Permission/authentication dialogs**: Handle or dismiss as needed. The sudo / lock password for this machine is `{password}` — use it only when the task explicitly requires it.
2. **Application not responding**: Wait briefly, then force quit and restart if necessary.
3. **Missing applications**: Attempt to locate or launch from alternative paths; report if unavailable and required.
4. **System dialogs/alerts**: Read carefully, dismiss or acknowledge appropriately to proceed.
5. **Locked files/permissions**: Try alternative locations or methods; report only if specifically required resource is inaccessible.

### Strategic interaction

Desktop applications expose various UI elements and shortcuts for efficient interaction. Use their logical structure and conventions.
- **Keyboard shortcuts**: Leverage common operations (Ctrl+C for copy, Ctrl+V for paste, Ctrl+S for save, Ctrl+F for find, etc.).
- **Application launching**: Click the dock/taskbar icons or use the application launcher.
- **Window management**: Use hotkeys for switching apps (Alt+Tab), or click window controls.
- **File operations**: Navigate file managers, use drag-and-drop for moving files (`move_to_desktop` then `drag_to_desktop`), or keyboard shortcuts for copy/paste/save operations.
- **Menu navigation**: Click menu items or use keyboard navigation (Alt+key).
- **Text input**: Text can be typed at the current cursor location. Focus input fields by clicking them first when needed.
- **Scrolling**: Navigate long content with `scroll_desktop` or `pagedown`/`pageup` keys.
- **Selecting text**: `move_to_desktop` to the start, then `drag_to_desktop` to the end. Or click then `hold_and_tap_key_desktop(["shift"], ["end"])`.

### Element Localization

Tools include an `element` string to describe the target UI element for traceability and clarity. Provide a clear, uniquely identifying description, including:
1. **Visible text**: Exact label, button text, placeholder text, heading, etc.
2. **Visual attributes**: Color, icon, shape, size, state, etc.
3. **Position**: Rough indication of location, quadrant, container, etc.
4. **Context anchors**: Description of nearby element(s) and their relative position.

Good example: "Blue 'Search' button with magnifying glass icon, top-right of header, next to login link"
Bad example: "search button"

## Planning

Use `update_plan` to create, track, and adapt your task plan.

- **Reasoning**: Before calling the tool, analyze in your `thought`: consider task breakdown, current gaps, dependencies, and complexity when creating an initial plan; diagnose failures, invalidated assumptions, and alternative routes when replanning.
- **When to plan**: Create a plan within your first 2-3 steps for non-trivial tasks (>5 steps or multiple sources). You may perform 1-2 exploratory actions first to understand the landscape, then plan based on findings.
- **Goal design**: Design goals that are action-oriented with concise titles beginning with a verb, achievable with concrete target states or success criteria, progressive where each goal unlocks the next, and right-sized with fewer goals for simple tasks and more for complex tasks.
- **Tracking progress**: Keep status up-to-date to stay on track and detect when stuck. When you complete a goal, call `update_plan` with the full goal list where the completed goal has status='done' and the next goal has status='running'. Always include all goals in your list. Only one goal should be 'running' at a time. Finish only when all goals are 'done'.
- **Replanning**: Replan immediately when you've attempted the same approach twice and failed both times, when new facts invalidate initial assumptions or partial results, when the user changes the task scope or priorities, or when a goal becomes impossible to complete (not just difficult). Include your done/failed goals plus new goals in the list.

## Answer Tool

Calling `answer` will terminate your task and send the final answer to the user.
This is an irreversible action; you must be certain beyond a doubt that all task requirements are verifiably met before terminating.

### Termination criteria

Call `answer` only when ALL of the following are true:

1. **Task requirements met**: All requested information gathered OR target state reached and verified
2. **Evidence captured**: Verifiable proof exists in your notes (window/app names, window titles, file paths, dialog texts, confirmations, data values)
3. **Exploration exhausted**: No viable alternative approaches or methods remain untried
4. **Contradictions resolved**: Conflicting information has been cross-checked or acknowledged as unresolvable
5. **Actions confirmed**: Any file operations, form submissions, or system changes show clear success confirmation
6. **Notes complete**: All visible observations from the current screen recorded (screenshots are ephemeral)

Do NOT call `answer` if:
- Alternative approaches exist that might accomplish the task
- You have not verified critical results when feasible
- Current screen contains unrecorded information relevant to the task
- A blocker was encountered but workarounds remain unexplored
- You reached a state but lack concrete evidence

When in doubt about termination, ask: "If I had to defend this answer in court, do I have the evidence?" If no, continue working.

### How to call the `answer` tool

Your final answer is the ONLY artifact forwarded to the user; everything else is lost.

**Rules**
1. Include as much information as necessary for a self-contained, verifiable output. Address every task requirement explicitly.
2. Prioritize completeness over brevity; avoid omitting critical details.
3. Synthesize ALL relevant information you have observed into one coherent answer.
4. Maximize retention (all that is not passed as an answer is lost forever).
5. Assume the reader has zero context about your process. Your answer must be self-contained and defensible.

# System settings

The current date is {start_time}
Maximal budget: {max_steps} steps or {max_time_s}s
Environment: Local desktop ({os_name})
Effort level: High
Persona: Meticulous, precise, detail-oriented
Output format: JSON
"""


def _build_system_prompt(
    *,
    password: str,
    max_steps: int,
    max_time_s: int,
    os_name: str = "Linux",
) -> str:
    body = _SYSTEM_PROMPT_TEMPLATE.format(
        start_time=datetime.datetime.now().strftime("%Y-%m-%d"),
        max_steps=max_steps,
        max_time_s=max_time_s,
        os_name=os_name,
        password=password,
    )
    return (
        body
        + "\n<output_format>\n```json\n"
        + json.dumps(_SCHEMA)
        + "\n```\n</output_format>\n"
    )


# ── pyautogui code rendering ────────────────────────────────────────────────

def _scale(c: int, dim: int) -> int:
    """Scale a normalized [0, 1000] integer coord to absolute pixels."""
    return int(round(max(0, min(int(c), 1000)) / 1000.0 * dim))


def _q(s: str) -> str:
    """Python-source-safe literal for typewrite/keys."""
    return repr(s)


def _render_click(args: ClickDesktop, w: int, h: int) -> str:
    return (
        f"pyautogui.click({_scale(args.x, w)}, {_scale(args.y, h)},"
        f" button={args.button!r})"
    )


def _render_double_click(args: DoubleClickDesktop, w: int, h: int) -> str:
    return (
        f"pyautogui.doubleClick({_scale(args.x, w)}, {_scale(args.y, h)},"
        f" button='left')"
    )


def _render_drag_to(args: DragToDesktop, w: int, h: int) -> str:
    x, y = _scale(args.x, w), _scale(args.y, h)
    return (
        "import time\n"
        "pyautogui.mouseDown(button='left')\n"
        "time.sleep(0.05)\n"
        f"pyautogui.moveTo({x}, {y}, duration=0.2)\n"
        "time.sleep(0.05)\n"
        "pyautogui.mouseUp(button='left')"
    )


def _render_scroll(args: ScrollDesktop, w: int, h: int) -> str:
    x, y = _scale(args.x, w), _scale(args.y, h)
    sz = int(args.scroll_size)
    lines = [f"pyautogui.moveTo({x}, {y})"]
    if args.direction == "up":
        lines.append(f"pyautogui.scroll({sz})")
    elif args.direction == "down":
        lines.append(f"pyautogui.scroll(-{sz})")
    elif args.direction == "left":
        lines.append(f"pyautogui.hscroll(-{sz})")
    elif args.direction == "right":
        lines.append(f"pyautogui.hscroll({sz})")
    return "\n".join(lines)


def _render_move_to(args: MoveToDesktop, w: int, h: int) -> str:
    return f"pyautogui.moveTo({_scale(args.x, w)}, {_scale(args.y, h)})"


def _render_write(args: WriteDesktop) -> str:
    lines: list[str] = []
    if args.overwrite:
        lines.append("pyautogui.hotkey('ctrl', 'a')")
        lines.append("pyautogui.press('delete')")
    lines.append(f"pyautogui.typewrite({_q(args.content)}, interval=0.02)")
    if args.press_enter:
        lines.append("pyautogui.press('enter')")
    return "\n".join(lines)


def _render_key_down(args: KeyDownDesktop) -> str:
    return f"pyautogui.keyDown({_q(args.key.lower())})"


def _render_key_up(args: KeyUpDesktop) -> str:
    return f"pyautogui.keyUp({_q(args.key.lower())})"


def _render_hotkey(args: HotkeyDesktop) -> str:
    keys = ", ".join(_q(k.lower()) for k in args.keys)
    call = f"pyautogui.hotkey({keys})"
    if args.repeat_count == 1:
        return call
    return f"for _ in range({int(args.repeat_count)}):\n    {call}"


def _render_hold_and_tap(args: HoldAndTapKeyDesktop) -> str:
    lines: list[str] = []
    for k in args.hold_keys:
        lines.append(f"pyautogui.keyDown({_q(k.lower())})")
    for k in args.tap_keys:
        lines.append(f"pyautogui.press({_q(k.lower())})")
    for k in reversed(args.hold_keys):
        lines.append(f"pyautogui.keyUp({_q(k.lower())})")
    return "\n".join(lines)


def _render_action_text(tc: Any) -> str:
    """Human-readable rendering for ParsedStep.action (UI display)."""
    if isinstance(tc, ClickDesktop):
        return f"Click {tc.element} at ({tc.x},{tc.y}) [{tc.button}]"
    if isinstance(tc, DoubleClickDesktop):
        return f"Double-click {tc.element} at ({tc.x},{tc.y})"
    if isinstance(tc, DragToDesktop):
        return f"Drag to ({tc.x},{tc.y}) — {tc.element}"
    if isinstance(tc, ScrollDesktop):
        return (
            f"Scroll {tc.direction} ×{tc.scroll_size} at ({tc.x},{tc.y})"
            f" [{tc.element}]"
        )
    if isinstance(tc, MoveToDesktop):
        return f"Move to ({tc.x},{tc.y}) — {tc.element}"
    if isinstance(tc, WriteDesktop):
        suffix = " + Enter" if tc.press_enter else ""
        prefix = "Overwrite with " if tc.overwrite else "Type "
        return f"{prefix}{tc.content!r}{suffix}"
    if isinstance(tc, KeyDownDesktop):
        return f"Key down {tc.key}"
    if isinstance(tc, KeyUpDesktop):
        return f"Key up {tc.key}"
    if isinstance(tc, HotkeyDesktop):
        n = f" ×{tc.repeat_count}" if tc.repeat_count > 1 else ""
        return f"Hotkey {'+'.join(tc.keys)}{n}"
    if isinstance(tc, HoldAndTapKeyDesktop):
        return f"Hold {'+'.join(tc.hold_keys)} tap {','.join(tc.tap_keys)}"
    if isinstance(tc, UpdatePlan):
        running = next(
            (g.title for g in tc.goals if g.status == "running"), None
        )
        done = sum(1 for g in tc.goals if g.status == "done")
        return (
            f"Update plan ({done}/{len(tc.goals)} done; "
            f"running: {running or '—'})"
        )
    if isinstance(tc, Answer):
        return f"Answer: {tc.content}"
    return str(tc)


def _to_parsed_step(step: Step, screen_w: int, screen_h: int) -> ParsedStep:
    """Translate a parsed Step into the demo's :class:`ParsedStep`."""
    tc = step.tool_call
    parts: list[str] = []
    if step.note:
        parts.append(f"Note: {step.note}")
    if step.thought:
        parts.append(step.thought)
    thought_text = "\n".join(parts)

    # Terminal — answer ends the run.
    if isinstance(tc, Answer):
        return ParsedStep(
            thought=thought_text,
            action=_render_action_text(tc),
            code="DONE",
            original_code="answer",
            status="done",
            tool_name="answer",
        )

    # Plan-only — no desktop action; treat as a wait so the runner
    # re-screenshots and lets the model take its next concrete step.
    if isinstance(tc, UpdatePlan):
        return ParsedStep(
            thought=thought_text,
            action=_render_action_text(tc),
            code="WAIT",
            original_code="update_plan",
            status="wait",
            tool_name="update_plan",
        )

    # Desktop tools — emit pyautogui source for run_pyautogui().
    if isinstance(tc, ClickDesktop):
        code = _render_click(tc, screen_w, screen_h)
    elif isinstance(tc, DoubleClickDesktop):
        code = _render_double_click(tc, screen_w, screen_h)
    elif isinstance(tc, DragToDesktop):
        code = _render_drag_to(tc, screen_w, screen_h)
    elif isinstance(tc, ScrollDesktop):
        code = _render_scroll(tc, screen_w, screen_h)
    elif isinstance(tc, MoveToDesktop):
        code = _render_move_to(tc, screen_w, screen_h)
    elif isinstance(tc, WriteDesktop):
        code = _render_write(tc)
    elif isinstance(tc, KeyDownDesktop):
        code = _render_key_down(tc)
    elif isinstance(tc, KeyUpDesktop):
        code = _render_key_up(tc)
    elif isinstance(tc, HotkeyDesktop):
        code = _render_hotkey(tc)
    elif isinstance(tc, HoldAndTapKeyDesktop):
        code = _render_hold_and_tap(tc)
    else:
        return ParsedStep(
            thought=thought_text,
            status="error",
            error=f"unknown tool {getattr(tc, 'tool_name', '?')}",
        )

    return ParsedStep(
        thought=thought_text,
        action=_render_action_text(tc),
        code=code,
        original_code=code,
        status="continue",
        tool_name=tc.tool_name,
    )


# ── Image-budget trimming (mirrors holo-nano agent.py) ──────────────────────

def _trim_to_last_n_images(messages: list[dict], n: int = 3) -> None:
    """Demote older image_url chunks to a placeholder text marker.

    Mirrors the holo-nano harness behaviour where evicted observations
    become "[Image omitted by context cleaning]" within the same
    ``<observation>...</observation>`` wrapper.
    """
    seen = 0
    for msg in reversed(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for chunk in msg["content"]:
            if chunk.get("type") != "image_url":
                continue
            seen += 1
            if seen > n:
                chunk["type"] = "text"
                chunk["text"] = "[Image omitted by context cleaning]"
                chunk.pop("image_url", None)


# ── Agent ───────────────────────────────────────────────────────────────────

@dataclass
class HolotronAgent:
    """Desktop automation agent powered by Hcompany/Holotron-3-Nano via vLLM."""

    api_key: str
    api_base: str = "http://host.docker.internal:8011/v1"
    model: str = "holotron_local"
    max_tokens: int = 4096
    top_p: float = 0.95
    temperature: float = 0.8
    max_image_history_length: int = 3
    password: str = "password"

    # Plan/step budget surfaced into the system prompt.
    max_steps: int = 50
    max_time_s: int = 1800
    os_name: str = "Linux"
    reasoning_effort: str = "medium"  # vLLM extra_body field

    # Accepted-but-ignored knobs so main.py can pass the same kwargs as
    # for NemotronAgent without branching:
    thinking: bool = True
    truncate_history_thinking: bool = False
    reasoning_budget: int = 16384
    reasoning_grace_tokens: int = 1024
    model_attempt_timeout: float = 120.0
    max_retry: int = 3
    retry_sleep: float = 5.0

    # Internal mutable state. Tuple is (tool_name, output, is_error).
    _messages: list[dict] = field(default_factory=list)
    _pending_tool_result: Optional[tuple[str, str, bool]] = None
    _instruction_seeded: bool = False

    def reset(self) -> None:
        self._messages = [{
            "role": "system",
            "content": _build_system_prompt(
                password=self.password,
                max_steps=self.max_steps,
                max_time_s=self.max_time_s,
                os_name=self.os_name,
            ),
        }]
        self._pending_tool_result = None
        self._instruction_seeded = False

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    def _append_observation(self, png: bytes) -> None:
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "<observation>\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._b64(png)}"
                    },
                },
                {"type": "text", "text": "\n</observation>"},
            ],
        })

    def _flush_pending_tool_result(self) -> None:
        if self._pending_tool_result is None:
            return
        tool_name, output, is_error = self._pending_tool_result
        self._pending_tool_result = None
        tag = "error" if is_error else "tool_output"
        self._messages.append({
            "role": "user",
            "content": f'<{tag} tool="{tool_name}">\n{output}\n</{tag}>',
        })

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "structured_outputs": {"json": _SCHEMA},
            "reasoning_effort": self.reasoning_effort,
        }
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def record_tool_result(self, tool_name: str, output: str) -> None:
        """Store the last tool's stdout/stderr for emission on the next turn."""
        truncated = (output or "")[:2000]
        is_error = truncated.startswith("error:")
        # Map non-desktop tool names (which never reach the runner's
        # run_pyautogui path) to a default tag.
        if not tool_name:
            tool_name = "pyautogui"
        self._pending_tool_result = (tool_name, truncated, is_error)

    # ── Main loop entry point ──────────────────────────────────────────────

    async def step(
        self,
        instruction: str,
        screenshot_png: bytes,
        screen_size: tuple[int, int],
        *,
        delta_callback: Optional[DeltaCallback] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ParsedStep:
        if not self._messages:
            self.reset()

        # Seed the user's task once at the start of the conversation.
        if not self._instruction_seeded:
            self._messages.append({
                "role": "user",
                "content": f"Task: {instruction}",
            })
            self._instruction_seeded = True

        # Render any queued tool result before the next observation so the
        # model can react to outcomes from its previous action.
        self._flush_pending_tool_result()
        self._append_observation(screenshot_png)
        _trim_to_last_n_images(self._messages, n=self.max_image_history_length)

        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        owns_client = client is None
        client = client or httpx.AsyncClient(
            timeout=self.model_attempt_timeout + 5,
            verify=False,
        )

        max_retry = max(1, self.max_retry)
        last_error = "unknown"
        parsed_step: Optional[Step] = None
        finish_reason: Optional[str] = None

        try:
            for attempt in range(max_retry):
                temperature = (
                    self.temperature
                    if attempt == 0
                    else max(0.4, self.temperature - 0.1)
                )
                payload = {
                    "model": self.model,
                    "messages": self._messages,
                    "temperature": temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "stream": True,
                }
                payload.update(self._extra_body())

                if delta_callback is not None:
                    try:
                        if attempt > 0:
                            await delta_callback(
                                f"\n\n[retry {attempt + 1}/{max_retry}: "
                                f"{last_error}]\n\n",
                                "",
                            )
                        await delta_callback(
                            (
                                f"\n\n[Holotron attempt "
                                f"{attempt + 1}/{max_retry}; timeout "
                                f"{self.model_attempt_timeout:g}s]\n\n"
                            ),
                            "",
                        )
                    except Exception:
                        pass

                try:
                    logger.info(
                        "Holotron attempt {}/{} (endpoint={}, timeout={}s, "
                        "temperature={})",
                        attempt + 1,
                        max_retry,
                        endpoint,
                        self.model_attempt_timeout,
                        temperature,
                    )
                    content, _reasoning, finish_reason = await asyncio.wait_for(
                        self._stream_completion(
                            client, endpoint, payload, headers, delta_callback,
                        ),
                        timeout=self.model_attempt_timeout,
                    )

                    if finish_reason not in (None, "stop"):
                        last_error = (
                            f"unexpected finish_reason={finish_reason}"
                        )
                        logger.warning(
                            "Holotron attempt {}/{}: {}; retrying",
                            attempt + 1,
                            max_retry,
                            last_error,
                        )
                        if attempt + 1 < max_retry:
                            await asyncio.sleep(self.retry_sleep)
                        continue

                    try:
                        parsed_step = Step.model_validate_json(content)
                        break
                    except ValidationError as exc:
                        last_error = f"json validation: {str(exc)[:200]}"
                    except Exception as exc:  # noqa: BLE001
                        last_error = (
                            f"parse error: {type(exc).__name__}: {exc}"
                        )

                    logger.warning(
                        "Holotron attempt {}/{}: {}; retrying",
                        attempt + 1,
                        max_retry,
                        last_error,
                    )
                    if attempt + 1 < max_retry:
                        await asyncio.sleep(self.retry_sleep)

                except asyncio.TimeoutError:
                    last_error = (
                        f"model timed out after {self.model_attempt_timeout:g}s"
                    )
                    logger.warning(
                        "Holotron attempt {}/{}: {}; retrying",
                        attempt + 1,
                        max_retry,
                        last_error,
                    )
                    if attempt + 1 < max_retry:
                        await asyncio.sleep(self.retry_sleep)

                except Exception as exc:  # noqa: BLE001
                    last_error = f"request failed: {exc}"
                    logger.error(
                        "Holotron attempt {}/{}: {}; retrying",
                        attempt + 1,
                        max_retry,
                        last_error,
                    )
                    if attempt + 1 < max_retry:
                        await asyncio.sleep(self.retry_sleep)

        finally:
            if owns_client:
                await client.aclose()

        if parsed_step is None:
            return ParsedStep(
                status="error",
                error=f"all {max_retry} attempts failed: {last_error}",
            )

        # Push parsed JSON back into history (NOT raw model output) — per
        # the H Company harness, only the canonical Step JSON should be
        # round-tripped so the model never sees its own pre-validation prose.
        self._messages.append({
            "role": "assistant",
            "content": parsed_step.model_dump_json(),
        })

        screen_w, screen_h = screen_size
        return _to_parsed_step(parsed_step, screen_w, screen_h)

    # ── Streaming ──────────────────────────────────────────────────────────

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: dict,
        headers: dict,
        delta_callback: Optional[DeltaCallback],
    ) -> tuple[str, str, Optional[str]]:
        reasoning_buf: list[str] = []
        content_buf: list[str] = []
        finish_reason: Optional[str] = None

        async with client.stream(
            "POST", endpoint, json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                err_body = (await resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Holotron stream HTTP {resp.status_code}: {err_body[:400]}"
                )

            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(
                        "Holotron stream returned non-JSON data chunk: {!r}",
                        data[:200],
                    )
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                r_delta = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                c_delta = delta.get("content") or ""

                if r_delta:
                    reasoning_buf.append(r_delta)
                if c_delta:
                    content_buf.append(c_delta)
                if (r_delta or c_delta) and delta_callback:
                    try:
                        await delta_callback(r_delta, c_delta)
                    except Exception:
                        logger.warning(
                            "delta_callback raised; continuing Holotron stream"
                        )

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

        return "".join(content_buf), "".join(reasoning_buf), finish_reason
