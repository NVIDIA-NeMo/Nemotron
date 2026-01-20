"""
NeMo Gym Resource Server for LangGraph CLI Tool Verification.

This resource server implements verification logic for training AI agents
to translate natural language instructions into correct LangGraph CLI commands.
It evaluates model outputs against expected structured JSON tool calls.

Architecture follows NeMo Gym patterns:
- FastAPI-based async server
- Pydantic models for request/response validation
- verify() function returns reward signals for RLVR training
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import json
import re
import unicodedata
from pathlib import Path

from fastapi import FastAPI


# =============================================================================
# Value Normalization for Robust Comparison
# =============================================================================

def normalize_unicode(s: str) -> str:
    """
    Normalize unicode characters for consistent comparison.

    Handles:
    - Non-breaking hyphens (U+2011) -> regular hyphen (U+002D)
    - Non-breaking spaces (U+00A0, U+202F) -> regular space
    - Other unicode normalization via NFKC
    """
    if not isinstance(s, str):
        return s
    # Replace common problematic unicode characters
    s = s.replace('\u2011', '-')  # non-breaking hyphen
    s = s.replace('\u2010', '-')  # hyphen
    s = s.replace('\u00A0', ' ')  # non-breaking space
    s = s.replace('\u202F', ' ')  # narrow no-break space
    # Apply NFKC normalization for compatibility
    return unicodedata.normalize('NFKC', s)


def normalize_path(p: str) -> str:
    """
    Normalize file paths for consistent comparison.

    Handles:
    - Removes leading ./
    - Converts to consistent format
    - Strips trailing slashes
    - Handles "current directory" equivalents
    """
    if not isinstance(p, str):
        return p

    p = normalize_unicode(p)

    # Handle "current directory" variations
    if p in ('.', './', ''):
        return '.'

    # Remove leading ./ if present
    if p.startswith('./'):
        p = p[2:]

    # Remove trailing slashes
    p = p.rstrip('/')

    # Extract just the filename/relative path (remove absolute path prefixes that look like data leakage)
    if p.startswith('/') and any(x in p.lower() for x in ['home/', 'users/', 'user/', 'documents/']):
        # This looks like an absolute path that shouldn't be here - extract basename
        p = Path(p).name

    return p


def normalize_value(value: Any, key: str = None) -> Any:
    """
    Normalize a value for comparison, handling type coercion and formatting.

    Args:
        value: The value to normalize
        key: Optional key name for type-specific handling

    Returns:
        Normalized value suitable for comparison
    """
    if value is None:
        return None

    # Handle port: convert float to int
    if key == 'port':
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.replace('.', '').isdigit():
            return int(float(value))
        return value

    # Handle paths
    if key in ('path', 'output_path'):
        if isinstance(value, str):
            return normalize_path(value)
        return value

    # Handle template names (unicode normalization)
    if key == 'template':
        if isinstance(value, str):
            return normalize_unicode(value)
        return value

    # Handle strings generally
    if isinstance(value, str):
        return normalize_unicode(value)

    return value


def normalize_cli_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an entire CLI output dictionary.

    Returns a new dict with all values normalized for comparison.
    """
    normalized = {}
    for key, value in data.items():
        normalized[key] = normalize_value(value, key)
    return normalized


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class CLIToolCall(BaseModel):
    """Expected structure for LangGraph CLI tool calls."""
    command: str = Field(..., description="CLI command: new, dev, up, build, or dockerfile")
    template: Optional[str] = Field(None, description="Template name for 'new' command")
    path: Optional[str] = Field(None, description="Project path for 'new' command")
    port: Optional[int] = Field(None, description="Port for 'dev' or 'up' command")
    no_browser: Optional[bool] = Field(None, description="Skip browser for 'dev' command")
    watch: Optional[bool] = Field(None, description="Watch mode for 'up' command")
    tag: Optional[str] = Field(None, description="Image tag for 'build' command")
    output_path: Optional[str] = Field(None, description="Output path for 'dockerfile' command")


class TaskInput(BaseModel):
    """Input structure for a single training task."""
    input: str = Field(..., description="Natural language user request")
    output: CLIToolCall = Field(..., description="Expected structured CLI tool call")


class VerifyRequest(BaseModel):
    """Request body for the verify endpoint."""
    task_id: str = Field(..., description="Unique identifier for this task")
    task_input: TaskInput = Field(..., description="The original task input")
    model_response: str = Field(..., description="The model's generated response")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Full conversation history if multi-turn"
    )


class VerifyResponse(BaseModel):
    """Response from the verify endpoint with reward signal."""
    task_id: str
    reward: float = Field(..., description="Reward signal in range [-1.0, 1.0]")
    exact_match: bool = Field(default=False)
    command_correct: bool = Field(default=False)
    flag_accuracy: float = Field(default=0.0)
    feedback: str = Field(default="")
    parsed_output: Optional[Dict[str, Any]] = Field(None)


class ParseCLIRequest(BaseModel):
    """Request for the parse_cli_command tool."""
    user_query: str = Field(..., description="Natural language CLI request")


class ParseCLIResponse(BaseModel):
    """Response from parse_cli_command tool."""
    parsed_command: Optional[Dict[str, Any]] = Field(None)
    success: bool = Field(default=False)
    error: Optional[str] = Field(None)


# =============================================================================
# Scoring Logic
# =============================================================================

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model response, handling various formats.

    Supports:
    - Raw JSON objects
    - JSON in code blocks (```json ... ```)
    - JSON within <answer> tags
    - Content after </think> tags (for reasoning models)
    """
    # Strip thinking content if present (for reasoning models)
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    # Try direct JSON parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, response)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try extracting from <answer> tags (for reasoning models)
    answer_pattern = r'<answer>\s*([\s\S]*?)\s*</answer>'
    matches = re.findall(answer_pattern, response)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding any JSON object in the response
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def score_cli_output(predicted: Dict[str, Any], reference: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """
    Compare predicted CLI JSON vs reference, returning reward and metrics.

    Uses normalization to handle:
    - Type coercion (float ports -> int)
    - Unicode normalization (non-breaking hyphens, etc.)
    - Path normalization (./ prefixes, trailing slashes)

    Scoring logic:
    - Wrong command: -1.0 (no partial credit)
    - Correct command: Reward based on flag accuracy
        - +1 for each correct flag (key & value match after normalization)
        - -1 for incorrect values or hallucinated extra flags
        - Normalized to [-1.0, 1.0] range
    - Exact match: +1.0

    Returns:
        tuple of (reward, metrics_dict)
    """
    metrics = {
        "exact_match": False,
        "command_correct": False,
        "flag_accuracy": 0.0,
        "correct_flags": 0,
        "wrong_flags": 0,
        "extra_flags": 0,
        "total_flags": 0
    }

    # Normalize both inputs for fair comparison
    predicted = normalize_cli_output(predicted)
    reference = normalize_cli_output(reference)

    # Extract command
    ref_cmd = reference.get("command")
    pred_cmd = predicted.get("command")

    # If command is wrong, full penalty
    if pred_cmd != ref_cmd:
        return -1.0, metrics

    metrics["command_correct"] = True

    # Build reference flags dict (non-null values only)
    ref_flags = {}
    flag_keys = ["template", "path", "port", "no_browser", "watch", "tag", "output_path"]
    for key in flag_keys:
        val = reference.get(key)
        if val is not None:
            ref_flags[key] = val

    # Build predicted flags dict (non-null values only)
    pred_flags = {}
    for key in flag_keys:
        val = predicted.get(key)
        if val is not None:
            pred_flags[key] = val

    total_flags = len(ref_flags)
    metrics["total_flags"] = total_flags

    if total_flags == 0:
        # No flags expected, check if model added any
        if len(pred_flags) == 0:
            metrics["exact_match"] = True
            return 1.0, metrics
        else:
            # Penalize hallucinated flags
            metrics["extra_flags"] = len(pred_flags)
            return max(-1.0, -0.5 * len(pred_flags)), metrics

    # Count correct, wrong, and extra flags
    correct_count = 0
    wrong_value_count = 0

    for key, ref_val in ref_flags.items():
        if key in pred_flags:
            if pred_flags[key] == ref_val:
                correct_count += 1
            else:
                wrong_value_count += 1

    extra_count = sum(1 for k in pred_flags.keys() if k not in ref_flags)

    metrics["correct_flags"] = correct_count
    metrics["wrong_flags"] = wrong_value_count
    metrics["extra_flags"] = extra_count
    metrics["flag_accuracy"] = correct_count / total_flags if total_flags > 0 else 0.0

    # Compute reward
    reward = (correct_count - wrong_value_count - extra_count) / total_flags
    reward = max(-1.0, min(1.0, reward))

    # Check for exact match
    if correct_count == total_flags and wrong_value_count == 0 and extra_count == 0:
        metrics["exact_match"] = True
        reward = 1.0

    return reward, metrics


# =============================================================================
# Resource Server
# =============================================================================

class LangGraphCLIResourceServer:
    """
    NeMo Gym Resource Server for LangGraph CLI verification.

    Implements:
    - verify(): Core verification function returning reward signals
    - parse_cli_command(): Tool for models to parse CLI commands (optional)
    """

    def __init__(self):
        self.app = self._setup_webserver()

    def _setup_webserver(self) -> FastAPI:
        """Set up FastAPI application with endpoints."""
        app = FastAPI(
            title="LangGraph CLI Resource Server",
            description="NeMo Gym resource server for CLI tool-calling verification",
            version="1.0.0"
        )

        # Register endpoints
        app.post("/verify", response_model=VerifyResponse)(self.verify)
        app.post("/parse_cli_command", response_model=ParseCLIResponse)(self.parse_cli_command)
        app.get("/health")(self.health_check)

        return app

    async def health_check(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        """
        Verify model output and compute reward signal.

        This is the core function for RLVR training. It evaluates whether
        the model correctly translated the natural language request into
        the expected structured CLI tool call.

        Args:
            body: VerifyRequest containing task input and model response

        Returns:
            VerifyResponse with reward signal and metrics
        """
        # Parse model response
        # Debug: show what the model generated
        response_preview = body.model_response if body.model_response else "(empty)"
        print(f"\n[VERIFY] Model output: {response_preview!r}")
        parsed_output = extract_json_from_response(body.model_response)

        if parsed_output is None:
            # Failed to parse JSON - heavy penalty
            return VerifyResponse(
                task_id=body.task_id,
                reward=-1.0,
                exact_match=False,
                command_correct=False,
                flag_accuracy=0.0,
                feedback="Failed to parse JSON from model response.",
                parsed_output=None
            )

        # Convert reference to dict
        reference = body.task_input.output.model_dump()

        # Compute reward and metrics
        reward, metrics = score_cli_output(parsed_output, reference)

        # Generate feedback message
        if metrics["exact_match"]:
            feedback = "Exact match! Command and all flags are correct."
        elif metrics["command_correct"]:
            if reward > 0:
                feedback = f"Partial match: correct command, {metrics['correct_flags']}/{metrics['total_flags']} flags correct."
            else:
                feedback = f"Command correct, but flags incorrect. Wrong: {metrics['wrong_flags']}, Extra: {metrics['extra_flags']}"
        else:
            feedback = "Incorrect command."

        return VerifyResponse(
            task_id=body.task_id,
            reward=reward,
            exact_match=metrics["exact_match"],
            command_correct=metrics["command_correct"],
            flag_accuracy=metrics["flag_accuracy"],
            feedback=feedback,
            parsed_output=parsed_output
        )

    async def parse_cli_command(self, body: ParseCLIRequest) -> ParseCLIResponse:
        """
        Tool endpoint for parsing CLI commands.

        This tool can be exposed to models during training/inference
        to help them understand CLI syntax. In most cases, the model
        should learn to generate commands directly without this tool.
        """
        # This is a placeholder - in production, you might implement
        # actual CLI parsing logic here
        return ParseCLIResponse(
            parsed_command=None,
            success=False,
            error="Direct parsing not implemented. Model should generate structured output."
        )


# =============================================================================
# Standalone Reward Functions for Unsloth GRPO
# =============================================================================

def cli_correctness_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    expected_outputs: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """
    Reward function for Unsloth GRPOTrainer.

    Evaluates whether completions match expected CLI tool calls.
    Returns reward of 2.0 for exact match, partial rewards for correct command,
    and -1.0 for wrong command or invalid JSON.

    Args:
        prompts: List of conversation prompts
        completions: List of model completions
        expected_outputs: List of expected output dictionaries

    Returns:
        List of reward values
    """
    rewards = []

    for completion, expected in zip(completions, expected_outputs):
        # Get the assistant's response
        response = completion[0]["content"] if completion else ""

        # Parse the response
        parsed = extract_json_from_response(response)

        if parsed is None:
            rewards.append(-1.0)
            continue

        # Score against expected
        reward, metrics = score_cli_output(parsed, expected)

        # Scale reward: exact match gets 2.0, partial gets proportional reward
        if metrics["exact_match"]:
            rewards.append(2.0)
        else:
            rewards.append(reward)

    return rewards


def json_format_reward(
    completions: List[List[Dict[str, str]]],
    **kwargs
) -> List[float]:
    """
    Reward function that checks if output is valid JSON.

    Returns:
        0.5 for valid JSON, 0.0 for invalid
    """
    rewards = []

    for completion in completions:
        response = completion[0]["content"] if completion else ""
        parsed = extract_json_from_response(response)
        rewards.append(0.5 if parsed is not None else 0.0)

    return rewards


def command_reward(
    completions: List[List[Dict[str, str]]],
    expected_outputs: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """
    Reward function that checks if command field is correct.

    Returns:
        0.5 for correct command, 0.0 otherwise
    """
    rewards = []

    for completion, expected in zip(completions, expected_outputs):
        response = completion[0]["content"] if completion else ""
        parsed = extract_json_from_response(response)

        if parsed is None:
            rewards.append(0.0)
            continue

        expected_cmd = expected.get("command")
        parsed_cmd = parsed.get("command")

        rewards.append(0.5 if parsed_cmd == expected_cmd else 0.0)

    return rewards


def flag_accuracy_reward(
    completions: List[List[Dict[str, str]]],
    expected_outputs: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """
    Reward function based on flag accuracy.

    Returns:
        Proportional reward based on correct flags (0.0 to 1.0)
    """
    rewards = []

    for completion, expected in zip(completions, expected_outputs):
        response = completion[0]["content"] if completion else ""
        parsed = extract_json_from_response(response)

        if parsed is None:
            rewards.append(0.0)
            continue

        _, metrics = score_cli_output(parsed, expected)
        rewards.append(metrics["flag_accuracy"])

    return rewards


# =============================================================================
# Server Entry Point
# =============================================================================

def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    server = LangGraphCLIResourceServer()
    return server.app


# For running with uvicorn directly
app = create_app()


# =============================================================================
# Data Cleaning Utilities
# =============================================================================

def clean_training_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Clean a single training example, normalizing values and fixing common issues.

    Args:
        example: A training example with 'input' and 'output' keys

    Returns:
        Cleaned example, or None if the example should be filtered out
    """
    # Filter out empty inputs
    if not example.get('input', '').strip():
        return None

    cleaned = {
        'input': normalize_unicode(example['input']),
        'output': {}
    }

    output = example.get('output', {})

    # Clean each field in output
    for key in ['command', 'template', 'path', 'port', 'no_browser', 'watch', 'tag', 'output_path']:
        val = output.get(key)
        cleaned['output'][key] = normalize_value(val, key)

    # Validate: must have a command
    if not cleaned['output'].get('command'):
        return None

    return cleaned


def clean_training_data(
    input_path: str,
    output_path: str = None,
    remove_duplicates: bool = True
) -> Dict[str, Any]:
    """
    Clean an entire JSONL training file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to write cleaned data (defaults to input_path with .cleaned suffix)
        remove_duplicates: Whether to remove near-duplicate examples

    Returns:
        Statistics about the cleaning process
    """
    if output_path is None:
        output_path = input_path.replace('.jsonl', '.cleaned.jsonl')

    stats = {
        'total_read': 0,
        'empty_inputs_removed': 0,
        'duplicates_removed': 0,
        'total_written': 0,
        'errors': []
    }

    cleaned_examples = []
    seen_inputs = set()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats['total_read'] += 1

            try:
                example = json.loads(line)
                cleaned = clean_training_example(example)

                if cleaned is None:
                    stats['empty_inputs_removed'] += 1
                    continue

                # Check for duplicates based on normalized input
                input_key = cleaned['input'].lower().strip()
                if remove_duplicates and input_key in seen_inputs:
                    stats['duplicates_removed'] += 1
                    continue

                seen_inputs.add(input_key)
                cleaned_examples.append(cleaned)

            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                stats['errors'].append(f"Line {line_num}: {e}")

    # Write cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in cleaned_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    stats['total_written'] = len(cleaned_examples)

    return stats


if __name__ == "__main__":
    import sys
    import uvicorn

    # Allow running as data cleaner or server
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        # Usage: python app.py clean input.jsonl [output.jsonl]
        if len(sys.argv) < 3:
            print("Usage: python app.py clean <input.jsonl> [output.jsonl]")
            sys.exit(1)

        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None

        print(f"Cleaning {input_file}...")
        stats = clean_training_data(input_file, output_file)

        print(f"\nCleaning complete:")
        print(f"  Total read: {stats['total_read']}")
        print(f"  Empty inputs removed: {stats['empty_inputs_removed']}")
        print(f"  Duplicates removed: {stats['duplicates_removed']}")
        print(f"  Total written: {stats['total_written']}")

        if stats['errors']:
            print(f"\nErrors ({len(stats['errors'])}):")
            for err in stats['errors'][:10]:
                print(f"  {err}")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
