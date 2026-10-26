"""Offline tests of the Super 3 getting-started notebook's streaming helper."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def stream_tools():
    notebook = (
        Path(__file__).resolve().parents[2]
        / "use-case-examples"
        / ("Nemotron-3-Super-Getting-Started-Guide")
        / "Nemotron-3-Super-Getting-Started-Guide.ipynb"
    )
    cells = json.loads(notebook.read_text())["cells"]
    source = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "def stream_reasoning_content(" in "".join(cell["source"])
    )
    namespace = {name: "" for name in ("STREAM_REASONING_COLOR", "STREAM_RESET_COLOR", "STREAM_TOOL_COLOR")}
    exec(compile(source, str(notebook), "exec"), namespace)
    return namespace["stream_reasoning_content"]


def chunk(tool_call, as_objects=False):
    if as_objects:
        function = tool_call.get("function")
        tool_call = SimpleNamespace(
            **{**tool_call, "function": (SimpleNamespace(**function) if isinstance(function, dict) else function)}
        )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    tool_calls=[tool_call],
                    content=None,
                    reasoning=None,
                    reasoning_content=None,
                )
            )
        ]
    )


@pytest.mark.parametrize("function", [None, {"name": None}, {"arguments": None}])
@pytest.mark.parametrize("as_objects", [False, True])
def test_nullable_deltas_preserve_the_accumulated_call(stream_tools, function, as_objects):
    frames = [
        chunk(
            {
                "index": 0,
                "id": "call_math",
                "function": {
                    "name": "get_math_answer",
                    "arguments": '{"expression":',
                },
            },
            as_objects,
        ),
        chunk({"index": 0, "function": function}, as_objects),
        chunk({"index": 0, "function": {"arguments": '"123+456"}'}}, as_objects),
    ]
    with redirect_stdout(StringIO()):
        calls = stream_tools(frames, show_tool_deltas=True)
    assert calls == {0: {"id": "call_math", "name": "get_math_answer", "arguments": '{"expression":"123+456"}'}}


@pytest.mark.parametrize("as_objects", [False, True])
def test_interleaved_calls_keep_their_ids_and_argument_fragments(stream_tools, as_objects):
    deltas = [
        {"index": 0, "id": "call_a", "function": {"name": "get_math_answer", "arguments": '{"expression":'}},
        {"index": 1, "id": "call_b", "function": {"name": "get_math_answer", "arguments": '{"expression":'}},
        {"index": 0, "function": {"arguments": '"1+2"}'}},
        {"index": 1, "function": {"arguments": '"3+4"}'}},
    ]
    if as_objects:
        deltas = [SimpleNamespace(**{**delta, "function": SimpleNamespace(**delta["function"])}) for delta in deltas]
    with redirect_stdout(StringIO()):
        calls = stream_tools([chunk(delta) for delta in deltas], show_tool_deltas=True)
    assert [(calls[i]["id"], json.loads(calls[i]["arguments"])) for i in (0, 1)] == [
        ("call_a", {"expression": "1+2"}),
        ("call_b", {"expression": "3+4"}),
    ]
