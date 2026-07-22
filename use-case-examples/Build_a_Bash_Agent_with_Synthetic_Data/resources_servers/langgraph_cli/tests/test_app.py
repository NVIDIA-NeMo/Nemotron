import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bash_agent.commands import TEMPLATE_IDS as RUNTIME_TEMPLATE_IDS
from bash_agent.commands import invocation_from_payload
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.langgraph_cli.app import (
    LangGraphCLIRunRequest,
    LangGraphCLIResourcesServer,
    LangGraphCLIResourcesServerConfig,
    LangGraphCLIVerifyRequest,
    TEMPLATE_IDS,
    extract_json_from_response,
    score_cli_output,
)


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp-test",
        created_at=0.0,
        model="test-model",
        object="response",
        output=[
            {
                "id": "msg-test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def make_server() -> LangGraphCLIResourcesServer:
    config = LangGraphCLIResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="langgraph_cli",
    )
    return LangGraphCLIResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def make_request(text: str, expected_output: dict) -> LangGraphCLIVerifyRequest:
    return LangGraphCLIVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "Do the requested CLI operation."}],
            "parallel_tool_calls": False,
            "tools": [],
        },
        response=make_response(text),
        expected_output=expected_output,
    )


def test_verifier_and_runtime_share_current_templates():
    assert TEMPLATE_IDS == RUNTIME_TEMPLATE_IDS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"command":"build","tag":"demo:v1"}', {"command": "build", "tag": "demo:v1"}),
        ('```json\n{"command":"dev","port":8123}\n```', {"command": "dev", "port": 8123}),
        (
            '<think>reasoning</think><answer>{"command":"up","wait":true}</answer>',
            {"command": "up", "wait": True},
        ),
    ],
)
def test_extract_json_from_response(text, expected):
    assert extract_json_from_response(text) == expected


@pytest.mark.parametrize("text", ["not json", "[]", "null", '"a string"'])
def test_extract_rejects_non_objects(text):
    assert extract_json_from_response(text) is None


def test_score_preserves_json_types_and_path_spelling():
    reward, metrics = score_cli_output(
        {"command": "dev", "port": "8123"},
        {"command": "dev", "port": 8123},
    )
    assert reward == -1.0
    assert metrics["exact_match"] is False

    reward, metrics = score_cli_output(
        {"command": "dockerfile", "output_path": "./deploy/Dockerfile"},
        {"command": "dockerfile", "output_path": "deploy/Dockerfile"},
    )
    assert reward == -1.0
    assert metrics["exact_match"] is False


@pytest.mark.asyncio
async def test_verify_exact_match():
    result = await make_server().verify(
        make_request('{"command":"build","tag":"demo:v1"}', {"command": "build", "tag": "demo:v1"})
    )
    assert result.reward == 1.0
    assert result.exact_match is True
    assert result.parsed_output["tag"] == "demo:v1"


@pytest.mark.asyncio
async def test_verify_rejects_unknown_fields():
    result = await make_server().verify(
        make_request(
            '{"command":"build","tag":"demo:v1","shell":"rm -rf /"}',
            {"command": "build", "tag": "demo:v1"},
        )
    )
    assert result.reward == -1.0
    assert "Invalid CLI tool-call schema" in result.feedback


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        ({"command": "dev", "port": "8123"}, {"command": "dev", "port": 8123}),
        ({"command": "dev", "port": 8123.0}, {"command": "dev", "port": 8123}),
        ({"command": "dev", "port": True}, {"command": "dev", "port": 8123}),
        (
            {"command": "up", "port": 8123, "watch": "true"},
            {"command": "up", "port": 8123, "watch": True},
        ),
        (
            {"command": "up", "port": 8123, "wait": 1},
            {"command": "up", "port": 8123, "wait": True},
        ),
        (
            {"command": "new", "template": "agent-python", "path": "/home/user/demo"},
            {"command": "new", "template": "agent-python", "path": "demo"},
        ),
        (
            {"command": "new", "template": "agent-python", "path": "../demo"},
            {"command": "new", "template": "agent-python", "path": "demo"},
        ),
        (
            {"command": "dockerfile", "output_path": "deploy/../../Dockerfile"},
            {"command": "dockerfile", "output_path": "Dockerfile"},
        ),
        (
            {"command": "dockerfile", "output_path": "."},
            {"command": "dockerfile", "output_path": "Dockerfile"},
        ),
        (
            {"command": "dockerfile", "output_path": "--help"},
            {"command": "dockerfile", "output_path": "Dockerfile"},
        ),
        (
            {"command": "new", "template": "react-agent", "path": "demo"},
            {"command": "new", "template": "agent-python", "path": "demo"},
        ),
        (
            {"command": "build", "tag": "bad tag"},
            {"command": "build", "tag": "demo:v1"},
        ),
        (
            {"command": "build", "tag": "demo:v1", "port": 8123},
            {"command": "build", "tag": "demo:v1"},
        ),
        ({"command": "build"}, {"command": "build", "tag": "demo:v1"}),
    ],
)
@pytest.mark.asyncio
async def test_verify_rejects_outputs_the_runtime_rejects(predicted, expected):
    result = await make_server().verify(make_request(json.dumps(predicted), expected))
    assert result.reward == -1.0
    assert "Invalid CLI tool-call schema" in result.feedback


def test_example_data_uses_current_gym_schema_and_runtime_contract(tmp_path):
    path = Path(__file__).parents[1] / "data" / "example.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    for record in records:
        validated = LangGraphCLIRunRequest.model_validate(record)
        assert validated.responses_create_params.input
        assert validated.expected_output.command
        payload = validated.expected_output.model_dump(mode="json")
        invocation = invocation_from_payload(payload, tmp_path)
        assert invocation.argv[:2] == ("langgraph", validated.expected_output.command)

        if validated.expected_output.command == "new":
            user_message = validated.responses_create_params.input[-1]
            assert validated.expected_output.template in user_message.content
