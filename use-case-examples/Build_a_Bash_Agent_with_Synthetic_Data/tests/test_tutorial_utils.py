import json

import pytest

from resources_servers.langgraph_cli.app import LangGraphCLIVerifyRequest
from bash_agent.config import Config
from tutorial_utils import (
    SYSTEM_PROMPT,
    build_expected_output,
    build_gym_record,
    build_verify_payload,
    create_nemo_gym_reward_function,
    deterministic_request,
    generated_request_matches_expected_output,
    normalize_expected_output,
    repair_generated_request,
)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {"command": "new", "template": "agent-python", "project_path": "agents/demo"},
            {"template": "agent-python", "path": "agents/demo"},
        ),
        (
            {"command": "dev", "port": 8123, "no_browser": True},
            {"port": 8123, "no_browser": True},
        ),
        (
            {"command": "up", "port": 9000, "watch": False, "wait": True},
            {"port": 9000, "watch": None, "wait": True},
        ),
        ({"command": "build", "image_tag": "demo:v1"}, {"tag": "demo:v1"}),
        (
            {"command": "dockerfile", "dockerfile_path": "deploy/Dockerfile"},
            {"output_path": "deploy/Dockerfile"},
        ),
    ],
)
def test_build_expected_output_is_deterministic(row, expected):
    target = build_expected_output(row)
    for key, value in expected.items():
        assert target[key] == value


def test_build_gym_record_uses_current_schema():
    record = build_gym_record("Build the image", {"command": "build", "tag": "demo:v1"})
    assert record["responses_create_params"]["input"][0]["role"] == "system"
    assert record["responses_create_params"]["input"][0]["content"] == SYSTEM_PROMPT
    assert Config().json_system_prompt == SYSTEM_PROMPT
    validated = LangGraphCLIVerifyRequest.model_validate(
        {
            **record,
            "response": {
                "id": "resp-1",
                "created_at": 0.0,
                "model": "test",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
        }
    )
    assert validated.expected_output.tag == "demo:v1"


def test_normalize_expected_output_restores_integral_ports_after_parquet_roundtrip():
    expected = normalize_expected_output({"command": "dev", "port": 8123.0})

    assert expected["port"] == 8123
    assert type(expected["port"]) is int

    with pytest.raises(ValueError, match="integral port"):
        normalize_expected_output({"command": "dev", "port": 8123.5})


@pytest.mark.parametrize(
    ("expected", "user_input"),
    [
        (
            {"command": "new", "template": "agent-python", "path": "agents/demo"},
            "Create a new project at agents/demo with the agent-python template.",
        ),
        (
            {"command": "dev", "port": 8123, "no_browser": True},
            "Start the development server on port 8123 without opening a browser.",
        ),
        (
            {"command": "up", "port": 8000, "watch": True, "wait": True},
            "Bring the services up on port 8000, watch files, and wait until ready.",
        ),
        (
            {"command": "build", "tag": "demo-agent:latest"},
            "Build the image with tag demo-agent:latest.",
        ),
        (
            {"command": "dockerfile", "output_path": "deploy/Dockerfile"},
            "Generate the Dockerfile at deploy/Dockerfile.",
        ),
    ],
)
def test_generated_request_validator_accepts_complete_requests(expected, user_input):
    assert generated_request_matches_expected_output(user_input, expected)


@pytest.mark.parametrize(
    ("expected", "bad_request"),
    [
        ({"command": "dev", "port": 8123}, "Start the development server."),
        (
            {"command": "dev", "port": 8123, "no_browser": True},
            "Start the development server on port 8123 and open the browser.",
        ),
        (
            {"command": "up", "port": 8000, "wait": True},
            "Start the services on port 8000.",
        ),
        (
            {"command": "up", "port": 8000, "wait": True},
            "Start the services on port 8000 and wait for them to finish.",
        ),
        (
            {"command": "build", "tag": "demo-agent:latest"},
            "Build the demo agent using the latest tag.",
        ),
        (
            {"command": "dockerfile", "output_path": "deploy/Dockerfile"},
            "Build a Docker image on port 8000.",
        ),
    ],
)
def test_repair_generated_request_replaces_incomplete_or_extra_phrasing(expected, bad_request):
    repaired = repair_generated_request(bad_request, expected)

    assert repaired == deterministic_request(expected)
    assert generated_request_matches_expected_output(repaired, expected)


def test_build_verify_payload_validates_with_gym_models():
    payload = build_verify_payload(
        [{"role": "assistant", "content": '{"command":"build","tag":"demo:v1"}'}],
        [{"role": "user", "content": "Build it"}],
        {"command": "build", "tag": "demo:v1"},
        "1",
    )
    assert payload["response"]["output"][0]["content"][0]["text"].endswith('"tag":"demo:v1"}')
    assert LangGraphCLIVerifyRequest.model_validate(payload)


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"reward": 0.75}


def test_reward_function_returns_list_and_forwards_expected_output():
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    reward_fn = create_nemo_gym_reward_function("http://127.0.0.1:8000/verify", post=post)
    rewards = reward_fn(
        completions=[[{"role": "assistant", "content": '{"command":"build","tag":"demo:v1"}'}]],
        prompts=[[{"role": "user", "content": "Build it"}]],
        expected_output=[json.dumps({"command": "build", "tag": "demo:v1"})],
    )
    assert rewards == [0.75]
    assert calls[0][1]["json"]["expected_output"]["tag"] == "demo:v1"
