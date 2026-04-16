"""
Unit tests for LangGraph CLI Resource Server.
"""

import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from app import (
    app,
    extract_json_from_response,
    score_cli_output,
    cli_correctness_reward,
    json_format_reward,
    command_reward,
    flag_accuracy_reward,
    normalize_value,
    normalize_unicode,
    normalize_path,
    clean_training_example,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestJSONExtraction:
    """Tests for JSON extraction from model responses."""

    def test_raw_json(self):
        """Test extraction from raw JSON."""
        response = '{"command": "build", "tag": "v1"}'
        result = extract_json_from_response(response)
        assert result == {"command": "build", "tag": "v1"}

    def test_code_block_json(self):
        """Test extraction from markdown code block."""
        response = '''Here is the command:
```json
{"command": "new", "template": "react-agent"}
```'''
        result = extract_json_from_response(response)
        assert result == {"command": "new", "template": "react-agent"}

    def test_answer_tags(self):
        """Test extraction from <answer> tags (reasoning models)."""
        response = '''<think>Let me analyze this request...</think>
<answer>{"command": "dev", "port": 3000}</answer>'''
        result = extract_json_from_response(response)
        assert result == {"command": "dev", "port": 3000}

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = "This is not valid JSON"
        result = extract_json_from_response(response)
        assert result is None


class TestScoring:
    """Tests for scoring logic."""

    def test_exact_match(self):
        """Test exact match scoring."""
        predicted = {"command": "build", "tag": "v1"}
        reference = {"command": "build", "tag": "v1"}
        reward, metrics = score_cli_output(predicted, reference)
        assert reward == 1.0
        assert metrics["exact_match"] is True

    def test_wrong_command(self):
        """Test wrong command penalty."""
        predicted = {"command": "dev", "tag": "v1"}
        reference = {"command": "build", "tag": "v1"}
        reward, metrics = score_cli_output(predicted, reference)
        assert reward == -1.0
        assert metrics["command_correct"] is False

    def test_partial_flags(self):
        """Test partial flag matching."""
        predicted = {"command": "new", "template": "react-agent"}
        reference = {"command": "new", "template": "react-agent", "path": "./myapp"}
        reward, metrics = score_cli_output(predicted, reference)
        assert metrics["command_correct"] is True
        assert metrics["correct_flags"] == 1
        assert metrics["total_flags"] == 2
        assert 0 < reward < 1

    def test_extra_flags_penalty(self):
        """Test penalty for hallucinated flags."""
        predicted = {"command": "build", "tag": "v1", "port": 3000}
        reference = {"command": "build", "tag": "v1"}
        reward, metrics = score_cli_output(predicted, reference)
        assert metrics["extra_flags"] == 1
        assert reward < 1.0


class TestVerifyEndpoint:
    """Tests for the /verify endpoint."""

    def test_verify_exact_match(self, client):
        """Test verification with exact match."""
        response = client.post("/verify", json={
            "task_id": "test-1",
            "task_input": {
                "input": "Build a Docker image with tag v1",
                "output": {"command": "build", "tag": "v1"}
            },
            "model_response": '{"command": "build", "tag": "v1"}'
        })
        assert response.status_code == 200
        data = response.json()
        assert data["reward"] == 1.0
        assert data["exact_match"] is True

    def test_verify_invalid_json(self, client):
        """Test verification with invalid JSON response."""
        response = client.post("/verify", json={
            "task_id": "test-2",
            "task_input": {
                "input": "Build a Docker image with tag v1",
                "output": {"command": "build", "tag": "v1"}
            },
            "model_response": "I don't understand"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["reward"] == -1.0
        assert data["exact_match"] is False

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestUnslothRewardFunctions:
    """Tests for standalone reward functions used with Unsloth."""

    def test_cli_correctness_reward(self):
        """Test CLI correctness reward function."""
        prompts = [[{"role": "user", "content": "Build image with tag v1"}]]
        completions = [[{"role": "assistant", "content": '{"command": "build", "tag": "v1"}'}]]
        expected = [{"command": "build", "tag": "v1"}]

        rewards = cli_correctness_reward(prompts, completions, expected)
        assert rewards[0] == 2.0  # Exact match

    def test_json_format_reward(self):
        """Test JSON format reward function."""
        completions = [
            [{"role": "assistant", "content": '{"command": "build"}'}],
            [{"role": "assistant", "content": "not json"}],
        ]

        rewards = json_format_reward(completions)
        assert rewards[0] == 0.5
        assert rewards[1] == 0.0

    def test_command_reward(self):
        """Test command reward function."""
        completions = [
            [{"role": "assistant", "content": '{"command": "build", "tag": "v1"}'}],
            [{"role": "assistant", "content": '{"command": "dev", "tag": "v1"}'}],
        ]
        expected = [
            {"command": "build", "tag": "v1"},
            {"command": "build", "tag": "v1"},
        ]

        rewards = command_reward(completions, expected)
        assert rewards[0] == 0.5
        assert rewards[1] == 0.0

    def test_flag_accuracy_reward(self):
        """Test flag accuracy reward function."""
        completions = [
            [{"role": "assistant", "content": '{"command": "new", "template": "react", "path": "./app"}'}],
        ]
        expected = [{"command": "new", "template": "react", "path": "./app"}]

        rewards = flag_accuracy_reward(completions, expected)
        assert rewards[0] == 1.0


class TestNormalization:
    """Tests for value normalization (type coercion, unicode, paths)."""

    def test_port_float_to_int(self):
        """Test that float ports are converted to int."""
        assert normalize_value(8080.0, 'port') == 8080
        assert normalize_value(3000.0, 'port') == 3000
        assert isinstance(normalize_value(8080.0, 'port'), int)

    def test_unicode_hyphen_normalization(self):
        """Test that non-breaking hyphens are normalized."""
        # U+2011 (non-breaking hyphen) -> U+002D (regular hyphen)
        assert normalize_unicode('data\u2011enrichment') == 'data-enrichment'
        assert normalize_unicode('react\u2011agent') == 'react-agent'

    def test_path_normalization(self):
        """Test path normalization."""
        assert normalize_path('./Dockerfile') == 'Dockerfile'
        assert normalize_path('.') == '.'
        assert normalize_path('./') == '.'
        assert normalize_path('docker/') == 'docker'

    def test_path_data_leakage_cleanup(self):
        """Test that absolute paths with user directories are cleaned."""
        # Should extract just the basename
        assert normalize_path('/home/user/project/docker') == 'docker'
        assert normalize_path('/Users/chris/Documents/Dockerfile') == 'Dockerfile'

    def test_scoring_with_float_port(self):
        """Test that scoring handles float vs int port comparison."""
        predicted = {"command": "dev", "port": 8080}
        reference = {"command": "dev", "port": 8080.0}
        reward, metrics = score_cli_output(predicted, reference)
        assert reward == 1.0
        assert metrics["exact_match"] is True

    def test_scoring_with_unicode_template(self):
        """Test that scoring handles unicode template names."""
        predicted = {"command": "new", "template": "data-enrichment"}
        reference = {"command": "new", "template": "data\u2011enrichment"}
        reward, metrics = score_cli_output(predicted, reference)
        assert reward == 1.0
        assert metrics["exact_match"] is True

    def test_scoring_with_path_variations(self):
        """Test that scoring handles path variations."""
        predicted = {"command": "dockerfile", "output_path": "Dockerfile"}
        reference = {"command": "dockerfile", "output_path": "./Dockerfile"}
        reward, metrics = score_cli_output(predicted, reference)
        assert reward == 1.0
        assert metrics["exact_match"] is True


class TestDataCleaning:
    """Tests for training data cleaning."""

    def test_clean_empty_input(self):
        """Test that empty inputs are filtered out."""
        example = {"input": "", "output": {"command": "build", "tag": "v1"}}
        result = clean_training_example(example)
        assert result is None

    def test_clean_normalizes_port(self):
        """Test that cleaning normalizes port type."""
        example = {
            "input": "Start server on port 8080",
            "output": {"command": "dev", "port": 8080.0}
        }
        result = clean_training_example(example)
        assert result is not None
        assert result["output"]["port"] == 8080
        assert isinstance(result["output"]["port"], int)

    def test_clean_normalizes_template_unicode(self):
        """Test that cleaning normalizes template unicode."""
        example = {
            "input": "Create project with data-enrichment template",
            "output": {"command": "new", "template": "data\u2011enrichment"}
        }
        result = clean_training_example(example)
        assert result is not None
        assert result["output"]["template"] == "data-enrichment"

    def test_clean_preserves_valid_data(self):
        """Test that valid data is preserved correctly."""
        example = {
            "input": "Build image with tag v1",
            "output": {"command": "build", "tag": "v1"}
        }
        result = clean_training_example(example)
        assert result is not None
        assert result["input"] == "Build image with tag v1"
        assert result["output"]["command"] == "build"
        assert result["output"]["tag"] == "v1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
