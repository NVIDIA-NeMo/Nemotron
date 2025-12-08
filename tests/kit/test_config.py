"""Tests for nemotron.kit.config (ConfigManager and cli)."""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nemotron.kit import cli, ConfigManager


@dataclass
class SimpleConfig:
    """Simple config for testing."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    name: str = "test"


@dataclass
class NestedConfig:
    """Config with nested dataclass."""

    @dataclass
    class ModelConfig:
        hidden_size: int = 256
        num_layers: int = 4

    @dataclass
    class TrainingConfig:
        steps: int = 1000
        lr: float = 1e-3

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_parse_args_defaults(self):
        """Test parsing with default values."""
        manager = ConfigManager(SimpleConfig)
        config = manager.parse_args([])

        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.name == "test"

    def test_parse_args_cli_override(self):
        """Test CLI args override defaults."""
        manager = ConfigManager(SimpleConfig)
        config = manager.parse_args(["--batch-size", "64", "--name", "custom"])

        assert config.batch_size == 64
        assert config.name == "custom"
        assert config.learning_rate == 1e-4  # unchanged

    def test_parse_args_yaml_file(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 128\nlearning_rate: 0.001\n")
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            config = manager.parse_args(["--config-file", config_path])

            assert config.batch_size == 128
            assert config.learning_rate == 0.001
            assert config.name == "test"  # default
        finally:
            Path(config_path).unlink()

    def test_parse_args_toml_file(self):
        """Test loading config from TOML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write('batch_size = 256\nname = "toml_test"\n')
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            config = manager.parse_args(["--config-file", config_path])

            assert config.batch_size == 256
            assert config.name == "toml_test"
        finally:
            Path(config_path).unlink()

    def test_parse_args_json_file(self):
        """Test loading config from JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"batch_size": 512, "learning_rate": 0.01}, f)
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            config = manager.parse_args(["--config-file", config_path])

            assert config.batch_size == 512
            assert config.learning_rate == 0.01
        finally:
            Path(config_path).unlink()

    def test_cli_override_config_file(self):
        """Test that CLI args override config file values."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 128\nlearning_rate: 0.001\n")
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            config = manager.parse_args([
                "--config-file", config_path,
                "--batch-size", "64",  # Override file value
            ])

            assert config.batch_size == 64  # CLI wins
            assert config.learning_rate == 0.001  # from file
        finally:
            Path(config_path).unlink()

    def test_nested_config_yaml(self):
        """Test nested dataclass config from YAML."""
        yaml_content = """
model:
  hidden_size: 512
  num_layers: 8
training:
  steps: 5000
seed: 123
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            config_path = f.name

        try:
            manager = ConfigManager(NestedConfig)
            config = manager.parse_args(["--config-file", config_path])

            assert config.model.hidden_size == 512
            assert config.model.num_layers == 8
            assert config.training.steps == 5000
            assert config.training.lr == 1e-3  # default
            assert config.seed == 123
        finally:
            Path(config_path).unlink()

    def test_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        manager = ConfigManager(SimpleConfig)

        with pytest.raises(FileNotFoundError):
            manager.parse_args(["--config-file", "/nonexistent/config.yaml"])

    def test_unsupported_format(self):
        """Test error for unsupported config format."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        ) as f:
            f.write("<config></config>")
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            with pytest.raises(ValueError, match="Unsupported config format"):
                manager.parse_args(["--config-file", config_path])
        finally:
            Path(config_path).unlink()

    def test_invalid_field_in_config(self):
        """Test error when config file has invalid fields."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 64\ninvalid_field: 123\n")
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            with pytest.raises(ValueError, match="Invalid fields"):
                manager.parse_args(["--config-file", config_path])
        finally:
            Path(config_path).unlink()

    def test_non_dataclass_raises(self):
        """Test that non-dataclass raises TypeError."""

        class NotADataclass:
            pass

        with pytest.raises(TypeError, match="must be a dataclass"):
            ConfigManager(NotADataclass)


class TestCli:
    """Tests for cli() function."""

    def test_cli_with_dataclass(self):
        """Test cli() with a dataclass type."""
        config = cli(SimpleConfig, args=["--batch-size", "128"])

        assert config.batch_size == 128
        assert config.learning_rate == 1e-4

    def test_cli_with_function(self):
        """Test cli() with a function."""

        def my_func(batch_size: int = 32, name: str = "default") -> dict:
            return {"batch_size": batch_size, "name": name}

        result = cli(my_func, args=["--batch-size", "64"])

        assert result == {"batch_size": 64, "name": "default"}

    def test_cli_with_config_taking_function(self):
        """Test cli() with function that takes a dataclass."""

        def train(config: SimpleConfig) -> int:
            return config.batch_size * 2

        result = cli(train, args=["--batch-size", "100"])

        assert result == 200

    def test_cli_with_config_file(self):
        """Test cli() with config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 256\n")
            config_path = f.name

        try:
            config = cli(SimpleConfig, args=["--config-file", config_path])
            assert config.batch_size == 256
        finally:
            Path(config_path).unlink()


class TestConfigFileFormats:
    """Tests for different config file format edge cases."""

    def test_empty_yaml_file(self):
        """Test empty YAML file uses defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")  # Empty file
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            config = manager.parse_args(["--config-file", config_path])

            # Should use all defaults
            assert config.batch_size == 32
            assert config.learning_rate == 1e-4
        finally:
            Path(config_path).unlink()

    def test_config_file_equals_syntax(self):
        """Test --config-file=path syntax."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 64\n")
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            config = manager.parse_args([f"--config-file={config_path}"])

            assert config.batch_size == 64
        finally:
            Path(config_path).unlink()

    def test_config_alias(self):
        """Test --config alias for --config-file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 96\n")
            config_path = f.name

        try:
            manager = ConfigManager(SimpleConfig)
            config = manager.parse_args(["--config", config_path])

            assert config.batch_size == 96
        finally:
            Path(config_path).unlink()
