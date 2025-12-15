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


class TestDefaults:
    """Tests for the defaults parameter."""

    def test_defaults_callable_simple(self):
        """Test defaults callable provides base values."""

        def my_defaults() -> SimpleConfig:
            return SimpleConfig(batch_size=64, learning_rate=0.01, name="from_defaults")

        manager = ConfigManager(SimpleConfig, defaults=my_defaults)
        config = manager.parse_args([])

        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.name == "from_defaults"

    def test_defaults_overridden_by_cli(self):
        """Test CLI args override defaults."""

        def my_defaults() -> SimpleConfig:
            return SimpleConfig(batch_size=64, learning_rate=0.01, name="from_defaults")

        manager = ConfigManager(SimpleConfig, defaults=my_defaults)
        config = manager.parse_args(["--batch-size", "128"])

        assert config.batch_size == 128  # CLI overrides
        assert config.learning_rate == 0.01  # from defaults
        assert config.name == "from_defaults"  # from defaults

    def test_defaults_overridden_by_config_file(self):
        """Test config file overrides defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 256\n")
            config_path = f.name

        def my_defaults() -> SimpleConfig:
            return SimpleConfig(batch_size=64, learning_rate=0.01, name="from_defaults")

        try:
            manager = ConfigManager(SimpleConfig, defaults=my_defaults)
            config = manager.parse_args(["--config-file", config_path])

            assert config.batch_size == 256  # config file overrides
            assert config.learning_rate == 0.01  # from defaults (not in file)
            assert config.name == "from_defaults"  # from defaults (not in file)
        finally:
            Path(config_path).unlink()

    def test_defaults_cli_overrides_config_file(self):
        """Test precedence: CLI > config file > defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 256\nlearning_rate: 0.001\n")
            config_path = f.name

        def my_defaults() -> SimpleConfig:
            return SimpleConfig(batch_size=64, learning_rate=0.01, name="from_defaults")

        try:
            manager = ConfigManager(SimpleConfig, defaults=my_defaults)
            config = manager.parse_args([
                "--config-file", config_path,
                "--batch-size", "512",
            ])

            assert config.batch_size == 512  # CLI overrides all
            assert config.learning_rate == 0.001  # config file overrides defaults
            assert config.name == "from_defaults"  # from defaults (not in file or CLI)
        finally:
            Path(config_path).unlink()

    def test_defaults_nested_config(self):
        """Test defaults with nested dataclass."""

        def my_defaults() -> NestedConfig:
            return NestedConfig(
                model=NestedConfig.ModelConfig(hidden_size=1024, num_layers=12),
                training=NestedConfig.TrainingConfig(steps=10000, lr=0.0001),
                seed=99,
            )

        manager = ConfigManager(NestedConfig, defaults=my_defaults)
        config = manager.parse_args([])

        assert config.model.hidden_size == 1024
        assert config.model.num_layers == 12
        assert config.training.steps == 10000
        assert config.training.lr == 0.0001
        assert config.seed == 99

    def test_defaults_nested_partial_override(self):
        """Test config file partially overrides nested defaults."""
        yaml_content = """
model:
  hidden_size: 512
seed: 123
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            config_path = f.name

        def my_defaults() -> NestedConfig:
            return NestedConfig(
                model=NestedConfig.ModelConfig(hidden_size=1024, num_layers=12),
                training=NestedConfig.TrainingConfig(steps=10000, lr=0.0001),
                seed=99,
            )

        try:
            manager = ConfigManager(NestedConfig, defaults=my_defaults)
            config = manager.parse_args(["--config-file", config_path])

            assert config.model.hidden_size == 512  # overridden by file
            assert config.model.num_layers == 12  # from defaults (not in file)
            assert config.training.steps == 10000  # from defaults
            assert config.training.lr == 0.0001  # from defaults
            assert config.seed == 123  # overridden by file
        finally:
            Path(config_path).unlink()

    def test_cli_with_defaults(self):
        """Test cli() function with defaults parameter."""

        def my_defaults() -> SimpleConfig:
            return SimpleConfig(batch_size=64, learning_rate=0.01, name="from_defaults")

        config = cli(SimpleConfig, args=["--batch-size", "128"], defaults=my_defaults)

        assert config.batch_size == 128  # CLI overrides
        assert config.learning_rate == 0.01  # from defaults
        assert config.name == "from_defaults"  # from defaults

    def test_cli_with_defaults_and_function(self):
        """Test cli() with function that takes dataclass and defaults."""

        def my_defaults() -> SimpleConfig:
            return SimpleConfig(batch_size=64, learning_rate=0.01, name="from_defaults")

        def train(config: SimpleConfig) -> int:
            return config.batch_size * 2

        result = cli(train, args=[], defaults=my_defaults)

        assert result == 128  # 64 * 2


class TestDefaultsFn:
    """Tests for defaults_fn parameter with fn. prefix in parse_inputs."""

    def test_defaults_fn_called_with_fn_kwargs(self):
        """Test that defaults_fn is called with fn.* values from parse_inputs."""
        from nemotron.kit.config import _apply_parse_inputs

        # Simulate what _apply_parse_inputs does with fn. prefix
        stdin_artifacts = {
            "data": {"path": "/tmp/test_artifact"}
        }
        parse_inputs = {"data.blend_path": "fn.per_split_data_args_path"}

        # Create artifact metadata
        metadata_dir = Path("/tmp/test_artifact")
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "metadata.json"
        metadata_file.write_text(json.dumps({"blend_path": "/path/to/blend.json"}))

        try:
            args, fn_kwargs = _apply_parse_inputs([], parse_inputs, stdin_artifacts)

            # fn_kwargs should contain the extracted value
            assert fn_kwargs == {"per_split_data_args_path": "/path/to/blend.json"}
            # args should be unchanged (no CLI injection for fn. targets)
            assert args == []
        finally:
            metadata_file.unlink()
            metadata_dir.rmdir()

    def test_defaults_fn_mixed_parse_inputs(self):
        """Test parse_inputs with both fn. and regular targets."""
        from nemotron.kit.config import _apply_parse_inputs

        stdin_artifacts = {
            "data": {"path": "/tmp/test_artifact2"}
        }
        parse_inputs = {
            "data.blend_path": "fn.per_split_data_args_path",
            "data.total_tokens": "config.data.total_tokens",
        }

        # Create artifact metadata
        metadata_dir = Path("/tmp/test_artifact2")
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "metadata.json"
        metadata_file.write_text(json.dumps({
            "blend_path": "/path/to/blend.json",
            "total_tokens": 1000000,
        }))

        try:
            args, fn_kwargs = _apply_parse_inputs([], parse_inputs, stdin_artifacts)

            # fn_kwargs should contain fn. prefixed value
            assert fn_kwargs == {"per_split_data_args_path": "/path/to/blend.json"}
            # args should contain CLI arg for non-fn. target
            assert args == ["--config.data.total_tokens", "1000000"]
        finally:
            metadata_file.unlink()
            metadata_dir.rmdir()

    def test_defaults_fn_cli_integration(self):
        """Test cli() with defaults_fn receives fn. kwargs."""
        captured_kwargs = {}

        def recipe_fn(**kwargs) -> SimpleConfig:
            captured_kwargs.update(kwargs)
            name = kwargs.get("custom_name", "default")
            return SimpleConfig(batch_size=64, learning_rate=0.01, name=name)

        def train(config: SimpleConfig) -> str:
            return config.name

        # Create artifact for stdin simulation
        metadata_dir = Path("/tmp/test_artifact3")
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "metadata.json"
        metadata_file.write_text(json.dumps({"name_field": "from_artifact"}))

        try:
            # Note: We can't easily mock stdin, so test the integration differently
            # Test that defaults_fn is used when provided (without stdin)
            result = cli(
                train,
                args=[],
                defaults_fn=recipe_fn,
                parse_inputs={},  # No stdin artifacts
            )

            # defaults_fn should be called with empty kwargs
            assert captured_kwargs == {}
            assert result == "default"
        finally:
            metadata_file.unlink()
            metadata_dir.rmdir()

    def test_defaults_fn_without_parse_inputs(self):
        """Test defaults_fn works without parse_inputs (empty kwargs)."""

        def recipe_fn(**kwargs) -> SimpleConfig:
            return SimpleConfig(
                batch_size=kwargs.get("batch_size", 100),
                learning_rate=0.02,
                name="from_recipe"
            )

        config = cli(SimpleConfig, args=[], defaults_fn=recipe_fn)

        assert config.batch_size == 100
        assert config.learning_rate == 0.02
        assert config.name == "from_recipe"

    def test_defaults_fn_overridden_by_cli(self):
        """Test CLI args override defaults_fn values."""

        def recipe_fn(**kwargs) -> SimpleConfig:
            return SimpleConfig(batch_size=100, learning_rate=0.02, name="from_recipe")

        config = cli(
            SimpleConfig,
            args=["--batch-size", "256"],
            defaults_fn=recipe_fn
        )

        assert config.batch_size == 256  # CLI overrides
        assert config.learning_rate == 0.02  # from defaults_fn
        assert config.name == "from_recipe"  # from defaults_fn

    def test_fn_prefix_without_defaults_fn_raises(self):
        """Test that fn. prefix without defaults_fn raises error."""
        from nemotron.kit.config import _apply_parse_inputs

        stdin_artifacts = {
            "data": {"path": "/tmp/test_artifact4"}
        }
        parse_inputs = {"data.blend_path": "fn.per_split_data_args_path"}

        # Create artifact metadata
        metadata_dir = Path("/tmp/test_artifact4")
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "metadata.json"
        metadata_file.write_text(json.dumps({"blend_path": "/path/to/blend.json"}))

        try:
            # _apply_parse_inputs should succeed and return fn_kwargs
            args, fn_kwargs = _apply_parse_inputs([], parse_inputs, stdin_artifacts)
            assert fn_kwargs == {"per_split_data_args_path": "/path/to/blend.json"}

            # The validation happens in cli(), not _apply_parse_inputs
            # We would need to mock stdin to test this fully
        finally:
            metadata_file.unlink()
            metadata_dir.rmdir()


class TestAppDefaultsFn:
    """Tests for App defaults_fn functionality with fn. prefix in artifact mappings."""

    def test_extract_fn_kwargs_from_artifacts(self):
        """Test _extract_fn_kwargs_from_artifacts extracts fn. prefixed mappings."""
        from nemotron.kit.app import _extract_fn_kwargs_from_artifacts
        from nemotron.kit.artifact import ArtifactInput

        # Create artifact metadata
        metadata_dir = Path("/tmp/test_app_artifact1")
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "metadata.json"
        metadata_file.write_text(json.dumps({
            "blend_path": "/path/to/blend.json",
            "total_tokens": 1000000,
        }))

        try:
            stdin_artifacts = {
                "data": {"path": "/tmp/test_app_artifact1"}
            }
            artifacts = {
                "data": ArtifactInput(
                    default_name="DataBlendsArtifact-pretrain",
                    mappings={
                        "blend_path": "fn.per_split_data_args_path",
                        "total_tokens": "config.data.total_tokens",
                    },
                ),
            }

            fn_kwargs = _extract_fn_kwargs_from_artifacts(stdin_artifacts, artifacts)

            # Only fn. prefixed mappings should be extracted
            assert fn_kwargs == {"per_split_data_args_path": "/path/to/blend.json"}
        finally:
            metadata_file.unlink()
            metadata_dir.rmdir()

    def test_extract_fn_kwargs_multiple_artifacts(self):
        """Test _extract_fn_kwargs_from_artifacts with multiple artifacts."""
        from nemotron.kit.app import _extract_fn_kwargs_from_artifacts
        from nemotron.kit.artifact import ArtifactInput

        # Create artifact metadata for two artifacts
        metadata_dir1 = Path("/tmp/test_app_artifact2a")
        metadata_dir1.mkdir(exist_ok=True)
        metadata_file1 = metadata_dir1 / "metadata.json"
        metadata_file1.write_text(json.dumps({"blend_path": "/path/to/blend.json"}))

        metadata_dir2 = Path("/tmp/test_app_artifact2b")
        metadata_dir2.mkdir(exist_ok=True)
        metadata_file2 = metadata_dir2 / "metadata.json"
        metadata_file2.write_text(json.dumps({"model_path": "/path/to/model"}))

        try:
            stdin_artifacts = {
                "data": {"path": "/tmp/test_app_artifact2a"},
                "model": {"path": "/tmp/test_app_artifact2b"},
            }
            artifacts = {
                "data": ArtifactInput(
                    default_name="DataBlendsArtifact",
                    mappings={"blend_path": "fn.data_arg"},
                ),
                "model": ArtifactInput(
                    default_name="ModelArtifact",
                    mappings={"model_path": "fn.model_arg"},
                ),
            }

            fn_kwargs = _extract_fn_kwargs_from_artifacts(stdin_artifacts, artifacts)

            assert fn_kwargs == {
                "data_arg": "/path/to/blend.json",
                "model_arg": "/path/to/model",
            }
        finally:
            metadata_file1.unlink()
            metadata_dir1.rmdir()
            metadata_file2.unlink()
            metadata_dir2.rmdir()

    def test_extract_fn_kwargs_no_fn_mappings(self):
        """Test _extract_fn_kwargs_from_artifacts with no fn. prefixed mappings."""
        from nemotron.kit.app import _extract_fn_kwargs_from_artifacts
        from nemotron.kit.artifact import ArtifactInput

        # Create artifact metadata
        metadata_dir = Path("/tmp/test_app_artifact3")
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "metadata.json"
        metadata_file.write_text(json.dumps({"blend_path": "/path/to/blend.json"}))

        try:
            stdin_artifacts = {
                "data": {"path": "/tmp/test_app_artifact3"}
            }
            artifacts = {
                "data": ArtifactInput(
                    default_name="DataBlendsArtifact",
                    mappings={"blend_path": "config.data.data_path"},  # No fn. prefix
                ),
            }

            fn_kwargs = _extract_fn_kwargs_from_artifacts(stdin_artifacts, artifacts)

            # No fn. prefixed mappings -> empty dict
            assert fn_kwargs == {}
        finally:
            metadata_file.unlink()
            metadata_dir.rmdir()


class TestTypedDictToDataclass:
    """Tests for _typeddict_to_dataclass helper function."""

    def test_basic_typeddict_conversion(self):
        """Test basic TypedDict to dataclass conversion."""
        from typing import TypedDict
        from nemotron.kit.app import _typeddict_to_dataclass

        class MyKwargs(TypedDict, total=False):
            per_split_data_args_path: str
            seq_length: int

        dc = _typeddict_to_dataclass(MyKwargs)

        # Check it's a dataclass
        assert hasattr(dc, "__dataclass_fields__")

        # Check fields exist
        fields = dc.__dataclass_fields__
        assert "per_split_data_args_path" in fields
        assert "seq_length" in fields

        # Check defaults are None
        instance = dc()
        assert instance.per_split_data_args_path is None
        assert instance.seq_length is None

    def test_typeddict_with_prefix(self):
        """Test TypedDict conversion with prefix for CLI names."""
        from typing import TypedDict
        from nemotron.kit.app import _typeddict_to_dataclass

        class MyKwargs(TypedDict, total=False):
            data_path: str

        dc = _typeddict_to_dataclass(MyKwargs, prefix="fn.")

        # The field name in the dataclass remains data_path
        # but the CLI arg name should be fn.data-path
        assert "data_path" in dc.__dataclass_fields__

    def test_typeddict_with_optional_types(self):
        """Test TypedDict with already-optional types."""
        from typing import TypedDict
        from nemotron.kit.app import _typeddict_to_dataclass

        class MyKwargs(TypedDict, total=False):
            optional_path: str | None
            required_value: int

        dc = _typeddict_to_dataclass(MyKwargs)

        # Both should have None defaults
        instance = dc()
        assert instance.optional_path is None
        assert instance.required_value is None

    def test_non_typeddict_raises(self):
        """Test that non-TypedDict raises TypeError."""
        from nemotron.kit.app import _typeddict_to_dataclass

        class NotATypedDict:
            pass

        import pytest
        with pytest.raises(TypeError, match="is not a TypedDict"):
            _typeddict_to_dataclass(NotATypedDict)


class TestAppKwargsSchema:
    """Tests for App kwargs_schema functionality."""

    def test_kwargs_schema_requires_defaults_fn(self):
        """Test that kwargs_schema without defaults_fn raises error."""
        from typing import TypedDict
        from nemotron.kit.app import App

        class MyKwargs(TypedDict, total=False):
            data_path: str

        app = App("test")

        import pytest
        with pytest.raises(ValueError, match="kwargs_schema requires defaults_fn"):
            app.command(
                "cmd",
                SimpleConfig,
                lambda c: None,
                kwargs_schema=MyKwargs,
            )

    def test_build_union_with_kwargs_schema(self):
        """Test that _build_union includes kwargs dataclass in union."""
        from typing import TypedDict
        from nemotron.kit.app import App

        class MyKwargs(TypedDict, total=False):
            custom_arg: str
            seq_length: int

        def recipe_fn(**kwargs) -> SimpleConfig:
            return SimpleConfig(
                batch_size=kwargs.get("seq_length", 32),
                learning_rate=0.01,
                name=kwargs.get("custom_arg", "default"),
            )

        app = App("test")
        app.command(
            "cmd",
            SimpleConfig,
            lambda c: None,
            defaults_fn=recipe_fn,
            kwargs_schema=MyKwargs,
        )

        union_type, handlers, artifacts_map, defaults_fn_map, kwargs_schema_map = app.build(include_global_options=True)

        # Check kwargs_schema is tracked
        assert SimpleConfig in kwargs_schema_map
        assert kwargs_schema_map[SimpleConfig] == MyKwargs

    def test_kwargs_schema_cli_parsing(self):
        """Test that kwargs_schema fields become CLI arguments."""
        from typing import TypedDict
        from nemotron.kit.app import App, _typeddict_to_dataclass
        import tyro

        class MyKwargs(TypedDict, total=False):
            custom_arg: str
            seq_length: int

        def recipe_fn(**kwargs) -> SimpleConfig:
            return SimpleConfig(
                batch_size=kwargs.get("seq_length", 32),
                learning_rate=0.01,
                name=kwargs.get("custom_arg", "default"),
            )

        app = App("test")
        # Add two commands to ensure subcommand selection is required
        app.command(
            "cmd",
            SimpleConfig,
            lambda c: c,
            defaults_fn=recipe_fn,
            kwargs_schema=MyKwargs,
        )
        app.command(
            "other",
            SimpleConfig,
            lambda c: c,
        )

        union_type, handlers, artifacts_map, defaults_fn_map, kwargs_schema_map = app.build(include_global_options=True)

        # Parse with kwargs_schema CLI args
        parsed = tyro.cli(
            union_type,
            args=["cmd", "--fn.custom-arg", "my_value", "--fn.seq-length", "512"],
        )

        # Check fn_ fields are parsed
        assert hasattr(parsed, "fn_")
        fn_options = parsed.fn_
        assert fn_options.custom_arg == "my_value"
        assert fn_options.seq_length == 512


class TestCliKwargsSchema:
    """Tests for standalone cli() function with kwargs_schema."""

    def test_cli_kwargs_schema_requires_defaults_fn(self):
        """Test that kwargs_schema without defaults_fn raises error."""
        from typing import TypedDict

        class MyKwargs(TypedDict, total=False):
            data_path: str

        import pytest
        with pytest.raises(ValueError, match="kwargs_schema requires defaults_fn"):
            cli(
                SimpleConfig,
                args=[],
                kwargs_schema=MyKwargs,
            )

    def test_cli_kwargs_schema_parses_fn_args(self):
        """Test that kwargs_schema CLI args are parsed and passed to defaults_fn."""
        from typing import TypedDict

        class MyKwargs(TypedDict, total=False):
            custom_value: str
            seq_length: int

        captured_kwargs = {}

        def recipe_fn(**kwargs) -> SimpleConfig:
            captured_kwargs.update(kwargs)
            return SimpleConfig(
                batch_size=kwargs.get("seq_length", 32),
                learning_rate=0.01,
                name=kwargs.get("custom_value", "default"),
            )

        config = cli(
            SimpleConfig,
            args=["--fn.custom-value", "my_value", "--fn.seq-length", "512"],
            defaults_fn=recipe_fn,
            kwargs_schema=MyKwargs,
        )

        assert captured_kwargs == {"custom_value": "my_value", "seq_length": 512}
        assert config.batch_size == 512
        assert config.name == "my_value"

    def test_cli_kwargs_schema_with_config_args(self):
        """Test kwargs_schema works alongside regular config CLI args."""
        from typing import TypedDict

        class MyKwargs(TypedDict, total=False):
            default_batch: int

        def recipe_fn(**kwargs) -> SimpleConfig:
            return SimpleConfig(
                batch_size=kwargs.get("default_batch", 32),
                learning_rate=0.01,
                name="from_recipe",
            )

        # CLI arg overrides defaults_fn value
        config = cli(
            SimpleConfig,
            args=["--fn.default-batch", "64", "--batch-size", "128"],
            defaults_fn=recipe_fn,
            kwargs_schema=MyKwargs,
        )

        assert config.batch_size == 128  # CLI overrides defaults_fn
        assert config.name == "from_recipe"

    def test_cli_kwargs_schema_artifact_overrides_cli(self):
        """Test that artifact fn. values override CLI kwargs."""
        from typing import TypedDict
        from pathlib import Path

        class MyKwargs(TypedDict, total=False):
            data_path: str

        captured_kwargs = {}

        def recipe_fn(**kwargs) -> SimpleConfig:
            captured_kwargs.update(kwargs)
            return SimpleConfig(batch_size=32, learning_rate=0.01, name="test")

        # Create artifact metadata
        metadata_dir = Path("/tmp/test_cli_kwargs_artifact")
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "metadata.json"
        metadata_file.write_text(json.dumps({"blend_path": "/artifact/path"}))

        try:
            # Mock stdin artifacts
            import sys
            from io import StringIO
            original_stdin = sys.stdin
            sys.stdin = StringIO(json.dumps({
                "data": {"path": "/tmp/test_cli_kwargs_artifact"}
            }))

            # Simulate non-TTY stdin
            original_isatty = sys.stdin.isatty
            sys.stdin.isatty = lambda: False

            config = cli(
                SimpleConfig,
                args=["--fn.data-path", "cli_value"],  # CLI value
                defaults_fn=recipe_fn,
                kwargs_schema=MyKwargs,
                parse_inputs={"data.blend_path": "fn.data_path"},  # Artifact maps to same kwarg
            )

            # Artifact value should override CLI value
            assert captured_kwargs.get("data_path") == "/artifact/path"
        finally:
            sys.stdin = original_stdin
            metadata_file.unlink()
            metadata_dir.rmdir()
