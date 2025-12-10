"""
Tests for nemotron artifact functionality.
"""

import json
import tempfile
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import Field

from nemotron.kit import Artifact, DataBlendsArtifact, ModelArtifact, apply_scale


class SampleDataset(Artifact):
    """Sample artifact for testing."""

    num_examples: Annotated[int, Field(gt=0)]
    quality: Annotated[float, Field(ge=0.0, le=1.0)]


def test_artifact_save_and_load():
    """Test saving and loading an artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_artifact"

        # Create artifact
        artifact = SampleDataset(
            path=output_dir,
            num_examples=100,
            quality=0.85,
        )

        # Save
        artifact.save()

        # Verify metadata.json exists
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        # Load and verify
        loaded = SampleDataset.load(path=output_dir)
        assert loaded.num_examples == 100
        assert loaded.quality == 0.85
        # Typed fields are synced to metadata
        assert loaded.metadata["num_examples"] == 100
        assert loaded.metadata["quality"] == 0.85


def test_artifact_validation():
    """Test Pydantic validation works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_validation"

        # Valid artifact
        valid = SampleDataset(path=output_dir, num_examples=100, quality=0.5)
        assert valid.num_examples == 100

        # Invalid: negative num_examples
        with pytest.raises(Exception):  # Pydantic ValidationError
            SampleDataset(path=output_dir, num_examples=-1, quality=0.5)

        # Invalid: quality out of range
        with pytest.raises(Exception):  # Pydantic ValidationError
            SampleDataset(path=output_dir, num_examples=100, quality=1.5)


def test_artifact_metadata_format():
    """Test that metadata.json has correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_format"

        artifact = SampleDataset(
            path=output_dir,
            num_examples=50,
            quality=0.75,
        )
        artifact.save()

        # Load metadata.json
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Verify required fields
        assert metadata["type"] == "SampleDataset"
        assert metadata["num_examples"] == 50
        assert metadata["quality"] == 0.75
        assert metadata["producer"] == "local"
        # Typed fields are in metadata dict
        assert metadata["metadata"]["num_examples"] == 50
        assert metadata["metadata"]["quality"] == 0.75


def test_artifact_metrics_property():
    """Test that metrics property extracts numeric values from metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_metrics"

        artifact = SampleDataset(
            path=output_dir,
            num_examples=100,
            quality=0.85,
        )

        # Metrics should be extracted from metadata
        metrics = artifact.metrics
        assert metrics["num_examples"] == 100.0
        assert metrics["quality"] == 0.85


def test_apply_scale():
    """Test scale factor utility."""
    assert apply_scale(100_000, "tiny") == 1_000  # 1%
    assert apply_scale(100_000, "small") == 10_000  # 10%
    assert apply_scale(100_000, "medium") == 30_000  # 30%
    assert apply_scale(100_000, "full") == 100_000  # 100%

    # Minimum 1 even for tiny scale
    assert apply_scale(10, "tiny") == 1

    # Cap tiny at 10k rows
    assert apply_scale(2_000_000, "tiny") == 10_000  # Would be 20k, capped at 10k
    assert apply_scale(500_000, "tiny") == 5_000  # Under cap, not affected

    # Invalid scale
    with pytest.raises(ValueError):
        apply_scale(100, "invalid")


def test_artifact_type_inference():
    """Test that artifact type is inferred from class name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_type"

        artifact = SampleDataset(path=output_dir, num_examples=10, quality=0.5)

        # Type should be class name
        assert artifact.type == "SampleDataset"


def test_data_blends_artifact():
    """Test DataBlendsArtifact typed subclass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blend_path = Path(tmpdir) / "blend.json"

        artifact = DataBlendsArtifact(
            path=blend_path,
            total_tokens=1_000_000,
            total_sequences=10_000,
            elapsed_sec=120.5,
        )

        # Typed fields accessible directly
        assert artifact.total_tokens == 1_000_000
        assert artifact.total_sequences == 10_000
        assert artifact.elapsed_sec == 120.5

        # Also in metadata
        assert artifact.metadata["total_tokens"] == 1_000_000
        assert artifact.metadata["total_sequences"] == 10_000

        # Type inferred from class name
        assert artifact.type == "DataBlendsArtifact"


def test_data_blends_artifact_save():
    """Test DataBlendsArtifact.save() writes metadata.json to parent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blend_path = Path(tmpdir) / "output" / "blend.json"

        # Create blend.json file first (simulating pipeline.py behavior)
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        blend_path.write_text('{"blends": []}')

        artifact = DataBlendsArtifact(
            path=blend_path,
            total_tokens=1_000_000,
            total_sequences=10_000,
            elapsed_sec=120.5,
        )

        # Save should not fail even though blend.json already exists
        artifact.save()

        # metadata.json should be in the parent directory (same as blend.json)
        metadata_path = blend_path.parent / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["type"] == "DataBlendsArtifact"
        assert metadata["total_tokens"] == 1_000_000
        assert metadata["total_sequences"] == 10_000
        assert metadata["producer"] == "local"


def test_model_artifact():
    """Test ModelArtifact typed subclass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint"

        artifact = ModelArtifact(
            path=checkpoint_path,
            step=10000,
            final_loss=1.234,
        )

        # Typed fields accessible directly
        assert artifact.step == 10000
        assert artifact.final_loss == 1.234

        # Also in metadata
        assert artifact.metadata["step"] == 10000
        assert artifact.metadata["final_loss"] == 1.234

        # Type inferred from class name
        assert artifact.type == "ModelArtifact"


def test_artifact_metadata_sync():
    """Test that typed fields sync to metadata and back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_sync"

        # Create artifact with typed fields
        artifact = SampleDataset(
            path=output_dir,
            num_examples=42,
            quality=0.99,
        )
        artifact.save()

        # Load from disk
        loaded = SampleDataset.load(path=output_dir)

        # Typed fields should be restored
        assert loaded.num_examples == 42
        assert loaded.quality == 0.99

        # And synced to metadata
        assert loaded.metadata["num_examples"] == 42
        assert loaded.metadata["quality"] == 0.99


def test_simple_artifact_no_subclass():
    """Test using Artifact directly with metadata dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_simple"

        # Create simple artifact without subclass
        artifact = Artifact(
            path=output_dir,
            type="custom",
            metadata={"custom_field": 123, "another": "value"},
        )
        artifact.save()

        # Load and verify
        loaded = Artifact.load(path=output_dir)
        assert loaded.type == "custom"
        assert loaded.metadata["custom_field"] == 123
        assert loaded.metadata["another"] == "value"


def test_artifact_art_path_with_name():
    """Test that art_path uses semantic name when set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_name"

        # Create artifact with semantic name
        artifact = SampleDataset(
            path=output_dir,
            num_examples=10,
            quality=0.5,
            name="nano3/pretrain/data",
        )

        # art_path should use the semantic name
        assert artifact.art_path == "art://nano3/pretrain/data"


def test_artifact_art_path_fallback():
    """Test that art_path falls back to full path when no name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_fallback"

        # Create artifact without name
        artifact = SampleDataset(
            path=output_dir,
            num_examples=10,
            quality=0.5,
        )

        # art_path should be the full path
        assert artifact.art_path == f"art://{output_dir.resolve()}"


def test_data_blends_artifact_with_source_uris():
    """Test DataBlendsArtifact with source URIs for lineage tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blend_path = Path(tmpdir) / "output" / "blend.json"
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        blend_path.write_text('{"blends": []}')

        # Create artifact with source URIs
        source_datasets = [
            "hf://nvidia/Nemotron-CC-v2",
            "s3://my-bucket/data/train.jsonl",
            "/local/path/to/data",
        ]
        tokenizer_uri = "https://huggingface.co/meta-llama/Llama-3.2-1B"

        artifact = DataBlendsArtifact(
            path=blend_path,
            total_tokens=1_000_000,
            total_sequences=10_000,
            elapsed_sec=120.5,
            source_datasets=source_datasets,
            tokenizer_uri=tokenizer_uri,
        )

        # Verify source URIs are accessible
        assert artifact.source_datasets == source_datasets
        assert artifact.tokenizer_uri == tokenizer_uri

        # Save and verify metadata includes source URIs
        artifact.save()
        metadata_path = blend_path.parent / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["source_datasets"] == source_datasets
        assert metadata["tokenizer_uri"] == tokenizer_uri
        # Also in nested metadata dict
        assert metadata["metadata"]["source_datasets"] == source_datasets
        assert metadata["metadata"]["tokenizer_uri"] == tokenizer_uri


def test_data_blends_artifact_default_source_uris():
    """Test DataBlendsArtifact defaults for source URIs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blend_path = Path(tmpdir) / "blend.json"

        # Create artifact without source URIs
        artifact = DataBlendsArtifact(
            path=blend_path,
            total_tokens=1_000_000,
            total_sequences=10_000,
        )

        # Default values
        assert artifact.source_datasets == []
        assert artifact.tokenizer_uri is None


def test_to_wandb_uri():
    """Test URI conversion for W&B artifact references."""
    from nemotron.kit.trackers import to_wandb_uri

    # HuggingFace dataset
    assert to_wandb_uri("hf://nvidia/Nemotron-CC-v2") == "https://huggingface.co/datasets/nvidia/Nemotron-CC-v2"
    assert to_wandb_uri("hf://allenai/c4") == "https://huggingface.co/datasets/allenai/c4"

    # S3 and GCS URIs pass through
    assert to_wandb_uri("s3://bucket/key") == "s3://bucket/key"
    assert to_wandb_uri("gs://bucket/key") == "gs://bucket/key"

    # HTTP/HTTPS pass through
    assert to_wandb_uri("https://example.com/data.json") == "https://example.com/data.json"

    # file:// pass through
    assert to_wandb_uri("file:///path/to/data") == "file:///path/to/data"

    # Local paths convert to file://
    result = to_wandb_uri("/data/train.jsonl")
    assert result.startswith("file:///")
    assert "train.jsonl" in result


def test_tokenizer_to_uri():
    """Test tokenizer URI generation."""
    from nemotron.kit.trackers import tokenizer_to_uri

    # HuggingFace model
    assert tokenizer_to_uri("meta-llama/Llama-3.2-1B") == "https://huggingface.co/meta-llama/Llama-3.2-1B"
    assert tokenizer_to_uri("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF") == "https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

    # With revision
    result = tokenizer_to_uri("meta-llama/Llama-3.2-1B", revision="abc123")
    assert result == "https://huggingface.co/meta-llama/Llama-3.2-1B/tree/abc123"

    # Local path
    result = tokenizer_to_uri("/path/to/tokenizer")
    assert result.startswith("file:///")
    assert "tokenizer" in result


def test_load_wandb_config_from_toml():
    """Test loading wandb config from run.toml."""
    from nemotron.kit.run import load_wandb_config

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a run.toml with wandb section
        run_toml = Path(tmpdir) / "run.toml"
        run_toml.write_text("""
[wandb]
project = "test-project"
entity = "test-team"
tags = ["tag1", "tag2"]
notes = "Test notes"

[local]
executor = "local"
nproc_per_node = 8
""")
        # Load wandb config
        config = load_wandb_config(config_path=run_toml)

        assert config is not None
        assert config.project == "test-project"
        assert config.entity == "test-team"
        assert config.tags == ("tag1", "tag2")
        assert config.notes == "Test notes"
        assert config.enabled is True


def test_load_wandb_config_missing_section():
    """Test loading wandb config when [wandb] section is missing."""
    from nemotron.kit.run import load_wandb_config

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a run.toml without wandb section
        run_toml = Path(tmpdir) / "run.toml"
        run_toml.write_text("""
[local]
executor = "local"
nproc_per_node = 8
""")
        # Load wandb config - should return None
        config = load_wandb_config(config_path=run_toml)
        assert config is None


def test_load_wandb_config_with_name_shorthand():
    """Test loading wandb config with 'name' shorthand for run_name."""
    from nemotron.kit.run import load_wandb_config

    with tempfile.TemporaryDirectory() as tmpdir:
        run_toml = Path(tmpdir) / "run.toml"
        run_toml.write_text("""
[wandb]
project = "test-project"
name = "my-run-name"
""")
        config = load_wandb_config(config_path=run_toml)

        assert config is not None
        assert config.run_name == "my-run-name"


def test_config_manager_extracts_wandb_section():
    """Test that ConfigManager extracts [wandb] section from config files."""
    from dataclasses import dataclass
    from nemotron.kit.config import ConfigManager

    @dataclass
    class TestConfig:
        batch_size: int = 32
        learning_rate: float = 1e-4

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a config file with wandb section
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text("""
batch_size = 64
learning_rate = 1e-3

[wandb]
project = "my-project"
entity = "my-team"
""")
        # Parse config
        manager = ConfigManager(TestConfig)
        config = manager.parse_args(["--config-file", str(config_file)])

        # Recipe config should be parsed
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3

        # Wandb config should be extracted
        wandb_config = manager.get_wandb_config()
        assert wandb_config is not None
        assert wandb_config.project == "my-project"
        assert wandb_config.entity == "my-team"


def test_config_manager_no_wandb_section():
    """Test ConfigManager when no [wandb] section is present."""
    from dataclasses import dataclass
    from nemotron.kit.config import ConfigManager

    @dataclass
    class TestConfig:
        batch_size: int = 32

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text("batch_size = 64")

        manager = ConfigManager(TestConfig)
        config = manager.parse_args(["--config-file", str(config_file)])

        assert config.batch_size == 64
        assert manager.get_wandb_config() is None
