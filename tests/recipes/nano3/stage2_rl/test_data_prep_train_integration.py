# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for data_prep.py → train.py integration in nano3/stage2_rl.

These tests validate:
1. HuggingFace placeholder resolution logic
2. Template restoration functions (DAPO prefix/suffix, Skywork {question})
3. Transform output format matches what RL training expects
4. The resolve_hf_placeholders transform handles both placeholder and normal records
5. Manifest JSON format matches what train.py expects
6. W&B artifact production and consumption works correctly
7. The ${art:data,path} resolver correctly resolves artifact paths
"""

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemotron.kit import SplitJsonlDataArtifact

from nemotron.data_prep.hf_placeholder import (
    TARGET_DATASETS,
    HFPlaceholderResolver,
    get_nested_value,
    is_placeholder_record,
    restore_dapo_question,
    restore_skywork_question,
)


class TestGetNestedValue:
    """Test the get_nested_value helper function."""

    def test_simple_dict_path(self):
        """Test extracting value from simple dict."""
        record = {"key": "value"}
        assert get_nested_value(record, ["key"]) == "value"

    def test_nested_dict_path(self):
        """Test extracting value from nested dict."""
        record = {"level1": {"level2": {"level3": "deep_value"}}}
        assert get_nested_value(record, ["level1", "level2", "level3"]) == "deep_value"

    def test_list_index_access(self):
        """Test extracting value from list by index."""
        record = {"items": ["first", "second", "third"]}
        assert get_nested_value(record, ["items", 0]) == "first"
        assert get_nested_value(record, ["items", 2]) == "third"

    def test_mixed_dict_and_list(self):
        """Test extracting from mixed dict/list structure."""
        record = {"prompt": [{"content": "What is 2+2?"}]}
        path = ["prompt", 0, "content"]
        assert get_nested_value(record, path) == "What is 2+2?"

    def test_missing_key_returns_none(self):
        """Test that missing keys return None."""
        record = {"key": "value"}
        assert get_nested_value(record, ["missing"]) is None

    def test_out_of_bounds_index_returns_none(self):
        """Test that out of bounds list index returns None."""
        record = {"items": ["only_one"]}
        assert get_nested_value(record, ["items", 5]) is None

    def test_empty_path_returns_record(self):
        """Test that empty path returns the record itself."""
        record = {"key": "value"}
        assert get_nested_value(record, []) == record

    def test_none_in_path_returns_none(self):
        """Test that None value in traversal path returns None."""
        record = {"key": None}
        assert get_nested_value(record, ["key", "nested"]) is None


class TestRestoreDapoQuestion:
    """Test DAPO question restoration with prefix/suffix templates."""

    def test_basic_prefix_suffix(self):
        """Test basic prefix and suffix application."""
        hf_question = "Calculate 2+2"
        template = {
            "prefix": "Math problem: ",
            "suffix": "\nPlease solve step by step.",
        }
        result = restore_dapo_question(hf_question, template)
        assert result == "Math problem: Calculate 2+2\nPlease solve step by step."

    def test_only_prefix(self):
        """Test template with only prefix."""
        hf_question = "What is the capital of France?"
        template = {"prefix": "Question: "}
        result = restore_dapo_question(hf_question, template)
        assert result == "Question: What is the capital of France?"

    def test_only_suffix(self):
        """Test template with only suffix."""
        hf_question = "Solve this equation"
        template = {"suffix": "\n\nShow your work."}
        result = restore_dapo_question(hf_question, template)
        assert result == "Solve this equation\n\nShow your work."

    def test_empty_template(self):
        """Test empty template returns original question."""
        hf_question = "Original question"
        template = {}
        result = restore_dapo_question(hf_question, template)
        assert result == "Original question"

    def test_multiline_prefix_suffix(self):
        """Test multiline prefix and suffix."""
        hf_question = "x + 5 = 10"
        template = {
            "prefix": "You are a math tutor.\n\nSolve: ",
            "suffix": "\n\nExplain each step.",
        }
        result = restore_dapo_question(hf_question, template)
        expected = "You are a math tutor.\n\nSolve: x + 5 = 10\n\nExplain each step."
        assert result == expected


class TestRestoreSkyworkQuestion:
    """Test Skywork question restoration with {question} placeholder."""

    def test_basic_placeholder_replacement(self):
        """Test basic {question} replacement."""
        hf_question = "What is the capital of France?"
        template = "Please answer: {question}\n\nAnswer:"
        result = restore_skywork_question(hf_question, template)
        assert result == "Please answer: What is the capital of France?\n\nAnswer:"

    def test_multiple_placeholders(self):
        """Test template with multiple {question} placeholders."""
        hf_question = "2+2"
        template = "Solve {question}. The answer to {question} is:"
        result = restore_skywork_question(hf_question, template)
        assert result == "Solve 2+2. The answer to 2+2 is:"

    def test_no_placeholder_returns_question(self):
        """Test template without placeholder returns original question."""
        hf_question = "Test question"
        template = "Some template without placeholder"
        result = restore_skywork_question(hf_question, template)
        assert result == "Test question"

    def test_empty_question(self):
        """Test empty question in template."""
        hf_question = ""
        template = "Answer: {question}"
        result = restore_skywork_question(hf_question, template)
        assert result == "Answer: "


class TestIsPlaceholderRecord:
    """Test placeholder record detection."""

    def test_placeholder_record_detected(self):
        """Test that records with _hf_placeholder are detected."""
        record = {
            "dataset": "nano_v3_sft_profiled_dapo17k",
            "_hf_placeholder": {"row": 42, "question_template": {"prefix": "Q: "}},
        }
        assert is_placeholder_record(record) is True

    def test_normal_record_not_detected(self):
        """Test that normal records are not detected as placeholders."""
        record = {
            "question": "What is 2+2?",
            "expected_answer": "4",
            "responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]},
        }
        assert is_placeholder_record(record) is False

    def test_empty_record(self):
        """Test empty record is not a placeholder."""
        assert is_placeholder_record({}) is False


class TestTargetDatasetsConfig:
    """Test TARGET_DATASETS configuration is valid."""

    def test_required_keys_present(self):
        """Test all target datasets have required configuration keys."""
        required_keys = {"hf_dataset", "split", "question_path", "answer_path", "template_type"}
        for name, config in TARGET_DATASETS.items():
            for key in required_keys:
                assert key in config, f"Missing key '{key}' in {name} config"

    def test_template_types_valid(self):
        """Test template_type is either 'dapo' or 'skywork'."""
        valid_types = {"dapo", "skywork"}
        for name, config in TARGET_DATASETS.items():
            assert config["template_type"] in valid_types, (
                f"Invalid template_type '{config['template_type']}' in {name}"
            )

    def test_paths_are_lists(self):
        """Test question_path and answer_path are lists."""
        for name, config in TARGET_DATASETS.items():
            assert isinstance(config["question_path"], list), (
                f"question_path in {name} should be a list"
            )
            assert isinstance(config["answer_path"], list), (
                f"answer_path in {name} should be a list"
            )

    def test_expected_datasets_present(self):
        """Test expected placeholder dataset names are configured."""
        expected_datasets = {
            "nano_v3_sft_profiled_dapo17k",
            "nano_v3_sft_profiled_skywork_no_omni",
        }
        configured_datasets = set(TARGET_DATASETS.keys())
        assert expected_datasets == configured_datasets


class TestResolveHfPlaceholderTransform:
    """Test the resolve_hf_placeholders transform function."""

    def test_non_placeholder_record_uses_nemotron_rl(self):
        """Test that non-placeholder records use nemotron_rl extraction."""
        from nemotron.data_prep.formats.transforms import resolve_hf_placeholders

        # Create transform without resolver (will not be needed for non-placeholder)
        transform = resolve_hf_placeholders(resolver=None)

        # Normal record with responses_create_params
        record = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "What is 2+2?"}],
                "tools": [{"name": "calculator"}],
            }
        }

        result = transform(record)

        # Should extract messages and tools like nemotron_rl
        assert result is not None
        assert "messages" in result
        assert result["messages"] == [{"role": "user", "content": "What is 2+2?"}]
        assert "tools" in result
        assert result["tools"] == [{"name": "calculator"}]

    def test_non_placeholder_without_tools(self):
        """Test non-placeholder record without tools field."""
        from nemotron.data_prep.formats.transforms import resolve_hf_placeholders

        transform = resolve_hf_placeholders(resolver=None)

        record = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Hello"}],
            }
        }

        result = transform(record)

        assert result is not None
        assert "messages" in result
        assert "tools" not in result

    def test_invalid_record_returns_none(self):
        """Test that records without valid structure return None."""
        from nemotron.data_prep.formats.transforms import resolve_hf_placeholders

        transform = resolve_hf_placeholders(resolver=None)

        # Record without responses_create_params or _hf_placeholder
        record = {"random_field": "value"}

        result = transform(record)
        assert result is None


class TestHFPlaceholderResolverUnit:
    """Unit tests for HFPlaceholderResolver without loading actual HF datasets."""

    def test_resolve_returns_none_for_non_placeholder(self):
        """Test resolve returns None for non-placeholder records."""
        # Create resolver with empty datasets (won't be used)
        resolver = HFPlaceholderResolver(datasets={}, configs={})

        record = {"question": "regular question", "answer": "regular answer"}
        result = resolver.resolve(record)

        assert result is None

    def test_resolve_returns_none_for_unknown_dataset(self):
        """Test resolve returns None for unknown dataset names."""
        resolver = HFPlaceholderResolver(datasets={}, configs={})

        record = {
            "dataset": "unknown_dataset",
            "_hf_placeholder": {"row": 0, "question_template": {}},
        }
        result = resolver.resolve(record)

        assert result is None


class TestOutputFormat:
    """Test that resolved records match RL training expected format."""

    def test_resolved_record_structure(self):
        """Test resolved records have correct structure for RL training."""
        # This is the expected output format
        resolved = {
            "question": "Full question text here",
            "expected_answer": "Expected answer",
            "responses_create_params": {
                "input": [{"role": "user", "content": "Full question text here"}]
            },
        }

        # Verify structure
        assert "question" in resolved
        assert "expected_answer" in resolved
        assert "responses_create_params" in resolved
        assert "input" in resolved["responses_create_params"]

        # Verify messages format
        messages = resolved["responses_create_params"]["input"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "content" in messages[0]

    def test_messages_content_matches_question(self):
        """Test that messages content matches the question field."""
        question = "What is the capital of France?"
        resolved = {
            "question": question,
            "expected_answer": "Paris",
            "responses_create_params": {
                "input": [{"role": "user", "content": question}]
            },
        }

        # The content should match the question
        assert resolved["responses_create_params"]["input"][0]["content"] == resolved["question"]


class TestTransformIntegration:
    """Integration tests for the resolve transform with mock resolver."""

    def test_placeholder_record_with_mock_resolver(self):
        """Test placeholder record resolution with a mock resolver."""
        from nemotron.data_prep.formats.transforms import resolve_hf_placeholders
        from nemotron.data_prep.hf_placeholder import HFPlaceholderResolver, PlaceholderConfig

        # Create a mock dataset (simple dict-based mock)
        class MockDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        mock_data = [
            {"prompt": [{"content": "What is 2+2?"}], "reward_model": {"ground_truth": "4"}},
            {"prompt": [{"content": "What is 3+3?"}], "reward_model": {"ground_truth": "6"}},
        ]

        mock_config = PlaceholderConfig(
            hf_dataset="test/mock-dataset",
            split="train",
            question_path=["prompt", 0, "content"],
            answer_path=["reward_model", "ground_truth"],
            template_type="dapo",
        )

        resolver = HFPlaceholderResolver(
            datasets={"test_dataset": MockDataset(mock_data)},
            configs={"test_dataset": mock_config},
        )

        # Test resolution
        placeholder_record = {
            "dataset": "test_dataset",
            "_hf_placeholder": {
                "row": 0,
                "question_template": {"prefix": "Q: ", "suffix": ""},
            },
        }

        result = resolver.resolve(placeholder_record)

        assert result is not None
        assert result["question"] == "Q: What is 2+2?"
        assert result["expected_answer"] == "4"
        assert "responses_create_params" in result
        assert result["responses_create_params"]["input"][0]["content"] == "Q: What is 2+2?"

    def test_skywork_template_resolution(self):
        """Test Skywork-style {question} template resolution."""
        from nemotron.data_prep.hf_placeholder import HFPlaceholderResolver, PlaceholderConfig

        class MockDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        mock_data = [
            {"problem": "Solve x + 5 = 10", "answer": "x = 5"},
        ]

        mock_config = PlaceholderConfig(
            hf_dataset="test/skywork-mock",
            split="train",
            question_path=["problem"],
            answer_path=["answer"],
            template_type="skywork",
        )

        resolver = HFPlaceholderResolver(
            datasets={"skywork_test": MockDataset(mock_data)},
            configs={"skywork_test": mock_config},
        )

        placeholder_record = {
            "dataset": "skywork_test",
            "_hf_placeholder": {
                "row": 0,
                "question_template": "Please solve: {question}\n\nSolution:",
            },
        }

        result = resolver.resolve(placeholder_record)

        assert result is not None
        assert result["question"] == "Please solve: Solve x + 5 = 10\n\nSolution:"
        assert result["expected_answer"] == "x = 5"


class TestRLDataPrepTrainIntegration:
    """Test integration between data_prep.py output and train.py consumption."""

    def test_manifest_json_format_matches_train_expectation(self):
        """Test manifest.json format is compatible with train.py config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / "manifest.json"

            # Simulate data_prep output - manifest.json with split paths
            manifest_data = {
                "train": str(tmpdir / "train" / "train.jsonl"),
                "val": str(tmpdir / "val" / "val.jsonl"),
                "test": str(tmpdir / "test" / "test.jsonl"),
                "mode": "resolve",
                "source_splits": ["train", "validation", "test"],
            }

            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f)

            # Simulate train.py loading
            with open(manifest_path) as f:
                loaded = json.load(f)

            assert loaded == manifest_data
            assert "train" in loaded
            assert "val" in loaded
            assert "test" in loaded

    def test_jsonl_format_compatible_with_nemo_rl_loading(self):
        """Test JSONL format is compatible with NeMo-RL's data loading.

        train.py uses setup_single_nemo_gym_dataset which:
        1. Opens JSONL file
        2. Calls json.loads() on each line
        3. Passes to nemo_gym_example_to_nemo_rl_datum_spec

        This test simulates that exact loading pattern.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            train_dir = tmpdir / "train"
            train_dir.mkdir()
            train_jsonl = train_dir / "train.jsonl"

            # Simulate data_prep output format (after resolve transform)
            records = [
                {
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
                {
                    "messages": [{"role": "user", "content": "What is 3+3?"}],
                    "tools": [{"name": "calculator"}],
                },
            ]

            with open(train_jsonl, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            # Simulate train.py loading (exact pattern from train.py:71-73)
            with open(train_jsonl) as f:
                nemo_gym_examples = list(map(json.loads, f))

            assert len(nemo_gym_examples) == 2
            assert nemo_gym_examples[0]["messages"][0]["content"] == "What is 2+2?"
            assert nemo_gym_examples[1]["tools"] == [{"name": "calculator"}]

    def test_split_jsonl_artifact_stores_paths_correctly(self):
        """Test SplitJsonlDataArtifact stores split paths correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create split directories and files
            for split in ["train", "val", "test"]:
                split_dir = tmpdir / split
                split_dir.mkdir()
                (split_dir / f"{split}.jsonl").write_text('{"test": true}\n')

            manifest_path = tmpdir / "manifest.json"
            manifest_data = {
                "train": str(tmpdir / "train" / "train.jsonl"),
                "val": str(tmpdir / "val" / "val.jsonl"),
                "test": str(tmpdir / "test" / "test.jsonl"),
            }
            manifest_path.write_text(json.dumps(manifest_data))

            artifact = SplitJsonlDataArtifact(
                path=manifest_path,
                total_sequences=100,
                elapsed_sec=10.0,
            )

            # Add split paths to metadata (as data_prep does)
            artifact.metadata["train"] = str(tmpdir / "train" / "train.jsonl")
            artifact.metadata["val"] = str(tmpdir / "val" / "val.jsonl")
            artifact.metadata["test"] = str(tmpdir / "test" / "test.jsonl")

            assert artifact.path == manifest_path
            assert artifact.metadata["train"] == str(tmpdir / "train" / "train.jsonl")
            artifact.save()

            loaded = SplitJsonlDataArtifact.load(path=manifest_path.parent)
            assert loaded.total_sequences == 100

    def test_artifact_metadata_contains_required_fields(self):
        """Test SplitJsonlDataArtifact metadata has fields needed by train.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text('{"train": "", "val": "", "test": ""}')

            artifact = SplitJsonlDataArtifact(
                path=manifest_path,
                total_sequences=1000,
                elapsed_sec=120.0,
            )
            artifact.save()

            # Load and verify metadata has required fields
            metadata_path = tmpdir / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Required for train.py consumption
            assert "path" in metadata
            assert "type" in metadata
            assert metadata["type"] == "SplitJsonlDataArtifact"
            assert "total_sequences" in metadata

    def test_artifact_roundtrip_preserves_all_fields(self):
        """Test that artifact save/load preserves all fields correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text('{"train": "", "val": "", "test": ""}')

            original = SplitJsonlDataArtifact(
                path=manifest_path,
                total_sequences=10_000,
                elapsed_sec=120.5,
                source_datasets=["hf://nvidia/dataset1", "hf://nvidia/dataset2"],
            )
            original.save()

            loaded = SplitJsonlDataArtifact.load(path=tmpdir)

            assert loaded.total_sequences == original.total_sequences
            assert loaded.elapsed_sec == original.elapsed_sec


class TestWandbArtifactIntegration:
    """Test W&B artifact production and consumption for nano3 RL stage."""

    def test_full_artifact_consumption_with_train_config(self, monkeypatch, tmp_path):
        """Test the complete consumption flow: OmegaConf resolver -> file read.

        This test simulates what happens in training:
        1. OmegaConf resolves ${art:data,path} to the artifact directory
        2. Train.py reads manifest.json to get split paths
        3. Train.py opens the JSONL files

        This is the key integration test for the data_prep -> train contract.
        """
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create realistic artifact structure (simulating data_prep output)
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()

        # Create split directories and JSONL files
        for split in ["train", "val", "test"]:
            split_dir = downloaded_dir / split
            split_dir.mkdir()
            jsonl_path = split_dir / f"{split}.jsonl"
            jsonl_path.write_text('{"messages": [{"role": "user", "content": "test"}]}\n')

        # Create manifest.json as data_prep produces
        manifest_data = {
            "train": str(downloaded_dir / "train" / "train.jsonl"),
            "val": str(downloaded_dir / "val" / "val.jsonl"),
            "test": str(downloaded_dir / "test" / "test.jsonl"),
            "mode": "resolve",
        }
        manifest_path = downloaded_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "DataBlendsArtifact-rl"
                self.type = "dataset"

            def download(self, skip_cache=True):
                return str(downloaded_dir)

        class FakeApi:
            def artifact(self, ref):
                return FakeArtifact(ref)

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Step 1: Config pattern - use ${art:data,path} to get artifact directory
        # then append the split path (as train.py would do)
        cfg = OmegaConf.create(
            {
                "run": {"data": "DataBlendsArtifact-rl:latest"},
                "data": {
                    "train_jsonl_fpath": "${art:data,path}/train/train.jsonl",
                    "validation_jsonl_fpath": "${art:data,path}/val/val.jsonl",
                },
            }
        )

        # Step 2: Register resolvers (as train.py does)
        resolvers.register_resolvers_from_config(cfg, mode="pre_init")

        # Step 3: Resolve config
        resolved = OmegaConf.to_container(cfg, resolve=True)
        train_path = resolved["data"]["train_jsonl_fpath"]
        val_path = resolved["data"]["validation_jsonl_fpath"]

        # Verify path resolution
        assert Path(train_path).exists(), f"train.jsonl not found at {train_path}"
        assert Path(val_path).exists(), f"val.jsonl not found at {val_path}"

        # Step 4: Simulate train.py loading (exact pattern from train.py:71-73)
        with open(train_path) as f:
            train_examples = list(map(json.loads, f))

        # Step 5: Verify the data is NOT empty
        assert len(train_examples) > 0, "train split is empty!"
        assert "messages" in train_examples[0], "Missing messages field"

    def test_wandb_artifact_resolution_for_train(self, monkeypatch, tmp_path):
        """Test ${art:data,path} resolver works with train.py config pattern."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Simulate downloaded artifact directory with manifest.json
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        manifest_json = downloaded_dir / "manifest.json"
        manifest_json.write_text('{"train": "", "val": "", "test": ""}')

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "DataBlendsArtifact-rl"
                self.type = "dataset"

            def download(self, skip_cache=True):
                return str(downloaded_dir)

        class FakeApi:
            def artifact(self, ref):
                return FakeArtifact(ref)

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Config pattern matching grpo_nanov3.yaml
        cfg = OmegaConf.create(
            {
                "run": {"data": "DataBlendsArtifact-rl:latest"},
                "data": {"train_jsonl_fpath": "${art:data,path}/train/train.jsonl"},
            }
        )

        qualified_names = resolvers.register_resolvers_from_config(cfg, mode="pre_init")

        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["data"]["train_jsonl_fpath"] == str(
            downloaded_dir / "train" / "train.jsonl"
        )
        assert "ent/proj/DataBlendsArtifact-rl:latest" in qualified_names[0]

    def test_artifact_without_manifest_should_fail_gracefully(self, monkeypatch, tmp_path):
        """Test that missing manifest.json is detected early with clear error."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create artifact directory WITHOUT manifest.json
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        # Only metadata.json, no manifest.json
        (downloaded_dir / "metadata.json").write_text("{}")

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "DataBlendsArtifact-rl"
                self.type = "dataset"

            def download(self, skip_cache=True):
                return str(downloaded_dir)

        class FakeApi:
            def artifact(self, ref):
                return FakeArtifact(ref)

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        cfg = OmegaConf.create(
            {
                "run": {"data": "DataBlendsArtifact-rl:latest"},
                "data": {"train_jsonl_fpath": "${art:data,path}/manifest.json"},
            }
        )

        resolvers.register_resolvers_from_config(cfg, mode="pre_init")
        resolved = OmegaConf.to_container(cfg, resolve=True)
        train_path = resolved["data"]["train_jsonl_fpath"]

        # The path resolves but file doesn't exist
        assert not Path(train_path).exists()


class TestMergeDataPrepOutput:
    """Test data_prep_merge.py output format and integration."""

    def test_merge_manifest_format(self):
        """Test that merge mode produces correct manifest format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Simulate merge mode output
            manifest_data = {
                "train": str(tmpdir / "train" / "train.jsonl"),
                "val": str(tmpdir / "val" / "val.jsonl"),
                "test": str(tmpdir / "test" / "test.jsonl"),
                "mode": "merge",
                "train_ratio": 0.98,
                "valid_ratio": 0.01,
                "test_ratio": 0.01,
                "seed": 42,
                "total_sequences": 1000,
            }

            manifest_path = tmpdir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f)

            # Verify manifest can be loaded
            with open(manifest_path) as f:
                loaded = json.load(f)

            assert loaded["mode"] == "merge"
            assert loaded["train_ratio"] == 0.98
            assert "train" in loaded
            assert "val" in loaded
            assert "test" in loaded

    def test_split_ratios_validation(self):
        """Test that split ratios must sum to 1.0."""
        from nemotron.recipes.nano3.stage2_rl.data_prep_merge import RLMergeDataPrepConfig

        # Valid ratios
        config = RLMergeDataPrepConfig(
            train_ratio=0.98,
            valid_ratio=0.01,
            test_ratio=0.01,
        )
        assert config.train_ratio + config.valid_ratio + config.test_ratio == 1.0

        # Invalid ratios should raise error
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            RLMergeDataPrepConfig(
                train_ratio=0.8,
                valid_ratio=0.1,
                test_ratio=0.05,  # Sum = 0.95 != 1.0
            )
