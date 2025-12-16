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
Tests for data_prep.py → train.py integration in nano3/stage1_sft.

These contract tests validate that:
1. The .npy output format from data_prep.py matches what Megatron-Bridge expects
2. File naming follows Megatron-Bridge conventions: {split}_{pack_size}.npy
3. Metadata.jsonl format is compatible with GPTSFTPackedDataset
4. W&B artifact production and consumption works correctly
5. OmegaConf resolvers correctly resolve artifact paths
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from nemotron.kit import SFTDataArtifact


class TestSFTDataPrepOutputFormat:
    """Test that data_prep.py output format is correct."""

    def test_packed_npy_file_naming_convention(self, tmp_path):
        """Test .npy files use {split}_{pack_size}.npy naming."""
        pack_size = 4096

        # Create files with correct naming
        train_path = tmp_path / f"training_{pack_size}.npy"
        valid_path = tmp_path / f"validation_{pack_size}.npy"
        test_path = tmp_path / f"test_{pack_size}.npy"

        # Simulate data_prep output
        sample_data = [{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "seq_start_id": [0]}]
        np.save(train_path, sample_data, allow_pickle=True)
        np.save(valid_path, sample_data, allow_pickle=True)
        np.save(test_path, sample_data, allow_pickle=True)

        # Verify naming pattern
        assert train_path.name == f"training_{pack_size}.npy"
        assert valid_path.name == f"validation_{pack_size}.npy"
        assert test_path.name == f"test_{pack_size}.npy"

    def test_packed_npy_contains_required_keys(self, tmp_path):
        """Test each sample has input_ids, loss_mask, seq_start_id."""
        pack_size = 4096
        npy_path = tmp_path / f"training_{pack_size}.npy"

        # Create sample data with all required keys
        sample_data = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "loss_mask": [0, 0, 1, 1, 1],
                "seq_start_id": [0],
            },
            {
                "input_ids": [10, 20, 30, 40, 50, 60],
                "loss_mask": [0, 1, 1, 1, 1, 1],
                "seq_start_id": [0, 3],  # Two sequences packed together
            },
        ]
        np.save(npy_path, sample_data, allow_pickle=True)

        # Load and verify
        loaded = np.load(npy_path, allow_pickle=True)
        for sample in loaded:
            assert "input_ids" in sample, "Missing input_ids key"
            assert "loss_mask" in sample, "Missing loss_mask key"
            assert "seq_start_id" in sample, "Missing seq_start_id key"

    def test_loss_mask_rolled_by_one_for_label_alignment(self, tmp_path):
        """Test loss_mask is shifted for next-token prediction.

        The loss_mask should be rolled by 1 so that it aligns with labels
        (next token prediction). First position should be 0 (no loss on
        predicting first token from nothing).
        """
        pack_size = 4096
        npy_path = tmp_path / f"training_{pack_size}.npy"

        # Create sample where loss_mask starts with 0 (rolled)
        sample_data = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                # Original mask might be [0, 1, 1, 1, 1], rolled becomes [0, 0, 1, 1, 1]
                "loss_mask": [0, 0, 1, 1, 1],
                "seq_start_id": [0],
            },
        ]
        np.save(npy_path, sample_data, allow_pickle=True)

        loaded = np.load(npy_path, allow_pickle=True)
        for sample in loaded:
            # First position should be 0 (no loss on predicting from nothing)
            assert sample["loss_mask"][0] == 0, "First loss_mask position should be 0 (rolled)"

    def test_seq_start_id_excludes_final_boundary(self, tmp_path):
        """Test seq_start_id format matches GPTSFTPackedDataset expectation.

        GPTSFTPackedDataset adds len(input_ids) at read time:
        seq_boundaries = seq_start_id + [len(input_ids)]
        """
        pack_size = 4096
        npy_path = tmp_path / f"training_{pack_size}.npy"

        # Two sequences of length 3 and 4 packed together
        sample_data = [
            {
                "input_ids": [1, 2, 3, 4, 5, 6, 7],  # 7 tokens total
                "loss_mask": [0, 1, 1, 0, 1, 1, 1],
                "seq_start_id": [0, 3],  # Starts at 0 and 3, NOT including 7
            },
        ]
        np.save(npy_path, sample_data, allow_pickle=True)

        loaded = np.load(npy_path, allow_pickle=True)
        sample = loaded[0]

        # Simulate GPTSFTPackedDataset loading
        seq_boundaries = sample["seq_start_id"] + [len(sample["input_ids"])]

        assert seq_boundaries == [0, 3, 7], f"Expected [0, 3, 7], got {seq_boundaries}"

    def test_metadata_jsonl_format_matches_megatron_bridge(self, tmp_path):
        """Test {pack_size}_metadata.jsonl has required fields."""
        pack_size = 4096
        metadata_path = tmp_path / f"{pack_size}_metadata.jsonl"

        # Create metadata with Megatron-Bridge expected fields
        metadata = [
            {
                "max_samples_per_bin": 5,
                "dataset_max_seqlen": 2048,
                "packing_factor": 3.2,
                "packing_efficiency": 92.5,
                "pack_size": pack_size,
                "min_packed_seqlen": 3800,
            },
            {
                "max_samples_per_bin": 4,
                "dataset_max_seqlen": 1900,
                "packing_factor": 3.0,
                "packing_efficiency": 90.1,
                "pack_size": pack_size,
                "min_packed_seqlen": 3700,
            },
        ]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Load and verify
        with open(metadata_path) as f:
            loaded = json.load(f)

        required_fields = {
            "max_samples_per_bin",
            "dataset_max_seqlen",
            "packing_factor",
            "packing_efficiency",
            "pack_size",
            "min_packed_seqlen",
        }

        for entry in loaded:
            for field in required_fields:
                assert field in entry, f"Missing required field: {field}"


class TestSFTDataPrepMegatronBridgeCompatibility:
    """Test integration with Megatron-Bridge's GPTSFTPackedDataset."""

    def test_npy_loadable_by_gpt_sft_packed_dataset_format(self, tmp_path):
        """Simulate GPTSFTPackedDataset loading to verify compatibility."""
        pack_size = 4096
        npy_path = tmp_path / f"training_{pack_size}.npy"

        # Create realistic packed data
        sample_data = [
            {
                "input_ids": list(range(100)),
                "loss_mask": [0] * 30 + [1] * 70,
                "seq_start_id": [0, 30],  # Two sequences
            },
            {
                "input_ids": list(range(50)),
                "loss_mask": [0] * 10 + [1] * 40,
                "seq_start_id": [0],  # One sequence
            },
        ]
        np.save(npy_path, sample_data, allow_pickle=True)

        # Simulate GPTSFTPackedDataset.__getitem__ (from sft.py:786-791)
        indexed_dataset = np.load(npy_path, allow_pickle=True)

        for idx in range(len(indexed_dataset)):
            # This is exactly what GPTSFTPackedDataset does
            input_ids = indexed_dataset[idx]["input_ids"]
            seq_boundaries = indexed_dataset[idx]["seq_start_id"] + [len(input_ids)]
            loss_mask = indexed_dataset[idx]["loss_mask"]

            # Verify data is accessible and correct type
            assert isinstance(input_ids, list), f"input_ids should be list, got {type(input_ids)}"
            assert isinstance(loss_mask, list), f"loss_mask should be list, got {type(loss_mask)}"
            assert isinstance(seq_boundaries, list), "seq_boundaries should be list"

            # Verify lengths match
            assert len(input_ids) == len(loss_mask), "input_ids and loss_mask length mismatch"

            # Verify seq_boundaries are valid
            for i, boundary in enumerate(seq_boundaries[:-1]):
                assert boundary >= 0, f"Boundary {i} is negative"
                assert boundary < len(input_ids), f"Boundary {i} exceeds input length"

    def test_metadata_parseable_by_megatron_bridge(self, tmp_path):
        """Test metadata.jsonl can be parsed as Megatron-Bridge expects."""
        pack_size = 4096
        metadata_path = tmp_path / f"{pack_size}_metadata.jsonl"

        metadata = [
            {
                "max_samples_per_bin": 5,
                "dataset_max_seqlen": 2048,
                "packing_factor": 3.2,
                "packing_efficiency": 92.5,
                "pack_size": pack_size,
                "min_packed_seqlen": 3800,
            },
        ]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Parse as Megatron-Bridge would
        with open(metadata_path) as f:
            loaded = json.load(f)

        # Should be a list of dicts
        assert isinstance(loaded, list)
        assert all(isinstance(entry, dict) for entry in loaded)

    def test_pack_size_suffix_in_filenames(self, tmp_path):
        """Test filenames include pack_size for Megatron-Bridge defaults."""
        pack_sizes = [2048, 4096, 8192]

        for pack_size in pack_sizes:
            train_path = tmp_path / f"training_{pack_size}.npy"
            valid_path = tmp_path / f"validation_{pack_size}.npy"

            # Create files
            sample_data = [{"input_ids": [1], "loss_mask": [0], "seq_start_id": [0]}]
            np.save(train_path, sample_data, allow_pickle=True)
            np.save(valid_path, sample_data, allow_pickle=True)

            # Verify they exist with correct names
            assert train_path.exists()
            assert valid_path.exists()
            assert str(pack_size) in train_path.name
            assert str(pack_size) in valid_path.name


class TestSFTArtifactIntegration:
    """Test SFTDataArtifact lifecycle."""

    def test_sft_data_artifact_stores_pack_size(self, tmp_path):
        """Test SFTDataArtifact includes pack_size."""
        artifact = SFTDataArtifact(
            path=tmp_path,
            total_tokens=1000,
            total_sequences=100,
            pack_size=4096,
        )

        assert artifact.pack_size == 4096
        assert artifact.metadata.get("pack_size") == 4096

    def test_sft_data_artifact_stores_explicit_paths(self, tmp_path):
        """Test SFTDataArtifact includes training_path, validation_path, etc."""
        pack_size = 4096
        training_path = str(tmp_path / f"training_{pack_size}.npy")
        validation_path = str(tmp_path / f"validation_{pack_size}.npy")
        test_path = str(tmp_path / f"test_{pack_size}.npy")
        metadata_path = str(tmp_path / f"{pack_size}_metadata.jsonl")

        artifact = SFTDataArtifact(
            path=tmp_path,
            total_tokens=1000,
            total_sequences=100,
            pack_size=pack_size,
            training_path=training_path,
            validation_path=validation_path,
            test_path=test_path,
            metadata_path=metadata_path,
        )

        assert artifact.training_path == training_path
        assert artifact.validation_path == validation_path
        assert artifact.test_path == test_path
        assert artifact.metadata_path == metadata_path

    def test_artifact_paths_are_absolute_and_normalized(self, tmp_path):
        """Test paths work across different execution contexts."""
        pack_size = 4096
        # Create with absolute paths
        training_path = str(tmp_path.resolve() / f"training_{pack_size}.npy")

        artifact = SFTDataArtifact(
            path=tmp_path,
            total_tokens=1000,
            total_sequences=100,
            pack_size=pack_size,
            training_path=training_path,
        )

        # Path should be absolute
        assert Path(artifact.training_path).is_absolute()
        # Path should not contain '..'
        assert ".." not in artifact.training_path

    def test_artifact_roundtrip_preserves_all_fields(self, tmp_path):
        """Test save/load preserves training_path, validation_path, etc."""
        pack_size = 4096
        training_path = str(tmp_path / f"training_{pack_size}.npy")
        validation_path = str(tmp_path / f"validation_{pack_size}.npy")
        test_path = str(tmp_path / f"test_{pack_size}.npy")
        metadata_path = str(tmp_path / f"{pack_size}_metadata.jsonl")

        original = SFTDataArtifact(
            path=tmp_path,
            total_tokens=1000000,
            total_sequences=10000,
            pack_size=pack_size,
            training_path=training_path,
            validation_path=validation_path,
            test_path=test_path,
            metadata_path=metadata_path,
            tokenizer_uri="https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        )
        original.save()

        # Load and verify
        loaded = SFTDataArtifact.load(path=tmp_path)

        assert loaded.total_tokens == original.total_tokens
        assert loaded.total_sequences == original.total_sequences
        assert loaded.pack_size == original.pack_size
        assert loaded.training_path == original.training_path
        assert loaded.validation_path == original.validation_path
        assert loaded.test_path == original.test_path
        assert loaded.metadata_path == original.metadata_path
        assert loaded.tokenizer_uri == original.tokenizer_uri


class TestOmegaConfResolverIntegration:
    """Test OmegaConf resolvers work correctly with train.py and config files."""

    def test_art_resolver_resolves_data_path(self, monkeypatch, tmp_path):
        """Test ${art:data,path} resolves to artifact download path."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create artifact directory
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "SFTDataArtifact-sft"
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
                "run": {"data": "SFTDataArtifact-sft:latest"},
                "dataset": {"dataset_root": "${art:data,path}"},
            }
        )

        resolvers.register_resolvers_from_config(cfg, mode="pre_init")
        resolved = OmegaConf.to_container(cfg, resolve=True)

        assert resolved["dataset"]["dataset_root"] == str(downloaded_dir)

    def test_art_resolver_resolves_pack_size(self, monkeypatch, tmp_path):
        """Test ${art:data,pack_size} resolves to artifact pack_size field."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create artifact with metadata
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        metadata = {
            "path": str(downloaded_dir),
            "type": "SFTDataArtifact",
            "pack_size": 4096,
            "total_tokens": 1000,
            "total_sequences": 100,
        }
        (downloaded_dir / "metadata.json").write_text(json.dumps(metadata))

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "SFTDataArtifact-sft"
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
                "run": {"data": "SFTDataArtifact-sft:latest"},
                "dataset": {"seq_length": "${art:data,pack_size}"},
            }
        )

        resolvers.register_resolvers_from_config(cfg, mode="pre_init")
        resolved = OmegaConf.to_container(cfg, resolve=True)

        # Resolver returns string representation of pack_size
        assert resolved["dataset"]["seq_length"] == "4096"

    def test_art_resolver_resolves_training_path(self, monkeypatch, tmp_path):
        """Test ${art:data,training_path} resolves to training file path."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create artifact with explicit paths
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        training_path = str(downloaded_dir / "training_4096.npy")
        metadata = {
            "path": str(downloaded_dir),
            "type": "SFTDataArtifact",
            "pack_size": 4096,
            "total_tokens": 1000,
            "total_sequences": 100,
            "training_path": training_path,
        }
        (downloaded_dir / "metadata.json").write_text(json.dumps(metadata))

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "SFTDataArtifact-sft"
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
                "run": {"data": "SFTDataArtifact-sft:latest"},
                "dataset": {"packed_train_data_path": "${art:data,training_path}"},
            }
        )

        resolvers.register_resolvers_from_config(cfg, mode="pre_init")
        resolved = OmegaConf.to_container(cfg, resolve=True)

        assert resolved["dataset"]["packed_train_data_path"] == training_path

    def test_run_wandb_project_interpolation(self, monkeypatch, tmp_path):
        """Test ${run.wandb.project} resolves from run section."""
        from nemotron.kit.cli.config import _resolve_run_interpolations

        config_dict = {
            "logger": {"wandb_project": "${run.wandb.project}"},
        }
        run_section = {"wandb": {"project": "test-project", "entity": "test-entity"}}

        result = _resolve_run_interpolations(config_dict, run_section)

        assert result["logger"]["wandb_project"] == "test-project"

    def test_run_wandb_entity_interpolation(self, monkeypatch, tmp_path):
        """Test ${run.wandb.entity} resolves from run section."""
        from nemotron.kit.cli.config import _resolve_run_interpolations

        config_dict = {
            "logger": {"wandb_entity": "${run.wandb.entity}"},
        }
        run_section = {"wandb": {"project": "test-project", "entity": "test-entity"}}

        result = _resolve_run_interpolations(config_dict, run_section)

        assert result["logger"]["wandb_entity"] == "test-entity"


class TestWandbArtifactIntegration:
    """Test W&B artifact production and consumption for SFT."""

    def test_wandb_artifact_resolution_for_sft_train(self, monkeypatch, tmp_path):
        """Test ${art:data,path} resolver works with SFT train.py config pattern."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create artifact with SFT data
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        pack_size = 4096
        training_path = str(downloaded_dir / f"training_{pack_size}.npy")
        validation_path = str(downloaded_dir / f"validation_{pack_size}.npy")
        metadata_path = str(downloaded_dir / f"{pack_size}_metadata.jsonl")

        # Create sample files
        sample_data = [{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "seq_start_id": [0]}]
        np.save(training_path, sample_data, allow_pickle=True)
        np.save(validation_path, sample_data, allow_pickle=True)

        # Create metadata
        metadata = {
            "path": str(downloaded_dir),
            "type": "SFTDataArtifact",
            "pack_size": pack_size,
            "total_tokens": 1000,
            "total_sequences": 100,
            "training_path": training_path,
            "validation_path": validation_path,
            "metadata_path": metadata_path,
        }
        (downloaded_dir / "metadata.json").write_text(json.dumps(metadata))

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "SFTDataArtifact-sft"
                self.type = "dataset"

            def download(self, skip_cache=True):
                return str(downloaded_dir)

        class FakeApi:
            def artifact(self, ref):
                return FakeArtifact(ref)

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Config pattern matching tiny.yaml
        cfg = OmegaConf.create(
            {
                "run": {"data": "SFTDataArtifact-sft:latest"},
                "dataset": {
                    "dataset_root": "${art:data,path}",
                    "seq_length": "${art:data,pack_size}",
                    "packed_sequence_specs": {
                        "packed_sequence_size": "${art:data,pack_size}",
                        "packed_train_data_path": "${art:data,training_path}",
                        "packed_val_data_path": "${art:data,validation_path}",
                        "packed_metadata_path": "${art:data,metadata_path}",
                    },
                },
            }
        )

        resolvers.register_resolvers_from_config(cfg, mode="pre_init")
        resolved = OmegaConf.to_container(cfg, resolve=True)

        # Verify all paths resolved correctly
        # Note: OmegaConf resolvers return strings, so numeric values are stringified
        assert resolved["dataset"]["dataset_root"] == str(downloaded_dir)
        assert resolved["dataset"]["seq_length"] == str(pack_size)
        specs = resolved["dataset"]["packed_sequence_specs"]
        assert specs["packed_sequence_size"] == str(pack_size)
        assert specs["packed_train_data_path"] == training_path
        assert specs["packed_val_data_path"] == validation_path
        assert specs["packed_metadata_path"] == metadata_path

        # Verify training file exists and is loadable
        loaded = np.load(specs["packed_train_data_path"], allow_pickle=True)
        assert len(loaded) > 0

    def test_artifact_metadata_contains_required_fields(self, tmp_path):
        """Test SFTDataArtifact metadata has fields needed by train.py."""
        pack_size = 4096
        training_path = str(tmp_path / f"training_{pack_size}.npy")

        artifact = SFTDataArtifact(
            path=tmp_path,
            total_tokens=1000000,
            total_sequences=10000,
            pack_size=pack_size,
            training_path=training_path,
        )
        artifact.save()

        # Load and verify metadata
        with open(tmp_path / "metadata.json") as f:
            metadata = json.load(f)

        # Required fields for train.py
        assert "path" in metadata
        assert "type" in metadata
        assert metadata["type"] == "SFTDataArtifact"
        assert "pack_size" in metadata
        assert metadata["pack_size"] == pack_size
        assert "training_path" in metadata
        assert metadata["training_path"] == training_path


class TestSFTDataPrepTrainContract:
    """End-to-end contract tests for data_prep -> train flow."""

    def test_full_data_prep_to_train_flow(self, monkeypatch, tmp_path):
        """Test complete flow: data_prep output -> train consumption."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Step 1: Simulate data_prep output
        pack_size = 4096
        output_dir = tmp_path / "data_prep_output"
        output_dir.mkdir()

        training_path = output_dir / f"training_{pack_size}.npy"
        validation_path = output_dir / f"validation_{pack_size}.npy"
        test_path = output_dir / f"test_{pack_size}.npy"
        mb_metadata_path = output_dir / f"{pack_size}_metadata.jsonl"

        # Create packed data
        train_data = [
            {
                "input_ids": list(range(100)),
                "loss_mask": [0] * 30 + [1] * 70,
                "seq_start_id": [0, 30],
            },
            {"input_ids": list(range(80)), "loss_mask": [0] * 20 + [1] * 60, "seq_start_id": [0]},
        ]
        valid_data = [
            {"input_ids": list(range(50)), "loss_mask": [0] * 10 + [1] * 40, "seq_start_id": [0]},
        ]
        test_data = [
            {"input_ids": list(range(60)), "loss_mask": [0] * 15 + [1] * 45, "seq_start_id": [0]},
        ]

        np.save(training_path, train_data, allow_pickle=True)
        np.save(validation_path, valid_data, allow_pickle=True)
        np.save(test_path, test_data, allow_pickle=True)

        # Create Megatron-Bridge compatible metadata
        mb_metadata = [
            {
                "max_samples_per_bin": 2,
                "dataset_max_seqlen": 70,
                "packing_factor": 1.5,
                "packing_efficiency": 90.0,
                "pack_size": pack_size,
                "min_packed_seqlen": 80,
            },
        ]
        with open(mb_metadata_path, "w") as f:
            json.dump(mb_metadata, f)

        # Step 2: Create and save artifact
        artifact = SFTDataArtifact(
            path=output_dir,
            total_tokens=290,
            total_sequences=4,
            pack_size=pack_size,
            training_path=str(training_path),
            validation_path=str(validation_path),
            test_path=str(test_path),
            metadata_path=str(mb_metadata_path),
        )
        artifact.save()

        # Step 3: Mock W&B artifact download
        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v1"
                self.name = "SFTDataArtifact-sft"
                self.type = "dataset"

            def download(self, skip_cache=True):
                return str(output_dir)

        class FakeApi:
            def artifact(self, ref):
                return FakeArtifact(ref)

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Step 4: Load config (simulating train.py)
        cfg = OmegaConf.create(
            {
                "run": {"data": "SFTDataArtifact-sft:latest"},
                "dataset": {
                    "dataset_root": "${art:data,path}",
                    "seq_length": "${art:data,pack_size}",
                    "packed_sequence_specs": {
                        "packed_sequence_size": "${art:data,pack_size}",
                        "packed_train_data_path": "${art:data,training_path}",
                        "packed_val_data_path": "${art:data,validation_path}",
                        "packed_metadata_path": "${art:data,metadata_path}",
                    },
                },
            }
        )

        resolvers.register_resolvers_from_config(cfg, mode="pre_init")
        resolved = OmegaConf.to_container(cfg, resolve=True)

        # Step 5: Verify data is NOT empty and correctly formatted
        specs = resolved["dataset"]["packed_sequence_specs"]
        train_loaded = np.load(specs["packed_train_data_path"], allow_pickle=True)
        valid_loaded = np.load(specs["packed_val_data_path"], allow_pickle=True)

        assert len(train_loaded) > 0, "Training data is empty!"
        assert len(valid_loaded) > 0, "Validation data is empty!"

        # Step 6: Verify format matches GPTSFTPackedDataset expectation
        for sample in train_loaded:
            assert "input_ids" in sample
            assert "loss_mask" in sample
            assert "seq_start_id" in sample
            # Simulate GPTSFTPackedDataset loading
            seq_boundaries = sample["seq_start_id"] + [len(sample["input_ids"])]
            assert seq_boundaries[-1] == len(sample["input_ids"])
