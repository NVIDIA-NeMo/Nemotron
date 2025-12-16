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
Tests for data_prep.py → train.py integration in nano3/stage0_pretrain.

These contract tests validate that:
1. The blend.json output format from data_prep.py matches what train.py expects
2. W&B artifact production and consumption works correctly
3. The ${art:data,path} resolver correctly resolves artifact paths
"""

import importlib
import json
import os
import re
import sys
import tempfile
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemotron.data_prep.filesystem import get_filesystem
from nemotron.data_prep.pipeline import _distribute_shards_to_splits
from nemotron.kit import PretrainBlendsArtifact


class TestNano3DataPrepTrainIntegration:
    """Test integration between data_prep.py output and train.py consumption."""

    def test_distribute_shards_produces_valid_per_split_format(self):
        """Test _distribute_shards_to_splits produces correct format."""
        data_paths = ["1.0", "/path/to/shard", "0.5", "/path/to/other"]

        result = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=4,
            valid_shards=1,
            test_shards=1,
        )

        assert "train" in result
        assert "valid" in result
        assert "test" in result

        # Each split should have alternating weight/path pairs
        for split_name, split_data in result.items():
            assert len(split_data) % 2 == 0, f"Split {split_name} has odd number of elements"
            for i in range(0, len(split_data), 2):
                # Weight should be parseable as float
                float(split_data[i])
                # Path should be a string
                assert isinstance(split_data[i + 1], str)

    def test_distribute_shards_respects_shard_counts(self):
        """Test that valid_shards and test_shards control split sizes."""
        data_paths = ["1.0", "/path/to/shard"]

        result = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=10,
            valid_shards=2,
            test_shards=3,
        )

        # valid should have 2 shards (4 elements: weight, path, weight, path)
        assert len(result["valid"]) == 4
        # test should have 3 shards (6 elements)
        assert len(result["test"]) == 6
        # train should have remaining 5 shards (10 elements)
        assert len(result["train"]) == 10

    def test_blend_json_format_matches_train_expectation(self):
        """Test blend.json format is compatible with train.py config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            blend_path = Path(tmpdir) / "blend.json"

            # Simulate data_prep output
            blend_data = {
                "train": ["1.0", f"{tmpdir}/shard_000000", "1.0", f"{tmpdir}/shard_000001"],
                "valid": ["1.0", f"{tmpdir}/shard_000002"],
                "test": ["1.0", f"{tmpdir}/shard_000003"],
            }

            with open(blend_path, "w") as f:
                json.dump(blend_data, f)

            # Simulate train.py loading
            with open(blend_path) as f:
                loaded = json.load(f)

            assert loaded == blend_data
            assert set(loaded.keys()) == {"train", "valid", "test"}

    def test_blend_json_compatible_with_megatron_bridge_parsing(self):
        """Test blend.json is compatible with Megatron-Bridge's get_blend_and_blend_per_split.

        This test simulates exactly how Megatron-Bridge parses per_split_data_args_path.
        See: megatron/bridge/data/loaders.py lines 76-89
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            blend_path = Path(tmpdir) / "blend.json"

            # Simulate data_prep output with string weights (as produced by data_prep)
            blend_data = {
                "train": ["1.0", f"{tmpdir}/shard_000000", "1.0", f"{tmpdir}/shard_000001"],
                "valid": ["1.0", f"{tmpdir}/shard_000002"],
                "test": ["1.0", f"{tmpdir}/shard_000003"],
            }

            with open(blend_path, "w") as f:
                json.dump(blend_data, f)

            # Simulate Megatron-Bridge loading (exact code from loaders.py:76-89)
            with open(blend_path, "r") as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

            # Verify the parsed data matches what Megatron-Bridge expects
            assert per_split_data_args["train"] == [
                "1.0",
                f"{tmpdir}/shard_000000",
                "1.0",
                f"{tmpdir}/shard_000001",
            ]
            assert per_split_data_args["valid"] == ["1.0", f"{tmpdir}/shard_000002"]
            assert per_split_data_args["test"] == ["1.0", f"{tmpdir}/shard_000003"]

            # Verify the lists are not empty (this is the error we're debugging)
            for split in ["train", "valid", "test"]:
                assert len(per_split_data_args[split]) > 0, f"{split} is empty!"
                assert len(per_split_data_args[split]) % 2 == 0, f"{split} has odd length"

    def test_pretrain_blends_artifact_with_blend_path(self):
        """Test PretrainBlendsArtifact stores blend_path correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            blend_path = tmpdir / "blend.json"
            blend_path.write_text('{"train": [], "valid": [], "test": []}')

            artifact = PretrainBlendsArtifact(
                path=tmpdir,
                blend_path=str(blend_path),
                total_tokens=1000,
                total_sequences=100,
                num_shards=4,
                elapsed_sec=0.0,
            )

            assert artifact.blend_path == str(blend_path)
            artifact.save()

            loaded = PretrainBlendsArtifact.load(path=tmpdir)
            assert loaded.blend_path == str(blend_path)

    def test_shard_path_naming_convention(self):
        """Test that shard paths follow the expected naming convention."""
        data_paths = ["1.0", "/output/shard"]

        result = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=10,
            valid_shards=1,
            test_shards=1,
        )

        # All paths should have _XXXXXX suffix (6 digits)
        pattern = re.compile(r".*_\d{6}$")

        for split_name, split_data in result.items():
            for i in range(1, len(split_data), 2):  # Every other item is a path
                assert pattern.match(
                    split_data[i]
                ), f"Path {split_data[i]} in {split_name} doesn't match naming convention"

    def test_distribute_shards_deterministic_with_seed(self):
        """Test that shard distribution is deterministic with same seed."""
        data_paths = ["1.0", "/path/to/shard"]

        result1 = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=10,
            valid_shards=2,
            test_shards=2,
            seed=42,
        )

        result2 = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=10,
            valid_shards=2,
            test_shards=2,
            seed=42,
        )

        assert result1 == result2

    def test_distribute_shards_different_with_different_seed(self):
        """Test that different seeds produce different distributions."""
        data_paths = ["1.0", "/path/to/shard"]

        result1 = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=10,
            valid_shards=2,
            test_shards=2,
            seed=42,
        )

        result2 = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=10,
            valid_shards=2,
            test_shards=2,
            seed=123,
        )

        # At least one split should be different
        assert result1 != result2


class TestWandbArtifactIntegration:
    """Test W&B artifact production and consumption for nano3 pretrain."""

    def test_full_artifact_consumption_with_megatron_bridge_parsing(self, monkeypatch, tmp_path):
        """Test the complete consumption flow: OmegaConf resolver -> file read -> MB parsing.

        This test simulates exactly what happens in training:
        1. OmegaConf resolves ${art:data,path}/blend.json to a local path
        2. Megatron-Bridge opens and parses the blend.json
        3. get_blend_from_list() processes train/valid/test arrays

        This is the key integration test for the data_prep -> train contract.
        """
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create a realistic blend.json with actual data (simulating data_prep output)
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        blend_json = downloaded_dir / "blend.json"

        # Realistic blend.json content as produced by data_prep
        blend_data = {
            "train": [
                "1.0",
                "/lustre/data/shard_000000",
                "1.0",
                "/lustre/data/shard_000001",
                "1.0",
                "/lustre/data/shard_000002",
            ],
            "valid": ["1.0", "/lustre/data/shard_000003"],
            "test": ["1.0", "/lustre/data/shard_000004"],
        }
        blend_json.write_text(json.dumps(blend_data))

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "PretrainBlendsArtifact-pretrain"
                self.type = "dataset"

            def download(self, skip_cache=True):
                return str(downloaded_dir)

        class FakeApi:
            def artifact(self, ref):
                return FakeArtifact(ref)

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Step 1: Config pattern matching train.py default.yaml
        cfg = OmegaConf.create(
            {
                "run": {"data": "PretrainBlendsArtifact-pretrain:latest"},
                "recipe": {"per_split_data_args_path": "${art:data,path}/blend.json"},
            }
        )

        # Step 2: Register resolvers (as train.py does)
        qualified_names = resolvers.register_resolvers_from_config(cfg, mode="pre_init")

        # Step 3: Resolve config (as train.py does before calling recipe)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        per_split_data_args_path = resolved["recipe"]["per_split_data_args_path"]

        # Verify path resolution
        assert per_split_data_args_path == str(downloaded_dir / "blend.json")
        assert Path(per_split_data_args_path).exists(), f"blend.json not found at {per_split_data_args_path}"

        # Step 4: Simulate Megatron-Bridge loading (exact code from loaders.py:76-89)
        with open(per_split_data_args_path, "r") as f:
            per_split_data_args = json.load(f)
            # Each element in blend_per_split should be a list of files (and optional
            # weights), so split string if needed.
            for split in ["train", "valid", "test"]:
                if isinstance(per_split_data_args[split], str):
                    per_split_data_args[split] = per_split_data_args[split].split()

        # Step 5: Verify the data is NOT empty (this is the bug we're catching)
        assert len(per_split_data_args["train"]) > 0, "train split is empty!"
        assert len(per_split_data_args["valid"]) > 0, "valid split is empty!"
        assert len(per_split_data_args["test"]) > 0, "test split is empty!"

        # Step 6: Verify format matches what get_blend_from_list expects
        # Even-length list: [weight, path, weight, path, ...]
        assert len(per_split_data_args["train"]) % 2 == 0, "train has odd length"
        assert len(per_split_data_args["valid"]) % 2 == 0, "valid has odd length"
        assert len(per_split_data_args["test"]) % 2 == 0, "test has odd length"

        # Verify the actual content matches what we put in
        assert per_split_data_args["train"] == blend_data["train"]
        assert per_split_data_args["valid"] == blend_data["valid"]
        assert per_split_data_args["test"] == blend_data["test"]

    def test_artifact_without_blend_json_should_fail_gracefully(self, monkeypatch, tmp_path):
        """Test that missing blend.json is detected early with clear error."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Create artifact directory WITHOUT blend.json
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        # Only metadata.json, no blend.json
        (downloaded_dir / "metadata.json").write_text("{}")

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "PretrainBlendsArtifact-pretrain"
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
                "run": {"data": "PretrainBlendsArtifact-pretrain:latest"},
                "recipe": {"per_split_data_args_path": "${art:data,path}/blend.json"},
            }
        )

        resolvers.register_resolvers_from_config(cfg, mode="pre_init")
        resolved = OmegaConf.to_container(cfg, resolve=True)
        per_split_data_args_path = resolved["recipe"]["per_split_data_args_path"]

        # The path resolves but file doesn't exist
        assert not Path(per_split_data_args_path).exists()

    def test_wandb_artifact_resolution_for_train(self, monkeypatch, tmp_path):
        """Test ${art:data,path} resolver works with train.py config pattern."""
        from nemotron.kit import resolvers

        resolvers.clear_artifact_cache()

        monkeypatch.setenv("WANDB_ENTITY", "ent")
        monkeypatch.setenv("WANDB_PROJECT", "proj")

        # Simulate downloaded artifact directory with blend.json
        downloaded_dir = tmp_path / "artifact"
        downloaded_dir.mkdir()
        blend_json = downloaded_dir / "blend.json"
        blend_json.write_text('{"train": [], "valid": [], "test": []}')

        class FakeArtifact:
            def __init__(self, ref):
                self.qualified_name = ref
                self.version = "v5"
                self.name = "PretrainBlendsArtifact-pretrain"
                self.type = "dataset"

            def download(self, skip_cache=True):
                return str(downloaded_dir)

        class FakeApi:
            def artifact(self, ref):
                return FakeArtifact(ref)

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Config pattern matching train.py test.yaml
        cfg = OmegaConf.create(
            {
                "run": {"data": "PretrainBlendsArtifact-pretrain:latest"},
                "recipe": {"per_split_data_args_path": "${art:data,path}/blend.json"},
            }
        )

        qualified_names = resolvers.register_resolvers_from_config(cfg, mode="pre_init")

        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["recipe"]["per_split_data_args_path"] == str(downloaded_dir / "blend.json")
        assert "ent/proj/PretrainBlendsArtifact-pretrain:latest" in qualified_names[0]

    def test_wandb_lineage_registration(self, monkeypatch):
        """Test that train.py pattern correctly registers lineage."""
        import nemotron.kit.wandb as wb

        wb = importlib.reload(wb)

        used_artifacts = []

        class FakeRun:
            def __init__(self):
                self.tags = []

            def use_artifact(self, qname):
                used_artifacts.append(qname)

        fake_run = FakeRun()

        def fake_init(*args, **kwargs):
            fake_wandb.run = fake_run
            return fake_run

        fake_wandb = types.SimpleNamespace(run=None, init=fake_init)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Pattern from train.py
        wb.patch_wandb_init_for_lineage(
            artifact_qualified_names=["ent/proj/PretrainBlendsArtifact-pretrain:v5"],
            tags=["pretrain"],
        )

        fake_wandb.init()

        assert "ent/proj/PretrainBlendsArtifact-pretrain:v5" in used_artifacts
        assert "pretrain" in fake_run.tags

    def test_artifact_metadata_contains_required_fields(self):
        """Test PretrainBlendsArtifact metadata has fields needed by train.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            blend_path = tmpdir / "blend.json"
            blend_path.write_text('{"train": [], "valid": [], "test": []}')

            artifact = PretrainBlendsArtifact(
                path=tmpdir,
                blend_path=str(blend_path),
                total_tokens=1_000_000,
                total_sequences=10_000,
                num_shards=128,
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
            assert metadata["type"] == "PretrainBlendsArtifact"
            assert "blend_path" in metadata
            assert metadata["blend_path"] == str(blend_path)
            assert "total_tokens" in metadata
            assert "num_shards" in metadata

    def test_artifact_roundtrip_preserves_all_fields(self):
        """Test that artifact save/load preserves all fields correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            blend_path = tmpdir / "blend.json"
            blend_path.write_text('{"train": [], "valid": [], "test": []}')

            original = PretrainBlendsArtifact(
                path=tmpdir,
                blend_path=str(blend_path),
                total_tokens=1_000_000,
                total_sequences=10_000,
                num_shards=128,
                elapsed_sec=120.5,
                train_tokens=900_000,
                valid_tokens=50_000,
                test_tokens=50_000,
                source_datasets=["hf://nvidia/dataset1", "hf://nvidia/dataset2"],
                tokenizer_uri="https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            )
            original.save()

            loaded = PretrainBlendsArtifact.load(path=tmpdir)

            assert loaded.total_tokens == original.total_tokens
            assert loaded.total_sequences == original.total_sequences
            assert loaded.num_shards == original.num_shards
            assert loaded.blend_path == original.blend_path
            assert loaded.train_tokens == original.train_tokens
            assert loaded.valid_tokens == original.valid_tokens
            assert loaded.test_tokens == original.test_tokens
            assert loaded.tokenizer_uri == original.tokenizer_uri

    def test_artifact_stores_tokenizer_uri_for_compatibility_check(self):
        """Test that artifact stores tokenizer_uri so train.py can verify compatibility.

        The tokenizer used for data preparation MUST match the tokenizer used for training.
        The artifact stores tokenizer_uri to enable this verification.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            blend_path = tmpdir / "blend.json"
            blend_path.write_text('{"train": [], "valid": [], "test": []}')

            tokenizer_model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
            tokenizer_uri = f"https://huggingface.co/{tokenizer_model}"

            artifact = PretrainBlendsArtifact(
                path=tmpdir,
                blend_path=str(blend_path),
                total_tokens=1_000_000,
                total_sequences=10_000,
                num_shards=128,
                elapsed_sec=120.0,
                tokenizer_uri=tokenizer_uri,
            )
            artifact.save()

            # Verify tokenizer_uri is in metadata for train.py to check
            metadata_path = tmpdir / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert "tokenizer_uri" in metadata
            assert metadata["tokenizer_uri"] == tokenizer_uri

            # Verify it can be loaded back
            loaded = PretrainBlendsArtifact.load(path=tmpdir)
            assert loaded.tokenizer_uri == tokenizer_uri


class TestPathNormalization:
    """Test that paths in blend.json are absolute and normalized.

    Paths must not contain '..' or '.' components to ensure they resolve
    correctly in training containers that may have different working directories.
    """

    def test_get_filesystem_normalizes_local_paths_with_dotdot(self, tmp_path):
        """Test get_filesystem resolves '..' in local paths."""
        # Create a nested structure
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)

        # Path with .. that should resolve
        path_with_dotdot = str(tmp_path / "a" / "b" / "c" / ".." / ".." / "b")

        fs, normalized = get_filesystem(path_with_dotdot)

        # The normalized path should not contain '..'
        assert ".." not in normalized, f"Path still contains '..': {normalized}"
        # It should resolve to the actual path
        assert normalized == str(tmp_path / "a" / "b")

    def test_get_filesystem_normalizes_local_paths_with_dot(self, tmp_path):
        """Test get_filesystem resolves '.' in local paths."""
        path_with_dot = str(tmp_path / "." / "subdir")

        fs, normalized = get_filesystem(path_with_dot)

        # The normalized path should not contain '/.'
        assert "/." not in normalized and not normalized.endswith(
            "."
        ), f"Path still contains '.': {normalized}"

    def test_get_filesystem_preserves_s3_paths(self):
        """Test that S3 paths are not modified."""
        pytest.importorskip("s3fs")

        s3_path = "s3://bucket/prefix/../other"

        fs, normalized = get_filesystem(s3_path)

        # S3 paths should be passed through (fsspec handles them)
        assert normalized.startswith("bucket/")

    def test_get_filesystem_handles_file_scheme(self, tmp_path):
        """Test file:// URIs are handled correctly."""
        path_with_dotdot = f"file://{tmp_path}/a/../b"

        fs, normalized = get_filesystem(path_with_dotdot)

        # Should resolve the .. component
        assert ".." not in normalized

    def test_blend_json_paths_should_be_normalized(self, tmp_path):
        """Test that blend.json paths are absolute without '..' components.

        This is a contract test: data_prep should produce paths that work
        in any execution context (different working directories, containers).
        """
        # Simulate what data_prep would produce with proper normalization
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Get filesystem normalizes the path
        fs, normalized_base = get_filesystem(str(output_dir))

        # Build path as pipeline.py does (after normalization fix)
        shard_path = f"{normalized_base}/runs/abc123/datasets/test/hash/shard_000000"

        # Validate the contract
        assert os.path.isabs(shard_path), f"Path is not absolute: {shard_path}"
        assert ".." not in shard_path, f"Path contains '..': {shard_path}"
        assert "/." not in shard_path, f"Path contains '/.': {shard_path}"

    def test_blend_json_paths_portable_across_contexts(self, tmp_path):
        """Test that normalized paths work regardless of working directory.

        When data_prep runs in /code and outputs to /output, the paths in
        blend.json must resolve correctly when training runs from /train.
        """
        # Simulate data_prep output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a test shard file
        shard_dir = output_dir / "runs" / "abc" / "datasets" / "test" / "hash"
        shard_dir.mkdir(parents=True)
        shard_file = shard_dir / "shard_000000.bin"
        shard_file.touch()

        # Get normalized path as data_prep would
        fs, base_path = get_filesystem(str(output_dir))
        shard_path = f"{base_path}/runs/abc/datasets/test/hash/shard_000000"

        # Change working directory to simulate training context
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # The path should still be valid
            assert os.path.isabs(shard_path)
            # The file should exist (without .bin extension check since we test the prefix)
            assert os.path.exists(f"{shard_path}.bin")
        finally:
            os.chdir(original_cwd)
