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

"""Tests for nemotron.kit.cli.utils module."""

import pytest

from nemotron.kit.cli.utils import (
    CONFIG_FILE_KEYS,
    extract_run_args,
    filter_config_file_args,
    resolve_run_interpolations,
    rewrite_paths_for_remote,
)


class TestResolveRunInterpolations:
    """Tests for resolve_run_interpolations function."""

    def test_resolve_simple_string(self):
        """Test resolving a simple ${run.*} interpolation."""
        run_data = {"wandb": {"project": "my-project"}}
        result = resolve_run_interpolations("${run.wandb.project}", run_data)
        assert result == "my-project"

    def test_resolve_nested_path(self):
        """Test resolving deeply nested paths."""
        run_data = {"a": {"b": {"c": {"d": "value"}}}}
        result = resolve_run_interpolations("${run.a.b.c.d}", run_data)
        assert result == "value"

    def test_resolve_in_dict(self):
        """Test resolving interpolations in a dict."""
        run_data = {"wandb": {"project": "my-project", "entity": "my-team"}}
        obj = {
            "project": "${run.wandb.project}",
            "entity": "${run.wandb.entity}",
            "static": "unchanged",
        }
        result = resolve_run_interpolations(obj, run_data)
        assert result == {
            "project": "my-project",
            "entity": "my-team",
            "static": "unchanged",
        }

    def test_resolve_in_list(self):
        """Test resolving interpolations in a list."""
        run_data = {"values": {"a": "1", "b": "2"}}
        obj = ["${run.values.a}", "${run.values.b}", "static"]
        result = resolve_run_interpolations(obj, run_data)
        assert result == ["1", "2", "static"]

    def test_resolve_in_nested_structure(self):
        """Test resolving interpolations in nested dict/list structures."""
        run_data = {"config": {"name": "test"}}
        obj = {"outer": [{"inner": "${run.config.name}"}]}
        result = resolve_run_interpolations(obj, run_data)
        assert result == {"outer": [{"inner": "test"}]}

    def test_preserve_non_run_interpolations(self):
        """Test that non-${run.*} interpolations are preserved."""
        run_data = {"wandb": {"project": "my-project"}}
        # ${art:data,path} should be preserved
        result = resolve_run_interpolations("${art:data,path}", run_data)
        assert result == "${art:data,path}"

    def test_preserve_unresolvable_paths(self):
        """Test that unresolvable paths are preserved."""
        run_data = {"wandb": {"project": "my-project"}}
        # nonexistent.path doesn't exist in run_data
        result = resolve_run_interpolations("${run.nonexistent.path}", run_data)
        assert result == "${run.nonexistent.path}"

    def test_preserve_scalars(self):
        """Test that scalar values are preserved."""
        run_data = {}
        assert resolve_run_interpolations(42, run_data) == 42
        assert resolve_run_interpolations(3.14, run_data) == 3.14
        assert resolve_run_interpolations(True, run_data) is True
        assert resolve_run_interpolations(None, run_data) is None

    def test_preserve_non_interpolation_strings(self):
        """Test that regular strings are preserved."""
        run_data = {"key": "value"}
        assert resolve_run_interpolations("regular string", run_data) == "regular string"
        assert resolve_run_interpolations("${incomplete", run_data) == "${incomplete"


class TestRewritePathsForRemote:
    """Tests for rewrite_paths_for_remote function."""

    def test_rewrite_pwd_env_var(self):
        """Test rewriting ${oc.env:PWD}/path."""
        result = rewrite_paths_for_remote("${oc.env:PWD}/src/main.py", "/local/repo")
        assert result == "/nemo_run/code/src/main.py"

    def test_rewrite_nemo_run_dir_simple(self):
        """Test rewriting ${oc.env:NEMO_RUN_DIR}/path."""
        result = rewrite_paths_for_remote("${oc.env:NEMO_RUN_DIR}/output", "/local/repo")
        assert result == "/nemo_run/output"

    def test_rewrite_nemo_run_dir_with_default(self):
        """Test rewriting ${oc.env:NEMO_RUN_DIR,.}/path."""
        result = rewrite_paths_for_remote("${oc.env:NEMO_RUN_DIR,.}/logs", "/local/repo")
        assert result == "/nemo_run/logs"

    def test_rewrite_absolute_path_under_repo(self):
        """Test rewriting absolute paths under repo root."""
        result = rewrite_paths_for_remote("/local/repo/src/config.yaml", "/local/repo")
        assert result == "/nemo_run/code/src/config.yaml"

    def test_preserve_paths_outside_repo(self):
        """Test that paths outside repo root are preserved."""
        result = rewrite_paths_for_remote("/other/path/file.txt", "/local/repo")
        assert result == "/other/path/file.txt"

    def test_rewrite_in_dict(self):
        """Test rewriting paths in a dict."""
        obj = {
            "config": "${oc.env:PWD}/config.yaml",
            "output": "${oc.env:NEMO_RUN_DIR}/output",
            "static": "/unchanged",
        }
        result = rewrite_paths_for_remote(obj, "/local/repo")
        assert result == {
            "config": "/nemo_run/code/config.yaml",
            "output": "/nemo_run/output",
            "static": "/unchanged",
        }

    def test_rewrite_in_list(self):
        """Test rewriting paths in a list."""
        obj = ["${oc.env:PWD}/a", "${oc.env:PWD}/b"]
        result = rewrite_paths_for_remote(obj, "/local/repo")
        assert result == ["/nemo_run/code/a", "/nemo_run/code/b"]

    def test_preserve_scalars(self):
        """Test that scalar values are preserved."""
        assert rewrite_paths_for_remote(42, "/local/repo") == 42
        assert rewrite_paths_for_remote(None, "/local/repo") is None


class TestExtractRunArgs:
    """Tests for extract_run_args function."""

    def test_no_run_args(self):
        """Test with no --run or --batch arguments."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["--config", "test.yaml", "--batch-size", "32"]
        )
        assert profile is None
        assert overrides == {}
        assert remaining == ["--config", "test.yaml", "--batch-size", "32"]
        assert is_launch is False

    def test_run_profile_long_form(self):
        """Test --run <profile> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["--run", "slurm", "--batch-size", "32"]
        )
        assert profile == "slurm"
        assert overrides == {}
        assert remaining == ["--batch-size", "32"]
        assert is_launch is False

    def test_run_profile_short_form(self):
        """Test -r <profile> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["-r", "slurm", "--batch-size", "32"]
        )
        assert profile == "slurm"
        assert remaining == ["--batch-size", "32"]
        assert is_launch is False

    def test_run_profile_equals_form(self):
        """Test --run=<profile> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["--run=slurm", "--batch-size", "32"]
        )
        assert profile == "slurm"
        assert remaining == ["--batch-size", "32"]

    def test_run_profile_short_equals_form(self):
        """Test -r=<profile> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["-r=slurm", "--batch-size", "32"]
        )
        assert profile == "slurm"
        assert remaining == ["--batch-size", "32"]

    def test_run_overrides(self):
        """Test --run.<key> <value> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["--run", "slurm", "--run.partition", "batch", "--run.nodes", "4"]
        )
        assert profile == "slurm"
        assert overrides == {"partition": "batch", "nodes": "4"}
        assert remaining == []

    def test_run_overrides_equals_form(self):
        """Test --run.<key>=<value> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["--run", "slurm", "--run.partition=batch"]
        )
        assert profile == "slurm"
        assert overrides == {"partition": "batch"}

    def test_batch_profile(self):
        """Test --batch <profile> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["--batch", "slurm", "--batch-size", "32"]
        )
        assert profile == "slurm"
        assert overrides == {}
        assert remaining == ["--batch-size", "32"]
        assert is_launch is True

    def test_batch_short_form(self):
        """Test -b <profile> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["-b", "slurm", "--batch-size", "32"]
        )
        assert profile == "slurm"
        assert is_launch is True

    def test_batch_overrides(self):
        """Test --batch.<key> <value> form."""
        profile, overrides, remaining, is_launch = extract_run_args(
            ["--batch", "slurm", "--batch.partition", "backfill"]
        )
        assert profile == "slurm"
        assert overrides == {"partition": "backfill"}
        assert is_launch is True

    def test_run_and_batch_mutually_exclusive(self):
        """Test that --run and --batch cannot be used together."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            extract_run_args(["--run", "slurm", "--batch", "local"])

    def test_run_requires_profile(self):
        """Test that --run requires a profile name."""
        with pytest.raises(ValueError, match="requires a profile name"):
            extract_run_args(["--run"])

    def test_run_requires_profile_not_flag(self):
        """Test that --run doesn't accept a flag as profile."""
        with pytest.raises(ValueError, match="requires a profile name"):
            extract_run_args(["--run", "--other-flag"])

    def test_batch_requires_profile(self):
        """Test that --batch requires a profile name."""
        with pytest.raises(ValueError, match="requires a profile name"):
            extract_run_args(["--batch"])

    def test_run_override_requires_value(self):
        """Test that --run.<key> requires a value."""
        with pytest.raises(ValueError, match="requires a value"):
            extract_run_args(["--run", "slurm", "--run.partition"])


class TestFilterConfigFileArgs:
    """Tests for filter_config_file_args function."""

    def test_filter_config_file_space_form(self):
        """Test filtering --config-file <path>."""
        result = filter_config_file_args(
            ["--config-file", "config.yaml", "--batch-size", "32"]
        )
        assert result == ["--batch-size", "32"]

    def test_filter_config_file_equals_form(self):
        """Test filtering --config-file=<path>."""
        result = filter_config_file_args(
            ["--config-file=config.yaml", "--batch-size", "32"]
        )
        assert result == ["--batch-size", "32"]

    def test_filter_config_underscore_form(self):
        """Test filtering --config_file <path>."""
        result = filter_config_file_args(
            ["--config_file", "config.yaml", "--batch-size", "32"]
        )
        assert result == ["--batch-size", "32"]

    def test_filter_config_short_form(self):
        """Test filtering --config <path>."""
        result = filter_config_file_args(["--config", "config.yaml", "--batch-size", "32"])
        assert result == ["--batch-size", "32"]

    def test_filter_c_short_form(self):
        """Test filtering -c <path>."""
        result = filter_config_file_args(["-c", "config.yaml", "--batch-size", "32"])
        assert result == ["--batch-size", "32"]

    def test_filter_c_equals_form(self):
        """Test filtering -c=<path>."""
        result = filter_config_file_args(["-c=config.yaml", "--batch-size", "32"])
        assert result == ["--batch-size", "32"]

    def test_no_config_file_args(self):
        """Test with no config file arguments."""
        result = filter_config_file_args(["--batch-size", "32", "--learning-rate", "0.001"])
        assert result == ["--batch-size", "32", "--learning-rate", "0.001"]

    def test_multiple_config_file_args(self):
        """Test filtering multiple config file arguments."""
        result = filter_config_file_args(
            ["--config", "a.yaml", "-c", "b.yaml", "--batch-size", "32"]
        )
        assert result == ["--batch-size", "32"]


class TestConfigFileKeys:
    """Tests for CONFIG_FILE_KEYS constant."""

    def test_contains_expected_keys(self):
        """Test that CONFIG_FILE_KEYS contains all expected variants."""
        assert "--config-file" in CONFIG_FILE_KEYS
        assert "--config_file" in CONFIG_FILE_KEYS
        assert "--config" in CONFIG_FILE_KEYS
        assert "-c" in CONFIG_FILE_KEYS
