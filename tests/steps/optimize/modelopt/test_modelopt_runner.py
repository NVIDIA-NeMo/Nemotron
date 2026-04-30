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

"""Unit checks for ModelOpt CLI argument translation."""

from nemotron.steps._runners.modelopt import to_cli_args


def test_hyphen_flag_translation_with_lists_and_bools() -> None:
    args = to_cli_args(
        {
            "hf_model_id": "model",
            "trust_remote_code": True,
            "compress": False,
            "data_paths": [1.0, "prefix"],
            "extra_args": ["--new-flag", "x"],
        },
        forwarded_fields=("hf_model_id", "trust_remote_code", "compress", "data_paths"),
        flag_style="hyphen",
    )

    assert args == [
        "--hf-model-id",
        "model",
        "--trust-remote-code",
        "--data-paths",
        "1.0",
        "prefix",
        "--new-flag",
        "x",
    ]


def test_underscore_flag_translation_with_dict_json() -> None:
    args = to_cli_args(
        {"prune_export_config": {"hidden_size": 3584}},
        forwarded_fields=("prune_export_config",),
        flag_style="underscore",
    )

    assert args == ["--prune_export_config", '{"hidden_size": 3584}']
