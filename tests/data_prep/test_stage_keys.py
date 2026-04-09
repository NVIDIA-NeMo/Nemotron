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

"""Unit tests for stage_keys module."""

from __future__ import annotations

import pytest

from nemotron.data_prep.stage_keys import (
    STAGE_DISPLAY_NAMES,
    canonical_stage_id,
    get_stage_display_name,
)


class TestCanonicalStageId:
    """Tests for canonical_stage_id function."""

    def test_plan_stage(self) -> None:
        """PlanStage becomes 'plan'."""
        assert canonical_stage_id("PlanStage") == "plan"

    def test_download_stage(self) -> None:
        """DownloadStage becomes 'download'."""
        assert canonical_stage_id("DownloadStage") == "download"

    def test_bin_idx_tokenization_stage(self) -> None:
        """BinIdxTokenizationStage becomes 'bin_idx_tokenization'."""
        assert canonical_stage_id("BinIdxTokenizationStage") == "bin_idx_tokenization"

    def test_packed_sft_parquet_stage(self) -> None:
        """PackedSftParquetStage becomes 'packed_sft_parquet'."""
        assert canonical_stage_id("PackedSftParquetStage") == "packed_sft_parquet"

    def test_sft_plan_stage(self) -> None:
        """SftPlanStage becomes 'sft_plan'."""
        assert canonical_stage_id("SftPlanStage") == "sft_plan"

    def test_prefixed_plan_stage(self) -> None:
        """'Stage 00 - PlanStage' becomes 'plan'."""
        assert canonical_stage_id("Stage 00 - PlanStage") == "plan"

    def test_prefixed_download_stage(self) -> None:
        """'Stage 01 - DownloadStage' becomes 'download'."""
        assert canonical_stage_id("Stage 01 - DownloadStage") == "download"

    def test_prefixed_tokenization_stage(self) -> None:
        """'Stage 02 - BinIdxTokenizationStage' becomes 'bin_idx_tokenization'."""
        assert canonical_stage_id("Stage 02 - BinIdxTokenizationStage") == "bin_idx_tokenization"

    def test_empty_string(self) -> None:
        """Empty string returns 'unknown'."""
        assert canonical_stage_id("") == "unknown"

    def test_none_value(self) -> None:
        """None value returns 'unknown'."""
        assert canonical_stage_id(None) == "unknown"  # type: ignore[arg-type]

    def test_whitespace_only(self) -> None:
        """Whitespace-only returns 'unknown'."""
        assert canonical_stage_id("   ") == "unknown"

    def test_simple_name_no_suffix(self) -> None:
        """Simple name without 'Stage' suffix works."""
        assert canonical_stage_id("Download") == "download"
        assert canonical_stage_id("Plan") == "plan"

    def test_camel_case_conversion(self) -> None:
        """CamelCase is converted to snake_case."""
        assert canonical_stage_id("MyCustomProcessor") == "my_custom_processor"

    def test_special_characters(self) -> None:
        """Special characters are replaced with underscores."""
        result = canonical_stage_id("My@Special#Stage!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_multiple_underscores_collapsed(self) -> None:
        """Multiple consecutive underscores are collapsed."""
        result = canonical_stage_id("My___Multiple___Underscores")
        assert "___" not in result

    def test_stage_suffix_only_at_end(self) -> None:
        """Only 'Stage' at the end is removed, not at the beginning."""
        result = canonical_stage_id("StageProcessor")
        # "Stage" at the beginning is preserved (only suffix is removed)
        assert result == "stage_processor"

    def test_various_prefixes(self) -> None:
        """Various Stage NN prefix formats are handled."""
        assert canonical_stage_id("Stage 0 - Test") == "test"
        assert canonical_stage_id("Stage 10 - Test") == "test"
        assert canonical_stage_id("Stage  5  -  Test") == "test"  # Extra spaces


class TestGetStageDisplayName:
    """Tests for get_stage_display_name function."""

    def test_known_stage(self) -> None:
        """Known stages return their display names."""
        assert get_stage_display_name("plan") == "Plan"
        assert get_stage_display_name("download") == "Download"
        assert get_stage_display_name("bin_idx_tokenization") == "Tokenize"

    def test_unknown_stage(self) -> None:
        """Unknown stages get title-cased display names."""
        assert get_stage_display_name("my_custom_stage") == "My Custom Stage"

    def test_sft_stages(self) -> None:
        """SFT-specific stages have display names."""
        assert get_stage_display_name("sft_plan") == "SFT Plan"
        assert get_stage_display_name("packed_sft_parquet") == "Pack SFT"


class TestStageDisplayNames:
    """Tests for STAGE_DISPLAY_NAMES constant."""

    def test_contains_common_stages(self) -> None:
        """Common stages are in the display names dict."""
        assert "plan" in STAGE_DISPLAY_NAMES
        assert "download" in STAGE_DISPLAY_NAMES
        assert "bin_idx_tokenization" in STAGE_DISPLAY_NAMES

    def test_display_names_are_readable(self) -> None:
        """Display names are human-readable."""
        for name in STAGE_DISPLAY_NAMES.values():
            assert len(name) > 0
            assert not name.startswith("_")
            assert not name.endswith("_")
