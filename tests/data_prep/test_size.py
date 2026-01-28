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

"""Tests for size parsing and formatting utilities."""

import pytest

from nemotron.data_prep.utils.size import (
    compute_num_shards,
    format_byte_size,
    format_count,
    parse_byte_size,
)


# =============================================================================
# parse_byte_size
# =============================================================================


class TestParseByteSize:
    def test_integer_passthrough(self) -> None:
        assert parse_byte_size(1024) == 1024

    def test_float_passthrough(self) -> None:
        assert parse_byte_size(1024.5) == 1024

    def test_bytes(self) -> None:
        assert parse_byte_size("100B") == 100

    def test_kilobytes(self) -> None:
        assert parse_byte_size("1KB") == 1024

    def test_kilobytes_no_b(self) -> None:
        assert parse_byte_size("1K") == 1024

    def test_megabytes(self) -> None:
        assert parse_byte_size("256MB") == 256 * 1024**2

    def test_megabytes_lowercase(self) -> None:
        assert parse_byte_size("256mb") == 256 * 1024**2

    def test_gigabytes(self) -> None:
        assert parse_byte_size("1GB") == 1024**3

    def test_gigabytes_no_b(self) -> None:
        assert parse_byte_size("1G") == 1024**3

    def test_terabytes(self) -> None:
        assert parse_byte_size("2TB") == 2 * 1024**4

    def test_mebibytes(self) -> None:
        assert parse_byte_size("256MiB") == 256 * 1024**2

    def test_fractional(self) -> None:
        assert parse_byte_size("1.5GB") == int(1.5 * 1024**3)

    def test_bare_number(self) -> None:
        assert parse_byte_size("1024") == 1024

    def test_with_spaces(self) -> None:
        assert parse_byte_size("  256MB  ") == 256 * 1024**2

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid size format"):
            parse_byte_size("not_a_size")


# =============================================================================
# format_byte_size
# =============================================================================


class TestFormatByteSize:
    def test_bytes(self) -> None:
        assert format_byte_size(100) == "100B"

    def test_kilobytes(self) -> None:
        assert format_byte_size(1024) == "1KB"

    def test_megabytes(self) -> None:
        assert format_byte_size(256 * 1024**2) == "256MB"

    def test_gigabytes(self) -> None:
        assert format_byte_size(1024**3) == "1GB"

    def test_fractional_gb(self) -> None:
        result = format_byte_size(int(1.5 * 1024**3))
        assert result == "1.5GB"

    def test_negative(self) -> None:
        result = format_byte_size(-1024)
        assert result == "-1KB"

    def test_zero(self) -> None:
        assert format_byte_size(0) == "0B"


# =============================================================================
# format_count
# =============================================================================


class TestFormatCount:
    def test_small_number(self) -> None:
        assert format_count(500) == "500"

    def test_thousands(self) -> None:
        assert format_count(1500) == "1.5K"

    def test_millions(self) -> None:
        assert format_count(1500000) == "1.5M"

    def test_billions(self) -> None:
        assert format_count(2300000000) == "2.3B"

    def test_exact_thousand(self) -> None:
        assert format_count(1000) == "1K"

    def test_negative(self) -> None:
        result = format_count(-1500)
        assert result == "-1.5K"

    def test_zero(self) -> None:
        assert format_count(0) == "0"


# =============================================================================
# compute_num_shards
# =============================================================================


class TestComputeNumShards:
    def test_exact_division(self) -> None:
        # 1GB / 256MB = 4
        assert compute_num_shards(1024**3, "256MB") == 4

    def test_ceiling_division(self) -> None:
        # 1GB + 1 byte / 256MB → still 5 shards
        assert compute_num_shards(1024**3 + 1, "256MB") == 5

    def test_minimum_one_shard(self) -> None:
        assert compute_num_shards(100, "256MB") == 1

    def test_integer_shard_size(self) -> None:
        assert compute_num_shards(1000, 500) == 2

    def test_zero_total_bytes(self) -> None:
        assert compute_num_shards(0, "256MB") == 1
