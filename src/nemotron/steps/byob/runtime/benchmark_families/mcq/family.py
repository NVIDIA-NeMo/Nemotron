# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCQ benchmark-family registration."""

from __future__ import annotations

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkFamilySpec
from nemotron.steps.byob.runtime.benchmark_families.mcq.pipeline import (
    generate_mcq,
    prepare_mcq_data,
    translate_mcq,
)

SPEC = BenchmarkFamilySpec(
    name="mcq",
    description="MMLU-Pro-style multiple-choice benchmark generation and translation.",
    prepare_data=prepare_mcq_data,
    generate=generate_mcq,
    translate=translate_mcq,
)
