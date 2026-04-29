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

"""Prompt templates for easiness filtering stage.

This module contains prompts for determining if a question is too easy. Multiple LLM
models are asked to answer the question without access to the source text passage.
Questions where most models answer correctly are flagged as too easy, as they can be
answered using general knowledge rather than requiring comprehension of the specific text.
This helps maintain appropriate difficulty levels in the generated question set.
"""

SYSTEM_PROMPT = """You are answering a multiple-choice question with {num_choices} choices.
You will be given a question and a list of choices.
Answer the question and choose the correct choice from the list of choices."""

PROMPT = """Answer the following question:

{{{{question_generated_formatted}}}}

The answer should be one of {choices}. Think step by step and then finish your answer with "The answer is (X)" where X is the correct letter choice."""