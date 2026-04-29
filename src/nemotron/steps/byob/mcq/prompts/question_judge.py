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

"""Prompt templates for question quality judgement stage.

This module contains prompts for evaluating generated question quality. The LLM judge
assesses questions on multiple criteria:
- Validity: Whether the question is clear and properly formed
- Reference style: Whether it inappropriately refers to "the text passage" in ways
  that assume the reader has the passage in front of them
- Category: Whether the question tests "knowledge" (memorization), "reasoning"
  (critical thinking/analysis), or "both"

The judgement provides a reason, validity flag, and category classification that can
be used to filter or organize the generated question set.
"""

SYSTEM_PROMPT = '''You are an expert in judging the quality of a question.
The question was created from a text passage on a topic.

1. Check if the question refers to the text passage. (Phrases like "According to the text passage", "From the text passage", "From the given text", etc.)
2. Check if the question is clear.
3. What is valid:
    - Referring to the text passage and being clear on the topic. Example:- "According to the Bill of Rights, what is the right to freedom of speech?"
    - Not referring to the text passage and being clear. Example:- "What is the capital of France?"
4. What is invalid:
    - Referring to the text passage as if it is given. Example:- "According to the text passage, what is the right to freedom of speech?, From the text passage, what is the right to freedom of speech?"
    - Not being clear. Example:- "What is its capital?"
5. Identify the category of the question:
    - "knowledge" questions rely on memorizing details without critical thinking, mathematical reasoning, and/or logical analysis. Example:- "What is the capital of France?"
    - "reasoning" questions require critical thinking, mathematical reasoning, and/or logical analysis. Example:- "If you have 10 apples and you give 2 to your friend, how many apples do you have left?"
    - "both" questions rely on both memorizing details and critical thinking, mathematical reasoning, and/or logical analysis. Example:- "According to the Bill of Rights, what is the right to freedom of speech?"
6. Provide a reason for your judgment.
'''

PROMPT = '''Here is the question:
Question: {{question}}

Return the question in JSON format with the following fields:
- reason: Reason for your judgment
- is_valid: Whether the question is valid
- category: Category of the question (knowledge/reasoning/both)
'''