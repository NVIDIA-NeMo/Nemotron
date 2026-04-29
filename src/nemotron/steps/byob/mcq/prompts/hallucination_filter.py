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

"""Prompt templates for hallucination filtering stage.

This module contains prompts for detecting hallucinated content in generated questions.
Multiple LLM models are given both the question and the source text passage and asked
to answer. Questions where most models fail to answer correctly (even with access to
the passage) are flagged as containing hallucinations - i.e., content not actually
supported by the source text. This quality control step helps maintain factual accuracy.
"""

SYSTEM_PROMPT = """You are answering a multiple-choice question with {num_choices} choices.
You will be given a passage on a topic and a question and a list of choices.
Read the passage and the question carefully and choose the correct choice from the list of choices."""

PROMPT = """Here is a passage on the given topic:
Topic: {{{{target_subject}}}}
<start of topic>
{{{{text}}}}
<end of topic>

Answer the following question:

{{{{question_generated_formatted}}}}

The answer should be one of {choices}. Think step by step and then finish your answer with "The answer is (X)" where X is the correct letter choice.
If there is no correct answer, finish your answer with "No correct answer"."""