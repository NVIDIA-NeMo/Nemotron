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

"""Prompt templates for distractor expansion stage.

This module contains prompts for expanding multiple-choice questions from 4 choices
(A, B, C, D) to 10 choices (A-J) by generating 6 additional plausible but incorrect
distractor options. The LLM is instructed to create distractors that are plausible
enough to be challenging but definitively incorrect.
"""

SYSTEM_PROMPT = '''You are an expert in text comprehension. You will be given a text passage on a topic and a question with four choices and one answer.

You should
1. Look at the question and the original options.
2. The original options are A, B, C, D.
3. Generate six additional plausible but incorrect options (E, F, G, H, I, J).
4. The new options should be plausible but incorrect.
'''

PROMPT = '''Here is a text passage on the given topic:
Topic: {{target_subject}}
<start of topic>
{{text}}
<end of topic>

Here is the question and the original options:
{{question_answer_generated_formatted}}

Now generate six additional plausible but incorrect options for the question E, F, G, H, I, J.
Write the options in the language: {{language}}.
Provide your answer in JSON format.

- choice_e: The fifth choice (E)
- choice_f: The sixth choice (F)
- choice_g: The seventh choice (G)
- choice_h: The eighth choice (H)
- choice_i: The ninth choice (I)
- choice_j: The tenth choice (J)
'''