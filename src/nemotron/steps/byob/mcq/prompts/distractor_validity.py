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

"""Prompt templates for distractor validity checking stage.

This module contains prompts for validating that distractor (incorrect) choices are
actually incorrect based on the source text passage. The LLM verifies each choice
to identify any that might actually be correct, which would make the question invalid
due to multiple correct answers. This quality control step helps ensure question integrity.
"""

SYSTEM_PROMPT = '''You are an expert in text comprehension. You will be given a text passage on a topic and a question.
The question claims that there is only one correct answer among the {num_choices} choices.

You should
1. Look at the question and the given answer.
2. Look at the options and see if they are same as the correct answer.
3. Check if any of the options that are not the correct answer are actually correct based on the text passage.

Example:
"True, False" is the same as "True, Not True"
'''

PROMPT = '''Here is a text passage on the given topic:
Topic: {{target_subject}}
<start of topic>
{{text}}
<end of topic>

Here is the question and the options:
{{question_answer_generated_formatted}}

Now mark the options that are same as the correct answer as "Yes" and the options that are not same as the correct answer as "No".
Provide your answer in JSON format.

- choice_a: Whether the first choice (A) is same as the correct answer (Yes/No)
- choice_b: Whether the second choice (B) is same as the correct answer (Yes/No)
- choice_c: Whether the third choice (C) is same as the correct answer (Yes/No)
- choice_d: Whether the fourth choice (D) is same as the correct answer (Yes/No)
{{"- choice_e: Whether the fifth choice (E) is same as the correct answer (Yes/No)" if num_choices == 10 else ""}}
{{"- choice_f: Whether the sixth choice (F) is same as the correct answer (Yes/No)" if num_choices == 10 else ""}}
{{"- choice_g: Whether the seventh choice (G) is same as the correct answer (Yes/No)" if num_choices == 10 else ""}}
{{"- choice_h: Whether the eighth choice (H) is same as the correct answer (Yes/No)" if num_choices == 10 else ""}}
{{"- choice_i: Whether the ninth choice (I) is same as the correct answer (Yes/No)" if num_choices == 10 else ""}}
{{"- choice_j: Whether the tenth choice (J) is same as the correct answer (Yes/No)" if num_choices == 10 else ""}}

Remember to mark the original correct answer as "Yes". Now mark which of the given options are same as the correct answer.
'''