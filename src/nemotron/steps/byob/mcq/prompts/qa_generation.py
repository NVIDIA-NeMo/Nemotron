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

"""Prompt templates for question-answer generation stage.

This module contains prompts for the core MCQ generation task. The LLM is provided
with a target text passage and few-shot examples from source subjects, then asked
to generate multiple-choice questions with 4 choices (A-D) that match the style,
difficulty, and format of the examples. The prompts emphasize:
- Following the reasoning style and difficulty of examples
- Creating questions grounded in the text but not explicitly referencing it
- Avoiding obvious answers to maintain appropriate challenge level
- Supporting multilingual generation through language parameters
"""

SYSTEM_PROMPT = '''You are an expert in creating questions from a description of a given topic.
You will be given {num_few_shot_samples} example questions and answers unrelated to the topic.

You should
1. Create {num_questions} questions, four choices and answers similar to the example question answer.
2. Follow the question style of the example question answers (Direct WH question/Completion/Best explanation/Best action/Equivalence/Other).
3. Try to create questions that are higher in the cognitive level scale, by mostly using concepts from the text passage.
4. The questions should not explicitly refer to the passage or example question answer.
5. Assume that the person reading the questions does not have access to the passage or example question answer. So make the question clear as to what the topic is.
'''

PROMPT = '''Definition of cognitive level (higher is better):
1: Recall
The question asks for recall of isolated facts, definitions, or simple formulas (e.g., “What is the full form of SEBI?”).

2: Understanding
The question checks comprehension of concepts, classifications, or simple explanations (e.g., interpret what an LTV ratio means for a borrower).

3: Application
The question requires using a concept, rule, or formula in a straightforward, familiar situation (e.g., compute post‑tax return on a fixed deposit given basic data).

4: Analysis
The question involves breaking down information, comparing alternatives, or identifying relationships/causes (e.g., infer the impact of an RBI rate change on bond prices or bank NIMs).

5: Evaluate
The question requires judgment among alternatives based on criteria, or synthesizing information to choose the best course of action or construct a plan (e.g., select the most appropriate investment strategy for a given Indian retail investor scenario).


Create {num_questions} questions, four choices and answers for the given topic:
Topic: {{{{target_subject}}}}
<start of topic>
{{{{text}}}}
<end of topic>

Now make {num_questions} questions, four choices and answers for the topic similar to the example question answer. Don't make the answer too obvious.

Example questions and answers:-
{{{{few_shot_examples}}}}

Write the questions and options in the language: {{{{language}}}}
Return the questions in JSON format with each question having the following fields:
- question: The question
- choice_a: The first choice (A)
- choice_b: The second choice (B)
- choice_c: The third choice (C)
- choice_d: The fourth choice (D)
- answer: The answer (A/B/C/D)
'''