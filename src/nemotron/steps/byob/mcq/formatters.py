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


import pandas as pd

def format_qa(question: str, choices: list[str], answer: str | None = None, append_choices: bool = True, append_answer: bool = False, append_choice_name=True) -> str:
    """Format a question and answer into a standardized text format.
    
    Args:
        question: The question text.
        choices: List of answer choices.
        answer: The correct answer letter (e.g., 'A', 'B', 'C', 'D'), or None.
        append_choices: Whether to include the options in the formatted output.
        append_answer: Whether to include the answer text in the formatted output.
        append_choice_name: Whether to include the answer letter in the formatted output.
        
    Returns:
        str: Formatted question-answer text.
        
    Raises:
        ValueError: If answer index is out of range for the provided choices.
    """
    text = f"Question: {question}"

    if append_choices:
        choices_flat = '\n'.join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
        text += f'\nOptions:\n{choices_flat}'

    if answer is not None:
        assert append_choice_name or append_answer, "Either append_choice_name or append_answer must be True when answer is not None"
        text += '\nAnswer:'
        if append_choice_name:
            text += f': {answer}'
        if append_answer:
            index = ord(answer) - ord('A')
            if index < 0 or index >= len(choices):
                raise ValueError(f"Answer index {index} is out of range for choices {choices}")
            text += f': {choices[index]}'

    return text

def format_qa_batch(df: pd.DataFrame) -> str:
    """Format a batch of questions and answers from a DataFrame.
    
    Args:
        df: DataFrame with columns: question, choices, answer.
        
    Returns:
        str: Formatted question-answers separated by double newlines.
    """
    questions = df['question'].tolist()
    choices = df['choices'].tolist()
    answers = df['answer'].tolist()

    formatted_qas = []
    for question, choices, answer in zip(questions, choices, answers):
        formatted_qa = format_qa(question, choices, chr(ord('A') + int(answer)))
        formatted_qas.append(formatted_qa)
    formatted_qas = '\n\n'.join(formatted_qas)
    return formatted_qas