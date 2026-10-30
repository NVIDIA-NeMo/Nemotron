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

from .distractor_expansion import (
    PROMPT as PROMPT_DISTRACTOR_EXPANSION,
)
from .distractor_expansion import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_DISTRACTOR_EXPANSION,
)
from .distractor_validity import (
    PROMPT as PROMPT_DISTRACTOR_VALIDITY,
)
from .distractor_validity import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_DISTRACTOR_VALIDITY,
)
from .easiness_filter import PROMPT as PROMPT_EASINESS_FILTER
from .easiness_filter import SYSTEM_PROMPT as SYSTEM_PROMPT_EASINESS_FILTER
from .hallucination_filter import (
    PROMPT as PROMPT_HALLUCINATION_FILTER,
)
from .hallucination_filter import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_HALLUCINATION_FILTER,
)
from .qa_generation import PROMPT as PROMPT_QA
from .qa_generation import SYSTEM_PROMPT as SYSTEM_PROMPT_QA
from .question_judge import PROMPT as PROMPT_JUDGE
from .question_judge import SYSTEM_PROMPT as SYSTEM_PROMPT_JUDGE


def get_prompts():
    """Get all default prompt templates for MCQ generation pipeline.

    Returns:
        dict: Dictionary containing system prompts and user prompts for each stage:
              - qa_generation: Question and answer generation
              - question_judge: Question quality assessment
              - hallucination_filter: Hallucination detection
              - easiness_filter: Question difficulty assessment
              - distractor_expansion: Expanding choices from 4 to 10
              - distractor_validity: Validating correctness of distractors
    """
    return {
        "qa_generation": {
            "system_prompt": SYSTEM_PROMPT_QA,
            "prompt": PROMPT_QA,
        },
        "question_judge": {
            "system_prompt": SYSTEM_PROMPT_JUDGE,
            "prompt": PROMPT_JUDGE,
        },
        "hallucination_filter": {
            "system_prompt": SYSTEM_PROMPT_HALLUCINATION_FILTER,
            "prompt": PROMPT_HALLUCINATION_FILTER,
        },
        "easiness_filter": {
            "system_prompt": SYSTEM_PROMPT_EASINESS_FILTER,
            "prompt": PROMPT_EASINESS_FILTER,
        },
        "distractor_expansion": {
            "system_prompt": SYSTEM_PROMPT_DISTRACTOR_EXPANSION,
            "prompt": PROMPT_DISTRACTOR_EXPANSION,
        },
        "distractor_validity": {
            "system_prompt": SYSTEM_PROMPT_DISTRACTOR_VALIDITY,
            "prompt": PROMPT_DISTRACTOR_VALIDITY,
        },
    }
