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


from typing import Literal
from pydantic import BaseModel, Field

class QuestionAnswerFourChoices(BaseModel):
    question: str = Field(description="A question related to the topic")
    choice_a: str = Field(description="The first choice (A)")
    choice_b: str = Field(description="The second choice (B)")
    choice_c: str = Field(description="The third choice (C)")
    choice_d: str = Field(description="The fourth choice (D)")
    answer: Literal['A', 'B', 'C', 'D'] = Field(description="The answer to the question (A/B/C/D)")

class QuestionAnswerList(BaseModel):
    questions: list[QuestionAnswerFourChoices] = Field(description="A list of questions and answers")

class Choice4(BaseModel):
    answer: Literal['A', 'B', 'C', 'D'] = Field(description="The answer to the question (A/B/C/D)")

class JudgeResult(BaseModel):
    reason: str = Field(description="Reason for the judgment")
    is_valid: bool = Field(description="Whether the question is valid")
    category: Literal['knowledge', 'reasoning', 'both'] = Field(description="Category of the question (knowledge/reasoning/both)")

class Choice10(BaseModel):
    answer: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] = Field(description="The answer to the question (A/B/C/D/E/F/G/H/I/J)")

class DistractorExpansion(BaseModel):
    choice_e: str = Field(description="The fifth choice (E)")
    choice_f: str = Field(description="The sixth choice (F)")
    choice_g: str = Field(description="The seventh choice (G)")
    choice_h: str = Field(description="The eighth choice (H)")
    choice_i: str = Field(description="The ninth choice (I)")
    choice_j: str = Field(description="The tenth choice (J)")

class DistractorValidityFourChoices(BaseModel):
    choice_a: Literal['Yes', 'No'] = Field(description="Whether the first choice (A) is correct (Yes/No)")
    choice_b: Literal['Yes', 'No'] = Field(description="Whether the second choice (B) is correct (Yes/No)")
    choice_c: Literal['Yes', 'No'] = Field(description="Whether the third choice (C) is correct (Yes/No)")
    choice_d: Literal['Yes', 'No'] = Field(description="Whether the fourth choice (D) is correct (Yes/No)")

class DistractorValidityTenChoices(BaseModel):
    choice_a: Literal['Yes', 'No'] = Field(description="Whether the first choice (A) is correct (Yes/No)")
    choice_b: Literal['Yes', 'No'] = Field(description="Whether the second choice (B) is correct (Yes/No)")
    choice_c: Literal['Yes', 'No'] = Field(description="Whether the third choice (C) is correct (Yes/No)")
    choice_d: Literal['Yes', 'No'] = Field(description="Whether the fourth choice (D) is correct (Yes/No)")
    choice_e: Literal['Yes', 'No'] = Field(description="Whether the fifth choice (E) is correct (Yes/No)")
    choice_f: Literal['Yes', 'No'] = Field(description="Whether the sixth choice (F) is correct (Yes/No)")
    choice_g: Literal['Yes', 'No'] = Field(description="Whether the seventh choice (G) is correct (Yes/No)")
    choice_h: Literal['Yes', 'No'] = Field(description="Whether the eighth choice (H) is correct (Yes/No)")
    choice_i: Literal['Yes', 'No'] = Field(description="Whether the ninth choice (I) is correct (Yes/No)")
    choice_j: Literal['Yes', 'No'] = Field(description="Whether the tenth choice (J) is correct (Yes/No)")