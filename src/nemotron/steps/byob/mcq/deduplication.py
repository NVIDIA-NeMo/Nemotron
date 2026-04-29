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

from nemotron.steps.byob.deduplication import TextSemanticDeduplication
from nemotron.steps.byob.config import ByobConfig
from nemotron.steps.byob.mcq.formatters import format_qa

class TextSemanticDeduplicationMCQ(TextSemanticDeduplication):
    """Semantic deduplication for MCQ datasets.
    
    Extends the base TextSemanticDeduplication to handle MCQ-specific data formats.
    """
    
    def __init__(self, config: ByobConfig):
        """Initialize MCQ semantic deduplication.
        
        Args:
            config: Configuration object containing deduplication parameters.
        """
        super().__init__(config)

    def prepare_input_data(self, dataset: pd.DataFrame):
        """Prepare MCQ dataset for deduplication.
        
        Formats questions and answers into text for semantic comparison.
        
        Args:
            dataset: DataFrame with columns: id_question, question_generated, 
                    choices_generated, answer_generated.
                    
        Returns:
            pd.DataFrame: DataFrame with columns: id, text.
        """
        dataset = dataset[['id_question', 'question_generated', 'choices_generated', 'answer_generated']].copy()
        dataset['text'] = dataset.apply(
            lambda x: format_qa(x['question_generated'], x['choices_generated'], x['answer_generated'], append_choices=False, append_answer=True), axis=1)
        dataset = dataset[['id_question', 'text']].copy().rename(columns={'id_question': 'id'})
        return dataset