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

from nemotron.steps.byob.config import ByobConfig
from nemotron.steps.byob.text_coverage import TextCoverage
from nemotron.steps.byob.mcq.dataset import McqByobDataset
from nemotron.steps.byob.mcq.formatters import format_qa

class TextCoverageMCQ(TextCoverage):
    """Text coverage analysis for MCQ datasets.
    
    Extends the base TextCoverage to analyze how well generated questions cover
    the source text passages.
    """
    
    def __init__(self, config: ByobConfig):
        """Initialize MCQ text coverage analysis.
        
        Args:
            config: Configuration object containing coverage analysis parameters.
        """
        super().__init__(config)

    def prepare_input_data(self, dataset: pd.DataFrame):
        """Prepare MCQ dataset for text coverage analysis.
        
        Extracts text passages and formats questions for coverage computation.
        
        Args:
            dataset: DataFrame with columns: document_id, text_path, id_question,
                    question_generated, choices_generated, answer_generated.
                    
        Returns:
            tuple: (dataset_text, dataset_queries) where dataset_text contains
                  document texts and dataset_queries contains formatted questions.
        """
        dataset_text = dataset[['document_id', 'text_path']].copy().drop_duplicates().reset_index(drop=True)
        dataset_queries = dataset[['document_id', 'id_question', 'question_generated', 'choices_generated', 'answer_generated']].copy().reset_index(drop=True)
        dataset_text['text'] = dataset_text['text_path'].apply(lambda x: McqByobDataset.extract_text_from_path(x)[0])
        dataset_text.drop(columns=['text_path'], inplace=True)
        dataset_queries['query'] = dataset_queries[['question_generated', 'choices_generated', 'answer_generated']].apply(lambda x: format_qa(x['question_generated'], x['choices_generated'], x['answer_generated'], append_choices=False, append_answer=True), axis=1)
        dataset_queries.drop(columns=['question_generated', 'choices_generated', 'answer_generated'], inplace=True)
        return dataset_text, dataset_queries