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


import os
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner

from nemotron.steps.byob.data_designer_utils import setup_model_config, batched_run
from nemotron.steps.byob.translation.prompts.translation_prompts import SYSTEM_PROMPT, PROMPT
from nemotron.steps.byob.config import ByobTranslationConfig

class Translation(BaseModel):
    """Pydantic model for translation output."""
    translation: str = Field(description="The translated text")

class LLMTranslationModule:
    """Translation module using Large Language Models (LLMs).
    
    Handles translation of text using DataDesigner with configurable LLM backends.
    Supports batch processing and custom prompts.
    """
    
    def __init__(
        self,
        model_params: dict,
        config: ByobTranslationConfig,
        text_field: str,
        translation_field: str,
        id_field: str,
        src_lang_field: str,
        tgt_lang_field: str,
        system_prompt: str = SYSTEM_PROMPT,
        prompt: str = PROMPT
    ):
        """Initialize the LLM translation module.
        
        Args:
            model_params: Dictionary containing model configuration parameters.
            config: Translation configuration object.
            text_field: Name of the column containing source text.
            translation_field: Name of the column to store translations.
            id_field: Name of the column containing unique identifiers.
            src_lang_field: Name of the column containing source language codes.
            tgt_lang_field: Name of the column containing target language codes.
            system_prompt: System prompt template for the LLM.
            prompt: User prompt template for the LLM.
        """
        self.model_params = model_params
        self.config = config
        self.model_config = setup_model_config(model_params)
        self.text_field = text_field
        self.translation_field = translation_field
        self.id_field = id_field
        self.src_lang_field = src_lang_field
        self.tgt_lang_field = tgt_lang_field
        self.system_prompt = system_prompt
        self.prompt = prompt

    def _translate(self, config: ByobTranslationConfig, seed_df: pd.DataFrame):
        """Internal method to translate a batch of text using DataDesigner.
        
        Args:
            config: Translation configuration object.
            seed_df: DataFrame containing text to translate with source and target language info.
            
        Returns:
            pd.DataFrame: DataFrame with translations added in the translation_field column.
        """
        seed_dataset = dd.DataFrameSeedSource(df=seed_df)

        data_designer = DataDesigner(artifact_path=f'{config.output_dir}/{config.expt_name}/artifacts/data_designer')
        config_builder = dd.DataDesignerConfigBuilder(model_configs=[self.model_config])
        config_builder.with_seed_dataset(seed_dataset)
        config_builder.add_column(
            name='result',
            column_type='llm-structured',
            system_prompt=self.system_prompt,
            prompt=self.prompt,
            output_format=Translation,
            model_alias=self.model_config.alias
        )
        config_builder.add_column(
            name=self.translation_field,
            column_type='expression',
            expr='{{result.translation}}'
        )
        job_results = data_designer.create(config_builder=config_builder, num_records=len(seed_df))
        dataset = job_results.load_dataset()
        dataset.dropna(inplace=True)
        return dataset

    def translate(self, dataframe: pd.DataFrame):
        """Translate text in the dataframe using batched LLM calls.
        
        Splits the dataframe into batches and processes them in parallel for efficiency.
        
        Args:
            dataframe: Input DataFrame with columns specified in __init__.
            
        Returns:
            pd.DataFrame: DataFrame with translations added.
        """
        return batched_run(self._translate, config=self.config, seed_df=dataframe, batch_size=self.config.ndd_batch_size)