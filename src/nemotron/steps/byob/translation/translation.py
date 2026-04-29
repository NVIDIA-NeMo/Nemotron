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

from nemotron.steps.byob.translation.llm import LLMTranslationModule
from nemotron.steps.byob.config import ByobTranslationConfig
from nemotron.steps.byob.translation.prompts.translation_prompts import SYSTEM_PROMPT, PROMPT

class TranslationPipeline:
    """Unified translation pipeline supporting multiple backend translation engines.
    
    Provides a consistent interface for translating text using different backends
    Currently supports LLM-based translation.
    """
    
    def __init__(
        self,
        mode: str,
        model_params: dict,
        config: ByobTranslationConfig,
        text_field: str = 'text',
        translation_field: str = 'translation',
        id_field: str = 'translation_id',
        src_lang_field: str = 'source_language_code',
        tgt_lang_field: str = 'target_language_code',
    ):
        """Initialize the translation pipeline.
        
        Args:
            mode: Translation mode ('llm' or 'indictrans2').
            model_params: Dictionary containing model configuration parameters.
            config: Translation configuration object.
            text_field: Name of the column containing source text.
            translation_field: Name of the column to store translations.
            id_field: Name of the column containing unique identifiers.
            src_lang_field: Name of the column containing source language codes.
            tgt_lang_field: Name of the column containing target language codes.
            
        Raises:
            ValueError: If mode is not supported.
        """
        self.mode = mode
        self.model_params = model_params
        self.text_field = text_field
        self.translation_field = translation_field
        self.id_field = id_field
        self.src_lang_field = src_lang_field
        self.tgt_lang_field = tgt_lang_field
        self.config = config

        if mode == 'llm':
            self.module = LLMTranslationModule(
                model_params=model_params,
                config=config,
                text_field=text_field,
                translation_field=translation_field,
                id_field=id_field,
                src_lang_field=src_lang_field,
                tgt_lang_field=tgt_lang_field,
                system_prompt=SYSTEM_PROMPT,
                prompt=PROMPT
            )
        elif mode == 'indictrans2':
            raise NotImplementedError("IndicTrans2TranslationModule is not implemented yet")
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose one of the following: llm, indictrans2")

    def translate(self, dataframe: pd.DataFrame):
        """Translate text in the dataframe using the configured backend.
        
        Args:
            dataframe: Input DataFrame with columns matching those specified in __init__.
                      Must contain text_field, src_lang_field, and tgt_lang_field.
                      
        Returns:
            pd.DataFrame: DataFrame with translations added in translation_field column.
        """
        return self.module.translate(dataframe)