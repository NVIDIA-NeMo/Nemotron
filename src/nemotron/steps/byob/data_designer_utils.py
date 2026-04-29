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
from typing import Callable
from tqdm import tqdm

import data_designer.config as dd

from nemotron.steps.byob.config import ByobConfig

def setup_model_config(model_config: dict):
    """Convert a model configuration dictionary to DataDesigner ModelConfig object.
    
    Args:
        model_config: Dictionary with keys: alias, model, provider, inference_parameters.
        
    Returns:
        ModelConfig: DataDesigner ModelConfig object.
    """
    return dd.ModelConfig(
        alias=model_config['alias'],
        model=model_config['model'],
        provider=model_config['provider'],
        inference_parameters=dd.ChatCompletionInferenceParams(**model_config['inference_parameters'])
    )

def batched_run(func: Callable, config: ByobConfig, seed_df: pd.DataFrame, batch_size: int):
    """Execute a function on a DataFrame in batches and concatenate results.
    
    Splits the input DataFrame into batches and processes each batch sequentially
    with progress tracking. Useful for processing large datasets and show a progress bar.
    
    Args:
        func: Function to execute on each batch. Should accept (config, batch_df) 
              and return a DataFrame.
        config: BYOB configuration object to pass to func.
        seed_df: Input DataFrame to process in batches.
        batch_size: Number of rows per batch.
        
    Returns:
        pd.DataFrame: Concatenated results from all batches.
        
    Raises:
        AssertionError: If no output data was generated.
    """
    df_out_list = []
    for idx in tqdm(range(0, len(seed_df), batch_size), desc=f"Running {func.__name__}"):
        df_batch = seed_df.iloc[idx:idx+batch_size].copy()
        df_out = func(config, df_batch)
        df_out_list.append(df_out)
    assert len(df_out_list) > 0, f"No output data was generated for {func.__name__}"
    return pd.concat(df_out_list)