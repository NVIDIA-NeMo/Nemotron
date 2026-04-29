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


import requests
import os
import datasets
import logging

logger = logging.getLogger(__name__)

def get_metadata(dataset_name):
    """Fetch metadata for a HuggingFace dataset from the datasets server API.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., 'cais/mmlu').
        
    Returns:
        dict: JSON response containing dataset metadata including splits.
        
    Raises:
        requests.HTTPError: If API request fails.
    """
    headers = {'Authorization': f'Bearer {os.getenv('HF_TOKEN')}'}
    api_url = f'https://datasets-server.huggingface.co/splits?dataset={dataset_name}'

    response = requests.get(api_url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()

def get_splits(dataset_name):
    """Get available splits for a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier.
        
    Returns:
        list: List of split metadata dictionaries.
    """
    metadata = get_metadata(dataset_name)
    return metadata['splits']

def get_subsets(dataset_name):
    """Get available subsets/configs for a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier.
        
    Returns:
        list: Sorted list of subset names.
    """
    splits = get_splits(dataset_name)
    return sorted(list(set([row['config'] for row in splits])))

def get_subjects(dataset_name, subset, split):
    """Get available subjects/categories for a HuggingFace benchmark dataset.
    
    Different datasets organize their subjects differently. This function provides
    a unified interface to extract subject lists from various MCQ benchmarks.
    
    Args:
        dataset_name: HuggingFace dataset identifier.
        subset: Dataset subset/config name.
        split: Dataset split (e.g., 'test', 'train').
        
    Returns:
        list: Sorted list of subject names.
        
    Raises:
        ValueError: If dataset is not supported.
    """
    if dataset_name == 'cais/mmlu':
        subsets = [s for s in get_subsets(dataset_name) if s not in {'all', 'auxiliary_train'}]
        return sorted(subsets)
    elif dataset_name == 'TIGER-Lab/MMLU-Pro':
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
        return sorted(dataset.to_pandas()['category'].unique().tolist())
    elif dataset_name == 'ai4bharat/MILU':
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
        subjects = dataset.to_pandas()['subject'].unique().tolist()
        return sorted(subjects)
    elif dataset_name in ['CohereLabs/Global-MMLU', 'CohereLabs/Global-MMLU-Lite']:
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
        subjects = dataset.to_pandas()['subject'].unique().tolist()
        return sorted(subjects)
    elif dataset_name == 'LinguaLift/IndicMMLU-Pro':
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
        subjects = dataset.to_pandas()['category'].unique().tolist()
        return sorted(subjects)
    elif dataset_name == 'openai/MMMLU':
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
        subjects = dataset.to_pandas()['Subject'].unique().tolist()
        return sorted(subjects)
    elif dataset_name == 'sarvamai/mmlu-indic':
        logger.info('`sarvamai/mmlu-indic` does not have subjects specified in the dataset. Use `all` as the subjects.')
        return ['all']
    elif dataset_name == 'Idavidrein/gpqa':
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
        subjects = dataset.to_pandas()['Subdomain'].unique().tolist()
        return sorted(subjects)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_dataset(dataset_name, subset, split):
    """Load a HuggingFace dataset.
    
    Wrapper around datasets.load_dataset for consistency across the codebase.
    
    Args:
        dataset_name: HuggingFace dataset identifier.
        subset: Dataset subset/config name.
        split: Dataset split (e.g., 'test', 'train').
        
    Returns:
        Dataset: HuggingFace Dataset object.
    """
    dataset = datasets.load_dataset(dataset_name, subset, split=split)
    return dataset