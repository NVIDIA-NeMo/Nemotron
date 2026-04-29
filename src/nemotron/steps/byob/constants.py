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

"""Constants for BYOB pipeline configuration.

This module defines supported datasets, their default subsets, module mappings,
and available quality metrics for the BYOB (Bring Your Own Benchmark) system.

Constants:
    ALLOWED_HF_DATASETS: List of supported HuggingFace benchmark datasets.
    HF_DATASET_TO_SUBSET: Default subset/config for each dataset.
    HF_DATASET_TO_MODULE: Python module path for dataset-specific implementations.
    AVAILABLE_QUALITY_METRICS: Supported translation quality metrics.
"""

ALLOWED_HF_DATASETS = [
    'cais/mmlu',
    'TIGER-Lab/MMLU-Pro',
    'ai4bharat/MILU',
    'CohereLabs/Global-MMLU',
    'CohereLabs/Global-MMLU-Lite',
    'LinguaLift/IndicMMLU-Pro',
    'openai/MMMLU',
    'sarvamai/mmlu-indic',
    'Idavidrein/gpqa',
]

# Default subsets for Hugging Face datasets
HF_DATASET_TO_SUBSET = {
    'cais/mmlu': 'all',
    'TIGER-Lab/MMLU-Pro': 'default',
    'ai4bharat/MILU': 'English',
    'CohereLabs/Global-MMLU': 'en',
    'CohereLabs/Global-MMLU-Lite': 'en',
    'LinguaLift/IndicMMLU-Pro': 'hindi',
    'openai/MMMLU': 'default',
    'sarvamai/mmlu-indic': 'en',
    'Idavidrein/gpqa': 'gpqa_main',
}

HF_DATASET_TO_MODULE = {
    'cais/mmlu': 'speaker.core.byob.mcq',
    'TIGER-Lab/MMLU-Pro': 'speaker.core.byob.mcq',
    'ai4bharat/MILU': 'speaker.core.byob.mcq',
    'CohereLabs/Global-MMLU': 'speaker.core.byob.mcq',
    'CohereLabs/Global-MMLU-Lite': 'speaker.core.byob.mcq',
    'LinguaLift/IndicMMLU-Pro': 'speaker.core.byob.mcq',
    'openai/MMMLU': 'speaker.core.byob.mcq',
    'sarvamai/mmlu-indic': 'speaker.core.byob.mcq',
    'Idavidrein/gpqa': 'speaker.core.byob.mcq',
}

AVAILABLE_QUALITY_METRICS = [
    'sacrebleu',
    'chrf',
    'ter',
]