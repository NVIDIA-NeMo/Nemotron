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


import logging
import importlib
import numpy as np

from abc import ABC, abstractmethod
from nemotron.steps.byob.config import ByobConfig
from nemotron.steps.byob.constants import HF_DATASET_TO_MODULE

logger = logging.getLogger(__name__)

class ByobDataset(ABC):
    """Abstract base class for BYOB dataset implementations.
    
    Defines the interface that all dataset-specific implementations must follow.
    Subclasses should implement methods for loading, parsing, sampling, and
    saving dataset-specific formats.
    """
    
    @abstractmethod
    def load_source_dataset(self):
        """Load and return the source dataset.
        
        Returns:
            Parsed dataset structure (implementation-specific).
        """
        pass

    @abstractmethod
    def parse_dataset(self, dataset):
        """Parse dataset into standardized format.
        
        Args:
            dataset: Raw dataset to parse.
            
        Returns:
            Parsed dataset structure.
        """
        pass

    @abstractmethod
    def make_samples(self, dataset_parsed):
        """Create samples from parsed dataset.
        
        Args:
            dataset_parsed: Parsed dataset.
            
        Returns:
            DataFrame with generated samples.
        """
        pass

    @abstractmethod
    def sample_and_dump(self):
        """Create samples and save to disk.
        
        Returns:
            DataFrame with generated samples.
        """
        pass

def make_from_config(config: ByobConfig):
    """Factory function to create dataset instance from configuration.
    
    Dynamically imports the appropriate dataset module based on the HuggingFace
    dataset name in the configuration and instantiates it.
    
    Args:
        config: BYOB configuration object.
        
    Returns:
        ByobDataset: Dataset implementation instance for the configured dataset.
    """
    np.random.seed(config.random_seed)
    module = importlib.import_module(HF_DATASET_TO_MODULE[config.hf_dataset])
    return module.ByobDataset(config)