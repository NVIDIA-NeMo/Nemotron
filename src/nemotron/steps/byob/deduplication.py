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
import glob
import logging
import pandas as pd
import shutil
from abc import ABC, abstractmethod

from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow
from nemo_curator.backends.experimental.ray_data import RayDataExecutor

from nemotron.steps.byob.config import ByobConfig

logger = logging.getLogger(__name__)

class TextSemanticDeduplication(ABC):
    def __init__(self, config: ByobConfig):
        self.config = config
        self.input_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_deduplication', 'input_data')
        self.output_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_deduplication', 'results')
        self.cache_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_deduplication', 'cache')

        self.workflow = TextSemanticDeduplicationWorkflow(
            input_path=self.input_path,
            output_path=self.output_path,
            cache_path=self.cache_path,
            model_identifier=self.config.semantic_deduplication_config['model_identifier'],
            n_clusters=self.config.semantic_deduplication_config['n_clusters'],
            text_field="text",
            id_field="id",
            perform_removal=False,
            eps=self.config.semantic_deduplication_config['eps'])
        self.executor = RayDataExecutor()
        os.makedirs(self.input_path, exist_ok=True)

        if os.path.exists(self.output_path):
            logger.warning(f"Output path {self.output_path} already exists. Removing it.")
            shutil.rmtree(self.output_path)
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)

    @abstractmethod
    def prepare_input_data(self, dataset: pd.DataFrame):
        pass

    def _mark_duplicates(self, dataset: pd.DataFrame):
        paths = glob.glob(os.path.join(self.output_path, 'duplicates/*.parquet'))
        if not paths:
            dataset['is_duplicate'] = False
            return dataset

        duplicates = pd.concat([pd.read_parquet(path) for path in paths])
        dataset['is_duplicate'] = dataset['id_question'].isin(duplicates['id'])
        return dataset

    def run(self, dataset: pd.DataFrame):
        dataset_temp = self.prepare_input_data(dataset)
        dataset_temp.to_parquet(os.path.join(self.input_path, 'questions.parquet'))
        self.workflow.run(self.executor)
        dataset_dedup = self._mark_duplicates(dataset)
        num_duplicates = dataset_dedup['is_duplicate'].sum()
        logger.info(f"Found {num_duplicates}/{len(dataset_dedup)} duplicate questions")
        if self.config.semantic_deduplication_config['remove_duplicates']:
            dataset_dedup = dataset_dedup[~dataset_dedup['is_duplicate']]
        return dataset_dedup