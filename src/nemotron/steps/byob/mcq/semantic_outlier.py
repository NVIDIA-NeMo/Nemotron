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
import pandas as pd
import os
import glob
import shutil
from collections import Counter
from sklearn.cluster import KMeans

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage

from nemotron.steps.byob.config import ByobConfig

logger = logging.getLogger(__name__)

class TextSemanticOutlierDetectionMCQ:
    """Detect and remove semantic outliers in MCQ datasets.
    
    Uses embeddings and clustering to identify questions where the correct answer
    is semantically dissimilar from all other choices, which may indicate issues
    with the question or answer choices.
    """

    def __init__(self, config: ByobConfig):
        """Initialize semantic outlier detection.

        Args:
            config: Configuration object containing outlier detection parameters.
        """
        self.config = config
        if os.path.exists(os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_outlier_detection')):
            shutil.rmtree(os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_outlier_detection'))
        self.input_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_outlier_detection', 'input_data')
        self.output_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_outlier_detection', 'results')
        self.cache_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'semantic_outlier_detection', 'cache')

        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)

    def _compute_embeddings(self, input_path: str, output_path: str, text_field: str, embedding_field: str, output_fields: list[str]):
        """Compute embeddings for text data using NeMo Curator pipeline.

        Args:
            input_path: Path to input parquet file.
            output_path: Path to output directory for embeddings.
            text_field: Name of the column containing text.
            embedding_field: Name of the column to store embeddings.
            output_fields: List of columns to include in output.

        Returns:
            pd.DataFrame: DataFrame with embeddings.
        """
        embedding_pipeline = Pipeline(
            stages=[
                ParquetReader(file_paths=input_path, _generate_ids=False),
                EmbeddingCreatorStage(
                    model_identifier=self.config.semantic_outlier_detection_config['model_identifier'],
                    text_field=text_field,
                    embedding_field=embedding_field
                ),
                ParquetWriter(path=output_path, fields=output_fields)
            ],
            name='embedding_pipeline'
        )
        embedding_pipeline.run()
        parquet_paths = glob.glob(os.path.join(output_path, '*.parquet'))
        if not parquet_paths:
            logger.warning("No embedding parquet files produced; returning empty DataFrame")
            return pd.DataFrame(columns=output_fields)
        return pd.concat([pd.read_parquet(path) for path in parquet_paths])

    def cleanup(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)

    def prepare_input_data(self, dataset: pd.DataFrame):
        """Prepare MCQ dataset for outlier detection.
        
        Flattens the choices into individual rows for embedding computation.

        Args:
            dataset: DataFrame with columns: id_question, choices_generated.

        Returns:
            pd.DataFrame: Flattened DataFrame with one row per choice.
        """
        dataset_choices = dataset[['id_question', 'choices_generated']].copy()
        dataset_choices['choice_index'] = dataset_choices['choices_generated'].apply(lambda x: list(range(len(x))))
        dataset_choices_flat = dataset_choices.explode(['choices_generated', 'choice_index'])
        dataset_choices_flat = dataset_choices_flat.rename(columns={'choices_generated': 'text'})

        return dataset_choices_flat

    def detect(self, dataset: pd.DataFrame):
        """Detect semantic outliers in the MCQ dataset.

        Computes embeddings for all choices, clusters them, and identifies questions
        where the correct answer has too few semantic neighbors in its cluster.

        Args:
            dataset: DataFrame with columns: id_question, choices_generated, 
                    answer_generated.

        Returns:
            pd.DataFrame: Original dataset augmented with outlier detection results:
                         answer_semantic_neighbours, is_outlier.
        """
        self.cleanup()
        dataset_prepared = self.prepare_input_data(dataset)
        dataset_prepared.to_parquet(os.path.join(self.input_path, 'choices.parquet'))
        dataset_embeddings = self._compute_embeddings(
            os.path.join(self.input_path, 'choices.parquet'),
            os.path.join(self.output_path),
            'text',
            'embeddings',
            ['id_question', 'choice_index', 'embeddings'])
        dataset_embeddings['choice_index'] = dataset_embeddings['choice_index'].astype(object)
        dataset_embeddings = dataset_embeddings.sort_values(by=['id_question', 'choice_index'])
        dataset_embeddings = dataset_embeddings.groupby(['id_question']).agg(list).reset_index()
        dataset_embeddings = pd.merge(
            dataset[['id_question', 'answer_generated']],
            dataset_embeddings,
            on='id_question',
            how='inner'
        )
        dataset_embeddings['answer_index'] = dataset_embeddings['answer_generated'].apply(lambda x: ord(x) - ord('A'))
        # Cluster into two groups and see if the answer has enough number of semantic neighbours in the cluster.
        dataset_embeddings['embedding_cluster'] = dataset_embeddings['embeddings'].apply(lambda x: KMeans(n_clusters=min(2, len(x)), random_state=42).fit_predict(x))
        dataset_embeddings['embedding_cluster_counter'] = dataset_embeddings['embedding_cluster'].apply(Counter)
        dataset_embeddings['answer_semantic_neighbours'] = dataset_embeddings[['answer_index', 'embedding_cluster', 'embedding_cluster_counter']].apply(
            lambda x: x['embedding_cluster_counter'][x['embedding_cluster'][x['answer_index']]] - 1,
            axis=1
        )
        dataset_embeddings['is_outlier'] = dataset_embeddings['answer_semantic_neighbours'] < self.config.semantic_outlier_detection_config['n_neighbours_min']
        logger.info(f"Found {dataset_embeddings['is_outlier'].sum()}/{len(dataset_embeddings)} outliers")

        if self.config.semantic_outlier_detection_config['remove_outliers']:
            dataset = dataset[~dataset_embeddings['is_outlier']]
            logger.info(f"Removed {dataset_embeddings['is_outlier'].sum()} outliers")

        dataset_out = pd.merge(
            dataset,
            dataset_embeddings[['id_question', 'answer_semantic_neighbours', 'is_outlier']],
            on='id_question',
            how='inner'
        )
        return dataset_out