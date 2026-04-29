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
import shutil
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod

from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.io.reader import ParquetReader

from nemotron.steps.byob.config import ByobConfig

def segment_text(text: str, window_size: int, hop_size: int = None, id_prefix: str = ''):
    """Segment text into overlapping or non-overlapping windows.
    
    Splits text into segments of fixed window size with configurable hop size for
    overlap. The last segment is extended if it would be incomplete.
    
    Args:
        text: Input text to segment.
        window_size: Size of each segment in characters.
        hop_size: Number of characters to advance between segments. If None, equals
                 window_size (no overlap).
        id_prefix: Prefix for segment IDs (e.g., 'doc_1').
        
    Returns:
        pd.DataFrame: DataFrame with columns: id, text, start, end for each segment.
        
    Raises:
        AssertionError: If window_size is None or <= 0.
    """
    assert window_size is not None and window_size > 0, "Field `window_size` must be greater than 0"
    if hop_size is None:
        hop_size = window_size
    segments = []
    for i in range(0, len(text), hop_size):
        if len(text[i:i+window_size]) < window_size and i != 0: # If the last segment is not complete, add the remaining text to the last segment
            segments[-1]['text'] += text[i:i+window_size]
            segments[-1]['end'] = segments[-1]['start'] + len(segments[-1]['text'])
        else:
            segments.append({'id': f'{id_prefix}#{len(segments)}', 'text': text[i:i+window_size], 'start': i, 'end': i+window_size})
    return pd.DataFrame(segments)

class TextCoverage(ABC):
    """Abstract base class for text coverage analysis.
    
    Analyzes how well generated questions cover the source text passages using
    semantic embeddings and similarity matching. Helps identify under-covered
    text regions and question distribution patterns.
    """
    
    def __init__(self, config: ByobConfig):
        """Initialize text coverage analyzer.
        
        Creates necessary directories for intermediate files and results.
        
        Args:
            config: BYOB configuration with coverage_check_config settings.
        """
        self.config = config
        if os.path.exists(os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'text_coverage')):
            shutil.rmtree(os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'text_coverage'))
        self.input_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'text_coverage', 'input_data')
        self.output_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'text_coverage', 'results')
        self.cache_path = os.path.join(self.config.output_dir, self.config.expt_name, 'artifacts', 'text_coverage', 'cache')
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)

    @abstractmethod
    def prepare_input_data(self, dataset: pd.DataFrame):
        """Prepare dataset for coverage analysis.
        
        Subclasses should implement this to extract text passages and queries
        from their specific dataset format.
        
        Args:
            dataset: Input dataset.
            
        Returns:
            tuple: (dataset_text, dataset_queries) DataFrames.
        """
        pass

    def _compute_embeddings(self, input_path: str, output_path: str, text_field: str, embedding_field: str, output_fields: list[str]):
        """Compute embeddings for text using NeMo Curator pipeline.
        
        Args:
            input_path: Path to input parquet file.
            output_path: Path to output directory.
            text_field: Name of column containing text.
            embedding_field: Name of column to store embeddings.
            output_fields: List of columns to include in output.
            
        Returns:
            pd.DataFrame: DataFrame with embeddings.
        """
        embedding_pipeline = Pipeline(
            stages=[
                ParquetReader(file_paths=input_path, _generate_ids=False),
                EmbeddingCreatorStage(
                    model_identifier=self.config.coverage_check_config['model_identifier'],
                    text_field=text_field,
                    embedding_field=embedding_field
                ),
                ParquetWriter(path=output_path, fields=output_fields)
            ],
            name='embedding_pipeline'
        )
        embedding_pipeline.run()

        return pd.concat([pd.read_parquet(path) for path in glob.glob(os.path.join(output_path, '*.parquet'))])

    def analyze(self, dataset: pd.DataFrame):
        """Analyze text coverage of generated questions.
        
        Segments source text into windows, computes embeddings for both text segments
        and questions, then matches questions to most similar text segments using
        cosine similarity. Calculates coverage metrics showing what fraction of
        text segments are covered by at least one question.
        
        Args:
            dataset: Dataset with questions and source text references.
            
        Returns:
            pd.DataFrame: Original dataset augmented with coverage metrics:
                         segment_match_start, segment_match_end, coverage, coverage_max.
        """
        dataset_text, dataset_queries = self.prepare_input_data(dataset)

        # Segment the text
        window_size = self.config.coverage_check_config['window_size']
        dataset_text['segments'] = dataset_text.apply(
            lambda row: segment_text(
                row['text'],
                window_size,
                id_prefix=str(row['document_id']),
            ),
            axis=1,
        )
        dataset_text.drop(columns='text', inplace=True) # Avoid memory issues

        dataset_text['segment_id'] = dataset_text['segments'].apply(lambda x: x['id'].tolist())
        dataset_text['segment_text'] = dataset_text['segments'].apply(lambda x: x['text'].tolist())
        dataset_text['segment_start'] = dataset_text['segments'].apply(lambda x: x['start'].tolist())
        dataset_text['segment_end'] = dataset_text['segments'].apply(lambda x: x['end'].tolist())
        dataset_text.drop(columns=['segments'], inplace=True)
        dataset_text = dataset_text.explode(['segment_id', 'segment_text', 'segment_start', 'segment_end']).reset_index(drop=True)

        # Compute the embeddings
        dataset_text.to_parquet(os.path.join(self.input_path, 'document_segments.parquet'))
        embeddings_documents = self._compute_embeddings(
            os.path.join(self.input_path, 'document_segments.parquet'),
            os.path.join(self.output_path, 'documents'),
            'segment_text',
            'embeddings',
            ['document_id', 'segment_id', 'segment_start', 'segment_end', 'embeddings']
        )

        dataset_queries.to_parquet(os.path.join(self.input_path, 'queries.parquet'))
        embeddings_queries = self._compute_embeddings(
            os.path.join(self.input_path, 'queries.parquet'),
            os.path.join(self.output_path, 'queries'),
            'query',
            'embeddings',
            ['document_id', 'id_question', 'embeddings']
        )

        question_to_segment = []
        for document_id in embeddings_documents['document_id'].unique():
            embeddings_segments_doc = embeddings_documents[embeddings_documents['document_id'] == document_id].reset_index(drop=True)
            embeddings_queries_doc = embeddings_queries[embeddings_queries['document_id'] == document_id].reset_index(drop=True)

            doc_matrix = torch.tensor(np.array(embeddings_segments_doc['embeddings'].tolist()), dtype=torch.float32).unsqueeze(0)
            question_matrix = torch.tensor(np.array(embeddings_queries_doc['embeddings'].tolist()), dtype=torch.float32).unsqueeze(1)
            similarity = torch.nn.functional.cosine_similarity(doc_matrix, question_matrix, dim=2)
            similarity_argmax = torch.argmax(similarity, dim=1).tolist()
            similarity_max = torch.max(similarity, dim=1).values.cpu().numpy().tolist()
            segment_match = [embeddings_segments_doc.iloc[i]['segment_id'] for i in similarity_argmax]
            question_to_segment.append(pd.DataFrame({
                'id_question': embeddings_queries_doc['id_question'],
                'segment_match_start': embeddings_segments_doc.iloc[similarity_argmax]['segment_start'].tolist(),
                'segment_match_end': embeddings_segments_doc.iloc[similarity_argmax]['segment_end'].tolist(),
                'similarity_max': similarity_max,
                'coverage': len(set(segment_match))/len(embeddings_segments_doc),
                'coverage_max': min(len(embeddings_segments_doc), len(segment_match))/len(embeddings_segments_doc) # Maximum achievable coverage with the given number of queries
            }))
        question_to_segment = pd.concat(question_to_segment).reset_index(drop=True)

        dataset_out = pd.merge(
            dataset,
            question_to_segment,
            on='id_question'
        )
        return dataset_out