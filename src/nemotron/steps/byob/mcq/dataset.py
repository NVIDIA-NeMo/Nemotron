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


import glob
import logging
import numpy as np
import pandas as pd
import os
import random

from collections import Counter

from nemotron.steps.byob.dataset import ByobDataset
from nemotron.steps.byob.config import ByobConfig
from nemotron.steps.byob.hf_utils import load_dataset

logger = logging.getLogger(__name__)

class McqByobDataset(ByobDataset):
    """Dataset class for multiple-choice question (MCQ) BYOB generation.
    
    This class handles loading, parsing, and sampling from various MCQ benchmark datasets
    to create seed data for question generation.
    """
    
    def __init__(self, config: ByobConfig):
        """Initialize the MCQ BYOB dataset.
        
        Args:
            config: Configuration object containing dataset parameters.
        """
        self.config = config
        self.dataset_parsed = self.load_source_dataset()

    def load_source_dataset(self):
        """Load and parse the source dataset from HuggingFace.
        
        Returns:
            dict: Dictionary mapping subjects to their parsed DataFrames.
        """
        subset = self.config.subset
        logger.info(f"Loading source dataset from {self.config.hf_dataset} with subset '{subset}' and split '{self.config.split}'")
        dataset = load_dataset(self.config.hf_dataset, subset, split=self.config.split)
        dataset_parsed = self.parse_dataset(dataset)
        if self.config.metadata_file is not None:
            metadata = pd.read_csv(self.config.metadata_file)
            for subject in dataset_parsed:
                dataset_parsed_merged = pd.merge(
                    dataset_parsed[subject],
                    metadata,
                    on='id',
                    how='left'
                ).dropna()
                assert len(dataset_parsed_merged) == len(dataset_parsed[subject]), f'Metadata and dataset IDs do not match in subject: {subject} ({len(dataset_parsed_merged)} != {len(dataset_parsed[subject])})'
                dataset_parsed[subject] = dataset_parsed_merged
        else:
            for subject in dataset_parsed:
                dataset_parsed[subject]['tags'] = '-'
        return dataset_parsed

    def parse_dataset(self, dataset):
        """Parse the HuggingFace dataset into a standardized format.
        
        Converts various MCQ benchmark datasets (MMLU, MMLU-Pro, MILU, Global-MMLU, etc.)
        into a unified format with columns: id, question, subject, choices, answer.
        
        Args:
            dataset: HuggingFace dataset to parse.
            
        Returns:
            dict: Dictionary mapping subjects to parsed DataFrames.
            
        Raises:
            ValueError: If the dataset format is not supported.
        """
        dataset = dataset.to_pandas()
        dataset['id'] = dataset.index
        dataset['id'] = dataset['id'].apply(lambda x: f'{self.config.hf_dataset}/{self.config.subset}/{self.config.split}#{x}')

        if self.config.hf_dataset == 'cais/mmlu':
            dataset_parsed = {}
            dataset = dataset[['id', 'question', 'subject', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        elif self.config.hf_dataset == 'TIGER-Lab/MMLU-Pro':
            dataset_parsed = {}
            dataset = dataset.rename(columns={'category': 'subject', 'options': 'choices'})
            dataset['answer'] = dataset['answer_index']
            dataset = dataset[['id', 'question', 'subject', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        elif self.config.hf_dataset == 'ai4bharat/MILU':
            dataset_parsed = {}
            dataset['choices'] = dataset[['option1', 'option2', 'option3', 'option4']].apply(lambda x: [x['option1'], x['option2'], x['option3'], x['option4']], axis=1)
            dataset['answer'] = dataset['target'].map({'option1': 0, 'option2': 1, 'option3': 2, 'option4': 3})
            dataset = dataset[['id', 'question', 'subject', 'language', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        elif self.config.hf_dataset in ['CohereLabs/Global-MMLU', 'CohereLabs/Global-MMLU-Lite']:
            dataset_parsed = {}
            dataset['choices'] = dataset[['option_a', 'option_b', 'option_c', 'option_d']].apply(lambda x: [x['option_a'], x['option_b'], x['option_c'], x['option_d']], axis=1)
            dataset['answer'] = dataset['answer'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
            dataset = dataset[['id', 'question', 'subject', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        elif self.config.hf_dataset == 'LinguaLift/IndicMMLU-Pro':
            dataset_parsed = {}
            dataset['subject'] = dataset['category']
            dataset['answer'] = dataset['answer_index']
            dataset['choices'] = dataset['options']
            dataset = dataset[['id', 'question', 'subject', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        elif self.config.hf_dataset == 'openai/MMMLU':
            dataset_parsed = {}
            dataset = dataset.rename(columns={'Question': 'question', 'Answer': 'answer', 'Subject': 'subject'})
            dataset['answer'] = dataset['answer'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
            dataset['choices'] = dataset[['A', 'B', 'C', 'D']].apply(lambda x: [x['A'], x['B'], x['C'], x['D']], axis=1)
            dataset = dataset[['id', 'question', 'subject', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        elif self.config.hf_dataset == 'sarvamai/mmlu-indic':
            dataset_parsed = {}
            dataset['subject'] = 'all'
            dataset = dataset[['id', 'question', 'subject', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        elif self.config.hf_dataset == 'Idavidrein/gpqa':
            dataset_parsed = {}
            dataset = dataset.rename(columns={'Subdomain': 'subject', 'Question': 'question'})
            dataset['choices'] = dataset.apply(lambda x: random.sample([
                x['Correct Answer'],
                x['Incorrect Answer 1'],
                x['Incorrect Answer 2'],
                x['Incorrect Answer 3'],
            ], k=4), axis=1)
            dataset['answer'] = dataset[['choices', 'Correct Answer']].apply(lambda x: x['choices'].index(x['Correct Answer']), axis=1)
            dataset = dataset[['id', 'question', 'subject', 'choices', 'answer']]
            for subject in self.config.source_subjects:
                dataset_parsed[subject] = dataset[dataset['subject'] == subject].reset_index(drop=True)
            return dataset_parsed
        else:
            raise ValueError(f"Unsupported dataset: {self.config.hf_dataset}")

    @staticmethod
    def extract_text_from_path(path: str):
        """Extract text content from a file path or parquet reference.
        
        Args:
            path: File path (e.g., '/path/to/file.txt') or parquet reference 
                  (e.g., '/path/to/file.parquet:file_name').
                  
        Returns:
            list: List containing the text content(s).
        """
        if '.parquet:' in path:
            # Extract from parquet
            parquet_path, file_name = path.rsplit(':', 1)
            df = pd.read_parquet(parquet_path)
            return df[df['file_name'] == file_name]['text'].values
        else:
            # Read from text file
            with open(path, 'r', encoding='utf-8') as f:
                return [f.read()]

    def chunk_text(self, text: str):
        """Extract a random chunk from the text based on window size.
        
        Args:
            text: Input text to chunk.
            
        Returns:
            pd.Series: Series with 'text' (chunked text), 'segment_start', and 'segment_end'.
        """
        window_size = self.config.chunking_config['window_size']
        if window_size is None:
            return pd.Series({'text': text, 'segment_start': 0, 'segment_end': len(text)})
        start_idx = np.random.randint(0, max(0, len(text) - window_size) + 1)
        end_idx = start_idx + window_size
        return pd.Series({'text': text[start_idx:end_idx], 'segment_start': start_idx, 'segment_end': end_idx})

    def make_samples(self, queries_per_target_subject_document: int | None = None):
        """Create seed samples for MCQ generation.
        
        Samples few-shot examples from source subjects and pairs them with target subject
        documents according to the configured mappings and weights.
        
        Args:
            queries_per_target_subject_document: Number of queries per document. 
                                                  Defaults to config value if None.
                                                  
        Returns:
            pd.DataFrame: Seed dataset with columns including text, few-shot examples, 
                          subject mappings, and metadata.
                          
        Raises:
            ValueError: If no samples were generated (e.g., due to missing files or 
                       incompatible tag filters).
        """
        if queries_per_target_subject_document is None:
            queries_per_target_subject_document = self.config.queries_per_target_subject_document

        dataframe_list = []
        for target_subject in self.config.target_source_mapping:
            target_subject_path = os.path.join(self.config.input_dir, target_subject)
            if os.path.isdir(target_subject_path):
                target_subject_files = glob.glob(os.path.join(target_subject_path, '*.txt'))
            else:
                target_subject_path = target_subject_path + '.parquet'
                assert os.path.exists(target_subject_path), f'Target subject path {target_subject_path} does not exist'
                target_subject_df = pd.read_parquet(target_subject_path)
                assert 'file_name' in target_subject_df.columns, f'`file_name` column not found in target subject dataframe for target subject {target_subject_path}'
                assert 'text' in target_subject_df.columns, f'`text` column not found in target subject dataframe for target subject {target_subject_path}'
                target_subject_files = [f'{target_subject_path}:{item}' for item in target_subject_df['file_name'].tolist()]

            for target_subject_file in target_subject_files:
                document_id = target_subject_file
                source_subjects = self.config.target_source_mapping[target_subject]['source_subjects']
                source_weights = self.config.target_source_mapping[target_subject]['source_weights']
                source_tags = self.config.target_source_mapping[target_subject]['source_tags']
                source_tag_weights = self.config.target_source_mapping[target_subject]['source_tag_weights']
                source_subjects_tags = [(subject, tag) for subject in source_subjects for tag in source_tags]
                source_subjects_tags_weights = np.array([w_subject * w_tag for w_subject in source_weights for w_tag in source_tag_weights])

                source_subjects_tags_sampled_indices = np.random.choice(
                    len(source_subjects_tags),
                    size=queries_per_target_subject_document,
                    replace=True,
                    p=source_subjects_tags_weights)
                source_subjects_tags_sampled = [source_subjects_tags[idx] for idx in source_subjects_tags_sampled_indices]
                source_subjects_tags_sampled_counter = Counter(source_subjects_tags_sampled)

                for source_subject, source_tag in source_subjects_tags_sampled_counter:
                    num_samples = source_subjects_tags_sampled_counter[source_subject, source_tag] * self.config.few_shot_samples_per_query
                    numberline = list(range(source_subjects_tags_sampled_counter[source_subject, source_tag])) * self.config.few_shot_samples_per_query
                    source_subject_df = self.dataset_parsed[source_subject].copy()
                    if source_tag != ('',):
                        source_subject_df = source_subject_df[source_subject_df['tags'].apply(lambda x: set(source_tag).issubset(set(x.split(','))))]

                    if len(source_subject_df) == 0:
                        logger.warning(f"No samples found for source subject '{source_subject}' and tags '{','.join(source_tag)}'")
                        logger.warning(f"Dropping {num_samples} samples")
                        continue

                    source_subject_df_sampled = source_subject_df.sample(n=num_samples, replace=True)
                    source_subject_df_sampled['numberline'] = numberline[:len(source_subject_df_sampled)]
                    source_subject_df_sampled['tags'] = [','.join(source_tag)] * len(source_subject_df_sampled)
                    # Group the few-shot samples
                    source_subject_df_sampled_grouped = source_subject_df_sampled.groupby('numberline').agg(list).reset_index()
                    source_subject_df_sampled_grouped = source_subject_df_sampled_grouped.drop(columns=['numberline'])
                    dataframe = source_subject_df_sampled_grouped.copy()
                    dataframe['target_subject'] = target_subject
                    dataframe['text'] = target_subject_file
                    dataframe['document_id'] = document_id
                    dataframe_list.append(dataframe)
                    logger.info(f"Adding {len(dataframe)} samples for target subject '{target_subject}' from source subject '{source_subject}' and tags '{','.join(source_tag)}' ({target_subject_file})")

        if not dataframe_list:
            raise ValueError(
                "No samples were generated in `make_samples`. "
                "Check `target_source_mapping`, input files, and tag filters."
            )
        dataframe_final = pd.concat(dataframe_list).reset_index(drop=True)
        dataframe_final['text_path'] = dataframe_final['text']
        dataframe_final['text'] = dataframe_final['text'].apply(self.extract_text_from_path)
        dataframe_final = dataframe_final.explode('text') # In case there are duplicates in the text column
        dataframe_final[['text', 'segment_start', 'segment_end']] = dataframe_final['text'].apply(self.chunk_text)
        dataframe_final = dataframe_final.rename(columns={'id': 'id_source'})
        dataframe_final['id_target'] = dataframe_final.index
        dataframe_final['id_target'] = dataframe_final['id_target'].apply(lambda x: f'{x}')

        return dataframe_final

    def sample_and_dump(self, queries_per_target_subject_document: int | None = None):
        """Create seed samples and save to parquet file.
        
        Args:
            queries_per_target_subject_document: Number of queries per document.
            
        Returns:
            pd.DataFrame: The seed dataset that was saved.
        """
        dataframe_seed = self.make_samples(queries_per_target_subject_document)
        out_path = os.path.join(self.config.output_dir, self.config.expt_name, 'seed.parquet')
        dataframe_seed.to_parquet(out_path)
        logger.info(f"Sampled and dumped {len(dataframe_seed)} samples to {out_path}")
        return dataframe_seed