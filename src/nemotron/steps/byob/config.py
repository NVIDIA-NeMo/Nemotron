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
import os
import yaml
from dataclasses import dataclass, field

from nemotron.steps.byob.constants import ALLOWED_HF_DATASETS, HF_DATASET_TO_SUBSET, AVAILABLE_QUALITY_METRICS
from nemotron.steps.byob.hf_utils import get_subjects

logger = logging.getLogger(__name__)

@dataclass
class ByobConfig:
    """Configuration for BYOB (Bring Your Own Benchmark) MCQ generation pipeline.
    
    This dataclass holds all configuration parameters for generating multiple-choice
    questions from custom text corpora using few-shot learning from existing benchmark
    datasets like MMLU, MMLU-Pro, etc.
    
    Attributes:
        expt_name: Unique identifier for this run/experiment.
        hf_dataset: HuggingFace dataset name (e.g., 'cais/mmlu').
        subset: Dataset subset/config name.
        split: Dataset split (e.g., 'test', 'train').
        input_dir: Directory containing target subject text files.
        output_dir: Directory for output files.
        language: Target language for generated questions.
        source_subjects: List of subjects from HF dataset to use as examples.
        target_source_mapping: Mapping from target to source subjects with weights.
        few_shot_samples_per_query: Number of few-shot examples per generation query.
        queries_per_target_subject_document: Number of queries per target document.
        prompt_config: Dictionary of prompt templates for each stage.
        generation_model_config: Configuration for question generation model.
        judge_model_config: Configuration for question quality judgement model.
        do_distractor_expansion: Whether to expand from 4 to 10 choices.
        distractor_expansion_model_config: Configuration for distractor expansion model.
        distractor_validity_model_config: Configuration for validity checking model.
        filtering_model_configs: Configurations for easiness/hallucination filter models.
        easiness_threshold: Threshold for marking questions as too easy (0-1).
        hallucination_threshold: Threshold for marking questions as hallucinated (0-1).
        ndd_batch_size: Batch size for DataDesigner operations.
        metadata_file: Optional CSV file with metadata tags for source questions.
        random_seed: Random seed for reproducibility.
        num_questions_per_query: Number of questions to generate per query.
        semantic_deduplication_config: Configuration for semantic deduplication.
        semantic_outlier_detection_config: Configuration for outlier detection.
        chunking_config: Configuration for text chunking.
        do_coverage_check: Whether to perform text coverage analysis.
        coverage_check_config: Configuration for coverage analysis.
    """
    expt_name: str
    hf_dataset: str
    subset: str
    split: str
    input_dir: str
    output_dir: str
    language: str
    source_subjects: list[str]
    target_subjects: list[str]
    target_source_mapping: dict[str, dict[str, list[str] | np.ndarray]]
    few_shot_samples_per_query: int
    queries_per_target_subject_document: int
    prompt_config: dict[str, dict] | None
    generation_model_config: dict
    judge_model_config: dict
    do_distractor_expansion: bool
    distractor_expansion_model_config: dict
    distractor_validity_model_config: dict
    filtering_model_configs: dict[str, list[dict]]
    easiness_threshold: float
    hallucination_threshold: float
    remove_hallucinated: bool = True
    remove_easy: bool = False
    ndd_batch_size: int = 1000
    metadata_file: str | None = None
    random_seed: int | None = None
    num_questions_per_query: int | None = None
    semantic_deduplication_config: dict = field(default_factory=lambda: {
        'model_identifier': 'sentence-transformers/all-MiniLM-L6-v2',
        'n_clusters': 1,
        'eps': 0.07,
        'remove_duplicates': False,
    })
    semantic_outlier_detection_config: dict = field(default_factory=lambda: {
        'model_identifier': 'sentence-transformers/all-MiniLM-L6-v2',
        'n_neighbours_min': 1,
        'remove_outliers': False,
    })
    chunking_config: dict = field(default_factory=lambda: {
        'window_size': None,
    })
    do_coverage_check: bool = False
    coverage_check_config: dict = field(default_factory=lambda: {
        'window_size': None,
        'model_identifier': None,
    })

    @staticmethod
    def from_yaml(path: str):
        """Load BYOB configuration from a YAML file.
        
        Performs comprehensive validation of all configuration parameters and creates
        necessary directories. Sets defaults for optional parameters and validates
        relationships between configuration fields.
        
        Args:
            path: Path to YAML configuration file.
            
        Returns:
            ByobConfig: Validated configuration object.
            
        Raises:
            AssertionError: If any required field is missing or validation fails.
            ValueError: If dataset is not supported or paths don't exist.
            PermissionError: If output directories are not writable.
        """
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        config['split'] = config.get('split', 'test')
        config['subset'] = config.get('subset', HF_DATASET_TO_SUBSET[config['hf_dataset']])
        config['random_seed'] = config.get('random_seed', None)
        config['ndd_batch_size'] = config.get('ndd_batch_size', ByobConfig.ndd_batch_size)
        config['chunking_config'] = config.get('chunking_config', {'window_size': None})
        config['do_coverage_check'] = config.get('do_coverage_check', False)
        config['coverage_check_config'] = config.get('coverage_check_config', {'window_size': None, 'model_identifier': None})
        config['prompt_config'] = config.get('prompt_config', None)

        assert 'expt_name' in config, "Field `expt_name` is required in the configuration file"
        assert 'output_dir' in config, "Field `output_dir` is required in the configuration file"
        assert 'hf_dataset' in config, "Field `hf_dataset` is required in the configuration file"
        assert 'language' in config, "Field `language` is required in the configuration file"
        assert 'source_subjects' in config, "Field `source_subjects` is required in the configuration file"
        assert 'target_source_mapping' in config, "Field `target_source_mapping` is required in the configuration file"
        assert 'input_dir' in config, "Field `input_dir` is required in the configuration file"
        assert 'few_shot_samples_per_query' in config, "Field `few_shot_samples_per_query` is required in the configuration file"
        assert config['few_shot_samples_per_query'] > 0, "Field `few_shot_samples_per_query` must be greater than 0"
        assert 'queries_per_target_subject_document' in config, "Field `queries_per_target_subject_document` is required in the configuration file"
        assert config['queries_per_target_subject_document'] > 0, "Field `queries_per_target_subject_document` must be greater than 0"
        assert 'num_questions_per_query' in config, "Field `num_questions_per_query` is required in the configuration file"
        assert config['num_questions_per_query'] > 0, "Field `num_questions_per_query` must be greater than 0"
        assert 'generation_model_config' in config, "Field `generation_model_config` is required in the configuration file"
        assert 'judge_model_config' in config, "Field `judge_model_config` is required in the configuration file"
        assert 'do_distractor_expansion' in config, "Field `do_distractor_expansion` is required in the configuration file"
        assert 'distractor_expansion_model_config' in config, "Field `distractor_expansion_model_config` is required in the configuration file"
        assert 'distractor_validity_model_config' in config, "Field `distractor_validity_model_config` is required in the configuration file"
        assert 'filtering_model_configs' in config, "Field `filtering_model_configs` is required in the configuration file"
        assert 'easiness' in config['filtering_model_configs'], "Field `easiness` is required in the filtering model configurations"
        assert 'hallucination' in config['filtering_model_configs'], "Field `hallucination` is required in the filtering model configurations"
        assert 'easiness_threshold' in config, "Field `easiness_threshold` is required in the configuration file"
        assert config['easiness_threshold'] > 0 and config['easiness_threshold'] < 1, "Field `easiness_threshold` must be between 0 and 1"
        assert 'hallucination_threshold' in config, "Field `hallucination_threshold` is required in the configuration file"
        assert config['hallucination_threshold'] > 0 and config['hallucination_threshold'] < 1, "Field `hallucination_threshold` must be between 0 and 1"
        assert 'semantic_deduplication_config' in config, "Field `semantic_deduplication_config` is required in the configuration file"
        assert 'model_identifier' in config['semantic_deduplication_config'], "Field `model_identifier` is required in the semantic deduplication configuration"
        assert 'n_clusters' in config['semantic_deduplication_config'], "Field `n_clusters` is required in the semantic deduplication configuration"
        assert 'eps' in config['semantic_deduplication_config'], "Field `eps` is required in the semantic deduplication configuration"
        assert config['semantic_deduplication_config']['n_clusters'] > 0, "Field `n_clusters` must be greater than 0"
        assert config['semantic_deduplication_config']['eps'] > 0 and config['semantic_deduplication_config']['eps'] < 1, "Field `eps` must be between 0 and 1"
        assert 'remove_duplicates' in config['semantic_deduplication_config'], "Field `remove_duplicates` is required in the semantic deduplication configuration"
        assert isinstance(config['semantic_deduplication_config']['remove_duplicates'], bool), "Field `remove_duplicates` must be a boolean"
        assert 'window_size' in config['chunking_config'], "Field `window_size` is required in the chunking configuration"
        assert config['chunking_config']['window_size'] is None or config['chunking_config']['window_size'] > 0, "Field `window_size` must be greater than 0 or None"

        if config['do_coverage_check']:
            assert 'window_size' in config['coverage_check_config'], "Field `window_size` is required in the coverage check configuration"
            assert config['coverage_check_config']['window_size'] is not None and config['coverage_check_config']['window_size'] > 0, "Field `window_size` must be greater than 0 when coverage check is enabled"
            assert 'model_identifier' in config['coverage_check_config'], "Field `model_identifier` is required in the coverage check configuration"

        assert 'semantic_outlier_detection_config' in config, "Field `semantic_outlier_detection_config` is required in the configuration file"
        assert 'model_identifier' in config['semantic_outlier_detection_config'], "Field `model_identifier` is required in the semantic outlier detection configuration"
        assert 'n_neighbours_min' in config['semantic_outlier_detection_config'], "Field `n_neighbours_min` is required in the semantic outlier detection configuration"
        assert config['semantic_outlier_detection_config']['n_neighbours_min'] > 0, "Field `n_neighbours_min` must be greater than 0"
        assert 'remove_outliers' in config['semantic_outlier_detection_config'], "Field `remove_outliers` is required in the semantic outlier detection configuration"
        assert isinstance(config['semantic_outlier_detection_config']['remove_outliers'], bool), "Field `remove_outliers` must be a boolean"

        if config['prompt_config'] is not None:
            with open(config['prompt_config'], 'r') as f:
                config['prompt_config'] = yaml.safe_load(f)
            for stage in ['qa_generation', 'question_judge', 'hallucination_filter', 'easiness_filter', 'distractor_expansion', 'distractor_validity']:
                assert stage in config['prompt_config'], f"Field `{stage}` is required in the prompt configuration"
                assert 'system_prompt' in config['prompt_config'][stage], f"Field `system_prompt` is required in the prompt configuration for stage {stage}"
                assert 'prompt' in config['prompt_config'][stage], f"Field `prompt` is required in the prompt configuration for stage {stage}"
                assert isinstance(config['prompt_config'][stage]['system_prompt'], str), f"Field `system_prompt` must be a string for stage {stage}"
                assert isinstance(config['prompt_config'][stage]['prompt'], str), f"Field `prompt` must be a string for stage {stage}"
        else:
            from speaker.core.byob.mcq.prompts.utils import get_prompts
            config['prompt_config'] = get_prompts()

        # Set to all subjects if not specified
        if not config['source_subjects']:
            config['source_subjects'] = get_subjects(config['hf_dataset'], config['subset'], config['split'])

        if config['hf_dataset'] not in ALLOWED_HF_DATASETS:
            raise ValueError(f"Invalid Hugging Face dataset: {config['hf_dataset']}. Choose one of the following: {ALLOWED_HF_DATASETS}")

        # Check permissions for output dir
        if os.path.exists(config['output_dir']):
            if not os.access(config['output_dir'], os.W_OK):
                raise PermissionError(f"Output directory {config['output_dir']} is not writable")
            logger.info(f"Output directory {config['output_dir']} already exists and is writable")
        else:
            os.makedirs(config['output_dir'], exist_ok=True)
            logger.info(f"Output directory {config['output_dir']} created")

        # Check if tag is already in output dir
        if os.path.exists(os.path.join(config['output_dir'], config['expt_name'])):
            logger.warning(f"Tag {config['expt_name']} already exists in output directory {config['output_dir']}. Files will be overwritten.")
            if not os.access(os.path.join(config['output_dir'], config['expt_name']), os.W_OK):
                raise PermissionError(f"Tag {config['expt_name']} in output directory {config['output_dir']} is not writable")
        else:
            os.makedirs(os.path.join(config['output_dir'], config['expt_name']), exist_ok=True)
            logger.info(f"Tag {config['expt_name']} created in output directory {config['output_dir']}")

        for subject in config['target_source_mapping']:
            # Check for directory or parquet file
            subject_path = os.path.join(config['input_dir'], subject)
            assert (os.path.exists(subject_path) and os.path.isdir(subject_path)) or os.path.exists(subject_path + '.parquet'), \
                f"Input data path {subject_path} does not exist or is not a directory or parquet file"
            if os.path.isdir(subject_path):
                if os.path.exists(subject_path + '.parquet'):
                    logger.warning(f"Both {subject_path} and {subject_path + '.parquet'} exist. Using {subject_path}")
                subject_files = glob.glob(os.path.join(subject_path, '*.txt'))
                assert len(subject_files) > 0, f"Input data directory {subject_path} does not contain any text files"

        for target_subject in config['target_source_mapping']:
            if isinstance(config['target_source_mapping'][target_subject]['subjects'], list):
                for source_subject in config['target_source_mapping'][target_subject]['subjects']:
                    assert source_subject in config['source_subjects'], \
                        f"Source subject '{source_subject}' in `target_source_mapping` is not in the list of source subjects ({config['source_subjects']})"
                # Use all source subjects if not specified
                if config['target_source_mapping'][target_subject]['subjects'] == []:
                    config['target_source_mapping'][target_subject]['subjects'] = config['source_subjects']
                # Make an array with equal weights for each source subject
                labels = config['target_source_mapping'][target_subject]['subjects']
                weights = np.ones(len(labels)) / len(labels)
            elif isinstance(config['target_source_mapping'][target_subject]['subjects'], dict):
                labels = list(config['target_source_mapping'][target_subject]['subjects'].keys())
                weights = np.array(list(config['target_source_mapping'][target_subject]['subjects'].values()))
                weights = weights / np.sum(weights)
            else:
                raise ValueError(f"Invalid type for `target_source_mapping` for target subject '{target_subject}'")

            if config.get('metadata_file') is None:
                assert 'tags' not in config['target_source_mapping'][target_subject], "`tags` should not be specified if `metadata_file` is not specified"
            tags = config['target_source_mapping'][target_subject].get('tags', [''])
            if isinstance(tags, list):
                tag_labels = [tuple(tag.split(',')) for tag in tags]
                tag_weights = np.ones(len(tags)) / len(tags)
            elif isinstance(tags, dict):
                tag_labels = [tuple(tag.split(',')) for tag in tags.keys()]
                tag_weights = np.array(list(tags.values()))
                tag_weights = tag_weights / np.sum(tag_weights)
            else:
                raise ValueError(f"Invalid type for `tags` for target subject '{target_subject}' ({type(tags)})")

            config['target_source_mapping'][target_subject] = {
                'source_subjects': labels,
                'source_weights': weights,
                'source_tags': tag_labels,
                'source_tag_weights': tag_weights,
            }

        _seen_aliases = set()
        for filter_type in ['easiness', 'hallucination']:
            for filtering_model_config in config['filtering_model_configs'][filter_type]:
                assert 'alias' in filtering_model_config, "Field `alias` is required in the filtering model configuration"
                assert filtering_model_config['alias'] not in _seen_aliases, f"Alias {filtering_model_config['alias']} is already used in the filtering model configurations"
                _seen_aliases.add(filtering_model_config['alias'])

        cfg = ByobConfig(
            expt_name=config['expt_name'],
            hf_dataset=config['hf_dataset'],
            subset=config['subset'],
            metadata_file=config.get('metadata_file'),
            split=config['split'],
            input_dir=config['input_dir'],
            output_dir=config['output_dir'],
            language=config['language'],
            source_subjects=config['source_subjects'],
            target_subjects=list(config['target_source_mapping'].keys()),
            target_source_mapping=config['target_source_mapping'],
            few_shot_samples_per_query=config['few_shot_samples_per_query'],
            queries_per_target_subject_document=config['queries_per_target_subject_document'],
            ndd_batch_size=config['ndd_batch_size'],
            random_seed=config['random_seed'],
            num_questions_per_query=config['num_questions_per_query'],
            prompt_config=config['prompt_config'],
            generation_model_config=config['generation_model_config'],
            judge_model_config=config['judge_model_config'],
            do_distractor_expansion=config['do_distractor_expansion'],
            distractor_expansion_model_config=config['distractor_expansion_model_config'],
            distractor_validity_model_config=config['distractor_validity_model_config'],
            filtering_model_configs=config['filtering_model_configs'],
            easiness_threshold=config['easiness_threshold'],
            hallucination_threshold=config['hallucination_threshold'],
            remove_hallucinated=config.get('remove_hallucinated', ByobConfig.remove_hallucinated),
            remove_easy=config.get('remove_easy', ByobConfig.remove_easy),
            semantic_deduplication_config=config['semantic_deduplication_config'],
            semantic_outlier_detection_config=config['semantic_outlier_detection_config'],
            chunking_config=config['chunking_config'],
            do_coverage_check=config['do_coverage_check'],
            coverage_check_config=config['coverage_check_config'],
        )
        logger.info(f"Loaded config: {cfg}")
        return cfg

@dataclass
class ByobTranslationConfig:
    """Configuration for BYOB translation pipeline.

    Holds configuration parameters for translating datasets between languages
    with optional backtranslation quality metrics.

    Attributes:
        expt_name: Unique identifier for this translation run.
        dataset_path: Path to the dataset to translate.
        output_dir: Directory for output files.
        source_language: Source language code (e.g., 'en').
        target_language: Target language code (e.g., 'hi').
        translation_model_config: Configuration for translation model.
        ndd_batch_size: Batch size for DataDesigner operations.
        backtranslation_quality_metrics: List of quality metrics for evaluation.
    """
    expt_name: str
    dataset_path: str
    output_dir: str
    source_language: str
    target_language: str
    translation_model_config: dict
    ndd_batch_size: int = 32
    backtranslation_quality_metrics: list[dict] = field(default_factory=lambda: [])
    remove_low_quality: bool = True

    @staticmethod
    def from_yaml(path: str):
        """Load translation configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file.
            
        Returns:
            ByobTranslationConfig: Validated configuration object.
            
        Raises:
            AssertionError: If any required field is missing or validation fails.
        """
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        config['ndd_batch_size'] = config.get('ndd_batch_size', ByobTranslationConfig.ndd_batch_size)

        assert 'translation_model_config' in config, "Field `translation_model_config` is required in the configuration file"
        assert 'backtranslation_quality_metrics' in config, "Field `backtranslation_quality_metrics` is required in the configuration file"
        for quality_metric in config['backtranslation_quality_metrics']:
            assert 'type' in quality_metric, "Field `type` is required in the backtranslation quality metric configuration"
            assert quality_metric['type'] in AVAILABLE_QUALITY_METRICS, f"Invalid backtranslation quality metric type: {quality_metric['type']}. Choose one of the following: {AVAILABLE_QUALITY_METRICS}"
            assert 'threshold' in quality_metric, "Field `threshold` is required in the backtranslation quality metric configuration"
            assert quality_metric['threshold'] >= 0, "Field `threshold` must be greater than or equal to 0"

        return ByobTranslationConfig(
            expt_name=config['expt_name'],
            dataset_path=config['dataset_path'],
            output_dir=config['output_dir'],
            source_language=config['source_language'],
            target_language=config['target_language'],
            translation_model_config=config['translation_model_config'],
            backtranslation_quality_metrics=config['backtranslation_quality_metrics'],
            ndd_batch_size=config['ndd_batch_size'],
            remove_low_quality=config.get('remove_low_quality', ByobTranslationConfig.remove_low_quality),
        )