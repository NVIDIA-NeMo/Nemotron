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
import sacrebleu

from nemotron.steps.byob.config import ByobTranslationConfig
from nemotron.steps.byob.mcq.utils import format_qa

AVAILABLE_QUALITY_METRICS = ['sacrebleu', 'chrf', 'ter']

def compute_metric(hypothesis: str, references: str, metric: str, threshold: float):
    """Compute translation quality metric and check if it passes threshold.
    
    Supports BLEU, chrF, and TER metrics for evaluating translation quality.
    
    Args:
        hypothesis: Translated text to evaluate.
        references: Reference translation(s) for comparison.
        metric: Metric type ('sacrebleu', 'chrf', or 'ter').
        threshold: Threshold value for pass/fail determination.
                   Higher values needed to pass for BLEU/chrF, lower for TER.
                   
    Returns:
        pd.Series: Series with 'score' (float) and 'passed' (bool) fields.
        
    Raises:
        ValueError: If metric type is not supported.
    """
    if metric == 'sacrebleu':
        score = sacrebleu.sentence_bleu(hypothesis, references).score
        passed = score >= threshold
    elif metric == 'chrf':
        score = sacrebleu.sentence_chrf(hypothesis, references).score
        passed = score >= threshold
    elif metric == 'ter':
        score = sacrebleu.sentence_ter(hypothesis, references).score
        passed = score <= threshold
    else:
        raise ValueError(f"Invalid metric: {metric}")
    return pd.Series({'score': score, 'passed': passed})

def evaluate_quality_metrics(dataset: pd.DataFrame, config: ByobTranslationConfig):
    """Evaluate translation quality using backtranslation and configured metrics.
    
    Compares original text with backtranslated text using multiple quality metrics
    to assess translation fidelity.
    
    Args:
        dataset: DataFrame with columns: question, options, 
                question_translated_backtranslated, options_translated_backtranslated.
        config: Translation configuration with quality metric specifications.
        
    Returns:
        pd.DataFrame: Original dataset augmented with score columns for each metric
                     (score_{metric}, score_{metric}_passed).
    """
    dataset_out = dataset.copy()
    for quality_metric in config.backtranslation_quality_metrics:
        metric = quality_metric['type']
        threshold = quality_metric['threshold']
        dataset_out[[f'score_{metric}', f'score_{metric}_passed']] = dataset_out[['question', 'options', 'question_translated_backtranslated', 'options_translated_backtranslated']].apply(lambda x: compute_metric(
            format_qa(x['question_translated_backtranslated'], x['options_translated_backtranslated']), [format_qa(x['question'], x['options'])], metric, threshold), axis=1)
    dataset_out['is_quality_metric_passed'] = dataset_out[[f'score_{metric['type']}_passed' for metric in config.backtranslation_quality_metrics]].all(axis=1)
    return dataset_out