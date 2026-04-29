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
import os
import random
import re

from pydantic import BaseModel

from nemotron.steps.byob.config import ByobConfig, ByobTranslationConfig
from nemotron.steps.byob.mcq.formatters import format_qa, format_qa_batch
from nemotron.steps.byob.mcq.response_model import QuestionAnswerFourChoices
from nemotron.steps.byob.translation.utils import get_language

def prepare_generation_seed_dataset(config: ByobConfig):
    """Prepare the seed dataset for question generation.
    
    Loads the seed parquet file and formats it with few-shot examples for LLM prompting.
    
    Args:
        config: Configuration object containing paths and settings.
        
    Returns:
        pd.DataFrame: Formatted seed dataset ready for question generation.
    """
    seed_df = pd.read_parquet(os.path.join(config.output_dir, config.expt_name, 'seed.parquet'))

    seed_df['few_shot_examples'] = seed_df.apply(format_qa_batch, axis=1)
    seed_df['id_source'] = seed_df['id_source'].apply(lambda x: ','.join([str(item) for item in x]))
    seed_df['subject'] = seed_df['subject'].apply(lambda x: ','.join(x))
    seed_df['tags'] = seed_df['tags'].apply(str)
    seed_df['language'] = get_language(config.language)
    seed_df.drop(columns=['question', 'choices', 'answer'], inplace=True)
    seed_df.rename(columns={'subject': 'source_subject'}, inplace=True)

    return seed_df

def prepare_distractor_expansion_seed_dataset(config: ByobConfig, dataset: pd.DataFrame):
    """Prepare dataset for distractor expansion stage.
    
    Formats questions with their 4 original choices for expansion to 10 choices.
    
    Args:
        config: Configuration object.
        dataset: DataFrame with generated questions.
        
    Returns:
        pd.DataFrame: Formatted dataset for distractor expansion.
    """
    dataset = dataset[['id_question', 'target_subject', 'text', 'language', 'question_generated', 'choices_generated', 'answer_generated']].copy()
    dataset['question_answer_generated_formatted'] = dataset[['question_generated', 'choices_generated', 'answer_generated']].apply(lambda x: format_qa(x['question_generated'], x['choices_generated'], x['answer_generated']), axis=1)
    dataset.drop(columns=['question_generated', 'choices_generated', 'answer_generated'], inplace=True)
    return dataset

def prepare_filtering_seed_dataset(dataset: pd.DataFrame, config: ByobConfig):
    """Prepare dataset for question filtering stage.
    
    Formats questions for easiness and hallucination filtering.
    
    Args:
        dataset: DataFrame with generated questions.
        config: Configuration object.
        
    Returns:
        pd.DataFrame: Formatted dataset for filtering.
    """
    dataset = dataset[['id_question', 'target_subject', 'text', 'question_generated', 'choices_generated', 'answer_generated']].copy()
    dataset['question_generated_formatted'] = dataset[['question_generated', 'choices_generated']].apply(lambda x: format_qa(x['question_generated'], x['choices_generated']), axis=1)
    dataset.drop(columns=['question_generated', 'choices_generated'], inplace=True)
    return dataset

def prepare_distractor_validity_seed_dataset(config: ByobConfig, dataset: pd.DataFrame):
    """Prepare dataset for distractor validity checking stage.
    
    Formats questions with all choices for validity verification.
    
    Args:
        config: Configuration object.
        dataset: DataFrame with generated questions.
        
    Returns:
        pd.DataFrame: Formatted dataset for distractor validity checking.
    """
    dataset = dataset[['id_question', 'document_id', 'target_subject', 'text', 'question_generated', 'choices_generated', 'answer_generated']].copy()
    dataset['question_answer_generated_formatted'] = dataset[['question_generated', 'choices_generated', 'answer_generated']].apply(
        lambda x: format_qa(x['question_generated'], x['choices_generated'], x['answer_generated'], append_choices=True, append_answer=True), axis=1)
    dataset.drop(columns=['choices_generated', 'answer_generated'], inplace=True)
    return dataset

def postprocess_generated_questions(generated_questions: pd.DataFrame, response_model: type(BaseModel) = QuestionAnswerFourChoices) -> pd.DataFrame:
    """Postprocess generated questions from LLM output format.
    
    Extracts individual questions from the structured LLM response and creates
    unique IDs for each question.
    
    Args:
        generated_questions: DataFrame with 'result' column containing LLM responses.
        response_model: Pydantic model type used for generation (default: QuestionAnswerFourChoices).
        
    Returns:
        pd.DataFrame: Flattened DataFrame with one row per question.
        
    Raises:
        ValueError: If response_model is not supported.
    """
    generated_questions = generated_questions.copy()
    generated_questions['result'] = generated_questions['result'].apply(lambda x: list(x['questions']) if x['questions'] is not None else None)
    generated_questions.dropna(inplace=True)
    generated_questions['id_question'] = generated_questions['result'].apply(lambda x: list(range(len(x))))
    generated_questions = generated_questions.explode(['result', 'id_question']).reset_index(drop=True)
    generated_questions['id_question'] = generated_questions[['id_target', 'id_question']].apply(lambda x: f'{x['id_target']}_q{x['id_question']}', axis=1)
    generated_questions['question_generated'] = generated_questions['result'].apply(lambda x: x['question'])

    if response_model == QuestionAnswerFourChoices:
        generated_questions['choices_generated'] = generated_questions['result'].apply(lambda x: [x['choice_a'], x['choice_b'], x['choice_c'], x['choice_d']])
        generated_questions['answer_generated'] = generated_questions['result'].apply(lambda x: x['answer'])
    else:
        raise ValueError(f"Unsupported response model: {response_model}")

    columns_drop = ['result', 'result__reasoning_trace'] if 'result__reasoning_trace' in generated_questions.columns else ['result']
    generated_questions.drop(columns=columns_drop, inplace=True)
    return generated_questions

def prepare_judgement_seed_dataset(config: ByobConfig, dataset: pd.DataFrame):
    """Prepare dataset for question judgement stage.
    
    Formats questions for quality and validity assessment.
    
    Args:
        config: Configuration object.
        dataset: DataFrame with generated questions.
        
    Returns:
        pd.DataFrame: Formatted dataset for judgement.
    """
    dataset = dataset[['id_question', 'question_generated', 'choices_generated', 'answer_generated']].copy()
    dataset['question_answer_formatted'] = dataset[['question_generated', 'choices_generated', 'answer_generated']].apply(lambda x: format_qa(x['question_generated'], x['choices_generated'], x['answer_generated']), axis=1)
    dataset.drop(columns=['question_generated', 'choices_generated', 'answer_generated'], inplace=True)
    dataset.rename(columns={'question_answer_formatted': 'question'}, inplace=True)
    return dataset

def postprocess_judged_questions(dataset_in: pd.DataFrame, dataset_out: pd.DataFrame):
    """Postprocess judged questions and filter to keep only valid ones.
    
    Extracts judgement results and filters to keep only questions marked as valid.
    
    Args:
        dataset_in: Original dataset before judgement.
        dataset_out: Dataset with judgement results.
        
    Returns:
        pd.DataFrame: Merged dataset containing only valid questions with judgement metadata.
    """
    dataset_out = dataset_out.copy()
    dataset_out['is_valid_judge'] = dataset_out['result'].apply(lambda x: x['is_valid'])
    dataset_out['category_judge'] = dataset_out['result'].apply(lambda x: x['category'])
    dataset_out['reason_judge'] = dataset_out['result'].apply(lambda x: x['reason'])
    dataset_out.drop(columns=['result'], inplace=True)
    dataset_out = pd.merge(
        dataset_in,
        dataset_out[['id_question', 'is_valid_judge', 'category_judge', 'reason_judge']],
        on='id_question',
        how='inner'
    )
    dataset_out = dataset_out[dataset_out['is_valid_judge']]
    return dataset_out

def postprocess_distractor_expansion(dataset_in: pd.DataFrame, dataset_out: pd.DataFrame, num_distractors: int = 6):
    """Postprocess distractor expansion results.
    
    Combines original 4 choices with 6 new distractors and shuffles them while
    tracking the new answer index.
    
    Args:
        dataset_in: Original dataset with 4 choices.
        dataset_out: Dataset with expanded distractors.
        num_distractors: Number of new distractors added (default: 6).
        
    Returns:
        pd.DataFrame: Dataset with 10 shuffled choices and updated answer indices.
    """

    def shuffle_choices(row: pd.Series):
        choices = row['choices_generated']
        answer = row['answer_generated']
        answer_index = ord(answer) - ord('A')
        indices_original, choices_shuffled = zip(*random.sample(list(enumerate(choices)), len(choices)))
        for idx_new, idx_original in enumerate(indices_original):
            if idx_original == answer_index:
                answer_index_new = idx_new
                break
        row['choices_generated'] = list(choices_shuffled)
        row['answer_generated'] = chr(ord('A') + answer_index_new)
        return row

    dataset_out = dataset_out.copy()
    dataset_out['new_choices'] = dataset_out['result_distractor_expansion'].apply(
        lambda x: [x[f'choice_{chr(ord('e') + idx)}'] for idx in range(0, num_distractors)])
    dataset_out.drop(columns=['result_distractor_expansion'], inplace=True)
    dataset_out = pd.merge(
        dataset_in,
        dataset_out[['id_question', 'new_choices']],
        on='id_question',
        how='left'
    )
    dataset_out.dropna(inplace=True)
    dataset_out['choices_generated_original'] = dataset_out['choices_generated']
    dataset_out['choices_generated'] = dataset_out[['choices_generated', 'new_choices']].apply(lambda x: list(x['choices_generated']) + list(x['new_choices']), axis=1)
    dataset_out.drop(columns=['new_choices'], inplace=True)
    dataset_out = dataset_out.apply(shuffle_choices, axis=1)
    return dataset_out

def parse_filter_response(row: pd.Series, config: ByobConfig):
    """Parse LLM filter responses to extract predicted answers.
    
    Extracts answer letters from filtering model responses using regex patterns.
    
    Args:
        row: DataFrame row containing filter responses.
        config: Configuration object with filter settings.
        
    Returns:
        pd.Series: Row with added answer columns for each filter type and model.
    """
    def parse(text: str, num_choices: int):
        char_start = 'A'
        char_end = chr(ord(char_start) + num_choices - 1)
        patterns = [
            r'[aA]nswer is\s*\(?\s*[' + f'{char_start}-{char_end}' + r']\s*\)?',                 # Answer is (B) or Answer is B, parenthesis optional
            r'[aA]nswer is\s*\\?\(?\s*\\boxed\{[' + f'{char_start}-{char_end}' + r']\}\s*\\?\)?',      # Answer is (\boxed{B}) or Answer is \boxed{B}, parenthesis optional
            r'[aA]nswer is\s*\\?\*?\\?\[\s*[' + f'{char_start}-{char_end}' + r']\s*\\?\]\\?\*?',              # Answer is *[D]* or answer is [D]
        ]
        answer_text = None
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                answer_text = match.group(0).strip()
                break

        if answer_text is not None:
            answer = answer_text.split(' ')[-1].replace('(', '').replace(')', '').replace('\\boxed{', '').replace('}', '').replace('\\', '').replace('*', '').replace('[', '').replace(']', '')
            if answer not in [chr(ord('A') + i) for i in range(num_choices)]:
                return '-'
            return answer
        else:
            return '-'

    num_choices = len(row['choices_generated'])
    for filter_type in ['easiness', 'hallucination']:
        for filtering_model_config in config.filtering_model_configs[filter_type]:
            row[f'answer_{filter_type}_{filtering_model_config['alias']}'] = parse(row[f'response_{filter_type}_{filtering_model_config['alias']}'], num_choices)
    return row

def postprocess_filtered_questions(dataset_in: pd.DataFrame, dataset_out: pd.DataFrame, config: ByobConfig):
    """Postprocess filtered questions and compute easiness/hallucination metrics.
    
    Parses filter responses, computes correct ratio for each filter type, and marks
    questions as easy or containing hallucinations based on thresholds.
    
    Args:
        dataset_in: Original dataset before filtering.
        dataset_out: Dataset with filter responses.
        config: Configuration object with thresholds.
        
    Returns:
        pd.DataFrame: Dataset with filtering metrics and flags.
    """
    def get_correct_ratio(row: pd.Series, column_names: list[str]):
        correct_answer = row['answer_generated']
        correct_count = sum([int(row[key] == correct_answer) for key in column_names])
        correct_ratio = correct_count / len(column_names)
        return correct_ratio

    # Parse the response of the easiness filter
    dataset_out = dataset_out.copy()
    dataset_out = pd.merge(
        dataset_out,
        dataset_in[['id_question', 'choices_generated']],
        on='id_question',
        how='inner'
    )
    dataset_out = dataset_out.apply(lambda x: parse_filter_response(x, config), axis=1)
    column_names_easiness = [f'answer_easiness_{filtering_model_config['alias']}' for filtering_model_config in config.filtering_model_configs['easiness']]
    column_names_hallucination = [f'answer_hallucination_{filtering_model_config['alias']}' for filtering_model_config in config.filtering_model_configs['hallucination']]
    # Find ratio of correct answers
    dataset_out['correct_ratio_easiness'] = dataset_out[['answer_generated'] + column_names_easiness].apply(lambda x: get_correct_ratio(x, column_names_easiness), axis=1)
    dataset_out['correct_ratio_hallucination'] = dataset_out[['answer_generated'] + column_names_hallucination].apply(lambda x: get_correct_ratio(x, column_names_hallucination), axis=1)
    dataset_out['is_easy'] = dataset_out['correct_ratio_easiness'] > config.easiness_threshold
    dataset_out['is_hallucination'] = dataset_out['correct_ratio_hallucination'] < config.hallucination_threshold
    dataset_out = pd.merge(
        dataset_in,
        dataset_out[['id_question', 'correct_ratio_easiness', 'correct_ratio_hallucination', 'is_easy', 'is_hallucination'] + column_names_easiness + column_names_hallucination],
        on='id_question',
        how='inner'
    )
    return dataset_out

def postprocess_distractor_validity(dataset_in: pd.DataFrame, dataset_out: pd.DataFrame):
    """Postprocess distractor validity checking results.
    
    Removes choices that are marked as correct (false negatives) while preserving
    the original correct answer.
    
    Args:
        dataset_in: Original dataset before validity checking.
        dataset_out: Dataset with validity checking results.
        
    Returns:
        pd.DataFrame: Dataset with validated choices, excluding false negative distractors.
    """

    def fix_choices(row: pd.Series):
        answer_index = ord(row['answer_generated']) - ord('A')
        choices = row['choices_generated']
        answer_text = choices[answer_index]

        choices_validated = []
        answer_index_validated = None
        has_dropped_choices = False
        for idx, choice in enumerate(choices):
            choice_letter = chr(ord('a') + idx)
            # Prioritize the original answer and drop false negatives.
            if idx == answer_index:
                answer_index_validated = len(choices_validated)
                choices_validated.append(answer_text)
            elif row['result_distractor_validity'][f'choice_{choice_letter}'] == 'No' and choice != answer_text and choice not in choices_validated:
                choices_validated.append(choice)
            else:
                has_dropped_choices = True
        row['choices_generated_validated'] = choices_validated
        row['answer_generated_validated'] = chr(ord('A') + answer_index_validated)
        row['has_dropped_choices'] = has_dropped_choices
        return row

    dataset_out = dataset_out.copy()
    dataset_out = pd.merge(
        dataset_out,
        dataset_in[['id_question', 'choices_generated', 'answer_generated']],
        on='id_question',
        how='inner'
    )
    dataset_out = dataset_out.apply(fix_choices, axis=1)
    dataset_out = dataset_out[dataset_out['choices_generated_validated'].apply(lambda x: len(x)) >= 4] # Keep only questions with at least 4 choices.

    dataset_in = dataset_in.rename(columns={'choices_generated': 'choices_generated_prevalidation', 'answer_generated': 'answer_generated_prevalidation'})
    dataset_out.drop(columns=['choices_generated', 'answer_generated'], inplace=True)
    dataset_out = dataset_out.rename(columns={'choices_generated_validated': 'choices_generated', 'answer_generated_validated': 'answer_generated'})
    dataset = pd.merge(
        dataset_in,
        dataset_out[['id_question', 'choices_generated', 'answer_generated', 'has_dropped_choices']],
        on='id_question',
        how='inner'
    )
    return dataset

def prepare_translation_seed_dataset(dataset: pd.DataFrame, source_language: str, target_language: str, id_field: str = 'question_id', text_field: str = 'question', options_field: str = 'options'):
    """Prepare MCQ dataset for translation.
    
    Separates questions and choices into individual rows for translation.
    
    Args:
        dataset: Source dataset to translate.
        source_language: Source language code (e.g., 'en').
        target_language: Target language code (e.g., 'hi').
        id_field: Name of the question ID column.
        text_field: Name of the question text column.
        options_field: Name of the options column.
        
    Returns:
        pd.DataFrame: Formatted dataset with one row per translatable item.
    """
    dataset = dataset.copy()
    seed_df_questions = dataset[[id_field, text_field]].copy().rename(columns={text_field: 'text'})
    seed_df_questions['type'] = 'question'
    seed_df_questions['translation_id'] = seed_df_questions[id_field].apply(lambda x: f'tq#{x}')
    seed_df_choices = dataset[[id_field, options_field]].copy()
    seed_df_choices['type'] = 'choice'
    seed_df_choices['choice_index'] = seed_df_choices[options_field].apply(lambda x: list(range(len(x))))
    seed_df_choices = seed_df_choices.explode([options_field, 'choice_index']).reset_index(drop=True)
    seed_df_choices['translation_id'] = seed_df_choices[[id_field, 'choice_index']].apply(lambda x: f'tc#{x[id_field]}#{x['choice_index']}', axis=1)
    seed_df_choices.rename(columns={options_field: 'text'}, inplace=True)
    seed_df_choices.drop(columns=['choice_index'], inplace=True)
    seed_df = pd.concat([seed_df_questions, seed_df_choices])

    seed_df['source_language_code'] = source_language
    seed_df['target_language_code'] = target_language
    seed_df['source_language'] = get_language(source_language)
    seed_df['target_language'] = get_language(target_language)
    return seed_df

def postprocess_translated_questions(dataset_in: pd.DataFrame, dataset_out: pd.DataFrame, id_field: str = 'question_id', text_field: str = 'question', options_field: str = 'options', answer_index_field: str = 'answer_index', suffix: str = 'translated'):
    """Postprocess translated questions and reconstruct MCQ format.
    
    Combines translated questions and choices back into MCQ format and updates
    answer indices to match the translated choice ordering.
    
    Args:
        dataset_in: Original dataset before translation.
        dataset_out: Dataset with translations.
        id_field: Name of the question ID column.
        text_field: Name of the question text column.
        options_field: Name of the options column.
        answer_index_field: Name of the answer index column.
        suffix: Suffix to append to translated column names.
        
    Returns:
        pd.DataFrame: Dataset with translated questions and updated answer indices.
    """
    def find_index(row: pd.Series):
        answer_index = row[answer_index_field]
        choice_index = row[f'choice_index_{suffix}']
        try:
            return choice_index.index(answer_index)
        except (IndexError, ValueError):
            return None

    dataset_out = pd.DataFrame({
        'translation_id': dataset_out['translation_id'].tolist(),
        id_field: dataset_out[id_field].tolist(),
        'translation': dataset_out['translation'].tolist(),
        'type': dataset_out['type'].tolist()
    })
    dataset_out_questions = dataset_out[dataset_out['type'] == 'question'].copy()
    dataset_out_questions.drop(columns=['type'], inplace=True)
    dataset_out_questions = dataset_out_questions.rename(columns={'translation': f'{text_field}_{suffix}'})
    dataset_out_choices = dataset_out[dataset_out['type'] == 'choice'].copy()
    dataset_out_choices.drop(columns=['type'], inplace=True)
    dataset_out_choices[f'choice_index_{suffix}'] = dataset_out_choices['translation_id'].apply(lambda x: int(x.split('#')[-1]))
    dataset_out_choices = dataset_out_choices.groupby(id_field).agg(list).reset_index()
    dataset_out_choices = dataset_out_choices.rename(columns={'translation': f'{options_field}_{suffix}'})
    dataset_out = pd.merge(
        dataset_out_questions[[id_field, f'{text_field}_{suffix}']],
        dataset_out_choices[[id_field, f'{options_field}_{suffix}', f'choice_index_{suffix}']],
        on=id_field,
        how='inner'
    )
    dataset_out = pd.merge(
        dataset_in,
        dataset_out,
        on=id_field,
        how='inner'
    )
    dataset_out[f'{answer_index_field}_{suffix}'] = dataset_out[[answer_index_field, f'choice_index_{suffix}']].apply(find_index, axis=1)
    return dataset_out