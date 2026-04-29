import os
import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner

from nemotron.steps.byob.data_designer_utils import setup_model_config
from nemotron.steps.byob.config import ByobConfig
from nemotron.steps.byob.mcq.response_model import QuestionAnswerList, DistractorExpansion, DistractorValidityFourChoices, DistractorValidityTenChoices, JudgeResult

def generate_questions(config: ByobConfig, seed_df: pd.DataFrame):
    """Generate multiple-choice questions using LLM.
    
    Uses DataDesigner to generate questions based on seed examples and target text passages.
    
    Args:
        config: Configuration object containing generation model settings and prompts.
        seed_df: Seed DataFrame with few-shot examples, target text, and metadata.
        
    Returns:
        pd.DataFrame: DataFrame containing generated questions with 'result' column.
    """
    generation_model_config = setup_model_config(config.generation_model_config)
    seed_dataset = dd.DataFrameSeedSource(df=seed_df)
    data_designer = DataDesigner(artifact_path=f'{config.output_dir}/{config.expt_name}/artifacts/data_designer')
    config_builder = dd.DataDesignerConfigBuilder(model_configs=[generation_model_config])
    config_builder.with_seed_dataset(seed_dataset)

    system_prompt_qa = config.prompt_config['qa_generation']['system_prompt'].format(
        num_few_shot_samples=config.few_shot_samples_per_query,
        num_questions=config.num_questions_per_query)
    prompt_qa = config.prompt_config['qa_generation']['prompt'].format(num_questions=config.num_questions_per_query)
    config_builder.add_column(
        name='result',
        column_type='llm-structured',
        system_prompt=system_prompt_qa,
        prompt=prompt_qa,
        output_format=QuestionAnswerList,
        model_alias=config.generation_model_config['alias']
    )

    job_results = data_designer.create(config_builder=config_builder, num_records=len(seed_df))
    dataset = job_results.load_dataset()
    dataset.dropna(inplace=True)
    return dataset

def judge_questions(config: ByobConfig, seed_df: pd.DataFrame):
    """Judge the quality and validity of generated questions using LLM.
    
    Evaluates questions for clarity, validity, and categorizes them as knowledge,
    reasoning, or both.
    
    Args:
        config: Configuration object containing judge model settings and prompts.
        seed_df: DataFrame with questions to judge.
        
    Returns:
        pd.DataFrame: DataFrame with 'result' column containing judgement results.
    """
    judge_model_config = setup_model_config(config.judge_model_config)
    seed_dataset = dd.DataFrameSeedSource(df=seed_df)
    data_designer = DataDesigner(artifact_path=f'{config.output_dir}/{config.expt_name}/artifacts/data_designer')
    config_builder = dd.DataDesignerConfigBuilder(model_configs=[judge_model_config])
    config_builder.with_seed_dataset(seed_dataset)

    system_prompt_judge = config.prompt_config['question_judge']['system_prompt']
    prompt_judge = config.prompt_config['question_judge']['prompt']
    config_builder.add_column(
        name='result',
        column_type='llm-structured',
        system_prompt=system_prompt_judge,
        prompt=prompt_judge,
        output_format=JudgeResult,
        model_alias=config.judge_model_config['alias']
    )

    job_results = data_designer.create(config_builder=config_builder, num_records=len(seed_df))
    dataset = job_results.load_dataset()
    dataset.dropna(inplace=True)
    return dataset

def expand_distractors(config: ByobConfig, seed_df: pd.DataFrame):
    """Expand the number of distractor (incorrect) choices from 4 to 10.
    
    Uses LLM to generate 6 additional plausible but incorrect choices.
    
    Args:
        config: Configuration object containing distractor expansion model settings.
        seed_df: DataFrame with questions containing 4 choices.
        
    Returns:
        pd.DataFrame: DataFrame with 'result_distractor_expansion' column containing
                     additional choices E-J.
    """
    distractor_expansion_model_config = setup_model_config(config.distractor_expansion_model_config)
    seed_dataset = dd.DataFrameSeedSource(df=seed_df)
    data_designer = DataDesigner(artifact_path=f'{config.output_dir}/{config.expt_name}/artifacts/data_designer')
    config_builder = dd.DataDesignerConfigBuilder(model_configs=[distractor_expansion_model_config])
    config_builder.with_seed_dataset(seed_dataset)

    system_prompt_distractor_expansion = config.prompt_config['distractor_expansion']['system_prompt']
    prompt_distractor_expansion = config.prompt_config['distractor_expansion']['prompt']
    config_builder.add_column(
        name='result_distractor_expansion',
        column_type='llm-structured',
        system_prompt=system_prompt_distractor_expansion,
        prompt=prompt_distractor_expansion,
        output_format=DistractorExpansion,
        model_alias=config.distractor_expansion_model_config['alias']
    )

    job_results = data_designer.create(config_builder=config_builder, num_records=len(seed_df))
    dataset = job_results.load_dataset()
    dataset.dropna(inplace=True)
    return dataset

def filter_questions(config: ByobConfig, dataset: pd.DataFrame):
    """Filter questions based on easiness and hallucination criteria.
    
    Uses multiple LLM models to answer questions and determines if they are too easy
    (high correct ratio) or contain hallucinations (low correct ratio).
    
    Args:
        config: Configuration object containing filtering model settings and prompts.
        dataset: DataFrame with questions to filter.
        
    Returns:
        pd.DataFrame: DataFrame with response columns for each filter type and model.
    """
    num_choices = 4 if not config.do_distractor_expansion else 10
    choices_text = '/'.join([chr(ord('A') + i) for i in range(num_choices)])

    system_prompt_filter = {
        'easiness': config.prompt_config['easiness_filter']['system_prompt'].format(num_choices=num_choices), # A/B/C/D when not expanding distractors, A/B/C/D/E/F/G/H/I/J when expanding distractors
        'hallucination': config.prompt_config['hallucination_filter']['system_prompt'].format(num_choices=num_choices)
    }
    prompt_filter = {
        'easiness': config.prompt_config['easiness_filter']['prompt'].format(choices=choices_text),
        'hallucination': config.prompt_config['hallucination_filter']['prompt'].format(choices=choices_text)
    }

    filter_model_configs = [setup_model_config(model_config) \
        for filter_type in ['easiness', 'hallucination'] \
        for model_config in config.filtering_model_configs[filter_type]]
    
    seed_dataset = dd.DataFrameSeedSource(df=dataset)
    data_designer = DataDesigner(artifact_path=f'{config.output_dir}/{config.expt_name}/artifacts/data_designer')
    config_builder = dd.DataDesignerConfigBuilder(model_configs=filter_model_configs)
    config_builder.with_seed_dataset(seed_dataset)

    for filter_type in ['easiness', 'hallucination']:
        for filtering_model_config in config.filtering_model_configs[filter_type]:
            config_builder.add_column(
                name=f"response_{filter_type}_{filtering_model_config['alias']}",
                column_type='llm-text',
                system_prompt=system_prompt_filter[filter_type],
                prompt=prompt_filter[filter_type],
                model_alias=filtering_model_config['alias'],
            )

    job_results = data_designer.create(config_builder=config_builder, num_records=len(dataset))
    dataset = job_results.load_dataset()
    return dataset

def check_distractor_validity(config: ByobConfig, dataset: pd.DataFrame):
    """Check if any distractor choices are actually correct.
    
    Uses LLM to verify that only the designated correct answer is actually correct
    given the source text passage.
    
    Args:
        config: Configuration object containing distractor validity model settings.
        dataset: DataFrame with questions to validate.
        
    Returns:
        pd.DataFrame: DataFrame with 'result_distractor_validity' column indicating
                     which choices are correct.
    """
    num_choices = 10 if config.do_distractor_expansion else 4
    dataset['num_choices'] = num_choices
    system_prompt_distractor_validity = config.prompt_config['distractor_validity']['system_prompt'].format(num_choices=num_choices)
    prompt_distractor_validity = config.prompt_config['distractor_validity']['prompt']

    distractor_validity_model_config = setup_model_config(config.distractor_validity_model_config)
    
    seed_dataset = dd.DataFrameSeedSource(df=dataset)
    data_designer = DataDesigner(artifact_path=f'{config.output_dir}/{config.expt_name}/artifacts/data_designer')
    config_builder = dd.DataDesignerConfigBuilder(model_configs=[distractor_validity_model_config])
    config_builder.with_seed_dataset(seed_dataset)

    config_builder.add_column(
        name='result_distractor_validity',
        column_type='llm-structured',
        system_prompt=system_prompt_distractor_validity,
        prompt=prompt_distractor_validity,
        output_format=DistractorValidityTenChoices if num_choices == 10 else DistractorValidityFourChoices,
        model_alias=config.distractor_validity_model_config['alias']
    )
    job_results = data_designer.create(config_builder=config_builder, num_records=len(dataset))
    dataset = job_results.load_dataset()
    dataset.dropna(inplace=True)
    return dataset