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

"""
Data preparation utilities for customization recipes.

Thin wrappers around NeMo Curator, Megatron-Bridge, and DataDesigner
for data acquisition, translation, synthetic generation, quality
assessment, tokenization/packing, and benchmark (BYOB) creation.
"""

from nemotron.customization_recipes.data_prep.acquire import (
    AcquireConfig,
    acquire_and_filter,
    download_dataset,
    classify_domains,
    identify_languages,
    apply_chat_template,
)
from nemotron.customization_recipes.data_prep.translate import (
    TranslationConfig,
    TranslationBackend,
    GoogleBackend,
    AWSBackend,
    LLMBackend,
    evaluate_faithfulness,
    translate_byob_benchmark,
)
from nemotron.customization_recipes.data_prep.sdg import (
    FunctionCall,
    ToolCall,
    Message,
    Conversation,
    ConversationList,
    SDGConfig,
    run_sdg_pipeline,
    generate_synthetic_data,
)
from nemotron.customization_recipes.data_prep.quality import (
    AssessmentConfig,
    AssessmentTool,
    FILTER_REGISTRY,
    create_filter,
    create_scorer_list,
    calculate_aggregates,
    evaluate_data_quality,
    evaluate_model,
)
from nemotron.customization_recipes.data_prep.tokenize_pack import (
    CPTConfig,
    SFTConfig,
    prepare_cpt_data,
    prepare_sft_data,
)
from nemotron.customization_recipes.data_prep.byob import (
    ByobConfig,
    MCQQuestion,
    MCQQuestionList,
    JudgeResult,
    DistractorExpansion,
    DistractorValidityFourChoices,
    DistractorValidityTenChoices,
    generate_questions,
    judge_questions,
    expand_distractors,
    filter_questions,
    check_distractor_validity,
    prepare_byob_seed,
    generate_byob_benchmark,
)
from nemotron.customization_recipes.data_prep.quantize import (
    QuantizeConfig,
    quantize_model,
)

__all__ = [
    # acquire
    "AcquireConfig",
    "acquire_and_filter",
    "download_dataset",
    "classify_domains",
    "identify_languages",
    "apply_chat_template",
    # translate
    "TranslationConfig",
    "TranslationBackend",
    "GoogleBackend",
    "AWSBackend",
    "LLMBackend",
    "evaluate_faithfulness",
    "translate_byob_benchmark",
    # sdg
    "FunctionCall",
    "ToolCall",
    "Message",
    "Conversation",
    "ConversationList",
    "SDGConfig",
    "run_sdg_pipeline",
    "generate_synthetic_data",
    # quality
    "AssessmentConfig",
    "AssessmentTool",
    "FILTER_REGISTRY",
    "create_filter",
    "create_scorer_list",
    "calculate_aggregates",
    "evaluate_data_quality",
    "evaluate_model",
    # tokenize_pack
    "CPTConfig",
    "SFTConfig",
    "prepare_cpt_data",
    "prepare_sft_data",
    # byob
    "ByobConfig",
    "MCQQuestion",
    "MCQQuestionList",
    "JudgeResult",
    "DistractorExpansion",
    "DistractorValidityFourChoices",
    "DistractorValidityTenChoices",
    "generate_questions",
    "judge_questions",
    "expand_distractors",
    "filter_questions",
    "check_distractor_validity",
    "prepare_byob_seed",
    "generate_byob_benchmark",
    # quantize
    "QuantizeConfig",
    "quantize_model",
]
