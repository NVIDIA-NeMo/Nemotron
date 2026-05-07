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

from nemotron.steps.byob.runtime.benchmark_families.mcq.utils import format_qa
from nemotron.steps.byob.runtime.config import ByobTranslationConfig


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
    from nemo_curator.stages.text.experimental.translation import TextQualityMetricStage
    from nemo_curator.tasks import DocumentBatch

    dataset_out = dataset.copy()
    dataset_out["_byob_reference_text"] = dataset_out[["question", "options"]].apply(
        lambda x: format_qa(x["question"], x["options"]),
        axis=1,
    )
    dataset_out["_byob_backtranslated_text"] = dataset_out[
        ["question_translated_backtranslated", "options_translated_backtranslated"]
    ].apply(
        lambda x: format_qa(x["question_translated_backtranslated"], x["options_translated_backtranslated"]),
        axis=1,
    )

    stage = TextQualityMetricStage(
        reference_text_field="_byob_reference_text",
        hypothesis_text_field="_byob_backtranslated_text",
        metrics=config.backtranslation_quality_metrics,
        filter_enabled=False,
    )
    batch = DocumentBatch(task_id=f"{config.expt_name}-quality", dataset_name=config.expt_name, data=dataset_out)
    dataset_out = stage.process(batch).to_pandas()
    return dataset_out.drop(columns=["_byob_reference_text", "_byob_backtranslated_text"])
