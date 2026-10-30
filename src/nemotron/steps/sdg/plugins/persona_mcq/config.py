# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer configuration for persona-grounded MCQ authoring."""

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class PersonaMCQConfig(SingleColumnConfig):
    """Configure one persona-grounded multiple-choice question column."""

    column_type: Literal["persona-mcq"] = "persona-mcq"
    model_alias: str = "question_model"
    persona_column: str = "persona"
    num_options: int = 4
    language: str = "English"
    random_seed: int = 13
    facet_weights: dict[str, float] | None = None
    difficulty_weights: dict[str, float] | None = None

    @property
    def required_columns(self) -> list[str]:
        return [self.persona_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return ["conversation_status"]
