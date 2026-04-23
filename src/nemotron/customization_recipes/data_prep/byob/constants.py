# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for the BYOB seed data preparation pipeline (ported from Speaker)."""

ALLOWED_HF_DATASETS = [
    "cais/mmlu",
    "TIGER-Lab/MMLU-Pro",
    "ai4bharat/MILU",
    "CohereLabs/Global-MMLU",
    "CohereLabs/Global-MMLU-Lite",
    "LinguaLift/IndicMMLU-Pro",
    "openai/MMMLU",
    "sarvamai/mmlu-indic",
    "Idavidrein/gpqa",
]

# Default subset/config for each supported Hugging Face dataset
HF_DATASET_TO_SUBSET = {
    "cais/mmlu": "all",
    "TIGER-Lab/MMLU-Pro": "default",
    "ai4bharat/MILU": "English",
    "CohereLabs/Global-MMLU": "en",
    "CohereLabs/Global-MMLU-Lite": "en",
    "LinguaLift/IndicMMLU-Pro": "hindi",
    "openai/MMMLU": "default",
    "sarvamai/mmlu-indic": "en",
    "Idavidrein/gpqa": "gpqa_main",
}

# All listed datasets use the same MCQ dataset implementation
HF_DATASET_TO_MODULE = {ds: "mcq" for ds in HF_DATASET_TO_SUBSET}
