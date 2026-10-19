# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training preflight shared only by the embed and rerank recipes."""

import json
import math
from pathlib import Path


def validate_training_data(
    train_data_path: Path,
    *,
    required_negatives: int,
    require_mined_negatives: bool,
) -> int:
    """Validate records without overstating model-mined negatives as judgements."""
    if required_negatives < 0:
        raise ValueError("required_negatives must be non-negative")

    try:
        payload = json.loads(train_data_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read training JSON from {train_data_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {train_data_path}")
    if not isinstance(payload.get("corpus"), dict):
        raise ValueError("Training data must contain a corpus object")
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError("Training data must contain a non-empty data list")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Training record {index} must be an object")
        if not record.get("pos_doc"):
            raise ValueError(f"Training record {index} has no positive document")
        negatives = record.get("neg_doc")
        if not isinstance(negatives, list) or len(negatives) < required_negatives:
            count = len(negatives) if isinstance(negatives, list) else 0
            raise ValueError(f"Training record {index} has {count} negatives; {required_negatives} are required")
        if require_mined_negatives:
            if record.get("negative_mining_performed") is not True:
                raise ValueError(
                    f"Training record {index} must set negative_mining_performed=True after downstream mining"
                )
            for negative_index, negative in enumerate(negatives[:required_negatives]):
                score = negative.get("score") if isinstance(negative, dict) else None
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                    raise ValueError(f"Training record {index} negative {negative_index} has no finite mining score")

    return len(records)
