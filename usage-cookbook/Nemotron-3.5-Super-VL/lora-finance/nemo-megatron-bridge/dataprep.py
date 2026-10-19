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
"""Build the finance training data in $WORKSPACE/dataprep. Run by `steps/dataprep.sh`.

Writes train.jsonl (one conversation per row), test.jsonl (held-out questions with their reference
answers, for `steps/try.sh`), and images/ (the page crops the image rows point to).
"""

import json
import os
import random
from pathlib import Path

from dataset_bizbench import DatasetBizBench
from dataset_tatdqa import DatasetTATDQA

output_dir = Path(os.environ["WORKSPACE"]) / "dataprep"
output_dir.mkdir(parents=True, exist_ok=True)

datasets = [
    DatasetTATDQA(image_dir=output_dir / "images"),  # annual-report pages: read off the image, or calculate
    DatasetBizBench(),  # finance questions answered by a short Python program
]

train, test = [], []
for dataset in datasets:
    print(f"Preparing {dataset.name}...")
    dataset_train, dataset_test = dataset.make_dataset()
    print(f"  train: {len(dataset_train)}  test: {len(dataset_test)}")
    train += dataset_train
    test += dataset_test

# Shuffle deterministically to interleave the datasets.
random.Random(1234).shuffle(train)
random.Random(1234).shuffle(test)

for name, rows in (("train.jsonl", train), ("test.jsonl", test)):
    with open(output_dir / name, "w") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"Wrote {len(rows)} rows to '{output_dir / name}'")
