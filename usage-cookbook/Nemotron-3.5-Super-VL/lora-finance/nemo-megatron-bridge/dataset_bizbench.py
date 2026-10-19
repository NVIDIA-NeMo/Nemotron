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
"""BizBench: finance questions answered by writing a short Python program.

Text only. The program is the model's reasoning, and the answer follows it.
"""

from typing import Iterator

from datasets import load_dataset

from base_sft_dataset import BaseSFTDataset, Example

PROMPT = """Write an executable Python calculation in the reasoning trace.

Context:
{context}

Question:
{question}"""


class DatasetBizBench(BaseSFTDataset):
    name = "BizBench"
    repo = "kensho/bizbench"
    revision = "0a793f2f886156902c72b4a22cab82bb9dceaecf"
    tasks = {"CodeFinQA", "CodeTAT-QA"}  # the BizBench tasks that come with a program

    def _load_examples(self) -> Iterator[Example]:
        for i, row in enumerate(load_dataset(self.repo, revision=self.revision, split="train+test")):
            if row["task"] not in self.tasks:
                continue
            context, question = row["context"].strip(), row["question"].strip()
            yield Example(
                id=f"bizbench:{i}",
                task=row["task"],
                split_key=context or question,  # questions about the same context stay together
                prompt=PROMPT.format(context=context or "No external document context is required.", question=question),
                reasoning=row["program"].strip(),
                answer=row["answer"].strip(),
            )
