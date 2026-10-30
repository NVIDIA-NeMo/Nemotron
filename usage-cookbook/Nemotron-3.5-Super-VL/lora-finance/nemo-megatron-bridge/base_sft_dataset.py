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
"""Turns a source dataset into chat rows for training, and holds out some for testing.

Each dataset subclass only loads its own data as `Example`s. This class gives every example the
same chat format, so all datasets train the model the same way.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

INSTRUCTION = (
    "Return a grounded reasoning trace inside <think>...</think>, followed by only the final answer. "
    "Preserve the requested financial unit or scale."
)


@dataclass
class Example:
    id: str
    task: str
    split_key: str  # examples with the same key land in the same split: use the source document's ID
    prompt: str  # the task-specific part of the question
    reasoning: str
    answer: str
    image: str | None = None  # absolute path under WORKSPACE, for questions answered from an image


class BaseSFTDataset(ABC):
    name: str

    def make_dataset(self) -> tuple[list[dict], list[dict]]:
        """Training rows (full conversations) and test rows (question plus reference answer)."""
        train, test = [], []
        for example in self._load_examples():
            if self._is_test(example):
                test.append({"messages": [self._user_turn(example)],
                             "id": example.id, "task": example.task, "answer": example.answer})
            else:
                train.append({"messages": [self._user_turn(example), self._assistant_turn(example)]})
        return train, test

    @abstractmethod
    def _load_examples(self) -> Iterator[Example]:
        """Load the source data as examples."""

    @staticmethod
    def _user_turn(example: Example) -> dict:
        content = [{"type": "text", "text": f"Task: {example.task}\n{INSTRUCTION}\n{example.prompt}"}]
        if example.image:
            content.insert(0, {"type": "image", "image": example.image})
        return {"role": "user", "content": content}

    @staticmethod
    def _assistant_turn(example: Example) -> dict:
        # The model's native reasoning format, byte for byte: no whitespace around </think>.
        text = f"<think>\n{example.reasoning}</think>{example.answer}"
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    @staticmethod
    def _is_test(example: Example) -> bool:
        """Hold out one in ten source documents: all examples from one document land in the same split."""
        return int(hashlib.sha256(example.split_key.encode()).hexdigest(), 16) % 10 == 0
