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
"""TAT-DQA: questions about pages of company annual reports.

Every question comes with its report page and the boxes on the page that hold the evidence. Span
questions ("What was the revenue in 2019?") show the model a crop of the page around the
evidence, so it learns to read the answer off the image. Arithmetic and count questions give it
the text of that crop instead, so it learns to calculate.
"""

import json
import zipfile
from pathlib import Path
from typing import Iterator

import pymupdf
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from base_sft_dataset import BaseSFTDataset, Example

READ_PROMPT = """Read the financial document image and extract the requested evidence exactly.
Question: {question}"""

CALCULATE_PROMPT = """Use the supplied financial-document evidence to calculate or count the requested result.

Evidence:
{evidence}

Question:
{question}"""

# The crop is the evidence plus a margin, at least MIN_HEIGHT tall, in the OCR's page units.
MARGIN_ABOVE, MARGIN_BELOW, MIN_HEIGHT = 130, 100, 420


class DatasetTATDQA(BaseSFTDataset):
    name = "TAT-DQA"
    repo = "next-tat/TAT-DQA"
    revision = "bc46208b22f0bfe47152c98fd0938d238ce868c2"
    splits = {"train": "train", "dev": "dev", "test_gold": "test"}  # question file -> PDF archive
    image_tasks = {"span", "multi-span"}  # answered from the page image; the rest from its text

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir

    def _load_examples(self) -> Iterator[Example]:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        for item, page, ocr in self._report_pages():
            for question in item["questions"]:
                yield self._make_example(question, item["doc"]["source"], page, ocr)

    def _make_example(self, question: dict, report: str, page: Image.Image, ocr: dict) -> Example:
        task, text = question["answer_type"], question["question"].strip()
        crop, evidence = crop_to_evidence(page, ocr, question)
        if task in self.image_tasks:
            image = str(self.image_dir / f"{question['uid']}.png")
            crop.save(image)
            prompt = READ_PROMPT.format(question=text)
        else:
            image = None
            prompt = CALCULATE_PROMPT.format(evidence=evidence, question=text)

        answer = format_answer(question)
        return Example(id=f"tatdqa:{question['uid']}", task=f"TAT-DQA/{task}", split_key=report,
                       prompt=prompt, reasoning=reasoning(question, answer), answer=answer, image=image)

    def _report_pages(self) -> Iterator[tuple[dict, Image.Image, dict]]:
        """Each annotated page: its dataset item (report and questions), the rendered page, and its OCR."""
        source = Path(snapshot_download(self.repo, repo_type="dataset", revision=self.revision,
                                        allow_patterns=["*.json", "*.zip"]))
        for questions_file, archive in self.splits.items():
            items = json.loads((source / f"tatdqa_dataset_{questions_file}.json").read_text())
            with zipfile.ZipFile(source / f"tatdqa_docs_{archive}.zip") as files:
                for item in tqdm(items, desc=f"TAT-DQA {archive}"):
                    uid, page_index = item["doc"]["uid"], int(item["doc"]["page"]) - 1
                    ocr_pages = json.loads(files.read(f"{archive}/{uid}.json"))["pages"]
                    ocr = ocr_pages[min(page_index, len(ocr_pages) - 1)]
                    yield item, render(files.read(f"{archive}/{uid}.pdf"), page_index), ocr


def render(pdf: bytes, page_index: int) -> Image.Image:
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        pixmap = document.load_page(min(page_index, document.page_count - 1)).get_pixmap(dpi=150)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def crop_to_evidence(page: Image.Image, ocr: dict, question: dict) -> tuple[Image.Image, str]:
    """The band of the page around the question's evidence, as an image and as text."""
    evidence_ids = {str(uid) for uids in question.get("block_mapping") or [] for uid in uids}
    evidence = [block for block in ocr["blocks"] if str(block["uuid"]) in evidence_ids]
    if not evidence:  # nothing marked: use the whole page
        return page, "\n".join(block["text"].strip() for block in ocr["blocks"])

    page_height = ocr["bbox"][3]
    top = max(0, min(block["bbox"][1] for block in evidence) - MARGIN_ABOVE)
    bottom = min(page_height, max(block["bbox"][3] for block in evidence) + MARGIN_BELOW)
    grow = max(0, MIN_HEIGHT - (bottom - top)) / 2
    top, bottom = max(0, top - grow), min(page_height, bottom + grow)

    text = "\n".join(block["text"].strip() for block in ocr["blocks"] if top <= block["bbox"][3] and block["bbox"][1] <= bottom)
    to_pixels = page.height / page_height
    return page.crop((0, round(top * to_pixels), page.width, round(bottom * to_pixels))), text


def format_answer(question: dict) -> str:
    """The gold answer with its scale, e.g. "8.6 million" or "37.4%"."""
    answer, scale = question["answer"], question["scale"]
    text = " | ".join(str(part).strip() for part in answer) if isinstance(answer, list) else str(answer).strip()
    if scale == "percent":
        return text if text.endswith("%") else f"{text}%"
    if scale and not text.lower().endswith(scale):
        return f"{text} {scale}"
    return text


def reasoning(question: dict, answer: str) -> str:
    """A short trace from the dataset's own annotations: the evidence values and the calculation."""
    facts = " ; ".join(fact for fact in map(str.strip, map(str, question.get("facts") or [])) if fact)
    steps = []
    if facts:
        steps.append(f"Evidence values: {facts}")
    if question.get("derivation"):
        steps.append(f"Calculation: {question['derivation'].strip()}")
    steps = steps or ["Read the document and extract the requested evidence."]
    return "\n".join([*steps, f"Answer scale: {question['scale'] or 'none'}", f"Final answer: {answer}"])
