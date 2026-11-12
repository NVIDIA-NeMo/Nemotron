"""Full-document Parse targets for Automodel's standard VLM SFT recipe."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ParseDocuments(Dataset):
    def __init__(self, path: str):
        manifest = Path(path).absolute()
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty training manifest: {manifest}")
        for row in self.rows:
            row["image"] = str((manifest.parent / row["image"]).absolute())
            if not row.get("target"):
                raise ValueError("SFT records require a nonempty target string")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_dataset(path: str) -> ParseDocuments:
    return ParseDocuments(path)


def collate(examples: list[dict], processor, max_length: int = 4096) -> dict:
    """Pre-shift labels, masking the visual prefix, prompt and right padding."""
    images = []
    for row in examples:
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        images.append(image)
    batch = dict(processor(images=images, return_tensors="pt"))
    tokenizer = processor.tokenizer
    sequences = []
    prompt_lengths = []
    for row, (height, width) in zip(examples, batch["image_sizes"].tolist(), strict=True):
        visual_tokens = (height // 16) * (width // 64) + 1
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        sequence = [tokenizer.pad_token_id] * visual_tokens + prompt + target
        if len(sequence) - 1 > max_length:
            raise ValueError(f"Complete document exceeds max_length: {row.get('id', 'unknown')}")
        sequences.append(sequence)
        prompt_lengths.append(visual_tokens + len(prompt))
    length = max(map(len, sequences)) - 1
    ids = torch.full((len(examples), length), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    for index, (sequence, prefix) in enumerate(zip(sequences, prompt_lengths, strict=True)):
        size = len(sequence) - 1
        ids[index, :size] = torch.tensor(sequence[:-1])
        shifted = torch.tensor(sequence[1:])
        shifted[: prefix - 1] = -100
        labels[index, :size] = shifted
        attention[index, :size] = True
    batch.update(input_ids=ids, labels=labels, attention_mask=attention, use_cache=False)
    batch["pixel_values"] = batch["pixel_values"].bfloat16()
    return batch
