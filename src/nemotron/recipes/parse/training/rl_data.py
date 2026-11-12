"""Native NeMo RL dataset/processor boundary for complete Parse pages."""

import json
from pathlib import Path

import torch
from datasets import Dataset
from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.multimodal_utils import PackedTensor
from PIL import Image


class ParsePages(RawDataset):
    def __init__(self, data_path, split=None, split_validation_size=0, seed=42, **kwargs):
        path = Path(data_path).absolute()
        self.task_name = "parse-page"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            row.update(image=str((path.parent / row["image"]).absolute()), task_name=self.task_name)
        self.dataset = Dataset.from_list(rows)
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)


def process_page(row, task_data_spec, processor, max_seq_length, idx):
    prompt = row["prompt"]
    if task_data_spec.prompt != prompt:
        raise ValueError("Keep the native full-page prompt identical across SFT and RLVR")
    tokenizer = processor.tokenizer
    with Image.open(row["image"]) as image:
        image = image.convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    height, width = inputs["image_sizes"][0].tolist()
    visual_tokens = (height // 16) * (width // 64) + 1
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([tokenizer.pad_token_id] * visual_tokens + prompt_ids)
    if len(ids) >= max_seq_length:
        raise ValueError("Prompt exceeds sequence length")
    pixels = inputs["pixel_values"]
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    message = dict(
        role="user",
        content=prompt,
        token_ids=ids,
        pixel_values=PackedTensor(pixels, dim_to_pack=0),
        image_sizes=PackedTensor(inputs["image_sizes"], dim_to_pack=0),
        content_sizes=PackedTensor(inputs["content_sizes"], dim_to_pack=0),
    )
    return dict(
        message_log=[message],
        length=len(ids),
        idx=idx,
        task_name=row["task_name"],
        loss_multiplier=1.0,
        extra_env_info=row,
        stop_strings=[],
        vllm_content=tokenizer.decode([tokenizer.pad_token_id] + prompt_ids, skip_special_tokens=False),
        vllm_multi_modal_data={"image": image},
    )
