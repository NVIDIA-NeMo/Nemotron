#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/sft/automodel"
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 4
# ///
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

"""SFT Training via AutoModel-style HF Trainer — reference implementation.

Minimal working example of JSONL-based SFT with HuggingFace Trainer and
optional LoRA adapters. The agent reads this to understand the simpler,
small-GPU path and then writes a customer-specific stage around it.

Expected config shape:
  model.name: HF model id
  dataset.train_jsonl: path to training JSONL
  dataset.validation_jsonl: optional validation JSONL
  training.output_dir: checkpoint directory
  training.seq_length: max token length
  peft.method: lora | null

Usage:
    python step.py --config /path/to/config.yaml
    torchrun --nproc_per_node=4 step.py --config /path/to/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference AutoModel SFT step")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def render_text(example: dict[str, Any], tokenizer: Any, seq_length: int) -> dict[str, list[int]]:
    if "messages" in example:
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=seq_length,
            )
        else:
            text = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in example["messages"]
            )
            input_ids = tokenizer(text, truncation=True, max_length=seq_length)["input_ids"]
    else:
        prompt = example.get("prompt") or example.get("instruction") or ""
        completion = example.get("completion") or example.get("response") or ""
        input_ids = tokenizer(f"{prompt}{completion}", truncation=True, max_length=seq_length)["input_ids"]

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def maybe_wrap_with_lora(model: Any, peft_config: dict[str, Any]) -> Any:
    method = (peft_config.get("method") or "null").lower()
    if method != "lora":
        return model

    lora = LoraConfig(
        r=peft_config.get("r", 16),
        lora_alpha=peft_config.get("alpha", 32),
        lora_dropout=peft_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return get_peft_model(model, lora)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = config["model"]["name"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    peft_cfg = config.get("peft", {})
    seq_length = training_cfg.get("seq_length", 4096)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    model = maybe_wrap_with_lora(model, peft_cfg)

    data_files = {"train": dataset_cfg["train_jsonl"]}
    if dataset_cfg.get("validation_jsonl"):
        data_files["validation"] = dataset_cfg["validation_jsonl"]
    raw = load_dataset("json", data_files=data_files)

    train_dataset = raw["train"].map(
        lambda example: render_text(example, tokenizer, seq_length),
        remove_columns=raw["train"].column_names,
    )
    eval_dataset = None
    if "validation" in raw:
        eval_dataset = raw["validation"].map(
            lambda example: render_text(example, tokenizer, seq_length),
            remove_columns=raw["validation"].column_names,
        )

    args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        learning_rate=training_cfg.get("learning_rate", 2.0e-5),
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 100),
        eval_steps=training_cfg.get("eval_steps", training_cfg.get("save_steps", 100)),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        bf16=torch.cuda.is_available(),
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])


if __name__ == "__main__":
    main()
