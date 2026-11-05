#!/usr/bin/env python3
"""Load the packaged RxnCaption BIVP data in NeMo AutoModel VLM format.

The extracted dataset stores ShareGPT-style ``messages`` plus a separate
``images`` list.  ``nemotron_omni_collate_fn`` instead consumes examples shaped
as::

    {"conversation": [
        {"role": "user", "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "..."},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "..."},
        ]},
    ]}

This adapter performs that conversion lazily, opens images only from
``__getitem__`` (so DataLoader workers do the I/O), removes empty text target
items, and filters known OCR-reasoning contamination by pattern.  The source
JSONL and PNG files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_TAG_RE = re.compile(r"(<image\s*>)", re.IGNORECASE)
CONTAMINATION_RE = re.compile(
    r"("
    r"<ctrl\d+>thought"
    r"|the image (?:shows|contains)"
    r"|determine content type"
    r"|according to (?:the )?(?:ocr )?(?:transcription )?rule"
    r"|however,?\s*rule\s*\d+"
    r")",
    re.IGNORECASE,
)
SPLIT_ALIASES = {"eval": "test", "evaluation": "test", "val": "validation"}
REACTION_FIELDS = ("reactants", "conditions", "products")


def _clean_target(
    target: str,
    *,
    remove_empty_text: bool,
) -> tuple[str, bool, int]:
    """Validate and clean one assistant target.

    Returns ``(serialized_target, contaminated, removed_empty_items)``.
    """
    try:
        reactions = json.loads(target)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant target is not valid JSON: {exc}") from exc
    if not isinstance(reactions, list):
        raise ValueError(f"assistant target must be a JSON list, got {type(reactions).__name__}")

    contaminated = False
    removed_empty = 0
    cleaned_reactions: list[dict[str, list[dict[str, Any]]]] = []

    for reaction_index, reaction in enumerate(reactions):
        if not isinstance(reaction, dict) or set(reaction) != set(REACTION_FIELDS):
            raise ValueError(f"reaction {reaction_index} must contain exactly {REACTION_FIELDS}; got {reaction!r}")

        cleaned_reaction: dict[str, list[dict[str, Any]]] = {}
        for field in REACTION_FIELDS:
            items = reaction[field]
            if not isinstance(items, list):
                raise ValueError(f"reaction {reaction_index}.{field} must be a list")

            cleaned_items: list[dict[str, Any]] = []
            for item_index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] must be an object")
                item_type = item.get("type")
                if item_type == "mol":
                    if set(item) != {"type", "index"} or not isinstance(item.get("index"), int):
                        raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] has invalid mol schema")
                elif item_type in {"txt", "idt"}:
                    if set(item) != {"type", "content"} or not isinstance(item.get("content"), str):
                        raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] has invalid text schema")
                    content = item["content"]
                    contaminated = contaminated or bool(CONTAMINATION_RE.search(content))
                    if remove_empty_text and not content.strip():
                        removed_empty += 1
                        continue
                else:
                    raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] has unknown type {item_type!r}")
                cleaned_items.append(dict(item))
            cleaned_reaction[field] = cleaned_items
        cleaned_reactions.append(cleaned_reaction)

    serialized = json.dumps(cleaned_reactions, ensure_ascii=False, separators=(",", ":"))
    return serialized, contaminated, removed_empty


def _split_user_prompt(prompt: str, image_path: Path) -> tuple[tuple[str, Any], ...]:
    """Replace the prompt's single ``<image>`` marker with an image content item."""
    parts = IMAGE_TAG_RE.split(prompt)
    image_tags = sum(1 for part in parts if IMAGE_TAG_RE.fullmatch(part or ""))
    if image_tags != 1:
        raise ValueError(f"expected exactly one <image> marker in user prompt, found {image_tags}")

    content: list[tuple[str, Any]] = []
    for part in parts:
        if not part:
            continue
        if IMAGE_TAG_RE.fullmatch(part):
            content.append(("image", image_path))
        elif part:
            content.append(("text", part))
    return tuple(content)


class RxnCaptionOmniDataset:
    """Map-style dataset that lazily returns Omni VLM conversations."""

    def __init__(
        self,
        path_or_dataset: str,
        split: str = "train",
        *,
        filter_contaminated: bool = True,
        remove_empty_text: bool = True,
        verify_images: bool = True,
        include_system_prompt: bool = False,
        validation_size: int = 0,
        validation_seed: int = 42,
        limit_dataset_samples: int | None = None,
    ) -> None:
        root = Path(path_or_dataset).expanduser()
        requested_split = SPLIT_ALIASES.get(split.lower(), split.lower())
        if requested_split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be train, validation, or test; got {split!r}")
        source_split = "train" if requested_split == "validation" else requested_split
        split_root = root / source_split
        jsonl_path = split_root / "dataset.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"RxnCaption split not found: {jsonl_path}. Expected <data-root>/{{train,test}}/dataset.jsonl."
            )

        self.root = root
        self.split = requested_split
        self.source_split = source_split
        self.include_system_prompt = include_system_prompt
        self.records: list[dict[str, Any]] = []
        stats: Counter[str] = Counter()

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                stats["source_rows"] += 1
                try:
                    row = json.loads(line)
                    record, contaminated, removed_empty = self._parse_row(
                        row,
                        split_root=split_root,
                        line_number=line_number,
                        remove_empty_text=remove_empty_text,
                        verify_images=verify_images,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{jsonl_path}:{line_number}: invalid RxnCaption example: {exc}") from exc

                if contaminated and filter_contaminated:
                    stats["filtered_contaminated_rows"] += 1
                    continue
                if contaminated:
                    stats["kept_contaminated_rows"] += 1
                stats["removed_empty_text_items"] += removed_empty
                self.records.append(record)

        stats["clean_rows_before_partition"] = len(self.records)
        if requested_split in {"train", "validation"} and validation_size:
            if validation_size >= len(self.records):
                raise ValueError(
                    f"validation_size={validation_size} must be smaller than "
                    f"the {len(self.records)} clean training rows"
                )

            def validation_rank(record: dict[str, Any]) -> bytes:
                value = f"{validation_seed}:{record['id']}".encode()
                return hashlib.sha256(value).digest()

            validation_ids = {record["id"] for record in sorted(self.records, key=validation_rank)[:validation_size]}
            if requested_split == "validation":
                self.records = [record for record in self.records if record["id"] in validation_ids]
            else:
                self.records = [record for record in self.records if record["id"] not in validation_ids]
            stats["reserved_validation_rows"] = validation_size
        elif requested_split == "validation":
            raise ValueError("split='validation' requires validation_size > 0")

        if limit_dataset_samples is not None:
            self.records = self.records[:limit_dataset_samples]
        stats["kept_rows"] = len(self.records)
        self.stats = dict(stats)
        if not self.records:
            raise ValueError(f"no usable examples loaded from {jsonl_path}")

    @staticmethod
    def _parse_row(
        row: dict[str, Any],
        *,
        split_root: Path,
        line_number: int,
        remove_empty_text: bool,
        verify_images: bool,
    ) -> tuple[dict[str, Any], bool, int]:
        messages = row["messages"]
        images = row["images"]
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError("images must contain exactly one path string")

        by_role: dict[str, list[str]] = {}
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("each message must be an object")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError("each message needs string role and content fields")
            by_role.setdefault(role, []).append(content)

        if len(by_role.get("user", [])) != 1 or len(by_role.get("assistant", [])) != 1:
            raise ValueError("expected exactly one user turn and one assistant turn")
        if len(by_role.get("system", [])) > 1:
            raise ValueError("expected at most one system turn")

        image_path = Path(images[0])
        if not image_path.is_absolute():
            image_path = split_root / image_path
        if verify_images and not image_path.is_file():
            raise FileNotFoundError(f"referenced image does not exist: {image_path}")

        cleaned_target, contaminated, removed_empty = _clean_target(
            by_role["assistant"][0],
            remove_empty_text=remove_empty_text,
        )
        record = {
            "id": image_path.stem,
            "source_line": line_number,
            "image_path": image_path,
            "system_prompt": by_role.get("system", [None])[0],
            "user_content": _split_user_prompt(by_role["user"][0], image_path),
            "target": cleaned_target,
        }
        return record, contaminated, removed_empty

    def __len__(self) -> int:
        return len(self.records)

    def _build_conversation(self, record: dict[str, Any], image: Any) -> list[dict[str, Any]]:
        conversation: list[dict[str, Any]] = []
        system_prompt = record["system_prompt"]
        if self.include_system_prompt and system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content = []
        for content_type, value in record["user_content"]:
            if content_type == "image":
                user_content.append({"type": "image", "image": image})
            else:
                user_content.append({"type": "text", "text": value})
        conversation.extend(
            [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": record["target"]}],
                },
            ]
        )
        return conversation

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required to load RxnCaption images. Install the NeMo AutoModel VLM media dependencies."
            ) from exc

        with Image.open(record["image_path"]) as opened:
            image = opened.convert("RGB")
        return {
            "id": record["id"],
            "source_line": record["source_line"],
            "conversation": self._build_conversation(record, image),
        }


def make_rxncaption_dataset(
    path_or_dataset: str,
    split: str = "train",
    filter_contaminated: bool = True,
    remove_empty_text: bool = True,
    verify_images: bool = True,
    include_system_prompt: bool = False,
    validation_size: int = 0,
    validation_seed: int = 42,
    limit_dataset_samples: int | None = None,
) -> RxnCaptionOmniDataset:
    """Dataset factory used by the LoRA and full-SFT recipe templates."""
    return RxnCaptionOmniDataset(
        path_or_dataset=path_or_dataset,
        split=split,
        filter_contaminated=filter_contaminated,
        remove_empty_text=remove_empty_text,
        verify_images=verify_images,
        include_system_prompt=include_system_prompt,
        validation_size=validation_size,
        validation_seed=validation_seed,
        limit_dataset_samples=limit_dataset_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the packaged RxnCaption dataset adapter.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", choices=("train", "test", "eval", "validation"), default="train")
    parser.add_argument("--keep-contaminated", action="store_true")
    parser.add_argument("--keep-empty-text", action="store_true")
    parser.add_argument("--no-verify-images", action="store_true")
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = make_rxncaption_dataset(
        args.data_dir,
        split=args.split,
        filter_contaminated=not args.keep_contaminated,
        remove_empty_text=not args.keep_empty_text,
        verify_images=not args.no_verify_images,
        validation_size=args.validation_size,
        validation_seed=args.validation_seed,
        limit_dataset_samples=args.limit,
    )
    print(
        json.dumps(
            {
                "data_dir": str(dataset.root),
                "split": dataset.split,
                "source_split": dataset.source_split,
                **dataset.stats,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
