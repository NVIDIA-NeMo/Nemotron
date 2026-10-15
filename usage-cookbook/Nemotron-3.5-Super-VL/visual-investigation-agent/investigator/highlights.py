# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw simple, hand-authored reader annotations over recorded evidence.

The model did not emit bounding boxes. Coordinates live in ``annotations.json``
beside the selected recording. They are presentation aids only: the agent never
saw them, and graders never use them.
"""

import io
import json
from PIL import Image, ImageDraw

LABEL = "Orange boxes were drawn after the run to guide you; Super VL saw the unmarked images."
COLOR = "#D55E00"


def load(run):
    """Read fixed presentation rectangles for a selected recording."""
    path = run.folder / "annotations.json" if run.folder else None
    if path and path.exists():
        return json.loads(path.read_text())
    return {}


def _selected(run, image_id, ids=None, limit=3):
    boxes = load(run).get(image_id, [])
    if ids:
        return [box for box in boxes if box["id"] in ids]
    return boxes[:limit]


def annotated(run, image_id, ids=None, limit=3):
    """Return the source image with a few post-run reader rectangles."""
    image = Image.open(run.image(image_id)).convert("RGB")
    boxes = _selected(run, image_id, ids, limit)
    if not boxes:
        return image
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x, y = box["x"], box["y"]
        right, bottom = x + box["width"], y + box["height"]
        draw.rectangle((x - 3, y - 3, right + 3, bottom + 3), outline=COLOR, width=4)
    return image


def crop(run, image_id, ids=None, limit=3, margin=28):
    """Crop around selected rectangles while retaining enough source context."""
    image = annotated(run, image_id, ids, limit)
    boxes = _selected(run, image_id, ids, limit)
    if not boxes:
        return image
    left = max(0, min(box["x"] for box in boxes) - margin)
    top = max(0, min(box["y"] for box in boxes) - margin)
    right = min(image.width, max(box["x"] + box["width"] for box in boxes) + margin)
    bottom = min(image.height, max(box["y"] + box["height"] for box in boxes) + margin)
    return image.crop((left, top, right, bottom))


def png_bytes(run, image_id, ids=None, crop_to_boxes=False, margin=28):
    buffer = io.BytesIO()
    image = crop(run, image_id, ids, margin=margin) if crop_to_boxes else annotated(run, image_id, ids)
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()
