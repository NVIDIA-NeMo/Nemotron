"""Read Parse page manifests without interpreting task-specific labels."""

import json
from pathlib import Path


def load_pages(manifest):
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for row in rows:
        row["image"] = str((manifest.parent / row["image"]).absolute())
    return rows
