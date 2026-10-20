# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create original, deterministic text/image/paired data for execution checks.

The fictional facts and rendered images are part of this Apache-2.0 fixture.
Queries share the corpus intentionally. This fixture is not a quality benchmark
and does not test hosted synthetic data generation.
"""

import json
from pathlib import Path


def write_fixture(root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    root.mkdir(parents=True, exist_ok=False)
    corpus = root / "corpus"
    corpus.mkdir()
    images = root / "images"
    images.mkdir()
    evaluation = root / "beir"
    (evaluation / "qrels").mkdir(parents=True)
    records, documents, queries, qrels, rows = [], [], [], [], []
    names = ("Amber", "Birch", "Cedar", "Dune", "Elm", "Flint", "Grove", "Harbor", "Iris", "Jade", "Kestrel", "Larch")
    for index, name in enumerate(names):
        identifier = name.lower()
        modality = ("text", "image", "paired")[index % 3]
        fact = f"The {name} Observatory gallery opens at {index + 8}:00."
        image_path, image_bytes = None, None
        if modality != "text":
            image_path = images / f"{identifier}.png"
            image = Image.new("RGB", (760, 160), "white")
            draw = ImageDraw.Draw(image)
            draw.text((20, 35), fact, fill="black", font_size=24)
            image.save(image_path)
            image_bytes = image_path.read_bytes()
        text = fact if modality != "image" else ""
        rows.append({"docid": identifier, "text": text, "image": image_bytes})
        document = {"_id": identifier, "title": "", "text": text}
        if image_path:
            document["image_path"] = str(image_path.relative_to(root))
        documents.append(document)
        records.append(
            {
                "question_id": f"train-{identifier}",
                "question": f"When does the {name} Observatory gallery open?",
                "corpus_id": "retrieval-smoke",
                "pos_doc": [{"id": identifier, "score": 1}],
                "neg_doc": [],
                "negative_mining_performed": False,
                "relevance_judgements_complete": False,
                "unlisted_document_disposition": "unjudged",
            }
        )
        queries.append({"_id": f"eval-{identifier}", "text": f"What is the opening time of the {name} gallery?"})
        qrels.append(f"eval-{identifier}\t{identifier}\t1\n")
    pq.write_table(pa.Table.from_pylist(rows), corpus / "part-00000.parquet")
    (corpus / "merlin_metadata.json").write_text(
        json.dumps(
            {
                "class": "WikiSSNQDataset",
                "corpus_id": "retrieval-smoke",
                "format": "parquet",
            }
        )
        + "\n"
    )
    (root / "train.json").write_text(json.dumps({"corpus": {"path": "corpus"}, "data": records}, indent=2) + "\n")
    for name, values in (("corpus", documents), ("queries", queries)):
        (evaluation / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    (evaluation / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(qrels))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_fixture(parser.parse_args().output.resolve())
