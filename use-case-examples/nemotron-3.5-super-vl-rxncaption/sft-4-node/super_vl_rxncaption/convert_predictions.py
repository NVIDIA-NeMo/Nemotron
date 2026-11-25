"""
Convert raw BIVP model predictions to evaluation format.

Takes the raw JSON output from model inference (keyed by filename, with
"response" containing the model's JSON string) and converts it to the
COCO-like format expected by metrics.py.

Requires a mapdict JSON that maps visual indices (1-based reading order)
to ground truth bounding box coordinates.

Usage:
    python convert_predictions.py \
        --predictions  raw_predictions.json \
        --mapdict      mapdicts/mapdict_gt.json \
        --ground_truth ground_truth/ground_truth.json \
        --output       converted_predictions.json

Input format (raw_predictions.json):
    {
      "filename.png": {
        "id": 0,
        "response": "[{\"reactants\": [{\"type\": \"mol\", \"index\": 1}], ...}]"
      },
      ...
    }

Output format: COCO-like JSON matching ground_truth.json schema, ready
for metrics.py.
"""

import argparse
import json
import re


def parse_response(raw):
    """Parse model JSON response, handling code fences."""
    if raw.startswith("ERROR"):
        return []
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def convert(pred_path, mapdict_path, gt_path, output_path):
    with open(pred_path) as f:
        raw_preds = json.load(f)
    with open(mapdict_path) as f:
        mapdict_list = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    # Build mapdict lookup: filename -> {yolo_idx_str -> info}
    map_lookup = {}
    for entry in mapdict_list:
        fname = entry["file_name"].split("/")[-1]
        map_lookup[fname] = entry.get("map_dict", {})

    # Start with GT schema (minus images)
    out = {k: ([] if k == "images" else v) for k, v in gt_data.items()}

    for fname, entry in raw_preds.items():
        rxns = parse_response(entry.get("response", ""))
        mapping = map_lookup.get(fname, {})

        new_img = {"file_name": fname, "bboxes": [], "reactions": []}

        # Populate mol bboxes from mapdict
        for yolo_idx_str, info in mapping.items():
            yolo_idx = int(yolo_idx_str)
            new_img["bboxes"].append(
                {
                    "id": yolo_idx - 1,
                    "category_id": 1,
                    "bbox": info["yolo_bbox"],
                }
            )

        max_id = max((int(k) for k in mapping), default=0) - 1

        # Convert reactions
        for rxn in rxns:
            if not isinstance(rxn, dict):
                continue
            new_rxn = {}
            for role in ("reactants", "conditions", "products"):
                new_rxn[role] = []
                for item in rxn.get(role, []):
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type", "")
                    if item_type == "mol":
                        raw_idx = item.get("index")
                        try:
                            idx = int(raw_idx)
                            new_rxn[role].append(idx - 1)  # 1-based -> 0-based
                        except (TypeError, ValueError):
                            pass
                    elif item_type in ("txt", "idt"):
                        max_id += 1
                        new_rxn[role].append(max_id)
                        new_img["bboxes"].append(
                            {
                                "id": max_id,
                                "category_id": 2,
                                "bbox": [0, 0, 0, 0],
                                "text": item.get("content", ""),
                            }
                        )
            new_img["reactions"].append(new_rxn)

        # Filter invalid indices
        valid_ids = set(b["id"] for b in new_img["bboxes"])
        for rxn in new_img["reactions"]:
            for role in ("reactants", "conditions", "products"):
                rxn[role] = [i for i in rxn[role] if i in valid_ids]

        out["images"].append(new_img)

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Converted {len(out['images'])} images -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert raw BIVP predictions to eval format")
    parser.add_argument("--predictions", required=True, help="Raw predictions JSON (from model inference)")
    parser.add_argument(
        "--mapdict", required=True, help="Mapdict JSON (mapdicts/mapdict_gt.json or mapdict_molyolo.json)"
    )
    parser.add_argument("--ground_truth", required=True, help="Ground truth JSON")
    parser.add_argument("--output", required=True, help="Output eval-format JSON")
    args = parser.parse_args()
    convert(args.predictions, args.mapdict, args.ground_truth, args.output)


if __name__ == "__main__":
    main()
