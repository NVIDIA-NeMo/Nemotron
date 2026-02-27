#!/usr/bin/env python3
"""
Build dataset/eval_mini_dev.jsonl from Hugging Face birdsql/bird_mini_dev.

Same input/output format as training (input = question + evidence, optionally schema;
output = gold SQL). Use this file for Step 8 eval so both NIMs (base and merged)
are evaluated on the same fixed dev set.

Run from this directory (e.g. in container or on host with `datasets` installed):
  python prepare_eval_mini_dev.py

Output: dataset/eval_mini_dev.jsonl (one {"input": "...", "output": "..."} per line).
"""

import json
import os

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    # bird_mini_dev has splits: mini_dev_sqlite, mini_dev_mysql, mini_dev_pg (500 rows each)
    # Use SQLite version for a single canonical eval set
    ds = load_dataset("birdsql/bird_mini_dev", split="mini_dev_sqlite")
    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_mini_dev.jsonl")

    rows = []
    for i in range(len(ds)):
        r = ds[i]
        question = r.get("question", "")
        evidence = r.get("evidence", "")
        schema = r.get("schema", "")  # may be missing in mini_dev
        sql = r.get("SQL", r.get("sql", "")).strip()
        parts = [p for p in (schema, question, evidence) if p]
        user_input = "\n".join(parts).strip()
        rows.append({"input": user_input, "output": sql})

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
