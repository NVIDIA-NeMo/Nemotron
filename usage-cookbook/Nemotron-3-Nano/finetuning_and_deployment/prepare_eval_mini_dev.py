#!/usr/bin/env python3
"""
Build dataset/eval_mini_dev.jsonl from Hugging Face birdsql/bird_mini_dev.

Same input/output format as training (input = question + evidence, optionally schema;
output = gold SQL). Use this file for Step 8 eval so both NIMs (base and merged)
are evaluated on the same fixed dev set.

Schema: bird_mini_dev on HF has db_id per row but no schema. To include schema,
either set BIRD_MINI_DEV_DATA to the root that contains dev_databases/, or set
BIRD_DOWNLOAD_MINI_DEV=1 to have the script download the package from Google Drive
(via gdown) to dataset/bird_mini_dev_data and use it.

Run from this directory (e.g. in container or on host with `datasets` installed):
  python prepare_eval_mini_dev.py
  # With schema (manual path):
  BIRD_MINI_DEV_DATA=/path/to/mini_dev_data python prepare_eval_mini_dev.py
  # With schema (auto-download; requires: pip install gdown):
  BIRD_DOWNLOAD_MINI_DEV=1 python prepare_eval_mini_dev.py

Output: dataset/eval_mini_dev.jsonl (one {"input": "...", "output": "..."} per line).
"""

import json
import os
import sys

# Ensure bird_schema is importable when run from another cwd (e.g. notebook)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Google Drive file ID for BIRD Mini-Dev complete package (from bird-bench/mini_dev README)
BIRD_MINI_DEV_DRIVE_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"


# Directories to skip when searching for dev_databases (e.g. macOS zip metadata)
_RESOLVE_SKIP_DIRS = frozenset(("__MACOSX", ".DS_Store"))
# Prefer SQLite MINIDEV over MINIDEV_mysql / MINIDEV_postgresql when present
_RESOLVE_PREFER = "MINIDEV"


def _resolve_mini_dev_data(root_dir, max_depth=4):
    """Return root path that contains dev_databases/, or None. Searches up to max_depth levels."""
    if not root_dir or not os.path.isdir(root_dir):
        return None
    if os.path.isdir(os.path.join(root_dir, "dev_databases")):
        return root_dir
    if max_depth <= 0:
        return None
    entries = sorted(os.listdir(root_dir))
    # If we have MINIDEV (SQLite) and others, check it first so we prefer schema from CSV/SQLite
    if _RESOLVE_PREFER in entries:
        entries = [_RESOLVE_PREFER] + [e for e in entries if e != _RESOLVE_PREFER]
    for name in entries:
        if name in _RESOLVE_SKIP_DIRS or name.startswith("."):
            continue
        sub = os.path.join(root_dir, name)
        if os.path.isdir(sub):
            found = _resolve_mini_dev_data(sub, max_depth - 1)
            if found:
                return found
    return None


def _download_mini_dev_package(cache_dir):
    """
    Download BIRD Mini-Dev complete package from Google Drive and unpack to cache_dir.
    Returns the root path that contains dev_databases/ (inside cache_dir), or None on failure.
    """
    try:
        import gdown
        import zipfile
    except ImportError:
        print("Warning: gdown not installed. Install with: pip install gdown")
        return None
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "bird_mini_dev.zip")
    if not os.path.isfile(zip_path):
        print("Downloading BIRD Mini-Dev package from Google Drive (may take a few minutes)...")
        try:
            gdown.download(id=BIRD_MINI_DEV_DRIVE_ID, output=zip_path, quiet=False, fuzzy=True)
        except Exception as e:
            print("Warning: download failed: %s" % (e,))
            return None
        if not os.path.isfile(zip_path):
            print("Warning: download did not produce %s" % (zip_path,))
            return None
    print("Unpacking %s..." % (zip_path,))
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)
    except Exception as e:
        print("Warning: unzip failed: %s" % (e,))
        return None
    root = _resolve_mini_dev_data(cache_dir)
    if root:
        print("Package ready at %s" % (root,))
    else:
        try:
            top = os.listdir(cache_dir)
            print("Warning: no dev_databases/ found under %s (top-level: %s)" % (cache_dir, top[:10]))
        except Exception:
            pass
    return root


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    # Optional: root of BIRD Mini-Dev complete package (contains dev_databases/)
    mini_dev_data = os.environ.get("BIRD_MINI_DEV_DATA", "").strip()
    if mini_dev_data and not os.path.isdir(mini_dev_data):
        print("Warning: BIRD_MINI_DEV_DATA=%r is not a directory; skipping schema." % (mini_dev_data,))
        mini_dev_data = ""
    if not mini_dev_data and os.environ.get("BIRD_DOWNLOAD_MINI_DEV", "").strip() == "1":
        cache_dir = os.path.join(_script_dir, "dataset", "bird_mini_dev_data")
        existing = _resolve_mini_dev_data(cache_dir)
        mini_dev_data = existing if existing else _download_mini_dev_package(cache_dir)
        if not mini_dev_data:
            print("Warning: could not obtain package; proceeding without schema.")

    # bird_mini_dev has splits: mini_dev_sqlite, mini_dev_mysql, mini_dev_pg (500 rows each)
    ds = load_dataset("birdsql/bird_mini_dev", split="mini_dev_sqlite")
    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_mini_dev.jsonl")

    # Build schema cache from complete package if available
    schema_cache = {}
    if mini_dev_data:
        try:
            from bird_schema import build_schema_cache
            db_ids = list({ds[i]["db_id"] for i in range(len(ds))})
            schema_cache = build_schema_cache(mini_dev_data, db_ids)
            with_schema = sum(1 for s in schema_cache.values() if s)
            print("Schema: loaded for %d/%d databases from %r" % (with_schema, len(db_ids), mini_dev_data))
        except Exception as e:
            print("Warning: could not load schema from %r: %s" % (mini_dev_data, e))

    rows = []
    for i in range(len(ds)):
        r = ds[i]
        question = r.get("question", "")
        evidence = r.get("evidence", "")
        db_id = r.get("db_id", "")
        schema = schema_cache.get(db_id, "") or r.get("schema", "")
        sql = r.get("SQL", r.get("sql", "")).strip()
        parts = [p for p in (schema, question, evidence) if p]
        user_input = "\n".join(parts).strip()
        rows.append({"input": user_input, "output": sql})

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Wrote %d examples to %s" % (len(rows), out_path))
    return out_path


if __name__ == "__main__":
    main()
