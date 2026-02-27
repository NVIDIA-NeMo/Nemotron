"""
Extract schema text for BIRD Mini-Dev databases from the complete package.

The BIRD Mini-Dev complete package (from Google Drive) contains:
  <root>/dev_databases/<db_id>/
    - database_description/  (CSV files describing schema)
    - *.sqlite or *.db       (SQLite database file, optional)

This module derives schema from (1) SQLite file via PRAGMA table_info if present,
or (2) database_description/*.csv (table = filename stem, columns = from CSV).

Usage:
  from bird_schema import get_schema_for_db_id, build_schema_cache
  schema = get_schema_for_db_id("debit_card_specializing", root_dir="/path/to/mini_dev_data")
"""

import csv
import os
import sqlite3
from typing import Optional


def _find_sqlite_in_db_dir(db_dir: str) -> Optional[str]:
    """Return path to first .sqlite or .db file under db_dir, or None."""
    if not os.path.isdir(db_dir):
        return None
    for name in os.listdir(db_dir):
        if name.endswith(".sqlite") or name.endswith(".db"):
            return os.path.join(db_dir, name)
    # Some layouts put the file in a subdir (e.g. database_content/)
    for sub in os.listdir(db_dir):
        subpath = os.path.join(db_dir, sub)
        if os.path.isdir(subpath):
            for name in os.listdir(subpath):
                if name.endswith(".sqlite") or name.endswith(".db"):
                    return os.path.join(subpath, name)
    return None


def _schema_from_sqlite(db_path: str) -> str:
    """Introspect SQLite file and return a concise schema string."""
    conn = sqlite3.connect(db_path)
    try:
        # Tables (SQLite 3.37+ has table_list; older fallback: sqlite_master)
        try:
            cur = conn.execute("PRAGMA table_list")
            tables = [row[1] for row in cur.fetchall()]
        except sqlite3.OperationalError:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [row[0] for row in cur.fetchall()]

        lines = []
        for table in tables:
            cur = conn.execute("PRAGMA table_info(%s)" % (table,))
            cols = cur.fetchall()
            # (cid, name, type, notnull, dflt_value, pk)
            col_strs = [f"{row[1]} ({row[2]})" for row in cols]
            lines.append("Table %s: %s" % (table, ", ".join(col_strs)))
        return "\n".join(lines) if lines else ""
    finally:
        conn.close()


def _schema_from_database_description(db_dir: str) -> str:
    """
    Build schema text from database_description/*.csv (BIRD format).
    Table name = CSV filename stem; columns = first column of CSV rows (original_column_name) or header.
    """
    desc_dir = os.path.join(db_dir, "database_description")
    if not os.path.isdir(desc_dir):
        return ""
    lines = []
    for name in sorted(os.listdir(desc_dir)):
        if not name.endswith(".csv"):
            continue
        table_name = os.path.splitext(name)[0]
        csv_path = os.path.join(desc_dir, name)
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                # BIRD description CSVs: header often has original_column_name, column_name, ...
                if header[0].strip().lower() == "original_column_name":
                    columns = [row[0].strip() for row in reader if row and row[0].strip()]
                else:
                    columns = [h.strip() for h in header if h.strip()]
                if columns:
                    lines.append("Table %s: %s" % (table_name, ", ".join(columns)))
        except (IOError, csv.Error):
            continue
    return "\n".join(lines) if lines else ""


def get_schema_for_db_id(db_id: str, root_dir: str) -> str:
    """
    Return schema text for the given db_id from the BIRD Mini-Dev complete package.

    root_dir: path to the package root that contains dev_databases/ (e.g. mini_dev_data
              or the unpacked folder from the Google Drive zip, e.g. minidev/MINIDEV).
    Tries (1) SQLite file under dev_databases/<db_id>/, then (2) database_description/*.csv.
    Returns empty string if schema cannot be determined.
    """
    dev_databases = os.path.join(root_dir, "dev_databases")
    db_dir = os.path.join(dev_databases, db_id)
    if not os.path.isdir(db_dir):
        return ""
    sqlite_path = _find_sqlite_in_db_dir(db_dir)
    if sqlite_path:
        try:
            return _schema_from_sqlite(sqlite_path)
        except (sqlite3.OperationalError, OSError):
            pass
    return _schema_from_database_description(db_dir)


def build_schema_cache(root_dir: str, db_ids: list) -> dict:
    """Build a dict mapping db_id -> schema text for the given db_ids."""
    cache = {}
    for db_id in db_ids:
        schema = get_schema_for_db_id(db_id, root_dir)
        cache[db_id] = schema
    return cache
