# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Load Nemotron-style personas from Hugging Face (or a local parquet/jsonl).

Config (``persona`` block in pipeline.yaml)::

    persona:
      enabled: true
      hf_dataset: nvidia/Nemotron-Personas-Vietnam   # repo id OR full HF URL
      # hf_split: train
      # local_path: ./personas.parquet               # skip download if set
      seed: 7

When ``hf_dataset`` / ``local_path`` is set, personas are attached to seeds at
``query_prep`` time and the DD Person sampler is skipped at ``generate``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

# Keys consumed by format_persona_for_prompt; region is remapped to state.
_PERSONA_KEYS = (
    "persona", "professional_persona", "finance_persona", "healthcare_persona",
    "sports_persona", "arts_persona", "travel_persona", "culinary_persona",
    "cultural_background", "skills_and_expertise", "hobbies_and_interests",
    "career_goals_and_ambitions", "sex", "age", "marital_status", "occupation",
    "education_level", "region", "city", "state", "country", "first_name",
    "last_name", "birth_date",
)

_HF_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?huggingface\.co/datasets/([^/\s?#]+/[^/\s?#]+)",
    re.IGNORECASE,
)


def parse_hf_dataset_id(value: str) -> str:
    """Accept a repo id (``org/name``) or a full Hugging Face dataset URL."""
    s = (value or "").strip().rstrip("/")
    if not s:
        raise ValueError("persona.hf_dataset is empty")
    m = _HF_URL_RE.search(s)
    if m:
        return m.group(1)
    # bare org/name (ignore query/fragment if pasted)
    path = urlparse(s).path if "://" in s else s
    path = path.strip("/")
    if path.startswith("datasets/"):
        path = path[len("datasets/"):]
    parts = path.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    raise ValueError(
        f"persona.hf_dataset must be 'org/name' or a HF dataset URL, got: {value!r}")


def normalize_persona_record(row: Dict[str, Any],
                             columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Keep formatter-relevant fields; map ``region`` -> ``state`` when needed."""
    keys = list(columns) if columns else list(_PERSONA_KEYS)
    out: Dict[str, Any] = {}
    for k in keys:
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if k == "age":
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = str(v)
        else:
            out[k] = v if isinstance(v, (dict, list, int, float, bool)) else str(v)
    if "state" not in out and out.get("region"):
        out["state"] = out["region"]
    return out


def _rows_from_dataframe(df: Any, columns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    # only pull columns that exist to avoid KeyError on locale-specific schemas
    want = [c for c in (columns or _PERSONA_KEYS) if c in df.columns]
    if not want:
        raise ValueError(
            f"persona dataset has none of the expected columns; got {list(df.columns)}")
    records = df[want].to_dict("records")
    return [normalize_persona_record(r, want) for r in records]


def load_personas_from_parquet(path: Path,
                               columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_parquet(path)
    return _rows_from_dataframe(df, columns)


def load_personas_from_jsonl(path: Path,
                             columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(normalize_persona_record(json.loads(line), columns))
    if not rows:
        raise ValueError(f"no persona rows in {path}")
    return rows


def load_personas_from_hf(dataset_id: str, *,
                         split: str = "train",
                         revision: Optional[str] = None,
                         columns: Optional[Sequence[str]] = None,
                         cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Download parquet shards via huggingface_hub (no ``datasets`` package required)."""
    repo_id = parse_hf_dataset_id(dataset_id)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "[persona] huggingface_hub is required to load persona.hf_dataset. "
            "Install it or set persona.local_path to a downloaded parquet/jsonl."
        ) from exc

    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        allow_patterns=["*.parquet"], cache_dir=cache_dir)
    parquets = sorted(Path(local_dir).rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"[persona] no parquet files found in HF dataset {repo_id}")

    import pandas as pd
    frames = [pd.read_parquet(p) for p in parquets]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    # optional split filter when a 'split' column exists (rare for Nemotron-Personas)
    if split and "split" in df.columns:
        df = df[df["split"] == split]
    return _rows_from_dataframe(df, columns)


def load_persona_pool(pcfg: Dict[str, Any], base: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Resolve persona.local_path / persona.hf_dataset into a list of persona dicts."""
    columns = pcfg.get("columns")
    local = pcfg.get("local_path") or pcfg.get("path")
    if local:
        path = Path(local)
        if not path.is_absolute() and base is not None:
            path = (base / path).resolve()
        if not path.exists():
            raise SystemExit(f"[persona] local_path not found: {path}")
        if path.suffix.lower() in {".jsonl", ".json"}:
            return load_personas_from_jsonl(path, columns)
        return load_personas_from_parquet(path, columns)

    hf = pcfg.get("hf_dataset") or pcfg.get("dataset") or pcfg.get("url")
    if hf:
        return load_personas_from_hf(
            str(hf),
            split=str(pcfg.get("hf_split", "train")),
            revision=pcfg.get("hf_revision"),
            columns=columns,
            cache_dir=pcfg.get("cache_dir"),
        )
    raise SystemExit(
        "[persona] set persona.hf_dataset (HF repo id/URL) or persona.local_path "
        "to load external personas.")


def sample_personas(pool: Sequence[Dict[str, Any]], n: int, seed: int = 7) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if not pool:
        raise SystemExit("[persona] empty persona pool")
    rng = random.Random(seed)
    if n <= len(pool):
        return [dict(p) for p in rng.sample(list(pool), n)]
    # with replacement when requesting more seeds than personas
    return [dict(pool[rng.randrange(len(pool))]) for _ in range(n)]


def attach_personas_to_seeds(seeds: List[Dict[str, Any]],
                             pcfg: Dict[str, Any],
                             base: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Attach one sampled persona (JSON string) per seed under the ``persona`` column."""
    pool = load_persona_pool(pcfg, base=base)
    seed = int(pcfg.get("seed", 7))
    picks = sample_personas(pool, len(seeds), seed=seed)
    col = str(pcfg.get("column", "persona"))
    out: List[Dict[str, Any]] = []
    for row, persona in zip(seeds, picks):
        r = dict(row)
        r[col] = json.dumps(persona, ensure_ascii=False)
        out.append(r)
    src = pcfg.get("local_path") or pcfg.get("hf_dataset") or pcfg.get("dataset") or "?"
    print(f"[persona] attached {len(out)} personas from {src} (pool={len(pool)}, seed={seed})")
    return out


def persona_source_is_external(pcfg: Dict[str, Any]) -> bool:
    """True when personas should be loaded from HF/local (not DD Person sampler)."""
    return bool(pcfg.get("hf_dataset") or pcfg.get("dataset") or pcfg.get("url")
                or pcfg.get("local_path") or pcfg.get("path"))
