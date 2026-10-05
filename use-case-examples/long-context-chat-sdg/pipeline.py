#!/usr/bin/env python3
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

"""Orchestrator — query_prep -> conversation generation -> raw trajectories.

    Stage A (query_prep, offline, MiniLM only):
        queries.jsonl (query source) -> dedup -> cluster -> sample -> seeds.jsonl
    Stage B (generate, Data Designer):
        seeds.jsonl -> per-row multi-turn trajectory (HTTP or simulated retrieval)
                    -> experiments/<exp_name>/output/raw.jsonl

Judging/filtering into the final SFT set is a SEPARATE, re-runnable stage
(``evaluate.py``) so you can re-judge without regenerating.

    python pipeline.py --config config/pipeline.yaml --stage query_prep
    python pipeline.py --config config/pipeline.yaml --stage generate
    python pipeline.py --config config/pipeline.yaml --stage all --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from long_context_chat_sdg.query_prep import cluster_queries, dedup, sample  # noqa: E402


# engine-block keys that are I/O, not ConversationSimulatorConfig knobs (popped before splat)
_ENGINE_IO = {"input", "output", "column_name", "metadata_fields", "resume", "artifact_path",
              "max_parallel_workers", "buffer_size"}   # read directly in run_generate; not config knobs


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def exp_paths(cfg: Dict[str, Any], base: Path) -> Dict[str, Path]:
    """Every run artifact lives under <exp_root>/<exp_name>/. Paths derive from exp_name
    (no per-stage path config). The corpus (query_gen.chunks_path) stays external."""
    root = _resolve(base, str(cfg.get("exp_root", "../experiments")))
    exp = root / str(cfg.get("exp_name") or "default")
    out = exp / "output"
    return {"exp": exp, "output": out, "artifacts": exp / "artifacts",
            "queries": out / "queries.jsonl", "seeds": out / "seeds.jsonl",
            "raw": out / "raw.jsonl", "sft": out / "sft.jsonl", "summary": out / "summary.json",
            "eval_details": out / "eval_details.jsonl",
            "generation_rejections": out / "generation_rejections.jsonl"}


def _atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Replace a JSONL artifact atomically so a crash cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as f:
            tmp_name = f.name
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def _positive_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer of at least 1") from exc
    if limit < 1:
        raise argparse.ArgumentTypeError("must be at least 1; omit --limit to process all rows")
    return limit


def _validate_dry_run_scope(stage: str, dry_run: bool, has_query_source: bool) -> None:
    if not dry_run:
        return
    if stage not in {"query_gen", "all"}:
        raise SystemExit("[pipeline] --dry-run is supported only for query_gen; no model calls were made.")
    if stage == "all" and not has_query_source:
        raise SystemExit("[pipeline] --stage all --dry-run needs a configured query_gen source; "
                         "generation dry-runs are not supported.")


def _build_model_clients(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    prov = {p["name"]: p for p in cfg.get("providers", [])}
    for m in cfg.get("models", []):
        p = prov.get(m.get("provider"), {})
        out[m["alias"]] = {"model": m["model"], "base_url": p.get("endpoint", ""),
                           "api_key_env": p.get("api_key", "NVIDIA_API_KEY"),
                           "params": dict(m.get("inference_parameters", {}))}
    return out


def validate_retrieval_config(cfg: Dict[str, Any]) -> str:
    """Return the retrieval mode or stop before an invalid generation run."""
    retrieval = cfg.get("retrieval", {}) or {}
    mode = str(retrieval.get("mode", "http")).lower()
    if mode not in {"http", "simulated"}:
        raise SystemExit("[generate] retrieval.mode must be 'http' or 'simulated'.")
    if mode == "http" and not str(retrieval.get("endpoint", "")).strip():
        raise SystemExit(
            "[generate] HTTP retrieval requires RETRIEVAL_ENDPOINT (or retrieval.endpoint). "
            "For a simulated demo run, pass --simulate-retrieval."
        )
    return mode


# ── Stage 0: query_gen (optional; corpus -> synthesized seed queries) ─────────
def run_query_gen(cfg: Dict[str, Any], base: Path, limit: Optional[int]) -> Path:
    qg = cfg.get("query_gen", {})
    source = qg.get("source", "lancedb")
    if source == "jsonl" and not qg.get("chunks_path"):
        raise SystemExit("[query_gen] source: jsonl needs query_gen.chunks_path (the corpus JSONL).")
    if source == "lancedb" and not qg.get("lancedb", {}).get("uri"):
        raise SystemExit("[query_gen] source: lancedb needs query_gen.lancedb.uri + table.")
    chunks_path = str(_resolve(base, qg["chunks_path"])) if qg.get("chunks_path") else ""
    if source == "jsonl" and not Path(chunks_path).is_file():
        raise SystemExit(f"[query_gen] corpus not found: {chunks_path}\n"
                         "  set query_gen.chunks_path to a readable JSONL corpus.")
    out_path = exp_paths(cfg, base)["queries"]
    models = _build_model_clients(cfg)

    client = None
    r = cfg.get("retrieval", {})
    if qg.get("validate", True) and r.get("endpoint"):
        from long_context_chat_sdg.retrieval.client import HttpRetrievalClient
        client = HttpRetrievalClient(r["endpoint"], oversample_factor=r.get("oversample_factor", 2),
                                     timeout=r.get("timeout", 30), field_map=r.get("field_map", {}),
                                     headers=r.get("headers") or None,
                                     max_retries=r.get("max_retries", 2))
    elif qg.get("validate", True):
        print("[query_gen] WARNING: validate requested but no retrieval.endpoint — "
              "queries will NOT be answerability-checked.")

    cl = qg.get("clustering", {})
    from long_context_chat_sdg.query_gen import run_query_gen as _run
    seeds = _run(models,
                 source=source,
                 lancedb_cfg=qg.get("lancedb"),
                 chunks_path=chunks_path,
                 coarse_cfg=qg.get("coarse"),
                 model_alias=qg.get("model_alias", "user_model"),
                 field_map=qg.get("field_map"),
                 n_queries=int(limit or qg.get("n_queries", 400)),
                 queries_per_cluster=int(qg.get("queries_per_cluster", 4)),
                 chunks_per_cluster=qg.get("chunks_per_cluster"),
                 pool_size=qg.get("pool_size"),          # None => derived from n_queries
                 candidate_headroom=float(qg.get("candidate_headroom", 1.5)),
                 max_rounds=int(qg.get("max_rounds", 3)),
                 embed_cfg=qg.get("embedding"),
                 cluster_algo=cl.get("algo", "kmeans"), n_clusters=cl.get("k"),
                 min_cluster_size=int(cl.get("min_cluster_size", 5)),
                 kind_weights=qg.get("kind_weights"),
                 multi_hop_chunks=int(qg.get("multi_hop_chunks", 3)),
                 cross_doc=bool(qg.get("cross_doc", True)),
                 client=client, top_k=int(r.get("top_k", 4)),
                 validate=qg.get("validate", True) and client is not None,
                 min_coverage=float(qg.get("min_coverage", 0.35)),
                 seed=int(qg.get("seed", 7)), max_workers=int(qg.get("max_workers", 8)),
                 dry_run=bool(qg.get("_dry_run", False)))
    if qg.get("_dry_run"):
        return out_path                                    # diagnostics only; nothing written
    if not seeds:                                          # never clobber input_path with an empty file
        raise SystemExit(f"[query_gen] produced 0 queries; refusing to overwrite {out_path}. "
                         "Check the source, endpoint, and validation settings.")

    _atomic_write_jsonl(out_path, seeds)
    print(f"[query_gen] wrote {len(seeds)} queries -> {out_path}")
    return out_path


# ── Stage A: query_prep ───────────────────────────────────────────────────────
def run_query_prep(cfg: Dict[str, Any], base: Path, limit: Optional[int]) -> Path:
    qp = cfg.get("query_prep", {})
    field = qp.get("query_field", "query")
    p = exp_paths(cfg, base)
    inp, seeds_path = p["queries"], p["seeds"]           # reads query_gen's queries; missing => error
    if not inp.exists():
        raise SystemExit(f"[query_prep] queries not found: {inp}\n"
                         "  run --stage query_gen first (or drop your queries there).")

    rows = [json.loads(l) for l in inp.open(encoding="utf-8") if l.strip()]
    rows = [r if isinstance(r, dict) else {field: r} for r in rows]
    print(f"[query_prep] loaded {len(rows)} queries from {inp}")

    dd_cfg = qp.get("dedup", {})
    kept = dedup(rows, threshold=dd_cfg.get("threshold", 0.92),
                 model_name=dd_cfg.get("embed_model", ""), query_field=field)
    print(f"[query_prep] deduped -> {len(kept)}")

    cl = qp.get("clustering", {})
    labels = cluster_queries(kept, algo=cl.get("algo", "kmeans"), k=cl.get("k"),
                             min_cluster_size=cl.get("min_cluster_size", 5),
                             model_name=dd_cfg.get("embed_model", ""), query_field=field)
    for r, lab in zip(kept, labels):
        r["cluster_id"] = f"c{lab:03d}"
    n_clusters = len(set(labels))
    print(f"[query_prep] clustered -> {n_clusters} clusters")

    sm = qp.get("sample", {})
    n_target = limit or sm.get("n_target")
    picked = sample(kept, n_target, seed=sm.get("seed", 7))
    print(f"[query_prep] sampled -> {len(picked)}")

    # Build seed rows. When persona.hf_dataset / local_path is set, attach one
    # persona per seed here (generator reads the "persona" column). Otherwise the
    # DD Person sampler may fill it at generate time.
    tools_json = json.dumps(cfg["tools"], ensure_ascii=False)
    seeds: List[Dict[str, Any]] = [
        {
            **r,
            "query": str(r.get(field, "")),
            "cluster_id": r.get("cluster_id", ""),
            "kind": r.get("kind", ""),  # -> planner opening kind (vague => clarify)
            "tools": tools_json,
        }
        for r in picked
    ]
    pcfg = cfg.get("persona", {}) or {}
    if pcfg.get("enabled", True):
        from long_context_chat_sdg.core.persona_loader import attach_personas_to_seeds, persona_source_is_external
        if persona_source_is_external(pcfg):
            seeds = attach_personas_to_seeds(seeds, pcfg, base=base)

    _atomic_write_jsonl(seeds_path, seeds)
    print(f"[query_prep] wrote {len(seeds)} seeds -> {seeds_path}")
    return seeds_path


# ── Stage B: conversation generation (Data Designer) ──────────────────────────
def build_config_builder(cfg: Dict[str, Any], seed_path: Path):
    import data_designer.config as dd
    from long_context_chat_sdg.conversation.config import ConversationSimulatorConfig
    from long_context_chat_sdg.core.persona_loader import persona_source_is_external

    model_configs = [
        dd.ModelConfig(alias=m["alias"], model=m["model"], provider=m.get("provider", "nvidia"),
                       skip_health_check=True,
                       inference_parameters=dd.ChatCompletionInferenceParams(
                           **dict(m.get("inference_parameters", {}))))
        for m in cfg["models"]
    ]

    cb = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    cb.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)),
                         sampling_strategy=dd.SamplingStrategy.SHUFFLE)

    # Persona source priority:
    #   1) persona.hf_dataset / local_path — already on seeds from query_prep; skip DD sampler
    #   2) DD Person sampler (NGC-managed locales) when enabled and no external source
    pcfg = cfg.get("persona", {}) or {}
    if pcfg.get("enabled", True) and not persona_source_is_external(pcfg):
        # Locale must be downloaded once: `data-designer download personas --locale <locale>`.
        cb.add_column(dd.SamplerColumnConfig(
            name="persona",                              # matches ConversationSimulatorConfig.persona_column
            sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(
                locale=pcfg.get("locale", "en_IN"),
                with_synthetic_personas=bool(pcfg.get("with_synthetic_personas", True)))))
    elif persona_source_is_external(pcfg):
        print("[generate] using personas from seeds (hf_dataset/local_path); DD Person sampler off")
    r = cfg.get("retrieval", {})
    eng = cfg.get("engine", {})
    knobs: Dict[str, Any] = {k: v for k, v in eng.items() if k not in _ENGINE_IO}   # I/O keys aren't config knobs
    knobs.update(retrieval_mode=r.get("mode", "http"),
                 retrieval_endpoint=r.get("endpoint", ""), retrieval_tools=r.get("tools", ["search"]),
                 retrieval_field_map=r.get("field_map", {}), retrieval_timeout=r.get("timeout", 30),
                 retrieval_max_retries=r.get("max_retries", 2),
                 retrieval_headers=r.get("headers", {}), top_k=r.get("top_k", 4),
                 oversample_factor=r.get("oversample_factor", 2))
    knobs["model_clients"] = _build_model_clients(cfg)
    name = eng.get("column_name", cfg.get("column_name", "conversation_messages"))
    cb.add_column(ConversationSimulatorConfig(name=name, **knobs))
    return cb


def _prepare_generated_rows(records: List[Dict[str, Any]], cfg: Dict[str, Any],
                            retrieval_mode: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert DD records without silently discarding malformed/empty trajectories."""
    tools = cfg["tools"]
    eng = cfg.get("engine", {})
    meta = eng.get("metadata_fields", cfg.get("metadata_fields",
                   ["kind", "difficulty", "cluster_id", "hops_taken",
                    "conversation_status", "trajectory_judgment", "retrieval_log"]))
    rows, rejections = [], []
    for index, rec in enumerate(records):
        src = rec.get("conversation_messages")
        try:
            messages = json.loads(src) if isinstance(src, str) else src
        except (json.JSONDecodeError, TypeError) as exc:
            rejections.append({"record_index": index, "reason": "invalid_conversation_messages",
                               "detail": str(exc)})
            continue
        if not isinstance(messages, list) or not messages:
            rejections.append({"record_index": index, "reason": "empty_conversation_messages",
                               "conversation_status": rec.get("conversation_status")})
            continue
        row = {"messages": messages, "tools": tools,
               "retrieval_mode": rec.get("retrieval_mode", retrieval_mode),
               "retrieval_tools": list((cfg.get("retrieval", {}) or {}).get("tools", ["search"]))}
        for mf in meta:
            if mf in rec:
                row[mf] = rec[mf]
        rows.append(row)
    return rows, rejections


def run_generate(cfg: Dict[str, Any], base: Path, seed_path: Path, limit: Optional[int]) -> Path:
    import data_designer.config as dd
    try:
        # Some Data Designer builds re-export this alongside RunConfig.
        from data_designer.config.run_config import ResumeMode
    except ImportError:
        # Data Designer 0.7.0 publishes ResumeMode from the storage API.
        from data_designer.engine.storage.artifact_storage import ResumeMode
    from data_designer.interface import DataDesigner

    retrieval_mode = validate_retrieval_config(cfg)
    eng = cfg.get("engine", {})
    cb = build_config_builder(cfg, seed_path)
    providers = [dd.ModelProvider(**p) for p in cfg.get("providers", [])]

    # resume from DD's per-row-group checkpoint (stable artifact_path + dataset_name).
    # if_possible: resume when the config matches, else restart. buffer_size left default
    # on both runs so it lines up.
    resume = ResumeMode(str(eng.get("resume", "never")).lower())
    artifact_path = exp_paths(cfg, base)["artifacts"] / "generate"
    artifact_path.mkdir(parents=True, exist_ok=True)
    client = DataDesigner(model_providers=providers, artifact_path=artifact_path) if providers \
        else DataDesigner(artifact_path=artifact_path)

    n_seeds = sum(1 for l in seed_path.open(encoding="utf-8") if l.strip())
    if limit == 0:
        raise SystemExit("[generate] --limit 0 selects no rows; omit --limit to process all rows.")
    n = min(limit, n_seeds) if limit is not None else n_seeds
    if n == 0:
        raise SystemExit(f"[generate] no non-empty seeds in {seed_path}.")

    # parallel trajectories + row-group size. buffer_size default = num_records/10
    # (incremental checkpoints), floored at workers so the pool doesn't starve.
    from data_designer.config.run_config import RunConfig
    workers = int(eng.get("max_parallel_workers", 4))
    buffer_size = int(eng.get("buffer_size") or max(workers, n // 10))
    client.set_run_config(RunConfig(non_inference_max_parallel_workers=workers, buffer_size=buffer_size))
    print(f"[generate] workers={workers} buffer_size={buffer_size} ({n} records)")

    if resume != ResumeMode.NEVER:
        print(f"[generate] resume={resume.value} artifacts={artifact_path} (re-run this stage to resume)")
    result = client.create(cb, num_records=n, dataset_name="long_context_chat_sdg", resume=resume)
    records = _records(result)

    paths = exp_paths(cfg, base)
    out = paths["raw"]
    rows, rejections = _prepare_generated_rows(records, cfg, retrieval_mode)

    if len(records) != n:
        rejections.append({"reason": "generated_record_count_mismatch", "expected": n,
                           "actual": len(records)})
    if rejections or len(rows) != n:
        _atomic_write_jsonl(paths["generation_rejections"], rejections)
        raise SystemExit(
            f"[generate] incomplete result: requested {n}, Data Designer returned {len(records)}, "
            f"and {len(rows)} valid trajectories. Previous raw/SFT artifacts were not changed; "
            f"details: {paths['generation_rejections']}"
        )

    _atomic_write_jsonl(out, rows)
    paths["generation_rejections"].unlink(missing_ok=True)
    # raw.jsonl has a new identity. Never leave derived artifacts that describe its
    # predecessor; eval checkpoints remain safe because evaluate.py keys them by input.
    invalidated = [p for p in (paths["sft"], paths["summary"], paths["eval_details"]) if p.exists()]
    for path in invalidated:
        path.unlink()
    if invalidated:
        print("[generate] invalidated stale derived artifacts: "
              + ", ".join(str(p) for p in invalidated))
    print(f"[generate] {len(records)} generated, {len(rows)} trajectories -> {out}")
    print("  next — judge & filter into the final SFT set:")
    print("    python evaluate.py --config config/pipeline.yaml --judge")
    return out


def _records(result) -> List[Dict[str, Any]]:
    if hasattr(result, "load_dataset"):
        ds = result.load_dataset()
        return ds.to_dict("records") if hasattr(ds, "to_dict") else list(ds)
    for attr in ("dataset", "records", "data"):
        obj = getattr(result, attr, None)
        if obj is not None:
            return obj.to_dict("records") if hasattr(obj, "to_dict") else list(obj)
    return list(result)


# ── driver ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="long_context_chat_sdg pipeline (query_prep + conversation gen).")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stage", choices=["query_gen", "query_prep", "generate", "all"], default="all")
    ap.add_argument("--limit", type=_positive_limit, default=None,
                    help="cap generated queries / sampled seeds / rows (must be >= 1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="query_gen only: embed+cluster+sample and print diagnostics, no LLM/tokens, no write")
    ap.add_argument("--resume", choices=["never", "if_possible", "always"], default=None,
                    help="generate stage: resume an interrupted run from its last completed "
                         "row-group checkpoint (overrides engine.resume in the config)")
    ap.add_argument("--simulate-retrieval", action="store_true",
                    help="explicitly use LLM-simulated search results instead of an HTTP retriever")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    base = args.config.resolve().parent
    _qg = cfg.get("query_gen", {})
    _has_source = _qg.get("lancedb", {}).get("uri") or _qg.get("chunks_path")
    _validate_dry_run_scope(args.stage, args.dry_run, bool(_has_source))
    if args.resume is not None:                         # CLI flag overrides engine.resume
        cfg.setdefault("engine", {})["resume"] = args.resume
    if args.simulate_retrieval:
        cfg.setdefault("retrieval", {})["mode"] = "simulated"
    if args.stage in ("generate", "all") and not args.dry_run:
        validate_retrieval_config(cfg)

    # all outputs live under experiments/<exp_name>/. Warn if it already exists (reuse
    # overwrites; change exp_name to keep both).
    P = exp_paths(cfg, base)
    if not args.dry_run:
        has_output = P["output"].exists() and any(P["output"].iterdir())
        has_artifacts = P["artifacts"].exists() and any(P["artifacts"].iterdir())
        if has_output or has_artifacts:
            print(f"[pipeline] ⚠️  experiment '{cfg.get('exp_name') or 'default'}' exists at {P['exp']} "
                  "— outputs will be overwritten (change exp_name to keep both).")
        P["output"].mkdir(parents=True, exist_ok=True)

    # Stage 0 (query_gen) is opt-in: explicit --stage query_gen, or "all" when a corpus
    # source (lancedb uri or chunks_path) is configured.
    if args.stage == "query_gen" or (args.stage == "all" and _has_source):
        if args.dry_run:
            cfg.setdefault("query_gen", {})["_dry_run"] = True
        run_query_gen(cfg, base, args.limit)            # writes query_gen.output; --limit caps it under `all`
        if args.stage == "query_gen" or args.dry_run:   # dry-run is diagnostics-only; never fall into generate
            return

    # `n_queries` is the single count knob: when query_gen feeds query_prep, it is
    # authoritative — query_prep dedups but must not silently re-cap below it.
    if _has_source:
        cfg.setdefault("query_prep", {}).setdefault("sample", {})["n_target"] = int(_qg.get("n_queries", 400))

    seeds_path = P["seeds"]                              # --stage generate reads seeds from the exp folder
    if args.stage in ("query_prep", "all"):
        seeds_path = run_query_prep(cfg, base, args.limit)
    if args.stage in ("generate", "all"):
        if not seeds_path.exists():
            raise SystemExit(f"[pipeline] no seeds at {seeds_path}; run --stage query_prep first.")
        run_generate(cfg, base, seeds_path, args.limit)


if __name__ == "__main__":
    main()
