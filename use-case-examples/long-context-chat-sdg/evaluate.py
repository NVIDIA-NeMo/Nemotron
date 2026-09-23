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

"""Judge/filter stage — a standalone DataDesigner pass over raw trajectories.

Decoupled from generation: run on any raw jsonl to (re)score without regenerating.
The expensive per-trajectory eval runs as ONE DD custom column, so DataDesigner
gives it concurrency + per-row-group checkpointing (crash-safe, resumable). Scoring:

  1. OBJECTIVE (deterministic): every tool call parses + validates against its
     schema; the trajectory made at least one retrieval call; it ends on a grounded
     (tool-free) answer; and no citation is fabricated.
  2. DEFECT GATE (optional, --judge): an LLM screens for binary train-harmful defects
     (unsupported claims / no real research / incoherent / request unresolved / user
     out of character) plus a soft 1-5 quality score.
  KEPT if it passes the objective gate and (when judging) NO defect fired and
  quality >= --min-quality. Thresholds are applied AFTER the DD pass, so re-thresholding
  reuses the cached judgments (resume) instead of re-judging.

    python evaluate.py --config config/pipeline.yaml --judge        # exp raw.jsonl -> sft.jsonl + summary.json
    python evaluate.py --input raw.jsonl --out sft.jsonl            # explicit paths, objective-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from long_context_chat_sdg.conversation.verifiers import ToolCallVerifier

# binary train-harmful defects; a row is rejected if ANY fire (see prompts.TRAJECTORY_RUBRIC_PROMPT)
DISQUALIFIERS = ["unsupported_claims", "no_real_research", "incoherent",
                 "request_unresolved", "user_out_of_character"]


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace an artifact atomically so interrupted evaluation keeps the old result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as f:
            tmp_name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _atomic_write_text(path, "".join(
        json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows
    ))


def _input_digest(rows: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _eval_cache_key(rows: List[Dict[str, Any]], *, judge: bool, model: str = "",
                    endpoint: str = "", api_key_env: str = "") -> str:
    """Namespace DD checkpoints by input and judge identity, never by threshold.

    The quality/overlap thresholds intentionally stay out of this key so an unchanged
    judged dataset can be re-thresholded without making another model call.
    """
    identity = {"input_sha256": _input_digest(rows), "judge": bool(judge)}
    if judge:
        identity.update(model=model, endpoint=endpoint, api_key_env=api_key_env)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"{'judge' if judge else 'objective'}-{digest}"


def _load(input_file: Optional[Path], raw_dir: Optional[Path]) -> List[Dict[str, Any]]:
    files = [input_file] if input_file else (sorted(raw_dir.glob("*.jsonl")) if raw_dir else [])
    rows: List[Dict[str, Any]] = []
    for fp in files:
        if not fp.is_file():
            raise SystemExit(f"[eval] input not found: {fp}")
        rows += [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]
    return rows


def _char_ngrams(text: str, n: int = 12) -> set:
    t = " ".join((text or "").split()).lower()
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else set()


def _evidence_overlap(row: Dict[str, Any], n: int = 12) -> float:
    """Grounding proxy: fraction of each answer's char n-grams also in the retrieved text."""
    evidence = _char_ngrams(" ".join(m.get("content", "") for m in row.get("messages", [])
                                     if m.get("role") == "tool"), n)
    if not evidence:
        return 0.0
    scores = []
    for m in row.get("messages", []):
        if m.get("role") == "assistant" and not m.get("tool_calls") and (m.get("content") or "").strip():
            ag = _char_ngrams(m["content"], n)
            if ag:
                scores.append(len(ag & evidence) / len(ag))
    return round(min(scores), 3) if scores else 0.0   # weakest answer sets the grounding


# Citation syntax requested by the assistant prompt. The fallback hash regex also
# catches ids mentioned in reasoning without brackets.
_HASH_ID = re.compile(r"h[0-9a-f]{12}")
_BRACKETED_ID = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:/-]{0,127})\]")


def _retrieved_ids(msgs: List[Dict[str, Any]]) -> set:
    ids: set = set()
    for message in msgs:
        if message.get("role") != "tool":
            continue
        content = message.get("content", "") or ""
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if isinstance(results, list):
            ids.update(str(item["id"]) for item in results
                       if isinstance(item, dict) and item.get("id") is not None)
        ids.update(_HASH_ID.findall(content))
    return ids


def _cited_ids(msgs: List[Dict[str, Any]], role_pred, field: str = "content") -> set:
    ids: set = set()
    for message in msgs:
        if not role_pred(message):
            continue
        content = message.get(field, "") or ""
        ids.update(_BRACKETED_ID.findall(content))
        ids.update(_HASH_ID.findall(content))
    return ids


def _citation_integrity(row: Dict[str, Any]) -> Dict[str, Any]:
    """Every chunk id CITED must have been RETRIEVED in this conversation — checked on
    both the answer (assistant content) and, since it's a kept training target, the CoT
    (assistant reasoning_content)."""
    msgs = row.get("messages", [])
    retrieved = _retrieved_ids(msgs)
    cited = _cited_ids(msgs, lambda m: m.get("role") == "assistant" and not m.get("tool_calls"))
    fabricated = sorted(cited - retrieved)
    r_cited = _cited_ids(msgs, lambda m: m.get("role") == "assistant", field="reasoning_content")
    r_fabricated = sorted(r_cited - retrieved)
    return {"cited_ids": len(cited), "fabricated_ids": len(fabricated),
            "fabricated": fabricated[:10], "citation_ok": not fabricated,
            "reasoning_cited_ids": len(r_cited), "reasoning_fabricated_ids": len(r_fabricated),
            "reasoning_fabricated": r_fabricated[:10], "reasoning_citation_ok": not r_fabricated}


def _objective(row: Dict[str, Any], verifier: ToolCallVerifier) -> Dict[str, Any]:
    msgs = row.get("messages", [])
    tools = row.get("tools", [])
    n_calls, invalid, retrievals, empty_retrievals, retrieval_results = 0, 0, 0, 0, 0
    retrieval_tool_names = set(row.get("retrieval_tools") or ["search"])
    retrieval_call_ids = set()
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            n_calls += 1
            if tc.get("function", {}).get("name") in retrieval_tool_names:
                retrieval_call_ids.add(str(tc.get("id", "")))
            try:
                json.loads(tc.get("function", {}).get("arguments", "{}"))
                ok, _ = verifier.verify_single(tc, tools)
            except (json.JSONDecodeError, TypeError):
                ok = False
            invalid += int(not ok)
    for m in msgs:
        if m.get("role") != "tool" or str(m.get("tool_call_id", "")) not in retrieval_call_ids:
            continue
        try:
            payload = json.loads(m.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        results = payload.get("results") if isinstance(payload, dict) else None
        usable = [item for item in results
                  if isinstance(item, dict) and str(item.get("text", "")).strip()] \
            if isinstance(results, list) else []
        if usable:
            retrievals += 1
            retrieval_results += len(usable)
        else:
            empty_retrievals += 1
    last = msgs[-1] if msgs else {}
    ends_on_answer = last.get("role") == "assistant" and not last.get("tool_calls") and bool(last.get("content"))
    cite = _citation_integrity(row)
    ok = (invalid == 0) and (n_calls > 0) and (retrievals > 0) and ends_on_answer and cite["citation_ok"]
    return {"tool_calls": n_calls, "invalid_calls": invalid, "retrievals": retrievals,
            "empty_retrievals": empty_retrievals, "retrieval_results": retrieval_results,
            "ends_on_answer": ends_on_answer, **cite, "objective_ok": ok}


def _judge_from_config(config_path):
    """(model, endpoint, api_key_env) for the judge_model alias in the config."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    models = {m.get("alias"): m for m in cfg.get("models", [])}
    provs = {p.get("name"): p for p in cfg.get("providers", [])}
    jm = models.get("judge_model", {})
    prov = provs.get(jm.get("provider"), {})
    return jm.get("model", ""), prov.get("endpoint", ""), prov.get("api_key", "NVIDIA_API_KEY")


def _make_judge(model: str, endpoint: str, api_key_env: str):
    from long_context_chat_sdg.core.caller import make_openai_caller
    from long_context_chat_sdg.core.messages import format_history_compact, format_tools_for_prompt
    from long_context_chat_sdg.conversation.prompts import JUDGE_SYSTEM_PROMPT, TRAJECTORY_RUBRIC_PROMPT
    caller = make_openai_caller(model, endpoint, api_key_env,
                                params={"temperature": 0.1, "max_tokens": 2000})

    def score(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # bound the conversation so it fits the judge model's context (some are 32k);
        # tool outputs are snipped, recent turns kept, so the judge still sees the flow.
        # wide tool snippets so the judge SEES the retrieved evidence (a tight snippet
        # truncates the chunk text/ids and makes faithful answers look unsupported).
        prompt = TRAJECTORY_RUBRIC_PROMPT.format(
            tools=format_tools_for_prompt(row.get("tools", [])),
            conversation=format_history_compact(row.get("messages", []), max_chars=180000, tool_snippet=4000))
        text = caller(JUDGE_SYSTEM_PROMPT, prompt)
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            return None
        try:
            d = json.loads(text[a:b + 1])
        except json.JSONDecodeError:
            return None
        dq = d.get("disqualifiers", {}) if isinstance(d.get("disqualifiers"), dict) else {}
        return {"disqualifiers": {k: bool(dq.get(k, False)) for k in DISQUALIFIERS},
                "quality": int(d.get("quality", 0) or 0), "notes": str(d.get("notes", ""))[:200]}

    return score


def _rubric_ok(scores: Optional[Dict[str, Any]], min_quality: int) -> bool:
    """Keep only if the judge returned a verdict, NO disqualifier fired, and the soft
    quality score clears the (low, reporting-oriented) floor."""
    if not scores:
        return False
    if any(scores.get("disqualifiers", {}).get(k) for k in DISQUALIFIERS):
        return False
    return scores.get("quality", 0) >= min_quality


# ── eval as a DataDesigner pass ────────────────────────────────────────────────
# One custom column computes the (expensive) objective gate + judge per trajectory;
# DataDesigner runs it concurrently and checkpoints it (crash-safe, resumable). The
# keep/drop THRESHOLDS are applied afterwards in Python, so re-thresholding reuses the
# cached judgments instead of re-judging.
from data_designer.config import custom_column_generator   # noqa: E402

_VERIFIER = ToolCallVerifier()
_JUDGE = None                                    # set in main() when --judge


def _as_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else (json.loads(v) if isinstance(v, str) and v.strip() else [])


@custom_column_generator(required_columns=["messages", "tools", "retrieval_tools"])
def _eval_column(row):
    """Per-trajectory eval detail (objective gate + grounding overlap + judge verdict),
    stored as a JSON string in the `eval` column."""
    r = {"messages": _as_list(row["messages"]), "tools": _as_list(row.get("tools")),
         "retrieval_tools": _as_list(row.get("retrieval_tools")) or ["search"]}
    obj = _objective(r, _VERIFIER)
    scores = _JUDGE(r) if (_JUDGE and obj["objective_ok"]) else None
    row["eval"] = json.dumps({**obj, "grounding_overlap": _evidence_overlap(r), "rubric": scores},
                             ensure_ascii=False, default=str)
    return row


def _records(result) -> List[Dict[str, Any]]:
    ds = result.load_dataset()
    return ds.to_dict("records") if hasattr(ds, "to_dict") else list(ds)


def _for_sft(row: Dict[str, Any], *, keep_reasoning: bool) -> Dict[str, Any]:
    """The row as written to the SFT set: drop the eval column. Assistant reasoning_content
    is kept by default (the CoT is a training target); --strip-reasoning drops it."""
    out = {k: v for k, v in row.items() if k != "eval"}
    if not keep_reasoning and isinstance(out.get("messages"), list):
        out["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"}
                           if isinstance(m, dict) else m for m in out["messages"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge raw trajectories (a DataDesigner pass) -> filtered SFT set.")
    ap.add_argument("--input", type=Path, default=None, help="raw jsonl (default: exp raw.jsonl from --config)")
    ap.add_argument("--out", type=Path, default=None, help="SFT set to write (default: exp sft.jsonl)")
    ap.add_argument("--judge", action="store_true", help="run the LLM defect gate (else objective-only)")
    ap.add_argument("--min-quality", type=int, default=3, help="soft quality floor (1-5)")
    ap.add_argument("--min-overlap", type=float, default=0.0,
                    help="min answer<->evidence char-ngram overlap (0 = report only; the metric is a weak "
                         "exact-substring proxy — median ~0.11 — so gate low, ~0.02-0.03, and calibrate per corpus)")
    ap.add_argument("--strip-reasoning", action="store_true",
                    help="drop assistant reasoning_content from the SFT rows (default: keep — the CoT is a training target)")
    # judge model/endpoint: from --config's judge_model, or explicit flags (which win)
    ap.add_argument("--config", type=Path, default=None, help="pipeline.yaml — resolves the judge_model + exp paths")
    ap.add_argument("--model", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16, help="concurrent judge calls")
    ap.add_argument("--resume", choices=["never", "if_possible", "always"], default="if_possible",
                    help="reuse DD's judge checkpoint (crash recovery; re-thresholding reuses cache)")
    args = ap.parse_args()

    # resolve the judge (explicit flags win; else --config's judge_model)
    global _JUDGE
    _JUDGE = None
    judge_model = judge_endpoint = judge_key = ""
    if args.judge:
        model, endpoint, key = args.model, args.endpoint, args.api_key_env
        if args.config and not (model and endpoint and key):
            cm, ce, ck = _judge_from_config(args.config)
            model, endpoint, key = model or cm, endpoint or ce, key or ck
        if not (model and endpoint):
            raise SystemExit("[eval] --judge needs a judge model + endpoint. Set a judge_model in "
                             "--config, or pass --model/--endpoint. (No silent hosted fallback.)")
        judge_model, judge_endpoint, judge_key = model, endpoint, key or "NVIDIA_API_KEY"
        _JUDGE = _make_judge(judge_model, judge_endpoint, judge_key)

    # paths from --config's exp_name (raw.jsonl -> sft.jsonl + summary.json), or explicit --input/--out
    input_path, out_path, summary_path, artifacts = args.input, args.out, None, None
    if args.config:
        from omegaconf import OmegaConf
        from pipeline import exp_paths
        P = exp_paths(OmegaConf.to_container(OmegaConf.load(args.config), resolve=True), args.config.resolve().parent)
        input_path, out_path = input_path or P["raw"], out_path or P["sft"]
        summary_path, artifacts = P["summary"], P["artifacts"] / "eval"
    if not (input_path and out_path):
        raise SystemExit("[eval] need --config (for exp paths) or explicit --input/--out")
    summary_path = summary_path or out_path.with_suffix(".summary.json")
    details_path = (summary_path.parent / "eval_details.jsonl" if args.config
                    else out_path.with_suffix(".eval.jsonl"))
    artifacts = artifacts or (out_path.parent / "eval_artifacts")

    raw = _load(input_path, None)
    if args.limit is not None:
        raw = raw[: args.limit]
    if not raw:
        raise SystemExit(f"[eval] no trajectories in {input_path}")

    input_sha256 = _input_digest(raw)
    cache_key = _eval_cache_key(raw, judge=args.judge, model=judge_model,
                                endpoint=judge_endpoint, api_key_env=judge_key)
    artifacts = artifacts / cache_key

    # the eval column, run + checkpointed by DataDesigner
    import data_designer.config as dd
    from data_designer.config.run_config import RunConfig
    try:
        # Some Data Designer builds re-export this alongside RunConfig.
        from data_designer.config.run_config import ResumeMode
    except ImportError:
        # Data Designer 0.7.0 publishes ResumeMode from the storage API.
        from data_designer.engine.storage.artifact_storage import ResumeMode
    from data_designer.interface import DataDesigner
    print(f"[eval] {len(raw)} trajectories, judge={'on' if args.judge else 'off'}, "
          f"workers={args.workers}, cache={cache_key}")
    # Keep the seed at a stable, input-keyed path so DD can resume across threshold
    # changes. Atomic replacement also removes the old shared-/tmp collision hazard.
    seed_path = artifacts.parent / "_seeds" / f"{cache_key}.jsonl"
    _atomic_write_jsonl(seed_path, [
        {"row_id": i,
         "messages": json.dumps(r.get("messages", []), ensure_ascii=False),
         "tools": json.dumps(r.get("tools", []), ensure_ascii=False),
         "retrieval_tools": json.dumps(r.get("retrieval_tools", ["search"]), ensure_ascii=False)}
        for i, r in enumerate(raw)
    ])
    cb = dd.DataDesignerConfigBuilder(model_configs=[])
    cb.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)),
                         sampling_strategy=dd.SamplingStrategy.SHUFFLE)
    cb.add_column(dd.CustomColumnConfig(name="eval", generator_function=_eval_column))
    client = DataDesigner(artifact_path=str(artifacts))
    client.set_run_config(RunConfig(non_inference_max_parallel_workers=max(1, args.workers)))
    result = client.create(cb, num_records=len(raw), dataset_name="long_context_chat_sdg_eval",
                           resume=ResumeMode(args.resume))
    by_id = {int(rec["row_id"]): rec for rec in _records(result)}
    missing_ids = sorted(set(range(len(raw))) - set(by_id))
    if missing_ids:
        raise SystemExit(f"[eval] checkpoint returned {len(by_id)}/{len(raw)} rows; "
                         f"missing row ids: {missing_ids[:10]}")

    # apply thresholds in Python (re-tunable without re-judging), then filter + score
    kept, scored = [], []
    for i, r in enumerate(raw):
        ev = by_id.get(i, {}).get("eval")
        ev = json.loads(ev) if isinstance(ev, str) else (ev or {})
        keep = (ev.get("objective_ok") and ev.get("grounding_overlap", 0.0) >= args.min_overlap
                and (args.strip_reasoning or ev.get("reasoning_citation_ok", True))
                and (not args.judge or _rubric_ok(ev.get("rubric"), args.min_quality)))
        scored.append({**r, "eval": {**ev, "kept": bool(keep)}})   # in-memory only (for the summary)
        if keep:
            kept.append(_for_sft(r, keep_reasoning=not args.strip_reasoning))

    objective_pass = sum(1 for row in scored if row["eval"].get("objective_ok"))
    judged_rows = sum(1 for row in scored if row["eval"].get("rubric") is not None)
    if args.judge and judged_rows != objective_pass:
        raise SystemExit(f"[eval] judge produced {judged_rows}/{objective_pass} required verdicts; "
                         "refusing to publish an incomplete judged dataset.")

    args.input_sha256 = input_sha256
    args.eval_cache_key = cache_key
    args.judge_model = judge_model or None
    _atomic_write_jsonl(out_path, kept)
    _atomic_write_jsonl(details_path, [
        {"row_id": i, "eval": row["eval"]} for i, row in enumerate(scored)
    ])
    summary = _summary(scored, kept, args)
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[eval] kept {len(kept)}/{len(raw)} -> {out_path}")


def _role_seq(msgs: List[Dict[str, Any]]) -> str:
    return "".join((m.get("role") or "?")[0].upper() for m in msgs)   # S/U/A/T


def _ctx_tokens(msgs: List[Dict[str, Any]]) -> int:
    return sum(len(m.get("content", "") or "") + len(m.get("reasoning_content", "") or "") for m in msgs) // 4


def _dist(vals: List[int]) -> Dict[str, Any]:
    v = sorted(vals)
    return {"min": v[0], "p25": v[len(v) // 4], "median": v[len(v) // 2],
            "p75": v[(3 * len(v)) // 4], "max": v[-1], "mean": round(sum(v) / len(v))}


def _summary(scored: List[Dict[str, Any]], kept: List[Dict[str, Any]], args) -> Dict[str, Any]:
    n = len(scored)
    objective_pass = sum(1 for r in scored if r["eval"].get("objective_ok"))
    rub = [r["eval"]["rubric"] for r in scored if r["eval"].get("rubric")]
    judged_rows = len(rub)
    judge_requested = bool(args.judge)
    out: Dict[str, Any] = {
        "total": n,
        "objective_pass": objective_pass,
        "empty_evidence_rows": sum(1 for r in scored if r["eval"].get("empty_retrievals", 0)),
        "retrieval_results_total": sum(r["eval"].get("retrieval_results", 0) for r in scored),
        "citation_fabrication_rows": sum(1 for r in scored if not r["eval"].get("citation_ok", True)),
        "reasoning_fabrication_rows": sum(1 for r in scored if not r["eval"].get("reasoning_citation_ok", True)),
        "cited_ids_total": sum(r["eval"].get("cited_ids", 0) for r in scored),
        "kept": len(kept),
        "keep_rate": round(len(kept) / n, 3) if n else None,
        "kept_hops_distribution": dict(sorted(Counter(r.get("hops_taken", 0) for r in kept).items())),
        "retrieval_mode_counts": dict(sorted(Counter(
            str(r.get("retrieval_mode", "unknown")) for r in scored).items())),
        "kept_retrieval_mode_counts": dict(sorted(Counter(
            str(r.get("retrieval_mode", "unknown")) for r in kept).items())),
        "judge_requested": judge_requested,
        "judged": bool(judge_requested and judged_rows > 0 and judged_rows == objective_pass),
        "judged_rows": judged_rows,
        "strip_reasoning": bool(getattr(args, "strip_reasoning", False)),
        "reasoning_citation_gate_applied": not bool(getattr(args, "strip_reasoning", False)),
        "input_sha256": getattr(args, "input_sha256", None),
        "eval_cache_key": getattr(args, "eval_cache_key", None),
    }
    if kept:
        turns = [sum(1 for m in r["messages"] if m.get("role") == "user") for r in kept]
        out["turns_distribution"] = dict(sorted(Counter(turns).items()))
        out["top_flow_patterns"] = [{"pattern": p, "count": c}
                                    for p, c in Counter(_role_seq(r["messages"]) for r in kept).most_common(5)]
        out["context_length_tokens"] = _dist([_ctx_tokens(r["messages"]) for r in kept])
        out["context_length_estimator"] = "characters_divided_by_4"
    ov = sorted(r["eval"].get("grounding_overlap", 0.0) for r in scored)
    if ov:
        out["grounding_overlap"] = {"mean": round(sum(ov) / len(ov), 3),
                                    "median": ov[len(ov) // 2],
                                    "p10": ov[len(ov) // 10], "min": ov[0], "max": ov[-1]}
    if args.judge:
        j = judged_rows
        # how often each defect fired among judged rows (lower = cleaner data)
        out["disqualifier_rate"] = ({k: round(sum(bool(s["disqualifiers"].get(k)) for s in rub) / j, 3)
                                     for k in DISQUALIFIERS} if j else {})
        out["clean_rate"] = round(sum(not any(s["disqualifiers"].values()) for s in rub) / j, 3) if j else None
        out["quality_distribution"] = dict(sorted(Counter(s["quality"] for s in rub).items())) if j else {}
        out["mean_quality"] = round(sum(s["quality"] for s in rub) / j, 2) if j else None
    return out


if __name__ == "__main__":
    main()
