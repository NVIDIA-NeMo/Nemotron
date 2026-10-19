#!/usr/bin/env python3
"""Prepare leakage-safe TAT-DQA + BizBench splits for NeMo AutoModel VLM SFT."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from PIL import Image


SPLITS = ("train", "validation", "test")
TAT_REVISION = "bc46208b22f0bfe47152c98fd0938d238ce868c2"
BIZ_REVISION = "0a793f2f886156902c72b4a22cab82bb9dceaecf"
BIZ_TASKS = ("CodeFinQA", "CodeTAT-QA")
TARGET_TEMPLATE = "<think>\n{trace}\n</think>\n{answer}"
TASK_DATASETS = {
    "visual_ocr": {
        "directory": "finance_visual_ocr_v1",
        "tasks": ("TAT-DQA/multi-span", "TAT-DQA/span"),
        "task_family": "visual_ocr",
        "input_mode": "image",
    },
    "reasoning": {
        "directory": "finance_reasoning_v1",
        "tasks": ("TAT-DQA/arithmetic", "TAT-DQA/count"),
        "task_family": "financial_reasoning",
        "input_mode": "text",
    },
    "code_generation": {
        "directory": "finance_code_generation_v1",
        "tasks": ("CodeFinQA", "CodeTAT-QA"),
        "task_family": "program_synthesis",
        "input_mode": "text",
    },
}
CAPABILITY_BY_TASK = {
    task: capability
    for capability, spec in TASK_DATASETS.items()
    for task in spec["tasks"]
}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def stable_hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8], "big")


def _answer_text(answer: object) -> str:
    if isinstance(answer, list):
        return " | ".join(str(value).strip() for value in answer)
    return str(answer).strip()


def _answer_with_scale(answer: object, scale: object) -> str:
    value = _answer_text(answer)
    unit = str(scale or "").strip().casefold()
    if not unit:
        return value
    if unit == "percent":
        return value if value.endswith("%") else f"{value}%"
    return value if normalize_text(value).endswith(unit) else f"{value} {unit}"


def _tat_trace(question: dict, answer: str) -> str:
    derivation = str(question.get("derivation") or "").strip()
    facts = [str(value).strip() for value in (question.get("facts") or []) if str(value).strip()]
    scale = str(question.get("scale") or "").strip() or "none"
    parts = []
    if facts:
        parts.append("Evidence values: " + " ; ".join(facts))
    if derivation:
        parts.append("Calculation: " + derivation)
    if not parts:
        parts.append("Read the document and extract the requested evidence.")
    parts.append(f"Answer scale: {scale}")
    parts.append(f"Final answer: {answer}")
    return "\n".join(parts)


def build_user_prompt(task: str, task_family: str, question: str, context: str = "") -> str:
    """Build the same task-tagged user text for training and evaluation."""
    instruction = (
        "Return a grounded reasoning trace inside <think>...</think>, followed by only the final answer. "
        "Preserve the requested financial unit or scale."
    )
    if task_family in {"visual_document_reasoning", "visual_ocr"}:
        return (
            f"Task: {task}\n{instruction}\n"
            "Read the financial document image and extract the requested evidence exactly.\n"
            f"Question: {question}"
        )
    if task_family == "financial_reasoning":
        return (
            f"Task: {task}\n{instruction}\n"
            "Use the supplied financial-document evidence to calculate or count the requested result.\n\n"
            f"Evidence:\n{context}\n\nQuestion:\n{question}"
        )
    return (
        f"Task: {task}\n{instruction}\n"
        "Write an executable Python calculation in the reasoning trace.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )


def _render_pdf_page(pdf_bytes: bytes, page_index: int, dpi: int) -> Image.Image:
    try:
        import fitz

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_index = min(max(page_index, 0), document.page_count - 1)
            pixmap = document.load_page(page_index).get_pixmap(dpi=dpi, alpha=False)
            return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        finally:
            document.close()
    except ImportError:
        executable = shutil.which("pdftoppm")
        if not executable:
            raise RuntimeError("High-resolution TAT-DQA rendering requires PyMuPDF or pdftoppm")
        with tempfile.TemporaryDirectory(prefix="tatdqa-render-") as temporary:
            pdf_path = Path(temporary) / "document.pdf"
            output_prefix = Path(temporary) / "page"
            pdf_path.write_bytes(pdf_bytes)
            subprocess.run([
                executable, "-f", str(page_index + 1), "-singlefile", "-r", str(dpi),
                "-png", str(pdf_path), str(output_prefix),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with Image.open(output_prefix.with_suffix(".png")) as rendered:
                return rendered.convert("RGB")


def _ocr_page(payload: dict, page_index: int) -> dict:
    pages = payload.get("pages") or []
    if not pages:
        return {"bbox": [0, 0, 1, 1], "blocks": []}
    return pages[min(max(page_index, 0), len(pages) - 1)]


def _evidence_crop(page: Image.Image, ocr_page: dict, question: dict) -> tuple[Image.Image, str]:
    blocks = ocr_page.get("blocks") or []
    by_uuid = {str(block.get("uuid")): block for block in blocks}
    mapped = []
    for item in question.get("block_mapping") or []:
        for block_id in item:
            if str(block_id) in by_uuid:
                mapped.append(by_uuid[str(block_id)])
    page_bbox = ocr_page.get("bbox") or [0, 0, page.width, page.height]
    source_width = max(float(page_bbox[2]) - float(page_bbox[0]), 1.0)
    source_height = max(float(page_bbox[3]) - float(page_bbox[1]), 1.0)
    if not mapped:
        evidence = "\n".join(str(block.get("text") or "").strip() for block in blocks if block.get("text"))
        return page.copy(), evidence

    top = max(0.0, min(float(block["bbox"][1]) for block in mapped) - 130.0)
    bottom = min(source_height, max(float(block["bbox"][3]) for block in mapped) + 100.0)
    if bottom - top < 420.0:
        extra = (420.0 - (bottom - top)) / 2.0
        top = max(0.0, top - extra)
        bottom = min(source_height, bottom + extra)
    selected = [
        block for block in blocks
        if block.get("bbox") and float(block["bbox"][3]) >= top and float(block["bbox"][1]) <= bottom
    ]
    evidence = "\n".join(str(block.get("text") or "").strip() for block in selected if block.get("text"))
    y0 = int(round(top / source_height * page.height))
    y1 = int(round(bottom / source_height * page.height))
    return page.crop((0, max(0, y0), page.width, min(page.height, max(y0 + 1, y1)))), evidence


def load_tat_records(source_dir: Path, media_dir: Path, *, render_dpi: int) -> list[dict]:
    records: list[dict] = []
    json_names = {
        "train": "tatdqa_dataset_train.json",
        "validation": "tatdqa_dataset_dev.json",
        "test": "tatdqa_dataset_test_gold.json",
    }
    archive_names = {"train": "train", "validation": "dev", "test": "test"}
    for split, json_name in json_names.items():
        path = source_dir / json_name
        if not path.is_file():
            raise FileNotFoundError(path)
        docs = json.loads(path.read_text())
        print(f"Preparing TAT-DQA {split}: {len(docs)} documents", flush=True)
        archive_split = archive_names[split]
        archive = source_dir / f"tatdqa_docs_{archive_split}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        destination = media_dir / archive_split
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for document_index, item in enumerate(docs, 1):
                if document_index == 1 or document_index % 100 == 0:
                    print(
                        f"TAT-DQA {split}: document {document_index}/{len(docs)}; "
                        f"{len(records)} records prepared across splits",
                        flush=True,
                    )
                doc = item["doc"]
                uid = str(doc["uid"])
                page_number = int(doc["page"])
                page_index = page_number - 1
                pdf_member = f"{archive_split}/{uid}.pdf"
                ocr_member = f"{archive_split}/{uid}.json"
                missing = {pdf_member, ocr_member} - names
                if missing:
                    raise FileNotFoundError(f"Missing TAT-DQA source assets: {sorted(missing)}")
                page_image = _render_pdf_page(zf.read(pdf_member), page_index, render_dpi)
                ocr = _ocr_page(json.loads(zf.read(ocr_member)), page_index)
                source = str(doc.get("source") or uid)
                for question in item["questions"]:
                    crop, ocr_context = _evidence_crop(page_image, ocr, question)
                    image_name = f"{uid}_{page_number}_{question['uid']}.png"
                    image_path = destination / image_name
                    if not image_path.is_file() or not image_path.stat().st_size:
                        crop.save(image_path, format="PNG", optimize=True)
                    crop.close()
                    if "answer" not in question:
                        raise ValueError(f"Gold answer missing for TAT-DQA {split} question {question.get('uid')}")
                    answer = _answer_with_scale(question["answer"], question.get("scale"))
                    if not answer or not str(question.get("question") or "").strip():
                        continue
                    trace = _tat_trace(question, answer)
                    records.append({
                        "sample_id": f"tatdqa:{question['uid']}",
                        "source_dataset": "TAT-DQA",
                        "source_split": split,
                        "task": f"TAT-DQA/{question.get('answer_type') or 'unknown'}",
                        "task_family": "visual_document_reasoning",
                        "lineage_key": f"tatdqa-report:{normalize_text(source)}",
                        "question_key": normalize_text(question["question"]),
                        "image_path": str(Path("data") / media_dir.name / archive_split / image_path.name),
                        "context": ocr_context,
                        "question": str(question["question"]).strip(),
                        "reasoning": trace,
                        "answer": answer,
                        "scale": str(question.get("scale") or "").strip(),
                        "program": "",
                        "target": TARGET_TEMPLATE.format(trace=trace, answer=answer),
                        "assigned_split": split,
                    })
                page_image.close()
    return records


def _biz_lineage(context: str, question: str) -> str:
    material = normalize_text(context) or normalize_text(question)
    return "bizbench-context:" + hashlib.sha256(material.encode()).hexdigest()


def _bizbench_parquet(source_dir: Path, split: str) -> Path:
    """Resolve either the normalized local name or Hugging Face snapshot layout."""
    normalized = source_dir / f"bizbench_{split}.parquet"
    if normalized.is_file():
        return normalized
    matches = sorted((source_dir / "data").glob(f"{split}-*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one BizBench {split} parquet under {source_dir / 'data'} "
            f"or {normalized}; found {matches}"
        )
    return matches[0]


def load_biz_records(source_dir: Path) -> list[dict]:
    records: list[dict] = []
    for split in ("train", "test"):
        dataset = load_dataset(
            "parquet", data_files=str(_bizbench_parquet(source_dir, split)), split="train"
        )
        for index, row in enumerate(dataset):
            task = str(row.get("task") or "")
            if task not in BIZ_TASKS:
                continue
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            program = str(row.get("program") or "").strip()
            context_value = row.get("context")
            context = "" if context_value is None else str(context_value).strip()
            if not question or not answer or not program:
                continue
            prompt_context = context if context else "No external document context is required."
            records.append({
                "sample_id": f"bizbench:{split}:{task}:{index}",
                "source_dataset": "BizBench",
                "source_split": split,
                "task": task,
                "task_family": "program_synthesis",
                "lineage_key": _biz_lineage(context, question),
                "question_key": normalize_text(question),
                "image_path": "",
                "context": prompt_context,
                "question": question,
                "reasoning": program,
                "answer": answer,
                "scale": "",
                "program": program,
                "target": TARGET_TEMPLATE.format(trace=program, answer=answer),
                "assigned_split": split,
            })
    return records


def _keys(records: Iterable[dict]) -> tuple[set[str], set[str]]:
    records = list(records)
    return ({row["question_key"] for row in records}, {row["lineage_key"] for row in records})


def _connected_components(records: list[dict]) -> list[list[dict]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    first_by_lineage: dict[str, int] = {}
    first_by_question: dict[str, int] = {}
    for index, row in enumerate(records):
        for key, seen in ((row["lineage_key"], first_by_lineage), (row["question_key"], first_by_question)):
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    components: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(records):
        components[find(index)].append(row)
    return list(components.values())


def make_splits(tat: list[dict], biz: list[dict], *, seed: int, validation_ratio: float) -> tuple[dict, dict]:
    """Regroup TAT-DQA by report and preserve BizBench's official test split."""
    output = {name: [] for name in SPLITS}
    excluded = Counter()

    fixed_test = [row for row in biz if row["assigned_split"] == "test"]
    test_questions, test_lineages = _keys(fixed_test)
    output["test"].extend(fixed_test)

    # TAT-DQA's released page-level splits reuse annual reports. Regroup every
    # released gold row by annual report so no report can appear in more than
    # one local split. Do not connect different reports merely because they use
    # a generic question such as "What does the table show?"; those duplicate
    # question rows are removed from lower-priority splits below.
    tat_reports: dict[str, list[dict]] = defaultdict(list)
    for row in tat:
        tat_reports[row["lineage_key"]].append(row)
    for key, rows in tat_reports.items():
        score = stable_hash(key, seed) / 2**64
        if score < 0.8:
            split = "train"
        elif score < 0.9:
            split = "validation"
        else:
            split = "test"
        output[split].extend(rows)

    heldout_questions, heldout_lineages = _keys(output["validation"] + output["test"])
    flexible = [row for row in biz if row["assigned_split"] == "train"]
    eligible: list[dict] = []
    for row in flexible:
        if row["question_key"] in heldout_questions or row["lineage_key"] in heldout_lineages:
            excluded["train_collision_with_official_heldout"] += 1
        else:
            eligible.append(row)

    # Keep every BizBench context atomic. Cross-dataset repeated questions are
    # handled by held-out priority after these context groups are assigned.
    by_task: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    biz_contexts: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        biz_contexts[row["lineage_key"]].append(row)
    for rows in biz_contexts.values():
        tasks = sorted({row["task"] for row in rows})
        task = tasks[0]
        component_key = min(row["lineage_key"] for row in rows)
        by_task[task].append((component_key, rows))
    for task, groups in by_task.items():
        groups.sort(key=lambda pair: stable_hash(f"{task}\0{pair[0]}", seed))
        validation_groups = max(1, math.ceil(len(groups) * validation_ratio))
        for group_index, (_, rows) in enumerate(groups):
            output["validation" if group_index < validation_groups else "train"].extend(rows)

    # Keep official/held-out examples when normalized questions recur across
    # documents; remove only the lower-priority rows. Duplicates inside one
    # split remain together and are harmless for the cross-split audit.
    seen_questions: set[str] = set()
    seen_lineages: set[str] = set()
    for split in ("test", "validation", "train"):
        split_questions = {row["question_key"] for row in output[split]}
        split_lineages = {row["lineage_key"] for row in output[split]}
        kept = []
        for row in output[split]:
            if row["question_key"] in seen_questions or row["lineage_key"] in seen_lineages:
                excluded[f"{split}_lower_priority_collision"] += 1
            else:
                kept.append(row)
        output[split] = kept
        seen_questions.update(split_questions)
        seen_lineages.update(split_lineages)

    for split in SPLITS:
        output[split].sort(key=lambda row: stable_hash(row["sample_id"], seed))
    audit = audit_splits(output)
    audit["excluded"] = dict(excluded)
    return output, audit


def audit_splits(
    splits,
    *,
    workspace: Path | None = None,
    trace_free_test: bool = False,
) -> dict:
    counts: dict[str, int] = {}
    sources: dict[str, dict] = {}
    questions: dict[str, set[str]] = {}
    lineages: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = list(splits[split])
        if not rows:
            raise AssertionError(f"Empty {split} split")
        counts[split] = len(rows)
        sources[split] = dict(Counter(f"{row['source_dataset']}/{row['task']}" for row in rows))
        questions[split] = {row["question_key"] for row in rows}
        lineages[split] = {row["lineage_key"] for row in rows}
        for row in rows:
            required = ("sample_id", "question", "answer", "target")
            if not all(str(row.get(key) or "").strip() for key in required):
                raise AssertionError(f"Incomplete row in {split}: {row.get('sample_id')}")
            if trace_free_test and split == "test":
                if str(row.get("reasoning") or "").strip() or str(row.get("program") or "").strip():
                    raise AssertionError(f"Test row exposes a reasoning trace: {row.get('sample_id')}")
                if str(row["target"]).strip() != str(row["answer"]).strip():
                    raise AssertionError(f"Test target exposes more than the final answer: {row.get('sample_id')}")
            elif not str(row.get("reasoning") or "").strip():
                raise AssertionError(f"Supervised row lacks a reasoning trace in {split}: {row.get('sample_id')}")
            if row["task_family"] in {"visual_document_reasoning", "visual_ocr"} and workspace is not None:
                image_path = workspace / row["image_path"]
                if not image_path.is_file():
                    raise AssertionError(f"Missing TAT-DQA image: {image_path}")
            if row["task_family"] == "financial_reasoning":
                if row.get("image_path"):
                    raise AssertionError(f"Reasoning row unexpectedly retains an image: {row.get('sample_id')}")
                if not str(row.get("context") or "").strip():
                    raise AssertionError(f"Reasoning row lacks evidence context: {row.get('sample_id')}")
    overlaps = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlaps[f"{left}__{right}"] = {
                "normalized_questions": len(questions[left] & questions[right]),
                "document_lineages": len(lineages[left] & lineages[right]),
            }
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise AssertionError(f"Cross-split leakage: {overlaps}")
    return {"counts": counts, "source_task_counts": sources, "overlaps": overlaps}


def prepare(
    tat_source_dir: Path,
    biz_source_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    validation_ratio: float,
    render_dpi: int,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    media_dir = output_dir.parent / f"tatdqa_pages_{render_dpi}dpi_crops"
    tat = load_tat_records(tat_source_dir, media_dir, render_dpi=render_dpi)
    biz = load_biz_records(biz_source_dir)
    rows, audit = make_splits(tat, biz, seed=seed, validation_ratio=validation_ratio)
    dataset = DatasetDict({name: Dataset.from_list(rows[name]) for name in SPLITS})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    manifest = {
        "name": output_dir.name,
        "seed": seed,
        "bizbench_validation_ratio": validation_ratio,
        "sources": {
            "TAT-DQA": {"revision": TAT_REVISION, "license": "CC-BY-4.0", "split_policy": "all public gold rows regrouped 80/10/10 by annual report; lower-priority duplicate questions pruned"},
            "BizBench": {"revision": BIZ_REVISION, "license": "Apache-2.0", "tasks": list(BIZ_TASKS), "split_policy": "official test; lineage-grouped validation from train"},
        },
        "trace_policy": "grounded evidence values, derivation, scale, and answer in every TAT-DQA split; executable source program in every BizBench split; no model-generated traces",
        "image_policy": {
            "source": "original PDFs contained in the pinned TAT-DQA archives",
            "render_dpi": render_dpi,
            "crop": "question evidence blocks plus vertical context; full page when no block mapping is available",
            "media_directory": str(media_dir.name),
        },
        "assistant_target_template": TARGET_TEMPLATE,
        "task_datasets": {
            capability: {"directory": spec["directory"], "tasks": list(spec["tasks"])}
            for capability, spec in TASK_DATASETS.items()
        },
        **audit,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_task_datasets(output_dir)
    return manifest


def capability_for_task(task: str) -> str:
    try:
        return CAPABILITY_BY_TASK[task]
    except KeyError as error:
        raise ValueError(f"Unclassified finance task: {task}") from error


def task_dataset_paths(dataset_root: str | Path) -> dict[str, Path]:
    root = Path(dataset_root).resolve()
    return {capability: root / spec["directory"] for capability, spec in TASK_DATASETS.items()}


def _project_task_row(row: dict, capability: str, split: str) -> dict:
    spec = TASK_DATASETS[capability]
    projected = {
        "capability": capability,
        "task_family": spec["task_family"],
    }
    if spec["input_mode"] == "text":
        projected["image_path"] = ""
    if split == "test":
        # Test inputs must never expose gold chains of thought or source code.
        # Keep only the reference answer for task-native scoring.
        projected.update({"reasoning": "", "program": "", "target": str(row["answer"]).strip()})
    return projected


def write_task_datasets(core_dir: str | Path) -> dict[str, dict]:
    """Materialize three disjoint DatasetDicts while preserving parent split assignments."""
    core_dir = Path(core_dir).resolve()
    core = load_from_disk(str(core_dir))
    workspace = core_dir.parents[1]
    outputs: dict[str, dict] = {}
    all_sample_ids: dict[str, set[str]] = {}
    for capability, spec in TASK_DATASETS.items():
        allowed = set(spec["tasks"])
        view = DatasetDict({
            split: core[split]
            .filter(lambda row, tasks=allowed: row["task"] in tasks)
            .map(lambda row, name=capability, split_name=split: _project_task_row(row, name, split_name))
            for split in SPLITS
        })
        observed = set().union(*(set(view[split]["task"]) for split in SPLITS))
        if observed != allowed:
            raise AssertionError(f"{capability} task mismatch: expected {sorted(allowed)}, found {sorted(observed)}")
        audit = audit_splits(view, workspace=workspace, trace_free_test=True)
        if any(
            not str(value).strip()
            for split in ("train", "validation")
            for value in view[split]["reasoning"]
        ):
            raise AssertionError(f"{capability} contains an empty train/validation reasoning trace")
        if any(
            not str(value).startswith("<think>\n")
            for split in ("train", "validation")
            for value in view[split]["target"]
        ):
            raise AssertionError(f"{capability} contains an inconsistent train/validation target")

        path = task_dataset_paths(core_dir.parent)[capability]
        manifest = {
            "name": path.name,
            "capability": capability,
            "tasks": list(spec["tasks"]),
            "input_mode": spec["input_mode"],
            "parent_dataset": core_dir.name,
            "split_policy": "filtered from the leakage-audited parent without changing split assignments",
            "trace_policy": (
                "train and validation retain non-empty source-grounded reasoning with <think>...</think> "
                "targets; test reasoning/program fields are empty and test targets contain only the final answer"
            ),
            **audit,
        }
        if path.exists():
            existing_manifest = path / "split_manifest.json"
            if not existing_manifest.is_file():
                raise FileExistsError(f"Refusing unverified existing task dataset: {path}")
            existing = json.loads(existing_manifest.read_text())
            comparable = {key: existing.get(key) for key in manifest}
            if comparable != manifest:
                raise FileExistsError(f"Existing task dataset does not match requested projection: {path}")
        else:
            view.save_to_disk(str(path))
            (path / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        outputs[capability] = manifest
        all_sample_ids[capability] = set().union(*(set(view[split]["sample_id"]) for split in SPLITS))

    for left_index, left in enumerate(TASK_DATASETS):
        for right in tuple(TASK_DATASETS)[left_index + 1:]:
            overlap = all_sample_ids[left] & all_sample_ids[right]
            if overlap:
                raise AssertionError(f"Task datasets overlap: {left} vs {right}: {len(overlap)} rows")
    core_manifest_path = core_dir / "split_manifest.json"
    if core_manifest_path.is_file():
        core_manifest = json.loads(core_manifest_path.read_text())
        core_manifest["task_datasets"] = {
            capability: {
                "directory": spec["directory"],
                "tasks": list(spec["tasks"]),
                "input_mode": spec["input_mode"],
            }
            for capability, spec in TASK_DATASETS.items()
        }
        core_manifest["task_dataset_trace_policy"] = (
            "reasoning traces are retained in train/validation only; task-view test splits contain "
            "no reference reasoning or programs"
        )
        core_manifest_path.write_text(json.dumps(core_manifest, indent=2, sort_keys=True) + "\n")
    return outputs


def _lazy_image(path: str):
    return Image.open(path).convert("RGB")


def load_finance_task_split(dataset_paths: Iterable[str | Path], split: str) -> Dataset:
    """Load the three task DatasetDicts in a stable capability order."""
    paths = [Path(path).resolve() for path in dataset_paths]
    expected_names = [spec["directory"] for spec in TASK_DATASETS.values()]
    if [path.name for path in paths] != expected_names:
        raise ValueError(f"Expected task dataset order {expected_names}, got {[path.name for path in paths]}")
    datasets = []
    workspace = paths[0].parents[1]
    for (capability, spec), path in zip(TASK_DATASETS.items(), paths):
        prepared = load_from_disk(str(path))
        audit_splits(prepared, workspace=workspace, trace_free_test=True)
        dataset = prepared[split]
        if set(dataset["task"]) - set(spec["tasks"]):
            raise AssertionError(f"Unexpected tasks in {path}")
        if set(dataset["capability"]) != {capability}:
            raise AssertionError(f"Unexpected capability labels in {path}")
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def load_finance_capability_split(
    path_or_dataset: str | Path,
    split: str,
    capability: str | None = None,
) -> Dataset:
    """Load one task-specific DatasetDict and enforce its declared capability."""
    path = Path(path_or_dataset).resolve()
    inferred = next(
        (name for name, spec in TASK_DATASETS.items() if spec["directory"] == path.name),
        None,
    )
    expected = capability or inferred
    if expected not in TASK_DATASETS or inferred != expected:
        raise ValueError(f"Dataset {path.name} is not the requested capability {expected!r}")
    prepared = load_from_disk(str(path))
    audit_splits(prepared, workspace=path.parents[1], trace_free_test=True)
    dataset = prepared[split]
    spec = TASK_DATASETS[expected]
    if set(dataset["task"]) - set(spec["tasks"]):
        raise AssertionError(f"Unexpected tasks in {path}")
    if set(dataset["capability"]) != {expected}:
        raise AssertionError(f"Unexpected capability labels in {path}")
    return dataset


def _balance_by_task(dataset: Dataset) -> Dataset:
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    for index, task in enumerate(dataset["task"]):
        indices_by_task[task].append(index)
    if not indices_by_task:
        raise ValueError("Cannot balance an empty training split")
    samples_per_task = max(len(indices) for indices in indices_by_task.values())
    balanced_indices = [
        indices[position % len(indices)]
        for task in sorted(indices_by_task)
        for position in range(samples_per_task)
        for indices in (indices_by_task[task],)
    ]
    return dataset.select(balanced_indices)


def _with_conversation_transform(dataset: Dataset, workspace: Path) -> Dataset:
    def transform(batch):
        conversations = []
        for image_path, context, question, target, family, task in zip(
            batch["image_path"], batch["context"], batch["question"], batch["target"],
            batch["task_family"], batch["task"],
        ):
            prompt = build_user_prompt(task, family, question, context)
            if image_path:
                user_content = [
                    {"type": "image", "image": _lazy_image(str(workspace / image_path))},
                    {"type": "text", "text": prompt},
                ]
            else:
                user_content = [{"type": "text", "text": prompt}]
            conversations.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ])
        return {"conversation": conversations}

    return dataset.with_transform(transform)


def make_finance_core_dataset(
    path_or_dataset: str,
    split: str = "train",
    limit_dataset_samples: int | None = None,
    task_balanced: bool = False,
    **kwargs,
):
    """Load prepared finance rows and expose AutoModel's conversation field."""
    del kwargs
    prepared = load_from_disk(path_or_dataset)
    workspace = Path(path_or_dataset).resolve().parents[1]
    audit_splits(prepared, workspace=workspace)
    dataset = prepared[split]
    if task_balanced:
        if split != "train":
            raise ValueError("Task-balanced sampling is only valid for the training split")
        dataset = _balance_by_task(dataset)
    if limit_dataset_samples is not None:
        dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))
    return _with_conversation_transform(dataset, workspace)


def make_finance_task_mixture_dataset(
    dataset_paths: list[str],
    split: str = "train",
    limit_dataset_samples: int | None = None,
    task_balanced: bool = True,
    **kwargs,
):
    """Load the three separate task datasets as one balanced AutoModel training view."""
    del kwargs
    dataset = load_finance_task_split(dataset_paths, split)
    if task_balanced:
        if split != "train":
            raise ValueError("Task-balanced sampling is only valid for the training split")
        dataset = _balance_by_task(dataset)
    if limit_dataset_samples is not None:
        dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))
    workspace = Path(dataset_paths[0]).resolve().parents[1]
    return _with_conversation_transform(dataset, workspace)


def make_finance_single_task_dataset(
    path_or_dataset: str,
    split: str = "train",
    limit_dataset_samples: int | None = None,
    capability: str | None = None,
    **kwargs,
):
    """Expose one separated finance capability as an AutoModel conversation dataset."""
    del kwargs
    dataset = load_finance_capability_split(path_or_dataset, split, capability)
    if limit_dataset_samples is not None:
        dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))
    workspace = Path(path_or_dataset).resolve().parents[1]
    return _with_conversation_transform(dataset, workspace)


def make_finance_core_validation_dataset(
    path_or_dataset: str,
    split: str = "validation",
    samples_per_task: int = 32,
    **kwargs,
):
    """Create a deterministic task-stratified validation view."""
    del kwargs
    prepared = load_from_disk(path_or_dataset)
    dataset = prepared[split]
    selected = []
    counts = Counter()
    for index, task in enumerate(dataset["task"]):
        if counts[task] < samples_per_task:
            selected.append(index)
            counts[task] += 1
    return make_finance_core_dataset(
        path_or_dataset,
        split=split,
    ).select(selected)


def make_finance_task_validation_dataset(
    dataset_paths: list[str],
    split: str = "validation",
    samples_per_task: int = 32,
    **kwargs,
):
    """Create a deterministic task-stratified validation view from three DatasetDicts."""
    del kwargs
    dataset = load_finance_task_split(dataset_paths, split)
    selected = []
    counts = Counter()
    for index, task in enumerate(dataset["task"]):
        if counts[task] < samples_per_task:
            selected.append(index)
            counts[task] += 1
    workspace = Path(dataset_paths[0]).resolve().parents[1]
    return _with_conversation_transform(dataset.select(selected), workspace)


def make_finance_single_task_validation_dataset(
    path_or_dataset: str,
    split: str = "validation",
    samples_per_task: int = 32,
    capability: str | None = None,
    **kwargs,
):
    """Create a deterministic sub-task-stratified validation view for one capability."""
    del kwargs
    dataset = load_finance_capability_split(path_or_dataset, split, capability)
    selected = []
    counts = Counter()
    for index, task in enumerate(dataset["task"]):
        if counts[task] < samples_per_task:
            selected.append(index)
            counts[task] += 1
    workspace = Path(path_or_dataset).resolve().parents[1]
    return _with_conversation_transform(dataset.select(selected), workspace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--tat-source-dir", type=Path)
    parser.add_argument("--biz-source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-views-from", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--render-dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_views_from:
        if args.source_dir or args.tat_source_dir or args.biz_source_dir or args.output_dir:
            raise ValueError(
                "--task-views-from cannot be combined with source/output arguments"
            )
        manifest = write_task_datasets(args.task_views_from.resolve())
    else:
        if args.source_dir and (args.tat_source_dir or args.biz_source_dir):
            raise ValueError("Use --source-dir or the two dataset-specific source arguments, not both")
        tat_source = args.tat_source_dir or args.source_dir
        biz_source = args.biz_source_dir or args.source_dir
        if not tat_source or not biz_source or not args.output_dir:
            raise ValueError(
                "--output-dir plus --tat-source-dir/--biz-source-dir are required "
                "(--source-dir remains available for a combined directory)"
            )
        manifest = prepare(
            tat_source.resolve(), biz_source.resolve(), args.output_dir.resolve(),
            seed=args.seed, validation_ratio=args.validation_ratio, render_dpi=args.render_dpi,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
