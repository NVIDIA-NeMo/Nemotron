"""Generate fictional page images and structured labels."""

from collections import Counter, defaultdict
import copy
import hashlib
import heapq
import json
from pathlib import Path
import random
from .render import PROMPT, patient_record, render_page

ORDER = "MODBIFL"


def universe():
    filling = []
    whole = []
    for tooth in range(1, 33):
        anterior = tooth in set(range(6, 12)) | set(range(22, 28))
        center = "I" if anterior else "O"
        outer = "F" if anterior else "B"
        for codes in [
            [center],
            ["M"],
            ["D"],
            [outer],
            ["L"],
            ["M", center],
            ["D", center],
            ["M", center, "D"],
        ]:
            filling.append((tooth, "filling", "".join(c for c in ORDER if c in codes)))
        whole.extend((tooth, procedure, "-") for procedure in ["crown", "extraction", "root canal"])
    return filling, whole


def plan(original, seed=2026110301):
    assert len(original) == 256
    rng = random.Random(seed)
    filling, whole = universe()
    events = filling * 3 + whole * 3 + rng.sample(filling, 72) + rng.sample(whole, 72)
    assert len(events) == 1200
    empty = {context * 8 + rng.randrange(8) for context in rng.sample(range(32), 16)}
    slots = [i for i in range(256) if i not in empty]
    rng.shuffle(slots)
    sizes = [3, 4, 5, 6, 7] * 48
    rng.shuffle(sizes)
    by_tooth = defaultdict(list)
    for event in events:
        by_tooth[event[0]].append(event)
    heap = [(-len(bag), rng.random(), tooth) for tooth, bag in by_tooth.items()]
    heapq.heapify(heap)
    tooth_slots = defaultdict(list)
    page_teeth = {}
    for index, size in zip(slots, sizes, strict=True):
        selected = [heapq.heappop(heap) for _ in range(size)]
        page_teeth[index] = [item[2] for item in selected]
        for negative, _, tooth in selected:
            tooth_slots[tooth].append(index)
            if negative + 1 < 0:
                heapq.heappush(heap, (negative + 1, rng.random(), tooth))
    assert not heap
    assignments = defaultdict(list)
    for tooth, indices in tooth_slots.items():
        bag = by_tooth[tooth]
        for attempt in range(100):
            rng.shuffle(bag)
            contexts = defaultdict(set)
            for index, event in zip(indices, bag, strict=True):
                contexts[event].add(index // 8)
            if min(map(len, contexts.values())) >= 2:
                break
        else:
            raise ValueError("Cannot distribute cases across patient contexts")
        for index, event in zip(indices, bag, strict=True):
            assignments[index].append(event)
    actual = Counter(event for values in assignments.values() for event in values)
    assert set(actual) == set(filling + whole) and set(actual.values()) <= {3, 4}
    rows = []
    for index, prior in enumerate(original):
        record = copy.deepcopy(prior["record"])
        record["odontogram"]["annotations"] = [
            {
                "tooth": t,
                "procedure": p,
                "status": "planned",
                "surface": s,
                "mark": "root-line"
                if p == "root canal"
                else "cross"
                if p == "extraction"
                else "solid",
            }
            for t, p, s in sorted(assignments[index])
        ]
        assert len({e["tooth"] for e in record["odontogram"]["annotations"]}) == len(
            record["odontogram"]["annotations"]
        )
        rows.append((prior, record))
    summary = {
        "training_pages": 256,
        "patient_contexts": 32,
        "variants_per_context": 8,
        "annotation_presentations": 1200,
        "distinct_exact_events": len(actual),
        "distinct_tooth_procedure_pairs": len({event[:2] for event in actual}),
        "distinct_tooth_surface_atoms": len(
            {(t, c) for t, p, s in actual if p == "filling" for c in s}
        ),
        "event_repetition_range": [min(actual.values()), max(actual.values())],
        "annotation_counts": dict(
            Counter(len(record["odontogram"]["annotations"]) for _, record in rows)
        ),
        "minimum_contexts_per_exact_event": 2,
        "seed": seed,
    }
    return rows, summary


def context_id(record):
    keys = ("patient", "visit", "medical_history", "treatment_history", "next_appointment", "notes")
    return hashlib.sha256(
        json.dumps({k: record[k] for k in keys}, sort_keys=True).encode()
    ).hexdigest()


def load_pages(manifest):
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for row in rows:
        row["image"] = str((manifest.parent / row["image"]).absolute())
    return rows


def save_pages(records, output, split):
    output = Path(output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records):
        image, target, geometry = render_page(record)
        name = f"{split}-{index:04d}"
        path = images / f"{name}.png"
        image.save(path)
        rows.append(
            dict(
                id=name,
                image=f"images/{name}.png",
                prompt=PROMPT,
                target=target,
                record=record,
                context_id=context_id(record),
                **geometry,
            )
        )
    manifest = output / f"{split}.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def make_records(seed, contexts, variants=1):
    rng = random.Random(seed)
    fillings, whole = universe()
    choices = {t: [e for e in fillings + whole if e[0] == t] for t in range(1, 33)}
    records = []
    for context in range(contexts):
        original = patient_record(rng, context, "fictional")
        original.update(palette={"planned": "#bf2635", "completed": "#185a9d"}, mirror=False)
        for variant in range(variants):
            record = copy.deepcopy(original)
            count = 0 if (context * variants + variant) % 16 == 15 else rng.choice([3, 4, 5, 6, 7])
            events = [rng.choice(choices[t]) for t in sorted(rng.sample(range(1, 33), count))]
            record["odontogram"]["annotations"] = [
                dict(
                    tooth=t,
                    procedure=p,
                    surface=s,
                    status="planned",
                    mark="root-line"
                    if p == "root canal"
                    else "cross"
                    if p == "extraction"
                    else "solid",
                )
                for t, p, s in events
            ]
            records.append(record)
    return records


def generate_dataset(output, seed=42):
    """256 balanced training pages, 32 monitoring pages, 256 held-out pages.

    Training uses 32 independent contexts with 8 chart variants each. The other
    splits have independent contexts and are never used for curriculum mining.
    """
    output = Path(output)
    manifests = {name: output / f"{name}.jsonl" for name in ("train", "monitor", "test")}
    metadata = output / "dataset.json"
    if metadata.is_file():
        if json.loads(metadata.read_text())["seed"] != seed:
            raise ValueError("Choose a new data directory when changing the seed")
        if all(path.is_file() for path in manifests.values()):
            return manifests
    originals = [{"record": r} for r in make_records(seed, 32, variants=8)]
    balanced, coverage = plan(originals, seed=seed)
    records = {
        "train": [r for _, r in balanced],
        "monitor": make_records(seed + 1, 32),
        "test": make_records(seed + 2, 256),
    }
    groups = {name: {context_id(r) for r in values} for name, values in records.items()}
    for a, b in [("train", "monitor"), ("train", "test"), ("monitor", "test")]:
        if not groups[a].isdisjoint(groups[b]):
            raise ValueError(f"Patient context overlap: {a}, {b}")
    manifests = {name: save_pages(values, output, name) for name, values in records.items()}
    metadata.write_text(
        json.dumps(
            dict(
                seed=seed,
                coverage=coverage,
                contexts={k: len(v) for k, v in groups.items()},
                pages={k: len(v) for k, v in records.items()},
            ),
            indent=2,
        )
    )
    return manifests
