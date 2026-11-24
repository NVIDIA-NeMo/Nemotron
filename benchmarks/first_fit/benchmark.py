"""Serial fresh-process benchmark; exact output hashes and Windows peak working set."""

import argparse
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(args):
    process = psutil.Process()
    affinity = process.cpu_affinity()
    process.cpu_affinity([affinity[-1]])
    module = load("measured", Path(args.source))
    if args.case == "short_synthetic":
        rng = random.Random(719)
        lengths = [rng.randrange(1, 33) for _ in range(10000)]
    elif args.case == "large_bin_synthetic":
        rng = random.Random(719)
        lengths = [rng.randrange(4097, 8193) for _ in range(10000)]
    else:
        lengths = json.loads((HERE / "alpaca-nemotron-lengths.json").read_text(encoding="utf-8"))["lengths"]
        if args.case == "alpaca_first_1000":
            lengths = lengths[:1000]
    repeats = 50 if args.case == "short_synthetic" else 1
    before = process.memory_info()
    start = time.perf_counter()
    cpu_start = time.process_time()
    bins = metrics = packer = None
    for iteration in range(repeats):
        if iteration:
            del bins, metrics, packer
        kwargs = {"seed": 42} if args.algorithm == "FirstFitShufflePacker" else {}
        packer = getattr(module, args.algorithm)(8192, **kwargs)
        bins, metrics = packer.pack(lengths)
    cpu_seconds = (time.process_time() - cpu_start) / repeats
    seconds = (time.perf_counter() - start) / repeats
    memory = process.memory_info()
    result = {
        "case": args.case,
        "algorithm": args.algorithm,
        "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "rows": len(lengths),
        "capacity": 8192,
        "seed": 42,
        "repeats": repeats,
        "seconds": seconds,
        "cpu_seconds": cpu_seconds,
        "sequences_per_second": len(lengths) / seconds,
        "rss_before_bytes": before.rss,
        "peak_wset_bytes": getattr(memory, "peak_wset", memory.rss),
        "peak_before_bytes": getattr(before, "peak_wset", before.rss),
        "metrics": dataclasses.asdict(metrics),
        "bins_sha256": hashlib.sha256(json.dumps(bins, separators=(",", ":")).encode()).hexdigest(),
        "input_sha256": hashlib.sha256(json.dumps(lengths, separators=(",", ":")).encode()).hexdigest(),
        "affinity": process.cpu_affinity(),
        "measured_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    print(json.dumps(result), flush=True)


def parent(args):
    directory = HERE / args.run
    directory.mkdir(exist_ok=False)
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": str(psutil.cpu_freq()),
        "psutil": psutil.__version__,
        "argv": sys.argv,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "memory": psutil.virtual_memory()._asdict(),
        "method": (
            "fresh serial process per version/case/algorithm/round; alternating order; "
            "pin last allowed logical CPU; OS peak_wset before hashing; tokenization excluded"
        ),
    }
    (directory / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    all_rows = []
    for case in args.cases:
        for algorithm in ("FirstFitDecreasingPacker", "FirstFitShufflePacker"):
            for round_idx in range(args.rounds):
                versions = ["baseline", "candidate"] if round_idx % 2 == 0 else ["candidate", "baseline"]
                pair = []
                for version in versions:
                    source = Path(args.checkout) / (
                        "baseline_algorithms.py"
                        if version == "baseline"
                        else "src/nemotron/data_prep/packing/algorithms.py"
                    )
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--source",
                        str(source),
                        "--case",
                        case,
                        "--algorithm",
                        algorithm,
                    ]
                    idle = psutil.cpu_percent(interval=0.2)
                    completed = subprocess.run(cmd, text=True, capture_output=True)
                    if completed.returncode:
                        raise RuntimeError(completed.stderr)
                    row = json.loads(completed.stdout)
                    row.update(version=version, round=round_idx, host_cpu_percent_before=idle, command=cmd)
                    all_rows.append(row)
                    pair.append(row)
                    with (directory / "raw.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps(row) + "\n")
                assert pair[0]["bins_sha256"] == pair[1]["bins_sha256"] and pair[0]["metrics"] == pair[1]["metrics"]
                print(
                    json.dumps({"case": case, "algorithm": algorithm, "round": round_idx, "equal": True}), flush=True
                )
    summaries = []
    for case in args.cases:
        for algorithm in ("FirstFitDecreasingPacker", "FirstFitShufflePacker"):
            versions = {}
            for version in ("baseline", "candidate"):
                rows = [
                    x
                    for x in all_rows
                    if x["case"] == case and x["algorithm"] == algorithm and x["version"] == version
                ]
                versions[version] = {
                    key: statistics.median(x[key] for x in rows)
                    for key in ("seconds", "cpu_seconds", "sequences_per_second", "peak_wset_bytes")
                }
                versions[version]["seconds_range"] = [min(x["seconds"] for x in rows), max(x["seconds"] for x in rows)]
            summary = {
                "case": case,
                "algorithm": algorithm,
                "rounds": args.rounds,
                "versions": versions,
                "speedup": versions["baseline"]["seconds"] / versions["candidate"]["seconds"],
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (directory / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--worker", action="store_true")
parser.add_argument("--source")
parser.add_argument("--case")
parser.add_argument("--algorithm")
parser.add_argument("--checkout")
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--run", default="run-final")
parser.add_argument(
    "--cases", nargs="+", default=["short_synthetic", "alpaca_first_1000", "alpaca_all", "large_bin_synthetic"]
)
args = parser.parse_args()
worker(args) if args.worker else parent(args)
