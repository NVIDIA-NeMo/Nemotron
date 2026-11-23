"""Pin public Alpaca records and Nemotron tokenizer; export only token lengths."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
out = Path(__file__).parent
cache = Path(__file__).parent / ".cache"
cache.mkdir(exist_ok=True)
sources = {
    "alpaca_data.json": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json",
    "tokenizer.json": "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/resolve/6533e8de2c68e4536bf7c411d7a3ce5734111476/tokenizer.json",
}
metadata = {}
for name, url in sources.items():
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            path.write_bytes(response.read())
    metadata[name] = {
        "url": url,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    print(json.dumps(metadata[name]), flush=True)
records = json.loads((cache / "alpaca_data.json").read_text(encoding="utf-8"))
tokenizer = Tokenizer.from_file(str(cache / "tokenizer.json"))
tokenizer.no_truncation()
tokenizer.no_padding()
lengths = []
for start in range(0, len(records), 512):
    texts = [
        "\n\n".join(x[key] for key in ("instruction", "input", "output") if x[key])
        for x in records[start : start + 512]
    ]
    lengths.extend(len(x.ids) for x in tokenizer.encode_batch(texts, add_special_tokens=False))
payload = {
    "lengths": lengths,
    "sources": metadata,
    "format": "nonempty instruction/input/output joined with two newlines, no chat template, no special tokens",
    "rows": len(lengths),
    "min": min(lengths),
    "max": max(lengths),
    "tokens": sum(lengths),
}
(out / "alpaca-nemotron-lengths.json").write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({key: value for key, value in payload.items() if key != "lengths"}), flush=True)
