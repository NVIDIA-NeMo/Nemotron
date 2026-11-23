# First-fit CPU benchmark (Windows)

Install `psutil tokenizers`. From this checkout run:

```powershell
python benchmarks/first_fit/prepare_public_lengths.py
git show c388f16f33f4057268e68aaf4d9b232618ea8597:src/nemotron/data_prep/packing/algorithms.py > baseline_algorithms.py
python benchmarks/first_fit/benchmark.py --checkout . --run run-public --cases short_synthetic alpaca_first_1000 alpaca_all
```

Use a UTF-8 shell for the baseline export. The pinned public Alpaca and Nemotron tokenizer revisions and SHA256 hashes are recorded by preparation. No chat template or special tokens. Raw data cached locally; only lengths are used by timed workers. Five serial fresh processes per version/case/algorithm, alternating order, pinned to one logical CPU. Output includes exact bin hashes and all metrics. Tokenization/imports excluded from timed packing. Windows OS peak working set is captured before hashing. This script targets Windows for the reported peak-memory methodology; the fallback on other platforms is current RSS, not peak RSS.

Results are algorithm-only, not Ray or model training throughput. Segment tree storage has a larger constant than the original list; do not infer memory reduction from process-level noise. No linked issue, no model quality claim.
