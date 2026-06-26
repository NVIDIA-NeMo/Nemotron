import os
import sys

from datasets import concatenate_datasets, disable_caching

disable_caching()

from dataset_bird import DatasetBIRD
from dataset_bird_reasoning import DatasetBIRDReasoning

# Config comes from the environment (set by the notebook / dataprep.sbatch).
for _required_var in ("MODEL_ID", "MAX_SEQ_LEN", "DATAPREP_OUTPUT_DIR"):
    if _required_var not in os.environ:
        raise RuntimeError(f"{_required_var} environment variable must be set (see config.env).")

model_id = os.environ["MODEL_ID"]
max_seq_len = int(os.environ["MAX_SEQ_LEN"])
output_dir = os.environ["DATAPREP_OUTPUT_DIR"]

# Keep the dataset small for a fast example run. 0/unset = full BIRD.
max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))  # 0 = full BIRD (both splits)

# Which BIRD splits to include in the mix.
use_bird_no_reasoning = os.environ.get("USE_BIRD_NO_REASONING", "1") == "1"
use_bird_reasoning = os.environ.get("USE_BIRD_REASONING", "1") == "1"

os.makedirs(output_dir, exist_ok=True)
output_fp = f"{output_dir}/training.jsonl"
if os.path.exists(output_fp):
    print(f"A prepared dataset already exists at '{output_fp}'. Skipping.")
    sys.exit(0)

print("Using tokenizer for model:", model_id)
datasets = []

if use_bird_no_reasoning:
    print("Preparing BIRD (no reasoning)...")
    datasets.append(DatasetBIRD(model_id_to_prep_for=model_id, max_seq_len=max_seq_len).make_dataset())
    print("  samples:", len(datasets[-1]))

if use_bird_reasoning:
    print("Preparing BIRD (with reasoning)...")
    datasets.append(DatasetBIRDReasoning(model_id_to_prep_for=model_id, max_seq_len=max_seq_len).make_dataset())
    print("  samples:", len(datasets[-1]))

if not datasets:
    sys.exit("No datasets selected (set USE_BIRD_NO_REASONING / USE_BIRD_REASONING).")

dataset = concatenate_datasets(datasets)
print("Total samples after concat:", len(dataset))

# When capping for a fast run, shuffle first so both splits are represented, then cap.
if max_train_samples and len(dataset) > max_train_samples:
    print(f"Truncating to MAX_TRAIN_SAMPLES={max_train_samples}.")
    dataset = dataset.shuffle(seed=1234).select(range(max_train_samples))

dataset = dataset.sort("length")  # sort by length for training stability

print(f"Writing {len(dataset)} samples to '{output_fp}'")
dataset.to_json(output_fp, orient="records", lines=True, force_ascii=True)
print("Done.")
