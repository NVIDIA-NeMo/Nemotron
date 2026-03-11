import os
import sys

from datasets import concatenate_datasets, disable_caching
disable_caching()

from dataset_bird import DatasetBIRD
from dataset_bird_reasoning import DatasetBIRDReasoning

# --------------------------------------------------------------------------------------------------
# Read configuration from environment variables (set by the notebook or caller)
# --------------------------------------------------------------------------------------------------
for _required_var in ("MODEL_ID", "MAX_SEQ_LEN", "DATAPREP_OUTPUT_DIR"):
    if _required_var not in os.environ:
        raise RuntimeError(
            f"{_required_var} environment variable must be set explicitly. "
            f"See the notebook configuration cells or the README for details."
        )

model_id = os.environ["MODEL_ID"]
max_seq_len = int(os.environ["MAX_SEQ_LEN"])
output_dir = os.environ["DATAPREP_OUTPUT_DIR"]

# Which datasets to prepare and add to the final mix?
use_bird_no_reasoning = True
use_bird_reasoning = True

# --------------------------------------------------------------------------------------------------
# No need to modify anything below this line
# --------------------------------------------------------------------------------------------------

os.makedirs(output_dir, exist_ok=True)
output_fp = f"{output_dir}/training.jsonl"

if os.path.exists(output_fp):
    print(f"A prepared dataset already exists at '{output_fp}'. Exiting to avoid overwriting.")
    sys.exit(1)


print("Using tokenizer for model: ", model_id)

datasets = []

if use_bird_no_reasoning:
    print(f"{'-' * 50}\nPreparing BIRD training dataset (no reasoning)...")
    dataset_bird = DatasetBIRD(
        model_id_to_prep_for=model_id,
        max_seq_len=max_seq_len,
    ).make_dataset()

    datasets.append(dataset_bird)
    print("Total samples: ", len(datasets[-1]))

if use_bird_reasoning:
    print(f"{'-' * 50}\nPreparing BIRD training dataset (with reasoning)...")
    dataset_bird_reasoning = DatasetBIRDReasoning(
        model_id_to_prep_for=model_id,
        max_seq_len=max_seq_len,
    ).make_dataset()

    datasets.append(dataset_bird_reasoning)
    print("Total samples: ", len(datasets[-1]))

print(f"\n\n{'-' * 50}\nConcatenating all datasets...")
datasets = concatenate_datasets(datasets)

# Sort by length (helps with training stability)
datasets = datasets.sort("length")

print("Total samples prepared after concat: ", len(datasets))
print(f"Saving to '{output_fp}'")
datasets.to_json(output_fp, orient="records", lines=True, force_ascii=True)
print(f"\nFinished preparing '{len(datasets)}' samples")
