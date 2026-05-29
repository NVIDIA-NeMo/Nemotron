#! /bin/bash

set -eux

export NUM_THREADS=128  # value can be set up to as many cores as you have on your machine
export LENGTH_THRESHOLD=100
export BYTES_PER_TOKEN=2

# TODO: Update paths
export INPUT_PATH="/path/to/input/data"
export MAIN_CACHE_PATH="/path/to/main/cache"
export OUTPUT_PATH="/path/to/output"
# Clear the main work path and output path if they exist and create new ones
rm -rf ${MAIN_CACHE_PATH} ${OUTPUT_PATH} || true
mkdir -p ${MAIN_CACHE_PATH} ${OUTPUT_PATH}

# Clear the deduplicate-text-datasets directory if it exists
rm -rf deduplicate-text-datasets || true
# Build the Rust code
echo "Building Rust code... - $(date)"
git clone https://github.com/google-research/deduplicate-text-datasets.git
( cd deduplicate-text-datasets ; cargo build --release )
echo
DEDUP_PATH="$(realpath deduplicate-text-datasets)"
DEDUP_COMMAND="${DEDUP_PATH}/target/release/dedup_dataset"

# Connects to Ray and uses NeMo Curator
echo "Preparing dataset... - $(date)"
python -u prepare_dataset.py --input-path ${INPUT_PATH} --output-path ${MAIN_CACHE_PATH}
echo

export DATA_CACHE_PATH="${MAIN_CACHE_PATH}/data"

echo "Making suffix arrays... - $(date)"
python -u make_suffix_array.py \
--input-path "${DATA_CACHE_PATH}/dataset.bin" \
--output-path "${DATA_CACHE_PATH}" \
--num-threads ${NUM_THREADS} \
--dedup-command "${DEDUP_COMMAND}"
echo

echo "Running self-similar... - $(date)"
${DEDUP_COMMAND} self-similar \
--data-file "${DATA_CACHE_PATH}/dataset.bin" \
--length-threshold ${LENGTH_THRESHOLD} \
--cache-dir "${DATA_CACHE_PATH}/cache" \
--num-threads ${NUM_THREADS}
echo

echo "Running collect... - $(date)"
${DEDUP_COMMAND} collect \
--data-file "${DATA_CACHE_PATH}/dataset.bin" \
--length-threshold ${LENGTH_THRESHOLD} \
--cache-dir "${DATA_CACHE_PATH}/cache" \
> "${DATA_CACHE_PATH}/dataset.bin.remove.byterange"
echo

echo "Removing duplicates... - $(date)"
python -u remove_duplicates.py \
--input-path "${DATA_CACHE_PATH}/dataset" \
--output-path "${DATA_CACHE_PATH}/deduped_dataset" \
--length-threshold ${LENGTH_THRESHOLD} \
--bytes-per-token ${BYTES_PER_TOKEN}
echo

echo "Reconstructing dataset... - $(date)"
python -u reconstruct_dataset.py \
--input-path "${DATA_CACHE_PATH}/deduped_dataset" \
--output-path "${OUTPUT_PATH}" \
--original-dataset-path "${INPUT_PATH}" \
--id-to-filename-path "${MAIN_CACHE_PATH}/id_to_filename.json"
echo
