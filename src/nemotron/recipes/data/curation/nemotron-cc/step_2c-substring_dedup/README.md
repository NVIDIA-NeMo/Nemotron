# Exact Substring Deduplication

This folder runs exact substring deduplication using Google Research's implementation: https://github.com/google-research/deduplicate-text-datasets
- Paper: https://arxiv.org/abs/2107.06499

Instructions for running exact substring deduplication:
1. Edit the parameters in the `exact_substring_dedup.sh` script.
2. Execute `bash exact_substring_dedup.sh`. The deduped dataset will be written to the `OUTPUT_PATH`.

If you are adapting it for Slurm:
1. Custom Ray cluster setup logic will be needed for `prepare_dataset` to parallelize it across many nodes.
2. Any step that uses `DEDUP_COMMAND` (`make_suffix_array`, `self-similar`, and `collect`) must run on a single exclusive node.
3. The `remove_duplicates` step must run on a single exclusive node.

Dependencies:
- `nemo_curator>=1.1.0`
- `cargo` (for building `deduplicate-text-datasets`). See https://doc.rust-lang.org/cargo/getting-started/installation.html

We recommend splitting up the dataset into 100 GB chunks or less, and executing `exact_substring_dedup.sh` on each 100 GB chunk.

Conservatively, we recommend to have at least 2-3x the input dataset size of RAM, and 10-15x the input dataset size of hard drive space. This means for a 100 GB input dataset, you should aim for 200-300 GB of RAM and 1-1.5 TB of hard drive space. Deduplication artifacts (everything in the `MAIN_CACHE_PATH`) can be deleted after the final `reconstruct_dataset.py` script writes the deduplicated dataset to the `OUTPUT_PATH`.

## Debugging Tips

- When running `make_suffix_array.py`, you may see "Killed" in the logs as the job is still running. This is fine. The script will rerun those failed jobs until there are no failures remaining.
- If `self-similar` is hanging, cancel it and try to increase `NUM_THREADS` (really, `NUM_THREADS` should always be as high as possible, meaning `NUM_CPUS`). Before rerunning, do `rm DATA_CACHE_PATH/cache`
- If `collect` stops with a message "Killed," it is likely an OOM. Do `rm DATA_CACHE_PATH/dataset.bin.remove.byterange` and try again with more RAM
