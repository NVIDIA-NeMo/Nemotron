## Nemotron-CC Data Curation

This script includes the recipe for curating datasets similar to the [Nemotron-CC datasets](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2).

The Nemotron-CC pipeline can be roughly split up into the following stages:

1. Download, Extract, and Clean
  - This is a CPU-only pipeline consisting of the following stages:
    - Download Common Crawl snapshots from the web and extract text from the HTML webpages.
    - Use a fasttext-based language identification model to annotate each document with a language.
    - Fix mojibake (encoding issues) for UTF-8.
  - We recommend a CPU node where each worker can get at least 3.5GB of RAM to prevent OOM errors.

2. Deduplication
  a. Exact Deduplication
  b. Fuzzy Deduplication
  c. Substring Deduplication

3. Annotation and Filtering

4. Synthetic Data Generation
