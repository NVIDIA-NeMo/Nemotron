## Nemotron-CC Data Curation

This script includes the recipe for Curating datasets similar to the [nemotron-cc datasets](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2).

The Nemotron-cc pipeline can be roughly split up into the following stages:

1. Download, Extract & Clean
  - This is a CPU only pipeline consisting of the following stages:
    - Download common crawl snapshots from the web & extract text from the html webpages.
    - Use a fasttext based language identification model to annotate each document with a language.
    - Fix mojibake (encoding issues) for utf-8.
  - We recommend a CPU node where each worker can get at-least 3.5GB of RAM to prevent OOM errors.

2. Deduplication
  a. Exact Deduplication
  b. Fuzzy Deduplication
  c. Substring Deduplication

3. Annotation and Quality Filtering

4. Synthetic Data generation
