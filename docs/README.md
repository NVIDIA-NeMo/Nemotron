# Building the Documentation

The authoritative recipe for building the documentation is what runs in CI, the
_build_docs job in <https://github.com/NVIDIA-NeMo/FW-CI-templates>.

1. Install documentation dependencies with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync --group docs
   ```

1. Build HTML from the `docs/` directory:

   ```bash
   cd docs
   sphinx-build --fail-on-warning --builder html . _build/html
   ```

Open `docs/_build/html/index.html` in a browser to preview the output.