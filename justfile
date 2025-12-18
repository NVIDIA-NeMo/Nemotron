# Nemotron development commands

# Build documentation
docs-render:
    uv run --group docs sphinx-build docs docs/_build/html

# Start documentation development server with live reload
docs-dev:
    uv run --group docs sphinx-autobuild docs docs/_build/html --port 12345 --open-browser
