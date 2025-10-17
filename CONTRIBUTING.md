# Contributing to Nemotron

We welcome contributions to the Nemotron repository!

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd nemotron
uv pip install -e .

# Run tests
uv run pytest tests/
```

## Contributing Workflow

1. **Fork** the repository and create a feature branch
2. **Make changes** following existing code patterns
3. **Write tests** for new features
4. **Update documentation** as needed
5. **Sign commits** with `git commit -s`
6. **Submit PR** with clear description

## Adding Training Recipes

When contributing a new recipe:

- Follow the three-step pattern (data curation, training, evaluation)
- Use the Artifact system for outputs
- Support scale factors (tiny/small/medium/full)
- Include comprehensive README with model overview, hardware requirements, and benchmarks
- Refer to [Adding Recipes Guide](docs/training/add-recipe.md)

## Code Quality

- Follow existing code style and patterns
- Use type hints and docstrings
- Write tests for new features
- Ensure all tests pass before submitting

## Developer Certificate of Origin

All commits must be signed off (`-s` flag) to certify that you have the right to submit the contribution under the project's open source license.

```bash
git commit -s -m "Your commit message"
```

## Questions?

- Check documentation in `docs/`
- Review examples in `examples/`
- Open an issue for discussions

---

Thank you for contributing to Nemotron!
