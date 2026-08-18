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

## Code Quality

- Follow existing code style and patterns
- Use type hints and docstrings
- Write tests for new features
- Ensure all tests pass before submitting

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Questions?

- Review examples in [`usage-cookbook/`](./usage-cookbook/)
- Open an issue for discussions

---

Thank you for contributing to Nemotron!
