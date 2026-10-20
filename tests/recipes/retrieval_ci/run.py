# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run offline CPU recipe contracts and reject silently skipped coverage.

From the repository root:
    uv sync --frozen --project tests/recipes/retrieval_ci
    uv pip install --python tests/recipes/retrieval_ci/.venv/bin/python \
        --no-deps --no-build-isolation -e .
    tests/recipes/retrieval_ci/.venv/bin/python tests/recipes/retrieval_ci/run.py
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TESTS = [
    "tests/recipes/test_nemotron_3_5_finetune.py",
    "tests/recipes/test_retrieval_training_preflight.py",
    "tests/recipes/test_retrieval_checkpoint_resources.py",
    "tests/recipes/embed/test_sdg_bundle_validation.py",
    "tests/recipes/embed/test_sdg_bundle_resolution.py",
    "tests/recipes/embed/test_sdg_directory_config.py",
    "tests/recipes/embed/test_sdg.py",
    "tests/recipes/embed/test_cli.py",
    "tests/recipes/embed/test_native_multimodal_mining.py",
    "tests/recipes/embed/test_vllm_mining.py",
    "tests/recipes/embed/test_api_truncation.py",
    "tests/recipes/embed/test_eval_batching.py",
    "tests/recipes/embed/test_eval_nim.py",
    "tests/recipes/embed/test_deploy.py",
    "tests/recipes/embed/test_stage_dependencies.py",
    "tests/recipes/embed/test_torch_sources.py",
    "tests/recipes/rerank/test_deploy.py",
    "tests/recipes/rerank/test_artifact_root.py",
    "tests/recipes/retrieval_ci/test_smoke.py",
]


class RequireCoverage:
    """Dependency-related collection or runtime skips must make CI fail."""

    def __init__(self):
        self.skipped = []

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.skipped:
            print(f"\nCPU contract tests unexpectedly skipped: {self.skipped}")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


if __name__ == "__main__":
    arguments = ["-q", "--strict-markers", *[str(ROOT / path) for path in TESTS]]
    raise SystemExit(pytest.main(arguments, plugins=[RequireCoverage()]))
