# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import subprocess
import tarfile
from unittest.mock import patch

import pytest


def test_self_contained_packager_produces_flat_tar(tmp_path):
    pytest.importorskip("nemo_run")

    from nemotron.kit.packaging.self_contained_packager import SelfContainedPackager

    repo_root = tmp_path / "repo"
    (repo_root / "src" / "nemotron").mkdir(parents=True)
    (repo_root / "src" / "nemotron" / "x.py").write_text(
        "def fx():\n    return 1\n",
        encoding="utf-8",
    )

    script_path = repo_root / "train.py"
    script_path.write_text(
        "from nemotron.x import fx\n\nprint(fx())\n",
        encoding="utf-8",
    )

    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text("a: 1\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class MockContext:
        @contextlib.contextmanager
        def cd(self, path):
            old = os.getcwd()
            os.chdir(path)
            try:
                yield
            finally:
                os.chdir(old)

        def run(self, cmd: str, **kwargs):
            subprocess.check_call(cmd, shell=True)

    with patch("nemo_run.core.packaging.pattern.Context", MockContext):
        packager = SelfContainedPackager(
            script_path=str(script_path.relative_to(repo_root)),
            train_path=str(train_cfg),
        )
        tar_path = packager.package(repo_root, str(out_dir), "pkg")

    with tarfile.open(tar_path, "r:gz") as tf:
        names = sorted(tf.getnames())
        assert names == ["config.yaml", "main.py"]
        main_src = tf.extractfile("main.py").read().decode("utf-8")
        assert "from nemotron" not in main_src


def test_self_contained_packager_inlines_nested_imports(tmp_path):
    """Verify that imports inside functions are also inlined."""
    pytest.importorskip("nemo_run")

    from nemotron.kit.packaging.self_contained_packager import SelfContainedPackager

    repo_root = tmp_path / "repo"
    (repo_root / "src" / "nemotron").mkdir(parents=True)
    (repo_root / "src" / "nemotron" / "x.py").write_text(
        "def fx():\n    return 1\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "nemotron" / "y.py").write_text(
        "def fy():\n    return 2\n",
        encoding="utf-8",
    )

    # Script with import inside a function (like stage2_rl/train.py)
    script_path = repo_root / "train.py"
    script_path.write_text(
        """\
from nemotron.x import fx

def main():
    from nemotron.y import fy
    print(fx(), fy())

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text("a: 1\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class MockContext:
        @contextlib.contextmanager
        def cd(self, path):
            old = os.getcwd()
            os.chdir(path)
            try:
                yield
            finally:
                os.chdir(old)

        def run(self, cmd: str, **kwargs):
            subprocess.check_call(cmd, shell=True)

    with patch("nemo_run.core.packaging.pattern.Context", MockContext):
        packager = SelfContainedPackager(
            script_path=str(script_path.relative_to(repo_root)),
            train_path=str(train_cfg),
        )
        tar_path = packager.package(repo_root, str(out_dir), "pkg")

    with tarfile.open(tar_path, "r:gz") as tf:
        names = sorted(tf.getnames())
        assert names == ["config.yaml", "main.py"]
        main_src = tf.extractfile("main.py").read().decode("utf-8")
        # Verify no nemotron imports remain (including nested ones)
        assert "from nemotron" not in main_src
        assert "import nemotron" not in main_src
        # Verify the inlined code is present
        assert "def fx():" in main_src
        assert "def fy():" in main_src
        # Verify the main function still exists
        assert "def main():" in main_src


def test_self_contained_packager_ignores_importlib_imports(tmp_path):
    """Verify that importlib.import_module calls are not detected as nemotron imports.

    This ensures the workaround in kit/wandb.py (using importlib instead of direct import)
    successfully prevents pydantic from being pulled into the RL training package.
    """
    pytest.importorskip("nemo_run")

    from nemotron.kit.packaging.self_contained_packager import SelfContainedPackager

    repo_root = tmp_path / "repo"
    (repo_root / "src" / "nemotron" / "kit").mkdir(parents=True)

    # Create a module that would pull in pydantic (simulating kit/__init__.py)
    (repo_root / "src" / "nemotron" / "kit" / "__init__.py").write_text(
        """\
from pydantic import BaseModel  # This would fail in nemo-rl container

def init():
    pass
""",
        encoding="utf-8",
    )

    # Create a module that uses importlib to import kit (like wandb.py does)
    (repo_root / "src" / "nemotron" / "wandb.py").write_text(
        """\
def patch_something():
    return "patched"

def init_wandb_if_configured():
    # Use importlib to avoid SelfContainedPackager detection
    import importlib
    kit = importlib.import_module("nemotron.kit")
    kit.init()
""",
        encoding="utf-8",
    )

    # Script that only imports the patch function (like RL train.py)
    script_path = repo_root / "train.py"
    script_path.write_text(
        """\
from nemotron.wandb import patch_something

def main():
    print(patch_something())

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text("a: 1\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class MockContext:
        @contextlib.contextmanager
        def cd(self, path):
            old = os.getcwd()
            os.chdir(path)
            try:
                yield
            finally:
                os.chdir(old)

        def run(self, cmd: str, **kwargs):
            subprocess.check_call(cmd, shell=True)

    with patch("nemo_run.core.packaging.pattern.Context", MockContext):
        packager = SelfContainedPackager(
            script_path=str(script_path.relative_to(repo_root)),
            train_path=str(train_cfg),
        )
        tar_path = packager.package(repo_root, str(out_dir), "pkg")

    with tarfile.open(tar_path, "r:gz") as tf:
        main_src = tf.extractfile("main.py").read().decode("utf-8")
        # Key assertion: pydantic should NOT be imported because importlib.import_module
        # is not detected as a nemotron import by the AST-based packager
        assert "from pydantic" not in main_src
        assert "import pydantic" not in main_src
        # The patch function should be inlined
        assert "def patch_something():" in main_src
        # The init_wandb_if_configured should also be inlined (it's in the same module)
        # but it uses importlib which doesn't trigger transitive inlining
        assert "importlib.import_module" in main_src
