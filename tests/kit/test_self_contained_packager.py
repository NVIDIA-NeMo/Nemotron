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
