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

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def test_code_packager_excludes_usage_cookbook_and_use_case_examples(tmp_path):
    pytest.importorskip("nemo_run")

    from nemo_runspec.packaging.code_packager import CodePackager

    repo_root = tmp_path / "repo"
    (repo_root / "src" / "nemotron").mkdir(parents=True)

    (repo_root / "usage-cookbook" / "x.txt").parent.mkdir(parents=True)
    (repo_root / "usage-cookbook" / "x.txt").write_text("nope", encoding="utf-8")
    (repo_root / "use-case-examples" / "y.txt").parent.mkdir(parents=True)
    (repo_root / "use-case-examples" / "y.txt").write_text("nope", encoding="utf-8")

    (repo_root / "src" / "nemotron" / "helper.py").write_text(
        "VALUE = 123\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "nemotron" / "__init__.py").write_text("", encoding="utf-8")

    script_rel = Path("src/nemotron/train.py")
    (repo_root / script_rel).write_text(
        """
import argparse

from nemotron.helper import VALUE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.parse_args()
    print(VALUE)


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )

    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text("a: 1\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    packager = CodePackager(script_path=str(script_rel), train_path=str(train_cfg))
    tar_path = packager.package(repo_root, str(out_dir), "pkg")

    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
        assert "main.py" in names
        assert "config.yaml" in names
        assert "src/nemotron/helper.py" in names
        assert not any(n.startswith("usage-cookbook/") for n in names)
        assert not any(n.startswith("use-case-examples/") for n in names)

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        tf.extractall(extract_dir)

    out = subprocess.check_output(
        [
            sys.executable,
            str(extract_dir / "main.py"),
            "--config",
            "config.yaml",
        ],
        cwd=extract_dir,
        text=True,
    )
    assert out.strip().endswith("123")
