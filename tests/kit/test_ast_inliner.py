from __future__ import annotations

from pathlib import Path

from nemotron.kit.packaging.self_contained_packager import inline_imports


def test_inline_imports_inlines_nemotron_modules_and_keeps_external_imports(tmp_path: Path):
    repo_root = tmp_path
    (repo_root / "src" / "nemotron").mkdir(parents=True)

    (repo_root / "src" / "nemotron" / "a.py").write_text(
        """
def f() -> int:
    return 1
""".lstrip(),
        encoding="utf-8",
    )

    (repo_root / "src" / "nemotron" / "b.py").write_text(
        """
from nemotron.a import f as g

def h() -> int:
    return g() + 1
""".lstrip(),
        encoding="utf-8",
    )

    entry = repo_root / "entry.py"
    entry.write_text(
        """
from __future__ import annotations

import math
from nemotron.b import h
import nemotron.a as a

def main() -> int:
    return h() + a.f() + int(math.sqrt(4))
""".lstrip(),
        encoding="utf-8",
    )

    out = inline_imports(entry, repo_root=repo_root, package_prefix="nemotron")
    for line in out.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("from nemotron")
        assert not stripped.startswith("import nemotron")
    assert "import math" in out

    ns: dict[str, object] = {}
    exec(compile(out, "<inlined>", "exec"), ns)
    assert ns["main"]() == 5
