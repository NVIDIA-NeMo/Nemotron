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

"""nemo-run packager for long-document SDG recipes.

The shared ``CodePackager`` injects ``--config config.yaml`` into a bare
``python <script>`` invocation.  The long-document scripts use PEP 723 inline
deps and need to run under ``uv run --no-project`` so those deps are
resolved at runtime inside the container.  This subclass only swaps the
launcher; everything else (tarballing, git-aware inclusion, exclusion of
``usage-cookbook``/``use-case-examples``) is inherited verbatim.

When ``sentinel_path`` is provided, the launcher additionally:
  - polls that path on shared storage for the serve task's published endpoint
  - exports / appends ``vllm_endpoint=<url>`` to the recipe invocation
  - traps process exit so the serve task receives ``<sentinel>.done`` and
    cleanly shuts vLLM down
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nemo_runspec.packaging import CodePackager


@dataclass(kw_only=True)
class LongDocPackager(CodePackager):
    """Packager that launches long-document SDG scripts via ``uv run``.

    Args:
        sentinel_path: When set, the launcher waits for an endpoint URL to
            be published at this path (by a sibling serve task) before
            invoking the recipe.  The URL is appended to argv as
            ``vllm_endpoint=<url>`` so the recipe's OmegaConf dotlist merge
            picks it up.  On exit, the launcher creates ``<sentinel>.done``
            so the serve task can shut down cleanly.
        sentinel_max_wait_secs: How long the launcher waits for the sentinel
            before giving up (default: 30 minutes).
    """

    sentinel_path: str | None = None
    sentinel_max_wait_secs: int = 1800

    def _build_launcher(self, repo_root: Path) -> str:
        script_file = Path(self.script_path)
        if not script_file.is_absolute():
            script_file = repo_root / self.script_path
        rel_script = script_file.relative_to(repo_root)
        rel_script_repr = repr(rel_script.as_posix())

        # Optional sentinel-aware preamble: poll a path on shared storage,
        # set VLLM_ENDPOINT, register atexit done-marker.
        sentinel_preamble = ""
        if self.sentinel_path:
            sentinel_preamble = (
                "import atexit\n"
                "import time\n\n"
                f"SENTINEL_PATH = {self.sentinel_path!r}\n"
                "SENTINEL_DONE = SENTINEL_PATH + '.done'\n\n"
                "def _signal_done() -> None:\n"
                "    try:\n"
                "        Path(SENTINEL_DONE).touch()\n"
                "    except OSError:\n"
                "        pass\n"
                "atexit.register(_signal_done)\n\n"
                f"_max_wait = {self.sentinel_max_wait_secs}\n"
                "_waited = 0\n"
                "print(f'[launcher] waiting for serve endpoint at {SENTINEL_PATH}', flush=True)\n"
                "while True:\n"
                "    try:\n"
                "        _size = Path(SENTINEL_PATH).stat().st_size\n"
                "    except OSError:\n"
                "        _size = 0\n"
                "    if _size > 0:\n"
                "        break\n"
                "    if _waited >= _max_wait:\n"
                "        print(f'[launcher] timed out after {_max_wait}s waiting for {SENTINEL_PATH}', flush=True)\n"
                "        sys.exit(1)\n"
                "    time.sleep(5)\n"
                "    _waited += 5\n"
                "endpoint = Path(SENTINEL_PATH).read_text().strip()\n"
                "print(f'[launcher] using vllm endpoint: {endpoint}', flush=True)\n"
                "extra_recipe_args = [f'vllm_endpoint={endpoint}']\n"
            )
        else:
            sentinel_preamble = "extra_recipe_args = []  # no serve sentinel\n"

        return (
            "from __future__ import annotations\n\n"
            "import os\n"
            "import shutil\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "ROOT = os.path.dirname(__file__)\n"
            "os.chdir(ROOT)\n"
            "print('[launcher] Starting long-document SDG launcher...', file=sys.stderr)\n\n"
            f"script_path = os.path.join(ROOT, {rel_script_repr})\n"
            "config_path = os.path.join(ROOT, 'config.yaml')\n\n"
            f"{sentinel_preamble}\n"
            "# Bootstrap uv if missing — required for `uv run` to resolve PEP 723 deps.\n"
            "uv = shutil.which('uv')\n"
            "if uv is None:\n"
            "    print('[launcher] uv not found; bootstrapping via pip install uv...', file=sys.stderr)\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'])\n"
            "    uv = shutil.which('uv')\n"
            "    if uv is None:\n"
            "        for candidate in (\n"
            "            os.path.join(sys.prefix, 'bin', 'uv'),\n"
            "            os.path.expanduser('~/.local/bin/uv'),\n"
            "        ):\n"
            "            if os.path.exists(candidate):\n"
            "                uv = candidate\n"
            "                break\n"
            "if uv is None:\n"
            "    print('[launcher] ERROR: uv could not be located after install', file=sys.stderr)\n"
            "    sys.exit(1)\n\n"
            "env = os.environ.copy()\n"
            "env.pop('VIRTUAL_ENV', None)\n"
            "env['PYTHONPATH'] = os.pathsep.join([ROOT, os.path.join(ROOT, 'src')]) + os.pathsep + env.get('PYTHONPATH', '')\n\n"
            "cmd = [uv, 'run', '--no-project', script_path, '--config', config_path] + sys.argv[1:] + extra_recipe_args\n"
            "print('[launcher] exec:', ' '.join(cmd), file=sys.stderr)\n"
            "result = subprocess.run(cmd, env=env)\n"
            "sys.exit(result.returncode)\n"
        )


__all__ = ["LongDocPackager"]
