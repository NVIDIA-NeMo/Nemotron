"""Container bootstrap helpers must work before stage packages are installed."""

import subprocess
import sys
from pathlib import Path


def test_metadata_and_uv_helpers_import_without_site_packages():
    source = Path(__file__).resolve().parents[2] / "src"
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from nemo_runspec import parse; "
            "from nemo_runspec._pyproject import _write_temp_pyproject; "
            "assert 'fsspec' not in sys.modules",
            str(source),
        ],
        check=True,
        capture_output=True,
    )


def test_manifest_tracker_public_import_still_resolves():
    from nemo_runspec import ManifestTracker
    from nemo_runspec.manifest_tracker import ManifestTracker as TrackerImplementation

    assert ManifestTracker is TrackerImplementation
