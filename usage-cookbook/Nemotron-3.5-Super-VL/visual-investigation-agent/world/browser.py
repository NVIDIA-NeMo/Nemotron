# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Screenshot an HTML page with headless Chrome or Chromium. Only live runs and image rebuilds need this."""

import os
import shutil
import subprocess
import threading
from pathlib import Path

CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]
render_lock = threading.Lock()  # one screenshot at a time


def find_browser():
    """Return the browser path from CHROME_PATH or a standard location, or None."""
    for candidate in [os.environ.get("CHROME_PATH")] + CANDIDATES:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return shutil.which(candidate) or candidate
    return None


def screenshot(html_path, png_path, size):
    browser = find_browser()
    if browser is None:
        raise RuntimeError("Chrome or Chromium is needed to render images; install one or set CHROME_PATH.")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    f"--window-size={size[0]},{size[1]}", f"--screenshot={png_path}", Path(html_path).resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    return png_path


def to_jpeg(png_path, quality=70):
    from PIL import Image

    jpg = Path(png_path).with_suffix(".jpg")
    Image.open(png_path).convert("RGB").save(jpg, "JPEG", quality=quality)
    return jpg
