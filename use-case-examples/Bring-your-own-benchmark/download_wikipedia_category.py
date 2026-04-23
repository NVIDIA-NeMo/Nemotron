#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Download Wikipedia article text for pages in a category tree.

This script writes one ``.txt`` file per article to::

    <output_dir>/<target_subject>/

That layout matches BYOB ``McqByobDataset`` expectations: keys in ``target_source_mapping``
correspond to ``input_dir/<target_subject>/*.txt``.

**Examples**

* Fixed depth and article limit::

      python download_wikipedia_category.py \\
          --category \"Finance in India\" \\
          --target-subject finance_india \\
          --max-articles 25

* Depth (subcategories, breadth-first; ``--depth 0`` is the default, root only)::

      # Depth 0: pages listed directly in the category.
      # Depth 1: also pages in immediate subcategories, and so on.
      python download_wikipedia_category.py --depth 2 --max-articles 100

* Full recursive subtree (root category and all nested subcategories). Reference:
  https://en.wikipedia.org/wiki/Category:Finance_in_India ::

      # ``--max-articles 0`` removes the cap on collected titles (long runs, many API calls).
      python download_wikipedia_category.py --recursive --max-articles 0 --delay 0.35

Set ``input_dir`` in the BYOB configuration to ``--output-dir`` (or the default
under ``use-case-examples/Bring-your-own-benchmark`` (e.g. ``byob_seed_preparation.ipynb``).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_BASE = Path(__file__).resolve().parent / "input" / "wikipedia"


def _category_title(name: str) -> str:
    """Normalize a category name to MediaWiki ``Category:...`` title form."""
    name = name.strip()
    if not name.lower().startswith("category:"):
        return "Category:" + name.replace(" ", "_")
    return name[0:9] + name[9:].strip().replace(" ", "_")


def _safe_filename(title: str) -> str:
    """Return a filesystem-safe basename stem for ``title`` (length-limited)."""
    s = re.sub(r"[^\w\-_.]+", "_", title, flags=re.UNICODE)
    s = s.strip("_") or "untitled"
    return s[:180]


def _api_get(params: dict[str, str], api_url: str, delay_s: float) -> dict:
    """Execute a single GET to the MediaWiki API and return the JSON body."""
    query = urllib.parse.urlencode(params)
    url = f"{api_url}?{query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Nemotron-BYOB-wikipedia-fetch/1.0 (data prep; https://github.com/NVIDIA-NeMo/Nemotron)"},
    )
    time.sleep(delay_s)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _paginate_category_members(
    cmtitle: str,
    *,
    api_url: str,
    delay_s: float,
    cmnamespace: str,
    cmtype: str,
) -> list[dict]:
    """List all members of ``cmtitle`` for the given namespace and type (follows ``cmcontinue``)."""
    members: list[dict] = []
    continue_token: str | None = None

    while True:
        params: dict[str, str] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": cmtitle,
            "cmnamespace": cmnamespace,
            "cmtype": cmtype,
            "cmlimit": "50",
        }
        if continue_token:
            params["cmcontinue"] = continue_token

        data = _api_get(params, api_url, delay_s)
        batch = data.get("query", {}).get("categorymembers", [])
        members.extend(batch)
        cont = data.get("continue", {})
        continue_token = cont.get("cmcontinue")
        if not continue_token or not batch:
            break

    return members


def collect_article_titles_by_depth(
    category_display: str,
    *,
    api_url: str,
    max_depth: int,
    max_articles: int,
    delay_s: float,
) -> tuple[list[str], dict[int, int]]:
    """Collect article titles from a category tree up to ``max_depth`` (breadth-first).

    Depth 0 includes only pages in the root category; greater depths add pages from
    subcategories at each deeper tier.

    Returns:
        A tuple of (ordered unique article titles, per-depth count of new articles).
    """

    root = _category_title(category_display)
    seen_titles: set[str] = set()
    ordered: list[str] = []
    stats: dict[int, int] = {}
    visited_categories: set[str] = set()

    current_level: list[str] = [root]

    for depth in range(max_depth + 1):
        at_depth_added = 0
        next_categories: list[str] = []
        queued_next: set[str] = set()

        for cmtitle in current_level:
            if cmtitle in visited_categories:
                continue
            visited_categories.add(cmtitle)

            pages = _paginate_category_members(
                cmtitle,
                api_url=api_url,
                delay_s=delay_s,
                cmnamespace="0",
                cmtype="page",
            )
            for m in pages:
                if len(ordered) >= max_articles:
                    break
                t = m.get("title")
                if not t or t in seen_titles:
                    continue
                seen_titles.add(t)
                ordered.append(t)
                at_depth_added += 1

            if len(ordered) >= max_articles:
                break

            # Enqueue subcategories for the next depth when the cap is not yet reached.
            if depth < max_depth and len(ordered) < max_articles:
                subcats = _paginate_category_members(
                    cmtitle,
                    api_url=api_url,
                    delay_s=delay_s,
                    cmnamespace="14",
                    cmtype="subcat",
                )
                for sm in subcats:
                    st = sm.get("title")
                    if (
                        st
                        and st not in visited_categories
                        and st not in queued_next
                    ):
                        queued_next.add(st)
                        next_categories.append(st)

        stats[depth] = at_depth_added
        log.info(
            "Depth %d: +%d new article(s) (total %d/%d); next tier: %d subcategor(ies) to walk",
            depth,
            at_depth_added,
            len(ordered),
            max_articles,
            len(next_categories),
        )

        if len(ordered) >= max_articles:
            break
        if depth >= max_depth:
            break

        current_level = next_categories
        if not current_level:
            log.info("No further subcategories below depth %d; stopping.", depth)
            break

    return ordered[:max_articles], stats


def collect_article_titles_recursive(
    category_display: str,
    *,
    api_url: str,
    max_articles: int | None,
    delay_s: float,
) -> tuple[list[str], dict[str, int]]:
    """Traverse the full category subtree in breadth-first order (all nested subcategories).

    Each category is visited at most once. Article titles are unique. Collection
    stops when ``max_articles`` is reached, or when the queue is empty.

    Args:
        max_articles: ``None`` or ``0`` means no cap on the number of titles.

    Returns:
        ``(titles, stats)`` with ``stats`` keys ``categories_visited`` and
        ``articles_collected``.
    """

    root = _category_title(category_display)
    seen_titles: set[str] = set()
    ordered: list[str] = []
    visited_categories: set[str] = set()
    q: deque[str] = deque([root])
    cap = None if max_articles in (None, 0) else int(max_articles)

    while q:
        cmtitle = q.popleft()
        if cmtitle in visited_categories:
            continue
        visited_categories.add(cmtitle)

        pages = _paginate_category_members(
            cmtitle,
            api_url=api_url,
            delay_s=delay_s,
            cmnamespace="0",
            cmtype="page",
        )
        for m in pages:
            if cap is not None and len(ordered) >= cap:
                break
            t = m.get("title")
            if not t or t in seen_titles:
                continue
            seen_titles.add(t)
            ordered.append(t)

        if cap is not None and len(ordered) >= cap:
            log.info(
                "Reached --max-articles=%d; stopping traversal (%d categories visited).",
                cap,
                len(visited_categories),
            )
            break

        subcats = _paginate_category_members(
            cmtitle,
            api_url=api_url,
            delay_s=delay_s,
            cmnamespace="14",
            cmtype="subcat",
        )
        for sm in subcats:
            st = sm.get("title")
            if st and st not in visited_categories:
                q.append(st)

        if len(visited_categories) % 15 == 0:
            log.info(
                "Recursive crawl: %d categories visited, %d unique articles, %d categories queued",
                len(visited_categories),
                len(ordered),
                len(q),
            )

    stats = {
        "categories_visited": len(visited_categories),
        "articles_collected": len(ordered),
    }
    return ordered, stats


def list_category_articles(
    category_display: str,
    *,
    api_url: str,
    max_articles: int,
    delay_s: float,
) -> list[str]:
    """List article titles in the root category only (``max_depth=0``)."""
    titles, _ = collect_article_titles_by_depth(
        category_display,
        api_url=api_url,
        max_depth=0,
        max_articles=max_articles,
        delay_s=delay_s,
    )
    return titles


def fetch_extracts(
    titles: list[str],
    *,
    api_url: str,
    delay_s: float,
    exchars: int,
) -> dict[str, str]:
    """Return a mapping of page title to plain-text extract.

    Uses one title per API request: batched multi-title ``extracts`` calls can
    return empty strings for long pages. A delay is applied between requests.
    """

    out: dict[str, str] = {}
    for title in titles:
        params: dict[str, str] = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "redirects": "1",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": title,
        }
        if exchars > 0:
            params["exchars"] = str(min(exchars, 1_000_000))

        data = _api_get(params, api_url, delay_s)
        pages = data.get("query", {}).get("pages", {})
        extract = ""
        resolved_title = title
        for _pid, page in pages.items():
            resolved_title = page.get("title", title)
            extract = page.get("extract") or ""
            if page.get("missing"):
                log.warning("Missing page: %r", title)
        if extract:
            out[resolved_title] = extract
            continue

        # Fallback: parse full page HTML when the extracts API returns no text.
        fallback = _fetch_parse_plaintext(title, api_url=api_url, delay_s=delay_s)
        if fallback:
            out[resolved_title] = fallback
        else:
            log.warning("Could not get text for %r", title)

    return out


def _fetch_parse_plaintext(title: str, *, api_url: str, delay_s: float) -> str:
    """Obtain plain text via ``action=parse`` (HTML) when ``prop=extracts`` is insufficient."""

    params: dict[str, str] = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "formatversion": "2",
    }
    try:
        data = _api_get(params, api_url, delay_s)
    except urllib.error.HTTPError as e:
        log.debug("parse HTTP error for %r: %s", title, e)
        return ""
    raw_html = data.get("parse", {}).get("text", "") or ""
    if not raw_html:
        return ""
    # Remove scripts and style blocks, strip tags, and unescape HTML entities.
    raw_html = re.sub(r"(?is)<script[^>]*>.*?</script>", "", raw_html)
    raw_html = re.sub(r"(?is)<style[^>]*>.*?</style>", "", raw_html)
    plain = re.sub(r"<[^>]+>", " ", raw_html)
    plain = html.unescape(plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain


def write_articles(
    texts: dict[str, str],
    out_dir: Path,
) -> int:
    """Write each article to ``<out_dir>/<safe_title>.txt``; return the number written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for title, body in texts.items():
        path = out_dir / f"{_safe_filename(title)}.txt"
        path.write_text(f"# {title}\n\n{body}", encoding="utf-8")
        n += 1
    return n


def main() -> None:
    """Parse arguments, list titles, fetch extracts, and write text files for BYOB."""
    parser = argparse.ArgumentParser(
        description="Download Wikipedia pages under a category as .txt files for a BYOB corpus ``input_dir``."
    )
    parser.add_argument(
        "--category",
        default="Finance in India",
        help='Wikipedia category (with or without "Category:" prefix), e.g. "Finance in India"',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Root folder; articles go under <output-dir>/<target-subject>/. Default: {DEFAULT_BASE}",
    )
    parser.add_argument(
        "--target-subject",
        default="finance_india",
        help="Subdirectory name matching BYOB target_source_mapping key (e.g. finance_docs, finance_india).",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=40,
        help=(
            "Maximum article titles to collect then download. "
            "With --recursive, use 0 for no limit (entire subtree; may take a long time and large disk)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk the full category tree: [Category:Finance in India](https://en.wikipedia.org/wiki/Category:Finance_in_India) "
            "and every nested subcategory, breadth-first, until the queue is empty (or --max-articles reached). "
            "Ignores --depth."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help=(
            "Category-tree depth (ignored if --recursive). 0 = only pages in the root category; "
            "1 = also pages in immediate subcategories; 2 = one more level; and so on. "
            "Visits each subcategory once (cycles skipped)."
        ),
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language code (builds https://{lang}.wikipedia.org/w/api.php).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to sleep between API requests (be polite to Wikimedia).",
    )
    parser.add_argument(
        "--exchars",
        type=int,
        default=80_000,
        help="Max characters per article extract (0 = API default / shorter snippets).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    api_url = f"https://{args.lang}.wikipedia.org/w/api.php"
    dest = args.output_dir.resolve() / args.target_subject

    log.info("Category (browse): %s", _category_title(args.category))
    log.info("API: %s", api_url)
    log.info("Output: %s", dest)

    if args.recursive:
        log.info("Mode: FULL recursive subtree (nested subcategories until done)")
        ma = None if args.max_articles == 0 else args.max_articles
        if ma is None:
            log.warning(
                "No --max-articles cap (0): downloading every unique article in the subtree. "
                "This can be slow; use a positive --max-articles to sample."
            )
        titles, rec_stats = collect_article_titles_recursive(
            args.category,
            api_url=api_url,
            max_articles=ma,
            delay_s=args.delay,
        )
        log.info("Recursive crawl stats: %s", rec_stats)
    else:
        log.info("Max depth: %d (0 = root category only)", args.depth)
        titles, depth_stats = collect_article_titles_by_depth(
            args.category,
            api_url=api_url,
            max_depth=args.depth,
            max_articles=args.max_articles,
            delay_s=args.delay,
        )
        log.info("Depth breakdown (new articles per depth): %s", depth_stats)
    log.info("Found %d article title(s) to fetch", len(titles))
    if not titles:
        log.error(
            "No articles in this category (check spelling; category may be empty or only subcategories). "
            "Try another category name that exists on %s.wikipedia.org.",
            args.lang,
        )
        raise SystemExit(1)

    texts = fetch_extracts(
        titles,
        api_url=api_url,
        delay_s=args.delay,
        exchars=args.exchars,
    )
    written = write_articles(texts, dest)
    log.info("Wrote %d file(s) under %s", written, dest)
    log.info(
        "BYOB: set input_dir=%s and use target_source_mapping keys that include %r",
        args.output_dir.resolve(),
        args.target_subject,
    )


if __name__ == "__main__":
    main()
