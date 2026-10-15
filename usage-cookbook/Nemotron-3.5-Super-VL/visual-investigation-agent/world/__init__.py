# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fernledger, a fictional company: the help desk's data, dashboard, and documents.

This package stands in for the systems a real team already has: a data warehouse (read-only SQL), a BI
tool that exports dashboard views as images, and a document store. To use the agent on your own
dashboards, replace this package; the agent's tools are the only code that calls it.

Besides the four hand-built Fernledger months, world/generator.py makes other worlds (other companies,
models, ticket types, dates, and causes) from a seed, for testing whether the agent generalizes.
"""

from .browser import find_browser
from .dashboard import DASHBOARDS, dashboards, page, render, render_path
from .data import (DEFAULT_COMPARE, DEFAULT_PERIOD, FIRST_DAY, LAST_DAY, PASS_PCT, PRICES, SCENARIOS, SLUGS,
                   TICKET_TYPES, WORLD, build_database, catalog, connect, content_sha256, current_prices, dates,
                   db_path, eval_in_db, model_tokens, pass_pct, price, stats, ticket_types)
from .documents import EVAL_REPORT, NEW_SCHEDULE, build_documents, document, images_ready, list_documents
from .spec import register, spec


def build(scenario, images=True):
    """Generate the scenario's database and documents. Without images, no browser is needed."""
    build_database(scenario)
    build_documents(scenario, images=images)
    return WORLD / scenario


def ensure(scenario, images=False):
    """Build the scenario once; the data is deterministic, so a cached build is identical."""
    ready = db_path(scenario).exists() and (not images or images_ready(scenario))
    return WORLD / scenario if ready else build(scenario, images=images)
