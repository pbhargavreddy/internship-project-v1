"""
conftest.py
-----------
Shared pytest configuration.

Tests marked ``@pytest.mark.live`` hit real external APIs and are skipped
unless pytest is run with ``--live``.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live", action="store_true", default=False, help="run tests that call real external APIs"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: test calls real external APIs (needs --live)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="live API test; run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
