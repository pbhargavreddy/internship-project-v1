"""
test_llm.py
-----------
Offline test for the LLM provider switch in backend/llm/__init__.py.
"""

from __future__ import annotations

import pytest

import backend.llm as llm
from backend.config import Settings


def test_get_llm_returns_api_client_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, llm_api_key="test-key")  # ignore the developer's .env
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "_llm", None)

    client = llm.get_llm()

    assert settings.llm_provider == "api"
    assert isinstance(client, llm.ApiLLM)
    assert llm.get_llm() is client  # cached
