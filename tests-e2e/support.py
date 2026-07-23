from __future__ import annotations

import os

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set — skipping e2e test")
    return value


def real_chat_model(**kwargs) -> BaseChatModel:
    api_key = require_env("WICA_ANTHROPIC_API_KEY")
    model = os.environ.get("WICA_E2E_MODEL", "claude-haiku-4-5")
    return init_chat_model(model, model_provider="anthropic", api_key=api_key, **kwargs)
