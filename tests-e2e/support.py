from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from wica.agent import Agent
from wica.config import MissingEnvError, WicaConfig

E2E_CONFIG = Path(__file__).parent / "e2e.config.json"


def _load_config() -> WicaConfig:
    try:
        return WicaConfig.from_json(E2E_CONFIG)
    except MissingEnvError as exc:
        pytest.skip(f"{exc.env_var} not set — skipping e2e test")


def real_chat_model(**kwargs: Any) -> BaseChatModel:
    cfg = _load_config().agent
    model_kwargs = dict(cfg.model_kwargs, **kwargs)
    if cfg.api_key is not None:
        model_kwargs.setdefault("api_key", cfg.api_key)
    return init_chat_model(cfg.model, model_provider=cfg.provider, **model_kwargs)


def real_agent(**kwargs: Any) -> Agent:
    """Build an Agent through the full config pipeline (file -> WicaConfig -> Agent), so the e2e
    tier exercises this pipeline against a live provider, not just a model builder. Deliberately
    doesn't call apply_logging: that's already covered deterministically by tests/test_config.py,
    and applying it here would reset the wica logger on every test, fighting any level a developer
    sets by hand while debugging a live e2e run."""
    return Agent.from_config(_load_config().agent, **kwargs)
