"""The live tier's rate-limit preflight (support.skip_if_rate_limited), key-less: the provider is
a stub, so this runs with the scripted-fake set. See specs/testing.md ("Live tier")."""

from __future__ import annotations

from typing import Any

import pytest

import support
from support import PROVIDER_CONFIGS, real_chat_model, real_wica, skip_if_rate_limited
from wica.config import MissingSecretError

CONFIG = PROVIDER_CONFIGS[0]


class QuotaError(Exception):
    """Shaped like a provider SDK's error: the HTTP status on `status_code`."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"{status_code} quota exceeded for this key")
        self.status_code = status_code


class StubModel:
    def __init__(self, error: Exception | None) -> None:
        self.error = error
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "OK"


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch):
    """Replace the provider behind `build_chat_model` with a stub, on a fresh preflight cache."""

    def install(error: Exception | None) -> StubModel:
        model = StubModel(error)
        monkeypatch.setattr(support, "_PREFLIGHT", {})
        monkeypatch.setattr(support, "build_chat_model", lambda config: model)
        return model

    return install


def test_a_rate_limited_provider_skips_every_test_after_one_call(provider):
    model = provider(QuotaError(429))

    with pytest.raises(pytest.skip.Exception, match="429 quota exceeded"):
        real_chat_model(CONFIG)
    # A Wica-driven test skips too, before any Wica is built, and the provider is not called
    # again: one call per provider per session.
    with pytest.raises(pytest.skip.Exception, match="at preflight"):
        real_wica(CONFIG)
    assert model.calls == 1


def test_a_429_wrapped_in_another_error_is_found(provider):
    wrapped = RuntimeError("retries exhausted")
    wrapped.__cause__ = QuotaError(429)
    provider(wrapped)
    with pytest.raises(pytest.skip.Exception):
        skip_if_rate_limited(CONFIG)


@pytest.mark.parametrize("error", [None, QuotaError(500), ValueError("bad request")])
def test_a_provider_that_answers_or_fails_otherwise_does_not_skip(provider, error: Any):
    model = provider(error)
    skip_if_rate_limited(CONFIG)
    skip_if_rate_limited(CONFIG)
    assert model.calls == 1


def test_a_missing_key_is_not_the_preflights_case(monkeypatch: pytest.MonkeyPatch):
    """With no key the build raises before any call: the caller's skip-or-fail rule applies."""
    monkeypatch.setattr(support, "_PREFLIGHT", {})
    monkeypatch.delenv("WICA_ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingSecretError):
        skip_if_rate_limited(CONFIG)
    assert CONFIG not in support._PREFLIGHT
