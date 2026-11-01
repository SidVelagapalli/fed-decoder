from types import SimpleNamespace

import anthropic
import pytest


class FakeClient:
    """Same shape as anthropic.Anthropic for what we use: client.messages.create(**kwargs)."""

    def __init__(self, reply_text=""):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._reply = reply_text

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._reply)])


@pytest.fixture(autouse=True)
def no_real_anthropic(monkeypatch):
    # Proves no test builds a real (paid) client.
    def boom(*a, **k):
        raise AssertionError("tests must not construct anthropic.Anthropic")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", boom)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
