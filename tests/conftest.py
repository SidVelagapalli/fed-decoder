from types import SimpleNamespace

import anthropic
import httpx
import pytest


class FakeClient:
    """Same shape as anthropic.Anthropic for what we use: client.messages.create(**kwargs)."""

    def __init__(self, reply_text="", stop_reason="end_turn"):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._reply = reply_text
        self._stop = stop_reason

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._reply)], stop_reason=self._stop)


@pytest.fixture(autouse=True)
def no_real_anthropic(monkeypatch):
    # Proves no test builds a real (paid) client.
    def boom(*a, **k):
        raise AssertionError("tests must not construct anthropic.Anthropic")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", boom)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Same for Groq: GroqClient posts with httpx.post unless a test passes its own fake.
    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
