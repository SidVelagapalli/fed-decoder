"""Groq (free tier) behind the same client.messages.create(...) shape that extract.py and explain.py already use.

Groq's API is OpenAI-compatible plain HTTP, so it's called with httpx (already installed): no new dependency.
The key is read from GROQ_API_KEY at call time and never logged.
"""

import os
import time
from types import SimpleNamespace

import httpx

URL = "https://api.groq.com/openai/v1/chat/completions"
# Strict JSON-schema mode only exists on gpt-oss models; no Llama chat model is available on the free key
# (checked against /models on 2026-11-06).
MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"  # gpt-oss thinks before answering; low keeps tokens inside the free 200K/day
STOP = {"stop": "end_turn", "length": "max_tokens"}  # map to the Anthropic stop reasons extract/explain check
MAX_WAIT = 120  # seconds; a longer retry-after means the daily limit, not the per-minute one


class DailyLimitReached(Exception):
    """The free tier's per-day quota is used up; stop and resume tomorrow."""


class GroqClient:
    def __init__(self, model: str = MODEL, post=None, sleep=None):
        self.model = model
        self._post = post or httpx.post
        self._sleep = sleep or time.sleep
        self.messages = self  # so callers can write client.messages.create(...)

    def create(self, *, model, max_tokens, system, messages, output_config=None):
        # `model` is extract.MODEL (an Anthropic id); Groq uses its own model.
        body = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": REASONING_EFFORT,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if output_config:
            schema = output_config["format"]["schema"]
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "signals", "strict": True, "schema": schema}}
        for _ in range(5):
            r = self._post(URL, headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                           json=body, timeout=60.0)
            if r.status_code != 429:
                break
            wait = float(r.headers.get("retry-after", 10))
            if wait > MAX_WAIT or "per day" in r.text:
                raise DailyLimitReached(f"Groq daily limit reached (retry-after {wait:.0f}s)")
            self._sleep(wait)
        r.raise_for_status()  # still 429 after 5 tries, or any other HTTP error: the run logs it as an error

        data = r.json()
        choice = data["choices"][0]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=choice["message"].get("content") or "")],
            stop_reason=STOP.get(choice["finish_reason"], choice["finish_reason"]),
            usage=SimpleNamespace(input_tokens=data["usage"]["prompt_tokens"],
                                  output_tokens=data["usage"]["completion_tokens"]),
        )
