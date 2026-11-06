import json
from types import SimpleNamespace

import pytest

from src import extract, groq_client
from src.extract import FIELDS

STATEMENT = "Inflation remains elevated. The Committee will keep tightening policy."


def reply(content, finish="stop", status=200, headers=None, text=""):
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    return SimpleNamespace(status_code=status, headers=headers or {}, text=text, json=lambda: body,
                           raise_for_status=lambda: None)


class FakePost:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


def test_extract_on_groq_sends_strict_schema_and_validates_quotes():
    signals = {f: {"level": "not_mentioned", "evidence_quote": ""} for f in FIELDS}
    signals["inflation_concern"] = {"level": "high", "evidence_quote": "Inflation remains elevated."}
    signals["growth_view"] = {"level": "strong", "evidence_quote": "growth is booming"}  # not in the statement
    post = FakePost(reply(json.dumps(signals)))
    out = extract.extract(groq_client.GroqClient(post=post), STATEMENT, "")

    body = post.calls[0]["json"]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"][0] == {"role": "system", "content": extract.SYSTEM}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == extract.JSON_SCHEMA
    assert out["inflation_concern"] == {"level": "high", "evidence_quote": "Inflation remains elevated."}
    assert out["growth_view"]["invalid"] == "quote not in statement"  # validate() unchanged on the Groq path


def test_truncated_groq_reply_maps_to_max_tokens_and_fails_extract():
    post = FakePost(reply('{"inflation_concern": {"le', finish="length"))
    out = extract.extract(groq_client.GroqClient(post=post), STATEMENT, "")
    assert all(v["invalid"] == "stop_reason max_tokens" for v in out.values())


def test_usage_is_reported_like_anthropic():
    r = groq_client.GroqClient(post=FakePost(reply("hi"))).messages.create(
        model="x", max_tokens=10, system="s", messages=[{"role": "user", "content": "u"}])
    assert (r.usage.input_tokens, r.usage.output_tokens, r.stop_reason) == (100, 20, "end_turn")
    assert r.content[0].text == "hi"


def test_per_minute_limit_waits_then_retries():
    waits = []
    post = FakePost(reply("", status=429, headers={"retry-after": "3"}, text="tokens per minute (TPM)"), reply("ok"))
    r = groq_client.GroqClient(post=post, sleep=waits.append).create(
        model="x", max_tokens=10, system="s", messages=[])
    assert waits == [3.0] and r.content[0].text == "ok"


def test_daily_limit_raises_instead_of_waiting_hours():
    post = FakePost(reply("", status=429, headers={"retry-after": "4000"}, text="tokens per day (TPD)"))
    with pytest.raises(groq_client.DailyLimitReached):
        groq_client.GroqClient(post=post, sleep=lambda s: None).create(model="x", max_tokens=10, system="s", messages=[])


def test_key_comes_from_env_and_real_http_is_blocked_in_tests():
    post = FakePost(reply("ok"))
    groq_client.GroqClient(post=post).create(model="x", max_tokens=10, system="s", messages=[])
    assert post.calls[0]["headers"]["Authorization"] == "Bearer test-key-not-real"  # conftest's fake key
    with pytest.raises(AssertionError):  # conftest guard: no real httpx.post
        groq_client.GroqClient().create(model="x", max_tokens=10, system="s", messages=[])
