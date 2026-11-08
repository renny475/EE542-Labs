import json

import httpx
import pytest

from config import Settings
from llm import LLMClient, LLMError


def make_client(handler, **overrides):
    settings = Settings(
        llm_api_base="https://llm.example/v1",
        llm_model="test-model",
        llm_api_key="key-123",
        embedding_model="embed-model",
        **overrides,
    )
    return LLMClient(settings, transport=httpx.MockTransport(handler))


async def test_chat_sends_model_tools_and_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    msg = await make_client(handler).chat([{"role": "user", "content": "yo"}], tools=tools, json_mode=True)
    assert msg["content"] == "hi"
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer key-123"
    body = seen["body"]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    assert body["tools"] == tools
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "user", "content": "yo"}]


async def test_chat_omits_optional_fields_by_default():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    await make_client(handler).chat([{"role": "user", "content": "x"}])
    assert "tools" not in seen["body"]
    assert "response_format" not in seen["body"]


async def test_chat_returns_tool_calls():
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [call]}}]})

    msg = await make_client(handler).chat([{"role": "user", "content": "x"}])
    assert msg["tool_calls"] == [call]


async def test_chat_http_error_raises_llmerror():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError):
        await make_client(handler).chat([{"role": "user", "content": "x"}])


async def test_chat_bad_shape_raises_llmerror():
    def handler(request):
        return httpx.Response(200, json={"nope": 1})

    with pytest.raises(LLMError):
        await make_client(handler).chat([{"role": "user", "content": "x"}])


async def test_embed_batches_and_orders_by_index():
    batches = []

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "embed-model"
        assert request.headers["authorization"] == "Bearer key-123"
        batches.append(body["input"])
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})

    texts = ["a" * (i + 1) for i in range(70)]
    vectors = await make_client(handler).embed(texts)
    assert [len(b) for b in batches] == [64, 6]
    assert [v[0] for v in vectors] == [float(i + 1) for i in range(70)]


async def test_embed_rejects_zero_vector():
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0]}]})

    with pytest.raises(LLMError):
        await make_client(handler).embed(["x"])


async def test_embed_rejects_incomplete_batch():
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    with pytest.raises(LLMError):
        await make_client(handler).embed(["a", "b"])


def test_embedding_model_is_exposed_read_only():
    client = make_client(lambda request: httpx.Response(500))
    assert client.embedding_model == "embed-model"
    with pytest.raises(AttributeError):
        client.embedding_model = "other"
