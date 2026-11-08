import pytest

from fake_llm import FakeLLM


async def test_script_pops_in_order_and_records_calls():
    fake = FakeLLM()
    fake.script({"content": "one"}, {"content": "two"})
    assert (await fake.chat([{"role": "user", "content": "a"}], json_mode=True))["content"] == "one"
    assert (await fake.chat([]))["content"] == "two"
    assert fake.calls[0] == ([{"role": "user", "content": "a"}], None, True)
    assert len(fake.calls) == 2


async def test_empty_queue_raises_assertion():
    with pytest.raises(AssertionError, match="no scripted"):
        await FakeLLM().chat([])


async def test_embed_is_deterministic_normalized_and_meaningful():
    fake = FakeLLM()
    a, b, c = await fake.embed(["take with food", "Take with FOOD.", "kidney stones diet"])
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert len(a) == 64
    assert abs(dot(a, a) - 1) < 1e-9
    assert a == b
    assert dot(a, c) < 0.5


def test_embedding_model_name_is_configurable_and_read_only():
    assert FakeLLM().embedding_model == "fake-bag-of-words-64"
    fake = FakeLLM(embedding_model="other-model")
    assert fake.embedding_model == "other-model"
    with pytest.raises(AttributeError):
        fake.embedding_model = "changed"
