import hashlib
import math
import re


class FakeLLM:
    """Scripted stand-in for LLMClient. Same chat/embed interface."""

    DIM = 64

    def __init__(self, embedding_model: str = "fake-bag-of-words-64"):
        self.queue: list[dict] = []
        self.calls: list[tuple] = []
        self._embedding_model = embedding_model

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def script(self, *messages: dict) -> None:
        self.queue.extend(messages)

    async def chat(self, messages, tools=None, json_mode=False):
        self.calls.append((messages, tools, json_mode))
        if not self.queue:
            raise AssertionError(
                f"FakeLLM.chat called with no scripted response left (call #{len(self.calls)})"
            )
        return self.queue.pop(0)

    async def embed(self, texts):
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            values[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            values[0], norm = 1.0, 1.0
        return [v / norm for v in values]
