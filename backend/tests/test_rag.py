import asyncio
from datetime import date

import numpy as np
import pytest

from corpus_db import ChunkMeta, CorpusSnapshot
from embeddings import FakeEmbedder
from rag import (
    EMBED_ERROR,
    LLM_ERROR,
    NOT_COVERED,
    SYSTEM_PROMPT,
    CorpusModelMismatch,
    Retriever,
    answer,
    build_messages,
)


def meta(i, content="c", page=None):
    return ChunkMeta(
        chunk_id=i, document_id=1, title=f"Doc{i}", tier="ois", url=f"https://x/{i}",
        page=page, published=None, content=content,
    )


def snapshot(vectors, model="fake"):
    v = np.array(vectors, dtype=np.float32)
    return CorpusSnapshot(
        chunks=[meta(i, content="x" * 700, page=3) for i in range(len(vectors))],
        vectors=v, model=model, document_count=1, corpus_as_of=date(2026, 10, 31),
    )


def test_search_ranks_and_applies_threshold():
    r = Retriever(snapshot([[1, 0], [0.8, 0.6], [0, 1]]), expected_model="fake")
    hits = r.search([1, 0], top_k=5, min_score=0.5)
    assert [h.title for h in hits] == ["Doc0", "Doc1"] and [h.n for h in hits] == [1, 2]
    assert hits[0].score > hits[1].score and len(hits[0].excerpt) == 600 and hits[0].page == 3


def test_search_respects_top_k_and_empty_corpus():
    r = Retriever(snapshot([[1, 0], [0.9, 0.1], [0.8, 0.2]]), expected_model="fake")
    assert len(r.search([1, 0], top_k=2, min_score=0.0)) == 2
    empty = Retriever(CorpusSnapshot([], np.zeros((0, 0), np.float32), "fake", 0, None), expected_model="fake")
    assert empty.size == 0 and empty.search([1, 0], top_k=5, min_score=0.0) == []


def test_retriever_rejects_model_mismatch():
    with pytest.raises(CorpusModelMismatch):
        Retriever(snapshot([[1, 0]], model="text-embedding-3-small"), expected_model="fake")


def test_hit_carries_document_id():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    assert r.search([1, 0], top_k=1, min_score=0.0)[0].document_id == 1


def test_build_messages_layout():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    hits = r.search([1, 0], top_k=1, min_score=0.0)
    msgs = build_messages(
        "When can I apply?", hits, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(SYSTEM_PROMPT)
    assert "[1] Doc0 (ois, p.3)" in msgs[0]["content"]
    assert msgs[1:3] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert msgs[-1] == {"role": "user", "content": "When can I apply?"}


class Emb(FakeEmbedder):
    def embed(self, texts):
        return [[1.0, 0.0]]


class BrokenEmb(FakeEmbedder):
    def embed(self, texts):
        raise RuntimeError("401")


def run(coro):
    return asyncio.run(coro)


def test_answer_happy_path_returns_sources():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    seen = {}

    async def chat(msgs):
        seen["msgs"] = msgs
        return "  You may apply 90 days before [1]. "

    out = run(answer("q", [], retriever=r, embedder=Emb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == "You may apply 90 days before [1]."
    assert out.sources == [
        {"n": 1, "title": "Doc0", "url": "https://x/0", "page": 3, "published": None, "excerpt": "x" * 600, "score": 1.0}
    ]
    assert seen["msgs"][-1]["content"] == "q"


def test_answer_not_covered_skips_llm():
    r = Retriever(snapshot([[0, 1]]), expected_model="fake")

    async def chat(msgs):
        raise AssertionError("must not be called")

    out = run(answer("q", [], retriever=r, embedder=Emb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == NOT_COVERED and out.sources == []


def test_answer_when_embedding_fails():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")

    async def chat(msgs):
        raise AssertionError("must not be called")

    out = run(answer("q", [], retriever=r, embedder=BrokenEmb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == EMBED_ERROR and out.sources == []


def test_answer_when_llm_fails():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")

    async def chat(msgs):
        raise TimeoutError("read timeout")

    out = run(answer("q", [], retriever=r, embedder=Emb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == LLM_ERROR and out.sources == []


def test_answer_when_llm_returns_no_text():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")

    async def chat(msgs):
        return None  # llama-server replied with "content": null

    out = run(answer("q", [], retriever=r, embedder=Emb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == LLM_ERROR and out.sources == []


def test_hit_source_includes_score():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    hit = r.search([1, 0], top_k=1, min_score=0.0)[0]
    assert hit.to_source()["score"] == round(hit.score, 2) == 1.0


def test_retriever_zero_query_yields_no_hits():
    r = Retriever(snapshot([[1, 0], [0, 1]]), expected_model="fake")
    assert r.search([0.0, 0.0], top_k=5, min_score=0.3) == []
