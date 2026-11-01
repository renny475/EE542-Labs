"""Retrieval and grounded answering.

The :class:`Retriever` holds a FAISS inner-product index over L2-normalised
chunk vectors (so inner product equals cosine similarity). :func:`answer`
embeds the question, retrieves, prompts the LLM and turns every failure mode
into a user-facing message instead of an exception.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date

import faiss
import httpx
import numpy as np

from corpus_db import CorpusSnapshot
from embeddings import Embedder, normalise

log = logging.getLogger("rag")

EXCERPT_CHARS = 600

NOT_COVERED = (
    "I couldn't find this in the OIS and USCIS materials I have. "
    "Please ask an OIS advisor: https://ois.usc.edu/contact/"
)
EMBED_ERROR = (
    "Sorry, I couldn't process your question right now (embedding service error). "
    "Please try again in a moment."
)
LLM_ERROR = "Sorry, the language model is not responding right now. Please try again in a moment."

SYSTEM_PROMPT = """You are the USC OIS Assistant, helping USC international students on F-1 status with questions about on-campus work, CPT, OPT, the STEM OPT extension, cap-gap, maintaining status, program extension and completion, travel and re-entry, and F-2 dependents.

Rules:
1. Answer only from the numbered sources below. Do not use outside knowledge.
2. After each sentence that relies on a source, cite it like [1] or [2][3].
3. If the sources do not answer the question, say so plainly and tell the student to ask an OIS advisor at https://ois.usc.edu/contact/.
4. For USC procedures (forms, requests, deadlines, processing times) prefer OIS sources. For the rule itself prefer USCIS, SEVP and CFR sources.
5. Be concise: a short paragraph or a few bullet points. Use plain English.
6. Never give legal advice, predict case outcomes, or tell the student what to do about a specific case; describe what the sources say and point them to OIS."""


class CorpusModelMismatch(RuntimeError):
    """The stored vectors were made with a different embedding model."""


@dataclass
class Hit:
    n: int
    document_id: int
    title: str
    tier: str
    url: str
    page: int | None
    published: date | None
    excerpt: str
    score: float
    content: str

    def to_source(self) -> dict:
        return {
            "n": self.n,
            "title": self.title,
            "url": self.url,
            "page": self.page,
            "published": self.published.isoformat() if self.published else None,
            "excerpt": self.excerpt,
            "score": round(self.score, 2),
        }


class Retriever:
    def __init__(self, snapshot: CorpusSnapshot, expected_model: str) -> None:
        if snapshot.chunks and snapshot.model != expected_model:
            raise CorpusModelMismatch(
                f"snapshot holds {snapshot.model!r} vectors but the retriever expects {expected_model!r}"
            )
        self.model = expected_model
        self._chunks = snapshot.chunks
        self.size = len(snapshot.chunks)
        self.document_count = len({c.document_id for c in snapshot.chunks})
        self._index: faiss.Index | None = None
        if self.size:
            vectors = np.ascontiguousarray(normalise(snapshot.vectors), dtype=np.float32)
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)

    def search(self, query_vec: Sequence[float], top_k: int, min_score: float) -> list[Hit]:
        if self._index is None or top_k <= 0:
            return []
        query = np.ascontiguousarray(normalise(np.asarray([query_vec], dtype=np.float32)))
        scores, ids = self._index.search(query, min(top_k, self.size))
        hits: list[Hit] = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0 or score < min_score:
                continue
            meta = self._chunks[int(idx)]
            hits.append(
                Hit(
                    n=len(hits) + 1,
                    document_id=meta.document_id,
                    title=meta.title,
                    tier=meta.tier,
                    url=meta.url,
                    page=meta.page,
                    published=meta.published,
                    excerpt=meta.content[:EXCERPT_CHARS],
                    score=float(score),
                    content=meta.content,
                )
            )
        return hits


def _source_block(hit: Hit) -> str:
    where = f"{hit.tier}, p.{hit.page}" if hit.page is not None else hit.tier
    return f"[{hit.n}] {hit.title} ({where})\n{hit.content}"


def build_messages(question: str, hits: list[Hit], history: list[dict]) -> list[dict]:
    system = SYSTEM_PROMPT + "\n\nSources:\n" + "\n\n".join(_source_block(h) for h in hits)
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]


ChatFn = Callable[[list[dict]], Awaitable[str]]


class LLMClient:
    """Minimal OpenAI-compatible chat client for llama-server."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def chat(self, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 500}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError(f"model returned no text (content={content!r})")
        return content


@dataclass
class BotAnswer:
    content: str
    sources: list[dict]


async def answer(
    question: str,
    history: list[dict],
    *,
    retriever: Retriever,
    embedder: Embedder,
    chat: ChatFn,
    top_k: int,
    min_score: float,
) -> BotAnswer:
    try:
        query_vec = (await asyncio.to_thread(embedder.embed, [question]))[0]
    except Exception:
        log.exception("embedding the question failed")
        return BotAnswer(EMBED_ERROR, [])

    hits = retriever.search(query_vec, top_k, min_score)
    if not hits:
        return BotAnswer(NOT_COVERED, [])

    try:
        text = (await chat(build_messages(question, hits, history))).strip()
    except Exception:
        log.exception("LLM call failed")
        return BotAnswer(LLM_ERROR, [])
    return BotAnswer(text, [h.to_source() for h in hits])
