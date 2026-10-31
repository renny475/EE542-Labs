"""Recall@k check for the retriever against a small golden question set.

Each question lists the document source_ids that would answer it; the
question passes when any one of them appears among the top-k hits.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import select

import settings
from corpus_db import documents, load_snapshot, make_engine
from embeddings import Embedder, OpenAIEmbedder
from rag import Retriever


@dataclass
class EvalQuestion:
    question: str
    expect: list[str]


@dataclass
class EvalResult:
    question: str
    expect: list[str]
    hit_ids: list[str]
    ok: bool


def load_eval(path: Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [EvalQuestion(question=str(item["question"]), expect=[str(x) for x in item["expect"]]) for item in raw]


def evaluate(
    questions: list[EvalQuestion],
    retriever: Retriever,
    embedder: Embedder,
    source_id_of: Callable[[int], str],
    top_k: int = 5,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for q in questions:
        # One call per question: this is how the app embeds a live question too.
        vec = embedder.embed([q.question])[0]
        hits = retriever.search(vec, top_k, min_score=0.0)
        hit_ids: list[str] = []
        for h in hits:
            sid = source_id_of(h.document_id)
            if sid not in hit_ids:
                hit_ids.append(sid)
        results.append(EvalResult(q.question, q.expect, hit_ids, any(sid in hit_ids for sid in q.expect)))
    return results


def main() -> int:
    engine = make_engine(settings.SYNC_DATABASE_URL)
    retriever = Retriever(load_snapshot(engine), settings.EMBEDDING_MODEL)
    with engine.connect() as conn:
        id_to_source = dict(conn.execute(select(documents.c.id, documents.c.source_id)).all())
    embedder = OpenAIEmbedder(settings.EMBEDDING_MODEL, settings.OPENAI_API_KEY)
    questions = load_eval(settings.DATA_DIR / "eval_questions.yaml")

    results = evaluate(questions, retriever, embedder, id_to_source.__getitem__, top_k=settings.RAG_TOP_K)
    for r in results:
        print(f"{'OK  ' if r.ok else 'MISS'} {r.question}  ->  {', '.join(r.hit_ids) or '-'}")
    passed = sum(r.ok for r in results)
    print(f"recall@{settings.RAG_TOP_K}: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
