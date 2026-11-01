"""Demo helper for the video: show one chunk, its two embeddings, and a retrieval.

    .venv/bin/python show_embedding_example.py --source ois-post-completion-opt --chunk 2 \
        --question "When can I apply for post-completion OPT?"
"""

from __future__ import annotations

import argparse

import numpy as np
from sqlalchemy import select

import settings
from corpus_db import chunk_embeddings, chunks, documents, load_snapshot, make_engine
from embeddings import from_blob, make_embedder
from histogram_embedding import HistogramEmbedder, histogram, tokenize
from rag import Retriever

HIST_MODEL = "w2v-hist-k256-idf-soft"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="ois-post-completion-opt")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--question", default="When can I apply for post-completion OPT?")
    a = p.parse_args()

    engine = make_engine(settings.SYNC_DATABASE_URL)
    with engine.connect() as c:
        row = c.execute(
            select(chunks.c.id, chunks.c.page, chunks.c.content, documents.c.title, documents.c.url)
            .join(documents, documents.c.id == chunks.c.document_id)
            .where(documents.c.source_id == a.source, chunks.c.chunk_index == a.chunk)
        ).one()
        vectors = dict(
            c.execute(select(chunk_embeddings.c.model, chunk_embeddings.c.embedding).where(chunk_embeddings.c.chunk_id == row.id)).all()
        )

    print("=" * 78)
    print(f"CHUNK  {a.source} #{a.chunk}  (page {row.page})  {row.title}")
    print(row.url)
    print("-" * 78)
    print(row.content[:700] + ("..." if len(row.content) > 700 else ""))
    print()

    v = from_blob(vectors["text-embedding-3-small"])
    print(f"OpenAI text-embedding-3-small: {v.shape[0]} dims, L2 norm {np.linalg.norm(v):.3f}")
    print("  first 8 values:", [round(float(x), 4) for x in v[:8]])
    print()

    emb: HistogramEmbedder = make_embedder(HIST_MODEL, models_dir=settings.MODELS_DIR)
    toks = tokenize(row.content)
    known = [t for t in toks if t in emb.art.vocab]
    print(f"Word histogram ({HIST_MODEL}): {len(toks)} tokens, {len(known)} in the Word2Vec vocabulary")
    print("  tokens:", " ".join(known[:25]) + (" ..." if len(known) > 25 else ""))
    h = histogram(toks, emb.art, emb.spec)
    top = np.argsort(-h)[:6]
    print(f"  {emb.spec.k} bins, {int((h > 0).sum())} non-zero; largest bins (bin: weight, closest words to that centroid):")
    for b in top:
        d = np.linalg.norm(emb.art.vectors - emb.art.centroids[b], axis=1)
        words = [w for w, i in sorted(emb.art.vocab.items(), key=lambda kv: d[kv[1]])[:5]]
        print(f"    bin {b:3d}: {h[b]:.3f}  {', '.join(words)}")
    hv = from_blob(vectors[HIST_MODEL])
    print(f"  stored vector: {hv.shape[0]} dims, L2 norm {np.linalg.norm(hv):.3f}, first 8 values {[round(float(x), 4) for x in hv[:8]]}")
    print()

    print("=" * 78)
    print(f"RETRIEVAL for: {a.question!r}")
    thresholds = settings.load_thresholds(settings.MODELS_DIR)
    for model in ("text-embedding-3-small", HIST_MODEL):
        e = make_embedder(model, api_key=settings.OPENAI_API_KEY, models_dir=settings.MODELS_DIR)
        r = Retriever(load_snapshot(engine, model), model)
        hits = r.search(e.embed([a.question])[0], 3, 0.0)
        print(f"  {model}  (not-covered threshold {settings.threshold_for(model, thresholds):.2f})")
        for hit in hits:
            print(f"    {hit.score:.2f}  {hit.title[:60]}  p.{hit.page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
