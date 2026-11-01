"""Fill ``chunk_embeddings`` for one embedding model.

Idempotent per model: only chunks without a vector for ``--model`` are
embedded; ``--refresh`` recomputes all of them. Run after ``ingest.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from sqlalchemy import Engine, func, select

import settings
from corpus_db import (
    chunks,
    chunks_missing_embedding,
    create_tables,
    delete_embeddings,
    make_engine,
    models_present,
    upsert_embeddings,
)
from embeddings import Embedder, make_embedder, to_blob

log = logging.getLogger("embed_corpus")


@dataclass
class EmbedReport:
    model: str
    embedded: int
    skipped: int
    dim: int


def embed_corpus(engine: Engine, embedder: Embedder, *, refresh: bool = False, batch_size: int = 100) -> EmbedReport:
    model = embedder.model
    with engine.begin() as conn:
        if refresh:
            delete_embeddings(conn, model)
        todo = chunks_missing_embedding(conn, model)
    skipped = models_present(engine).get(model, 0)

    embedded = 0
    dim = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors = embedder.embed([content for _, content in batch])
        if not vectors:
            continue
        dim = len(vectors[0])
        rows = [(chunk_id, to_blob(vec)) for (chunk_id, _), vec in zip(batch, vectors, strict=True)]
        with engine.begin() as conn:
            embedded += upsert_embeddings(conn, model, rows, dim=dim)
        log.info("%s: %d/%d chunks embedded", model, embedded, len(todo))
    if not dim:
        dim = getattr(embedder, "dim", 0)
    return EmbedReport(model=model, embedded=embedded, skipped=skipped, dim=dim)


def check_vocab_age(engine: Engine, embedder: Embedder) -> bool:
    """Warn when a histogram embedder was trained on a different number of chunks. Returns True when stale."""
    art = getattr(embedder, "art", None)
    if art is None:
        return False
    with engine.connect() as conn:
        current = conn.execute(select(func.count()).select_from(chunks)).scalar_one()
    if current == art.chunk_count:
        return False
    log.warning(
        "Word2Vec was trained on %d chunks but the corpus now has %d; retrain with train_histogram.py",
        art.chunk_count, current,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="embedding model name, e.g. text-embedding-3-small or w2v-hist-k256-idf")
    parser.add_argument("--refresh", action="store_true", help="recompute vectors that already exist for this model")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = make_engine(settings.SYNC_DATABASE_URL)
    create_tables(engine)
    try:
        embedder = make_embedder(args.model, api_key=settings.OPENAI_API_KEY, models_dir=settings.MODELS_DIR)
    except (ValueError, FileNotFoundError) as exc:
        print(f"cannot build embedder {args.model!r}: {exc}", file=sys.stderr)
        return 1
    check_vocab_age(engine, embedder)
    report = embed_corpus(engine, embedder, refresh=args.refresh, batch_size=args.batch_size)
    print(f"model {report.model}: embedded {report.embedded}, skipped {report.skipped}, dim {report.dim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
