"""The corpus tables and the per-model in-memory snapshot.

``documents`` and ``chunks`` hold the text (written by ``ingest.py``);
``chunk_embeddings`` holds one vector per (chunk, model) so several
embeddings can coexist (written by ``embed_corpus.py``). MySQL is the
source of truth; the app rebuilds a FAISS index per model from
:func:`load_snapshot` at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.pool import StaticPool

from embeddings import from_blob

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("source_id", String(64), nullable=False, unique=True),
    Column("title", String(255), nullable=False),
    Column("tier", String(8), nullable=False),
    Column("url", String(512), nullable=False),
    Column("file_path", String(255), nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("fetched_on", Date, nullable=False),
    Column("published", Date, nullable=True),
    Column("chunk_count", Integer, nullable=False),
    Column("ingested_at", DateTime, server_default=func.now()),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("page", Integer, nullable=True),
    Column("content", Text, nullable=False),
    UniqueConstraint("document_id", "chunk_index"),
)

chunk_embeddings = Table(
    "chunk_embeddings",
    metadata,
    Column("chunk_id", ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
    Column("model", String(64), primary_key=True),
    Column("dim", Integer, nullable=False),
    Column("embedding", LargeBinary, nullable=False),
)


@dataclass
class DocumentRecord:
    source_id: str
    title: str
    tier: str
    url: str
    file_path: str
    sha256: str
    fetched_on: date
    published: date | None = None


@dataclass
class ChunkRecord:
    chunk_index: int
    page: int | None
    content: str


@dataclass
class ChunkMeta:
    chunk_id: int
    document_id: int
    title: str
    tier: str
    url: str
    page: int | None
    published: date | None
    content: str


@dataclass
class CorpusSnapshot:
    chunks: list[ChunkMeta]
    vectors: np.ndarray
    model: str
    document_count: int = 0
    corpus_as_of: date | None = None


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        # One shared in-memory database across connections, usable from any thread.
        return create_engine(url, poolclass=StaticPool, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    metadata.create_all(engine)


def has_legacy_schema(engine: Engine) -> bool:
    """True when the database still has the Part 4 shape (vector stored on the chunk)."""
    insp = inspect(engine)
    if "chunks" not in insp.get_table_names():
        return False
    return "embedding" in {c["name"] for c in insp.get_columns("chunks")}


def reset_corpus_tables(engine: Engine) -> None:
    """Drop the three corpus tables (FK-safe order) and recreate them empty."""
    with engine.begin() as conn:
        for name in ("chunk_embeddings", "chunks", "documents"):
            conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
    create_tables(engine)


def existing_sha(conn: Connection, source_id: str) -> str | None:
    return conn.execute(select(documents.c.sha256).where(documents.c.source_id == source_id)).scalar_one_or_none()


def _delete_documents(conn: Connection, document_ids: list[int]) -> None:
    if not document_ids:
        return
    # Explicit so SQLite without foreign-key enforcement behaves like MySQL's CASCADE.
    chunk_ids = select(chunks.c.id).where(chunks.c.document_id.in_(document_ids))
    conn.execute(delete(chunk_embeddings).where(chunk_embeddings.c.chunk_id.in_(chunk_ids)))
    conn.execute(delete(chunks).where(chunks.c.document_id.in_(document_ids)))
    conn.execute(delete(documents).where(documents.c.id.in_(document_ids)))


def replace_document(conn: Connection, doc: DocumentRecord, chunk_records: list[ChunkRecord]) -> int:
    old_ids = conn.execute(select(documents.c.id).where(documents.c.source_id == doc.source_id)).scalars().all()
    _delete_documents(conn, list(old_ids))
    document_id = conn.execute(
        insert(documents).values(
            source_id=doc.source_id,
            title=doc.title,
            tier=doc.tier,
            url=doc.url,
            file_path=doc.file_path,
            sha256=doc.sha256,
            fetched_on=doc.fetched_on,
            published=doc.published,
            chunk_count=len(chunk_records),
        )
    ).inserted_primary_key[0]
    if chunk_records:
        conn.execute(
            insert(chunks),
            [
                {"document_id": document_id, "chunk_index": c.chunk_index, "page": c.page, "content": c.content}
                for c in chunk_records
            ],
        )
    return document_id


def prune_documents(conn: Connection, keep: set[str]) -> list[str]:
    rows = conn.execute(select(documents.c.id, documents.c.source_id)).all()
    doomed = [(i, sid) for i, sid in rows if sid not in keep]
    _delete_documents(conn, [i for i, _ in doomed])
    return sorted(sid for _, sid in doomed)


def chunks_missing_embedding(conn: Connection, model: str) -> list[tuple[int, str]]:
    """(chunk_id, content) of every chunk that has no vector for ``model``, by chunk id."""
    have = select(chunk_embeddings.c.chunk_id).where(chunk_embeddings.c.model == model)
    rows = conn.execute(
        select(chunks.c.id, chunks.c.content).where(chunks.c.id.not_in(have)).order_by(chunks.c.id)
    ).all()
    return [(int(cid), content) for cid, content in rows]


def upsert_embeddings(conn: Connection, model: str, rows: list[tuple[int, bytes]], dim: int) -> int:
    if not rows:
        return 0
    ids = [cid for cid, _ in rows]
    conn.execute(
        delete(chunk_embeddings).where(chunk_embeddings.c.model == model, chunk_embeddings.c.chunk_id.in_(ids))
    )
    conn.execute(
        insert(chunk_embeddings),
        [{"chunk_id": cid, "model": model, "dim": dim, "embedding": blob} for cid, blob in rows],
    )
    return len(rows)


def delete_embeddings(conn: Connection, model: str) -> int:
    return conn.execute(delete(chunk_embeddings).where(chunk_embeddings.c.model == model)).rowcount


def models_present(engine: Engine) -> dict[str, int]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(chunk_embeddings.c.model, func.count()).group_by(chunk_embeddings.c.model)
        ).all()
    return {model: int(n) for model, n in rows}


def load_snapshot(engine: Engine, model: str) -> CorpusSnapshot:
    with engine.connect() as conn:
        doc_rows = conn.execute(select(documents.c.fetched_on)).all()
        rows = conn.execute(
            select(
                chunks.c.id,
                chunks.c.document_id,
                documents.c.title,
                documents.c.tier,
                documents.c.url,
                chunks.c.page,
                documents.c.published,
                chunks.c.content,
                chunk_embeddings.c.embedding,
            )
            .select_from(chunk_embeddings)
            .join(chunks, chunks.c.id == chunk_embeddings.c.chunk_id)
            .join(documents, documents.c.id == chunks.c.document_id)
            .where(chunk_embeddings.c.model == model)
            .order_by(chunks.c.document_id, chunks.c.chunk_index)
        ).all()

    metas = [ChunkMeta(*row[:8]) for row in rows]
    if rows:
        vectors = np.vstack([from_blob(row[8]) for row in rows]).astype(np.float32)
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)
    fetched = [d.fetched_on for d in doc_rows if d.fetched_on is not None]
    return CorpusSnapshot(
        chunks=metas,
        vectors=vectors,
        model=model,
        document_count=len(doc_rows),
        corpus_as_of=max(fetched) if fetched else None,
    )
