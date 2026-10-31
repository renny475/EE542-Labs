"""The corpus tables (``documents``, ``chunks``) and the in-memory snapshot.

MySQL is the source of truth; the FAISS index the app searches is rebuilt
from :func:`load_snapshot` at every startup. The ingester talks to the same
tables through a synchronous engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    select,
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
    Column("embedding_model", String(64), nullable=False),
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
    Column("embedding", LargeBinary, nullable=False),
    UniqueConstraint("document_id", "chunk_index"),
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
    embedding_model: str
    published: date | None = None


@dataclass
class ChunkRecord:
    chunk_index: int
    page: int | None
    content: str
    embedding: bytes


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
    embedding_models: set[str] = field(default_factory=set)
    document_count: int = 0
    corpus_as_of: date | None = None


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        # One shared in-memory database across connections, usable from any thread.
        return create_engine(url, poolclass=StaticPool, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    metadata.create_all(engine)


def existing_sha(conn: Connection, source_id: str) -> str | None:
    return conn.execute(select(documents.c.sha256).where(documents.c.source_id == source_id)).scalar_one_or_none()


def _delete_documents(conn: Connection, document_ids: list[int]) -> None:
    if not document_ids:
        return
    # Explicit so SQLite without foreign-key enforcement behaves like MySQL's CASCADE.
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
            embedding_model=doc.embedding_model,
            chunk_count=len(chunk_records),
        )
    ).inserted_primary_key[0]
    if chunk_records:
        conn.execute(
            insert(chunks),
            [
                {
                    "document_id": document_id,
                    "chunk_index": c.chunk_index,
                    "page": c.page,
                    "content": c.content,
                    "embedding": c.embedding,
                }
                for c in chunk_records
            ],
        )
    return document_id


def prune_documents(conn: Connection, keep: set[str]) -> list[str]:
    rows = conn.execute(select(documents.c.id, documents.c.source_id)).all()
    doomed = [(i, sid) for i, sid in rows if sid not in keep]
    _delete_documents(conn, [i for i, _ in doomed])
    return sorted(sid for _, sid in doomed)


def load_snapshot(engine: Engine) -> CorpusSnapshot:
    with engine.connect() as conn:
        doc_rows = conn.execute(
            select(documents.c.embedding_model, documents.c.fetched_on)
        ).all()
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
                chunks.c.embedding,
            )
            .join(documents, documents.c.id == chunks.c.document_id)
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
        embedding_models={d.embedding_model for d in doc_rows},
        document_count=len(doc_rows),
        corpus_as_of=max(fetched) if fetched else None,
    )
