from datetime import date

import numpy as np
import pytest
from sqlalchemy import text

from corpus_db import (
    ChunkRecord,
    DocumentRecord,
    chunks_missing_embedding,
    create_tables,
    delete_embeddings,
    existing_sha,
    has_legacy_schema,
    load_snapshot,
    make_engine,
    models_present,
    prune_documents,
    replace_document,
    reset_corpus_tables,
    upsert_embeddings,
)
from embeddings import to_blob


@pytest.fixture
def engine():
    e = make_engine("sqlite://")
    create_tables(e)
    return e


def doc(sid="a", sha="1" * 64, fetched=date(2026, 10, 31)):
    return DocumentRecord(
        source_id=sid, title="T", tier="ois", url="u", file_path="data/ois/a.pdf", sha256=sha, fetched_on=fetched
    )


def test_replace_document_stores_chunks_without_vectors(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, 1, "one"), ChunkRecord(1, 2, "two")])
        assert [cid for cid, _ in chunks_missing_embedding(c, "m")] and existing_sha(c, "a") == "1" * 64
    assert load_snapshot(engine, "m").vectors.shape == (0, 0)


def test_upsert_embeddings_fills_one_model_and_snapshot_filters_by_model(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, None, "x"), ChunkRecord(1, None, "y")])
        ids = [cid for cid, _ in chunks_missing_embedding(c, "m1")]
        assert upsert_embeddings(c, "m1", [(ids[0], to_blob([1, 0]))], dim=2) == 1
        assert chunks_missing_embedding(c, "m1") == [(ids[1], "y")]
        upsert_embeddings(c, "m2", [(ids[0], to_blob([0, 0, 1])), (ids[1], to_blob([0, 1, 0]))], dim=3)
    s1, s2 = load_snapshot(engine, "m1"), load_snapshot(engine, "m2")
    assert s1.vectors.shape == (1, 2) and [m.content for m in s1.chunks] == ["x"] and s1.model == "m1"
    assert s1.vectors.dtype == np.float32
    assert s2.vectors.shape == (2, 3) and s2.document_count == 1
    assert models_present(engine) == {"m1": 1, "m2": 2}


def test_replace_document_drops_its_old_vectors(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, None, "x")])
        cid = chunks_missing_embedding(c, "m")[0][0]
        upsert_embeddings(c, "m", [(cid, to_blob([1, 0]))], dim=2)
        replace_document(c, doc(sha="2" * 64), [ChunkRecord(0, None, "new")])
    assert load_snapshot(engine, "m").vectors.shape == (0, 0) and models_present(engine) == {}


def test_upsert_replaces_an_existing_vector(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, None, "x")])
        cid = chunks_missing_embedding(c, "m")[0][0]
        upsert_embeddings(c, "m", [(cid, to_blob([1, 0]))], dim=2)
        upsert_embeddings(c, "m", [(cid, to_blob([0, 1]))], dim=2)
    snap = load_snapshot(engine, "m")
    assert snap.vectors.tolist() == [[0.0, 1.0]] and models_present(engine) == {"m": 1}


def test_delete_embeddings_and_prune(engine):
    with engine.begin() as c:
        replace_document(c, doc("a"), [ChunkRecord(0, None, "x")])
        replace_document(c, doc("b"), [ChunkRecord(0, None, "y")])
        rows = chunks_missing_embedding(c, "m")
        upsert_embeddings(c, "m", [(cid, to_blob([1, 0])) for cid, _ in rows], dim=2)
        upsert_embeddings(c, "n", [(rows[0][0], to_blob([1, 0]))], dim=2)
        assert delete_embeddings(c, "n") == 1
    # models_present opens its own connection: with SQLite's StaticPool that would
    # roll back an open transaction, so it is only called between transactions.
    assert models_present(engine) == {"m": 2}
    with engine.begin() as c:
        assert prune_documents(c, keep={"b"}) == ["a"]
    assert models_present(engine) == {"m": 1} and load_snapshot(engine, "m").document_count == 1


def test_snapshot_metadata_and_empty(engine):
    assert load_snapshot(engine, "m").chunks == []
    assert load_snapshot(engine, "m").vectors.shape == (0, 0)
    assert load_snapshot(engine, "m").corpus_as_of is None
    with engine.begin() as c:
        replace_document(c, doc("a", fetched=date(2026, 10, 1)), [ChunkRecord(0, None, "x")])
        replace_document(c, doc("b", fetched=date(2026, 10, 31)), [])
    snap = load_snapshot(engine, "m")
    assert snap.corpus_as_of == date(2026, 10, 31) and snap.document_count == 2 and snap.chunks == []


def test_legacy_schema_detection_and_reset():
    e = make_engine("sqlite://")
    with e.begin() as c:
        c.execute(text("CREATE TABLE documents (id INTEGER PRIMARY KEY, embedding_model TEXT)"))
        c.execute(text("CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, embedding BLOB)"))
    assert has_legacy_schema(e) is True
    reset_corpus_tables(e)
    assert has_legacy_schema(e) is False and models_present(e) == {}
    assert has_legacy_schema(make_engine("sqlite://")) is False
