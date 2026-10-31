from datetime import date

import numpy as np
import pytest

from corpus_db import (
    ChunkRecord,
    DocumentRecord,
    create_tables,
    existing_sha,
    load_snapshot,
    make_engine,
    prune_documents,
    replace_document,
)
from embeddings import to_blob


@pytest.fixture
def engine():
    e = make_engine("sqlite://")
    create_tables(e)
    return e


def doc(sid="a", sha="1" * 64, fetched=date(2026, 10, 31)):
    return DocumentRecord(
        source_id=sid, title="T", tier="ois", url="u", file_path="data/ois/a.pdf",
        sha256=sha, fetched_on=fetched, embedding_model="fake",
    )


def test_replace_document_inserts_and_replaces(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, 1, "one", to_blob([1, 0])), ChunkRecord(1, 2, "two", to_blob([0, 1]))])
        assert existing_sha(c, "a") == "1" * 64
        replace_document(c, doc(sha="2" * 64), [ChunkRecord(0, None, "new", to_blob([1, 1]))])
    snap = load_snapshot(engine)
    assert [c.content for c in snap.chunks] == ["new"] and snap.document_count == 1
    assert snap.vectors.shape == (1, 2) and snap.vectors.dtype == np.float32


def test_prune_removes_documents_not_kept(engine):
    with engine.begin() as c:
        replace_document(c, doc("a"), [ChunkRecord(0, None, "x", to_blob([1, 0]))])
        replace_document(c, doc("b"), [ChunkRecord(0, None, "y", to_blob([0, 1]))])
        assert prune_documents(c, keep={"a"}) == ["b"]
    assert [m.title for m in load_snapshot(engine).chunks] == ["T"] and load_snapshot(engine).document_count == 1


def test_snapshot_metadata(engine):
    with engine.begin() as c:
        replace_document(c, doc("a", fetched=date(2026, 10, 1)), [ChunkRecord(0, None, "x", to_blob([1, 0]))])
        replace_document(c, doc("b", fetched=date(2026, 10, 31)), [])
    snap = load_snapshot(engine)
    assert snap.corpus_as_of == date(2026, 10, 31) and snap.embedding_models == {"fake"}


def test_empty_snapshot(engine):
    snap = load_snapshot(engine)
    assert snap.chunks == [] and snap.vectors.shape == (0, 0) and snap.corpus_as_of is None
