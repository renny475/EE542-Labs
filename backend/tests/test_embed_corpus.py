import pytest

from corpus_db import load_snapshot, models_present
from embed_corpus import embed_corpus
from embeddings import FakeEmbedder, is_histogram_model, make_embedder
from ingest import ingest
from tests.test_ingest import setup


def corpus(tmp_path):
    m, d, e = setup(tmp_path)
    ingest(m, d, e)
    return e


def test_embed_corpus_fills_missing_then_skips(tmp_path):
    e = corpus(tmp_path)
    r1 = embed_corpus(e, FakeEmbedder(dim=4, model="fake"))
    assert r1.embedded >= 3 and r1.skipped == 0 and r1.dim == 4 and r1.model == "fake"
    r2 = embed_corpus(e, FakeEmbedder(dim=4, model="fake"))
    assert r2.embedded == 0 and r2.skipped == r1.embedded
    assert load_snapshot(e, "fake").vectors.shape == (r1.embedded, 4)


def test_embed_corpus_refresh_recomputes(tmp_path):
    e = corpus(tmp_path)
    n = embed_corpus(e, FakeEmbedder(dim=4)).embedded
    r = embed_corpus(e, FakeEmbedder(dim=4), refresh=True)
    assert r.embedded == n and r.skipped == 0 and models_present(e) == {"fake": n}


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4, model="rec")
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)


def test_embed_corpus_batches(tmp_path):
    e = corpus(tmp_path)
    emb = RecordingEmbedder()
    embed_corpus(e, emb, batch_size=2)
    assert emb.calls and all(len(call) <= 2 for call in emb.calls)
    sent = [t for call in emb.calls for t in call]
    assert sent == [m.content for m in load_snapshot(e, "rec").chunks] or sorted(sent) == sorted(
        m.content for m in load_snapshot(e, "rec").chunks
    )


def test_make_embedder_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_embedder("bert-base")


def test_is_histogram_model():
    assert is_histogram_model("w2v-hist-k256-idf-soft") and not is_histogram_model("text-embedding-3-small")


class FingerprintedEmbedder(FakeEmbedder):
    def __init__(self, fingerprint, dim=4, model="hist"):
        super().__init__(dim=dim, model=model)
        self.fingerprint = fingerprint


def test_embed_corpus_re_embeds_everything_when_the_artefact_fingerprint_changed(tmp_path, caplog):
    import logging

    from corpus_db import embedding_fingerprints

    e = corpus(tmp_path)
    n = embed_corpus(e, FingerprintedEmbedder("aaaa")).embedded
    assert embed_corpus(e, FingerprintedEmbedder("aaaa")).embedded == 0
    with caplog.at_level(logging.WARNING):
        r = embed_corpus(e, FingerprintedEmbedder("bbbb"))
    assert r.embedded == n and "retrained" in caplog.text.lower()
    assert embedding_fingerprints(e, "hist") == {"bbbb"}


class FailsOnSecondBatch(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4, model="fake")
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("network")
        return super().embed(texts)


def test_refresh_never_leaves_fewer_vectors_than_before(tmp_path):
    e = corpus(tmp_path)
    n = embed_corpus(e, FakeEmbedder(dim=4)).embedded
    assert n >= 3
    with pytest.raises(RuntimeError, match="network"):
        embed_corpus(e, FailsOnSecondBatch(), refresh=True, batch_size=1)
    assert models_present(e) == {"fake": n}


def test_require_current_schema_rejects_part4_tables():
    from sqlalchemy import text

    from corpus_db import make_engine
    from embed_corpus import require_current_schema

    e = make_engine("sqlite://")
    with e.begin() as c:
        c.execute(text("CREATE TABLE documents (id INTEGER PRIMARY KEY, embedding_model TEXT)"))
        c.execute(text("CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, embedding BLOB)"))
    with pytest.raises(RuntimeError, match="ingest.py"):
        require_current_schema(e)
