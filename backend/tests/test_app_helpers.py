import asyncio
import os

os.environ.setdefault("JWT_SECRET", "test-secret-that-is-long-enough-0123456789")
from datetime import datetime, timedelta, timezone  # noqa: E402

from jose import jwt  # noqa: E402

import settings  # noqa: E402
from app import ConnectionManager, Message, message_json, user_id_from_token  # noqa: E402


def tok(sub, minutes=5):
    return jwt.encode(
        {"sub": str(sub), "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)},
        settings.JWT_SECRET,
        algorithm="HS256",
    )


def test_user_id_from_token_rejects_bad_tokens():
    assert user_id_from_token(tok(7)) == 7
    assert user_id_from_token(tok(7, minutes=-1)) is None
    assert user_id_from_token("garbage") is None
    assert user_id_from_token(jwt.encode({"sub": "7"}, "other-secret", algorithm="HS256")) is None


class FakeWS:
    def __init__(self):
        self.sent = []

    async def accept(self):
        pass

    async def send_text(self, s):
        self.sent.append(s)


def test_send_to_routes_only_to_that_user():
    m = ConnectionManager()
    a1, a2, b = FakeWS(), FakeWS(), FakeWS()

    async def go():
        await m.connect(a1, 1)
        await m.connect(a2, 1)
        await m.connect(b, 2)
        await m.send_to(1, {"type": "message", "message": {"id": 1}})
        m.disconnect(a1)
        await m.send_to(1, {"type": "message", "message": {"id": 2}})

    asyncio.run(go())
    assert len(a1.sent) == 1 and len(a2.sent) == 2 and b.sent == []


def test_message_json_includes_sources_and_utc():
    m = Message(id=1, user_id=3, content="hi", is_bot=True, sources=None, created_at=datetime(2026, 10, 31, 12, 0, 0))
    out = message_json(m, "OIS Assistant")
    assert out["sources"] == [] and out["created_at"].endswith("+00:00") and out["username"] == "OIS Assistant"


# --- Part 5: several embeddings ---------------------------------------------

import logging  # noqa: E402
import pathlib  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, inspect, text  # noqa: E402

from app import (  # noqa: E402
    EmbeddingService,
    User,
    app,
    build_services,
    current_user,
    default_model,
    embedding_label,
    ensure_messages_columns,
)
from corpus_db import ChunkRecord, DocumentRecord, chunks_missing_embedding, create_tables, make_engine, replace_document, upsert_embeddings  # noqa: E402
from embeddings import FakeEmbedder, to_blob  # noqa: E402


def test_embedding_label():
    assert embedding_label("w2v-hist-k256-idf-soft") == "Word histogram (k=256), idf-weighted, soft assignment"
    assert embedding_label("w2v-hist-k256") == "Word histogram (k=256)"
    assert embedding_label("text-embedding-3-small") == "OpenAI text-embedding-3-small"


def svc(model):
    return EmbeddingService(model=model, label=model, retriever=None, embedder=None, threshold=0.3)


def test_default_model_prefers_openai_then_first_histogram():
    assert default_model({"w2v-hist-k256-idf": svc("x"), "text-embedding-3-small": svc("y")}) == "text-embedding-3-small"
    assert default_model({"w2v-hist-k256-idf": svc("x"), "w2v-hist-k256": svc("y")}) == "w2v-hist-k256"


def two_model_engine():
    e = make_engine("sqlite://")
    create_tables(e)
    with e.begin() as c:
        from datetime import date

        replace_document(
            c,
            DocumentRecord("a", "T", "ois", "u", "data/ois/a.pdf", "1" * 64, date(2026, 10, 31)),
            [ChunkRecord(0, None, "opt"), ChunkRecord(1, None, "cpt")],
        )
        ids = [cid for cid, _ in chunks_missing_embedding(c, "x")]
        fake = FakeEmbedder(dim=4)
        for model in ("text-embedding-3-small", "w2v-hist-k256"):
            upsert_embeddings(c, model, [(cid, to_blob(fake.embed([str(cid)])[0])) for cid in ids], dim=4)
    return e


def test_build_services_skips_unavailable_models(tmp_path, caplog):
    e = two_model_engine()
    with pytest.raises(RuntimeError, match="no embedding model available"):
        build_services(e, api_key="", models_dir=tmp_path, thresholds={})
    with caplog.at_level(logging.WARNING):
        s = build_services(e, api_key="sk-x", models_dir=tmp_path, thresholds={})
    assert list(s) == ["text-embedding-3-small"] and s["text-embedding-3-small"].threshold == settings.RAG_MIN_SCORE
    assert s["text-embedding-3-small"].retriever.size == 2
    assert "word2vec.kv" in caplog.text


def test_message_json_defaults_embedding_to_none():
    m = Message(id=1, user_id=3, content="hi", is_bot=True, sources=None, embedding=None, created_at=datetime(2026, 10, 31, 12, 0, 0))
    out = message_json(m, "OIS Assistant")
    assert out["embedding"] is None and out["sources"] == []


def test_post_message_rejects_unknown_embedding():
    app.dependency_overrides[current_user] = lambda: User(id=1, username="u")
    app.state.services = {"text-embedding-3-small": svc("text-embedding-3-small")}
    app.state.default_model = "text-embedding-3-small"
    try:
        client = TestClient(app)  # no context manager: lifespan (MySQL, corpus) is not run
        r = client.post("/api/messages", json={"content": "hi", "embedding": "nope"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 400 and "text-embedding-3-small" in r.json()["detail"]


def test_ensure_messages_columns_adds_sources_and_embedding():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE messages (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, is_bot BOOLEAN)"))
        assert ensure_messages_columns(conn) == ["sources", "embedding"]
        assert ensure_messages_columns(conn) == []
    assert {"sources", "embedding"} <= {c["name"] for c in inspect(engine).get_columns("messages")}


def test_default_model_falls_back_to_any_available_model_when_env_names_an_absent_one(monkeypatch, caplog):
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "text-embedding-3-large")
    with caplog.at_level(logging.WARNING):
        assert default_model({"text-embedding-3-small": svc("x")}) == "text-embedding-3-small"
    assert "text-embedding-3-large" in caplog.text


def test_build_services_rejects_a_part4_database():
    from sqlalchemy import text as sql_text

    e = make_engine("sqlite://")
    with e.begin() as c:
        c.execute(sql_text("CREATE TABLE documents (id INTEGER PRIMARY KEY, embedding_model TEXT)"))
        c.execute(sql_text("CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, embedding BLOB)"))
    with pytest.raises(RuntimeError, match="ingest.py"):
        build_services(e, api_key="sk-x", models_dir=pathlib.Path("/nonexistent"), thresholds={})


def test_build_services_skips_a_model_whose_vectors_predate_the_artefacts(tmp_path, caplog, monkeypatch):
    import embeddings as embeddings_module

    e = two_model_engine()  # vectors for w2v-hist-k256 carry no fingerprint

    class Stale(FakeEmbedder):
        fingerprint = "ffff"

    monkeypatch.setattr(
        "app.make_embedder",
        lambda model, **kw: Stale(dim=4, model=model) if embeddings_module.is_histogram_model(model) else FakeEmbedder(dim=4, model=model),
    )
    with caplog.at_level(logging.WARNING):
        s = build_services(e, api_key="sk-x", models_dir=tmp_path, thresholds={})
    assert "w2v-hist-k256" not in s and "--refresh" in caplog.text
