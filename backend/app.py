"""USC OIS Assistant backend.

Each user has a private thread with the assistant. A posted question is
saved, echoed to the asker's own WebSocket connections, and answered in a
background task that retrieves from the curated corpus (see ``rag.py``) and
calls the local ``llama-server``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, func, inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

import corpus_db
import rag
import settings
from embeddings import Embedder, is_histogram_model, make_embedder
from rag import LLMClient, Retriever

log = logging.getLogger("app")

BOT_NAME = "OIS Assistant"
HISTORY_MESSAGES = 4
HISTORY_CHARS = 500
WS_UNAUTHORIZED = 4401

# Force every connection to UTC so TIMESTAMP columns are read/written as UTC
# wall time. Timestamps are stored in UTC and converted only in the browser.
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"init_command": "SET time_zone = '+00:00'"},
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
passwords = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    sources: Mapped[Optional[list]] = mapped_column(JSON(none_as_null=True), nullable=True)
    embedding: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class NewMessage(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    embedding: str | None = Field(default=None, max_length=64)


async def get_session():
    async with SessionLocal() as session:
        yield session


def make_token(user: User) -> str:
    expiry = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user.id), "username": user.username, "exp": expiry},
        settings.JWT_SECRET,
        algorithm="HS256",
    )


def user_id_from_token(token: str | None) -> int | None:
    """Decode a login token; None when missing, malformed, forged or expired."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError, TypeError):
        return None


async def current_user(
    authorization: Optional[str] = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing login token")
    user_id = user_id_from_token(authorization.split(" ", 1)[1])
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired login token")
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


class ConnectionManager:
    """Open WebSockets grouped by user id, so a message reaches only its owner."""

    def __init__(self) -> None:
        self.connections: dict[int, set[WebSocket]] = {}
        self._owner: dict[WebSocket, int] = {}

    async def connect(self, websocket: WebSocket, user_id: int) -> None:
        await websocket.accept()
        self.connections.setdefault(user_id, set()).add(websocket)
        self._owner[websocket] = user_id

    def disconnect(self, websocket: WebSocket) -> None:
        user_id = self._owner.pop(websocket, None)
        if user_id is None:
            return
        sockets = self.connections.get(user_id)
        if sockets is not None:
            sockets.discard(websocket)
            if not sockets:
                del self.connections[user_id]

    async def send_to(self, user_id: int, data: dict) -> None:
        text = json.dumps(data)
        stale = []
        for connection in list(self.connections.get(user_id, ())):
            try:
                await connection.send_text(text)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


manager = ConnectionManager()


def message_json(message: Message, username: str) -> dict:
    created_at = message.created_at
    if created_at and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return {
        "id": message.id,
        "username": username,
        "content": message.content,
        "is_bot": message.is_bot,
        "sources": message.sources or [],
        "embedding": message.embedding,
        "created_at": created_at.isoformat() if created_at else None,
    }


async def save_message(
    content: str,
    user_id: Optional[int],
    is_bot: bool,
    username: str,
    sources: list | None = None,
    embedding: str | None = None,
) -> dict:
    async with SessionLocal() as session:
        message = Message(content=content, user_id=user_id, is_bot=is_bot, sources=sources, embedding=embedding)
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message_json(message, username)


async def recent_history(user_id: int, before_message_id: int) -> list[dict]:
    """The user's last few messages before the question, oldest first, as chat turns."""
    async with SessionLocal() as session:
        rows = (
            await session.scalars(
                select(Message)
                .where(Message.user_id == user_id, Message.id < before_message_id)
                .order_by(Message.id.desc())
                .limit(HISTORY_MESSAGES)
            )
        ).all()
    return [
        {"role": "assistant" if m.is_bot else "user", "content": m.content[:HISTORY_CHARS]}
        for m in reversed(rows)
    ]


async def respond(user_id: int, username: str, question: str, message_id: int, model: str) -> None:
    """Background task: answer one question with the chosen embedding and push it to its asker."""
    try:
        service: EmbeddingService = app.state.services[model]
        history = await recent_history(user_id, message_id)
        result = await rag.answer(
            question,
            history,
            retriever=service.retriever,
            embedder=service.embedder,
            chat=app.state.llm.chat,
            top_k=settings.RAG_TOP_K,
            min_score=service.threshold,
        )
        bot_message = await save_message(result.content, user_id, True, BOT_NAME, result.sources, embedding=model)
        await manager.send_to(user_id, {"type": "message", "message": bot_message})
    except Exception:
        log.exception("answering message %s for user %s (%s) failed", message_id, user_id, username)


MESSAGE_COLUMNS = {"sources": "JSON NULL", "embedding": "VARCHAR(64) NULL"}


def ensure_messages_columns(conn) -> list[str]:
    """Add columns that older databases lack (Part 1: sources; Part 4: embedding). Returns the names added.

    create_all never alters an existing table, so without this every POST
    /api/messages on an upgraded database fails with "Unknown column".
    """
    present = {c["name"] for c in inspect(conn).get_columns("messages")}
    added = []
    for name, ddl in MESSAGE_COLUMNS.items():
        if name in present:
            continue
        conn.execute(text(f"ALTER TABLE messages ADD COLUMN {name} {ddl}"))
        added.append(name)
    if added:
        log.warning("added missing messages column(s) %s to an older database", ", ".join(added))
    return added


@dataclass
class EmbeddingService:
    model: str
    label: str
    retriever: Retriever
    embedder: Embedder
    threshold: float


def embedding_label(model: str) -> str:
    if not is_histogram_model(model):
        return f"OpenAI {model}"
    from histogram_embedding import HistogramSpec

    spec = HistogramSpec.parse(model)
    parts = [f"Word histogram (k={spec.k})"]
    if spec.idf:
        parts.append("idf-weighted")
    if spec.soft:
        parts.append("soft assignment")
    return ", ".join(parts)


def build_services(engine, *, api_key: str, models_dir: Path, thresholds: dict[str, float]) -> dict[str, EmbeddingService]:
    """One retriever + embedder per model that has vectors and whose prerequisites exist."""
    services: dict[str, EmbeddingService] = {}
    for model in sorted(corpus_db.models_present(engine)):
        if not is_histogram_model(model) and not api_key:
            log.warning("skipping %s: OPENAI_API_KEY is not set", model)
            continue
        try:
            embedder = make_embedder(model, api_key=api_key, models_dir=models_dir)
        except (ValueError, FileNotFoundError) as exc:
            log.warning("skipping %s: %s", model, exc)
            continue
        retriever = Retriever(corpus_db.load_snapshot(engine, model), model)
        services[model] = EmbeddingService(
            model=model,
            label=embedding_label(model),
            retriever=retriever,
            embedder=embedder,
            threshold=settings.threshold_for(model, thresholds),
        )
    if not services:
        raise RuntimeError("no embedding model available; run embed_corpus.py --model ... first")
    return services


def default_model(services: dict[str, EmbeddingService]) -> str:
    if settings.EMBEDDING_MODEL in services:
        return settings.EMBEDDING_MODEL
    return sorted(m for m in services if is_histogram_model(m))[0]


def _load_services() -> tuple[dict[str, EmbeddingService], corpus_db.CorpusSnapshot]:
    sync_engine = corpus_db.make_engine(settings.SYNC_DATABASE_URL)
    try:
        services = build_services(
            sync_engine,
            api_key=settings.OPENAI_API_KEY,
            models_dir=settings.MODELS_DIR,
            thresholds=settings.load_thresholds(settings.MODELS_DIR),
        )
        snapshot = corpus_db.load_snapshot(sync_engine, default_model(services))
    finally:
        sync_engine.dispose()
    return services, snapshot


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(corpus_db.metadata.create_all)
        await conn.run_sync(ensure_messages_columns)

    # A RuntimeError here (no usable embedding) stops the server on purpose.
    services, snapshot = await asyncio.to_thread(_load_services)
    app.state.services = services
    app.state.default_model = default_model(services)
    app.state.corpus_as_of = snapshot.corpus_as_of
    app.state.llm = LLMClient(settings.LLM_API_BASE, settings.LLM_MODEL, settings.LLM_API_KEY)
    for service in services.values():
        log.info("embedding %s: %d chunks, threshold %.2f", service.model, service.retriever.size, service.threshold)
    log.info("corpus: %d documents, as of %s; default embedding %s",
             snapshot.document_count, snapshot.corpus_as_of, app.state.default_model)
    yield
    await engine.dispose()


app = FastAPI(title="USC OIS Assistant", lifespan=lifespan)

FRONTEND_DIR = settings.PROJECT_DIR / "frontend"
app.mount("/.well-known", StaticFiles(directory=FRONTEND_DIR / ".well-known"), name="well-known")


@app.get("/")
async def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
async def styles():
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def javascript():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/api/meta")
async def meta():
    services: dict[str, EmbeddingService] = app.state.services
    default = app.state.default_model
    corpus_as_of = app.state.corpus_as_of
    return {
        "document_count": services[default].retriever.document_count,
        "chunk_count": services[default].retriever.size,
        "corpus_as_of": corpus_as_of.isoformat() if corpus_as_of else None,
        "llm_model": settings.LLM_MODEL,
        "default_embedding": default,
        "embeddings": [
            {"model": s.model, "label": s.label, "chunk_count": s.retriever.size, "threshold": s.threshold}
            for s in services.values()
        ],
    }


@app.post("/api/signup")
async def signup(data: Credentials, session: AsyncSession = Depends(get_session)):
    username = data.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be blank")
    existing = await session.scalar(select(User).where(User.username == username))
    if existing:
        raise HTTPException(status_code=409, detail="Username is already taken")
    user = User(username=username, password_hash=passwords.hash(data.password))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return {"token": make_token(user), "username": user.username}


@app.post("/api/login")
async def login(data: Credentials, session: AsyncSession = Depends(get_session)):
    user = await session.scalar(select(User).where(User.username == data.username.strip()))
    if not user or not passwords.verify(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return {"token": make_token(user), "username": user.username}


@app.get("/api/messages")
async def get_messages(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    rows = (
        await session.scalars(
            select(Message).where(Message.user_id == user.id).order_by(Message.id.desc()).limit(100)
        )
    ).all()
    return {"messages": [message_json(m, BOT_NAME if m.is_bot else user.username) for m in reversed(rows)]}


@app.post("/api/messages")
async def post_message(data: NewMessage, background_tasks: BackgroundTasks, user: User = Depends(current_user)):
    content = data.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be blank")
    model = data.embedding or app.state.default_model
    if model not in app.state.services:
        available = ", ".join(app.state.services)
        raise HTTPException(status_code=400, detail=f"unknown embedding {model!r}; available: {available}")
    saved = await save_message(content, user.id, False, user.username, embedding=model)
    await manager.send_to(user.id, {"type": "message", "message": saved})
    background_tasks.add_task(respond, user.id, user.username, content, saved["id"], model)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    user_id = user_id_from_token(websocket.query_params.get("token"))
    if user_id is not None:
        async with SessionLocal() as session:
            if await session.get(User, user_id) is None:
                user_id = None
    if user_id is None:
        await websocket.accept()
        await websocket.close(code=WS_UNAUTHORIZED)
        return

    await manager.connect(websocket, user_id)
    try:
        while True:
            # The frontend only receives on this socket; it posts questions over HTTP.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
