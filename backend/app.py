import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Load groupchat_app_src/.env, one directory above backend/
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+asyncmy://chatuser:chatpass@localhost:3306/groupchat",
)
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "43200"))
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8001/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3-8b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

if not JWT_SECRET or JWT_SECRET == "please_change_me_to_a_long_random_value":
    raise RuntimeError("Set a unique JWT_SECRET in the project .env file.")

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
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
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text)
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class NewMessage(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


async def get_session():
    async with SessionLocal() as session:
        yield session


def make_token(user: User) -> str:
    expiry = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user.id), "username": user.username, "exp": expiry},
        JWT_SECRET,
        algorithm="HS256",
    )


async def authenticated_user(
    session: AsyncSession = Depends(get_session),
    authorization: Optional[str] = None,
) -> User:
    # FastAPI injects the Authorization header below via Header.
    raise NotImplementedError


from fastapi import Header


async def current_user(
    authorization: Optional[str] = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing login token")

    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired login token")

    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


class ConnectionManager:
    def __init__(self):
        self.connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)

    async def broadcast(self, data: dict):
        stale = []
        for connection in self.connections:
            try:
                await connection.send_text(json.dumps(data))
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
        "created_at": created_at.isoformat() if created_at else None,
    }


async def save_message(
    content: str, user_id: Optional[int], is_bot: bool, username: str
) -> dict:
    async with SessionLocal() as session:
        message = Message(content=content, user_id=user_id, is_bot=is_bot)
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message_json(message, username)


async def ask_llm(question: str) -> str:
    headers = {}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    body = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant in a group chat. Give concise, useful answers.",
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        return f"Sorry, I could not reach the LLM server. Check its address and status. ({exc})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables automatically. You can also run sql/schema.sql first.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Group Chat + LLM Bot", lifespan=lifespan)

FRONTEND_DIR = PROJECT_DIR / "frontend"
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
async def get_messages(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    rows = (
        await session.execute(
            select(Message, User.username)
            .outerjoin(User, Message.user_id == User.id)
            .order_by(Message.id.desc())
            .limit(100)
        )
    ).all()

    messages = [
        message_json(message, "Bot" if message.is_bot else (username or "unknown"))
        for message, username in reversed(rows)
    ]
    return {"messages": messages}


@app.post("/api/messages")
async def post_message(
    data: NewMessage,
    user: User = Depends(current_user),
):
    content = data.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be blank")

    saved = await save_message(content, user.id, False, user.username)
    await manager.broadcast({"type": "message", "message": saved})

    if "?" in content:
        answer = await ask_llm(content)
        bot_message = await save_message(answer, None, True, "Bot")
        await manager.broadcast({"type": "message", "message": bot_message})

    return {"ok": True}


@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # The frontend receives broadcasts; it does not send chat through this socket.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)