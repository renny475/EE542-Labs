import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models import User

router = APIRouter()


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


def _prehash(password: str) -> bytes:
    # bcrypt only reads 72 bytes; hash first so long passwords are fully used.
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("ascii"))
    except ValueError:
        return False


def make_token(request: Request, user: User) -> str:
    settings = request.app.state.settings
    expiry = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode(
        {"sub": str(user.id), "username": user.username, "exp": expiry},
        settings.jwt_secret,
        algorithm="HS256",
    )


async def current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing login token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, request.app.state.settings.jwt_secret, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired login token")
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


@router.post("/api/signup")
async def signup(
    data: Credentials, request: Request, session: AsyncSession = Depends(get_session)
):
    username = data.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be blank")
    if await session.scalar(select(User).where(User.username == username)):
        raise HTTPException(status_code=409, detail="Username is already taken")
    user = User(username=username, password_hash=hash_password(data.password))
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return {"token": make_token(request, user), "username": user.username}


@router.post("/api/login")
async def login(
    data: Credentials, request: Request, session: AsyncSession = Depends(get_session)
):
    user = await session.scalar(select(User).where(User.username == data.username.strip()))
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return {"token": make_token(request, user), "username": user.username}


@router.get("/api/me")
async def me(user: User = Depends(current_user)):
    return {"username": user.username}
