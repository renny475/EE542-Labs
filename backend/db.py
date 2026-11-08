from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool
from fastapi import Request


class Base(DeclarativeBase):
    pass


def make_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        kwargs = {}
        if ":memory:" in url or url.endswith("///"):
            # One shared connection so every session sees the same in-memory DB.
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        return create_async_engine(url, **kwargs)
    # Force every connection to UTC so DATETIME columns are read/written as UTC.
    return create_async_engine(
        url,
        pool_pre_ping=True,
        connect_args={"init_command": "SET time_zone = '+00:00'"},
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request):
    async with request.app.state.session_factory() as session:
        yield session


async def init_db(engine: AsyncEngine) -> None:
    import models  # noqa: F401  (register tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
