import httpx
import pytest
import pytest_asyncio
from config import Settings
from app import create_app
from fake_llm import FakeLLM
from knowledge import KnowledgeStore


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest_asyncio.fixture
async def client(fake_llm, tmp_path):
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-secret-not-for-production",
    )
    # An empty store under tmp_path, so tests never create the repo's index file.
    app = create_app(settings, llm=fake_llm, store=KnowledgeStore(tmp_path / "index.sqlite3", fake_llm))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def auth_headers(client, username="alice"):
    resp = await client.post(
        "/api/signup", json={"username": username, "password": "password123"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


@pytest_asyncio.fixture
async def file_client(fake_llm, tmp_path):
    """Like `client`, but on a file-backed SQLite DB so sessions get separate
    connections (needed for real concurrency tests)."""
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        jwt_secret="test-secret-not-for-production",
    )
    # An empty store under tmp_path, so tests never create the repo's index file.
    app = create_app(settings, llm=fake_llm, store=KnowledgeStore(tmp_path / "index.sqlite3", fake_llm))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
