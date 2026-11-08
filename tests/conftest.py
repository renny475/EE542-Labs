import httpx
import pytest_asyncio
from config import Settings
from app import create_app


@pytest_asyncio.fixture
async def client():
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-secret-not-for-production",
    )
    # Task 2 replaces llm=None with a FakeLLM.
    app = create_app(settings, llm=None)
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
