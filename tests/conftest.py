import httpx
import pytest
import pytest_asyncio
from config import Settings
from app import create_app
from fake_llm import FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest_asyncio.fixture
async def client(fake_llm):
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-secret-not-for-production",
    )
    app = create_app(settings, llm=fake_llm)
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
