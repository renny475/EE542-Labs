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
