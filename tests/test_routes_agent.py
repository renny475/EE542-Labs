import json

import httpx
import pytest_asyncio

from agent import FALLBACK
from app import create_app
from config import Settings
from conftest import auth_headers
from knowledge import KnowledgeStore
from llm import LLMClient, LLMError
from test_routes_documents import DOC, upload_text

# DOC paragraphs: [1] discharge date, [2] amoxicillin 500 mg, [3] fever above 101 F.
ITEMS = [
    {
        "category": "medication",
        "text": "Take amoxicillin 500 mg three times daily for 7 days.",
        "source_quote": "Take amoxicillin 500 mg by mouth three times daily for 7 days.",
        "details": {"name": "Amoxicillin"},
    },
]


def answer(text, citations, kind="answer", care_team_question=None):
    return {"content": json.dumps({
        "kind": kind, "answer": text, "citations": citations,
        "care_team_question": care_team_question,
    })}


async def confirmed(client, fake_llm, headers=None, text=DOC):
    h = headers or await auth_headers(client)
    resp = await upload_text(client, h, fake_llm, text=text, items=ITEMS)
    assert resp.status_code == 200, resp.text
    resp = await client.post("/api/documents/active/confirm", headers=h)
    assert resp.status_code == 200, resp.text
    return h


async def post_ask(client, h, question):
    return await client.post("/api/ask", json={"question": question}, headers=h)


async def care_team(client, h):
    return (await client.get("/api/care-team", headers=h)).json()["questions"]


async def test_ask_returns_cited_answer_and_saves_conversation(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Take amoxicillin 500 mg three times daily [2].", [2]))
    resp = await post_ask(client, h, "  How often do I take amoxicillin?  ")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "kind": "answer",
        "answer": "Take amoxicillin 500 mg three times daily [2].",
        "sources": [{
            "number": 2, "kind": "document", "title": "Your discharge document", "url": None,
            "content": "Take amoxicillin 500 mg by mouth three times daily for 7 days.",
        }],
        "flagged": False,
        "emergency": {"terms": [], "banner": False},
    }
    assert fake_llm.calls[-1][0][-1]["content"].endswith("How often do I take amoxicillin?")

    messages = (await client.get("/api/conversation", headers=h)).json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "How often do I take amoxicillin?"),
        ("assistant", body["answer"]),
    ]
    assert messages[0]["payload"] is None
    assert messages[1]["payload"] == body
    assert all(m["created_at"] for m in messages)
    assert await care_team(client, h) == []


async def test_routes_ask_sets_banner_even_if_model_ignores(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Take amoxicillin 500 mg three times daily [2].", [2]))
    resp = await post_ask(client, h, "I have chest pain, should I still take amoxicillin?")
    body = resp.json()
    assert body["kind"] == "answer"
    assert body["emergency"] == {"terms": ["chest pain"], "banner": True}
    assert "A possible emergency was detected in the patient's message: chest pain" in (
        fake_llm.calls[-1][0][-1]["content"]
    )


async def test_ask_without_confirmed_doc_409(client, fake_llm):
    h = await auth_headers(client)
    assert (await post_ask(client, h, "Hello?")).status_code == 409
    assert (await upload_text(client, h, fake_llm, items=ITEMS)).status_code == 200
    resp = await post_ask(client, h, "Hello?")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Confirm your discharge document first."


async def test_conversation_scoped_to_active_document(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Take amoxicillin 500 mg three times daily [2].", [2]))
    assert (await post_ask(client, h, "How often?")).status_code == 200
    fake_llm.script(answer("I could not find this.", [], kind="not_covered"))
    assert (await post_ask(client, h, "Can I swim?")).status_code == 200
    assert len((await client.get("/api/conversation", headers=h)).json()["messages"]) == 4

    await confirmed(client, fake_llm, headers=h, text=DOC + "\n\nRest at home.")
    assert (await client.get("/api/conversation", headers=h)).json() == {"messages": []}
    fake_llm.script(answer("Rest at home [4].", [4]))
    assert (await post_ask(client, h, "Can I go to work?")).status_code == 200
    messages = fake_llm.calls[-1][0]
    assert [m["role"] for m in messages] == ["system", "user"]  # no old-document history
    # The care-team list keeps the question from the old document.
    assert [q["question"] for q in await care_team(client, h)] == ["Can I swim?"]


async def test_history_sends_last_six_messages(client, fake_llm):
    h = await confirmed(client, fake_llm)
    words = ["one", "two", "three", "four"]
    for word in words:
        fake_llm.script(answer(f"Answer {word} [2].", [2]))
        assert (await post_ask(client, h, f"Question {word}")).status_code == 200
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    await post_ask(client, h, "Last question")
    history = [m["content"] for m in fake_llm.calls[-1][0][1:-1]]
    assert history == [
        f"{prefix} {word}" + ("." if prefix == "Answer" else "")  # markers stripped
        for word in words[1:] for prefix in ("Question", "Answer")
    ]


async def test_not_covered_saves_exactly_one_care_team_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(
        {"content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "add_care_team_question", "arguments": json.dumps(
                {"question": "Can I swim with my IV site?", "reason": "Not in my document"})},
        }]},
        answer("I added this to your care-team questions.", [], kind="not_covered"),
    )
    body = (await post_ask(client, h, "Can I swim?")).json()
    assert body["kind"] == "not_covered" and body["flagged"] is True
    fake_llm.script(answer("I could not find this.", [], kind="not_covered"))
    await post_ask(client, h, "Can I fly?")
    questions = await care_team(client, h)
    assert [(q["question"], q["reason"], q["source"]) for q in questions] == [
        ("Can I fly?", "Not answered by the discharge document or approved sources.", "agent"),
        ("Can I swim with my IV site?", "Not in my document", "agent"),
    ]


async def test_validation_fallback_saves_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Take amoxicillin 900 mg [2].", [2]))
    body = (await post_ask(client, h, "What is my dose?")).json()
    assert body["kind"] == "not_covered" and body["answer"] == FALLBACK
    assert body["sources"] == [] and body["flagged"] is True
    questions = await care_team(client, h)
    assert [(q["question"], q["source"]) for q in questions] == [
        ("What is my dose?", "validation_fallback")
    ]


async def test_ask_validates_question_length(client, fake_llm):
    h = await confirmed(client, fake_llm)
    for question in ("", "   \n ", "x" * 2001):
        assert (await post_ask(client, h, question)).status_code == 422
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    assert (await post_ask(client, h, " " + "x" * 2000 + " ")).status_code == 200


async def test_ask_llm_error_502(client, fake_llm):
    h = await confirmed(client, fake_llm)

    async def broken(messages, tools=None, json_mode=False):
        raise LLMError("down")

    fake_llm.chat = broken
    resp = await post_ask(client, h, "How often?")
    assert resp.status_code == 502
    assert (await client.get("/api/conversation", headers=h)).json() == {"messages": []}


@pytest_asyncio.fixture
async def app_and_client(fake_llm, tmp_path):
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production"
    )
    app = create_app(settings, llm=fake_llm, store=KnowledgeStore(tmp_path / "i.sqlite3", fake_llm))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield app, c


async def test_ask_without_llm_503(app_and_client, fake_llm):
    app, client = app_and_client
    h = await confirmed(client, fake_llm)
    app.state.llm = None
    resp = await post_ask(client, h, "How often?")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "The language model is not configured."


def test_create_app_without_model_has_no_llm_or_store():
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production"
    )
    app = create_app(settings)
    assert app.state.llm is None and app.state.store is None


def test_create_app_builds_llm_client_and_store_from_settings(tmp_path):
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production",
        llm_model="local-model", rag_db_path=str(tmp_path / "index.sqlite3"),
    )
    app = create_app(settings)
    assert isinstance(app.state.llm, LLMClient)
    assert isinstance(app.state.store, KnowledgeStore)
    assert app.state.store.path == tmp_path / "index.sqlite3"
    assert app.state.foods.lookup("almonds")[0]["name"] == "Almonds"
