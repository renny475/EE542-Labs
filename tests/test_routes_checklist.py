from conftest import auth_headers
from test_routes_documents import DOC, extraction, upload_text

ITEMS = [
    {
        "category": "medication",
        "text": "Take amoxicillin 500 mg three times daily for 7 days.",
        "source_quote": "Take amoxicillin 500 mg by mouth three times daily for 7 days.",
        "details": {
            "name": "Amoxicillin", "dose": "500 mg", "times_per_day": 3, "duration_days": 7,
        },
    },
    {
        "category": "follow_up",
        "text": "Call your doctor if your fever is above 101 F.",
        "source_quote": "Call your doctor if your fever is above 101 F.",
        "details": {},
    },
]


async def confirmed(client, fake_llm, username="alice", items=ITEMS):
    h = await auth_headers(client, username)
    resp = await upload_text(client, h, fake_llm, items=items)
    assert resp.status_code == 200, resp.text
    conf = await client.post("/api/documents/active/confirm", headers=h)
    assert conf.status_code == 200, conf.text
    return h


async def test_checklist_lists_today_items_by_slot(client, fake_llm):
    h = await confirmed(client, fake_llm)
    resp = await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["date"] == "2026-03-02"
    assert [s["slot"] for s in body["slots"]] == ["morning", "midday", "evening"]
    first = body["slots"][0]["items"][0]
    assert set(first) == {"id", "label", "done", "source_quote", "approximate"}
    assert first["label"] == "Amoxicillin 500 mg" and first["done"] is False
    assert first["source_quote"].startswith("Take amoxicillin")
    assert [u["label"] for u in body["undated"]] == [
        "Call your doctor if your fever is above 101 F."
    ]
    assert set(body["undated"][0]) == {"id", "label", "source_quote"}


async def test_checklist_empty_before_anchor_and_after_course(client, fake_llm):
    h = await confirmed(client, fake_llm)
    for day in ("2026-03-01", "2026-03-09"):
        body = (await client.get("/api/checklist", params={"date": day}, headers=h)).json()
        assert body["slots"] == []
    last = (await client.get("/api/checklist", params={"date": "2026-03-08"}, headers=h)).json()
    assert len(last["slots"]) == 3


async def test_one_time_follow_up_only_on_its_date(client, fake_llm):
    items = [{
        "category": "follow_up", "text": "See cardiology",
        "source_quote": "Call your doctor if your fever is above 101 F.",
        "details": {"date": "2026-03-10"},
    }]
    h = await confirmed(client, fake_llm, items=items)
    on = (await client.get("/api/checklist", params={"date": "2026-03-10"}, headers=h)).json()
    assert on["slots"][0]["slot"] == "anytime"
    for day in ("2026-03-09", "2026-03-11"):
        assert (await client.get("/api/checklist", params={"date": day}, headers=h)).json()["slots"] == []


async def test_completion_toggle_persists(client, fake_llm):
    h = await confirmed(client, fake_llm)
    day = {"date": "2026-03-02"}
    item_id = (await client.get("/api/checklist", params=day, headers=h)).json()["slots"][0]["items"][0]["id"]
    url = f"/api/checklist/{item_id}/completion"
    assert (await client.put(url, json={"date": "2026-03-02", "done": True}, headers=h)).status_code == 200
    assert (await client.put(url, json={"date": "2026-03-02", "done": True}, headers=h)).status_code == 200
    body = (await client.get("/api/checklist", params=day, headers=h)).json()
    assert body["slots"][0]["items"][0]["done"] is True
    other = (await client.get("/api/checklist", params={"date": "2026-03-03"}, headers=h)).json()
    assert other["slots"][0]["items"][0]["done"] is False
    assert (await client.put(url, json={"date": "2026-03-02", "done": False}, headers=h)).status_code == 200
    body = (await client.get("/api/checklist", params=day, headers=h)).json()
    assert body["slots"][0]["items"][0]["done"] is False


async def test_completion_outside_range_400(client, fake_llm):
    h = await confirmed(client, fake_llm)
    item_id = (await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)).json()[
        "slots"][0]["items"][0]["id"]
    url = f"/api/checklist/{item_id}/completion"
    for day in ("2026-03-01", "2026-03-09"):
        resp = await client.put(url, json={"date": day, "done": True}, headers=h)
        assert resp.status_code == 400


async def test_completion_other_user_404(client, fake_llm):
    h = await confirmed(client, fake_llm)
    item_id = (await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)).json()[
        "slots"][0]["items"][0]["id"]
    h2 = await auth_headers(client, "bob")
    resp = await client.put(
        f"/api/checklist/{item_id}/completion", json={"date": "2026-03-02", "done": True}, headers=h2
    )
    assert resp.status_code == 404
    assert (await client.put("/api/checklist/9999/completion",
                             json={"date": "2026-03-02", "done": True}, headers=h)).status_code == 404


async def test_checklist_without_confirmed_doc_409(client, fake_llm):
    h = await auth_headers(client)
    assert (await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)).status_code == 409
    await upload_text(client, h, fake_llm)
    assert (await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)).status_code == 409


async def test_invalid_date_422(client, fake_llm):
    h = await confirmed(client, fake_llm)
    assert (await client.get("/api/checklist", params={"date": "03/02/2026"}, headers=h)).status_code == 422
    assert (await client.get("/api/checklist", headers=h)).status_code == 422
    item_id = (await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)).json()[
        "slots"][0]["items"][0]["id"]
    assert (await client.put(f"/api/checklist/{item_id}/completion",
                             json={"date": "nope", "done": True}, headers=h)).status_code == 422


async def test_reupload_hides_old_checklist(client, fake_llm):
    h = await confirmed(client, fake_llm)
    await upload_text(client, h, fake_llm)
    assert (await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)).status_code == 409


async def test_rejected_item_not_in_checklist(client, fake_llm):
    h = await auth_headers(client)
    doc = (await upload_text(client, h, fake_llm, items=ITEMS)).json()
    med_id = doc["items"][0]["id"]
    await client.post(f"/api/documents/active/items/{med_id}/reject", headers=h)
    await client.post("/api/documents/active/confirm", headers=h)
    body = (await client.get("/api/checklist", params={"date": "2026-03-02"}, headers=h)).json()
    assert body["slots"] == [] and len(body["undated"]) == 1
