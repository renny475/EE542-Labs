from conftest import auth_headers
from test_routes_documents import upload_text


async def make_question(client, headers, fake_llm):
    doc = (await upload_text(client, headers, fake_llm)).json()
    await client.post(f"/api/documents/active/items/{doc['items'][0]['id']}/reject", headers=headers)
    return doc


async def test_resolve_question(client, fake_llm):
    h = await auth_headers(client)
    await make_question(client, h, fake_llm)
    qid = (await client.get("/api/care-team", headers=h)).json()["questions"][0]["id"]
    assert (await client.post(f"/api/care-team/{qid}/resolve", headers=h)).status_code == 200
    assert (await client.get("/api/care-team", headers=h)).json()["questions"][0]["resolved"] is True


async def test_care_team_newest_first(client, fake_llm):
    h = await auth_headers(client)
    doc = (await upload_text(client, h, fake_llm)).json()
    for item in doc["items"]:
        await client.post(f"/api/documents/active/items/{item['id']}/reject", headers=h)
    qs = (await client.get("/api/care-team", headers=h)).json()["questions"]
    assert [q["question"] for q in qs] == [
        "Please confirm this instruction: " + i["text"] for i in reversed(doc["items"])
    ]


async def test_other_users_question_404(client, fake_llm):
    ha = await auth_headers(client, "alice")
    hb = await auth_headers(client, "bob")
    await make_question(client, ha, fake_llm)
    qid = (await client.get("/api/care-team", headers=ha)).json()["questions"][0]["id"]
    assert (await client.post(f"/api/care-team/{qid}/resolve", headers=hb)).status_code == 404
    assert (await client.get("/api/care-team", headers=hb)).json() == {"questions": []}
    assert (await client.get("/api/care-team", headers=ha)).json()["questions"][0]["resolved"] is False
    assert (await client.post("/api/care-team/99999/resolve", headers=ha)).status_code == 404


async def test_care_team_list_survives_reupload(client, fake_llm):
    h = await auth_headers(client)
    first = await make_question(client, h, fake_llm)
    second = (await upload_text(client, h, fake_llm, text="Second document body text.")).json()
    qs = (await client.get("/api/care-team", headers=h)).json()["questions"]
    assert len(qs) == 1 and qs[0]["document_id"] == first["id"] != second["id"]
