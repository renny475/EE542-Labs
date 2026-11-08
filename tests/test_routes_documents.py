import json
from datetime import datetime, timezone

from conftest import auth_headers

DOC = (
    "Discharge date: 2026-03-02\n\n"
    "Take amoxicillin 500 mg by mouth three times daily for 7 days.\n\n"
    "Call your doctor if your fever is above 101 F."
)


def extraction(items=None, discharge_date="2026-03-02"):
    if items is None:
        items = [
            {
                "category": "medication",
                "text": "Take amoxicillin 500 mg three times daily for 7 days.",
                "source_quote": "Take amoxicillin 500 mg by mouth three times daily for 7 days.",
                "details": {"name": "amoxicillin"},
            },
            {
                "category": "warning_signs",
                "text": "Call your doctor if fever is above 101 F.",
                "source_quote": "Call your doctor if your fever is above 101 F.",
                "details": {},
            },
            {
                "category": "diet",
                "text": "Avoid salt.",
                "source_quote": "this quote is not in the document",
                "details": {},
            },
        ]
    return {"content": json.dumps({"discharge_date": discharge_date, "items": items})}


async def upload_text(client, headers, fake_llm, text=DOC, **kw):
    fake_llm.script(extraction(**kw))
    return await client.post("/api/documents", data={"text": text}, headers=headers)


async def test_upload_text_creates_pending_doc_with_items(client, fake_llm):
    h = await auth_headers(client)
    resp = await upload_text(client, h, fake_llm)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending_review"
    assert body["filename"] is None
    assert body["discharge_date"] == "2026-03-02"
    assert body["anchor_date"] is None
    assert len(body["paragraphs"]) == 3
    assert [i["category"] for i in body["items"]] == ["medication", "warning_signs"]
    assert set(body["items"][0]) == {"id", "category", "text", "source_quote", "details", "status"}
    assert body["items"][0]["status"] == "active"
    assert body["unverified"] == [{"category": "diet", "text": "Avoid salt."}]
    active = await client.get("/api/documents/active", headers=h)
    assert active.status_code == 200 and active.json()["id"] == body["id"]


async def test_upload_file_uses_filename(client, fake_llm):
    h = await auth_headers(client)
    fake_llm.script(extraction())
    resp = await client.post(
        "/api/documents", files={"file": ("d.txt", DOC.encode(), "text/plain")}, headers=h
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["filename"] == "d.txt"


async def test_upload_requires_exactly_one_of_file_or_text(client, fake_llm):
    h = await auth_headers(client)
    neither = await client.post("/api/documents", headers=h)
    assert neither.status_code == 400
    both = await client.post(
        "/api/documents",
        data={"text": DOC},
        files={"file": ("d.txt", DOC.encode(), "text/plain")},
        headers=h,
    )
    assert both.status_code == 400
    blank = await client.post("/api/documents", data={"text": "   "}, headers=h)
    assert blank.status_code == 400
    assert fake_llm.calls == []
    assert (await client.get("/api/documents/active", headers=h)).status_code == 404


async def test_upload_too_large_413(client, fake_llm):
    h = await auth_headers(client)
    big = await client.post("/api/documents", data={"text": "a" * 200_001}, headers=h)
    assert big.status_code == 413
    bigfile = await client.post(
        "/api/documents",
        files={"file": ("d.txt", b"a" * (10 * 1024 * 1024 + 1), "text/plain")},
        headers=h,
    )
    assert bigfile.status_code == 413


async def test_upload_requires_auth(client):
    assert (await client.post("/api/documents", data={"text": DOC})).status_code == 401
    assert (await client.get("/api/documents/active")).status_code == 401
    assert (await client.get("/api/care-team")).status_code == 401


async def test_upload_scanned_pdf_400(client, fake_llm):
    from pypdf import PdfWriter
    import io

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    h = await auth_headers(client)
    resp = await client.post(
        "/api/documents", files={"file": ("scan.pdf", buf.getvalue(), "application/pdf")}, headers=h
    )
    assert resp.status_code == 400
    assert "Scanned PDFs need OCR first" in resp.json()["detail"]


async def test_upload_llm_failure_502_and_no_doc(client, fake_llm):
    h = await auth_headers(client)
    first = await upload_text(client, h, fake_llm)
    fake_llm.script({"content": "not json"})
    resp = await client.post("/api/documents", data={"text": "Another document body."}, headers=h)
    assert resp.status_code == 502
    active = await client.get("/api/documents/active", headers=h)
    assert active.json()["id"] == first.json()["id"]
    assert active.json()["status"] == "pending_review"


async def test_upload_all_quotes_unverified_gives_empty_review(client, fake_llm):
    h = await auth_headers(client)
    resp = await upload_text(
        client, h, fake_llm,
        items=[{"category": "diet", "text": "Eat less salt.", "source_quote": "nothing like this here", "details": {}}],
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == []
    assert len(resp.json()["unverified"]) == 1


async def test_reupload_archives_previous(client, fake_llm):
    h = await auth_headers(client)
    first = (await upload_text(client, h, fake_llm)).json()
    await client.post("/api/documents/active/confirm", headers=h)
    second = (await upload_text(client, h, fake_llm, text=DOC + "\n\nExtra paragraph here.")).json()
    assert second["id"] != first["id"]
    active = (await client.get("/api/documents/active", headers=h)).json()
    assert active["id"] == second["id"] and active["status"] == "pending_review"

    from sqlalchemy import select
    from models import DischargeDocument

    async with client._transport.app.state.session_factory() as s:
        rows = (await s.execute(select(DischargeDocument).order_by(DischargeDocument.id))).scalars().all()
    assert [r.status for r in rows] == ["archived", "pending_review"]


async def test_reject_item_adds_care_team_question(client, fake_llm):
    h = await auth_headers(client)
    doc = (await upload_text(client, h, fake_llm)).json()
    item = doc["items"][0]
    resp = await client.post(f"/api/documents/active/items/{item['id']}/reject", headers=h)
    assert resp.status_code == 200
    assert resp.json()["items"][0]["status"] == "rejected"
    qs = (await client.get("/api/care-team", headers=h)).json()["questions"]
    assert len(qs) == 1
    q = qs[0]
    assert q["question"] == "Please confirm this instruction: " + item["text"]
    assert q["reason"] == "Extracted item did not match my document"
    assert q["source"] == "rejected_item"
    assert q["resolved"] is False
    assert q["document_id"] == doc["id"]
    # idempotent: rejecting again does not duplicate the question
    again = await client.post(f"/api/documents/active/items/{item['id']}/reject", headers=h)
    assert again.status_code == 200
    assert len((await client.get("/api/care-team", headers=h)).json()["questions"]) == 1
    missing = await client.post("/api/documents/active/items/99999/reject", headers=h)
    assert missing.status_code == 404


async def test_reject_after_confirm_409(client, fake_llm):
    h = await auth_headers(client)
    doc = (await upload_text(client, h, fake_llm)).json()
    assert (await client.post("/api/documents/active/confirm", headers=h)).status_code == 200
    resp = await client.post(f"/api/documents/active/items/{doc['items'][0]['id']}/reject", headers=h)
    assert resp.status_code == 409
    assert (await client.post("/api/documents/active/confirm", headers=h)).status_code == 409
    fake_llm.script(extraction())
    assert (await client.post("/api/documents/active/re-extract", headers=h)).status_code == 409


async def test_confirm_uses_document_date_else_today(client, fake_llm):
    h = await auth_headers(client)
    await upload_text(client, h, fake_llm)
    resp = await client.post("/api/documents/active/confirm", headers=h)
    body = resp.json()
    assert body["status"] == "confirmed"
    assert body["anchor_date"] == "2026-03-02"
    assert body["date_source"] == "document"

    h2 = await auth_headers(client, "bob")
    await upload_text(client, h2, fake_llm, discharge_date=None)
    body2 = (await client.post("/api/documents/active/confirm", headers=h2)).json()
    assert body2["anchor_date"] == datetime.now(timezone.utc).date().isoformat()
    assert body2["date_source"] == "confirmation"


async def test_re_extract_replaces_items(client, fake_llm):
    h = await auth_headers(client)
    await upload_text(client, h, fake_llm)
    fake_llm.script(extraction(items=[{
        "category": "activity", "text": "No heavy lifting.",
        "source_quote": "Call your doctor if your fever", "details": {},
    }]))
    resp = await client.post("/api/documents/active/re-extract", headers=h)
    assert resp.status_code == 200
    assert [i["category"] for i in resp.json()["items"]] == ["activity"]
    assert resp.json()["unverified"] == []
    # failure leaves things unchanged
    fake_llm.script({"content": "garbage"})
    bad = await client.post("/api/documents/active/re-extract", headers=h)
    assert bad.status_code == 502
    kept = (await client.get("/api/documents/active", headers=h)).json()
    assert [i["category"] for i in kept["items"]] == ["activity"]


async def test_other_users_document_404(client, fake_llm):
    ha = await auth_headers(client, "alice")
    hb = await auth_headers(client, "bob")
    doc = (await upload_text(client, ha, fake_llm)).json()
    assert (await client.get("/api/documents/active", headers=hb)).status_code == 404
    rej = await client.post(f"/api/documents/active/items/{doc['items'][0]['id']}/reject", headers=hb)
    assert rej.status_code == 404
    assert (await client.post("/api/documents/active/confirm", headers=hb)).status_code == 404
    assert (await client.post("/api/documents/active/re-extract", headers=hb)).status_code == 404
