import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_user
from db import get_session
from documents import paragraphs, read_upload
from extraction import extract
from llm import LLMError
from checklist import build_checklist
from models import ChecklistItem, DischargeDocument, ExtractedItem, User
from routes_care_team import add_question

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
ACTIVE_STATUSES = ("pending_review", "confirmed")


def document_view(doc: DischargeDocument, items: list[ExtractedItem]) -> dict:
    return {
        "id": doc.id,
        "status": doc.status,
        "filename": doc.filename,
        "discharge_date": doc.discharge_date.isoformat() if doc.discharge_date else None,
        "anchor_date": doc.anchor_date.isoformat() if doc.anchor_date else None,
        "date_source": doc.date_source,
        "paragraphs": paragraphs(doc.text),
        "items": [
            {
                "id": i.id,
                "category": i.category,
                "text": i.text,
                "source_quote": i.source_quote,
                "details": i.details or {},
                "status": i.status,
            }
            for i in items
        ],
        "unverified": doc.unverified or [],
    }


async def load_items(session: AsyncSession, doc: DischargeDocument) -> list[ExtractedItem]:
    rows = await session.scalars(
        select(ExtractedItem)
        .where(ExtractedItem.document_id == doc.id)
        .order_by(ExtractedItem.id)
        .execution_options(populate_existing=True)
    )
    return list(rows)


async def lock_user(session: AsyncSession, user: User) -> None:
    """Serialize this user's document mutations (row lock on MySQL; SQLite ignores it).
    Ends the transaction opened by current_user first, so that under MySQL
    REPEATABLE READ the reads after the lock see a fresh snapshot."""
    await session.commit()
    await session.execute(select(User.id).where(User.id == user.id).with_for_update())


async def active_document(session: AsyncSession, user: User) -> DischargeDocument:
    doc = await session.scalar(
        select(DischargeDocument)
        .where(DischargeDocument.user_id == user.id, DischargeDocument.status.in_(ACTIVE_STATUSES))
        .order_by(DischargeDocument.id.desc())
        .limit(1)
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="No active document")
    return doc


async def pending_document(session: AsyncSession, user: User) -> DischargeDocument:
    doc = await active_document(session, user)
    if doc.status != "pending_review":
        raise HTTPException(status_code=409, detail="This document is already confirmed")
    return doc


async def run_extraction(request: Request, text: str):
    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="The language model is not configured.")
    try:
        return await extract(llm, text)
    except LLMError as exc:
        raise HTTPException(
            status_code=502, detail="The language model could not read this document. Try again."
        ) from exc


def new_items(document_id: int, extraction) -> list[ExtractedItem]:
    return [
        ExtractedItem(
            document_id=document_id,
            category=i["category"],
            text=i["text"],
            source_quote=i["source_quote"],
            details=i["details"],
            status="active",
        )
        for i in extraction.items
    ]


async def generate_checklist(session: AsyncSession, doc: DischargeDocument) -> None:
    """Create the checklist rows for a just-confirmed document. Flushes only;
    the confirm transaction commits."""
    items = await load_items(session, doc)
    for spec in build_checklist(items, doc.anchor_date):
        session.add(
            ChecklistItem(
                document_id=doc.id,
                extracted_item_id=spec.extracted_item_id,
                label=spec.label[:500],
                slot=spec.slot,
                start_date=spec.start_date,
                end_date=spec.end_date,
                one_time=spec.one_time,
                approximate=spec.approximate,
                undated=spec.undated,
            )
        )
    await session.flush()


@router.post("/api/documents")
async def upload_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    has_file = file is not None and bool(file.filename)
    has_text = text is not None and text.strip() != ""
    if has_file == has_text:
        raise HTTPException(status_code=400, detail="Provide either a file or pasted text, not both.")

    if has_file:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Use a file smaller than 10 MB.")
        try:
            body = await asyncio.to_thread(read_upload, data, file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        filename = file.filename[:255]
    else:
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=413, detail="Pasted text is too long.")
        body, filename = text.strip(), None

    # Extract before touching the database: on failure nothing changes and the
    # previous active document stays active.
    extraction = await run_extraction(request, body)

    await lock_user(session, user)
    await session.execute(
        update(DischargeDocument)
        .where(DischargeDocument.user_id == user.id, DischargeDocument.status.in_(ACTIVE_STATUSES))
        .values(status="archived", active_user_id=None)
    )
    doc = DischargeDocument(
        user_id=user.id,
        active_user_id=user.id,
        status="pending_review",
        filename=filename,
        text=body,
        discharge_date=extraction.discharge_date,
        unverified=extraction.unverified,
    )
    session.add(doc)
    await session.flush()
    items = new_items(doc.id, extraction)
    session.add_all(items)
    try:
        await session.commit()
    except IntegrityError as exc:
        # Lost a race with a concurrent upload (unique active_user_id); nothing was saved.
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="Another upload is in progress. Try again in a moment."
        ) from exc
    return document_view(doc, await load_items(session, doc))


@router.get("/api/documents/active")
async def get_active(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    doc = await active_document(session, user)
    return document_view(doc, await load_items(session, doc))


@router.post("/api/documents/active/items/{item_id}/reject")
async def reject_item(
    item_id: int, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    await lock_user(session, user)
    doc = await pending_document(session, user)
    item = await session.scalar(
        select(ExtractedItem).where(ExtractedItem.id == item_id, ExtractedItem.document_id == doc.id)
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    # Conditional transition: only the request that flips the status adds the question.
    flipped = await session.execute(
        update(ExtractedItem)
        .where(ExtractedItem.id == item.id, ExtractedItem.status == "active")
        .values(status="rejected")
    )
    if flipped.rowcount == 1:
        await add_question(
            session,
            user.id,
            doc.id,
            question="Please confirm this instruction: " + item.text,
            reason="Extracted item did not match my document",
            source="rejected_item",
        )
    await session.commit()
    return document_view(doc, await load_items(session, doc))


@router.post("/api/documents/active/re-extract")
async def re_extract(
    request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    doc = await pending_document(session, user)
    doc_id = doc.id
    extraction = await run_extraction(request, doc.text)
    await lock_user(session, user)
    doc = await pending_document(session, user)  # re-check under the lock
    if doc.id != doc_id:  # replaced by a new upload during extraction
        raise HTTPException(status_code=409, detail="This document was replaced")
    await session.refresh(doc)  # the identity map would otherwise keep the stale status
    if doc.status != "pending_review":
        raise HTTPException(status_code=409, detail="This document is already confirmed")
    # Conditional touch of the document: 409 if it left pending_review meanwhile.
    touched = await session.execute(
        update(DischargeDocument)
        .where(DischargeDocument.id == doc.id, DischargeDocument.status == "pending_review")
        .values(discharge_date=extraction.discharge_date, unverified=extraction.unverified)
    )
    if touched.rowcount != 1:
        raise HTTPException(status_code=409, detail="This document is already confirmed")
    await session.execute(delete(ExtractedItem).where(ExtractedItem.document_id == doc.id))
    session.add_all(new_items(doc.id, extraction))
    await session.commit()
    await session.refresh(doc)
    return document_view(doc, await load_items(session, doc))


@router.post("/api/documents/active/confirm")
async def confirm(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    await lock_user(session, user)
    doc = await pending_document(session, user)
    now = datetime.now(timezone.utc)
    if doc.discharge_date:
        anchor, source = doc.discharge_date, "document"
    else:
        anchor, source = now.date(), "confirmation"
    # Conditional transition: exactly one concurrent confirm wins, the rest get 409.
    flipped = await session.execute(
        update(DischargeDocument)
        .where(DischargeDocument.id == doc.id, DischargeDocument.status == "pending_review")
        .values(
            status="confirmed",
            confirmed_at=now.replace(tzinfo=None),
            anchor_date=anchor,
            date_source=source,
        )
    )
    if flipped.rowcount != 1:
        raise HTTPException(status_code=409, detail="This document is already confirmed")
    await session.refresh(doc)
    await generate_checklist(session, doc)
    await session.commit()
    return document_view(doc, await load_items(session, doc))
