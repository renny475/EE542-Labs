from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_user
from checklist import SLOT_ORDER
from db import get_session
from models import ChecklistCompletion, ChecklistItem, DischargeDocument, ExtractedItem, User
from routes_documents import active_document

router = APIRouter()


class CompletionBody(BaseModel):
    date: date
    done: bool


async def confirmed_document(session: AsyncSession, user: User) -> DischargeDocument:
    try:
        doc = await active_document(session, user)
    except HTTPException:
        doc = None
    if doc is None or doc.status != "confirmed":
        raise HTTPException(status_code=409, detail="Confirm your discharge document first.")
    return doc


def scheduled_on(day: date):
    """SQL condition: a dated checklist item that applies on `day`."""
    return and_(
        ChecklistItem.undated.is_(False),
        ChecklistItem.start_date <= day,
        or_(ChecklistItem.end_date.is_(None), ChecklistItem.end_date >= day),
    )


@router.get("/api/checklist")
async def get_checklist(
    date: date, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    day = date
    doc = await confirmed_document(session, user)
    rows = (
        await session.execute(
            select(ChecklistItem, ExtractedItem.source_quote)
            .outerjoin(ExtractedItem, ExtractedItem.id == ChecklistItem.extracted_item_id)
            .where(ChecklistItem.document_id == doc.id)
            .order_by(ChecklistItem.id)
        )
    ).all()
    done = set(
        await session.scalars(
            select(ChecklistCompletion.checklist_item_id)
            .join(ChecklistItem, ChecklistItem.id == ChecklistCompletion.checklist_item_id)
            .where(ChecklistItem.document_id == doc.id, ChecklistCompletion.day == day)
        )
    )
    by_slot: dict[str, list[dict]] = {}
    undated = []
    for item, quote in rows:
        if item.undated:
            undated.append({"id": item.id, "label": item.label, "source_quote": quote or ""})
        elif item.start_date <= day and (item.end_date is None or day <= item.end_date):
            by_slot.setdefault(item.slot, []).append(
                {
                    "id": item.id,
                    "label": item.label,
                    "done": item.id in done,
                    "source_quote": quote or "",
                    "approximate": item.approximate,
                }
            )
    return {
        "date": day.isoformat(),
        "slots": [{"slot": s, "items": by_slot[s]} for s in SLOT_ORDER if s in by_slot],
        "undated": undated,
    }


@router.put("/api/checklist/{item_id}/completion")
async def set_completion(
    item_id: int,
    body: CompletionBody,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    # Only items of the user's current confirmed document exist for this endpoint.
    item = await session.scalar(
        select(ChecklistItem)
        .join(DischargeDocument, DischargeDocument.id == ChecklistItem.document_id)
        .where(
            ChecklistItem.id == item_id,
            DischargeDocument.user_id == user.id,
            DischargeDocument.status == "confirmed",
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Checklist item not found")
    in_range = await session.scalar(
        select(ChecklistItem.id).where(ChecklistItem.id == item.id, scheduled_on(body.date))
    )
    if in_range is None:
        raise HTTPException(status_code=400, detail="This item is not scheduled on that date.")
    existing = await session.scalar(
        select(ChecklistCompletion.id).where(
            ChecklistCompletion.checklist_item_id == item.id, ChecklistCompletion.day == body.date
        )
    )
    if body.done and existing is None:
        session.add(ChecklistCompletion(checklist_item_id=item.id, day=body.date))
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()  # a concurrent identical request already recorded it
    elif not body.done and existing is not None:
        await session.execute(delete(ChecklistCompletion).where(ChecklistCompletion.id == existing))
        await session.commit()
    return {"id": item.id, "date": body.date.isoformat(), "done": body.done}
