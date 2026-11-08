from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_user
from db import get_session
from models import CareTeamQuestion, User

router = APIRouter()


def question_view(q: CareTeamQuestion) -> dict:
    return {
        "id": q.id,
        "question": q.question,
        "reason": q.reason,
        "source": q.source,
        "resolved": q.resolved,
        "created_at": q.created_at.isoformat() if q.created_at else None,
        "document_id": q.document_id,
    }


async def add_question(
    session: AsyncSession,
    user_id: int,
    document_id: int | None,
    question: str,
    reason: str,
    source: str,
) -> CareTeamQuestion:
    """Add a question to the user's care-team list. The caller commits."""
    row = CareTeamQuestion(
        user_id=user_id,
        document_id=document_id,
        question=question,
        reason=reason,
        source=source,
    )
    session.add(row)
    await session.flush()
    return row


@router.get("/api/care-team")
async def list_questions(
    user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    rows = await session.scalars(
        select(CareTeamQuestion)
        .where(CareTeamQuestion.user_id == user.id)
        .order_by(CareTeamQuestion.id.desc())
    )
    return {"questions": [question_view(q) for q in rows]}


@router.post("/api/care-team/{question_id}/resolve")
async def resolve_question(
    question_id: int,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    row = await session.scalar(
        select(CareTeamQuestion).where(
            CareTeamQuestion.id == question_id, CareTeamQuestion.user_id == user.id
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Question not found")
    row.resolved = True
    await session.commit()
    return question_view(row)
