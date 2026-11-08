from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import HISTORY_LIMIT, emergency_result, run_agent
from auth import current_user
from db import get_session
from llm import LLMError
from models import ChatMessage, User
from routes_care_team import add_question
from routes_documents import active_document, confirmed_document, load_items
from safety import detect_emergency, warning_terms

router = APIRouter()

MAX_QUESTION_CHARS = 2000


class AskBody(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]


def message_view(m: ChatMessage) -> dict:
    return {
        "role": m.role,
        "content": m.content,
        "payload": m.payload,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


async def recent_history(session: AsyncSession, document_id: int) -> list[dict]:
    rows = await session.scalars(
        select(ChatMessage)
        .where(ChatMessage.document_id == document_id)
        .order_by(ChatMessage.id.desc())
        .limit(HISTORY_LIMIT)
    )
    return [{"role": m.role, "content": m.content} for m in reversed(list(rows))]


@router.post("/api/ask")
async def ask(
    body: AskBody,
    request: Request,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    doc = await confirmed_document(session, user)
    items = await load_items(session, doc)
    # The backstop runs first on every message; the emergency path never depends
    # on the model (R19).
    terms = detect_emergency(body.question, warning_terms(items))
    llm = request.app.state.llm
    if llm is None:
        if not terms:
            raise HTTPException(status_code=503, detail="The language model is not configured.")
        result = emergency_result()
    else:
        history = await recent_history(session, doc.id)
        await session.commit()  # end the read transaction before the slow model calls
        try:
            result = await run_agent(
                llm, request.app.state.store, request.app.state.foods,
                doc, items, history, body.question, terms,
            )
        except LLMError as exc:
            if not terms:
                raise HTTPException(
                    status_code=502,
                    detail="The assistant could not answer right now. Please try again.",
                ) from exc
            result = emergency_result()

    payload = {
        "kind": result.kind,
        "answer": result.answer,
        "sources": [
            {"number": e.number, "kind": e.kind, "title": e.title, "url": e.url, "content": e.content}
            for e in result.sources
        ],
        "flagged": result.flagged,
        "emergency": {"terms": terms, "banner": bool(terms) or result.kind == "emergency"},
    }
    session.add(ChatMessage(document_id=doc.id, role="user", content=body.question))
    session.add(ChatMessage(document_id=doc.id, role="assistant", content=result.answer,
                            payload=payload))
    if result.care_team_question:
        await add_question(
            session, user.id, doc.id,
            question=result.care_team_question,
            reason=result.care_team_reason,
            source=result.care_team_source,
        )
    await session.commit()
    return payload


@router.get("/api/conversation")
async def conversation(
    user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    try:
        doc = await active_document(session, user)
    except HTTPException:
        return {"messages": []}
    rows = await session.scalars(
        select(ChatMessage).where(ChatMessage.document_id == doc.id).order_by(ChatMessage.id)
    )
    return {"messages": [message_view(m) for m in rows]}
