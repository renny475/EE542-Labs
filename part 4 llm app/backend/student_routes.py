"""Private Q&A and curator-only file ingestion for the Part 4 website/TWA."""
import asyncio
import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from knowledge import KnowledgeStore, answer_question, extract_document, official_url, make_record, save_record

log = logging.getLogger(__name__)

class Turn(BaseModel):
    role: Literal['user','assistant']
    content: str = Field(min_length=1, max_length=8000)

class Profile(BaseModel):
    degree: Literal['Not specified','Undergraduate',"Master's",'Doctoral','Other'] = 'Not specified'
    stage: Literal['Not specified','Enrolled','Graduating','On OPT','On STEM OPT'] = 'Not specified'
    school: str = Field(default='', max_length=100)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=6)
    profile: Profile = Field(default_factory=Profile)


def create_student_router(auth_dependency):
    router = APIRouter(prefix='/api/student', tags=['USC student assistant'])
    store = KnowledgeStore()
    # Bound local-model/embedding requests; the existing login token is required.
    queue = asyncio.Semaphore(2)
    upload_lock = asyncio.Lock()

    @router.get('/knowledge')
    async def knowledge_status():
        return await asyncio.to_thread(store.status)

    @router.post('/ask')
    async def ask(data: Question, user=Depends(auth_dependency)):
        question = data.question.strip()
        if not question:
            raise HTTPException(400, 'Enter a question.')
        try:
            async with queue:
                return await answer_question(store, question,
                    [t.model_dump() for t in data.history], data.profile.model_dump())
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError):
            log.warning('Student assistant upstream request failed', exc_info=True)
            raise HTTPException(502, 'The embedding or chat service is unavailable. Check the server configuration and try again.')

    @router.post('/documents')
    async def upload_document(
        file: UploadFile = File(...), title: str = Form(..., min_length=1, max_length=200),
        url: str = Form(..., max_length=1000), topic: Literal['F-1','CPT','OPT','STEM OPT','Travel'] = Form(...),
        x_knowledge_admin_key: str = Header(default=''), user=Depends(auth_dependency),
    ):
        expected = os.getenv('KNOWLEDGE_ADMIN_KEY','')
        if not expected or not secrets.compare_digest(x_knowledge_admin_key, expected):
            raise HTTPException(403, 'A knowledge-base curator key is required.')
        if not official_url(url):
            raise HTTPException(400, 'Provide an official USC OIS, USCIS, DHS Study in the States, or eCFR source URL.')
        data = await file.read(10 * 1024 * 1024 + 1)
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(413, 'Use a file smaller than 10 MB.')
        try:
            pages = await asyncio.to_thread(extract_document, data, Path(file.filename or '').suffix)
            record = make_record(data, title, url, topic)
            async with upload_lock:
                chunks = await store.add(record, pages)
                await asyncio.to_thread(save_record, record, pages)
            return {'ok':True, 'chunks':chunks}
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except httpx.HTTPError:
            raise HTTPException(502, 'Embedding service unavailable. The existing index was preserved.')
        except Exception:
            log.warning('Document extraction failed', exc_info=True)
            raise HTTPException(400, 'Could not process this document. Check that it contains readable text.')

    return router
