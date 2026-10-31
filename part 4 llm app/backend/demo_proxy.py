"""Temporary public-demo gateway: expose the assistant, not the group-chat app.

Run on localhost:8003 and point cloudflared at it. The main app remains on 8002.
Signup, curator uploads, legacy chat, WebSockets, and API docs are not published.
"""
import asyncio
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

load_dotenv(Path(__file__).resolve().parents[1]/'.env')
UPSTREAM='http://127.0.0.1:8002'
ALLOWED={
    ('GET','/'),('GET','/student.js'),('GET','/student.css'),
    ('GET','/.well-known/assetlinks.json'),('GET','/api/student/knowledge'),
    ('POST','/api/login'),('POST','/api/student/ask')}
windows=defaultdict(deque)
question_count=0
inflight=asyncio.Semaphore(2)

@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=125) as client:
        app.state.client=client
        yield

app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)

@app.api_route('/{path:path}',methods=['GET','POST','PUT','PATCH','DELETE','OPTIONS'])
async def forward(request:Request,path:str):
    global question_count
    key=(request.method,'/'+path)
    if key not in ALLOWED:
        return JSONResponse({'detail':'This route is not available in the public demo.'},status_code=404)
    if request.method=='POST':
        ip=request.headers.get('cf-connecting-ip',request.client.host)
        bucket=(ip,path)
        now=time.monotonic(); recent=windows[bucket]
        while recent and recent[0]<now-60:recent.popleft()
        limit=6 if path=='api/login' else 12
        if len(recent)>=limit:
            return JSONResponse({'detail':'Please wait a minute before trying again.'},status_code=429)
        recent.append(now)
        if path=='api/student/ask':
            if question_count>=int(os.getenv('DEMO_MAX_QUESTIONS','200')):
                return JSONResponse({'detail':'This demonstration has reached its question limit.'},status_code=429)
            question_count+=1
    body=await request.body()
    if len(body)>100_000:
        return JSONResponse({'detail':'Request is too large.'},status_code=413)
    headers={k:v for k,v in request.headers.items() if k.lower() in {'authorization','content-type'}}
    try:
        async with inflight:
            upstream=await request.app.state.client.request(request.method,UPSTREAM+'/'+path,headers=headers,content=body)
    except httpx.HTTPError:
        return JSONResponse({'detail':'The local assistant is unavailable. Please try again later.'},status_code=503)
    content=upstream.content
    if path=='' and upstream.is_success:
        text=upstream.text.replace('class="legacy-link"','class="legacy-link hidden"')
        text=text.replace('id="signupBtn"','id="signupBtn" hidden')
        text=text.replace('Use your team’s chat account, or create one.','Log in with an existing project account.')
        content=text.encode()
    return Response(content,status_code=upstream.status_code,media_type=upstream.headers.get('content-type'),
        headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})
