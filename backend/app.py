from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import auth
import routes_agent
import routes_care_team
import routes_checklist
import routes_documents
from config import PROJECT_DIR, Settings, get_settings
from db import init_db, make_engine, make_session_factory
from foods import FoodTable
from knowledge import KnowledgeStore
from llm import LLMClient

FRONTEND_DIR = PROJECT_DIR / "frontend"


def create_app(settings: Optional[Settings] = None, llm=None, store=None, foods=None) -> FastAPI:
    """`llm`, `store` and `foods` default to what `settings` configures. Without
    LLM_MODEL there is no model (model routes answer 503) and no library store."""
    settings = settings or get_settings()
    settings.validate()
    if llm is None and settings.llm_model:
        llm = LLMClient(settings)  # no API key required, so a self-hosted server works
    if store is None and llm is not None:
        store = KnowledgeStore(settings.rag_db_path, llm, settings.rag_min_similarity)
    if foods is None:
        foods_path = Path(settings.foods_path)
        foods = FoodTable.load(foods_path if foods_path.is_absolute() else PROJECT_DIR / foods_path)
    engine = make_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Tables are created automatically; sql/schema.sql is the manual alternative.
        await init_db(engine)
        yield
        await engine.dispose()

    app = FastAPI(title="AfterCare", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.llm = llm
    app.state.store = store
    app.state.foods = foods

    app.include_router(auth.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_care_team.router)
    app.include_router(routes_checklist.router)
    app.include_router(routes_agent.router)

    def page(name: str):
        async def handler():
            path = FRONTEND_DIR / name
            if not path.is_file():
                raise HTTPException(status_code=404, detail="Page not found")
            return FileResponse(path)
        return handler

    app.add_api_route("/", page("index.html"), include_in_schema=False)
    app.add_api_route("/app", page("app.html"), include_in_schema=False)
    # Mounted last so it never shadows /api routes.
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    return app


def __getattr__(name: str):
    # `uvicorn app:app` resolves this lazily so importing the module (tests)
    # does not require a configured JWT_SECRET.
    if name == "app":
        instance = create_app()
        globals()["app"] = instance
        return instance
    raise AttributeError(name)
