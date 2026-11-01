# USC Student Guide

Unofficial USC international-student assistant for F-1 status, CPT, OPT, and STEM OPT. The web app retrieves passages from a curated knowledge base and generates answers with source citations.

## Run the website

1. Install Python 3.10+ and MySQL 8+. Create the database with `sql/schema.sql`; use your own database password.
2. From the project root:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   cp .env.example .env
   ```

3. In `.env`, set `DATABASE_URL`, a unique `JWT_SECRET`, and `OPENAI_API_KEY`. Enable the `RAG_CHAT_API_BASE` and `RAG_CHAT_MODEL` lines to use the configured OpenAI chat model. Set `KNOWLEDGE_ADMIN_KEY` if using curator uploads. Keep `.env` private.
4. Start the app:

   ```bash
   cd backend
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. Open `http://localhost:8000`, create a project account, and log in. Questions appear in the student guide; source citations expand to show the retrieved passages.

## Knowledge base

`knowledge/index.sqlite3` contains the ready index: 23 source documents and 258 embedded passages. `knowledge/sources/` preserves the source text and provenance. The eight USC notes are curated summaries; fifteen federal sources contain regulation subsections. `knowledge/EMBEDDING_EXAMPLE.json` provides a stored-vector example for the lab demonstration.

From `backend/`, run `python manage_knowledge.py status` to inspect the index. To rebuild after reviewing changed source snapshots, run `python manage_knowledge.py index`; this makes paid embedding calls. `import_ecfr.py` refreshes the federal source snapshots.

## Android

`USC-Student-Guide-demo.apk` is the signed demonstration app. `twa_android_src/` contains its buildable source and Gradle wrapper. Build it using JDK 17+ and the Android SDK, with `./gradlew assembleDebug` inside that folder.
