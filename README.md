# AfterCare

AfterCare is an English web app for patients who go home after a hospital stay. The patient uploads the discharge document their doctor wrote (family medicine), reviews and confirms what the app extracted from it, and then asks an agent lifestyle and medication questions such as "is almond butter OK, and how much?". Every answer carries citations, and the doctor's document is always the top authority: the agent explains and applies it, and never overrides it. A daily checklist generated from the confirmed document is a secondary feature. Questions the agent cannot answer from approved sources are saved to a Care team list the patient can take to the next appointment.

AfterCare is a class project, not medical advice. In an emergency, call 911.

This is EE542 final project, Team 2, milestone 1. The "Status" section below lists plainly what is not built or not verified yet.

## Architecture

| Part | What it is |
|------|-----------|
| Backend | FastAPI, async SQLAlchemy. MySQL 8 in production (`asyncmy`), SQLite (`aiosqlite`) for local dev and tests |
| Auth | Sign up / log in with JWT; every query is scoped to the current user, another user's IDs return 404 |
| Frontend | Plain HTML/CSS/JS in `frontend/`, served by FastAPI. No framework, no build step |
| Models | Any OpenAI-compatible server: `/chat/completions` with standard `tools`/`tool_calls`, and `/embeddings` |
| Education library | Curated, allowlisted official sources (MedlinePlus, NIDDK, CDC, DailyMed, kidney.org), chunked and embedded into a SQLite vector index, exact cosine search with NumPy |
| Food table | `data/foods.json`: USDA FoodData Central nutrients plus oxalate categories from the UChicago oxalate table |
| Grounding | The server validates every answer. Citations must exist in the evidence of that request, and every number in an answer must appear in the cited evidence. Otherwise the user gets a safe fallback and the question is added to the Care team list automatically |
| Safety | Local emergency-symptom detection (a static symptom list plus the document's own warning signs) shows an emergency banner above the answer. It never depends on the model: with no model configured, or when the model fails, an emergency message still gets the call-911 answer and banner |

The app has four tabs: Instructions (upload, then review and confirm the extracted items), Today (daily checklist), Ask (the agent), Care team (questions the app could not answer).

## Run the website

1. Install Python 3.10+. MySQL 8+ is only needed for the production setup; for a quick local run use SQLite (below).
2. Get the code. The project lives on the `aftercare` branch, and the repository root is the project root:

   ```bash
   git clone https://github.com/renny475/EE542-Labs.git
   cd EE542-Labs
   git checkout aftercare
   ```

3. From the project root:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   cp .env.example .env
   ```

4. In `.env`, set a unique `JWT_SECRET` (startup fails if it is empty or still the placeholder). Keep `.env` private; it is gitignored.
5. Choose a database.

   Quick local run with SQLite - set this in `.env`:

   ```
   DATABASE_URL=sqlite+aiosqlite:///./aftercare-dev.sqlite3
   ```

   MySQL - edit the password in `sql/schema.sql`, run `mysql -u root -p < sql/schema.sql`, and put the same password in `DATABASE_URL`. The app also creates missing tables on startup. `schema.sql` uses `CREATE TABLE IF NOT EXISTS`, so a database created by an earlier version will not gain new columns (for example `discharge_documents.active_user_id`). Drop and recreate it.
6. Start the app from the project root. A relative SQLite path in `DATABASE_URL` resolves against the current directory; `RAG_DB_PATH` (`knowledge/index.sqlite3`) and `FOODS_PATH` (`data/foods.json`) resolve against the project root wherever you start it:

   ```bash
   uvicorn --app-dir backend app:app --host 0.0.0.0 --port 8000
   ```

7. Open `http://localhost:8000`, create an account, and log in.

## Model configuration

All settings live in `.env` and are read by `backend/config.py`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | MySQL on localhost | SQLAlchemy async URL |
| `JWT_SECRET` | none, required | Token signing secret |
| `JWT_EXPIRE_MINUTES` | 43200 | Token lifetime |
| `LLM_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | empty | Chat model name, for example `gpt-4.1-mini` (see `.env.example`) |
| `LLM_API_KEY` | empty | May stay empty for a self-hosted server |
| `EMBEDDING_API_BASE` | `LLM_API_BASE` | Embeddings endpoint base |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_API_KEY` | `LLM_API_KEY` | Embeddings key |
| `RAG_DB_PATH` | `knowledge/index.sqlite3` | Education library index |
| `RAG_MIN_SIMILARITY` | 0.25 | Minimum cosine similarity for a passage |
| `FOODS_PATH` | `data/foods.json` | Food table |

Without `LLM_MODEL` the app still starts and you can sign up, but uploading a document (extraction) and Ask return a "not configured" error (HTTP 503). The one exception is an emergency message in Ask: it still gets the call-911 answer and banner. A temporary model failure returns HTTP 502 and the page asks the patient to try again.

To use a self-hosted vLLM server, set `LLM_API_BASE` to its OpenAI-compatible URL (for example `http://host:8000/v1`) and `LLM_MODEL` to the served model name; `LLM_API_KEY` can be empty (no `Authorization` header is sent then, and `manage_knowledge.py index` only requires a key for `api.openai.com`). If you change `EMBEDDING_MODEL` or the embeddings endpoint, rebuild the index with `python backend/manage_knowledge.py index` (`status` shows whether the index matches the configured model).

## Knowledge library

`knowledge/manifest.json` lists the approved sources and `knowledge/sources/` holds one snapshot per source with its provenance. Only HTTPS sources on the allowlisted official hosts are accepted. Run from the project root:

```bash
python backend/manage_knowledge.py status    # what is indexed, and whether it matches EMBEDDING_MODEL
python backend/manage_knowledge.py collect   # fetch every manifest URL into knowledge/sources/ (network)
python backend/manage_knowledge.py index     # embed reviewed snapshots and curated summaries (paid embedding calls)
```

Collected snapshots are unreviewed; review each one and set `"reviewed": true` before indexing, as unreviewed snapshots are skipped.

The index `knowledge/index.sqlite3` is not built in this branch because no API key is configured. Without it the agent still answers from the patient's own document and the food table, but not from the education library.

## Food data

`data/foods.json` is generated by `data/build_foods.py` from USDA FoodData Central (SR Legacy items, patient-facing servings) and `data/oxalate.json`, whose category words are copied verbatim from the UChicago oxalate table. Foods missing from that table get oxalate "unknown" and the agent says it does not know rather than estimating.

```bash
FDC_API_KEY=... python data/build_foods.py [--cache DIR]      # from the FDC API
python data/build_foods.py --sr-legacy-csv DIR                # offline, from the unzipped SR Legacy CSV download
```

## Demo documents

`demo/cases/*.json` holds seven fictional family-medicine discharge cases: pneumonia, cellulitis, uti, copd, heart_failure, diabetes_htn, kidney_stones. `demo/out/` has the generated `.txt` and `.pdf` for each, marked "FICTIONAL - FOR DEMO". No real patient data is used anywhere.

Dates in the documents are relative to the day you generate them. Regenerate before a demo:

```bash
python demo/generate_demo.py --days-ago 2
```

To see the Instructions review and Today screens without a model, `scripts/seed_demo.py` creates the user `demo` / `demo-password-1` in the database named by `DATABASE_URL`, with a fictional kidney-stones document waiting for review. It is dev only and resets everything that user owns when re-run.

```bash
python scripts/seed_demo.py            # pending review
python scripts/seed_demo.py --confirm  # confirmed, checklist generated
```

## Tests

```bash
pytest
```

The unit suite uses SQLite and a scripted fake model, so it needs no network and no key. Currently 358 tests pass (`pytest -q -W error`).

## Status: not done or not verified yet

- No real-model run yet. Extraction and the agent have only been verified against a scripted fake model.
- The education index (`knowledge/index.sqlite3`) is not built, so library citations are untested end to end.
- The evaluation set and the end-to-end browser test are not written.
- The MySQL path (`sql/schema.sql`, `asyncmy`) has not been run against a real MySQL server; local runs and tests use SQLite.
- The Android TWA wrapper from the reference project is not ported.
- Out of scope for this milestone: follow-up summary generation, OCR for scanned PDFs (they return "No readable text found. Scanned PDFs need OCR first."), and languages other than English.

## Project layout

```
README.md  pytest.ini  .env.example
backend/     FastAPI app: app.py (create_app), config.py, auth.py, models.py, db.py,
             documents.py, extraction.py, checklist.py, agent.py, foods.py, safety.py,
             knowledge.py, manage_knowledge.py, llm.py, routes_*.py, requirements.txt
frontend/    index.html + login.js (login), app.html + app.css + api.js (shell),
             review.js, today.js, ask.js, careteam.js (tabs)
knowledge/   manifest.json, sources/*.json (index.sqlite3 is built locally)
data/        foods.json, oxalate.json, build_foods.py
demo/        generate_demo.py, cases/*.json, out/ (generated .txt and .pdf)
scripts/     seed_demo.py
sql/         schema.sql (MySQL)
tests/       unit tests and the fake LLM
docs/        design spec
```
