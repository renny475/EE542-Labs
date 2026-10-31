# Setup (macOS, tested 2026-10-31)

1. `brew install mysql && brew services start mysql`
2. `mysql -u root < sql/schema.sql`

   Creates the database, the `chatuser` account and every table. `schema.sql` is the
   reference schema; `ingest.py` and the backend also create any missing table
   themselves. Upgrading a Part 1 database: the backend adds the missing
   `messages.sources` column at startup; run `DROP TABLE messages;` first only if
   you also want to discard the old public group-chat rows.
3. `cd backend && python -m venv .venv && source .venv/bin/activate`
4. `pip install -r requirements.txt greenlet "bcrypt<4.1"`

   `greenlet` is required by SQLAlchemy async, and `bcrypt` 5.x is incompatible with `passlib`.
5. `cp ../.env.example ../.env`, then set:
   - `JWT_SECRET` (generate one with `python -c "import secrets; print(secrets.token_hex(32))"`)
   - `OPENAI_API_KEY`, used only for `text-embedding-3-small` embeddings (ingest and the
     question embedding at query time). The answers come from the local model below.
6. Corpus (run from `backend/`, with MySQL up and `OPENAI_API_KEY` set):

   ```bash
   .venv/bin/python fetch_sources.py            # downloads the USCIS / SEVP / CFR documents
   # OIS pages block scripts: open each ois.usc.edu URL listed in data/sources.yaml in Chrome,
   # print to PDF and save it as data/ois/<slug>.pdf, or run  bash scripts/save_ois_pdfs.sh
   .venv/bin/python fetch_sources.py --check    # lists OIS files still missing
   .venv/bin/python ingest.py                   # extract, chunk, embed into MySQL (idempotent)
   .venv/bin/python eval_retrieval.py           # optional: recall@5 on data/eval_questions.yaml
   ```

   Restart uvicorn after every ingest; the backend loads the corpus into memory at startup.
7. llama.cpp: `brew install llama.cpp` (or build from source into `../llama.cpp/build/bin/`), then start the server from the `lab9` directory:

   ```bash
   ./llama.cpp/build/bin/llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
     --port 8001 --alias llama-3.2-3b-instruct -c 8192 -np 2 --api-key <key>
   ```

   `-c 8192` leaves room for five retrieved chunks plus history. The 8B model will freeze
   machines with <=16 GB RAM; use the 3B model.
8. In `.env`, `LLM_API_KEY` must match the `--api-key` value above and `LLM_MODEL` must match `--alias`.
9. From `backend/`: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`, then open http://localhost:8000

   `curl localhost:8000/api/meta` shows the document and chunk counts the server loaded.
10. Tests: `cd backend && .venv/bin/python -m pytest`
