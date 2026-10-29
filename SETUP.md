# Setup (macOS, tested 2026-10-28)

1. `brew install mysql && brew services start mysql`
2. `mysql -u root < sql/schema.sql`
3. `cd backend && python -m venv .venv && source .venv/bin/activate`
4. `pip install -r requirements.txt greenlet "bcrypt<4.1"`

   The template's requirements.txt is missing `greenlet` (required by SQLAlchemy async), and `bcrypt` 5.x is incompatible with `passlib`.
5. `cp ../.env.example ../.env` and set `JWT_SECRET` (generate one with `python -c "import secrets; print(secrets.token_hex(32))"`).
6. llama.cpp: `brew install llama.cpp` (or build from source into `../llama.cpp/build/bin/`), then start the server from the `lab9` directory:

   ```bash
   ./llama.cpp/build/bin/llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
     --port 8001 --alias llama-3.2-3b-instruct -c 4096 -np 2 --api-key <key>
   ```

   The 8B model will freeze machines with <=16 GB RAM; use the 3B model.
7. In `.env`, `LLM_API_KEY` must match the `--api-key` value above and `LLM_MODEL` must match `--alias`.
8. From `backend/`: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`, then open http://localhost:8000
