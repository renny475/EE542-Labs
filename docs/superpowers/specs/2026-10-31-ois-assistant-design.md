# USC OIS Assistant: design (EE542 Lab 9, Part 4)

Status: approved by Kai on 2026-10-31. Android packaging is deferred (see
"Out of scope for now").

## 1. Intent

Part 4 asks for a custom-content chatbot: a web app, later wrapped as an
Android app, that answers questions on one topic from a curated knowledge
base, using embeddings plus an LLM.

Our topic is **USC international students on F-1 status**: on-campus work,
CPT, OPT, STEM OPT, cap-gap, maintaining status, program extension and
completion, travel and re-entry, F-2 dependents. These questions are asked
constantly, the answers are spread over USC OIS pages and dense federal
documents, and getting them wrong has real consequences. The assistant
answers only from official sources and shows exactly which passage each
answer came from, so a student can verify it and knows where to read more.

The web app is the existing Lab 9 group chat (FastAPI + MySQL + vanilla JS)
turned into a single-topic personal assistant. The Lab 9 Part 2 Streamlit
app (`embedding/`) stays as-is; the separate `ois_assistant` Streamlit repo
is an experiment bench and not part of the deliverable.

## 2. Goals, non-goals, anti-goals

Goals
- A logged-in student types a question and gets an answer grounded in the
  curated corpus, with numbered citations and a "Sources" list showing
  title, link, page and the quoted passage.
- When the corpus does not cover the question, the assistant says so and
  points to OIS advising instead of guessing.
- The corpus is reproducible: every file in `data/` is listed with source
  URL and fetch date, and re-running ingestion is idempotent.
- The whole thing runs on a laptop: MySQL, llama.cpp `llama-server` with
  Llama 3.2 3B Instruct, FastAPI; the only network dependency is the
  OpenAI embeddings API.
- The web UI works at phone width, because it will be shown inside a TWA.

Non-goals
- No J-1 scholars, taxes, SSN/ITIN, or dependents' employment content.
- No account roles, no admin UI, no conversation sharing.
- No multi-turn query rewriting; retrieval uses the current question only.
- No Android work in this phase.

Anti-goals (would make this a failure even if it "works")
- Answering confidently from model memory when the retrieved passages do
  not support the answer.
- Mixing in unofficial sources (forums, law-firm blogs) so that the corpus
  contradicts itself.
- Showing one user another user's questions.
- Committing `.env`, `.venv`, or any API key.

## 3. Constraints

- LLM: local `llama-server` serving `Llama-3.2-3B-Instruct` Q4_K_M at
  `http://localhost:8001/v1`, OpenAI-compatible, `--api-key` enforced.
  Decided by Kai; quality limits of a 3B model are accepted.
- Embeddings: OpenAI `text-embedding-3-small` (1536 dims), called directly
  with the `openai` Python package. Kai has a funded key. My call, for
  quality and simplicity; a fully local alternative (second `llama-server`
  with a GGUF embedding model) is possible later behind the same interface.
- Database: the existing MySQL `groupchat` database and `chatuser` account.
- Code lives in `groupchat_app_src` on branch `Lab9` of
  `renny475/EE542-Labs`, where the teammate already has access.
- Python 3.13 venv under `backend/`; pinned versions in
  `backend/requirements.txt`.
- The dev database may be reset; existing group-chat history is not kept.

## 4. Corpus

### 4.1 Scope and tiers

Three tiers, in order of authority for USC procedure vs. federal rule:

| Tier | Source | How it is fetched |
|---|---|---|
| `ois` | ois.usc.edu pages (about 25) | Saved manually from Chrome as PDF (the site blocks scripted fetches; we do not work around that) |
| `uscis` | USCIS pages and the I-765 instructions PDF | `fetch_sources.py` |
| `sevp` | ICE SEVP / Study in the States pages, Form I-983 and instructions, STEM designated degree list | `fetch_sources.py` |
| `cfr` | 8 CFR 214.2, paragraph (f) only | `fetch_sources.py`, then the ingester keeps only paragraph (f) |

OIS pages to include (all under ois.usc.edu):
- employment/: f1-student-employment, on-campus-employment,
  curricular-practical-training-cpt, optional-practical-training-opt,
  pre-completion-opt, post-completion-opt, stem-opt-extension,
  stem-opt-extension-prior-degree, cap-gap-extension,
  additional-forms-of-employment, social-security-number
- maintaining-status/: guidelines-maintaining-status, change-of-status-f1,
  extending-program-students, progressive-degree-program,
  attending-another-institution, reduced-course-load, leave-of-absence,
  program-completion-students
- travel/: entering-united-states, renewing-passport-f1-j1-visa,
  requirements-for-reentry, automatic-revalidation
- getting-started/: student-visa-types, transferring-to-usc, f2-dependents
- news: duration-status-rule-change, update-regarding-proposed-opt-fees
  (time-sensitive; keep their dates visible in metadata)

Federal sources (verified live on 2026-10-31):
- https://www.uscis.gov/sites/default/files/document/forms/i-765instr.pdf
- https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students
- https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt
- https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/students-and-employment
- https://www.uscis.gov/policy-manual/volume-2-part-f plus every
  `/policy-manual/volume-2-part-f-chapter-N` page it links to
- https://studyinthestates.dhs.gov/stem-opt-hub
- https://studyinthestates.dhs.gov/students/work/training-opportunities-in-the-united-states
- https://www.ice.gov/sevis/practical-training
- https://www.ice.gov/doclib/sevis/pdf/i983.pdf
- https://www.ice.gov/doclib/sevis/pdf/i983Instructions.pdf
- https://www.ice.gov/doclib/sevis/pdf/stemList2024.pdf
- https://www.ecfr.gov/current/title-8/chapter-I/subchapter-B/part-214/section-214.2

Expected size: about 37 documents, 400 to 800 chunks.

### 4.2 Layout and manifest

```
data/
  sources.yaml        manifest, one entry per document
  ois/                manually saved PDFs
  federal/            files written by fetch_sources.py
  eval_questions.yaml small golden set for retrieval checks
```

`sources.yaml` entry fields: `id` (stable slug, used as the DB key),
`title`, `tier`, `url`, `file` (path under data/), `kind` (`pdf` | `html`),
`fetched` (YYYY-MM-DD), `manual` (true for OIS). `fetch_sources.py`
downloads every non-manual entry that is missing or whose `--refresh` flag
is given, and writes `fetched` back. For manual entries it only checks the
file exists and reports the ones still missing.

Files in `data/` are committed. They are the curated deliverable and the
teammate needs the identical corpus. Keep the total under roughly 30 MB.

## 5. Ingestion (`backend/ingest.py`)

Run from the backend venv after `fetch_sources.py`; safe to re-run.

1. Load `sources.yaml`. For each entry, hash the file (sha256).
2. If a `documents` row with the same `source_id` and `sha256` exists, skip.
   If the hash differs, delete the row (chunks cascade) and re-ingest.
   `--prune` deletes `documents` rows whose `source_id` is no longer in the
   manifest.
3. Extract text. PDF: `pypdf`, page by page, keeping the page number.
   HTML: BeautifulSoup; drop `script`, `style`, `nav`, `header`, `footer`,
   `aside`; prefer `<main>` or `<article>` when present; emit headings and
   paragraphs as separate lines. For the eCFR page keep only the subtree of
   paragraph (f). Collapse whitespace and strip repeated page headers.
4. Chunk with `RecursiveCharacterTextSplitter(chunk_size=1000,
   chunk_overlap=200)` (characters). My call: larger than the Part 2
   setting because legal text needs more surrounding context, and five
   such chunks still fit easily in the model's window.
5. Embed in batches of up to 100 with `text-embedding-3-small`.
6. Insert `documents` and `chunks` rows in one transaction per document.
7. Print a summary: documents added / replaced / skipped, chunk counts, and
   the list of manual files still missing.

Schema additions (also added to `sql/schema.sql`):

```sql
CREATE TABLE documents (
  id INT AUTO_INCREMENT PRIMARY KEY,
  source_id VARCHAR(64) NOT NULL UNIQUE,
  title VARCHAR(255) NOT NULL,
  tier ENUM('ois','uscis','sevp','cfr') NOT NULL,
  url VARCHAR(512) NOT NULL,
  file_path VARCHAR(255) NOT NULL,
  sha256 CHAR(64) NOT NULL,
  fetched_on DATE NOT NULL,
  embedding_model VARCHAR(64) NOT NULL,
  chunk_count INT NOT NULL,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE chunks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  document_id INT NOT NULL,
  chunk_index INT NOT NULL,
  page INT NULL,
  content TEXT NOT NULL,
  embedding BLOB NOT NULL,          -- float32 little-endian, 1536 values
  UNIQUE KEY uq_doc_chunk (document_id, chunk_index),
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

MySQL is the source of truth for text and vectors; FAISS is only an
in-memory cache built at backend startup.

## 6. Retrieval and answering (`backend/rag.py`)

Startup: load all chunks, L2-normalise the vectors, build a FAISS
`IndexFlatIP` (cosine), keep a parallel list of chunk metadata. Refuse to
start if `documents.embedding_model` disagrees with `EMBEDDING_MODEL`. Log
document and chunk counts. After re-ingesting, restart the backend; there
is no hot reload.

Per question:
1. Embed the question with the same model. If the embeddings API fails,
   the bot message is an apology naming the failure; the LLM is not called
   without context.
2. Search top `RAG_TOP_K` (default 5), drop hits below `RAG_MIN_SCORE`
   (default 0.30, cosine). If nothing survives, reply with the canned
   not-covered message and no LLM call:
   "I couldn't find this in the OIS and USCIS materials I have. Please ask
   an OIS advisor: https://ois.usc.edu/contact/".
3. Build the prompt:
   - system: persona (USC OIS assistant for F-1 students), rules: answer
     only from the numbered sources; cite as [1], [2] after the sentence
     they support; if the sources do not answer, say so and point to OIS;
     prefer OIS sources for USC procedure and federal sources for the rule
     itself; be concise; never give legal advice or predict case outcomes.
   - the sources block: `[n] <title> (<tier>, p.<page>)` followed by the
     chunk text.
   - the last 4 messages of this user's thread (user and bot), each
     truncated to 500 characters, then the current question.
4. Call `llama-server` `/v1/chat/completions` with `temperature=0.2`,
   `max_tokens=500`, 120 s timeout.
5. Save the bot message with `sources` = the surviving hits as JSON
   `[{n, title, url, page, excerpt}]` (excerpt is the chunk text, at most
   600 characters) and push it to the asker's sockets.

Context budget: `llama-server` runs with `-c 8192 -np 2`, so each slot has
4096 tokens. System + 5 chunks (about 1300 tokens) + history (about 500) +
answer (500) is under 2500. SETUP.md is updated from `-c 4096`.

## 7. Backend changes (`backend/app.py`)

- Conversations become private. `messages.user_id` is the thread owner for
  both the user's and the bot's rows (`is_bot` tells them apart), so the
  FK stays `ON DELETE SET NULL`. `GET /api/messages` returns only the
  caller's rows. The old `is_bot` rows with `user_id NULL` are dropped
  with the table reset.
- `messages.sources JSON NULL` is added.
- Every user message triggers an answer; the `?` rule is removed.
- `POST /api/messages` saves and pushes the user message, then runs
  retrieval and the LLM in a FastAPI background task, returning at once.
  The bot message arrives over the WebSocket.
- The WebSocket authenticates: the client connects to `/ws?token=<jwt>`;
  invalid tokens are closed with code 4401. `ConnectionManager` keeps
  `dict[user_id, set[WebSocket]]` and `send_to(user_id, data)` replaces
  `broadcast`.
- `GET /api/meta` returns `{document_count, chunk_count, corpus_as_of
  (latest fetched_on), llm_model, embedding_model}` for the UI banner.
- The bot's display name is "OIS Assistant".
- New env: `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `RAG_TOP_K`,
  `RAG_MIN_SCORE`. `.env.example` and SETUP.md are updated.
- New dependencies: `openai`, `faiss-cpu`, `numpy`, `pypdf`,
  `beautifulsoup4`, `langchain-text-splitters`, `pyyaml`, `pymysql` (sync
  driver for ingest). Pin versions.
- `rag.py` and `ingest.py` are importable without a running server so they
  can be unit-tested.

## 8. Frontend changes (`frontend/`)

- Title and header: "USC OIS Assistant", subtitle "F-1 · CPT · OPT · STEM
  OPT". Remove the "?" hint.
- A persistent notice under the header: "Not legal advice. Answers are
  generated from OIS and USCIS documents as of <corpus_as_of>. Confirm
  anything important with an OIS advisor." `<corpus_as_of>` comes from
  `/api/meta`.
- Empty thread: a welcome card with four clickable example questions
  (e.g. "When can I apply for post-completion OPT?", "How many hours can I
  work on campus during the semester?", "Do I qualify for the STEM OPT
  extension?", "What do I need to re-enter the U.S. after travelling?").
- Bot messages: the answer text, then a collapsed "Sources (n)" section;
  each source shows `[n] title`, a link to `url`, the page if any, and the
  excerpt. Citation markers in the answer are plain text.
- A typing indicator appears when the user sends and disappears when the
  bot message arrives or after 120 s.
- Phone layout: the message list fills the viewport (`100dvh` minus header
  and composer), the composer stays at the bottom, 16 px side gutters, no
  horizontal scroll. Dark theme kept; the builder may introduce a USC
  cardinal accent for primary buttons as long as contrast stays readable.
- `connectWS` passes the token in the query string and stops reconnecting
  on close code 4401.

## 9. Verification

- Unit tests (pytest, run against SQLite where a DB is needed; ingest and
  rag use SQLAlchemy Core so this works): HTML cleaning keeps headings and
  drops navigation; eCFR extraction keeps only paragraph (f); chunker
  settings; ingest skips unchanged files and replaces changed ones;
  retrieval threshold and the not-covered path with a fake embedder; prompt
  assembly includes exactly the surviving sources.
- `backend/eval_retrieval.py` reads `data/eval_questions.yaml` (15 to 20
  questions, each with the `source_id`s that should be retrieved) and
  prints recall@5 per question and overall. Target: every question hits at
  least one expected source.
- Manual end-to-end on the laptop: sign up, ask each example question,
  confirm the sources shown match the answer, confirm a second user does
  not see the first user's thread, confirm an off-topic question gets the
  not-covered reply, and check the layout at 390 px width in Chrome
  devtools.

## 10. Out of scope for now

- Android: later, set the TWA `defaultUrl` to the final https URL and put
  the signing-key SHA-256 into `frontend/.well-known/assetlinks.json`,
  then rebuild the APK. A stable https URL (Cloudflare named tunnel with an
  owned domain, or a small VPS) is needed before that; quick tunnels change
  URL on every restart.
- Local embedding model, hot reload of the index, query rewriting for
  follow-up questions, per-answer feedback buttons.

## 11. Left to the builder

- Exact CSS, colours and spacing; the welcome card copy.
- HTML cleaning heuristics beyond those in section 5, and how repeated PDF
  page headers are detected.
- Batch sizes, retry policy for the embeddings API, logging format.
- Whether `fetch_sources.py` uses `httpx` or `requests`.
- The eval question set.
