# USC OIS Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Lab 9 group chat into a private, single-topic RAG assistant for USC F-1 students that answers only from a curated OIS/USCIS corpus and shows its sources.

**Architecture:** A manifest (`data/sources.yaml`) drives a fetch script and an idempotent ingester that extracts, chunks and embeds documents into two new MySQL tables. The FastAPI backend builds an in-memory FAISS index from those tables at startup and, for every user message, retrieves top-k chunks, prompts the local `llama-server`, and pushes the cited answer over a per-user WebSocket. The vanilla-JS frontend becomes a phone-friendly assistant UI with a disclaimer, example questions and collapsible sources.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy 2 (async `asyncmy` for the app, sync `pymysql` for ingest), MySQL, `openai` (embeddings only), `faiss-cpu`, `numpy`, `pypdf`, `beautifulsoup4`, `langchain-text-splitters`, `PyYAML`, `httpx`, pytest; llama.cpp `llama-server`; vanilla HTML/JS/CSS.

**Spec:** `docs/superpowers/specs/2026-10-31-ois-assistant-design.md`

## Global Constraints

- All code lives in `groupchat_app_src/` on branch `Lab9`; commit after every task; never commit `.env`, `.venv`, `__pycache__` or any key.
- Backend interpreter is `backend/.venv/bin/python` (Python 3.13.13). Every command below runs from `groupchat_app_src/backend/` unless stated.
- All new Python dependencies are pinned in `backend/requirements.txt` to the exact installed version (`pip freeze` style).
- Embeddings: `text-embedding-3-small`, 1536 dims, cosine similarity on L2-normalised float32 vectors stored as little-endian bytes.
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, characters.
- Retrieval defaults: `RAG_TOP_K=5`, `RAG_MIN_SCORE=0.30`. LLM call: `temperature=0.2`, `max_tokens=500`, 120 s timeout. History: last 4 messages, each truncated to 500 characters. Excerpt: at most 600 characters.
- Fetch policy: User-Agent `EE542-OIS-Assistant/0.1 (USC student course project)`, 2 s pause between requests, no retry on 403, never a browser User-Agent.
- Bot display name: `OIS Assistant`. Not-covered reply, verbatim: `I couldn't find this in the OIS and USCIS materials I have. Please ask an OIS advisor: https://ois.usc.edu/contact/`
- Conversations are private: a user sees only rows whose `messages.user_id` is theirs; bot rows carry the asker's `user_id` with `is_bot = true`.
- WebSocket auth: `/ws?token=<jwt>`; bad token closes with code 4401.
- Frontend renders model output as text only (`textContent`, `white-space: pre-wrap`); no Markdown/HTML interpretation.
- Phone layout at 390 px: no horizontal scroll, 16 px side gutters, composer pinned to the bottom.

## Review Focus

1. A PDF page with no extractable text (scanned page) must yield no chunk and must not abort ingestion of that document or the run. Test in Task 6 (`test_chunk_pages_skips_blank_pages`) and Task 8 (`test_ingest_document_with_no_text_is_recorded_with_zero_chunks`).
2. A manifest entry whose file is absent (OIS PDF not yet saved) must be reported as missing and skipped, never crash ingest. Test in Task 8 (`test_ingest_reports_missing_files`).
3. `llama-server` down or timing out must produce an apologetic bot message, with no traceback to the user and no unhandled background-task error. Test in Task 9 (`test_answer_when_llm_fails`).
4. A WebSocket opened with an expired or garbage token must be closed with 4401 and the frontend must stop reconnecting instead of hammering the server every 2 s. Test in Task 10 (`test_user_id_from_token_rejects_bad_tokens`); manual check in Task 11.
5. Starting the backend against a corpus embedded with a different model than `EMBEDDING_MODEL` must refuse to start with a clear message, not silently return garbage similarities. Test in Task 9 (`test_retriever_rejects_model_mismatch`).

---

### Task 1: Settings module and test harness

**Files:**
- Create: `backend/settings.py`
- Create: `backend/tests/__init__.py`, `backend/tests/conftest.py`
- Modify: `backend/requirements.txt`, `.env.example`, `.gitignore`
- Test: `backend/tests/test_settings.py`

**Interfaces:**
- Produces: `settings.PROJECT_DIR: Path`, `settings.DATA_DIR: Path` (= `PROJECT_DIR / "data"`), `settings.DATABASE_URL: str`, `settings.SYNC_DATABASE_URL: str`, `settings.OPENAI_API_KEY: str`, `settings.EMBEDDING_MODEL: str` (default `text-embedding-3-small`), `settings.RAG_TOP_K: int` (default 5), `settings.RAG_MIN_SCORE: float` (default 0.30), `settings.LLM_API_BASE`, `settings.LLM_MODEL`, `settings.LLM_API_KEY`, `settings.JWT_SECRET`, `settings.JWT_EXPIRE_MINUTES`, `settings.USER_AGENT: str`, and `settings.sync_url(url: str) -> str`.

- [ ] **Step 1: Install dependencies and pin them**

Run:
```bash
.venv/bin/pip install openai faiss-cpu numpy pypdf beautifulsoup4 langchain-text-splitters pyyaml pymysql pytest
.venv/bin/pip freeze > requirements.txt
```
Expected: `requirements.txt` lists every package with `==` versions, including the pre-existing fastapi, uvicorn, sqlalchemy, asyncmy, httpx, python-jose, passlib, python-dotenv entries.

- [ ] **Step 2: Write the failing test**

`backend/tests/conftest.py` inserts `Path(__file__).resolve().parents[1]` at the front of `sys.path`.

```python
# backend/tests/test_settings.py
from settings import sync_url

def test_sync_url_swaps_asyncmy_for_pymysql():
    assert sync_url("mysql+asyncmy://u:p@h:3306/db") == "mysql+pymysql://u:p@h:3306/db"

def test_sync_url_leaves_sync_urls_alone():
    assert sync_url("sqlite:///x.db") == "sqlite:///x.db"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'settings'`

- [ ] **Step 4: Implement `backend/settings.py`**

Loads `PROJECT_DIR / ".env"` with `load_dotenv`, exposes the names in Interfaces with the defaults in Global Constraints. `sync_url` replaces `+asyncmy` with `+pymysql`. Keep the existing `JWT_SECRET` placeholder check (raise `RuntimeError`) here so `app.py` can drop it later.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`
Expected: 2 passed

- [ ] **Step 6: Update `.env.example` and `.gitignore`**

Append to `.env.example`:
```
# --- OIS Assistant (RAG) ---
# OpenAI key, used only for embeddings (text-embedding-3-small)
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
RAG_TOP_K=5
RAG_MIN_SCORE=0.30
```
Change `LLM_MODEL=llama-3-8b-instruct` to `LLM_MODEL=llama-3.2-3b-instruct`. Add `.pytest_cache/` to `.gitignore`.

- [ ] **Step 7: Commit**

```bash
git add backend/settings.py backend/tests backend/requirements.txt ../.env.example ../.gitignore
git commit -m "Part 4: settings module, pinned dependencies, pytest harness"
```

---

### Task 2: Manifest model and the real `data/sources.yaml`

**Files:**
- Create: `backend/manifest.py`, `data/sources.yaml`, `data/ois/.gitkeep`, `data/federal/.gitkeep`
- Test: `backend/tests/test_manifest.py`

**Interfaces:**
- Produces:
  ```python
  TIERS = ("ois", "uscis", "sevp", "cfr"); KINDS = ("pdf", "html")
  @dataclass
  class Source:
      id: str; title: str; tier: str; url: str; file: str; kind: str
      manual: bool = False
      fetched: date | None = None
      published: date | None = None
      extractor: str | None = None      # only value used: "cfr_paragraph_f"
      def path(self, data_dir: Path) -> Path   # data_dir.parent / self.file
  class ManifestError(ValueError)
  def load_manifest(path: Path) -> list[Source]
  def save_manifest(path: Path, sources: list[Source]) -> None
  def slug_from_url(url: str) -> str
  ```
  `file` is relative to the repo root (for example `data/ois/x.pdf`), so `path(data_dir)` resolves against `data_dir.parent`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_manifest.py
import pytest
from datetime import date
from manifest import Source, ManifestError, load_manifest, save_manifest, slug_from_url

def test_roundtrip(tmp_path):
    p = tmp_path / "sources.yaml"
    src = [Source(id="a", title="A", tier="ois", url="https://x/a/", file="data/ois/a.pdf",
                  kind="pdf", manual=True, fetched=date(2026, 10, 31), published=None)]
    save_manifest(p, src)
    assert load_manifest(p) == src

def test_duplicate_id_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("- {id: a, title: A, tier: ois, url: u, file: data/ois/a.pdf, kind: pdf}\n"
                 "- {id: a, title: B, tier: ois, url: v, file: data/ois/b.pdf, kind: pdf}\n")
    with pytest.raises(ManifestError):
        load_manifest(p)

def test_bad_tier_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("- {id: a, title: A, tier: blog, url: u, file: data/ois/a.pdf, kind: pdf}\n")
    with pytest.raises(ManifestError):
        load_manifest(p)

def test_slug_from_url():
    assert slug_from_url("https://ois.usc.edu/employment/optional-practical-training-opt/") == "optional-practical-training-opt"
    assert slug_from_url("https://www.ice.gov/doclib/sevis/pdf/i983.pdf") == "i983.pdf"

def test_real_manifest_loads_and_files_match_slugs():
    from settings import DATA_DIR
    sources = load_manifest(DATA_DIR / "sources.yaml")
    assert len(sources) >= 45
    for s in sources:
        if s.tier == "ois":
            assert s.manual and s.file == f"data/ois/{slug_from_url(s.url)}.pdf"
        else:
            assert not s.manual and s.file.startswith("data/federal/")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manifest'`

- [ ] **Step 3: Implement `backend/manifest.py`**

PyYAML `safe_load`/`safe_dump`; dates serialised as ISO strings; omit `None`/default fields on save; validate `tier in TIERS`, `kind in KINDS`, unique ids, `file` starts with `data/`.

- [ ] **Step 4: Write `data/sources.yaml`**

OIS entries are `manual: true`, `kind: pdf`, `file: data/ois/<slug>.pdf`, id `ois-<slug>`. Federal entries are `kind: pdf` or `html`, `file: data/federal/<id>.<kind>`. Exact content:

```yaml
# OIS (manual: save each URL from Chrome as PDF under the given file name)
- {id: ois-f1-student-employment, title: F-1 Student Employment, tier: ois, url: https://ois.usc.edu/employment/f1-student-employment/, file: data/ois/f1-student-employment.pdf, kind: pdf, manual: true}
- {id: ois-on-campus-employment, title: On-Campus Employment, tier: ois, url: https://ois.usc.edu/employment/on-campus-employment/, file: data/ois/on-campus-employment.pdf, kind: pdf, manual: true}
- {id: ois-cpt, title: Curricular Practical Training (CPT), tier: ois, url: https://ois.usc.edu/employment/curricular-practical-training-cpt/, file: data/ois/curricular-practical-training-cpt.pdf, kind: pdf, manual: true}
- {id: ois-opt, title: Optional Practical Training (OPT), tier: ois, url: https://ois.usc.edu/employment/optional-practical-training-opt/, file: data/ois/optional-practical-training-opt.pdf, kind: pdf, manual: true}
- {id: ois-pre-completion-opt, title: Pre-Completion OPT, tier: ois, url: https://ois.usc.edu/employment/pre-completion-opt/, file: data/ois/pre-completion-opt.pdf, kind: pdf, manual: true}
- {id: ois-post-completion-opt, title: Post-Completion OPT, tier: ois, url: https://ois.usc.edu/employment/post-completion-opt/, file: data/ois/post-completion-opt.pdf, kind: pdf, manual: true}
- {id: ois-stem-opt-extension, title: STEM OPT Extension, tier: ois, url: https://ois.usc.edu/employment/stem-opt-extension/, file: data/ois/stem-opt-extension.pdf, kind: pdf, manual: true}
- {id: ois-stem-opt-prior-degree, title: STEM OPT Extension Based on a Prior Degree, tier: ois, url: https://ois.usc.edu/employment/stem-opt-extension-prior-degree/, file: data/ois/stem-opt-extension-prior-degree.pdf, kind: pdf, manual: true}
- {id: ois-cap-gap-extension, title: Cap-Gap Extension, tier: ois, url: https://ois.usc.edu/employment/cap-gap-extension/, file: data/ois/cap-gap-extension.pdf, kind: pdf, manual: true}
- {id: ois-additional-forms-of-employment, title: Additional Forms of Employment, tier: ois, url: https://ois.usc.edu/employment/additional-forms-of-employment/, file: data/ois/additional-forms-of-employment.pdf, kind: pdf, manual: true}
- {id: ois-social-security-number, title: Social Security Number, tier: ois, url: https://ois.usc.edu/employment/social-security-number/, file: data/ois/social-security-number.pdf, kind: pdf, manual: true}
- {id: ois-guidelines-maintaining-status, title: Guidelines for Maintaining Status, tier: ois, url: https://ois.usc.edu/maintaining-status/guidelines-maintaining-status/, file: data/ois/guidelines-maintaining-status.pdf, kind: pdf, manual: true}
- {id: ois-change-of-status-f1, title: Change of Status to F-1, tier: ois, url: https://ois.usc.edu/maintaining-status/change-of-status-f1/, file: data/ois/change-of-status-f1.pdf, kind: pdf, manual: true}
- {id: ois-extending-program, title: Extending Your Program (Students), tier: ois, url: https://ois.usc.edu/maintaining-status/extending-program-students/, file: data/ois/extending-program-students.pdf, kind: pdf, manual: true}
- {id: ois-progressive-degree, title: Adding a Progressive Degree Program, tier: ois, url: https://ois.usc.edu/maintaining-status/progressive-degree-program/, file: data/ois/progressive-degree-program.pdf, kind: pdf, manual: true}
- {id: ois-attending-another-institution, title: Attending Another Institution, tier: ois, url: https://ois.usc.edu/maintaining-status/attending-another-institution/, file: data/ois/attending-another-institution.pdf, kind: pdf, manual: true}
- {id: ois-reduced-course-load, title: Reduced Course Load, tier: ois, url: https://ois.usc.edu/maintaining-status/reduced-course-load/, file: data/ois/reduced-course-load.pdf, kind: pdf, manual: true}
- {id: ois-leave-of-absence, title: Leave of Absence, tier: ois, url: https://ois.usc.edu/maintaining-status/leave-of-absence/, file: data/ois/leave-of-absence.pdf, kind: pdf, manual: true}
- {id: ois-program-completion, title: Program Completion (Students), tier: ois, url: https://ois.usc.edu/maintaining-status/program-completion-students/, file: data/ois/program-completion-students.pdf, kind: pdf, manual: true}
- {id: ois-entering-united-states, title: Entering the United States, tier: ois, url: https://ois.usc.edu/travel/entering-united-states/, file: data/ois/entering-united-states.pdf, kind: pdf, manual: true}
- {id: ois-renewing-passport-visa, title: Renewing Passport or F/J Visa, tier: ois, url: https://ois.usc.edu/travel/renewing-passport-f1-j1-visa/, file: data/ois/renewing-passport-f1-j1-visa.pdf, kind: pdf, manual: true}
- {id: ois-requirements-for-reentry, title: Requirements for Reentry, tier: ois, url: https://ois.usc.edu/travel/requirements-for-reentry/, file: data/ois/requirements-for-reentry.pdf, kind: pdf, manual: true}
- {id: ois-automatic-revalidation, title: Automatic Revalidation, tier: ois, url: https://ois.usc.edu/travel/automatic-revalidation/, file: data/ois/automatic-revalidation.pdf, kind: pdf, manual: true}
- {id: ois-student-visa-types, title: Student Visa Types, tier: ois, url: https://ois.usc.edu/getting-started/student-visa-types/, file: data/ois/student-visa-types.pdf, kind: pdf, manual: true}
- {id: ois-transferring-to-usc, title: Transferring to USC, tier: ois, url: https://ois.usc.edu/getting-started/transferring-to-usc/, file: data/ois/transferring-to-usc.pdf, kind: pdf, manual: true}
- {id: ois-f2-dependents, title: F-2 Dependents, tier: ois, url: https://ois.usc.edu/getting-started/f2-dependents/, file: data/ois/f2-dependents.pdf, kind: pdf, manual: true}
- {id: ois-news-duration-of-status-rule-change, title: Duration of Status Rule Change (OIS news), tier: ois, url: https://ois.usc.edu/duration-status-rule-change/, file: data/ois/duration-status-rule-change.pdf, kind: pdf, manual: true, published: 2026-09-14}
- {id: ois-news-duration-of-status-rule-paused, title: Duration of Status Final Rule Paused (OIS news), tier: ois, url: https://ois.usc.edu/dhs-final-rule-postponed/, file: data/ois/dhs-final-rule-postponed.pdf, kind: pdf, manual: true, published: 2026-09-14}
- {id: ois-news-proposed-opt-fees, title: Update Regarding Proposed OPT Fees (OIS news), tier: ois, url: https://ois.usc.edu/update-regarding-proposed-opt-fees/, file: data/ois/update-regarding-proposed-opt-fees.pdf, kind: pdf, manual: true, published: 2026-10-07}
# USCIS
- {id: uscis-i-765-instructions, title: Instructions for Form I-765, Application for Employment Authorization, tier: uscis, url: https://www.uscis.gov/sites/default/files/document/forms/i-765instr.pdf, file: data/federal/uscis-i-765-instructions.pdf, kind: pdf}
- {id: uscis-opt-for-f1-students, title: Optional Practical Training (OPT) for F-1 Students, tier: uscis, url: https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students, file: data/federal/uscis-opt-for-f1-students.html, kind: html}
- {id: uscis-stem-opt, title: Optional Practical Training Extension for STEM Students (STEM OPT), tier: uscis, url: https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt, file: data/federal/uscis-stem-opt.html, kind: html}
- {id: uscis-students-and-employment, title: Students and Employment, tier: uscis, url: https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/students-and-employment, file: data/federal/uscis-students-and-employment.html, kind: html}
- {id: uscis-pm-2f-ch1, title: "USCIS Policy Manual Vol. 2 Part F Ch. 1: Purpose and Background", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-1, file: data/federal/uscis-pm-2f-ch1.html, kind: html}
- {id: uscis-pm-2f-ch2, title: "USCIS Policy Manual Vol. 2 Part F Ch. 2: Eligibility Requirements", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-2, file: data/federal/uscis-pm-2f-ch2.html, kind: html}
- {id: uscis-pm-2f-ch3, title: "USCIS Policy Manual Vol. 2 Part F Ch. 3: Courses and Enrollment, Full Course of Study, and Reduced Course Load", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-3, file: data/federal/uscis-pm-2f-ch3.html, kind: html}
- {id: uscis-pm-2f-ch4, title: "USCIS Policy Manual Vol. 2 Part F Ch. 4: School Transfer", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-4, file: data/federal/uscis-pm-2f-ch4.html, kind: html}
- {id: uscis-pm-2f-ch5, title: "USCIS Policy Manual Vol. 2 Part F Ch. 5: Practical Training", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-5, file: data/federal/uscis-pm-2f-ch5.html, kind: html}
- {id: uscis-pm-2f-ch6, title: "USCIS Policy Manual Vol. 2 Part F Ch. 6: Employment", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-6, file: data/federal/uscis-pm-2f-ch6.html, kind: html}
- {id: uscis-pm-2f-ch7, title: "USCIS Policy Manual Vol. 2 Part F Ch. 7: Absences From the United States", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-7, file: data/federal/uscis-pm-2f-ch7.html, kind: html}
- {id: uscis-pm-2f-ch8, title: "USCIS Policy Manual Vol. 2 Part F Ch. 8: Change of Status, Extension of Stay, and Length of Stay", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-8, file: data/federal/uscis-pm-2f-ch8.html, kind: html}
- {id: uscis-pm-2f-ch9, title: "USCIS Policy Manual Vol. 2 Part F Ch. 9: Dependents", tier: uscis, url: https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-9, file: data/federal/uscis-pm-2f-ch9.html, kind: html}
# ICE SEVP / Study in the States
- {id: sevp-stem-opt-hub, title: STEM OPT Hub (Study in the States), tier: sevp, url: https://studyinthestates.dhs.gov/stem-opt-hub, file: data/federal/sevp-stem-opt-hub.html, kind: html}
- {id: sevp-training-opportunities, title: Training Opportunities in the United States (Study in the States), tier: sevp, url: https://studyinthestates.dhs.gov/students/work/training-opportunities-in-the-united-states, file: data/federal/sevp-training-opportunities.html, kind: html}
- {id: sevp-practical-training, title: Practical Training (ICE SEVP), tier: sevp, url: https://www.ice.gov/sevis/practical-training, file: data/federal/sevp-practical-training.html, kind: html}
- {id: sevp-form-i-983, title: Form I-983, Training Plan for STEM OPT Students, tier: sevp, url: https://www.ice.gov/doclib/sevis/pdf/i983.pdf, file: data/federal/sevp-form-i-983.pdf, kind: pdf}
- {id: sevp-form-i-983-instructions, title: Instructions for Form I-983, tier: sevp, url: https://www.ice.gov/doclib/sevis/pdf/i983Instructions.pdf, file: data/federal/sevp-form-i-983-instructions.pdf, kind: pdf}
- {id: sevp-stem-designated-degree-list, title: DHS STEM Designated Degree Program List (2024), tier: sevp, url: https://www.ice.gov/doclib/sevis/pdf/stemList2024.pdf, file: data/federal/sevp-stem-designated-degree-list.pdf, kind: pdf}
# Regulation
- {id: cfr-8-214-2-f, title: "8 CFR 214.2(f): Students in colleges, universities, seminaries, conservatories, academic high schools, elementary schools, other academic institutions, and in language training programs", tier: cfr, url: https://www.govinfo.gov/content/pkg/CFR-2025-title8-vol1/pdf/CFR-2025-title8-vol1-sec214-2.pdf, file: data/federal/cfr-8-214-2.pdf, kind: pdf, extractor: cfr_paragraph_f}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add backend/manifest.py backend/tests/test_manifest.py ../data
git commit -m "Part 4: source manifest model and curated sources.yaml"
```

---

### Task 3: `fetch_sources.py`

**Files:**
- Create: `backend/fetch_sources.py`
- Test: `backend/tests/test_fetch_sources.py`

**Interfaces:**
- Consumes: `manifest.load_manifest/save_manifest/Source`, `settings.USER_AGENT`.
- Produces:
  ```python
  @dataclass
  class FetchReport:
      downloaded: list[str]; skipped: list[str]; missing: list[str]; failed: list[tuple[str, str]]
  def fetch_sources(manifest_path: Path, data_dir: Path, client: httpx.Client, *,
                    refresh: bool = False, check_only: bool = False,
                    today: date | None = None, sleep: Callable[[float], None] = time.sleep) -> FetchReport
  def main() -> int   # argparse: --refresh, --check; prints the report; exit 1 if failed or missing
  ```
  Behaviour: non-manual entry with missing file (or `refresh`) is fetched with `GET`, header `User-Agent: settings.USER_AGENT`, `follow_redirects=True`; status 200 writes bytes and sets `fetched = today`; any other status goes to `failed` with no retry. `sleep(2.0)` after every request. Manual entries are never fetched; a present manual file with `fetched is None` gets `fetched` from the file's mtime date. Any entry whose file is absent after the run goes to `missing`. The manifest is saved at the end (also in `check_only` mode, since it may fill `fetched`).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_fetch_sources.py
from datetime import date
import httpx
from manifest import load_manifest
from fetch_sources import fetch_sources

MANIFEST = """\
- {id: ois-a, title: A, tier: ois, url: https://ois.usc.edu/a/, file: data/ois/a.pdf, kind: pdf, manual: true}
- {id: uscis-b, title: B, tier: uscis, url: https://www.uscis.gov/b, file: data/federal/uscis-b.html, kind: html}
- {id: sevp-c, title: C, tier: sevp, url: https://www.ice.gov/c.pdf, file: data/federal/sevp-c.pdf, kind: pdf}
"""

def make(tmp_path, handler):
    (tmp_path / "data" / "ois").mkdir(parents=True)
    (tmp_path / "data" / "federal").mkdir()
    m = tmp_path / "data" / "sources.yaml"
    m.write_text(MANIFEST)
    seen = []
    def h(request):
        seen.append(request)
        return handler(request)
    return m, tmp_path / "data", httpx.Client(transport=httpx.MockTransport(h)), seen

def test_downloads_missing_federal_files_and_reports_missing_manual(tmp_path):
    def handler(req):
        return httpx.Response(200, content=b"<html>B</html>" if req.url.path == "/b" else b"%PDF-1.4 c")
    m, data_dir, client, seen = make(tmp_path, handler)
    sleeps = []
    rep = fetch_sources(m, data_dir, client, today=date(2026, 10, 31), sleep=sleeps.append)
    assert rep.downloaded == ["uscis-b", "sevp-c"] and rep.missing == ["ois-a"] and rep.failed == []
    assert (data_dir / "federal" / "uscis-b.html").read_bytes() == b"<html>B</html>"
    assert all(r.headers["User-Agent"].startswith("EE542-OIS-Assistant/") for r in seen)
    assert sleeps == [2.0, 2.0]
    assert {s.id: s.fetched for s in load_manifest(m)}["uscis-b"] == date(2026, 10, 31)

def test_present_files_are_skipped_unless_refresh(tmp_path):
    m, data_dir, client, seen = make(tmp_path, lambda r: httpx.Response(200, content=b"x"))
    (data_dir / "federal" / "uscis-b.html").write_bytes(b"old")
    rep = fetch_sources(m, data_dir, client, sleep=lambda s: None)
    assert "uscis-b" in rep.skipped and len(seen) == 1
    rep = fetch_sources(m, data_dir, client, refresh=True, sleep=lambda s: None)
    assert "uscis-b" in rep.downloaded

def test_403_is_failed_with_no_retry(tmp_path):
    m, data_dir, client, seen = make(tmp_path, lambda r: httpx.Response(403))
    rep = fetch_sources(m, data_dir, client, sleep=lambda s: None)
    assert [f[0] for f in rep.failed] == ["uscis-b", "sevp-c"] and len(seen) == 2

def test_check_only_fills_fetched_for_present_manual_file(tmp_path):
    m, data_dir, client, seen = make(tmp_path, lambda r: httpx.Response(200, content=b"x"))
    (data_dir / "ois" / "a.pdf").write_bytes(b"%PDF")
    rep = fetch_sources(m, data_dir, client, check_only=True, sleep=lambda s: None)
    assert seen == [] and "ois-a" not in rep.missing
    assert {s.id: s.fetched for s in load_manifest(m)}["ois-a"] == date.today()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fetch_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fetch_sources'`

- [ ] **Step 3: Implement `backend/fetch_sources.py`**

As specified in Interfaces. `main()` builds `httpx.Client(timeout=60)` and uses `settings.DATA_DIR`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fetch_sources.py -v`
Expected: 4 passed

- [ ] **Step 5: Fetch the real federal corpus**

Run: `.venv/bin/python fetch_sources.py`
Expected: report shows 20 downloaded, 0 failed, 29 missing (all `ois-*`). `ls -la ../data/federal` shows 20 files; total under 10 MB.

- [ ] **Step 6: Commit**

```bash
git add backend/fetch_sources.py backend/tests/test_fetch_sources.py ../data
git commit -m "Part 4: fetch_sources.py and fetched federal documents"
```

---

### Task 4: Text extraction

**Files:**
- Create: `backend/extract.py`
- Test: `backend/tests/test_extract.py`

**Interfaces:**
- Consumes: `manifest.Source`.
- Produces:
  ```python
  Page = tuple[int | None, str]            # (1-based page number or None for HTML, text)
  class ExtractError(ValueError)
  def pdf_pages(path: Path) -> list[Page]  # one entry per page, "" when no text
  def html_to_text(html: str) -> str
  def strip_repeated_lines(pages: list[Page], min_fraction: float = 0.5) -> list[Page]
  def cfr_paragraph_f(text: str) -> str
  def extract_source(source: Source, data_dir: Path) -> list[Page]
  ```
  `html_to_text`: BeautifulSoup `html.parser`; remove `script, style, nav, header, footer, aside, form, noscript`; root is first `<main>`, else `<article>`, else `<body>`; emit `h1..h4`, `p`, `li` as one line each, table rows as cells joined by ` | `; collapse runs of blank lines to one.
  `strip_repeated_lines`: only when `len(pages) >= 3`; a line is dropped when its digit-normalised form (every digit replaced by `#`) occurs on at least `min_fraction` of the pages.
  `cfr_paragraph_f`: slice from the first line that starts with `(f) Students in colleges` to the first later line matching `^\(g\) ` (exclusive); `ExtractError` if either marker is absent.
  `extract_source`: `pdf` → `strip_repeated_lines(pdf_pages(...))`; `html` → `[(None, html_to_text(...))]`; when `source.extractor == "cfr_paragraph_f"`, join all page texts with `\n`, apply `cfr_paragraph_f`, return `[(None, text)]`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_extract.py
import pytest
from pathlib import Path
from extract import html_to_text, strip_repeated_lines, cfr_paragraph_f, pdf_pages, ExtractError

HTML = """<html><body><nav><a>Home</a></nav><header>USCIS</header>
<main><h1>OPT</h1><p>Students may apply.</p><ul><li>Pre-completion</li><li>Post-completion</li></ul>
<table><tr><td>Type</td><td>Months</td></tr><tr><td>OPT</td><td>12</td></tr></table></main>
<footer>Privacy</footer><script>x()</script></body></html>"""

def test_html_to_text_keeps_main_content_and_drops_chrome():
    text = html_to_text(HTML)
    assert text.splitlines() == ["OPT", "Students may apply.", "Pre-completion", "Post-completion",
                                 "Type | Months", "OPT | 12"]

def test_strip_repeated_lines_removes_chrome_print_headers():
    pages = [(i, f"10/31/26, 2:1{i} PM OPT - OIS\nBody {i}\nhttps://ois.usc.edu/x/ {i}/3") for i in (1, 2, 3)]
    assert strip_repeated_lines(pages) == [(1, "Body 1"), (2, "Body 2"), (3, "Body 3")]

def test_strip_repeated_lines_leaves_short_documents_alone():
    pages = [(1, "Same\nA"), (2, "Same\nB")]
    assert strip_repeated_lines(pages) == pages

def test_cfr_paragraph_f_slices_between_markers():
    text = "(e) Treaty traders\nblah\n(f) Students in colleges, universities\n(1) Admission\n(2) Stuff\n(g) Representatives\nmore"
    assert cfr_paragraph_f(text) == "(f) Students in colleges, universities\n(1) Admission\n(2) Stuff"

def test_cfr_paragraph_f_requires_markers():
    with pytest.raises(ExtractError):
        cfr_paragraph_f("(f) Students in colleges\nno g")

I983 = Path(__file__).resolve().parents[2] / "data" / "federal" / "sevp-form-i-983-instructions.pdf"

@pytest.mark.skipif(not I983.exists(), reason="federal corpus not fetched")
def test_pdf_pages_reads_real_pdf():
    pages = pdf_pages(I983)
    assert pages[0][0] == 1 and len(pages) >= 2
    assert "STEM" in "\n".join(t for _, t in pages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extract'`

- [ ] **Step 3: Implement `backend/extract.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_extract.py -v`
Expected: 6 passed

- [ ] **Step 5: Verify the real CFR extraction works**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path; from manifest import load_manifest; from extract import extract_source; from settings import DATA_DIR
s=[x for x in load_manifest(DATA_DIR/'sources.yaml') if x.id=='cfr-8-214-2-f'][0]
t=extract_source(s, DATA_DIR)[0][1]; print(len(t)); print(t[:200]); print('...', t[-150:])"
```
Expected: length between 150,000 and 400,000 characters; starts with `(f) Students in colleges`; the tail does not contain `(g) Representatives`. If the PDF's text has the `(f)` heading on the same line as other text, adjust the marker regex in `cfr_paragraph_f` and add the real line shape to the test.

- [ ] **Step 6: Commit**

```bash
git add backend/extract.py backend/tests/test_extract.py
git commit -m "Part 4: PDF/HTML text extraction and 8 CFR 214.2(f) slicing"
```

---

### Task 5: Embedders and vector serialisation

**Files:**
- Create: `backend/embeddings.py`
- Test: `backend/tests/test_embeddings.py`

**Interfaces:**
- Produces:
  ```python
  class Embedder(Protocol):
      model: str
      dim: int
      def embed(self, texts: list[str]) -> list[list[float]]: ...
  class OpenAIEmbedder:      # __init__(self, model: str, api_key: str, batch_size: int = 100)
      ...                    # uses openai.OpenAI(api_key=..., max_retries=3); dim = 1536
  class FakeEmbedder:        # __init__(self, dim: int = 8, model: str = "fake")
      ...                    # deterministic: sha256(text) bytes -> floats, L2-normalised
  def to_blob(vec: Sequence[float]) -> bytes            # float32 little-endian
  def from_blob(blob: bytes) -> np.ndarray              # dtype float32
  def normalise(vectors: np.ndarray) -> np.ndarray      # rows L2-normalised, zero rows left as zero
  ```

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_embeddings.py
import struct
import numpy as np
from embeddings import FakeEmbedder, to_blob, from_blob, normalise

def test_fake_embedder_is_deterministic_and_unit_length():
    e = FakeEmbedder(dim=8)
    a, b = e.embed(["opt", "opt"]); c = e.embed(["cpt"])[0]
    assert a == b and a != c and len(a) == 8
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6

def test_blob_roundtrip_is_float32_little_endian():
    blob = to_blob([1.0, -2.5, 0.0])
    assert len(blob) == 12 and blob[:4] == struct.pack("<f", 1.0)
    assert from_blob(blob).tolist() == [1.0, -2.5, 0.0]

def test_normalise_rows():
    out = normalise(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(out[0], [0.6, 0.8]) and np.allclose(out[1], [0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'embeddings'`

- [ ] **Step 3: Implement `backend/embeddings.py`**

`OpenAIEmbedder.embed` slices `texts` into `batch_size` batches and calls `client.embeddings.create(model=..., input=batch)`, preserving order.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_embeddings.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/embeddings.py backend/tests/test_embeddings.py
git commit -m "Part 4: embedder interface, OpenAI and fake embedders, vector blobs"
```

---

### Task 6: Chunking

**Files:**
- Create: `backend/chunking.py`
- Test: `backend/tests/test_chunking.py`

**Interfaces:**
- Consumes: `extract.Page`.
- Produces:
  ```python
  @dataclass
  class ChunkText:
      index: int; page: int | None; content: str
  def chunk_pages(pages: list[Page], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[ChunkText]
  ```
  Join page texts with `"\n"`, remembering each page's start offset; split with `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)`; locate each chunk in the joined text by searching forward from the previous chunk's start; `page` is the page whose span contains the chunk's start offset. Blank pages (whitespace only) contribute nothing. Chunks that are whitespace only are dropped; `index` is sequential from 0 after dropping.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_chunking.py
from chunking import chunk_pages

def test_empty_input_gives_no_chunks():
    assert chunk_pages([]) == []
    assert chunk_pages([(1, ""), (2, "   \n")]) == []

def test_chunk_pages_skips_blank_pages_and_attributes_pages():
    p1 = "\n".join(f"Page one line {i}." for i in range(40))   # about 680 chars
    p3 = "\n".join(f"Page three line {i}." for i in range(40))
    chunks = chunk_pages([(1, p1), (2, ""), (3, p3)], chunk_size=300, chunk_overlap=50)
    assert chunks[0].page == 1 and chunks[0].index == 0
    assert chunks[-1].page == 3
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.content) <= 300 and c.content.strip() for c in chunks)

def test_consecutive_chunks_overlap():
    text = "\n".join(f"Sentence number {i} about OPT rules." for i in range(100))
    chunks = chunk_pages([(None, text)], chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 5
    tail = chunks[0].content[-60:]
    assert tail.splitlines()[-1] in chunks[1].content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chunking'`

- [ ] **Step 3: Implement `backend/chunking.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chunking.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/chunking.py backend/tests/test_chunking.py
git commit -m "Part 4: page-aware recursive chunking"
```

---

### Task 7: Corpus tables and store (`corpus_db.py`) plus `sql/schema.sql`

**Files:**
- Create: `backend/corpus_db.py`
- Modify: `sql/schema.sql`
- Test: `backend/tests/test_corpus_db.py`

**Interfaces:**
- Consumes: `embeddings.to_blob/from_blob`.
- Produces (SQLAlchemy Core, sync engine):
  ```python
  metadata = MetaData()
  documents = Table("documents", metadata,
      Column("id", Integer, primary_key=True),
      Column("source_id", String(64), nullable=False, unique=True),
      Column("title", String(255), nullable=False),
      Column("tier", String(8), nullable=False),
      Column("url", String(512), nullable=False),
      Column("file_path", String(255), nullable=False),
      Column("sha256", String(64), nullable=False),
      Column("fetched_on", Date, nullable=False),
      Column("published", Date, nullable=True),
      Column("embedding_model", String(64), nullable=False),
      Column("chunk_count", Integer, nullable=False),
      Column("ingested_at", DateTime, server_default=func.now()))
  chunks = Table("chunks", metadata,
      Column("id", Integer, primary_key=True),
      Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
      Column("chunk_index", Integer, nullable=False),
      Column("page", Integer, nullable=True),
      Column("content", Text, nullable=False),
      Column("embedding", LargeBinary, nullable=False),
      UniqueConstraint("document_id", "chunk_index"))

  @dataclass
  class DocumentRecord: source_id: str; title: str; tier: str; url: str; file_path: str; sha256: str; fetched_on: date; embedding_model: str; published: date | None = None
  @dataclass
  class ChunkRecord: chunk_index: int; page: int | None; content: str; embedding: bytes
  @dataclass
  class ChunkMeta: chunk_id: int; document_id: int; title: str; tier: str; url: str; page: int | None; published: date | None; content: str
  @dataclass
  class CorpusSnapshot: chunks: list[ChunkMeta]; vectors: np.ndarray; embedding_models: set[str]; document_count: int; corpus_as_of: date | None

  def make_engine(url: str) -> Engine            # sqlite URLs get StaticPool + check_same_thread=False
  def create_tables(engine: Engine) -> None
  def existing_sha(conn: Connection, source_id: str) -> str | None
  def replace_document(conn: Connection, doc: DocumentRecord, chunk_records: list[ChunkRecord]) -> int
  def prune_documents(conn: Connection, keep: set[str]) -> list[str]
  def load_snapshot(engine: Engine) -> CorpusSnapshot   # vectors shape (n, dim) float32, (0, 0) when empty
  ```
  `replace_document` deletes any existing row with the same `source_id` (explicitly deleting its chunks first, so SQLite without FK enforcement behaves like MySQL), inserts the document and its chunks, and returns the new document id. The caller owns the transaction (`with engine.begin() as conn`).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_corpus_db.py
from datetime import date
import numpy as np
import pytest
from corpus_db import (make_engine, create_tables, existing_sha, replace_document, prune_documents,
                       load_snapshot, DocumentRecord, ChunkRecord)
from embeddings import to_blob

@pytest.fixture
def engine():
    e = make_engine("sqlite://"); create_tables(e); return e

def doc(sid="a", sha="1"*64, fetched=date(2026, 10, 31)):
    return DocumentRecord(source_id=sid, title="T", tier="ois", url="u", file_path="data/ois/a.pdf",
                          sha256=sha, fetched_on=fetched, embedding_model="fake")

def test_replace_document_inserts_and_replaces(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, 1, "one", to_blob([1, 0])), ChunkRecord(1, 2, "two", to_blob([0, 1]))])
        assert existing_sha(c, "a") == "1"*64
        replace_document(c, doc(sha="2"*64), [ChunkRecord(0, None, "new", to_blob([1, 1]))])
    snap = load_snapshot(engine)
    assert [c.content for c in snap.chunks] == ["new"] and snap.document_count == 1
    assert snap.vectors.shape == (1, 2) and snap.vectors.dtype == np.float32

def test_prune_removes_documents_not_kept(engine):
    with engine.begin() as c:
        replace_document(c, doc("a"), [ChunkRecord(0, None, "x", to_blob([1, 0]))])
        replace_document(c, doc("b"), [ChunkRecord(0, None, "y", to_blob([0, 1]))])
        assert prune_documents(c, keep={"a"}) == ["b"]
    assert [m.title for m in load_snapshot(engine).chunks] == ["T"] and load_snapshot(engine).document_count == 1

def test_snapshot_metadata(engine):
    with engine.begin() as c:
        replace_document(c, doc("a", fetched=date(2026, 10, 1)), [ChunkRecord(0, None, "x", to_blob([1, 0]))])
        replace_document(c, doc("b", fetched=date(2026, 10, 31)), [])
    snap = load_snapshot(engine)
    assert snap.corpus_as_of == date(2026, 10, 31) and snap.embedding_models == {"fake"}

def test_empty_snapshot(engine):
    snap = load_snapshot(engine)
    assert snap.chunks == [] and snap.vectors.shape == (0, 0) and snap.corpus_as_of is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_corpus_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'corpus_db'`

- [ ] **Step 3: Implement `backend/corpus_db.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_corpus_db.py -v`
Expected: 4 passed

- [ ] **Step 5: Update `sql/schema.sql`**

Append `documents` and `chunks` `CREATE TABLE IF NOT EXISTS` statements matching the Table definitions (use `ENUM('ois','uscis','sevp','cfr')` for `tier`, `BLOB` for `embedding`, `DATE NULL` for `published`). Also add `sources JSON NULL` to `messages` (used in Task 10) and a comment that the dev `messages` table must be dropped and recreated when upgrading from Part 1.

- [ ] **Step 6: Commit**

```bash
git add backend/corpus_db.py backend/tests/test_corpus_db.py ../sql/schema.sql
git commit -m "Part 4: documents/chunks tables and corpus store"
```

---

### Task 8: `ingest.py`

**Files:**
- Create: `backend/ingest.py`
- Test: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `manifest.load_manifest`, `extract.extract_source`, `chunking.chunk_pages`, `embeddings.Embedder/to_blob`, `corpus_db.*`.
- Produces:
  ```python
  @dataclass
  class IngestReport:
      added: list[str]; replaced: list[str]; skipped: list[str]; pruned: list[str]
      missing: list[str]; empty: list[str]          # empty = ingested with zero chunks
  def sha256_of(path: Path) -> str
  def ingest(manifest_path: Path, data_dir: Path, engine: Engine, embedder: Embedder, *, prune: bool = False) -> IngestReport
  def main() -> int    # argparse --prune; engine from settings.SYNC_DATABASE_URL; calls create_tables(engine) first; OpenAIEmbedder(settings.EMBEDDING_MODEL, settings.OPENAI_API_KEY); prints report
  ```
  Per source: file absent → `missing`; `existing_sha == sha256_of(file)` → `skipped`; else extract, chunk, embed (one `embed` call per document, in `batch_size` slices inside the embedder), `replace_document` inside `engine.begin()`; `added` if there was no previous row, `replaced` otherwise; zero chunks also go to `empty`. `fetched_on` is `source.fetched or date.today()`. An exception in one document is logged with the source id and the run continues; it is reported in `missing` only if the file was absent, otherwise re-raised at the end as `IngestError` listing failed ids after all other documents were processed.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_ingest.py
from pathlib import Path
from corpus_db import make_engine, create_tables, load_snapshot
from embeddings import FakeEmbedder
from ingest import ingest

def setup(tmp_path, html="<html><body><main><p>" + "OPT text. " * 300 + "</p></main></body></html>"):
    d = tmp_path / "data"; (d / "federal").mkdir(parents=True); (d / "ois").mkdir()
    (d / "federal" / "x.html").write_text(html)
    (d / "sources.yaml").write_text(
        "- {id: uscis-x, title: X, tier: uscis, url: u, file: data/federal/x.html, kind: html, fetched: 2026-10-31}\n"
        "- {id: ois-a, title: A, tier: ois, url: v, file: data/ois/a.pdf, kind: pdf, manual: true}\n")
    e = make_engine("sqlite://"); create_tables(e)
    return d / "sources.yaml", d, e

def test_ingest_adds_then_skips_then_replaces(tmp_path):
    m, d, e = setup(tmp_path)
    r1 = ingest(m, d, e, FakeEmbedder())
    assert r1.added == ["uscis-x"] and r1.missing == ["ois-a"]
    assert len(load_snapshot(e).chunks) >= 3
    r2 = ingest(m, d, e, FakeEmbedder())
    assert r2.skipped == ["uscis-x"] and r2.added == []
    (d / "federal" / "x.html").write_text("<html><body><main><p>changed</p></main></body></html>")
    r3 = ingest(m, d, e, FakeEmbedder())
    assert r3.replaced == ["uscis-x"] and [c.content for c in load_snapshot(e).chunks] == ["changed"]

def test_ingest_reports_missing_files(tmp_path):
    m, d, e = setup(tmp_path)
    (d / "federal" / "x.html").unlink()
    r = ingest(m, d, e, FakeEmbedder())
    assert sorted(r.missing) == ["ois-a", "uscis-x"] and load_snapshot(e).document_count == 0

def test_ingest_document_with_no_text_is_recorded_with_zero_chunks(tmp_path):
    m, d, e = setup(tmp_path, html="<html><body><main></main></body></html>")
    r = ingest(m, d, e, FakeEmbedder())
    assert r.added == ["uscis-x"] and r.empty == ["uscis-x"]
    assert load_snapshot(e).document_count == 1 and load_snapshot(e).chunks == []

def test_prune_removes_documents_missing_from_manifest(tmp_path):
    m, d, e = setup(tmp_path)
    ingest(m, d, e, FakeEmbedder())
    m.write_text("- {id: ois-a, title: A, tier: ois, url: v, file: data/ois/a.pdf, kind: pdf, manual: true}\n")
    r = ingest(m, d, e, FakeEmbedder(), prune=True)
    assert r.pruned == ["uscis-x"] and load_snapshot(e).document_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest'`

- [ ] **Step 3: Implement `backend/ingest.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: 4 passed

- [ ] **Step 5: Ingest the real federal tier into MySQL**

Precondition: MySQL running, `.env` has a funded `OPENAI_API_KEY`. `ingest.py` creates the `documents` and `chunks` tables itself.

Run: `.venv/bin/python ingest.py`
Expected: `added: 20`, `missing: 29` (all `ois-*`), `empty: 0`, `failed: none`; then
```bash
mysql -u chatuser -pchatpass groupchat -e "SELECT tier, COUNT(*) docs, SUM(chunk_count) chunks FROM documents GROUP BY tier"
```
shows `uscis 13`, `sevp 6`, `cfr 1`, with total chunks between 400 and 900. If `cfr` has more than 500 chunks on its own, the `(g)` marker was not found: fix `cfr_paragraph_f` (Task 4) before continuing.

- [ ] **Step 6: Commit**

```bash
git add backend/ingest.py backend/tests/test_ingest.py
git commit -m "Part 4: idempotent ingest into documents/chunks"
```

---

### Task 9: Retrieval and answering (`rag.py`)

**Files:**
- Create: `backend/rag.py`
- Test: `backend/tests/test_rag.py`

**Interfaces:**
- Consumes: `corpus_db.CorpusSnapshot/ChunkMeta`, `embeddings.Embedder/normalise`.
- Produces:
  ```python
  NOT_COVERED = "I couldn't find this in the OIS and USCIS materials I have. Please ask an OIS advisor: https://ois.usc.edu/contact/"
  EMBED_ERROR = "Sorry, I couldn't process your question right now (embedding service error). Please try again in a moment."
  LLM_ERROR = "Sorry, the language model is not responding right now. Please try again in a moment."
  SYSTEM_PROMPT: str   # fixed text, below
  class CorpusModelMismatch(RuntimeError)

  @dataclass
  class Hit:
      n: int; title: str; tier: str; url: str; page: int | None; published: date | None; excerpt: str; score: float; content: str
      def to_source(self) -> dict     # {"n","title","url","page","published" (ISO or None),"excerpt"}

  class Retriever:
      def __init__(self, snapshot: CorpusSnapshot, expected_model: str)   # raises CorpusModelMismatch if snapshot.embedding_models - {expected_model} is non-empty
      size: int                                                              # number of chunks
      def search(self, query_vec: Sequence[float], top_k: int, min_score: float) -> list[Hit]   # n numbered 1..len, excerpt = content[:600]

  def build_messages(question: str, hits: list[Hit], history: list[dict]) -> list[dict]
      # [system(SYSTEM_PROMPT + "\n\nSources:\n" + "[n] title (tier, p.page)\ncontent" per hit), *history, user(question)]

  ChatFn = Callable[[list[dict]], Awaitable[str]]
  class LLMClient:
      def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 120.0)
      async def chat(self, messages: list[dict]) -> str   # POST {base_url}/chat/completions, temperature 0.2, max_tokens 500; raises httpx errors / KeyError

  @dataclass
  class BotAnswer: content: str; sources: list[dict]
  async def answer(question: str, history: list[dict], *, retriever: Retriever, embedder: Embedder,
                   chat: ChatFn, top_k: int, min_score: float) -> BotAnswer
  ```
  `answer`: `await asyncio.to_thread(embedder.embed, [question])`; any exception → `BotAnswer(EMBED_ERROR, [])`. Empty hits → `BotAnswer(NOT_COVERED, [])` with no `chat` call. Else `await chat(build_messages(...))`; any exception → `BotAnswer(LLM_ERROR, [])`. Success → `BotAnswer(text.strip(), [h.to_source() for h in hits])`.

  `SYSTEM_PROMPT` (exact):
  ```
  You are the USC OIS Assistant, helping USC international students on F-1 status with questions about on-campus work, CPT, OPT, the STEM OPT extension, cap-gap, maintaining status, program extension and completion, travel and re-entry, and F-2 dependents.

  Rules:
  1. Answer only from the numbered sources below. Do not use outside knowledge.
  2. After each sentence that relies on a source, cite it like [1] or [2][3].
  3. If the sources do not answer the question, say so plainly and tell the student to ask an OIS advisor at https://ois.usc.edu/contact/.
  4. For USC procedures (forms, requests, deadlines, processing times) prefer OIS sources. For the rule itself prefer USCIS, SEVP and CFR sources.
  5. Be concise: a short paragraph or a few bullet points. Use plain English.
  6. Never give legal advice, predict case outcomes, or tell the student what to do about a specific case; describe what the sources say and point them to OIS.
  ```

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_rag.py
import asyncio
from datetime import date
import numpy as np
import pytest
from corpus_db import CorpusSnapshot, ChunkMeta
from embeddings import FakeEmbedder
from rag import Retriever, build_messages, answer, NOT_COVERED, EMBED_ERROR, LLM_ERROR, CorpusModelMismatch, SYSTEM_PROMPT

def meta(i, content="c", page=None):
    return ChunkMeta(chunk_id=i, document_id=1, title=f"Doc{i}", tier="ois", url=f"https://x/{i}", page=page, published=None, content=content)

def snapshot(vectors, models={"fake"}):
    v = np.array(vectors, dtype=np.float32)
    return CorpusSnapshot(chunks=[meta(i, content="x" * 700, page=3) for i in range(len(vectors))],
                          vectors=v, embedding_models=models, document_count=1, corpus_as_of=date(2026, 10, 31))

def test_search_ranks_and_applies_threshold():
    r = Retriever(snapshot([[1, 0], [0.8, 0.6], [0, 1]]), expected_model="fake")
    hits = r.search([1, 0], top_k=5, min_score=0.5)
    assert [h.title for h in hits] == ["Doc0", "Doc1"] and [h.n for h in hits] == [1, 2]
    assert hits[0].score > hits[1].score and len(hits[0].excerpt) == 600 and hits[0].page == 3

def test_search_respects_top_k_and_empty_corpus():
    r = Retriever(snapshot([[1, 0], [0.9, 0.1], [0.8, 0.2]]), expected_model="fake")
    assert len(r.search([1, 0], top_k=2, min_score=0.0)) == 2
    empty = Retriever(CorpusSnapshot([], np.zeros((0, 0), np.float32), set(), 0, None), expected_model="fake")
    assert empty.size == 0 and empty.search([1, 0], top_k=5, min_score=0.0) == []

def test_retriever_rejects_model_mismatch():
    with pytest.raises(CorpusModelMismatch):
        Retriever(snapshot([[1, 0]], models={"text-embedding-3-small"}), expected_model="fake")

def test_build_messages_layout():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    hits = r.search([1, 0], top_k=1, min_score=0.0)
    msgs = build_messages("When can I apply?", hits, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(SYSTEM_PROMPT)
    assert "[1] Doc0 (ois, p.3)" in msgs[0]["content"]
    assert msgs[1:3] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert msgs[-1] == {"role": "user", "content": "When can I apply?"}

class Emb(FakeEmbedder):
    def embed(self, texts): return [[1.0, 0.0]]

class BrokenEmb(FakeEmbedder):
    def embed(self, texts): raise RuntimeError("401")

def run(coro): return asyncio.run(coro)

def test_answer_happy_path_returns_sources():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    seen = {}
    async def chat(msgs): seen["msgs"] = msgs; return "  You may apply 90 days before [1]. "
    out = run(answer("q", [], retriever=r, embedder=Emb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == "You may apply 90 days before [1]."
    assert out.sources == [{"n": 1, "title": "Doc0", "url": "https://x/0", "page": 3, "published": None, "excerpt": "x" * 600}]
    assert seen["msgs"][-1]["content"] == "q"

def test_answer_not_covered_skips_llm():
    r = Retriever(snapshot([[0, 1]]), expected_model="fake")
    async def chat(msgs): raise AssertionError("must not be called")
    out = run(answer("q", [], retriever=r, embedder=Emb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == NOT_COVERED and out.sources == []

def test_answer_when_embedding_fails():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    async def chat(msgs): raise AssertionError("must not be called")
    out = run(answer("q", [], retriever=r, embedder=BrokenEmb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == EMBED_ERROR and out.sources == []

def test_answer_when_llm_fails():
    r = Retriever(snapshot([[1, 0]]), expected_model="fake")
    async def chat(msgs): raise TimeoutError("read timeout")
    out = run(answer("q", [], retriever=r, embedder=Emb(dim=2), chat=chat, top_k=5, min_score=0.3))
    assert out.content == LLM_ERROR and out.sources == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag'`

- [ ] **Step 3: Implement `backend/rag.py`**

`Retriever` normalises `snapshot.vectors` with `embeddings.normalise`, builds `faiss.IndexFlatIP(dim)` when `size > 0`, normalises the query the same way, searches `min(top_k, size)`, filters `score >= min_score`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rag.py -v`
Expected: 8 passed

- [ ] **Step 5: Smoke-test against the real corpus**

Run (MySQL and `OPENAI_API_KEY` needed, LLM not needed):
```bash
.venv/bin/python -c "
import settings
from corpus_db import make_engine, load_snapshot
from embeddings import OpenAIEmbedder
from rag import Retriever
r = Retriever(load_snapshot(make_engine(settings.SYNC_DATABASE_URL)), settings.EMBEDDING_MODEL)
e = OpenAIEmbedder(settings.EMBEDDING_MODEL, settings.OPENAI_API_KEY)
for h in r.search(e.embed(['How long can I be unemployed on post-completion OPT?'])[0], 5, 0.0): print(round(h.score,3), h.title, h.page)"
```
Expected: five hits, top scores above 0.45, at least one from `Practical Training` (Policy Manual Ch. 5) or `8 CFR 214.2(f)`. If every score is below 0.3, re-check that ingest stored normalised-compatible float32 blobs before touching `RAG_MIN_SCORE`.

- [ ] **Step 6: Commit**

```bash
git add backend/rag.py backend/tests/test_rag.py
git commit -m "Part 4: FAISS retriever, prompt assembly and grounded answer flow"
```

---

### Task 10: Backend wiring (`app.py`)

**Files:**
- Modify: `backend/app.py` (whole file; it is 312 lines)
- Test: `backend/tests/test_app_helpers.py`

**Interfaces:**
- Consumes: `settings.*`, `corpus_db.make_engine/create_tables/load_snapshot`, `embeddings.OpenAIEmbedder`, `rag.Retriever/LLMClient/answer/BotAnswer`.
- Produces (HTTP/WS contract the frontend relies on):
  - `GET /api/messages` → `{"messages": [...]}` only the caller's rows, oldest first, max 100; each message JSON gains `"sources": list` (`[]` when null).
  - `POST /api/messages` `{content}` → `{"ok": true}` immediately; the user message and later the bot message arrive on the caller's WebSocket as `{"type": "message", "message": {...}}`.
  - `GET /api/meta` → `{"document_count", "chunk_count", "corpus_as_of" (ISO date or null), "llm_model", "embedding_model"}`; no auth.
  - `WS /ws?token=<jwt>`: accepted then closed with code `4401` when the token is missing/invalid/expired or the user no longer exists.
  - Bot username `"OIS Assistant"`.
  - Module-level helpers: `def user_id_from_token(token: str) -> int | None`; `class ConnectionManager` with `async connect(ws, user_id)`, `disconnect(ws)`, `async send_to(user_id, data: dict)`; `def message_json(message, username) -> dict`; `async def respond(user_id: int, username: str, question: str, message_id: int) -> None` (the background task).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_app_helpers.py
import asyncio, os
os.environ.setdefault("JWT_SECRET", "test-secret-that-is-long-enough-0123456789")
from datetime import datetime, timedelta, timezone
from jose import jwt
import settings
from app import user_id_from_token, ConnectionManager, message_json, Message

def tok(sub, minutes=5):
    return jwt.encode({"sub": str(sub), "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}, settings.JWT_SECRET, algorithm="HS256")

def test_user_id_from_token_rejects_bad_tokens():
    assert user_id_from_token(tok(7)) == 7
    assert user_id_from_token(tok(7, minutes=-1)) is None
    assert user_id_from_token("garbage") is None
    assert user_id_from_token(jwt.encode({"sub": "7"}, "other-secret", algorithm="HS256")) is None

class FakeWS:
    def __init__(self): self.sent = []
    async def accept(self): pass
    async def send_text(self, s): self.sent.append(s)

def test_send_to_routes_only_to_that_user():
    m = ConnectionManager(); a1, a2, b = FakeWS(), FakeWS(), FakeWS()
    async def go():
        await m.connect(a1, 1); await m.connect(a2, 1); await m.connect(b, 2)
        await m.send_to(1, {"type": "message", "message": {"id": 1}})
        m.disconnect(a1)
        await m.send_to(1, {"type": "message", "message": {"id": 2}})
    asyncio.run(go())
    assert len(a1.sent) == 1 and len(a2.sent) == 2 and b.sent == []

def test_message_json_includes_sources_and_utc():
    m = Message(id=1, user_id=3, content="hi", is_bot=True, sources=None, created_at=datetime(2026, 10, 31, 12, 0, 0))
    out = message_json(m, "OIS Assistant")
    assert out["sources"] == [] and out["created_at"].endswith("+00:00") and out["username"] == "OIS Assistant"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_app_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'user_id_from_token'` (or the `JWT_SECRET` RuntimeError if `.env` is absent; the test sets a default before import).

- [ ] **Step 3: Rewrite `backend/app.py`**

Keep: models, `Credentials`, `NewMessage`, signup/login, static routes, UTC engine setup (now from `settings`), `save_message`. Change:
- Import config from `settings`; delete the local `os.getenv` block, the unused `authenticated_user` stub and the mid-file `from fastapi import Header`.
- `Message.sources: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)`.
- `user_id_from_token` is the single place that decodes JWTs; `current_user` uses it.
- `ConnectionManager` keyed by user id as in Interfaces.
- `lifespan`: `Base.metadata.create_all` and `corpus_db.metadata.create_all` via `conn.run_sync`; then `snapshot = await asyncio.to_thread(lambda: load_snapshot(make_engine(settings.SYNC_DATABASE_URL)))`, `app.state.retriever = Retriever(snapshot, settings.EMBEDDING_MODEL)` (let `CorpusModelMismatch` propagate so startup fails loudly), `app.state.embedder = OpenAIEmbedder(...)`, `app.state.llm = LLMClient(settings.LLM_API_BASE, settings.LLM_MODEL, settings.LLM_API_KEY)`, `app.state.corpus_as_of = snapshot.corpus_as_of`; log a warning when `retriever.size == 0`.
- `GET /api/messages` filters `Message.user_id == user.id`.
- `POST /api/messages`: save, `send_to(user.id, ...)`, `background_tasks.add_task(respond, user.id, user.username, content, saved["id"])`, return `{"ok": True}`.
- `respond`: loads up to 4 messages of this user with `id < message_id` ordered desc, reverses them, maps to `{"role": "assistant" if is_bot else "user", "content": content[:500]}`; calls `rag.answer(question, history, retriever=..., embedder=..., chat=app.state.llm.chat, top_k=settings.RAG_TOP_K, min_score=settings.RAG_MIN_SCORE)`; saves `Message(content=..., user_id=user_id, is_bot=True, sources=...)`; `send_to`. Wrap the whole body in `try/except Exception` that logs with traceback; the background task must never raise.
- `GET /api/meta` as in Interfaces; `chunk_count` is `retriever.size`.
- `/ws`: `token = websocket.query_params.get("token")`; `uid = user_id_from_token(token)`; verify the user exists; on failure `await websocket.accept(); await websocket.close(code=4401); return`.
- Bot username constant `BOT_NAME = "OIS Assistant"` used in `get_messages` and `respond`.

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass (Tasks 1 to 10).

- [ ] **Step 5: Reset the dev `messages` table and run the server**

Run:
```bash
mysql -u chatuser -pchatpass groupchat -e "DROP TABLE IF EXISTS messages"
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Expected: startup recreates `messages` with the `sources` column via `create_all`; the log shows documents and chunks loaded (20 documents), no `CorpusModelMismatch`. In another shell:
```bash
curl -s localhost:8000/api/meta
```
Expected: JSON with `document_count: 20`, `chunk_count` > 400, `llm_model: "llama-3.2-3b-instruct"`.

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/tests/test_app_helpers.py
git commit -m "Part 4: private threads, per-user WebSocket, background RAG answers, /api/meta"
```

---

### Task 11: Frontend

**Files:**
- Modify: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

**Interfaces:**
- Consumes: the HTTP/WS contract from Task 10.

- [ ] **Step 1: Rewrite `frontend/index.html`**

Structure (ids are referenced by `app.js`):
```html
<title>USC OIS Assistant</title>
<div class="container">
  <section id="auth" class="auth">
    <h1>USC OIS Assistant</h1>
    <p class="subtitle">F-1 · CPT · OPT · STEM OPT</p>
    <p>Log in to ask questions. Your conversation is private to your account.</p>
    <!-- username/password inputs and Sign Up / Log In buttons unchanged; the "?" hint removed -->
    <div id="authMsg" class="msg"></div>
  </section>
  <section id="chat" class="chat hidden">
    <header class="header"><div><h2>USC OIS Assistant</h2><p class="subtitle">F-1 · CPT · OPT · STEM OPT</p></div><button id="logoutBtn">Log Out</button></header>
    <div id="notice" class="notice">Not legal advice. Answers are generated from OIS and USCIS documents as of <span id="corpusDate">…</span>. Confirm anything important with an OIS advisor.</div>
    <div id="messages" class="messages">
      <div id="welcome" class="welcome">
        <p>Ask about working, staying in status or travelling as an F-1 student. Try one of these:</p>
        <button class="example">When can I apply for post-completion OPT?</button>
        <button class="example">How many hours can I work on campus during the semester?</button>
        <button class="example">Do I qualify for the STEM OPT extension?</button>
        <button class="example">What do I need to re-enter the U.S. after travelling?</button>
      </div>
    </div>
    <div id="typing" class="typing hidden">OIS Assistant is thinking…</div>
    <div class="composer"><input id="chatInput" placeholder="Ask a question…" autocomplete="off"><button id="sendBtn">Send</button></div>
  </section>
</div>
```

- [ ] **Step 2: Update `frontend/app.js`**

- `loadMeta()`: `GET /api/meta` (no auth) and set `#corpusDate` to `corpus_as_of || "an unknown date"`; call it on page load.
- `addMessage(m)`: as before, but username/time line; the body `div` gets class `body` and `textContent = m.content`; when `m.is_bot && m.sources.length`, append `<details class="sources"><summary>Sources (n)</summary><ol>…</ol></details>` where each `<li>` has `[n] ` + `<a href=url target=_blank rel=noopener>title</a>` + (page ? ` · p.${page}` : "") + (published ? ` · ${published}` : "") + `<blockquote>` with `textContent = excerpt`. Hide `#welcome` when any message is added; show it again when the list is empty after `loadMessages`.
- Typing indicator: show `#typing` when the user sends; hide it when a message with `is_bot` arrives or after 120 s.
- `.example` buttons: on click, put the text into `#chatInput` and call the same send routine.
- `connectWS()`: URL becomes `` `${proto}://${location.host}/ws?token=${encodeURIComponent(token)}` ``; in `onclose`, if `ev.code === 4401` then clear the token, `showAuth()`, and do not reconnect.
- `sendBtn.onclick`: also triggered by Enter in `#chatInput`; on API error show the message in a transient `.error` line under the composer instead of silently swallowing it.

- [ ] **Step 3: Update `frontend/styles.css`**

Keep the dark theme. Add: `.subtitle` (small, muted), `.notice` (amber border, 13 px, `margin: 12px 0`), `.welcome` (centered card with `.example` buttons stacked, full width on phones), `.message .body { white-space: pre-wrap; overflow-wrap: anywhere }`, `.sources` (`font-size: 13px`, `blockquote` muted with left border), `.typing` (muted, italic). Layout: `.container { padding: 16px }`, `.chat { display: flex; flex-direction: column; height: calc(100dvh - 32px) }`, `.messages { flex: 1; min-height: 0; overflow: auto }` (remove the fixed `height: 60vh`), `.composer` not shrinking; `@media (max-width: 480px)` stacks the auth `.row` inputs vertically and reduces border radius. Primary button colour `#990000` (USC cardinal) with white text; hover `filter: brightness(1.15)`.

- [ ] **Step 4: Syntax check and manual end-to-end**

Run: `node --check ../frontend/app.js` (Expected: no output). Then, with MySQL, `llama-server` (`-c 8192 -np 2`) and uvicorn running:
1. Open http://localhost:8000 in Chrome, sign up as `alice-test`, confirm the notice shows the corpus date and the welcome card shows four example questions.
2. Click "When can I apply for post-completion OPT?": the typing indicator appears, the question renders, and within about 30 s an `OIS Assistant` message appears with citation markers and a `Sources (n)` section whose links open the real USCIS/ICE pages.
3. Ask "What is the best pizza in Los Angeles?": the reply is exactly the not-covered text with no sources.
4. Stop `llama-server`, ask anything: the reply is the LLM error sentence; restart it.
5. In a second browser profile sign up as `bob-test`: the thread is empty; ask a question; switch back to Alice and confirm Bob's messages did not appear.
6. In devtools, set `localStorage.token = "bad"` and reload: the app returns to the login screen and the Network tab shows one failed WebSocket, not one every 2 s.
7. Devtools device toolbar at 390 × 844: no horizontal scroll, composer visible at the bottom, sources readable, 16 px gutters.
Record anything off and fix it before committing.

- [ ] **Step 5: Commit**

```bash
git add ../frontend
git commit -m "Part 4: OIS Assistant UI with disclaimer, example questions, sources and phone layout"
```

---

### Task 12: Retrieval eval set and script

**Files:**
- Create: `data/eval_questions.yaml`, `backend/eval_retrieval.py`
- Test: `backend/tests/test_eval_retrieval.py`

**Interfaces:**
- Consumes: `rag.Retriever`, `embeddings.Embedder`, `corpus_db.load_snapshot`.
- Produces:
  ```python
  @dataclass
  class EvalQuestion: question: str; expect: list[str]          # expected document source_ids (any one suffices)
  @dataclass
  class EvalResult: question: str; expect: list[str]; hit_ids: list[str]; ok: bool
  def load_eval(path: Path) -> list[EvalQuestion]
  def evaluate(questions: list[EvalQuestion], retriever: Retriever, embedder: Embedder,
               source_id_of: Callable[[int], str], top_k: int = 5) -> list[EvalResult]
  def main() -> int   # prints one line per question (OK/MISS, question, hit ids) and "recall@5: x/y"; exit 1 if any MISS
  ```
  `source_id_of` maps a `ChunkMeta.document_id` to its `documents.source_id`; `main` builds it from a `SELECT id, source_id FROM documents`. `Hit` needs the document id to do this: add `document_id: int` to `rag.Hit` (populated from `ChunkMeta.document_id`) in this task and keep `to_source()` unchanged.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_eval_retrieval.py
from datetime import date
import numpy as np
from corpus_db import CorpusSnapshot, ChunkMeta
from rag import Retriever
from embeddings import FakeEmbedder
from eval_retrieval import evaluate, EvalQuestion

class Emb(FakeEmbedder):
    def embed(self, texts): return [[1.0, 0.0] if "opt" in texts[0] else [0.0, 1.0]]

def test_evaluate_marks_hits_and_misses():
    chunks = [ChunkMeta(1, 10, "A", "ois", "u", None, None, "a"), ChunkMeta(2, 20, "B", "ois", "v", None, None, "b")]
    snap = CorpusSnapshot(chunks, np.array([[1, 0], [0, 1]], np.float32), {"fake"}, 2, date(2026, 10, 31))
    r = Retriever(snap, "fake")
    res = evaluate([EvalQuestion("opt?", ["ois-opt"]), EvalQuestion("cpt?", ["ois-opt"])], r, Emb(dim=2),
                   source_id_of={10: "ois-opt", 20: "ois-cpt"}.__getitem__, top_k=1)
    assert [x.ok for x in res] == [True, False] and res[1].hit_ids == ["ois-cpt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_retrieval'`

- [ ] **Step 3: Implement `backend/eval_retrieval.py` and add `document_id` to `rag.Hit`**

- [ ] **Step 4: Write `data/eval_questions.yaml`**

```yaml
- {question: "When can I apply for post-completion OPT?", expect: [ois-post-completion-opt, ois-opt, uscis-opt-for-f1-students, uscis-pm-2f-ch5]}
- {question: "How many hours can I work on campus during the semester?", expect: [ois-on-campus-employment, ois-f1-student-employment, uscis-pm-2f-ch6, cfr-8-214-2-f]}
- {question: "Do I qualify for the STEM OPT extension?", expect: [ois-stem-opt-extension, uscis-stem-opt, sevp-stem-opt-hub]}
- {question: "What do I need to re-enter the U.S. after travelling?", expect: [ois-requirements-for-reentry, ois-entering-united-states, uscis-pm-2f-ch7]}
- {question: "How many days of unemployment am I allowed on OPT?", expect: [uscis-pm-2f-ch5, cfr-8-214-2-f, uscis-opt-for-f1-students, ois-post-completion-opt]}
- {question: "What is CPT and do I need a course for it?", expect: [ois-cpt, uscis-pm-2f-ch5, cfr-8-214-2-f]}
- {question: "What is Form I-983 and who fills it out?", expect: [sevp-form-i-983-instructions, sevp-form-i-983, sevp-stem-opt-hub, uscis-stem-opt]}
- {question: "Is my major on the STEM designated degree list?", expect: [sevp-stem-designated-degree-list, uscis-stem-opt]}
- {question: "What is the cap-gap extension for H-1B?", expect: [ois-cap-gap-extension, uscis-pm-2f-ch5, uscis-opt-for-f1-students]}
- {question: "Can I take fewer classes in my last semester?", expect: [ois-reduced-course-load, uscis-pm-2f-ch3, cfr-8-214-2-f]}
- {question: "How do I extend my I-20 if I need more time to finish?", expect: [ois-extending-program, uscis-pm-2f-ch8, cfr-8-214-2-f]}
- {question: "What is the 60-day grace period after graduation?", expect: [ois-program-completion, uscis-pm-2f-ch8, cfr-8-214-2-f]}
- {question: "Can my spouse on F-2 work or study?", expect: [ois-f2-dependents, uscis-pm-2f-ch9, cfr-8-214-2-f]}
- {question: "How do I file Form I-765 for OPT?", expect: [uscis-i-765-instructions, uscis-opt-for-f1-students, ois-post-completion-opt]}
- {question: "Can I transfer my SEVIS record to another school?", expect: [ois-attending-another-institution, ois-transferring-to-usc, uscis-pm-2f-ch4]}
- {question: "Is there a new fee for OPT applications?", expect: [ois-news-proposed-opt-fees]}
```

- [ ] **Step 5: Run the test and the real eval**

Run: `.venv/bin/python -m pytest tests/test_eval_retrieval.py -v` → Expected: 1 passed.
Run: `.venv/bin/python eval_retrieval.py` → Expected (federal tier only): every question except the OIS-only ones (`Is there a new fee for OPT applications?`, and any other whose `expect` has no federal id) is `OK`; print shows `recall@5: 15/16` or better. After the OIS PDFs are ingested, rerun and expect `16/16`.

- [ ] **Step 6: Commit**

```bash
git add backend/eval_retrieval.py backend/tests/test_eval_retrieval.py backend/rag.py ../data/eval_questions.yaml
git commit -m "Part 4: retrieval eval set and recall@5 script"
```

---

### Task 13: Documentation and operator steps

**Files:**
- Modify: `SETUP.md`, `README.md`, `.env.example`

- [ ] **Step 1: Update `SETUP.md`**

- Step 6 `llama-server` command: change `-c 4096` to `-c 8192`, keep `-np 2`.
- New step after the `.env` step: "Corpus: `cd backend && .venv/bin/python fetch_sources.py` (downloads federal documents), save each OIS page listed in `data/sources.yaml` from Chrome as PDF into `data/ois/<slug>.pdf`, run `.venv/bin/python fetch_sources.py --check`, then `.venv/bin/python ingest.py`; restart uvicorn after every ingest."
- Mention `OPENAI_API_KEY` is required only for embeddings, that `ingest.py` and the backend create their own tables (`sql/schema.sql` is the reference), and that upgrading a Part 1 database requires `DROP TABLE messages` once.
- Tests: `cd backend && .venv/bin/python -m pytest`.

- [ ] **Step 2: Update `README.md`**

Rename the title to "USC OIS Assistant (EE542 Lab 9 Part 4)", describe the product in three sentences, link the spec and SETUP.md, and keep the Part 2 section.

- [ ] **Step 3: Verify docs match reality**

Follow SETUP.md from a clean shell exactly as written (skipping model download if present); every command must work as typed.

- [ ] **Step 4: Commit and push**

```bash
git add ../SETUP.md ../README.md ../.env.example
git commit -m "Part 4: setup and README for the OIS Assistant"
git push origin Lab9
```

---

## Self-review notes

- Spec coverage: §4 corpus → Tasks 2, 3, 13; §5 ingest → Tasks 4 to 8; §6 retrieval/answer → Task 9; §7 backend → Task 10; §8 frontend → Task 11; §9 verification → tests in each task, eval in Task 12, manual E2E in Task 11 step 4; §10/§11 untouched by design.
- Types: `Page = tuple[int | None, str]` is shared by Tasks 4, 6, 8; `ChunkMeta` field order used positionally in Task 12's test matches Task 7's dataclass order; `Hit.document_id` is added in Task 12 and is the only interface change after a task ships.
- Review Focus items 1 to 5 each have a named test in Tasks 6, 8, 9, 9 and 10, with the frontend reconnect behaviour checked manually in Task 11 step 6.
