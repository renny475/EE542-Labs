"""Education library: approved health sources, chunked and embedded for search.

Port of the reference store: 500-character chunks with 100-character overlap,
SQLite for text, provenance and float vectors, and exact cosine search with
NumPy. Vectors come from `llm.embed`, and every chunk records the embedding
model name so a model change shows up as "not ready" until the index is rebuilt.
"""
import hashlib
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "knowledge"

# Public, freely readable patient-education and drug-label hosts. HTTPS only.
# www.upmc.com is the one academic kidney-stone host (UPMC Department of Urology,
# University of Pittsburgh), used for the "high oxalate" mg-per-serving cutoff.
ALLOWED_HOSTS = {
    "medlineplus.gov",
    "www.niddk.nih.gov",
    "www.cdc.gov",
    "dailymed.nlm.nih.gov",
    "www.kidney.org",
    "www.upmc.com",
}

SCENARIO_TOPICS = (
    "Pneumonia",
    "Cellulitis",
    "Urinary tract infection",
    "COPD",
    "Heart failure",
    "Diabetes and hypertension",
    "Kidney stones",
)
MEDICATION_TOPIC = "Medications"

HEALTH_TERMS = (
    "health", "doctor", "symptom", "treatment", "disease", "infection", "medicine",
    "medication", "patient", "diet", "blood", "kidney", "heart", "lung", "dose",
)

USER_AGENT = "AfterCare-ClassProject/1.0 (USC EE542 educational health source collection)"
DAILYMED_SPL_XML = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
HL7 = "{urn:hl7-org:v3}"


class KnowledgeError(Exception):
    """The library is not usable as asked (not indexed, model changed, bad source)."""


def official_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    return p.scheme == "https" and p.hostname in ALLOWED_HOSTS and not p.username and not p.password


def normalize_text(text: str) -> str:
    """One stripped line per non-empty line; em dashes become " - " (house style)."""
    text = text.replace("\u2014", " - ").replace("\u00a0", " ")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def split_text(text: str) -> list[str]:
    """500-character chunks with 100-character overlap, split at line breaks first.

    The reference ran CharacterTextSplitter and then a recursive fallback for long
    paragraphs; the recursive splitter alone gives the same bounded chunks without
    logging an oversize warning for every long paragraph.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,
                                              separators=["\n\n", "\n", " ", ""])
    chunks = [piece.strip() for piece in splitter.split_text(text) if piece.strip()]
    if not chunks:
        raise KnowledgeError("The source does not contain usable text.")
    return chunks


_HTML_BLOCKS = ("p, li, h1, h2, h3, h4, h5, h6, div, section, ul, ol, dl, dt, dd, table, tr, "
                "blockquote, figcaption, caption")


def clean_html(html: str, selector: str | None = None, exclude: list[str] | None = None) -> str:
    """Readable text of the page's main region (or `selector`), one block per line.

    `exclude` lists extra CSS selectors (promos, link lists) to drop from that region.
    """
    soup = BeautifulSoup(html, "html.parser")
    if selector:
        main = soup.select_one(selector)
        if main is None:
            raise ValueError(f"The selector {selector!r} matched nothing. Review this page manually.")
    else:
        main = soup.select_one("main") or soup.select_one("article")
        if main is None:
            raise ValueError("No main/article region found. Review this page manually.")
    boilerplate = ["script, style, nav, header, footer, form, aside, noscript, .screen-reader-text", *(exclude or [])]
    for unwanted in main.select(", ".join(boilerplate)):
        unwanted.replace_with(" ")  # a space, so text around an inline citation stays apart
    # Source line breaks inside a paragraph are layout, not content. Inline elements
    # (links, emphasis) stay on their sentence's line; blocks get their own.
    for piece in main.find_all(string=True):
        piece.replace_with(re.sub(r"\s+", " ", str(piece)))
    for br in main.select("br"):
        br.replace_with("\n")
    for block in main.select(_HTML_BLOCKS):
        block.insert_before("\n")
        block.insert_after("\n")
    for cell in main.select("td, th"):
        cell.insert_after(" ")
    text = normalize_text(main.get_text())
    if len(text) < 250 or not any(term in text.lower() for term in HEALTH_TERMS):
        raise ValueError("Page appears empty or unrelated; review its URL.")
    return text


_BLOCK_TAGS = {"section", "title", "text", "paragraph", "list", "item", "table", "tr", "caption", "br"}


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _render(element, out: list[str]) -> None:
    tag = element.tag.removeprefix(HL7)
    if tag == "code":
        return
    if tag in _BLOCK_TAGS:
        out.append("\n")
    if element.text:
        out.append(_collapse(element.text))
    for child in element:
        _render(child, out)
        if child.tail:
            out.append(_collapse(child.tail))
    if tag in {"td", "th"}:
        out.append(" ")
    if tag in _BLOCK_TAGS:
        out.append("\n")


_LABEL_NOISE = re.compile(r"[\u2022\u25cf\u25aa\u043e\-*]|Figure( [A-Z0-9]{1,3})?|Image")


def dailymed_sections(xml: str, codes: list[str]) -> str:
    """Text of the label sections with these LOINC codes (with their subsections)."""
    root = ET.fromstring(xml)
    wanted = set(codes)
    picked = []

    def visit(element, inside: bool) -> None:
        for child in element:
            if child.tag == HL7 + "section":
                code = child.find(HL7 + "code")
                match = not inside and code is not None and code.get("code") in wanted
                if match:
                    picked.append(child)
                visit(child, inside or match)
            else:
                visit(child, inside)

    visit(root, False)
    if not picked:
        raise ValueError("None of the requested label sections were found; review the setid and codes.")
    parts: list[str] = []
    for section in picked:
        _render(section, parts)
    # Bullet glyphs and figure/image placeholders carry no text of their own.
    lines = normalize_text("".join(parts)).splitlines()
    return "\n".join(line for line in lines if not _LABEL_NOISE.fullmatch(line))


async def _fetch(client: httpx.AsyncClient, url: str) -> httpx.Response:
    for _ in range(6):
        response = await client.get(url)
        if response.is_redirect:
            url = str(response.url.join(response.headers["location"]))
            if not official_url(url):
                raise ValueError("Redirect left the approved source sites.")
            continue
        response.raise_for_status()
        return response
    raise ValueError("Too many redirects.")


async def collect_sources(directory: Path = DATA, transport: httpx.AsyncBaseTransport | None = None) -> list[dict]:
    """Fetch every non-curated manifest URL into an unreviewed snapshot."""
    directory = Path(directory)
    destination = directory / "sources"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if any(not official_url(item["url"]) for item in manifest):
        raise KnowledgeError("Manifest contains a URL outside the approved source sites.")
    report = []
    async with httpx.AsyncClient(timeout=45, follow_redirects=False, transport=transport,
                                 headers={"User-Agent": USER_AGENT}) as client:
        for item in manifest:
            if item.get("curated"):
                report.append({"id": item["id"], "ok": True, "skipped": "curated summary, edited by hand"})
                continue
            try:
                if "label_sections" in item:
                    setid = parse_qs(urlparse(item["url"]).query)["setid"][0]
                    response = await _fetch(client, DAILYMED_SPL_XML.format(setid=setid))
                    text = dailymed_sections(response.text, item["label_sections"])
                    url = item["url"]
                else:
                    response = await _fetch(client, item["url"])
                    text = clean_html(response.text, item.get("selector"), item.get("exclude"))
                    url = str(response.url)
                record = {**item, "url": url, "collected_at": datetime.now(timezone.utc).isoformat(),
                          "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text,
                          "reviewed": False}
                (destination / (item["id"] + ".json")).write_text(json.dumps(record, indent=2) + "\n")
                report.append({"id": item["id"], "ok": True, "characters": len(text)})
            except (httpx.HTTPError, ValueError, KeyError, ET.ParseError) as exc:
                report.append({"id": item["id"], "ok": False, "error": str(exc)})
                # A blocked refresh must not erase the curator's material, but the
                # old snapshot needs a fresh review before it is indexed again.
                previous = destination / (item["id"] + ".json")
                if previous.exists():
                    saved = json.loads(previous.read_text())
                    saved.update(reviewed=False, refresh_failed_at=datetime.now(timezone.utc).isoformat(),
                                 refresh_error=str(exc))
                    previous.write_text(json.dumps(saved, indent=2) + "\n")
    (directory / "collection-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class KnowledgeStore:
    def __init__(self, path, llm, min_similarity: float = 0.25):
        path = Path(path)
        self.path = path if path.is_absolute() else ROOT / path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.model = llm.embedding_model
        self.min_similarity = min_similarity
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
                    topic TEXT NOT NULL, collected_at TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER, content TEXT NOT NULL, vector TEXT NOT NULL, model TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    async def add(self, record: dict) -> int:
        if not official_url(record["url"]):
            raise KnowledgeError(f"Source {record['id']} must link to an approved health site over HTTPS.")
        chunks = split_text(record["text"])
        # Embed before the transaction so a failed API request leaves existing data intact.
        vectors = await self.llm.embed(chunks)
        if len(vectors) != len(chunks):
            raise KnowledgeError("Embedding service returned the wrong number of vectors.")
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE id=?", (record["id"],))
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)",
                       tuple(record[k] for k in ["id", "title", "url", "topic", "collected_at", "sha256"]))
            db.executemany("INSERT INTO chunks(document_id,page,content,vector,model) VALUES (?,?,?,?,?)",
                           [(record["id"], None, chunk, json.dumps(vector), self.model)
                            for chunk, vector in zip(chunks, vectors)])
        return len(chunks)

    def status(self) -> dict:
        with self.connect() as db:
            docs = [dict(row) for row in db.execute("""SELECT d.*, count(c.id) AS chunks
                FROM documents d LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.title""")]
            models = [row[0] for row in db.execute("SELECT DISTINCT model FROM chunks")]
        ready = bool(docs) and models == [self.model]
        topics = sorted({d["topic"] for d in docs})
        if ready:
            message = "Ready"
        elif docs:
            message = "Rebuild the index after changing EMBEDDING_MODEL."
        else:
            message = "The education library has not been indexed yet."
        return {"ready": ready, "documents": docs, "chunks": sum(d["chunks"] for d in docs),
                "embedding_model": self.model, "topics": topics,
                "missing_scenario_topics": [t for t in SCENARIO_TOPICS if t not in topics],
                "message": message}

    async def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        status = self.status()
        if not status["ready"]:
            raise KnowledgeError(status["message"])
        vector = np.asarray((await self.llm.embed([query]))[0], dtype=float)
        with self.connect() as db:
            rows = db.execute("""SELECT c.content, c.vector, d.id AS source_id, d.title, d.url, d.topic
                FROM chunks c JOIN documents d ON c.document_id=d.id WHERE c.model=?""", (self.model,)).fetchall()
        matrix = np.asarray([json.loads(row["vector"]) for row in rows], dtype=float)
        if matrix.shape[1] != len(vector):
            raise KnowledgeError("Embedding dimensions changed. Rebuild the index.")
        scores = matrix @ vector / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))
        ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)
        return [
            {"source_id": row["source_id"], "title": row["title"], "url": row["url"],
             "topic": row["topic"], "content": row["content"], "score": round(float(score), 4)}
            for row, score in ranked[:limit] if score >= self.min_similarity
        ]


def load_sources(directory: Path = DATA) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((Path(directory) / "sources").glob("*.json"))]


async def index_sources(store: KnowledgeStore, directory: Path = DATA) -> dict:
    """Rebuild the whole index from reviewed snapshots and curated summaries.

    A complete replacement is staged next to the index so a failed run cannot
    leave a mix of old and new snapshots or embedding models.
    """
    records = load_sources(directory)
    reviewed = [r for r in records if r.get("reviewed") is True]
    skipped = [r["id"] for r in records if r.get("reviewed") is not True]
    if not reviewed:
        raise KnowledgeError("No reviewed sources. Review the snapshots and set reviewed: true first.")
    staging = store.path.with_suffix(".staging.sqlite3")
    staging.unlink(missing_ok=True)
    try:
        candidate = KnowledgeStore(staging, store.llm, store.min_similarity)
        indexed = {r["id"]: await candidate.add(r) for r in reviewed}
        staging.replace(store.path)
    finally:
        staging.unlink(missing_ok=True)
    return {"indexed": indexed, "skipped": skipped}
