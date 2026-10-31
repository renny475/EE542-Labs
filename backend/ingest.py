"""Build the corpus: extract, chunk and embed every manifest source into MySQL.

Idempotent: a document whose file hash already matches the stored row is
skipped, a changed file replaces its rows, and ``--prune`` removes rows for
sources that left the manifest. Run after ``fetch_sources.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sqlalchemy import Engine

import settings
from chunking import chunk_pages
from corpus_db import (
    ChunkRecord,
    DocumentRecord,
    create_tables,
    existing_sha,
    make_engine,
    prune_documents,
    replace_document,
)
from embeddings import Embedder, OpenAIEmbedder, to_blob
from extract import extract_source
from manifest import Source, load_manifest

log = logging.getLogger("ingest")


class IngestError(RuntimeError):
    """One or more documents failed; the rest of the run completed."""


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ingest_one(source: Source, data_dir: Path, engine: Engine, embedder: Embedder, report: IngestReport) -> None:
    path = source.path(data_dir)
    sha = sha256_of(path)
    with engine.begin() as conn:
        previous = existing_sha(conn, source.id)
    if previous == sha:
        report.skipped.append(source.id)
        return

    pages = extract_source(source, data_dir)
    chunk_texts = chunk_pages(pages)
    vectors = embedder.embed([c.content for c in chunk_texts]) if chunk_texts else []
    records = [
        ChunkRecord(chunk_index=c.index, page=c.page, content=c.content, embedding=to_blob(v))
        for c, v in zip(chunk_texts, vectors, strict=True)
    ]
    doc = DocumentRecord(
        source_id=source.id,
        title=source.title,
        tier=source.tier,
        url=source.url,
        file_path=source.file,
        sha256=sha,
        fetched_on=source.fetched or date.today(),
        embedding_model=embedder.model,
        published=source.published,
    )
    with engine.begin() as conn:
        replace_document(conn, doc, records)

    (report.replaced if previous is not None else report.added).append(source.id)
    if not records:
        report.empty.append(source.id)
    log.info("%s: %d chunks (%s)", source.id, len(records), "replaced" if previous else "added")


def ingest(manifest_path: Path, data_dir: Path, engine: Engine, embedder: Embedder, *, prune: bool = False) -> IngestReport:
    sources = load_manifest(manifest_path)
    report = IngestReport()
    failed: list[tuple[str, str]] = []

    for source in sources:
        if not source.path(data_dir).exists():
            report.missing.append(source.id)
            continue
        try:
            _ingest_one(source, data_dir, engine, embedder, report)
        except Exception as exc:  # keep going; re-raised collectively below
            log.exception("%s: ingest failed", source.id)
            failed.append((source.id, f"{type(exc).__name__}: {exc}"))

    if prune:
        with engine.begin() as conn:
            report.pruned = prune_documents(conn, keep={s.id for s in sources})

    if failed:
        details = "; ".join(f"{sid} ({err})" for sid, err in failed)
        raise IngestError(f"{len(failed)} document(s) failed: {details}")
    return report


def _print_report(report: IngestReport) -> None:
    for name in ("added", "replaced", "skipped", "pruned", "missing", "empty"):
        ids = getattr(report, name)
        print(f"{name:9s} {len(ids):3d}  {', '.join(ids) if ids else '-'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prune", action="store_true", help="delete documents no longer in the manifest")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = make_engine(settings.SYNC_DATABASE_URL)
    create_tables(engine)
    embedder = OpenAIEmbedder(settings.EMBEDDING_MODEL, settings.OPENAI_API_KEY)
    try:
        report = ingest(settings.DATA_DIR / "sources.yaml", settings.DATA_DIR, engine, embedder, prune=args.prune)
    except IngestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    print("failed: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
