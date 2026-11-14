"""AfterCare retrieval component. No text-generation model or Ollama required.

Uses document_processing.py for PDF ingestion and section chunking.
The default BM25 baseline needs no additional packages or downloads.
Optional semantic retrieval uses a Sentence Transformers encoder on CPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np

from document_processing import Chunk, make_chunks, read_pdf


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STOP_WORDS = set("""a an the and or to of for in on at by from with as is are was
were be been being i me my we our you your he she it its they their this that these
those what which who whom whose where when why how do does did can could should
would will shall may might must please about if then than there here into am not
""".split())


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return [word for word in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
            if word not in STOP_WORDS]


class RetrievalIndex:
    def __init__(self, chunks: list[Chunk], document_id: str,
                 backend: str = "bm25", model: str = MODEL,
                 vectors: np.ndarray | None = None):
        if not chunks or backend not in {"bm25", "semantic"}:
            raise ValueError("An index needs sections and a supported backend.")
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("Duplicate section IDs.")
        self.chunks, self.document_id = chunks, document_id
        self.backend, self.model, self.vectors = backend, model, vectors
        self.encoder = None
        self.term_counts = [Counter(tokenize(c.heading + "\n" + c.text)) for c in chunks]
        self.lengths = np.array([sum(c.values()) for c in self.term_counts], dtype=float)
        self.average_length = max(float(self.lengths.mean()), 1.0)
        self.document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )
        if vectors is not None:
            self._validate_vectors(vectors)

    def _validate_vectors(self, vectors):
        if (vectors.ndim != 2 or vectors.shape[0] != len(self.chunks)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
            raise ValueError("Invalid or unnormalized index vectors.")

    def _encoder(self):
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic backend needs sentence-transformers. "
                    "Install it, or build with --backend bm25."
                ) from exc
            self.encoder = SentenceTransformer(self.model, device="cpu")
        return self.encoder

    def prepare_semantic_vectors(self):
        encoder = self._encoder()
        texts = [c.heading + "\n" + c.text for c in self.chunks]
        for chunk, text in zip(self.chunks, texts):
            count = len(encoder.tokenizer(text, truncation=False)["input_ids"])
            if count > encoder.max_seq_length:
                raise ValueError(
                    f"Section {chunk.id} exceeds the encoder's token limit. "
                    "Split it with its qualifiers intact; do not truncate silently."
                )
        self.vectors = np.asarray(encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ), dtype=np.float32)
        self._validate_vectors(self.vectors)

    def search(self, query: str, top_k: int = 3, min_score: float | None = None) -> dict:
        if not query.strip() or top_k < 1:
            raise ValueError("Provide a nonempty query and top_k >= 1.")
        if self.backend == "semantic":
            if self.vectors is None:
                raise ValueError("Semantic index has no embeddings; rebuild it.")
            encoder = self._encoder()
            if len(encoder.tokenizer(query, truncation=False)["input_ids"]) > encoder.max_seq_length:
                raise ValueError("Query exceeds the encoder's token limit.")
            query_vector = np.asarray(encoder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            ), dtype=np.float32)[0]
            if query_vector.shape[0] != self.vectors.shape[1]:
                raise ValueError("Query/index embedding dimensions differ; rebuild the index.")
            scores = self.vectors @ query_vector
            # A semantic nearest neighbor is NOT proof that the sheet answers a query.
            threshold = -math.inf if min_score is None else min_score
        else:
            scores = np.zeros(len(self.chunks), dtype=float)
            for term in set(tokenize(query)):
                frequency = self.document_frequency.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.chunks) - frequency + 0.5) / (frequency + 0.5))
                for i, counts in enumerate(self.term_counts):
                    tf = counts.get(term, 0)
                    denominator = tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average_length)
                    scores[i] += idf * tf * 2.5 / denominator
            threshold = 0.0 if min_score is None else max(0.0, min_score)
        hits = []
        for i in np.argsort(-scores, kind="stable"):
            if float(scores[i]) <= threshold:
                continue
            chunk = self.chunks[i]
            hits.append({"source_id": f"{self.document_id}:{chunk.id}",
                         "page": chunk.page, "chunk_id": chunk.id,
                         "heading": chunk.heading, "text": chunk.text,
                         "score": round(float(scores[i]), 6)})
            if len(hits) >= top_k:
                break
        return {"document_id": self.document_id, "query": query,
                "backend": self.backend, "matches": hits,
                "score_is_confidence": False}

    def save(self, directory: Path):
        if directory.exists():
            raise ValueError("Index directory exists; choose a fresh directory.")
        if self.backend == "semantic":
            if self.vectors is None:
                raise ValueError("Build semantic vectors before saving.")
            self._validate_vectors(self.vectors)
        directory.mkdir(parents=True)
        data = {"format_version": 1, "document_id": self.document_id,
                "backend": self.backend,
                "embedding_model": self.model if self.backend == "semantic" else None,
                "chunks": [c.__dict__ for c in self.chunks]}
        (directory / "index.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        if self.vectors is not None and self.backend == "semantic":
            np.save(directory / "embeddings.npy", self.vectors)

    @classmethod
    def load(cls, directory: Path):
        data = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if data.get("format_version") != 1:
            raise ValueError("Unsupported index version.")
        backend = data["backend"]
        vectors = np.load(directory / "embeddings.npy", allow_pickle=False) if backend == "semantic" else None
        return cls([Chunk(**c) for c in data["chunks"]], data["document_id"],
                   backend, data["embedding_model"] or MODEL, vectors)

    def export_all(self):
        """Checklist generation needs every section, not a top-k subset."""
        return {"document_id": self.document_id,
                "pages_received": sorted({c.page for c in self.chunks}),
                "sections": [{"source_id": f"{self.document_id}:{c.id}", **c.__dict__}
                             for c in self.chunks]}


# Fixture-specific evaluation only. Expected sections never influence ranking.
GOUT_CASES = [
    ("How often do I take colchicine?", "Medications"),
    ("Should I take naproxen with food?", "Medications"),
    ("What foods and drinks should I avoid?", "Diet"),
    ("How much water should I drink?", "Diet"),
    ("When can I return to normal activity?", "Activity"),
    ("How long should I put an ice pack on my toe?", "Joint care"),
    ("What shoes should I wear?", "Joint care"),
    ("When should I write down my pain score?", "Monitoring"),
    ("When is the follow-up with Dr Helen Okafor?", "Follow-up"),
    ("When is my repeat kidney blood test?", "Follow-up"),
    ("When should I call 911?", "Call 911 or return to the ER if"),
    ("What should I do about chest pain or fainting?", "Call 911 or return to the ER if"),
    ("When should I call the clinic?", "Call the clinic at 555-0177 if"),
    ("What if pain is not better after 3 days?", "Call the clinic at 555-0177 if"),
    ("What should I do for black stools or vomiting?", "Call the clinic at 555-0177 if"),
]


def evaluate_gout(index: RetrievalIndex) -> dict:
    headings = {c.heading for c in index.chunks}
    if not {expected for _, expected in GOUT_CASES}.issubset(headings):
        raise ValueError("eval-gout is only for the supplied fictional gout reference sheet.")
    details = []
    for query, expected in GOUT_CASES:
        matches = index.search(query, top_k=3)["matches"]
        retrieved = [m["heading"] for m in matches]
        rank = retrieved.index(expected) + 1 if expected in retrieved else None
        details.append({"query": query, "expected_section": expected,
                        "rank_within_top_3": rank, "retrieved_sections": retrieved})
    return {"backend": index.backend, "query_count": len(details),
            "hit_at_1": sum(r["rank_within_top_3"] == 1 for r in details) / len(details),
            "hit_at_3": sum(r["rank_within_top_3"] is not None for r in details) / len(details),
            "details": details,
            "scope": "Small same-document development benchmark, not a generalization result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("pdf", type=Path)
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--backend", choices=["bm25", "semantic"], default="bm25")
    build.add_argument("--model", default=MODEL)
    search = commands.add_parser("search")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--min-score", type=float)
    export = commands.add_parser("export")
    export.add_argument("--index", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("eval-gout")
    evaluate.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if args.index.exists():
            raise ValueError("Index directory exists; choose a fresh directory.")
        pages = read_pdf(args.pdf)
        chunks = make_chunks(pages)
        index = RetrievalIndex(chunks, hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                               args.backend, args.model)
        if args.backend == "semantic":
            index.prepare_semantic_vectors()
        index.save(args.index)
        print(f"Indexed {len(pages)} pages and {len(chunks)} sections using {args.backend}.")
    else:
        index = RetrievalIndex.load(args.index)
        if args.command == "search":
            print(json.dumps(index.search(args.query, args.top_k, args.min_score), indent=2))
        elif args.command == "eval-gout":
            print(json.dumps(evaluate_gout(index), indent=2))
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(index.export_all(), handle, indent=2)
            print(f"Exported all {len(index.chunks)} source sections to {args.output}.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(1)
