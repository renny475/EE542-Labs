"""Train the Part 5 artefacts: Word2Vec on the chunk texts, hand-written
k-means over the word vectors, idf per word. Writes ``data/models/``:
``word2vec.kv``, ``centroids-k<K>.npy`` per K, ``idf.json``, ``meta.json``.
Deterministic: fixed seeds, one worker, explicit hash function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sqlalchemy import select

import settings
from corpus_db import chunks, make_engine
from embeddings import normalise
from histogram_embedding import tokenize
from kmeans import kmeans

log = logging.getLogger("train_histogram")

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 25
SEED = 42


def deterministic_hash(s: str) -> int:
    """Replacement for Python's randomised str hash so Word2Vec is reproducible."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def train_word2vec(token_lists, *, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, epochs=EPOCHS, seed=SEED):
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs,
        seed=seed,
        workers=1,
        hashfxn=deterministic_hash,
    )
    return model.wv


def compute_idf(token_lists) -> dict[str, float]:
    n = len(token_lists)
    df = Counter(tok for tokens in token_lists for tok in set(tokens))
    return {tok: math.log((1 + n) / (1 + d)) + 1 for tok, d in df.items()}


def cluster_report(kv, centroids: np.ndarray, labels: np.ndarray, n_clusters: int = 10, n_words: int = 5) -> str:
    """A few clusters with their most central words, for the eye (and the video)."""
    vectors = normalise(np.asarray(kv.vectors, dtype=np.float32))
    sizes = np.bincount(labels, minlength=centroids.shape[0])
    lines = [f"cluster sizes: min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}"]
    for j in np.argsort(-sizes)[:n_clusters]:
        members = np.flatnonzero(labels == j)
        dist = np.linalg.norm(vectors[members] - centroids[j], axis=1)
        words = [kv.index_to_key[i] for i in members[np.argsort(dist)[:n_words]]]
        lines.append(f"  cluster {j:3d} ({sizes[j]:4d} words): {', '.join(words)}")
    return "\n".join(lines)


def train(chunk_texts: list[str], models_dir: Path, *, k_values=(256,), vector_size=VECTOR_SIZE, seed=SEED) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    token_lists = [tokenize(t) for t in chunk_texts]
    kv = train_word2vec(token_lists, vector_size=vector_size, seed=seed)
    kv.save(str(models_dir / "word2vec.kv"))
    log.info("word2vec: %d words, %d dims", len(kv.index_to_key), vector_size)

    word_vectors = normalise(np.asarray(kv.vectors, dtype=np.float32))
    k_done = []
    for k in k_values:
        k_eff = min(k, word_vectors.shape[0])
        result = kmeans(word_vectors, k_eff, seed=seed)
        np.save(models_dir / f"centroids-k{k}.npy", result.centroids.astype(np.float32))
        log.info("k-means k=%d: %d iterations, inertia %.2f", k, result.iterations, result.inertia)
        print(cluster_report(kv, result.centroids, result.labels))
        k_done.append(k)

    (models_dir / "idf.json").write_text(json.dumps(compute_idf(token_lists), indent=0, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(kv.vectors, dtype=np.float32).tobytes())
    for k in k_done:
        digest.update(np.load(models_dir / f"centroids-k{k}.npy").tobytes())
    meta = {
        "fingerprint": digest.hexdigest()[:16],
        "chunk_count": len(chunk_texts),
        "vocab_size": len(kv.index_to_key),
        "vector_size": vector_size,
        "k_values": k_done,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "epochs": EPOCHS,
        "seed": seed,
    }
    (models_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=int, action="append", help="number of clusters (repeatable); default 256")
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = make_engine(settings.SYNC_DATABASE_URL)
    with engine.connect() as conn:
        texts = [row[0] for row in conn.execute(select(chunks.c.content).order_by(chunks.c.id)).all()]
    if not texts:
        print("no chunks in the database; run ingest.py first", file=sys.stderr)
        return 1
    meta = train(texts, settings.MODELS_DIR, k_values=tuple(args.k or (256,)), vector_size=args.vector_size)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
