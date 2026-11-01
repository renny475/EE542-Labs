import logging
import math

import numpy as np

from corpus_db import create_tables, make_engine
from embed_corpus import embed_corpus
from histogram_embedding import HistogramEmbedder
from ingest import ingest
from tests.test_ingest import setup
from train_histogram import compute_idf, train

TEXTS = [f"opt employment work authorization student {w}" for w in ("alpha", "beta", "gamma")] * 10 + [
    "travel visa passport reentry port of entry"
] * 10


def test_train_writes_all_artifacts_and_is_deterministic(tmp_path):
    meta1 = train(TEXTS, tmp_path / "a", k_values=(4,), vector_size=16)
    meta2 = train(TEXTS, tmp_path / "b", k_values=(4,), vector_size=16)
    for f in ("word2vec.kv", "centroids-k4.npy", "idf.json", "meta.json"):
        assert (tmp_path / "a" / f).exists()
    assert meta1["chunk_count"] == len(TEXTS) and meta1["k_values"] == [4] and meta1["vector_size"] == 16
    assert np.array_equal(np.load(tmp_path / "a" / "centroids-k4.npy"), np.load(tmp_path / "b" / "centroids-k4.npy"))
    assert meta1 == meta2


def test_compute_idf_matches_formula():
    idf = compute_idf([["a", "b"], ["a"], ["c"]])
    assert np.isclose(idf["a"], math.log(4 / 3) + 1) and np.isclose(idf["c"], math.log(4 / 2) + 1)


def test_trained_artifacts_load_into_embedder(tmp_path):
    train(TEXTS, tmp_path, k_values=(4,), vector_size=16)
    e = HistogramEmbedder.load("w2v-hist-k4-idf", tmp_path)
    a, b = e.embed(["opt employment student", "travel visa passport"])
    assert np.dot(a, a) > 0 and np.dot(a, b) < 0.9  # different topics land in different bins


def test_embed_corpus_warns_when_vocab_predates_chunks(tmp_path, caplog):
    from embed_corpus import check_vocab_age

    m, d, e = setup(tmp_path)
    ingest(m, d, e)  # several chunks
    train(TEXTS[:3], tmp_path / "models", k_values=(2,), vector_size=8)  # trained on 3 "chunks"
    emb = HistogramEmbedder.load("w2v-hist-k2", tmp_path / "models")
    with caplog.at_level(logging.WARNING):
        stale = check_vocab_age(e, emb)
    assert stale is True and "retrain" in caplog.text.lower()
