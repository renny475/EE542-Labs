import logging
from datetime import date

import numpy as np

import settings
from corpus_db import ChunkMeta, CorpusSnapshot
from embeddings import FakeEmbedder
from eval_retrieval import EvalQuestion, EvalResult, calibrate, evaluate, summarise
from rag import Retriever
from settings import threshold_for


class Emb(FakeEmbedder):
    def embed(self, texts):
        return [[1.0, 0.0] if "opt" in texts[0] else [0.0, 1.0]]


def make_retriever():
    chunks = [ChunkMeta(1, 10, "A", "ois", "u", None, None, "a"), ChunkMeta(2, 20, "B", "ois", "v", None, None, "b")]
    snap = CorpusSnapshot(chunks, np.array([[1, 0], [0.6, 0.8]], np.float32), "fake", 2, date(2026, 10, 31))
    return Retriever(snap, "fake")


def test_evaluate_marks_hits_misses_and_rejections():
    r = make_retriever()
    res = evaluate(
        [
            EvalQuestion("opt?", ["ois-opt"]),        # top hit is document 10 -> ois-opt, score 1.0
            EvalQuestion("opt?", ["ois-cpt"]),        # expected id never appears first... but appears second: still a hit within top_k
            EvalQuestion("opt?", ["ois-none"]),       # miss
            EvalQuestion("cpt?", []),                 # off-topic: query [0,1] -> top score 0.8 >= 0.5 -> not rejected
        ],
        r, Emb(dim=2), source_id_of={10: "ois-opt", 20: "ois-cpt"}.__getitem__, top_k=2, threshold=0.5,
    )
    assert [x.ok for x in res] == [True, True, False, False]
    assert res[2].hit_ids == ["ois-opt", "ois-cpt"] and np.isclose(res[0].top_score, 1.0)
    off = evaluate([EvalQuestion("cpt?", [])], r, Emb(dim=2), {10: "a", 20: "b"}.__getitem__, top_k=2, threshold=0.9)
    assert off[0].ok is True and np.isclose(off[0].top_score, 0.8)


def test_summarise_counts():
    results = [
        EvalResult("q1", ["a"], ["a"], 0.70, True),
        EvalResult("q2", ["a"], ["b"], 0.20, False),
        EvalResult("x1", [], ["a"], 0.40, True),
        EvalResult("x2", [], ["a"], 0.60, False),
    ]
    s = summarise("m", results, threshold=0.5)
    assert s.model == "m" and s.recall == (1, 2) and s.answered == (1, 2) and s.rejected == (1, 2) and s.threshold == 0.5


def on_(score):
    return EvalResult("q", ["a"], ["a"], score, True)


def off_(score):
    return EvalResult("x", [], ["a"], score, True)


def test_calibrate_separable_sets_use_the_gap_midpoint():
    assert np.isclose(calibrate([on_(0.70), on_(0.60), off_(0.40), off_(0.30)]), 0.50)


def test_calibrate_overlap_maximises_answered_plus_rejected_and_prefers_rejection_on_ties():
    # on {0.70, 0.60}, off {0.65}: answering both (2) ties with answer-one-reject-one (2); the higher cut wins.
    assert np.isclose(calibrate([on_(0.70), on_(0.60), off_(0.65)]), 0.675)


def test_calibrate_does_not_let_one_off_topic_outlier_silence_the_model():
    # Real shape: one off-topic question scores above most on-topic ones. Rejecting it
    # would cost twelve answers, so the cut goes below it.
    on = [on_(s) for s in (0.79, 0.64, 0.89, 0.45, 0.57, 0.67, 0.77, 0.89, 0.63, 0.36, 0.55, 0.71, 0.65)]
    off = [off_(s) for s in (0.71, 0.26, 0.00, 0.53, 0.37)]
    t = calibrate(on + off)
    assert 0.53 < t < 0.55


def test_threshold_for_prefers_model_entry_and_falls_back(caplog):
    assert threshold_for("w2v-hist-k256", {"w2v-hist-k256": 0.55}) == 0.55
    with caplog.at_level(logging.WARNING):
        assert threshold_for("w2v-hist-k256", {}) == settings.RAG_MIN_SCORE
    assert "threshold" in caplog.text.lower()
    assert threshold_for("text-embedding-3-small", {"text-embedding-3-small": 0.9}) == settings.RAG_MIN_SCORE
