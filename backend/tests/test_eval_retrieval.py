from datetime import date

import numpy as np

from corpus_db import ChunkMeta, CorpusSnapshot
from embeddings import FakeEmbedder
from eval_retrieval import EvalQuestion, evaluate
from rag import Retriever


class Emb(FakeEmbedder):
    def embed(self, texts):
        return [[1.0, 0.0] if "opt" in texts[0] else [0.0, 1.0]]


def test_evaluate_marks_hits_and_misses():
    chunks = [ChunkMeta(1, 10, "A", "ois", "u", None, None, "a"), ChunkMeta(2, 20, "B", "ois", "v", None, None, "b")]
    snap = CorpusSnapshot(chunks, np.array([[1, 0], [0, 1]], np.float32), "fake", 2, date(2026, 10, 31))
    r = Retriever(snap, "fake")
    res = evaluate(
        [EvalQuestion("opt?", ["ois-opt"]), EvalQuestion("cpt?", ["ois-opt"])], r, Emb(dim=2),
        source_id_of={10: "ois-opt", 20: "ois-cpt"}.__getitem__, top_k=1,
    )
    assert [x.ok for x in res] == [True, False] and res[1].hit_ids == ["ois-cpt"]
