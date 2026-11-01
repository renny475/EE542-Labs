"""Word-histogram embedding (Lab 9 Part 5).

Every word in the corpus has a Word2Vec vector; the vectors are clustered
into K groups (``kmeans.py``); a chunk is represented by the normalised
histogram of its words' nearest clusters. Variants: ``idf`` weights each
word by its inverse document frequency, ``soft`` spreads a word over its
T nearest clusters by distance. Artefacts come from ``train_histogram.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embeddings import HISTOGRAM_MODEL_RE, normalise
from kmeans import nearest_centroids

TOKEN_RE = re.compile(r"[a-z][a-z0-9-]+")
MIN_TOKEN_LEN = 3
SOFT_T = 3
SOFT_TAU = 0.1

STOP_WORDS: frozenset[str] = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself just let may
    me might more most must my myself no nor not now of off on once only or other our ours ourselves out
    over own same shall she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up upon us very was we were what when where which
    while who whom why will with within without would you your yours yourself yourselves""".split()
)

ARTIFACT_FILES = ("word2vec.kv", "centroids-k{k}.npy", "idf.json", "meta.json")


def tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in TOKEN_RE.findall(text.lower())
        if len(tok) >= MIN_TOKEN_LEN and tok not in STOP_WORDS
    ]


@dataclass(frozen=True)
class HistogramSpec:
    k: int
    idf: bool
    soft: bool

    @property
    def name(self) -> str:
        return f"w2v-hist-k{self.k}" + ("-idf" if self.idf else "") + ("-soft" if self.soft else "")

    @classmethod
    def parse(cls, name: str) -> HistogramSpec:
        m = HISTOGRAM_MODEL_RE.match(name)
        if m is None:
            raise ValueError(f"not a histogram model name: {name!r} (expected w2v-hist-k<K>[-idf][-soft])")
        return cls(k=int(m.group(1)), idf=m.group(2) is not None, soft=m.group(3) is not None)


@dataclass
class Artifacts:
    vectors: np.ndarray  # (V, N) L2-normalised word vectors, row i = word with vocab index i
    vocab: dict[str, int]
    centroids: np.ndarray  # (K, N)
    idf: dict[str, float]
    chunk_count: int
    fingerprint: str | None = None  # from meta.json; identifies this training run


class ArtifactsMissing(FileNotFoundError):
    """A trained artefact is absent; run train_histogram.py first."""


def load_artifacts(models_dir: Path, k: int) -> Artifacts:
    for name in ARTIFACT_FILES:
        path = models_dir / name.format(k=k)
        if not path.exists():
            raise ArtifactsMissing(f"{path} is missing; run train_histogram.py --k {k} first")
    from gensim.models import KeyedVectors  # heavy import, only when artefacts are used

    kv = KeyedVectors.load(str(models_dir / "word2vec.kv"))
    vectors = normalise(np.asarray(kv.vectors, dtype=np.float32))
    vocab = {word: i for i, word in enumerate(kv.index_to_key)}
    centroids = np.load(models_dir / f"centroids-k{k}.npy").astype(np.float32)
    idf = json.loads((models_dir / "idf.json").read_text(encoding="utf-8"))
    meta = json.loads((models_dir / "meta.json").read_text(encoding="utf-8"))
    return Artifacts(
        vectors=vectors,
        vocab=vocab,
        centroids=centroids,
        idf=idf,
        chunk_count=int(meta["chunk_count"]),
        fingerprint=meta.get("fingerprint"),
    )


def histogram(tokens: list[str], art: Artifacts, spec: HistogramSpec) -> np.ndarray:
    """The normalised K-bin histogram of the tokens (zeros when none is in the vocabulary)."""
    hist = np.zeros(spec.k, dtype=np.float64)
    known = [tok for tok in tokens if tok in art.vocab]
    if not known:
        return hist
    word_vecs = art.vectors[[art.vocab[tok] for tok in known]]
    weights = np.array([art.idf.get(tok, 1.0) for tok in known]) if spec.idf else np.ones(len(known))
    if spec.soft:
        idx, dist = nearest_centroids(word_vecs, art.centroids, t=SOFT_T)
        logits = -dist / SOFT_TAU
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        np.add.at(hist, idx.ravel(), (probs * weights[:, None]).ravel())
    else:
        idx, _ = nearest_centroids(word_vecs, art.centroids, t=1)
        np.add.at(hist, idx[:, 0], weights)
    return hist / len(known)


class HistogramEmbedder:
    """``Embedder`` over the trained artefacts; vectors are L2-normalised for cosine search."""

    def __init__(self, art: Artifacts, spec: HistogramSpec) -> None:
        if art.centroids.shape[0] != spec.k:
            raise ValueError(f"centroids have {art.centroids.shape[0]} rows but the model asks for k={spec.k}")
        self.art = art
        self.spec = spec
        self.model = spec.name
        self.dim = spec.k
        self.fingerprint = art.fingerprint

    @classmethod
    def load(cls, model: str, models_dir: Path) -> HistogramEmbedder:
        spec = HistogramSpec.parse(model)
        return cls(load_artifacts(models_dir, spec.k), spec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        hists = np.vstack([histogram(tokenize(t), self.art, self.spec) for t in texts]) if texts else np.zeros((0, self.dim))
        return normalise(hists.astype(np.float32)).tolist()
