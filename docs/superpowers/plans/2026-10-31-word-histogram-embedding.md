# Word-Histogram Embedding (Lab 9 Part 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, hand-built chunk embedding (Word2Vec words → hand-written K-means clusters → normalised K-bin histogram per chunk, with idf and soft-assignment variants) that coexists with the OpenAI embedding so a question can be asked with either and the retrievals compared.

**Architecture:** Vectors move from `chunks.embedding` into a `chunk_embeddings` table keyed by `(chunk_id, model)`; `ingest.py` stops embedding and a new `embed_corpus.py --model` fills vectors per model. `train_histogram.py` trains Word2Vec on the chunks, clusters the word vectors with `kmeans.py`, and saves artefacts under `data/models/`; `histogram_embedding.py` turns text into histograms from those artefacts behind the existing `Embedder` protocol. The app holds one retriever per available model, the request names which to use, and the eval reports recall and off-topic rejection per model.

**Tech Stack:** Python 3.13 (`backend/.venv`), numpy, FAISS, SQLAlchemy Core, MySQL (SQLite in tests), gensim 4.4 (Word2Vec only), pytest; vanilla JS frontend.

**Spec:** `docs/superpowers/specs/2026-10-31-word-histogram-embedding-design.md` (extends `2026-10-31-ois-assistant-design.md`)

## Global Constraints

- Branch `Lab9`, all code under `groupchat_app_src/`; commit after every task; never commit `.env`, `.venv`, keys. Commands run from `groupchat_app_src/backend/` with `.venv/bin/python` unless stated.
- K-means is hand-written with numpy only: no scikit-learn, no gensim/faiss clustering. Word2Vec comes from gensim.
- Determinism: Word2Vec `seed=42`, `workers=1`, explicit `hashfxn`; K-means seeded numpy RNG, k-means++ init, stop when assignments stop changing or after 100 iterations.
- Word2Vec: `vector_size=100`, `window=5`, `min_count=2`, `sg=1`, `epochs=25`. K-means on L2-normalised word vectors, default `K=256`.
- Tokeniser: lowercase, tokens matching `[a-z][a-z0-9-]+`, drop stop words and tokens shorter than 3 characters.
- Histogram variants and names: `w2v-hist-k256`, `w2v-hist-k256-idf`, `w2v-hist-k256-soft`, `w2v-hist-k256-idf-soft`. `idf` adds `idf[word]` instead of 1; `soft` spreads a word over its T=3 nearest centroids with `softmax(-d / 0.1)`. Divide by counted words, then L2-normalise. Zero in-vocabulary words → zero vector.
- idf formula: `log((1 + N) / (1 + df)) + 1`, N = number of chunks.
- `chunk_embeddings(chunk_id, model VARCHAR(64), dim INT, embedding BLOB, PRIMARY KEY (chunk_id, model))`; `chunks.embedding` and `documents.embedding_model` are removed.
- Default embedding `text-embedding-3-small` when available, else the first histogram model; an unavailable model in a request → 400; no available model → startup error.
- Thresholds: OpenAI uses `RAG_MIN_SCORE`; histogram models read `data/models/thresholds.json`; a missing entry falls back to `RAG_MIN_SCORE` with a logged warning.
- Committed artefacts under `data/models/` stay under 15 MB total.

## Review Focus

1. A query whose words are all out of vocabulary (typo-only or non-English) must get the not-covered reply with the histogram embedding, not a crash or a random hit. Test in Task 4 (`test_histogram_embedder_returns_zero_vector_for_unknown_words`) and Task 6 (`test_retriever_zero_query_yields_no_hits`).
2. A histogram model name the frontend remembers in `localStorage` but that is no longer served must not break the composer: the select falls back to the default. Manual check in Task 8 step 5.
3. Running `embed_corpus.py --model w2v-hist-k256` before `train_histogram.py` must fail with a message naming the missing artefact, not a stack trace from numpy. Test in Task 4 (`test_load_artifacts_reports_missing_files`).
4. Old bot messages (no `embedding`, sources without `score`) must still render. Test in Task 7 (`test_message_json_defaults_embedding_to_none`) and manual check in Task 8.
5. A corpus whose chunks changed after training (re-ingest) must still embed with the old Word2Vec vocabulary instead of failing; the mismatch is only a warning. Test in Task 5 (`test_embed_corpus_warns_when_vocab_predates_chunks`) via the artefact's recorded chunk count.

---

### Task 1: `chunk_embeddings` table and per-model snapshots

**Files:**
- Modify: `backend/corpus_db.py`
- Modify: `backend/tests/test_corpus_db.py`

**Interfaces:**
- Produces (replacing the old shapes):
  ```python
  chunk_embeddings = Table("chunk_embeddings", metadata,
      Column("chunk_id", ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
      Column("model", String(64), primary_key=True),
      Column("dim", Integer, nullable=False),
      Column("embedding", LargeBinary, nullable=False))
  # chunks: no `embedding` column. documents: no `embedding_model` column.
  @dataclass
  class DocumentRecord: source_id; title; tier; url; file_path; sha256; fetched_on; published=None
  @dataclass
  class ChunkRecord: chunk_index: int; page: int | None; content: str
  @dataclass
  class CorpusSnapshot: chunks: list[ChunkMeta]; vectors: np.ndarray; model: str; document_count: int; corpus_as_of: date | None
  def existing_sha(conn, source_id) -> str | None          # unchanged
  def replace_document(conn, doc, chunk_records) -> int     # unchanged signature, no vectors
  def prune_documents(conn, keep) -> list[str]              # unchanged
  def chunks_missing_embedding(conn, model: str) -> list[tuple[int, str]]   # (chunk_id, content), ordered by chunk id
  def upsert_embeddings(conn, model: str, rows: list[tuple[int, bytes]], dim: int) -> int   # delete+insert per chunk_id; returns count
  def delete_embeddings(conn, model: str) -> int
  def models_present(engine) -> dict[str, int]              # model -> number of vectors
  def load_snapshot(engine, model: str) -> CorpusSnapshot   # only chunks with a vector for `model`; vectors (n, dim) or (0, 0)
  def has_legacy_schema(engine) -> bool                     # True when `chunks` exists and has an `embedding` column
  def reset_corpus_tables(engine) -> None                   # drop chunk_embeddings, chunks, documents (in that order) and create_all
  ```
  `CorpusSnapshot.embedding_models` is gone; `Retriever` (Task 6) compares `snapshot.model` instead. `existing_ingest` is removed.

- [ ] **Step 1: Rewrite the failing tests**

Replace `tests/test_corpus_db.py` with tests using `make_engine("sqlite://")`:

```python
def doc(sid="a", sha="1"*64, fetched=date(2026, 10, 31)):
    return DocumentRecord(source_id=sid, title="T", tier="ois", url="u", file_path="data/ois/a.pdf", sha256=sha, fetched_on=fetched)

def test_replace_document_stores_chunks_without_vectors(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, 1, "one"), ChunkRecord(1, 2, "two")])
        assert [cid for cid, _ in chunks_missing_embedding(c, "m")] and existing_sha(c, "a") == "1"*64
    assert load_snapshot(engine, "m").vectors.shape == (0, 0)

def test_upsert_embeddings_fills_one_model_and_snapshot_filters_by_model(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, None, "x"), ChunkRecord(1, None, "y")])
        ids = [cid for cid, _ in chunks_missing_embedding(c, "m1")]
        assert upsert_embeddings(c, "m1", [(ids[0], to_blob([1, 0]))], dim=2) == 1
        assert chunks_missing_embedding(c, "m1") == [(ids[1], "y")]
        upsert_embeddings(c, "m2", [(ids[0], to_blob([0, 0, 1])), (ids[1], to_blob([0, 1, 0]))], dim=3)
    s1, s2 = load_snapshot(engine, "m1"), load_snapshot(engine, "m2")
    assert s1.vectors.shape == (1, 2) and [m.content for m in s1.chunks] == ["x"] and s1.model == "m1"
    assert s2.vectors.shape == (2, 3) and s2.document_count == 1
    assert models_present(engine) == {"m1": 1, "m2": 2}

def test_replace_document_drops_its_old_vectors(engine):
    with engine.begin() as c:
        replace_document(c, doc(), [ChunkRecord(0, None, "x")])
        cid = chunks_missing_embedding(c, "m")[0][0]
        upsert_embeddings(c, "m", [(cid, to_blob([1, 0]))], dim=2)
        replace_document(c, doc(sha="2"*64), [ChunkRecord(0, None, "new")])
    assert load_snapshot(engine, "m").vectors.shape == (0, 0) and models_present(engine) == {}

def test_delete_embeddings_and_prune(engine): ...   # delete_embeddings(c, "m") returns 1 and models_present drops the key; prune_documents(c, keep={"b"}) == ["a"] removes a's vectors too

def test_snapshot_metadata_and_empty(engine): ...   # corpus_as_of = max fetched_on over all documents even without vectors; empty engine → chunks == [], vectors.shape == (0, 0), corpus_as_of None

def test_legacy_schema_detection_and_reset():
    e = make_engine("sqlite://")
    with e.begin() as c:
        c.execute(text("CREATE TABLE documents (id INTEGER PRIMARY KEY, embedding_model TEXT)"))
        c.execute(text("CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, embedding BLOB)"))
    assert has_legacy_schema(e) is True
    reset_corpus_tables(e)
    assert has_legacy_schema(e) is False and models_present(e) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_corpus_db.py -v`
Expected: FAIL with `ImportError: cannot import name 'chunks_missing_embedding'` (and the removed fields).

- [ ] **Step 3: Implement the new `corpus_db.py`**

Keep `make_engine`, `create_tables`, `_delete_documents` (now also deleting `chunk_embeddings` rows for the chunks first, explicitly, for SQLite). `has_legacy_schema` uses `sqlalchemy.inspect`. `reset_corpus_tables` drops with `DROP TABLE IF EXISTS` in FK-safe order then `create_tables`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_corpus_db.py -v`
Expected: all pass. The whole suite will now fail in `test_ingest.py`, `test_rag.py`, `test_eval_retrieval.py`, `test_app_helpers.py` (consumers of the old shapes); Task 2 and Task 6 repair them. Do not commit a red suite: Task 1 and Task 2 share one commit.

---

### Task 2: `ingest.py` without embedding, `embed_corpus.py`, embedder factory

**Files:**
- Modify: `backend/ingest.py`, `backend/embeddings.py`, `backend/settings.py`
- Create: `backend/embed_corpus.py`
- Modify: `backend/tests/test_ingest.py`; Create: `backend/tests/test_embed_corpus.py`
- Temporarily adjust: `backend/tests/test_rag.py`, `backend/tests/test_eval_retrieval.py` (only the `CorpusSnapshot(...)` constructor calls: replace the `embedding_models` set with `model="fake"`), `backend/rag.py` line `foreign = snapshot.embedding_models - {expected_model}` → `if snapshot.model != expected_model and snapshot.chunks:` raising the same `CorpusModelMismatch`.

**Interfaces:**
- Consumes: Task 1.
- Produces:
  ```python
  # settings.py
  MODELS_DIR = DATA_DIR / "models"
  # embeddings.py
  HISTOGRAM_MODEL_RE = re.compile(r"^w2v-hist-k(\d+)(-idf)?(-soft)?$")
  def is_histogram_model(name: str) -> bool
  def make_embedder(model: str, *, api_key: str = "", models_dir: Path | None = None) -> Embedder
      # OpenAI name → OpenAIEmbedder(model, api_key); histogram name → histogram_embedding.HistogramEmbedder.load(model, models_dir) (Task 4); else ValueError(f"unknown embedding model {model!r}")
  # ingest.py
  def ingest(manifest_path, data_dir, engine, *, prune=False) -> IngestReport     # embedder parameter removed; `empty` keeps meaning zero chunks
  def main() -> int   # if has_legacy_schema(engine): log.warning(...) and reset_corpus_tables(engine); else create_tables
  # embed_corpus.py
  @dataclass
  class EmbedReport: model: str; embedded: int; skipped: int; dim: int
  def embed_corpus(engine, embedder: Embedder, *, refresh: bool = False, batch_size: int = 100) -> EmbedReport
      # refresh → delete_embeddings first; rows = chunks_missing_embedding; embed in batch_size slices; upsert_embeddings per slice (own transaction each); dim = len(first vector)
  def main() -> int   # argparse --model (required) --refresh; embedder = make_embedder(args.model, api_key=settings.OPENAI_API_KEY, models_dir=settings.MODELS_DIR); prints "model m: embedded n, skipped s, dim d"
  ```

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest.py`: drop the `FakeEmbedder` argument everywhere; keep the four existing behaviours (add/skip/replace, missing, zero-chunk → `empty`, prune); delete `test_ingest_re_embeds_when_embedding_model_changes`; add:

```python
def test_ingest_leaves_vectors_to_embed_corpus(tmp_path):
    m, d, e = setup(tmp_path)
    ingest(m, d, e)
    assert models_present(e) == {} and len(chunks_missing_embedding_all(e)) >= 3   # helper: with e.begin() as c: chunks_missing_embedding(c, "any")
```

`tests/test_embed_corpus.py`:

```python
def corpus(tmp_path):   # setup() from test_ingest, then ingest(); returns engine
def test_embed_corpus_fills_missing_then_skips(tmp_path):
    e = corpus(tmp_path)
    r1 = embed_corpus(e, FakeEmbedder(dim=4, model="fake"))
    assert r1.embedded >= 3 and r1.skipped == 0 and r1.dim == 4 and r1.model == "fake"
    r2 = embed_corpus(e, FakeEmbedder(dim=4, model="fake"))
    assert r2.embedded == 0 and r2.skipped == r1.embedded
    assert load_snapshot(e, "fake").vectors.shape == (r1.embedded, 4)
def test_embed_corpus_refresh_recomputes(tmp_path): ...   # refresh=True → embedded == n again
def test_embed_corpus_batches(tmp_path):   # embedder that records call sizes; batch_size=2 → every call ≤ 2 texts, order preserved
def test_make_embedder_rejects_unknown_name():
    with pytest.raises(ValueError): make_embedder("bert-base")
def test_is_histogram_model():
    assert is_histogram_model("w2v-hist-k256-idf-soft") and not is_histogram_model("text-embedding-3-small")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ingest.py tests/test_embed_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'embed_corpus'` and `TypeError` on the old `ingest` signature.

- [ ] **Step 3: Implement**

`ingest.py`: remove the embedder, `to_blob`, `ChunkRecord.embedding`; `IngestError` and the rest unchanged. `embed_corpus.py` as in Interfaces; logging per slice. `embeddings.make_embedder` imports `histogram_embedding` lazily (Task 4 creates it; until then the histogram branch raises `ImportError`, which is fine for this task's tests).

- [ ] **Step 4: Make the temporarily broken tests consistent and run the whole suite**

Apply the `CorpusSnapshot(... model="fake" ...)` edits in `test_rag.py` and `test_eval_retrieval.py`, the `rag.py` one-line mismatch change, and in `test_app_helpers.py` nothing (it does not build snapshots).

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Real run: reset the dev corpus and re-embed with OpenAI**

Precondition: MySQL up, `OPENAI_API_KEY` in `.env`. Stop the running uvicorn first (it holds the old snapshot).

Run:
```bash
.venv/bin/python ingest.py
.venv/bin/python embed_corpus.py --model text-embedding-3-small
```
Expected: ingest logs the legacy-schema reset, then `added 49`; embed prints `model text-embedding-3-small: embedded 785, skipped 0, dim 1536`. Then `mysql -u chatuser -pchatpass groupchat -e "SELECT model, COUNT(*) FROM chunk_embeddings GROUP BY model"` shows one row with 785.

- [ ] **Step 6: Commit (Tasks 1 and 2 together)**

```bash
git add backend/corpus_db.py backend/ingest.py backend/embed_corpus.py backend/embeddings.py backend/settings.py backend/rag.py backend/tests
git commit -m "Part 5: per-model chunk_embeddings table, ingest without vectors, embed_corpus.py"
```

---

### Task 3: Hand-written K-means

**Files:**
- Create: `backend/kmeans.py`
- Test: `backend/tests/test_kmeans.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class KMeansResult: centroids: np.ndarray; labels: np.ndarray; iterations: int; inertia: float
  def kmeans_plusplus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray   # (k, d)
  def kmeans(x: np.ndarray, k: int, *, seed: int = 42, max_iter: int = 100) -> KMeansResult
      # Lloyd's algorithm; squared Euclidean; empty cluster → re-seed its centroid to the point farthest from its current centroid; stop when labels unchanged
  def nearest_centroids(x: np.ndarray, centroids: np.ndarray, t: int = 1) -> tuple[np.ndarray, np.ndarray]
      # (indices (n, t), distances (n, t)) of the t nearest centroids per row, ascending distance; Euclidean (not squared)
  ```

- [ ] **Step 1: Write the failing tests**

```python
def blobs(rng):   # 3 clusters of 50 points around (0,0), (10,0), (0,10) in 2-D with std 0.5
def test_kmeans_recovers_separable_clusters():
    x, truth = blobs(np.random.default_rng(0))
    r = kmeans(x, 3)
    assert r.centroids.shape == (3, 2) and r.iterations <= 100
    # each true cluster maps to exactly one label
    assert len({tuple(sorted(set(r.labels[truth == g]))) for g in range(3)}) == 3 and all(len(set(r.labels[truth == g])) == 1 for g in range(3))
    assert r.inertia < 100
def test_kmeans_is_deterministic():
    x, _ = blobs(np.random.default_rng(1))
    assert np.array_equal(kmeans(x, 3).centroids, kmeans(x, 3).centroids)
def test_kmeans_plusplus_picks_distinct_points():
    x = np.array([[0, 0], [0, 0], [5, 5], [9, 9]], float)
    init = kmeans_plusplus_init(x, 3, np.random.default_rng(0))
    assert len({tuple(r) for r in init}) == 3
def test_nearest_centroids_orders_by_distance():
    c = np.array([[0, 0], [1, 0], [5, 0]], float)
    idx, dist = nearest_centroids(np.array([[0.9, 0]]), c, t=2)
    assert idx.tolist() == [[1, 0]] and np.allclose(dist, [[0.1, 0.9]])
def test_kmeans_handles_k_greater_than_unique_points_without_nan():
    x = np.array([[0, 0], [0, 0], [1, 1]], float)
    r = kmeans(x, 3)
    assert not np.isnan(r.centroids).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kmeans.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kmeans'`

- [ ] **Step 3: Implement `backend/kmeans.py`**

Distances via `((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)` in row blocks of 2048 to bound memory (vocab ~8k × k 256 × d 100 floats). k-means++: first centroid uniform, then `D²`-weighted choice with `rng.choice`; if all remaining `D²` are zero, pick uniformly among points not yet chosen.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_kmeans.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/kmeans.py backend/tests/test_kmeans.py
git commit -m "Part 5: hand-written k-means with k-means++ seeding"
```

---

### Task 4: Tokeniser, artefacts and `HistogramEmbedder`

**Files:**
- Create: `backend/histogram_embedding.py`
- Test: `backend/tests/test_histogram_embedding.py`
- Modify: `backend/requirements.txt` (add `gensim`), `backend/embeddings.py` (wire `make_embedder`'s histogram branch)

**Interfaces:**
- Consumes: `kmeans.nearest_centroids`, `embeddings.normalise`, `embeddings.HISTOGRAM_MODEL_RE`.
- Produces:
  ```python
  STOP_WORDS: frozenset[str]        # about 120 common English words; builder's list
  TOKEN_RE = re.compile(r"[a-z][a-z0-9-]+")
  def tokenize(text: str) -> list[str]
  @dataclass(frozen=True)
  class HistogramSpec: k: int; idf: bool; soft: bool
      @property
      def name(self) -> str                        # "w2v-hist-k256-idf-soft"
      @classmethod
      def parse(cls, name: str) -> "HistogramSpec"  # ValueError on non-matching names
  @dataclass
  class Artifacts: vectors: np.ndarray; vocab: dict[str, int]; centroids: np.ndarray; idf: dict[str, float]; chunk_count: int
      # vectors (V, N) L2-normalised word vectors in vocab order
  ARTIFACT_FILES = ("word2vec.kv", "centroids-k{k}.npy", "idf.json", "meta.json")   # meta.json: {"chunk_count", "vector_size", "k_values": [...]}
  class ArtifactsMissing(FileNotFoundError)
  def load_artifacts(models_dir: Path, k: int) -> Artifacts    # ArtifactsMissing naming the first absent file
  SOFT_T = 3; SOFT_TAU = 0.1
  def histogram(tokens: list[str], art: Artifacts, spec: HistogramSpec) -> np.ndarray   # shape (k,), the normalised histogram (divided by counted words), zeros when no counted words; NOT L2-normalised
  class HistogramEmbedder:            # implements Embedder
      model: str; dim: int
      def __init__(self, art: Artifacts, spec: HistogramSpec)
      @classmethod
      def load(cls, model: str, models_dir: Path) -> "HistogramEmbedder"
      def embed(self, texts: list[str]) -> list[list[float]]   # normalise(histogram(...)) per text; zero stays zero
  ```

- [ ] **Step 1: Write the failing tests**

Build a tiny `Artifacts` by hand (no gensim): vocab `{"opt": 0, "employment": 1, "travel": 2, "visa": 3}`, vectors = unit vectors along 4 axes in 4-D, centroids = 2 rows `[1,0,0,0]` and `[0,0,1,0]` (so k=2: opt/employment → bin 0, travel/visa → bin 1), idf `{"opt": 1.0, "employment": 2.0, "travel": 1.0, "visa": 3.0}`, `chunk_count=10`.

```python
def test_tokenize_lowercases_filters_stopwords_and_short_tokens():
    assert tokenize("The F-1 student's OPT and CPT!") == ["f-1", "student", "opt", "cpt"] or tokenize("The F-1 student's OPT and CPT!") == ["f-1", "student", "opt", "cpt"]
    # (decide the possessive handling in the tokeniser; the assertion must pin what you chose: "student's" → "student" is the builder's choice recorded here)
    assert "the" not in tokenize("the and of") and tokenize("a an I") == []
def test_spec_parse_and_name_roundtrip():
    s = HistogramSpec.parse("w2v-hist-k256-idf-soft"); assert (s.k, s.idf, s.soft) == (256, True, True) and s.name == "w2v-hist-k256-idf-soft"
    assert HistogramSpec.parse("w2v-hist-k64").name == "w2v-hist-k64"
    with pytest.raises(ValueError): HistogramSpec.parse("w2v-hist-idf")
def test_raw_histogram_counts_nearest_bins_and_normalises():
    h = histogram(["opt", "employment", "travel", "unknownword"], ART, HistogramSpec(2, False, False))
    assert np.allclose(h, [2/3, 1/3])
def test_idf_histogram_weights_rare_words():
    h = histogram(["opt", "visa"], ART, HistogramSpec(2, True, False))
    assert np.allclose(h, [1.0/2, 3.0/2])          # weights divided by counted words (2)
def test_soft_histogram_spreads_over_nearest_centroids():
    h = histogram(["opt"], ART, HistogramSpec(2, False, True))
    assert h[0] > h[1] > 0 and np.isclose(h.sum(), 1.0)
def test_histogram_embedder_returns_zero_vector_for_unknown_words():
    e = HistogramEmbedder(ART, HistogramSpec(2, False, False))
    assert e.embed(["zzzz qqqq"]) == [[0.0, 0.0]] and e.dim == 2 and e.model == "w2v-hist-k2"
def test_histogram_embedder_output_is_unit_length():
    v = np.array(HistogramEmbedder(ART, HistogramSpec(2, True, True)).embed(["opt travel visa"])[0])
    assert np.isclose(np.linalg.norm(v), 1.0)
def test_load_artifacts_reports_missing_files(tmp_path):
    with pytest.raises(ArtifactsMissing, match="word2vec.kv"): load_artifacts(tmp_path, 256)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_histogram_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'histogram_embedding'`

- [ ] **Step 3: Install gensim, implement, wire the factory**

Run `.venv/bin/pip install gensim` then `.venv/bin/pip freeze > requirements.txt`. Implement per Interfaces; `load_artifacts` reads `word2vec.kv` with `gensim.models.KeyedVectors.load`, normalises rows, builds `vocab` from `kv.index_to_key`. `histogram` for `soft`: `nearest_centroids(vec, centroids, t=SOFT_T)` then `w = softmax(-d / SOFT_TAU)` times (idf weight or 1). In `embeddings.make_embedder`, the histogram branch calls `HistogramEmbedder.load(model, models_dir or settings.MODELS_DIR)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_histogram_embedding.py tests/test_embed_corpus.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/histogram_embedding.py backend/tests/test_histogram_embedding.py backend/embeddings.py backend/requirements.txt
git commit -m "Part 5: tokeniser, histogram embedder with idf and soft-assignment variants"
```

---

### Task 5: `train_histogram.py` and the real artefacts

**Files:**
- Create: `backend/train_histogram.py`, `data/models/` artefacts
- Test: `backend/tests/test_train_histogram.py`
- Modify: `backend/embed_corpus.py` (vocab-age warning)

**Interfaces:**
- Consumes: `kmeans.kmeans`, `histogram_embedding.tokenize/ARTIFACT_FILES`, `corpus_db.load_snapshot`-free chunk text access via `select(chunks.c.id, chunks.c.content)`.
- Produces:
  ```python
  def deterministic_hash(s: str) -> int                    # int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
  def train_word2vec(token_lists: list[list[str]], *, vector_size=100, window=5, min_count=2, epochs=25, seed=42) -> KeyedVectors
      # gensim Word2Vec(sg=1, workers=1, hashfxn=deterministic_hash, ...).wv
  def compute_idf(token_lists: list[list[str]]) -> dict[str, float]     # log((1+N)/(1+df)) + 1
  def train(chunk_texts: list[str], models_dir: Path, *, k_values=(256,), vector_size=100, seed=42) -> dict   # writes word2vec.kv, centroids-k{k}.npy per k, idf.json, meta.json; returns the meta dict
  def cluster_report(kv: KeyedVectors, centroids: np.ndarray, labels: np.ndarray, n_clusters=10, n_words=5) -> str
  def main() -> int   # argparse --k (repeatable, default 256) --vector-size 100; reads chunk texts from settings.SYNC_DATABASE_URL; prints vocab size, iterations, cluster sizes min/median/max and cluster_report
  ```
  `embed_corpus.main` after loading a histogram embedder compares `meta.json["chunk_count"]` with the current chunk count and logs a warning "Word2Vec trained on N chunks, corpus now has M; retrain with train_histogram.py" when they differ.

- [ ] **Step 1: Write the failing tests**

```python
TEXTS = [f"opt employment work authorization student {w}" for w in ("alpha", "beta", "gamma")] * 10 + ["travel visa passport reentry port of entry"] * 10
def test_train_writes_all_artifacts_and_is_deterministic(tmp_path):
    meta1 = train(TEXTS, tmp_path / "a", k_values=(4,), vector_size=16)
    meta2 = train(TEXTS, tmp_path / "b", k_values=(4,), vector_size=16)
    for f in ("word2vec.kv", "centroids-k4.npy", "idf.json", "meta.json"): assert (tmp_path / "a" / f).exists()
    assert meta1["chunk_count"] == len(TEXTS) and meta1["k_values"] == [4]
    assert np.array_equal(np.load(tmp_path / "a" / "centroids-k4.npy"), np.load(tmp_path / "b" / "centroids-k4.npy"))
def test_compute_idf_matches_formula():
    idf = compute_idf([["a", "b"], ["a"], ["c"]])
    assert np.isclose(idf["a"], math.log(4 / 3) + 1) and np.isclose(idf["c"], math.log(4 / 2) + 1)
def test_trained_artifacts_load_into_embedder(tmp_path):
    train(TEXTS, tmp_path, k_values=(4,), vector_size=16)
    e = HistogramEmbedder.load("w2v-hist-k4-idf", tmp_path)
    a, b = e.embed(["opt employment student", "travel visa passport"])
    assert np.dot(a, a) > 0 and np.dot(a, b) < 0.9      # different topics land in different bins
def test_embed_corpus_warns_when_vocab_predates_chunks(tmp_path, caplog): ...   # meta chunk_count 3 vs engine with 5 chunks → warning text contains "retrain"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_histogram.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_histogram'`

- [ ] **Step 3: Implement `train_histogram.py` and the warning**

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train_histogram.py -v`
Expected: 4 passed

- [ ] **Step 5: Train on the real corpus and embed the four variants**

Run:
```bash
.venv/bin/python train_histogram.py
for m in w2v-hist-k256 w2v-hist-k256-idf w2v-hist-k256-soft w2v-hist-k256-idf-soft; do .venv/bin/python embed_corpus.py --model $m; done
du -sh ../data/models
```
Expected: vocab between 4,000 and 12,000 words; k-means converges under 100 iterations; cluster report shows topical groups (e.g. a cluster with opt/cpt/training/employment); four `embedded 785` lines; `data/models` under 15 MB. Paste the cluster report into the commit message body.

- [ ] **Step 6: Commit**

```bash
git add backend/train_histogram.py backend/tests/test_train_histogram.py backend/embed_corpus.py ../data/models
git commit -m "Part 5: Word2Vec + k-means training script and trained artefacts (k=256)"
```

---

### Task 6: Retriever per model, thresholds, eval with negatives and calibration

**Files:**
- Modify: `backend/rag.py`, `backend/eval_retrieval.py`, `backend/settings.py`, `data/eval_questions.yaml`
- Modify: `backend/tests/test_rag.py`, `backend/tests/test_eval_retrieval.py`

**Interfaces:**
- Consumes: `corpus_db.load_snapshot(engine, model)/models_present`, `embeddings.make_embedder/is_histogram_model`.
- Produces:
  ```python
  # rag.py
  Hit.to_source() gains "score": round(score, 2)
  class Retriever: __init__(self, snapshot: CorpusSnapshot, expected_model: str)   # mismatch check on snapshot.model; `model` attribute
  # settings.py
  def threshold_for(model: str, thresholds: dict[str, float]) -> float   # OpenAI/default → RAG_MIN_SCORE; histogram → thresholds.get(model) else RAG_MIN_SCORE with log.warning
  def load_thresholds(models_dir: Path) -> dict[str, float]              # thresholds.json or {}
  # eval_retrieval.py
  @dataclass
  class EvalQuestion: question: str; expect: list[str]           # expect == [] means off-topic: must be rejected
  @dataclass
  class EvalResult: question: str; expect: list[str]; hit_ids: list[str]; top_score: float; ok: bool   # ok: on-topic → any expected id in hit_ids (threshold ignored); off-topic → top_score < threshold
  @dataclass
  class EvalSummary: model: str; recall: tuple[int, int]; answered: tuple[int, int]; rejected: tuple[int, int]; threshold: float
  def evaluate(questions, retriever, embedder, source_id_of, *, top_k=5, threshold: float) -> list[EvalResult]
  def summarise(model, results, threshold) -> EvalSummary
  def calibrate(results: list[EvalResult]) -> float    # midpoint between min top_score of on-topic hits and max top_score of off-topic; if overlapping, max off-topic top_score + 0.01 (prefers rejection)
  def main() -> int   # --model NAME (default settings.EMBEDDING_MODEL) | --all ; --calibrate writes data/models/thresholds.json for histogram models (never for OpenAI); prints per-question lines "OK  0.62 question -> ids", then "recall@5 (no threshold): x/y", "with threshold t: answered a/y, rejected r/n"; --all prints one summary line per model present; exit 1 if any on-topic MISS for --model, 0 for --all
  ```

- [ ] **Step 1: Write the failing tests**

`test_rag.py`: existing tests keep passing with `model="fake"` snapshots; add:
```python
def test_hit_source_includes_score(): ...   # to_source()["score"] == round(hits[0].score, 2)
def test_retriever_zero_query_yields_no_hits():
    r = Retriever(snapshot([[1, 0], [0, 1]]), expected_model="fake")
    assert r.search([0.0, 0.0], top_k=5, min_score=0.3) == []
```
`test_eval_retrieval.py`: replace with
```python
def test_evaluate_marks_hits_misses_and_rejections(): ...  # on-topic hit → ok, on-topic miss → not ok with hit_ids, off-topic with top_score 0.9 and threshold 0.5 → not ok, off-topic with top_score 0.2 → ok
def test_summarise_counts(): ...                            # recall (hits/on-topic), answered (top_score >= threshold among on-topic), rejected (off-topic below threshold)
def test_calibrate_midpoint_and_overlap():
    on = [EvalResult("q", ["a"], ["a"], 0.70, True), EvalResult("q", ["a"], ["a"], 0.60, True)]
    off = [EvalResult("x", [], ["a"], 0.40, True), EvalResult("x", [], ["a"], 0.30, True)]
    assert np.isclose(calibrate(on + off), 0.50)
    assert np.isclose(calibrate(on + [EvalResult("x", [], ["a"], 0.65, True)]), 0.66)
def test_threshold_for_prefers_model_entry_and_falls_back(caplog):
    assert threshold_for("w2v-hist-k256", {"w2v-hist-k256": 0.55}) == 0.55
    assert threshold_for("w2v-hist-k256", {}) == settings.RAG_MIN_SCORE and "threshold" in caplog.text.lower()
    assert threshold_for("text-embedding-3-small", {"text-embedding-3-small": 0.9}) == settings.RAG_MIN_SCORE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rag.py tests/test_eval_retrieval.py -v`
Expected: FAIL on the new names (`threshold_for`, `summarise`, `calibrate`, `score` key).

- [ ] **Step 3: Implement, and append the five off-topic questions**

Append to `data/eval_questions.yaml`:
```yaml
- {question: "What is the best pizza in Los Angeles?", expect: []}
- {question: "How do I change a flat bicycle tire?", expect: []}
- {question: "Who won the 2022 FIFA World Cup?", expect: []}
- {question: "Write a Python function that reverses a string.", expect: []}
- {question: "What is the weather like in Seattle in March?", expect: []}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass

- [ ] **Step 5: Real eval and calibration**

Run:
```bash
.venv/bin/python eval_retrieval.py --all --calibrate
.venv/bin/python eval_retrieval.py --all
```
Expected: a table with five rows (OpenAI + four histogram variants); OpenAI recall 15/16 or 16/16 with rejected 5/5 at 0.30; raw histogram recall lower than `idf` and `idf-soft`; `data/models/thresholds.json` has four entries. Copy the table into the ledger; it goes into the README in Task 9.

- [ ] **Step 6: Commit**

```bash
git add backend/rag.py backend/eval_retrieval.py backend/settings.py backend/tests ../data/eval_questions.yaml ../data/models/thresholds.json
git commit -m "Part 5: per-model thresholds, eval with off-topic rejection and calibration"
```

---

### Task 7: Backend: several embeddings, request field, meta

**Files:**
- Modify: `backend/app.py`
- Modify: `backend/tests/test_app_helpers.py`

**Interfaces:**
- Consumes: Task 6 `threshold_for/load_thresholds`, `corpus_db.models_present/load_snapshot`, `embeddings.make_embedder/is_histogram_model`, `rag.Retriever`.
- Produces:
  ```python
  @dataclass
  class EmbeddingService: model: str; label: str; retriever: Retriever; embedder: Embedder; threshold: float
  def embedding_label(model: str) -> str      # "OpenAI text-embedding-3-small" | "Word histogram (k=256)" + ", idf-weighted" / ", soft assignment" as applicable
  def build_services(engine, *, api_key: str, models_dir: Path, thresholds: dict[str, float]) -> dict[str, EmbeddingService]
      # for each model in models_present(engine): skip OpenAI name when api_key is empty (log.warning), skip histogram names whose artefacts are missing (log.warning); raise RuntimeError("no embedding model available; run embed_corpus.py") when empty
  def default_model(services: dict[str, EmbeddingService]) -> str   # settings.EMBEDDING_MODEL if present else first histogram key (sorted)
  class NewMessage(BaseModel): content: str; embedding: str | None = None
  Message.embedding: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)   # added at startup like `sources` (generalise ensure_messages_sources_column → ensure_messages_columns(conn) -> list[str] added)
  message_json(...) gains "embedding": message.embedding  (None for old rows)
  POST /api/messages: 400 {"detail": "unknown embedding 'x'; available: [...]"} before saving when the name is not served
  GET /api/meta → {"document_count", "chunk_count" (default model), "corpus_as_of", "llm_model", "default_embedding", "embeddings": [{"model", "label", "chunk_count", "threshold"}]}
  respond(user_id, username, question, message_id, model: str)   # uses services[model]
  ```

- [ ] **Step 1: Write the failing tests**

```python
def test_embedding_label(): assert embedding_label("w2v-hist-k256-idf-soft") == "Word histogram (k=256), idf-weighted, soft assignment"
def test_default_model_prefers_openai_then_first_histogram(): ...  # dict with both → OpenAI; only {"w2v-hist-k256-idf": ..., "w2v-hist-k256": ...} → "w2v-hist-k256"
def test_build_services_skips_unavailable_models(tmp_path, caplog):
    e = make_engine("sqlite://"); create_tables(e)   # one doc, two chunks, vectors for "text-embedding-3-small" and "w2v-hist-k256" (FakeEmbedder-made blobs)
    s = build_services(e, api_key="", models_dir=tmp_path, thresholds={})
    assert s == {} or "no embedding model available" ... → use pytest.raises(RuntimeError)
    # with api_key="sk-x": OpenAI present, histogram still skipped (no artefacts) and caplog mentions "word2vec.kv"
def test_message_json_defaults_embedding_to_none(): ...   # Message(..., embedding=None) → out["embedding"] is None and out["sources"] == []
def test_post_message_rejects_unknown_embedding():
    # app.dependency_overrides[current_user] = lambda: User(id=1, username="u"); app.state.services = {"text-embedding-3-small": <stub>}; TestClient(app) WITHOUT lifespan; POST {"content": "hi", "embedding": "nope"} → 400 and detail lists the available name
def test_ensure_messages_columns_adds_sources_and_embedding(): ...   # Part 1 table → returns ["sources", "embedding"]; second call → []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_app_helpers.py -v`
Expected: FAIL with `ImportError` on the new names.

- [ ] **Step 3: Implement**

`lifespan`: `services = await asyncio.to_thread(build_services, ...)`, `app.state.services`, `app.state.default_model`, `app.state.corpus_as_of` from the default model's snapshot. `post_message` validates `data.embedding or app.state.default_model` against `app.state.services` before `save_message`. `respond` passes `top_k=settings.RAG_TOP_K, min_score=service.threshold` and stores `embedding=model` on the bot row.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass

- [ ] **Step 5: Start the server and check meta**

Run: `.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000` then `curl -s localhost:8000/api/meta`
Expected: `embeddings` lists five models with chunk_count 785 each and the histogram thresholds from `thresholds.json`; `default_embedding` is `text-embedding-3-small`.

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/tests/test_app_helpers.py
git commit -m "Part 5: serve several embeddings, per-request choice, meta listing"
```

---

### Task 8: Frontend: embedding selector, labels and scores

**Files:**
- Modify: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

**Interfaces:**
- Consumes: Task 7 `/api/meta.embeddings/default_embedding`, `POST /api/messages {content, embedding}`, message `embedding` and source `score`.

- [ ] **Step 1: index.html**

In `.composer`, before the input: `<select id="embeddingSelect" aria-label="Embedding"></select>`. In `#welcome`, after the examples: `<p class="hint">Pick an embedding next to the box to compare retrieval: OpenAI vs. the word-histogram method (Part 5).</p>`.

- [ ] **Step 2: app.js**

- `loadMeta()` fills `#embeddingSelect` with one `<option value=model>label</option>` per entry, selects `localStorage.embedding` if it is among them else `default_embedding`; `change` stores the choice.
- `sendQuestion` posts `{content, embedding: embeddingSelect.value}`.
- `addMessage`: bot meta line becomes `OIS Assistant · <label> · <time>` when `m.embedding` is set (label looked up from the meta list, falling back to the raw name); each source `<li>` appends ` · score 0.62` when `s.score` is a number.

- [ ] **Step 3: styles.css**

`.composer select` styled like the input (same background, border, radius, 13 px, `max-width: 42%`); on `max-width: 480px` the composer wraps: select full width on its own row above input + Send (`flex-wrap: wrap`, select `flex-basis: 100%`).

- [ ] **Step 4: Syntax check and end-to-end with both embeddings**

Run `node --check ../frontend/app.js`. With MySQL, llama-server and uvicorn running, in Chrome (or the CDP script from the previous plan):
1. Welcome screen shows the select with five entries, OpenAI selected.
2. Ask "When can I apply for post-completion OPT?" with OpenAI: answer with sources carrying scores; meta line names the embedding.
3. Switch to `Word histogram (k=256), idf-weighted, soft assignment`, ask the same question: answer arrives; sources differ in order/scores.
4. Ask "What is the best pizza in Los Angeles?" with the histogram model: not-covered reply (if it answers instead, note the top score in the ledger; the threshold came from calibration and this is the finding to show).
5. Set `localStorage.embedding = "w2v-hist-k9"` and reload: the select shows the default, no console error.
6. 390 px width: the select sits on its own row, no horizontal scroll.
Record screenshots paths in the ledger.

- [ ] **Step 5: Commit**

```bash
git add ../frontend
git commit -m "Part 5: embedding selector, embedding label and scores in the UI"
```

---

### Task 9: Schema, docs, results table, push

**Files:**
- Modify: `sql/schema.sql`, `SETUP.md`, `README.md`, `.env.example`

- [ ] **Step 1: schema.sql**

Replace the `documents`/`chunks` definitions with the Task 1 shapes (no `embedding_model`, no `chunks.embedding`), add `chunk_embeddings`, add `embedding VARCHAR(64) NULL` to `messages`, and note that `ingest.py` resets the three corpus tables when it finds the old shape.

- [ ] **Step 2: SETUP.md and README.md**

SETUP step 6 becomes: `ingest.py` → `embed_corpus.py --model text-embedding-3-small` → `train_histogram.py` → `embed_corpus.py --model w2v-hist-k256-idf-soft` (and the other variants) → `eval_retrieval.py --all --calibrate`; note which steps need the OpenAI key (only the first embed) and that artefacts/thresholds are committed so a teammate can skip training. README: a "Part 5: word-histogram embedding" section with the method in five sentences, the four variants, the eval table from Task 6 step 5, and how to pick the embedding in the UI. `.env.example`: comment that `RAG_MIN_SCORE` applies to OpenAI and histogram thresholds live in `data/models/thresholds.json`.

- [ ] **Step 3: Verify docs**

Run every command in SETUP step 6 as written (they are idempotent and the OpenAI step skips 785 existing vectors).
Expected: each exits 0; eval prints the same table as the README.

- [ ] **Step 4: Commit and push**

```bash
git add ../sql/schema.sql ../SETUP.md ../README.md ../.env.example
git commit -m "Part 5: schema, setup steps and results for the word-histogram embedding"
git push origin Lab9
```

---

## Self-review notes

- Spec coverage: §4 data model → Task 1; §5 pipeline (ingest/embed split, training, variants) → Tasks 2-5; §6 retrieval/app (services, default, 400, meta, thresholds, UI) → Tasks 6-8; §7 eval (negatives, two numbers, --all, --calibrate, hashfxn) → Tasks 5-6; §8 verification → tests per task, real runs in Tasks 2, 5, 6, 7, 8; §9 builder items left open on purpose.
- Type consistency: `CorpusSnapshot.model` (Task 1) is what `Retriever` checks (Task 6) and what Task 2 patches in `rag.py`; `HistogramEmbedder.load(model, models_dir)` (Task 4) is what `make_embedder` (Task 2) and `build_services` (Task 7) call; `threshold_for/load_thresholds` live in `settings.py` (Task 6) and are consumed by Task 7; `EvalResult.top_score` (Task 6) feeds `calibrate`.
- Review Focus 1-5 each have a named test or a manual step in Tasks 4, 5, 6, 7, 8.
- Proportion: the plan carries signatures, test names and assertions; no function bodies beyond the k-means++ and soft-assignment formulas the tests do not pin.
