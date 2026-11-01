# Word-histogram embedding (Lab 9 Part 5) - design

Date: 2026-10-31. Builds on the OIS Assistant design
(`2026-10-31-ois-assistant-design.md`); everything not mentioned here stays
as that document says.

## 1. Intent

Lab 9 Part 5 asks for a second way to vectorise the context chunks: train
Word2Vec on the corpus, cluster the word vectors into K groups with custom
clustering code, and represent each chunk as a normalised K-bin histogram of
its words' nearest clusters. The assistant must be able to answer the same
question once with the OpenAI embedding and once with the histogram
embedding, so the video can show both retrievals side by side, with scores,
and show what the improvements to the histogram method changed.

## 2. Goals, non-goals, anti-goals

Goals
- Both embeddings coexist in the database and in the running app; the
  person picks one per question.
- The histogram method is implemented from the definition in the
  assignment, with K-means written by hand (numpy only), and has measurable
  variants ("improvements") compared on the existing retrieval eval.
- Trained artefacts are committed so a teammate can run the app without
  retraining; retraining is one command and deterministic.

Non-goals
- Replacing the OpenAI embedding. It stays the default.
- Changing the LLM, prompt, private threads, auth or the Android work.

Anti-goals
- A histogram path that silently falls back to OpenAI when something is
  missing. If the histogram model or vectors are absent, the option is not
  offered and the API refuses it.
- Hidden mixing of the two vector spaces (one FAISS index with both).

## 3. Constraints

- Same stack as before: Python 3.13 in `backend/.venv`, numpy, FAISS,
  MySQL, pinned requirements. New dependency: `gensim` (Word2Vec only).
- K-means is hand-written with numpy: no scikit-learn, no gensim/faiss
  clustering. Word2Vec itself may come from gensim (the assignment says
  "find and install Word2Vec").
- Training must be deterministic: fixed seeds, `workers=1` for Word2Vec,
  fixed iteration count for K-means.
- Committed artefacts under `data/models/` stay under 15 MB in total.

## 4. Data model

`chunks.embedding` and `documents.embedding_model` go away. Vectors move to
a new table so one chunk can carry several:

```
chunk_embeddings
  chunk_id   INT  FK chunks.id ON DELETE CASCADE
  model      VARCHAR(64)      e.g. text-embedding-3-small, w2v-hist-k256-idf-soft
  dim        INT
  embedding  BLOB             float32 little-endian, L2-normalised
  PRIMARY KEY (chunk_id, model)
```

`documents` keeps `sha256` for file-level idempotency. The corpus snapshot
the app loads is per model: `load_snapshot(engine, model)` returns only the
chunks that have a vector for that model, plus the list of all models
present (for `/api/meta`).

Migration: this is a dev database; `ingest.py` drops and recreates
`chunk_embeddings`/`chunks`/`documents` when it detects the old
`chunks.embedding` column (logged loudly), then re-ingests. Re-embedding
with OpenAI costs well under a dollar.

## 5. Pipeline

Two scripts replace the single `ingest.py` embedding step:

1. `ingest.py` - unchanged responsibilities minus embedding: manifest →
   extract → chunk → `documents`/`chunks` rows, idempotent by sha256,
   `--prune`. No API key needed.
2. `embed_corpus.py --model <name>` - for every chunk without a vector for
   that model, compute and insert one. `--model` is one of:
   - `text-embedding-3-small` (OpenAI, needs the key)
   - any histogram model name produced by step 3 below.
   `--refresh` recomputes existing vectors for that model.
3. `train_histogram.py` - the Part 5 training step:
   - Tokenise every chunk: lowercase, `[a-z][a-z0-9-]+` tokens, drop a
     small built-in English stop-word list and tokens shorter than 3.
   - Word2Vec (gensim): `vector_size=100`, `window=5`, `min_count=2`,
     `sg=1`, `epochs=25`, `seed=42`, `workers=1`. Save as
     `data/models/word2vec.kv` (KeyedVectors).
   - K-means by hand (`backend/kmeans.py`): k-means++ seeding with a seeded
     numpy RNG, Lloyd iterations until assignments stop changing or 100
     iterations, on L2-normalised word vectors (so Euclidean nearest ≈
     cosine nearest). Default `K=256`. Save `data/models/centroids-k256.npy`.
   - Also save `data/models/idf.json`: inverse document frequency of every
     vocabulary word over chunks (`log((1+N)/(1+df)) + 1`).
   - Print cluster diagnostics: sizes, five sample words of ten clusters,
     so the video can show what a cluster "means".

Chunk/query vectorisation (`backend/histogram_embedding.py`,
`HistogramEmbedder` implementing the existing `Embedder` protocol):
- Tokenise as above; words outside the Word2Vec vocabulary are skipped.
- Base method ("raw"): for each word, nearest centroid → `hist[k] += 1`;
  divide by the number of counted words.
- Improvements, each a flag in the model name so vectors never mix:
  - `idf`: add `idf[word]` instead of 1 (rare, topical words count more).
  - `soft`: distribute each word over the T=3 nearest centroids with
    weights `softmax(-d / tau)`, `tau=0.1`, instead of one bin.
  - Both: `w2v-hist-k256-idf-soft`.
- The vector is then L2-normalised so the existing cosine/FAISS retriever
  works unchanged. (The histogram's own "divide by word count" step is kept
  before normalisation so the stored vector is still the normalised
  histogram the assignment describes; the L2 step only makes cosine valid.)
- A chunk or query with zero in-vocabulary words gets the zero vector; the
  retriever already treats a zero query as "no hits" (score 0 < threshold).

Model names: `w2v-hist-k256`, `w2v-hist-k256-idf`, `w2v-hist-k256-soft`,
`w2v-hist-k256-idf-soft`. `embed_corpus.py` accepts any of them; the app
serves whichever are present in `chunk_embeddings`.

## 6. Retrieval and the app

- The backend builds one `Retriever` per model found in `chunk_embeddings`
  at startup, and one embedder per model (OpenAI for the OpenAI name,
  `HistogramEmbedder` for histogram names, loading `data/models/`).
  `/api/meta` lists `embeddings: [{"model", "label", "chunk_count"}]` and
  the default.
- `POST /api/messages` accepts an optional `embedding` field; missing →
  default (`text-embedding-3-small`); unknown → 400.
- The bot message stores `embedding` (model name) and each source gains
  `score` (cosine, 2 decimals). `messages` gets an `embedding VARCHAR(64)
  NULL` column (added at startup like `sources`).
- Thresholds are per model: `RAG_MIN_SCORE` stays for OpenAI; each
  histogram model has its own threshold stored next to its centroids in
  `data/models/thresholds.json` (written by the eval's `--calibrate` step,
  section 7). Histogram cosines are all non-negative and generally high
  (common-word bins dominate), so one shared threshold would not work.
- Starting without the OpenAI model (builder's call, for a teammate's
  machine without a key): the backend builds embedders only for the models
  that have vectors in `chunk_embeddings` and whose prerequisites exist
  (key for OpenAI, `data/models/` for histogram). `/api/meta.default` is
  `text-embedding-3-small` when it is available, else the first histogram
  model; a request without `embedding` uses that default. There is no
  fallback in the other direction: a request that names an unavailable
  model gets 400. Zero available models is a startup error.
- Frontend: a `<select>` in the composer listing the embeddings from
  `/api/meta`, remembered in `localStorage`; the bot message's meta line
  shows the embedding label; each source line shows its score. Everything
  else unchanged.

## 7. Eval

`data/eval_questions.yaml` gains five off-topic questions with
`expect: []`, meaning "must be answered not-covered" (every top score below
the model's threshold).

`eval_retrieval.py --model <name>` (default OpenAI) prints, per question,
OK/MISS, the top score and the hit ids, then two summary numbers:
- `recall@5` ignoring the threshold (the official number for comparing the
  variants; it is what the README table and the video quote; today's
  OpenAI value 15/16 is an expectation, not an acceptance gate), and
- `with threshold: answered x/y, rejected r/5` using the model's threshold
  (what the app would actually do).
`--all` prints that table for every model present. `--calibrate` picks each
histogram model's threshold as the cut (midpoint between neighbouring top
scores) that maximises answered on-topic hits plus rejected off-topic
questions, ties going to the higher cut, and writes
`data/models/thresholds.json`. (Revised during implementation: the earlier
"prefer rejection whenever the ranges overlap" rule let one off-topic
question that happened to mention Los Angeles push the threshold to 0.72 and
silence 12 of 16 on-topic questions.) Hashing in Word2Vec uses an
explicit deterministic `hashfxn` so results do not depend on
`PYTHONHASHSEED`.

## 8. Verification

- Unit tests (sqlite + fake vectors): kmeans converges on separable data
  and is deterministic; histogram of a known chunk against known centroids
  equals the hand-computed vector; idf and soft variants change the vector
  as expected; `load_snapshot(engine, model)` filters by model;
  `embed_corpus` is idempotent and `--refresh` recomputes; the API rejects
  an unknown embedding with 400 and records the chosen one.
- Real run: `ingest.py`, `train_histogram.py`, `embed_corpus.py` for the
  OpenAI model and all four histogram models, `eval_retrieval.py --all`.
  Expected: OpenAI 15/16 as today (unthresholded); raw histogram clearly
  lower; `idf` and `idf-soft` higher than raw; off-topic rejection reported
  per model. Record the table in the README. A histogram model that cannot
  reject off-topic questions without losing most on-topic ones is a
  legitimate finding to show, not a failure of the build.
- End to end: the same question asked twice in the UI with the two
  embeddings, both answered with sources and scores; off-topic question
  gets not-covered with both.

## 9. Left to the builder

- Stop-word list contents, tokeniser edge cases.
- Exact k-means++ implementation details and convergence tolerance.
- How `/api/meta` labels the models for the dropdown.
- Logging and progress output of the training script.
