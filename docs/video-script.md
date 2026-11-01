# Lab 9 video script (Parts 4 and 5)

Read the English aloud. Lines marked 提示 are what to show on screen.

Before recording (提示): have these running in separate terminals: `cloudflared tunnel ...`,
`llama-server ...`, `uvicorn app:app ...`; the Android emulator open with the app installed;
VS Code open on `groupchat_app_src/`; a browser tab on http://localhost:8000 logged in.
Total length target: 8 to 10 minutes.

---

## 0. Opening (30 s)

提示: 鏡頭或螢幕先停在 VS Code 的檔案樹 `groupchat_app_src/`。

"Hi, this is my Lab 9 video. I built the USC OIS Assistant: a question-answering app for
international students on F-1 status. It answers questions about CPT, OPT, the STEM OPT
extension, maintaining status and travel, using only a curated set of documents from USC's
Office of International Services and from USCIS, ICE and the Code of Federal Regulations.
It cites its sources, and it refuses questions the documents do not cover.

I will cover four things: the source code, real examples of the data and its embeddings and
how retrieval uses them, the word-histogram embedding I built by hand for Part 5 and how I
improved it, and finally the working website and Android app."

## 1. Source code (2 min)

提示: 在 VS Code 依序點開這些檔案，每個停 10 到 20 秒，不要逐行念。

"The repository is on GitHub, branch Lab9. The backend is Python with FastAPI and MySQL."

提示: 打開 `data/sources.yaml`。

"Everything starts from this manifest. It lists 49 documents: 29 pages from the USC OIS
website, 13 from USCIS including nine chapters of the Policy Manual, 6 from ICE and Study in
the States, and the full text of 8 CFR 214.2 paragraph f, the regulation for F-1 students.
Each entry has a stable id, a title, a tier, the URL, and where the file lives."

提示: 打開 `backend/fetch_sources.py`，再 `backend/extract.py`。

"fetch_sources.py downloads the federal documents with a self-identifying user agent and a
two-second pause between requests. The OIS pages block scripts, so those were saved as PDFs
by hand. extract.py turns PDFs and HTML into text: it strips the print headers that repeat on
every page, repairs words broken across lines, and for the regulation it cuts out just
paragraph f out of a 181-page PDF."

提示: 打開 `backend/chunking.py` 和 `backend/ingest.py`。

"chunking.py splits each document into chunks of 1,000 characters with a 200-character
overlap, remembering the page each chunk came from. ingest.py writes documents and chunks
into MySQL. It is idempotent: it hashes each file, so re-running it only touches documents
that changed."

提示: 打開 `backend/embed_corpus.py`，然後 `sql/schema.sql` 捲到 chunk_embeddings。

"embed_corpus.py computes the vectors, one model at a time. Vectors live in their own
table, chunk_embeddings, keyed by chunk and model, so the same chunk can have an OpenAI
vector and my own histogram vector side by side."

提示: 打開 `backend/rag.py`。

"rag.py is the retrieval and answering layer. The retriever holds a FAISS index over the
normalised vectors, so inner product equals cosine similarity. The answer function embeds
the question, takes the top five chunks above a threshold, builds a prompt with numbered
sources, and sends it to a local Llama 3.2 3B model running in llama.cpp. If nothing
clears the threshold it returns a fixed not-covered message without calling the model."

提示: 打開 `backend/app.py`，捲到 `respond` 和 `websocket_chat`。

"app.py is the web backend. Each user has a private thread. A question is saved, echoed to
the user's own WebSocket, and answered in a background task. Login uses JWT tokens."

提示: 打開 `backend/tests/` 資料夾，跑 `cd backend && .venv/bin/python -m pytest -q`。

"There are 97 automated tests covering extraction, chunking, the store, retrieval, the
k-means, the histogram embedding and the API. They all run against SQLite and a fake
embedder, so they need no API key."

## 2. Contextual data, embeddings, and how RAG uses them (2.5 min)

提示: 在終端機跑：
`cd backend && .venv/bin/python show_embedding_example.py --source ois-post-completion-opt --chunk 0 --question "When can I apply for post-completion OPT?"`

"Let me show a real example. This script prints one chunk from the corpus and everything
derived from it."

提示: 指著 CHUNK 區塊。

"This is chunk zero of the OIS Post-Completion OPT page: about a thousand characters of
text, with its page number and URL kept as provenance, so the app can cite it."

提示: 指著 OpenAI 那兩行。

"The first embedding is OpenAI text-embedding-3-small: 1,536 numbers, normalised to length
one. These are the first eight values. Stored as float32 bytes in MySQL."

提示: 指著 Word histogram 區塊。

"The second embedding is the one I built for Part 5. The chunk is tokenised into 89
words, all of them in my Word2Vec vocabulary. Each word is assigned to one of 256 word
clusters and the chunk becomes a 256-bin histogram. You can see the largest bins and the
words closest to each cluster centre: bin 51 is the OPT and STEM-extension cluster, bin 19
holds unemployment and counting words. The stored vector has 256 dimensions, mostly zeros."

提示: 指著 RETRIEVAL 區塊。

"And this is how retrieval uses them. For the question 'When can I apply for
post-completion OPT?' both embeddings search their own index. OpenAI returns the USCIS OPT
page and the Policy Manual chapter with cosine 0.76. The histogram model returns the STEM
OPT Extension page first, then the OPT pages, with slightly different scores. Each model
has its own not-covered threshold: 0.30 for OpenAI, 0.54 for the histogram model."

提示: 切到瀏覽器 localhost:8000，點範例問題「When can I apply for post-completion OPT?」，等回答，展開 Sources。

"In the app, the retrieved chunks become numbered sources in the prompt. The model's answer
cites them, and the Sources panel shows each source with its link, page and similarity
score, so a student can check the original text."

提示: 問「What is the best pizza in Los Angeles?」。

"If I ask something off topic, nothing clears the threshold and the app answers with the
not-covered message and no sources."

## 3. Word-histogram embedding and improvements (3 min)

提示: 打開 `backend/train_histogram.py`。

"Part 5 asked for a second way to vectorise the chunks: train Word2Vec on the corpus,
cluster the word vectors into K groups, and represent a chunk by a normalised histogram of
its words' clusters.

train_histogram.py does the training. I tokenise all 785 chunks, lowercase, drop stop
words and short tokens, and train gensim Word2Vec with skip-gram, 100 dimensions, window
5, 25 epochs, a fixed seed, one worker and a deterministic hash function, so the result is
reproducible. The corpus is small, so the vocabulary is 2,483 words."

提示: 打開 `backend/kmeans.py`。

"The clustering is my own code in numpy, no scikit-learn. kmeans_plusplus_init picks the
first centroid uniformly and each next one with probability proportional to the squared
distance to the nearest chosen centroid. Then Lloyd's algorithm alternates assignment and
mean update until the labels stop changing. Empty clusters are re-seeded at the farthest
point. With K equal to 256 it converges in 13 iterations."

提示: 在終端機跑 `.venv/bin/python train_histogram.py`（約 10 秒），指著 cluster report。

"Here is the cluster report: for example one cluster holds engineering fields from the STEM
degree list, another holds biology words, another holds the visa-category codes."

提示: 打開 `backend/histogram_embedding.py`，捲到 `histogram` 函式。

"histogram_embedding.py turns text into the vector. The basic method: for each word, find
the nearest centroid and add one to that bin, then divide by the number of words. That is
the normalised K-dimensional histogram from the assignment.

I added two improvements as switches in the model name. First, idf weighting: instead of
adding one for every word, I add the word's inverse document frequency, so rare, topical
words like 'unemployment' count more than words that appear in every chunk. Second, soft
assignment: instead of one bin per word, each word is spread over its three nearest
centroids with softmax weights over the distances, which smooths the histogram when a word
sits between two clusters. Finally the vector is L2-normalised so cosine similarity works
with the same FAISS retriever."

提示: 在終端機跑 `.venv/bin/python eval_retrieval.py --all`，指著表格。也可以開 README 的表格。

"To measure the improvements I wrote a retrieval eval: 16 real student questions with the
documents that answer them, plus 5 off-topic questions that must be rejected.

Recall at five, the share of questions where a correct document is in the top five:
OpenAI scores 15 out of 16. The plain histogram scores 12. idf alone and soft alone also
score 12, but idf plus soft reaches 13, so that combination is the best variant and the one
the app uses by default for Part 5.

The weakness of the histogram method is telling off-topic questions apart. Its cosines are
all non-negative and dominated by common-word bins, so an off-topic question that happens
to share vocabulary with the corpus, like 'best pizza in Los Angeles', matches the OIS page
that lists Social Security offices in Los Angeles. The calibration therefore chooses each
model's threshold to maximise answered on-topic questions plus rejected off-topic ones,
and the language model's instructions catch the remaining leaks."

提示: 回到瀏覽器，下拉選單切到「Word histogram (k=256), idf-weighted, soft assignment」，再問一次「When can I apply for post-completion OPT?」，展開 Sources 比較分數。

"In the app you can switch embeddings per question. Here is the same question with the
histogram embedding: the answer is still correct and cited, but the retrieved sources and
their scores differ from the OpenAI run above, which is exactly the comparison the video
asks for."

## 4. Working website and Android app (1.5 min)

提示: 瀏覽器視窗用 DevTools 切到手機尺寸或直接縮窄，展示登入、歡迎卡、發問、來源摺疊。

"The website is a single page app in plain HTML and JavaScript. It has a disclaimer with
the corpus date, example questions, a private thread per account, the embedding selector,
and collapsible sources with scores. It is responsive down to a 390-pixel phone."

提示: 切到 Android 模擬器，點開 USC OIS Assistant，登入，問一題。

"For the Android app I used a Trusted Web Activity. The website is published over https
through a Cloudflare tunnel; the Android project points at that URL and declares the
site in its asset statements; the server publishes assetlinks.json with the app's package
name and signing-key fingerprint. When Chrome verifies that link, it opens the site full
screen with no address bar, as you can see here, and the same login, questions and sources
work on the phone."

提示: 可以秀一下 `twa_android_src/app/src/main/res/values/strings.xml` 和 `frontend/.well-known/assetlinks.json`。

## 5. Closing (20 s)

"To sum up: a curated corpus of 49 documents and 785 chunks, two interchangeable
embeddings, a hand-written k-means, a local Llama model, a cited and bounded answer
pipeline, 97 tests, and the same app on the web and on Android. Thank you."
