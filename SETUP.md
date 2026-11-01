# Setup (macOS, tested 2026-10-31)

1. `brew install mysql && brew services start mysql`
2. `mysql -u root < sql/schema.sql`

   Creates the database, the `chatuser` account and every table. `schema.sql` is the
   reference schema; `ingest.py` and the backend also create any missing table
   themselves. Upgrading a Part 1 database: the backend adds the missing
   `messages.sources` column at startup; run `DROP TABLE messages;` first only if
   you also want to discard the old public group-chat rows.
3. `cd backend && python -m venv .venv && source .venv/bin/activate`
4. `pip install -r requirements.txt greenlet "bcrypt<4.1"`

   `greenlet` is required by SQLAlchemy async, and `bcrypt` 5.x is incompatible with `passlib`.
5. `cp ../.env.example ../.env`, then set:
   - `JWT_SECRET` (generate one with `python -c "import secrets; print(secrets.token_hex(32))"`)
   - `OPENAI_API_KEY`, used only for `text-embedding-3-small` embeddings (ingest and the
     question embedding at query time). The answers come from the local model below.
6. Corpus (run from `backend/`, with MySQL up):

   ```bash
   .venv/bin/python fetch_sources.py            # downloads the USCIS / SEVP / CFR documents
   # OIS pages block scripts: open each ois.usc.edu URL listed in data/sources.yaml in Chrome,
   # print to PDF and save it as data/ois/<slug>.pdf, or run  bash scripts/save_ois_pdfs.sh
   .venv/bin/python fetch_sources.py --check    # lists OIS files still missing
   .venv/bin/python ingest.py                   # extract + chunk into MySQL (idempotent, no key needed)
   .venv/bin/python embed_corpus.py --model text-embedding-3-small   # OpenAI vectors (needs OPENAI_API_KEY)
   .venv/bin/python train_histogram.py          # Part 5: Word2Vec + k-means -> data/models/ (committed, optional)
   .venv/bin/python embed_corpus.py --model w2v-hist-k256-idf-soft   # Part 5 vectors; also: w2v-hist-k256, -idf, -soft
   .venv/bin/python eval_retrieval.py --all --calibrate   # recall@5 per model, writes data/models/thresholds.json
   ```

   Only the `text-embedding-3-small` step needs the OpenAI key; the histogram
   models work offline. `data/models/` (trained artefacts and thresholds) is
   committed, so a teammate can skip `train_histogram.py` and `--calibrate`.
   The app serves every model that has vectors; without an OpenAI key it serves
   the histogram models only. Restart uvicorn after every embed; the backend
   loads the vectors into memory at startup.
7. llama.cpp: `brew install llama.cpp` (or build from source into `../llama.cpp/build/bin/`), then start the server from the `lab9` directory:

   ```bash
   ./llama.cpp/build/bin/llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
     --port 8001 --alias llama-3.2-3b-instruct -c 8192 -np 2 --api-key <key>
   ```

   `-c 8192` leaves room for five retrieved chunks plus history. The 8B model will freeze
   machines with <=16 GB RAM; use the 3B model.
8. In `.env`, `LLM_API_KEY` must match the `--api-key` value above and `LLM_MODEL` must match `--alias`.
9. From `backend/`: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`, then open http://localhost:8000

   `curl localhost:8000/api/meta` shows the document and chunk counts the server loaded.
10. Tests: `cd backend && .venv/bin/python -m pytest`

## Android app (Lab 9 Part 2 wrapper, Trusted Web Activity)

`twa_android_src/` wraps the deployed website in an Android app (no Kotlin or
Java code: `android-browser-helper` opens the site full screen once Digital
Asset Links verify). Needs Android Studio (or JDK 17 + Android SDK).

1. Publish the site over https. Quick demo: `cloudflared tunnel --url http://localhost:8000 --protocol http2`
   prints a `https://<random>.trycloudflare.com` URL (it changes every run; a
   named tunnel on your own domain gives a stable one).
2. Signing key (once): `keytool -genkeypair -v -keystore ~/ois-twa.jks -alias ois-twa -keyalg RSA -keysize 2048 -validity 3650 -storepass android -keypass android -dname "CN=USC OIS Assistant"`
   then `keytool -list -v -keystore ~/ois-twa.jks -alias ois-twa -storepass android -keypass android | grep SHA256`.
   Keep the keystore out of git; `export OIS_TWA_KEYSTORE=~/ois-twa.jks` makes Gradle sign debug builds with it.
3. From the repo root: `bash scripts/set_twa_url.sh https://<your-domain> <SHA256 fingerprint>`.
   This writes `defaultUrl` and `asset_statements` into `twa_android_src/app/src/main/res/values/strings.xml`
   and the fingerprint into `frontend/.well-known/assetlinks.json` (served by the backend at
   `/.well-known/assetlinks.json`; restart uvicorn, then check `curl https://<your-domain>/.well-known/assetlinks.json`).
4. Build: open `twa_android_src/` in Android Studio and Run, or `cd twa_android_src && ./gradlew assembleDebug`
   (APK at `app/build/outputs/apk/debug/app-debug.apk`). Install on a phone or an emulator
   (Android Studio AVD, LDPlayer, BlueStacks) that has Chrome.
5. Launch: Chrome opens the site without a URL bar when the asset links verify; with a
   URL bar it means the fingerprint or package name (`edu.usc.ee542.oisassistant`) does not
   match, or the URL is not https. Changing the tunnel URL means re-running step 3 and rebuilding.
