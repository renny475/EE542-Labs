-- Optional: If you prefer manual SQL instead of SQLAlchemy auto-creation
CREATE DATABASE IF NOT EXISTS groupchat CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'chatuser'@'localhost' IDENTIFIED BY 'chatpass';
GRANT ALL PRIVILEGES ON groupchat.* TO 'chatuser'@'localhost';
FLUSH PRIVILEGES;

USE groupchat;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS messages (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  content TEXT NOT NULL,
  is_bot BOOLEAN DEFAULT FALSE,
  -- Part 4: cited sources of a bot answer, JSON list of {n, title, url, page, published, excerpt, score}
  sources JSON NULL,
  -- Part 5: embedding model the question was retrieved with (NULL for rows older than Part 5)
  embedding VARCHAR(64) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
-- Upgrading a Part 1 database: the backend adds the missing `sources` column at
-- startup. Old public group-chat rows stay but are only ever shown to their own
-- author; to start clean, drop the table before starting the Part 4 backend:
--   DROP TABLE IF EXISTS messages;

-- Part 4/5: curated corpus. Source of truth for the in-memory FAISS indexes the
-- backend rebuilds at startup (one per embedding model). documents and chunks are
-- written by backend/ingest.py, chunk_embeddings by backend/embed_corpus.py.
-- ingest.py drops and recreates these three tables when it finds the older Part 4
-- shape (vector stored on the chunk, embedding_model on the document).
CREATE TABLE IF NOT EXISTS documents (
  id INT AUTO_INCREMENT PRIMARY KEY,
  source_id VARCHAR(64) NOT NULL UNIQUE,
  title VARCHAR(255) NOT NULL,
  tier ENUM('ois','uscis','sevp','cfr') NOT NULL,
  url VARCHAR(512) NOT NULL,
  file_path VARCHAR(255) NOT NULL,
  sha256 CHAR(64) NOT NULL,
  fetched_on DATE NOT NULL,
  published DATE NULL,
  chunk_count INT NOT NULL,
  ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chunks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  document_id INT NOT NULL,
  chunk_index INT NOT NULL,
  page INT NULL,
  content TEXT NOT NULL,
  UNIQUE KEY uq_chunk (document_id, chunk_index),
  CONSTRAINT fk_chunk_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id INT NOT NULL,
  -- text-embedding-3-small (1536 dims) or a Part 5 histogram model such as w2v-hist-k256-idf-soft (256 dims)
  model VARCHAR(64) NOT NULL,
  dim INT NOT NULL,
  -- dim float32 values, little-endian, L2-normalised
  embedding BLOB NOT NULL,
  -- Part 5: fingerprint of the trained artefacts (data/models/meta.json) a histogram vector was
  -- computed with; NULL for OpenAI. embed_corpus.py re-embeds a model whose fingerprint changed.
  fingerprint VARCHAR(32) NULL,
  PRIMARY KEY (chunk_id, model),
  CONSTRAINT fk_embedding_chunk FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
