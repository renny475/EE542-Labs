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
  -- Part 4: cited sources of a bot answer, JSON list of {n, title, url, page, published, excerpt}
  sources JSON NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
-- Upgrading a Part 1 database: the old messages table lacks `sources` and holds
-- public group-chat rows that the private assistant must not show. Drop and
-- recreate it (the backend does this once at startup when the column is missing):
--   DROP TABLE IF EXISTS messages;

-- Part 4: curated corpus. Source of truth for the in-memory FAISS index that the
-- backend rebuilds at startup. Written only by backend/ingest.py.
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
  embedding_model VARCHAR(64) NOT NULL,
  chunk_count INT NOT NULL,
  ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chunks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  document_id INT NOT NULL,
  chunk_index INT NOT NULL,
  page INT NULL,
  content TEXT NOT NULL,
  -- 1536 float32 values, little-endian (6144 bytes) for text-embedding-3-small
  embedding BLOB NOT NULL,
  UNIQUE KEY uq_chunk (document_id, chunk_index),
  CONSTRAINT fk_chunk_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
