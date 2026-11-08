-- Manual MySQL 8+ setup. The app also creates these tables on startup.
-- Run as a MySQL admin: mysql -u root -p < sql/schema.sql
-- Change the password below and put the same one in DATABASE_URL in .env.
CREATE DATABASE IF NOT EXISTS aftercare CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'aftercare'@'localhost' IDENTIFIED BY 'change_me';
GRANT ALL PRIVILEGES ON aftercare.* TO 'aftercare'@'localhost';
FLUSH PRIVILEGES;

USE aftercare;

CREATE TABLE IF NOT EXISTS users (
  id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(50) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY ix_users_username (username)
);

CREATE TABLE IF NOT EXISTS discharge_documents (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'pending_review',  -- pending_review | confirmed | archived
  filename VARCHAR(255) NULL,
  text LONGTEXT NOT NULL,
  discharge_date DATE NULL,
  anchor_date DATE NULL,
  date_source VARCHAR(20) NULL,                           -- document | confirmation
  unverified JSON NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  confirmed_at DATETIME NULL,
  active_user_id INT NULL,                                -- = user_id while pending_review/confirmed, NULL when archived
  PRIMARY KEY (id),
  UNIQUE KEY uq_discharge_documents_active_user (active_user_id),
  KEY ix_discharge_documents_user_id (user_id),
  FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS extracted_items (
  id INT NOT NULL AUTO_INCREMENT,
  document_id INT NOT NULL,
  category VARCHAR(50) NOT NULL,
  text TEXT NOT NULL,
  source_quote TEXT NOT NULL,
  details JSON NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'active',           -- active | rejected
  PRIMARY KEY (id),
  KEY ix_extracted_items_document_id (document_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checklist_items (
  id INT NOT NULL AUTO_INCREMENT,
  document_id INT NOT NULL,
  extracted_item_id INT NULL,
  label VARCHAR(500) NOT NULL,
  slot VARCHAR(20) NOT NULL DEFAULT 'anytime',            -- morning | midday | evening | bedtime | anytime
  start_date DATE NOT NULL,
  end_date DATE NULL,
  one_time BOOLEAN NOT NULL DEFAULT FALSE,
  approximate BOOLEAN NOT NULL DEFAULT FALSE,
  undated BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (id),
  KEY ix_checklist_items_document_id (document_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE,
  FOREIGN KEY (extracted_item_id) REFERENCES extracted_items (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS checklist_completions (
  id INT NOT NULL AUTO_INCREMENT,
  checklist_item_id INT NOT NULL,
  day DATE NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_completion_item_day (checklist_item_id, day),
  FOREIGN KEY (checklist_item_id) REFERENCES checklist_items (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id INT NOT NULL AUTO_INCREMENT,
  document_id INT NOT NULL,
  role VARCHAR(20) NOT NULL,                              -- user | assistant
  content TEXT NOT NULL,
  payload JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_chat_messages_document_id (document_id),
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS care_team_questions (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  document_id INT NULL,
  question TEXT NOT NULL,
  reason TEXT NOT NULL,
  source VARCHAR(30) NOT NULL,                            -- agent | rejected_item | validation_fallback
  resolved BOOLEAN NOT NULL DEFAULT FALSE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_care_team_questions_user_id (user_id),
  FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
  FOREIGN KEY (document_id) REFERENCES discharge_documents (id) ON DELETE SET NULL
);
