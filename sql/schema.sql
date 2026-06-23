-- DDL for the support_tickets table.
-- Matches the raw dataset columns used by src/data_cleaning.py.
-- SQLite-compatible; run with: sqlite3 support_tickets.db < sql/schema.sql

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id             TEXT    PRIMARY KEY,
    created_date          TEXT    NOT NULL,
    resolved_date         TEXT,
    priority              TEXT    NOT NULL DEFAULT 'Unknown',
    category              TEXT    NOT NULL DEFAULT 'Unknown',
    channel               TEXT    NOT NULL DEFAULT 'Unknown',
    team                  TEXT    NOT NULL DEFAULT 'Unknown',
    status                TEXT    NOT NULL DEFAULT 'Unknown',
    escalated             INTEGER NOT NULL DEFAULT 0,
    customer_satisfaction INTEGER,
    customer_message      TEXT    DEFAULT 'No message provided'
);

CREATE INDEX IF NOT EXISTS idx_tickets_priority     ON support_tickets (priority);
CREATE INDEX IF NOT EXISTS idx_tickets_team         ON support_tickets (team);
CREATE INDEX IF NOT EXISTS idx_tickets_created_date ON support_tickets (created_date);
