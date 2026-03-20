"""
SQLite schema and helper functions for documents and chunks.
"""

import sqlite3
import struct
from typing import Optional

import numpy as np


def init_db(db: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
    """)
    db.commit()


def clear_all(db: sqlite3.Connection) -> None:
    """Delete all documents and chunks."""
    db.execute("DELETE FROM chunks")
    db.execute("DELETE FROM documents")
    db.commit()


def insert_document(db: sqlite3.Connection, filename: str, content: str) -> int:
    """Insert a document and return its id."""
    cur = db.execute(
        "INSERT INTO documents(filename, content) VALUES(?, ?)",
        (filename, content),
    )
    db.commit()
    return cur.lastrowid


def insert_chunks(
    db: sqlite3.Connection,
    document_id: int,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Insert chunks with their embeddings for a document."""
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        emb_blob = _embedding_to_blob(emb)
        rows.append((document_id, chunk["chunk_index"], chunk["text"], emb_blob))
    db.executemany(
        "INSERT INTO chunks(document_id, chunk_index, content, embedding) VALUES(?, ?, ?, ?)",
        rows,
    )
    db.commit()


def get_all_embeddings(db: sqlite3.Connection) -> list[tuple[int, np.ndarray]]:
    """Return all (chunk_id, embedding_vector) pairs."""
    rows = db.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
    result = []
    for chunk_id, blob in rows:
        if blob:
            result.append((chunk_id, _blob_to_embedding(blob)))
    return result


def get_chunks_by_ids(db: sqlite3.Connection, chunk_ids: list[int]) -> list[dict]:
    """Fetch chunk records by their IDs, preserving order."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = db.execute(
        f"SELECT c.id, c.document_id, c.chunk_index, c.content, d.filename "
        f"FROM chunks c JOIN documents d ON c.document_id = d.id "
        f"WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()

    # Build lookup and preserve requested order
    lookup = {r[0]: r for r in rows}
    result = []
    for cid in chunk_ids:
        if cid in lookup:
            r = lookup[cid]
            result.append({
                "chunk_id": r[0],
                "document_id": r[1],
                "chunk_index": r[2],
                "content": r[3],
                "filename": r[4],
            })
    return result


def get_document_list(db: sqlite3.Connection) -> list[dict]:
    """Return a summary of all documents."""
    rows = db.execute(
        "SELECT d.id, d.filename, COUNT(c.id) as chunk_count "
        "FROM documents d LEFT JOIN chunks c ON d.id = c.document_id "
        "GROUP BY d.id ORDER BY d.id"
    ).fetchall()
    return [{"id": r[0], "filename": r[1], "chunk_count": r[2]} for r in rows]


# blob helpers 

def _embedding_to_blob(embedding: list[float]) -> bytes:
    """Pack a list of floats blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Unpack a blob back into a numpy array."""
    n = len(blob) // 4  
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
