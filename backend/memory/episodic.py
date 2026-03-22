"""
Episodic memory — stores past Q&A interactions and retrieves relevant ones
for new queries using cosine similarity, COALA framework.

Each conversation (question + answer) is embedded and stored. When a new
question comes in, search past interactions for relevant context,
allowing the system to build on previous answers.
"""

import sqlite3
import struct

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def save_interaction(
    db: sqlite3.Connection,
    client: OpenAI,
    question: str,
    answer: str,
) -> None:
    """Embed and store a Q&A pair in episodic memory."""
    combined = f"Q: {question}\nA: {answer}"
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[combined])
    embedding = response.data[0].embedding
    blob = struct.pack(f"{len(embedding)}f", *embedding)

    db.execute(
        "INSERT INTO conversations(question, answer, embedding) VALUES(?, ?, ?)",
        (question, answer, blob),
    )
    db.commit()


def recall(
    db: sqlite3.Connection,
    client: OpenAI,
    query: str,
    top_k: int = 3,
    min_score: float = 0.6,
) -> list[dict]:
    """Search episodic memory for past interactions relevant to the query."""
    rows = db.execute(
        "SELECT id, question, answer, embedding FROM conversations WHERE embedding IS NOT NULL"
    ).fetchall()

    if not rows:
        return []

    # Embed the query
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Unpack stored embeddings into a matrix
    ids, questions, answers, blobs = zip(*rows)
    n_floats = len(blobs[0]) // 4
    emb_matrix = np.array(
        [struct.unpack(f"{n_floats}f", b) for b in blobs], dtype=np.float32
    )

    # Normalize and compute cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_matrix = emb_matrix / norms

    similarities = np.dot(emb_matrix, query_vec)

    # Filter by minimum score and take top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in top_indices:
        score = float(similarities[i])
        if score < min_score:
            break
        results.append({
            "question": questions[i],
            "answer": answers[i],
            "score": score,
        })

    return results


def format_episodic_context(memories: list[dict]) -> str:
    """Format retrieved memories into a context string for the LLM."""
    if not memories:
        return ""

    parts = []
    for memory in memories:
        parts.append(f"[Previous Q&A, relevance: {memory['score']:.2f}]\nQ: {memory['question']}\nA: {memory['answer']}")

    return "\n\n---\n\n".join(parts)
