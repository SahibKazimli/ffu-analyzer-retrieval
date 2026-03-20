"""
Embedding generation and similarity search using OpenAI embeddings.
"""

import logging
from typing import Optional

import numpy as np
from openai import OpenAI

from utils.database import get_all_embeddings, get_chunks_by_ids

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI's embedding API."""
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def search_chunks(
    db,
    client: OpenAI,
    query: str,
    top_k: int = 15,
) -> list[dict]:
    """Embed *query*, compute cosine similarity against all stored chunks, and return the top-k most relevant chunks with scores."""
    # Embed the query
    query_embedding = embed_texts(client, [query])[0]
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Load all chunk embeddings from SQLite
    all_embeddings = get_all_embeddings(db)
    if not all_embeddings:
        return []

    chunk_ids = [cid for cid, _ in all_embeddings]
    emb_matrix = np.stack([emb for _, emb in all_embeddings])

    # Normalise rows
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    emb_matrix = emb_matrix / norms

    # Cosine similarities via matrix-vector product
    similarities = np.dot(emb_matrix, query_vec)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_chunk_ids = [chunk_ids[i] for i in top_indices]
    top_scores = [float(similarities[i]) for i in top_indices]

    # Fetch chunk content from DB
    chunks = get_chunks_by_ids(db, top_chunk_ids)

    # Attach scores
    score_lookup = dict(zip(top_chunk_ids, top_scores))
    for chunk in chunks:
        chunk["score"] = score_lookup.get(chunk["chunk_id"], 0.0)

    return chunks
