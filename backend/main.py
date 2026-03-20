import os
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf4llm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

from utils.database import init_db, clear_all, insert_document, insert_chunks, get_document_list
from utils.chunking import chunk_document
from utils.embeddings import embed_texts, search_chunks
from utils.llm import generate_sub_queries, build_context, generate_answer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
db = sqlite3.connect(Path(__file__).with_name("ffu.db"), check_same_thread=False)
db.execute("PRAGMA foreign_keys = ON")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
data_dir = Path("data")
extract = lambda path: pymupdf4llm.to_markdown(str(path), ignore_images=True, ignore_graphics=True)


@asynccontextmanager
async def lifespan(app):
    init_db(db)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/process")
def process():
    """Extract PDFs, chunk them, embed chunks, and store everything in SQLite."""
    logger.info("Processing documents...")
    db.execute("DELETE FROM chunks")
    db.execute("DELETE FROM documents")
    db.commit()
    paths = sorted(data_dir.rglob("*.pdf"))
    total_chunks = 0

    logger.info(f"Extracting {len(paths)} PDFs...")
    extracted: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(extract, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                content = future.result()
                extracted.append((path.name, content))
                logger.info(f"  Extracted {path.name}")
            except Exception as e:
                logger.error(f"  Failed to extract {path.name}: {e}")

    print(f"Extraction complete: {len(extracted)} documents, chunking...")

    # Chunk all documents first 
    doc_data: list[tuple[int, str, list[dict]]] = []
    all_chunk_texts: list[str] = []

    for filename, content in extracted:
        doc_id = insert_document(db, filename, content)
        chunks = chunk_document(content, filename)
        if not chunks:
            continue
        doc_data.append((doc_id, filename, chunks))
        all_chunk_texts.extend(c["text"] for c in chunks)

    print(f"Chunked into {len(all_chunk_texts)} total chunks, embedding...")

    # Embed all chunks in one batched call 
    try:
        all_embeddings = embed_texts(client, all_chunk_texts)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    print(f"Embedding complete, storing in database...")

    # Store chunks with their embeddings
    offset = 0
    for doc_id, filename, chunks in doc_data:
        doc_embeddings = all_embeddings[offset : offset + len(chunks)]
        insert_chunks(db, doc_id, chunks, doc_embeddings)
        total_chunks += len(chunks)
        offset += len(chunks)

    logger.info(f"Done! {len(extracted)} documents, {total_chunks} chunks total.")
    return {
        "status": "ok",
        "documents": len(extracted),
        "chunks": total_chunks,
    }


@app.post("/chat")
def chat(body: dict):
    """Multi-query retrieval + LLM answer generation."""
    question = body.get("message", "")
    history = body.get("history", [])

    if not question.strip():
        return {"response": "Please ask a question."}

    try:
        # Generate diverse sub-queries for better retrieval
        logger.info(f"Question: {question}")
        sub_queries = generate_sub_queries(client, question)

        # Retrieve top chunks for each sub-query
        all_chunks: dict[int, dict] = {}  # chunk_id -> chunk (dedup)
        for query in sub_queries:
            results = search_chunks(db, client, query, top_k=10)
            for chunk in results:
                chunk_id = chunk["chunk_id"]
                # Keep the highest score for each chunk
                if chunk_id not in all_chunks or chunk["score"] > all_chunks[chunk_id]["score"]:
                    all_chunks[chunk_id] = chunk

        # Sort by score and take top 15
        ranked_chunks = sorted(all_chunks.values(), key=lambda chunk: chunk["score"], reverse=True)[:15]
        logger.info(f"Retrieved {len(ranked_chunks)} unique chunks from {len(sub_queries)} queries")

        for chunk in ranked_chunks[:5]:
            logger.info(f"  [{chunk['score']:.3f}] {chunk['filename']} (chunk {chunk['chunk_index']})")

        # Build context and generate answer
        context = build_context(ranked_chunks)
        answer = generate_answer(client, question, context, history)

        # Build source references
        sources = []
        seen_files = set()
        for chunk in ranked_chunks:
            if chunk["filename"] not in seen_files:
                sources.append({
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "score": round(chunk["score"], 3),
                })
                seen_files.add(chunk["filename"])

        return {
            "response": answer,
            "sources": sources[:10],  # Top 10 unique source documents
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return {"response": f"Error: {e}"}
