import json
import os
import logging
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf4llm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv

from utils.database import init_db, clear_all, insert_document, insert_chunks, get_document_list
from utils.chunking import chunk_document
from utils.embeddings import embed_texts, search_chunks
from utils.llm import generate_sub_queries, build_context, generate_answer
from eval.judge import refine_with_feedback
from memory.episodic import save_interaction, recall, format_episodic_context

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
db = sqlite3.connect(Path(__file__).with_name("ffu.db"), check_same_thread=False)
db.execute("PRAGMA foreign_keys = ON")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
data_dir = Path("data")
def extract(path):
    return pymupdf4llm.to_markdown(str(path), ignore_images=True, ignore_graphics=True)


@asynccontextmanager
async def lifespan(app):
    init_db(db)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _sse(data: dict) -> str:
    """Format a dict to send progress to the browser"""
    return f"data: {json.dumps(data)}\n\n"


@app.post("/api/process")
def process():
    """Extract PDFs, chunk them, embed chunks, and store in SQLite. """

    # Generator that yields SSE events, each yield is like a print() to the browser
    def generate():
        yield _sse({"type": "log", "msg": "Processing documents..."})

    def stream():
        try:
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM documents")
            db.commit()

            paths = sorted(data_dir.rglob("*.pdf"))
            yield _sse({"type": "log", "msg": f"Extracting {len(paths)} PDFs..."})

            extracted = []
            with ProcessPoolExecutor(max_workers=6) as pool:
                futures = {pool.submit(extract, path): path for path in paths}
                for future in as_completed(futures):
                    name = futures[future].name
                    extracted.append((name, future.result()))
                    yield _sse({"type": "log", "msg": f"  Extracted {name}"})

            yield _sse({"type": "log", "msg": "Chunking..."})

            doc_chunks = []
            all_texts = []
            for filename, content in extracted:
                doc_id = insert_document(db, filename, content)
                chunks = chunk_document(content, filename)
                if not chunks:
                    continue
                doc_chunks.append((doc_id, chunks))
                all_texts.extend(c["text"] for c in chunks)
                yield _sse({"type": "log", "msg": f"  Chunked {filename} → {len(chunks)} chunks"})

            yield _sse({"type": "log", "msg": f"{len(all_texts)} chunks. Embedding..."})
            all_embeddings = embed_texts(client, all_texts)

            yield _sse({"type": "log", "msg": "Storing in database..."})
            total, offset = 0, 0
            for doc_id, chunks in doc_chunks:
                insert_chunks(db, doc_id, chunks, all_embeddings[offset:offset + len(chunks)])
                total += len(chunks)
                offset += len(chunks)

            yield _sse({"type": "done", "documents": len(extracted), "chunks": total})
        except Exception as e:
            yield _sse({"type": "error", "error": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/chat")
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

        # Recall relevant past interactions from episodic memory
        memories = recall(db, client, question, top_k=3)
        episodic_context = format_episodic_context(memories)

        # Build context from document chunks + episodic memory
        doc_context = build_context(ranked_chunks)
        context = doc_context
        if episodic_context:
            context += "\n\nPREVIOUS RELEVANT INTERACTIONS:\n" + episodic_context

        # Generate answer, judge it, refine up to 2 times if score < 0.7
        answer, evaluation, refine_count = refine_with_feedback(
            client, question, context, history,
            generate_fn=generate_answer,
            max_attempts=2, threshold=0.7,
        )

        # Save this interaction to episodic memory for future recall
        save_interaction(db, client, question, answer)

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
            "sources": sources[:10],
            "debug": {
                "sub_queries": sub_queries,
                "chunks_retrieved": len(ranked_chunks),
                "top_scores": [{"file": c["filename"], "score": round(c["score"], 3)} for c in ranked_chunks[:5]],
                "episodic_memories_used": len(memories),
                "judge": evaluation,
                "refinements": refine_count,
            },
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return {"response": f"Error: {e}"}


# Serve frontend static files in production
static_dir = Path(__file__).with_name("static")
if static_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{path:path}")
    def serve_frontend(path: str):
        file = static_dir / path
        if file.is_file():
            return FileResponse(file)
        return FileResponse(static_dir / "index.html")
