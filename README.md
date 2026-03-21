# FFU Analyzer

A RAG application for querying Swedish tender documents (förfrågningsunderlag). Process PDF construction documents, then ask questions in natural language and get answers grounded in the source material.

My chosen track for this assignment was Backend/AI/Infra, and therefore chose to implement better retrieval methods and querying.

Key additions are:
- **Chunking** — Documents split into 500-token chunks with 100-token overlap
- **Embedding** — Chunks are embedded via OpenAI in concurrent batches with threads
- **Multi-query generation** — A big problem with RAG is scalability, large datasets drastically decrease performance in regards to both time and precision. Multiple sub-queries improve recall across different document sections.

## Live demo

**https://ffu-analyzer-851230900226.europe-north1.run.app**

1. Click **Process FFU** — wait for extraction, chunking, and embedding to complete (progress shown in the log panel)
2. Ask questions in the chat input

> **Note:** The database is ephemeral. If the container has restarted since the last use, one person needs to click Process FFU before chatting. Subsequent users can chat immediately, pressing Process FFU again will affect all other users as well. 

![FFU Analyzer screenshot](screenshot.png)

## How it works

1. **Process FFU** is clicked — the backend extracts all PDFs to markdown using `pymupdf4llm` (4 parallel workers)
2. Each document is split into overlapping chunks (~500 tokens each, 100-token overlap) to preserve context across chunk boundaries
3. All chunks are embedded via OpenAI in concurrent batches and stored alongside their vectors in SQLite
4. When a user asks a question, `gpt-4o-mini` generates 3 alternative search queries to cover different angles of the question
5. Each query is embedded and compared against all stored chunks using cosine similarity. Results are deduplicated and the top 15 are selected
6. `gpt-4o` synthesizes an answer from the retrieved chunks, citing source documents

## Project structure

```
ffu-analyzer-retrieval/
├── backend/
│   ├── main.py                 # FastAPI app endpoints
│   ├── requirements.txt        # Python dependencies
│   └── utils/
│       ├── database.py         # SQLite 
│       ├── chunking.py         # Paragraph chunking with overlap
│       ├── embeddings.py       # Batch embedding and cosine similarity search
│       └── llm.py              # Sub-query generation and answer synthesis
├── frontend/
│   ├── src/main.tsx            
│   ├── package.json
│   └── vite.config.js          
├── Dockerfile                  
├── .env.example                        
└── README.md
```

## Tech stack

- **Backend:** Python, FastAPI, SQLite, OpenAI API
- **Frontend:** React, Vite, TypeScript
- **Deployment:** Docker, GCP Cloud Run

## What I would do with more time

- **Persistent storage** — replace SQLite with a vector database so data survives, also user based storage.
- **File upload** — let users upload their own PDFs through the UI instead of relying on pre-bundled files in the container
- **Streaming answers** — stream the LLM response token-by-token to the chat UI for a more responsive feel
