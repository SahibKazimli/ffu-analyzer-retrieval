"""
Document chunking utilities.

Splits extracted markdown text into overlapping segments.
"""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


def chunk_document(
    text: str,
    filename: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Split *text* into overlapping chunks of approximately *chunk_size* tokens.

    Split on double-newlines to get natural paragraphs.
    Merge paragraphs until the token budget is reached.
    Slide back by *overlap* tokens worth of paragraphs for the next chunk.

    Returns a list of dicts:
        {text, filename, chunk_index, token_count}
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs (double-newline boundaries)
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    if not paragraphs:
        return []

    chunks: list[dict] = []
    current_paragraphs: list[str] = []
    current_tokens = 0
    chunk_index = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_tokens = _token_len(para)

        # If a paragraph exceeds chunk_size, add it as its own chunk
        if para_tokens > chunk_size and not current_paragraphs:
            chunks.append({
                "text": para,
                "filename": filename,
                "chunk_index": chunk_index,
                "token_count": para_tokens,
            })
            chunk_index += 1
            i += 1
            continue

        # If adding this paragraph would exceed the budget, flush
        if current_tokens + para_tokens > chunk_size and current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append({
                "text": chunk_text,
                "filename": filename,
                "chunk_index": chunk_index,
                "token_count": current_tokens,
            })
            chunk_index += 1

            # Calculate overlap, walk backwards keeping paragraphs until
            overlap_paragraphs: list[str] = []
            overlap_tokens = 0
            for prev_para in reversed(current_paragraphs):
                prev_para_tokens = _token_len(prev_para)
                if overlap_tokens + prev_para_tokens > overlap:
                    break
                overlap_paragraphs.insert(0, prev_para)
                overlap_tokens += prev_para_tokens

            current_paragraphs = overlap_paragraphs
            current_tokens = overlap_tokens
            # Re-evaluate this paragraph
            continue

        current_paragraphs.append(para)
        current_tokens += para_tokens
        i += 1

    # Flush remaining
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunks.append({
            "text": chunk_text,
            "filename": filename,
            "chunk_index": chunk_index,
            "token_count": current_tokens,
        })

    return chunks
