"""
LLM utilities: multi-query generation and answer synthesis.
"""

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


def generate_sub_queries(
    client: OpenAI,
    question: str,
    num: int = 3,
) -> list[str]:
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du är en hjälpare som genererar sökfrågor för ett svenskt bygghandlingssystem (FFU / förfrågningsunderlag). "
                    f"Givet användarens fråga, generera {num} olika och kompletterande sökfrågor som täcker olika aspekter "
                    "av frågan. Skriv ENBART frågorna, en per rad. Ingen numrering, inga förklaringar."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=300,
    )

    raw = response.choices[0].message.content or ""
    sub_queries = [query.strip() for query in raw.strip().split("\n") if query.strip()]

    # Always include the original question
    all_queries = [question] + sub_queries[:num]
    logger.info(f"Generated {len(all_queries)} queries: {all_queries}")
    return all_queries


def build_context(chunks: list[dict], max_tokens: int = 6000) -> str:
    """Format retrieved chunks into a context string for the LLM.

    Includes source information and truncates to stay within *max_tokens*
    (approximate count based on character length / 4).
    """
    if not chunks:
        return "Inga relevanta dokument hittades."

    parts: list[str] = []
    total_chars = 0
    char_budget = max_tokens * 4  

    for i, chunk in enumerate(chunks):
        header = f"[Källa: {chunk['filename']}, del {chunk.get('chunk_index', '?')}]"
        block = f"{header}\n{chunk['content']}"

        if total_chars + len(block) > char_budget:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(parts)


def generate_answer(
    client: OpenAI,
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> str:
    """Generate an answer using retrieved context."""

    system_prompt = (
        "Du är en expert på svenska bygghandlingar och förfrågningsunderlag (FFU). "
        "Svara på användarens fråga baserat på den kontext som tillhandahålls nedan. "
        "Om kontexten inte innehåller tillräcklig information, säg det tydligt. "
        "Referera alltid till källdokument när du citerar information.\n\n"
        "KONTEXT:\n" + context
    )

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=2000,
    )

    return response.choices[0].message.content or ""
