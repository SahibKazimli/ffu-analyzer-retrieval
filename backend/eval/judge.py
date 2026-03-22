import json
from openai import OpenAI

JUDGE_MODEL = "gpt-4o"

JUDGE_PROMPT = (
    "Du är en opartisk domare som utvärderar ett RAG-systems svar. "
    "Var rättvis men noggrann — de flesta okej svar bör hamna i intervallet 0.6–0.8. "
    "Reservera 0.9+ för riktigt utmärkta svar och under 0.5 för tydliga misslyckanden.\n\n"
    "Utvärdera utifrån:\n"
    "1. Trovärdighet — använder svaret endast information från kontexten? "
    "Påståenden utan stöd är hallucinationer.\n"
    "2. Relevans — besvarar svaret den faktiska frågan?\n"
    "3. Fullständighet — täcker det de viktigaste punkterna i kontexten?\n\n"
    "Poängsättning:\n"
    "- 0.9-1.0: Utmärkt — helt trovärdigt, relevant och fullständigt\n"
    "- 0.7-0.8: Bra — mestadels korrekt, mindre brister\n"
    "- 0.5-0.6: Blandat — viss nytta men tydliga problem\n"
    "- 0.3-0.4: Dåligt — betydande hallucinationer eller missar poängen\n"
    "- 0.0-0.2: Underkänt — felaktigt eller påhittat\n\n"
    "Svara ENDAST med giltig JSON: {\"score\": 0.0-1.0, \"reasoning\": \"2-3 meningar: vad som är bra, vad som är fel, och vilka specifika påståenden som saknar stöd i kontexten\"}"
)


def judge_answer(
    client: OpenAI,
    question: str,
    answer: str,
    context: str,
) -> dict:
    """LLM-as-a-judge, evaluate answer quality against the retrieved context.

    Using gpt-4o as the judge (different from gpt-4o-mini used for sub-queries)
    to reduce same-model bias.
    """
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n\n"
                    f"CONTEXT (retrieved chunks):\n{context[:3000]}\n\n"
                    f"ANSWER:\n{answer}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=300,
    )

    raw = response.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": -1, "reasoning": "Judge failed to return valid JSON"}


def refine_with_feedback(
    client: OpenAI,
    question: str,
    context: str,
    history: list[dict] | None,
    generate_fn,
    max_attempts: int = 2,
    threshold: float = 0.7,
) -> tuple[str, dict, int]:
    """Generate an answer, judge it, and refine up to max_attempts times if below threshold.

    Returns (final_answer, final_evaluation, refinement_count).
    """
    answer = generate_fn(client, question, context, history)
    evaluation = judge_answer(client, question, answer, context)
    attempts = 0

    while evaluation.get("score", 1) < threshold and attempts < max_attempts:
        score = evaluation.get("score", 0)
        criticism = evaluation.get("reasoning", "")
        refinement_history = (history or []) + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": (
                f"En oberoende utvärdering gav ditt svar {score}/1.0 med följande kritik:\n"
                f"\"{criticism}\"\n\n"
                f"Skriv om ditt svar och åtgärda dessa specifika problem. "
                f"Använd BARA information som finns i den givna kontexten. "
                f"Om du inte hittar stöd för ett påstående i kontexten, ta bort det."
            )},
        ]
        answer = generate_fn(client, question, context, refinement_history)
        evaluation = judge_answer(client, question, answer, context)
        attempts += 1

    return answer, evaluation, attempts