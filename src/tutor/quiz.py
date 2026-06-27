import json
import re
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, field_validator

NUM_QUESTIONS = 10
_OPTION_KEYS = ("A", "B", "C", "D")


class Question(BaseModel):
    question: str
    options: dict[str, str]
    answer: str
    explanation: str

    @field_validator("options")
    @classmethod
    def _check_options(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v) != set(_OPTION_KEYS):
            raise ValueError(f"options must have keys {_OPTION_KEYS}, got {sorted(v)}")
        return v

    @field_validator("answer")
    @classmethod
    def _check_answer(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in _OPTION_KEYS:
            raise ValueError(f"answer must be one of {_OPTION_KEYS}, got {v!r}")
        return v


class Quiz(BaseModel):
    id: str
    topic: str
    questions: list[Question]


_GENERATION_PROMPT = """You are a senior engineer running a system design interview. Write \
exactly {n} multiple-choice questions about "{topic}" of the kind that actually come up in a \
real system design interview.

Style — questions MUST be:
- Practical and concept-focused: trade-offs, "when would you use X vs Y", failure modes, \
scaling decisions, real-world scenarios ("Your service needs ... which approach fits best?").
- The kind a candidate is expected to reason about, not memorize.

Hard bans — NEVER write trivia questions. Do NOT reference or ask about:
- specific papers, authors, researchers, book titles, companies, or publication years \
(e.g. "In the 1987 paper by Demers et al. ..." is forbidden);
- who invented something, what something is named after, or any historical/biographical fact.
Use the reference context only to ground the technical concepts — not as a source of citations.

Other rules:
- Each question has exactly 4 options keyed "A", "B", "C", "D"; exactly one is correct.
- Vary difficulty and cover different facets of the topic. Do not repeat questions.
- Make all 4 options plausible and tempting; avoid obvious throwaway answers.
- The "explanation" briefly says why the correct option is right (still no citations).

Reference context:
{context}

Return ONLY a JSON array of {n} objects, no prose, no markdown fences. Each object:
{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, \
"answer": "A", "explanation": "..."}}"""


def _extract_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in LLM response")
    return json.loads(text[start : end + 1])


def generate_quiz(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    topic: str,
    n: int = NUM_QUESTIONS,
) -> Quiz:
    topic = topic.strip()
    if not topic:
        raise ValueError("topic is empty")

    docs = retriever.invoke(topic)
    context = "\n\n".join(doc.page_content for doc in docs) or "(no reference material found)"

    prompt = _GENERATION_PROMPT.format(n=n, topic=topic, context=context)
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    questions = [Question(**item) for item in _extract_json_array(raw)]
    if not questions:
        raise ValueError("LLM returned no questions")

    return Quiz(id=uuid.uuid4().hex, topic=topic, questions=questions[:n])


def grade(quiz: Quiz, answers: dict[str, str]) -> dict:
    results = []
    score = 0
    for idx, q in enumerate(quiz.questions):
        given = (answers.get(str(idx)) or "").strip().upper()
        is_correct = given == q.answer
        score += is_correct
        results.append(
            {
                "question": q.question,
                "options": q.options,
                "your_answer": given or None,
                "correct_answer": q.answer,
                "is_correct": is_correct,
                "explanation": q.explanation,
            }
        )
    return {"score": score, "total": len(quiz.questions), "results": results}
