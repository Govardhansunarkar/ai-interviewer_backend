import json
from openai import OpenAI
from core.config import settings

_client = None


def get_client():
    global _client
    if _client is None and settings.NVIDIA_API_KEY and settings.NVIDIA_API_KEY != "your_nvidia_api_key_here":
        _client = OpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            timeout=30
        )
    return _client


def _generate(prompt: str) -> str:
    """Generate content using NVIDIA API."""
    client = get_client()
    if not client:
        return ""
    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048
    )
    return response.choices[0].message.content.strip() if response.choices else ""


def evaluate_answer(question: str, answer: str, category: str) -> dict:
    """Evaluate an interview answer using LLM with detailed analysis."""
    client = get_client()
    if not client:
        return _fallback_evaluate(answer)

    prompt = f"""You are an expert interviewer evaluating a candidate's answer.

Question: {question}
Category: {category}
Candidate's Answer: {answer}

Analyze the answer thoroughly:
1. What did the candidate get RIGHT?
2. What did the candidate get WRONG or miss?
3. What key points were missing?
4. Overall quality assessment

Return ONLY valid JSON:
{{
    "score": 7,
    "feedback": "Brief constructive feedback",
    "correct_points": "What they got right",
    "wrong_points": "What they got wrong or missed",
    "missing_topics": "Key topics they didn't cover"
}}

Scoring guide:
- 1-3: Poor or irrelevant answer
- 4-5: Below average, missing key points
- 6-7: Good answer, covers basics well
- 8-9: Excellent, detailed and insightful
- 10: Outstanding, exceptional answer"""

    try:
        text = _generate(prompt)
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        result = json.loads(text)
        return {
            "score": min(max(int(result.get("score", 5)), 1), 10),
            "feedback": result.get("feedback", "Answer recorded.")
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return _fallback_evaluate(answer)


def _fallback_evaluate(answer: str) -> dict:
    """Fallback evaluation without LLM."""
    word_count = len(answer.split())
    if word_count < 5:
        return {"score": 3, "feedback": "Answer was too brief. Try to elaborate more."}
    elif word_count < 20:
        return {"score": 5, "feedback": "Decent answer, but could use more detail."}
    elif word_count < 50:
        return {"score": 7, "feedback": "Good answer with reasonable detail."}
    else:
        return {"score": 8, "feedback": "Detailed and comprehensive answer."}


def generate_overall_feedback(results: list) -> str:
    """Generate overall interview feedback."""
    client = get_client()

    answered = [r for r in results if not r.get("skipped")]
    if not answered:
        return "No questions were answered in this interview."

    avg_score = sum(r["score"] for r in answered) / len(answered)

    if not client:
        if avg_score >= 8:
            return "Excellent performance! You demonstrated strong knowledge and communication skills."
        elif avg_score >= 6:
            return "Good performance. You showed solid understanding with room for improvement in some areas."
        elif avg_score >= 4:
            return "Average performance. Consider practicing more detailed and structured responses."
        else:
            return "Needs improvement. Focus on building deeper knowledge and practicing your responses."

    summary = "\n".join([
        f"Q{r['question_number']}: {r['question']}\nA: {r.get('answer', 'Skipped')}\nScore: {r['score']}/10"
        for r in results[:10]
    ])

    prompt = f"""Based on this interview performance, provide a 2-3 sentence overall feedback:

{summary}

Average Score: {avg_score:.1f}/10
Total Questions: {len(results)}
Answered: {len(answered)}
Skipped: {len(results) - len(answered)}

Provide constructive, encouraging feedback."""

    try:
        text = _generate(prompt)
        return text if text else f"Interview completed with an average score of {avg_score:.1f}/10."
    except Exception:
        return f"Interview completed with an average score of {avg_score:.1f}/10."