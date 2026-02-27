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


def _clean_json(text: str) -> str:
    """Remove markdown code fences from LLM JSON output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ─────────────────────────────────────────
# Resume Analysis
# ─────────────────────────────────────────

def analyze_resume(resume_text: str) -> dict:
    """Use LLM to analyze resume and extract structured data."""
    client = get_client()
    if not client:
        return _fallback_analyze(resume_text)

    prompt = f"""Analyze the following resume and extract information in JSON format.
Return ONLY valid JSON with these fields:
{{
    "name": "candidate name",
    "email": "email if found",
    "phone": "phone if found",
    "skills": ["list", "of", "skills"],
    "projects": ["project 1 description", "project 2 description"],
    "experience": ["experience 1", "experience 2"],
    "education": ["education 1", "education 2"],
    "interests": ["interest 1", "interest 2"],
    "summary": "brief professional summary"
}}

Resume:
{resume_text}"""

    try:
        text = _generate(prompt)
        text = _clean_json(text)
        return json.loads(text)
    except Exception as e:
        print(f"Resume analysis error: {e}")
        return _fallback_analyze(resume_text)


def _fallback_analyze(text: str) -> dict:
    common_skills = [
        "python", "javascript", "react", "node", "java", "c++", "sql",
        "html", "css", "typescript", "mongodb", "docker", "aws", "git",
        "machine learning", "deep learning", "flask", "django", "fastapi",
        "angular", "vue", "express", "postgresql", "mysql", "redis",
        "kubernetes", "linux", "rust", "go", "swift", "kotlin"
    ]
    found_skills = [s.title() for s in common_skills if s in text.lower()]
    return {
        "name": "", "email": "", "phone": "",
        "skills": found_skills if found_skills else ["General Programming"],
        "projects": ["See resume for project details"],
        "experience": ["See resume for experience details"],
        "education": ["See resume for education details"],
        "interests": [],
        "summary": text[:300] if text else "Resume uploaded successfully"
    }


# ─────────────────────────────────────────
# First Question (from resume)
# ─────────────────────────────────────────

def generate_first_question(resume_data: dict) -> dict:
    """Generate the first warm-up question based on resume."""
    client = get_client()
    if not client:
        return {"question": "Tell me about yourself and your background.", "category": "general", "topic": "introduction"}

    skills = ", ".join(resume_data.get("skills", []))
    projects = ", ".join(resume_data.get("projects", []))
    experience = ", ".join(resume_data.get("experience", []))

    prompt = f"""You are a very friendly, warm, and encouraging technical interviewer starting a live interview.

Candidate's profile:
- Skills: {skills}
- Projects: {projects}
- Experience: {experience}

Generate one simple, friendly opening warm-up question.
Ask them to briefly introduce themselves OR ask about something exciting from their resume
(a project they enjoyed, a skill they like using, etc.).

RULES:
- Keep it VERY EASY and conversational — make the candidate feel comfortable and confident.
- Do NOT ask any technical or tricky question in the opening.
- Think of this as a casual, friendly chat — not a grilling session.
- One short sentence question is ideal.

Return ONLY valid JSON:
{{"question": "your question", "category": "general", "topic": "introduction"}}"""

    try:
        text = _generate(prompt)
        text = _clean_json(text)
        result = json.loads(text)
        result.setdefault("topic", "introduction")
        return result
    except Exception as e:
        print(f"First question error: {e}")
        return {"question": "Tell me about yourself and your background.", "category": "general", "topic": "introduction"}


# ─────────────────────────────────────────
# CORE: Analyze Answer + Generate Next Question
# ─────────────────────────────────────────

def analyze_and_next_question(
    resume_data: dict,
    conversation_history: list,
    current_topic: str,
    topic_question_count: int,
    total_questions_asked: int,
    total_weak_streak: int,
) -> dict:
    """
    The CORE function. Analyzes the user's last answer and decides:
    1. What was correct/wrong in the answer
    2. Should we continue on same topic (2-3 Qs per topic) or switch?
    3. Should we end the interview early (candidate too weak)?
    4. Generate the next question accordingly

    Returns:
    {
        "analysis": "...",
        "score": 7,
        "next_question": "...",
        "category": "...",
        "topic": "...",
        "should_end": false,
        "end_reason": ""
    }
    """
    client = get_client()
    if not client:
        return _fallback_next(resume_data, total_questions_asked, topic_question_count, current_topic)

    skills = ", ".join(resume_data.get("skills", []))
    projects = ", ".join(resume_data.get("projects", []))
    experience = ", ".join(resume_data.get("experience", []))

    # Build conversation context (last 16 messages max)
    recent = conversation_history[-16:] if len(conversation_history) > 16 else conversation_history
    conv_text = "\n".join(
        f"{'Interviewer' if h['role'] == 'interviewer' else 'Candidate'}: {h['content']}"
        for h in recent
    )

    prompt = f"""You are a very friendly, warm, and encouraging technical interviewer conducting a LIVE adaptive interview.
Your job is to ANALYZE the candidate's LAST answer and then generate an EASY next question.

╔══════════════════════════════════════════════════════════════╗
║  CRITICAL RULE: ALL QUESTIONS MUST BE EASY & SIMPLE!        ║
║  Ask basic-level, beginner-friendly questions ONLY.         ║
║  NEVER ask advanced, tricky, obscure, or complex questions. ║
║  Think: "Could a fresher / junior developer answer this?"   ║
║  If NO → make it simpler. If YES → good to go.             ║
╚══════════════════════════════════════════════════════════════╝

Examples of GOOD easy questions:
- "What does HTML stand for?"
- "What is the difference between a list and a dictionary in Python?"
- "Can you explain what an API is in simple terms?"
- "What is version control and why do we use it?"
- "Tell me about a project you worked on — what was your role?"
- "What is the purpose of CSS in web development?"
- "What is a database and why do we need one?"

Examples of BAD questions (TOO HARD — DO NOT ASK THESE):
- "Explain the time complexity of red-black tree rebalancing."
- "How does the V8 engine optimize JIT compilation?"
- "Describe the CAP theorem and its implications for distributed databases."
- "Implement a lock-free concurrent queue."

═══ CANDIDATE PROFILE ═══
Skills: {skills}
Projects: {projects}
Experience: {experience}

═══ CONVERSATION SO FAR ═══
{conv_text}

═══ INTERVIEW STATE ═══
- Current topic being explored: "{current_topic}"
- Questions asked on this topic so far: {topic_question_count}
- Total questions asked in interview: {total_questions_asked}
- Consecutive weak answers streak: {total_weak_streak}

═══ YOUR INSTRUCTIONS ═══

STEP 1 — ANALYZE the candidate's LAST answer:
- What did they say that was CORRECT? Highlight positives first.
- What did they get WRONG or MISS?
- Rate their answer from 1 to 10 (be GENEROUS — give partial credit)
- Write a clear, encouraging analysis summary
- Always find something positive to note, even if the answer was weak.

STEP 2 — DECIDE what to do next based on these RULES:

RULE A — STAY ON SAME TOPIC (if topic_question_count < 3):
  • Weak answer (score 1-4): Ask a VERY EASY follow-up on the SAME topic.
    Use simple words. Include hints or context in the question to help them.
    Example: "That's okay! Let me ask it differently — in simple terms, X is like Y. Can you tell me…?"
  • Decent answer (score 5-7): Ask another EASY question on the SAME topic.
    Stay at the same basic level — do NOT increase difficulty.
  • Great answer (score 8-10): Ask a SLIGHTLY deeper but still SIMPLE question.
    Keep it practical and straightforward — absolutely no trick questions.

RULE B — SWITCH TOPIC (if topic_question_count >= 3):
  • Move to a DIFFERENT skill, project, or concept from their resume.
  • Pick something NOT yet covered in the conversation.
  • Start the new topic with the EASIEST possible introductory question.

RULE C — END INTERVIEW (set should_end = true):
  • IF weak_streak >= 3 AND total_questions >= 5: end gracefully and kindly. Thank them warmly.
  • IF total_questions >= 8 AND overall performance is strong: enough ground covered.
  • Do NOT end before asking at least 5 questions.

RULE D — NO FIXED LIMIT:
  • Minimum ~5 questions, maximum ~20 questions.
  • Interview should feel like a friendly conversation, NOT an exam.
  • Be supportive, patient, and encouraging throughout.

Return ONLY valid JSON (no extra text):
{{
    "analysis": "Encouraging analysis — correct points first, then areas to improve",
    "score": 7,
    "next_question": "Your next EASY question here",
    "category": "technical|project|behavioral|skill|general",
    "topic": "the topic/concept this question is about",
    "should_end": false,
    "end_reason": ""
}}

If should_end is true, set next_question to a warm closing like "Thank you so much for your time! That wraps up our interview. You did great!"
"""

    try:
        text = _generate(prompt)
        text = _clean_json(text)
        result = json.loads(text)

        # Ensure required fields
        result.setdefault("analysis", "Answer analyzed.")
        result.setdefault("score", 5)
        result.setdefault("next_question", "Can you tell me more about that?")
        result.setdefault("category", "general")
        result.setdefault("topic", current_topic)
        result.setdefault("should_end", False)
        result.setdefault("end_reason", "")
        result["score"] = min(max(int(result["score"]), 1), 10)

        return result
    except Exception as e:
        print(f"Analyze & next question error: {e}")
        return _fallback_next(resume_data, total_questions_asked, topic_question_count, current_topic)


# ─────────────────────────────────────────
# After Skip
# ─────────────────────────────────────────

def generate_after_skip(
    resume_data: dict,
    conversation_history: list,
    total_questions_asked: int,
    total_weak_streak: int,
) -> dict:
    """Generate next question after a skip — always switch topic."""
    client = get_client()

    # If too many weak/skips, end interview
    if total_weak_streak >= 3 and total_questions_asked >= 5:
        return {
            "analysis": "Candidate skipped again with too many weak responses.",
            "score": 0,
            "next_question": "Thank you for your time. That concludes our interview.",
            "category": "general",
            "topic": "closing",
            "should_end": True,
            "end_reason": "Too many consecutive skips and weak answers"
        }

    if not client:
        return _fallback_next(resume_data, total_questions_asked, 0, "new")

    skills = ", ".join(resume_data.get("skills", []))
    projects = ", ".join(resume_data.get("projects", []))

    asked_qs = [h["content"] for h in conversation_history if h["role"] == "interviewer"]
    asked_text = "\n- ".join(asked_qs[-8:]) if asked_qs else "None"

    prompt = f"""You are a very friendly, warm interviewer. The candidate just SKIPPED the last question (they didn't know the answer).
That's totally fine! Move to a COMPLETELY DIFFERENT topic from their resume.

Profile:
- Skills: {skills}
- Projects: {projects}

Questions already asked:
- {asked_text}

Total questions so far: {total_questions_asked}

Generate a question on a NEW topic not yet covered. Make it specific to their profile.

CRITICAL: The candidate just skipped, so they might be feeling less confident.
Ask the SIMPLEST, most BASIC question you can think of on the new topic.
Think: simple definitions, "what is X?", "what does Y stand for?", "why do we use Z?" level.
Help them regain confidence with an easy win!

Examples of good post-skip questions:
- "What does CSS stand for and what is it used for?"
- "Can you tell me about a project you really enjoyed working on?"
- "What programming language do you feel most comfortable with and why?"
- "In simple terms, what is an API?"

Return ONLY valid JSON:
{{"next_question": "your EASY question", "category": "technical|project|behavioral|skill|general", "topic": "the new topic"}}"""

    try:
        text = _generate(prompt)
        text = _clean_json(text)
        result = json.loads(text)
        return {
            "analysis": "Candidate skipped — moving to different topic.",
            "score": 0,
            "next_question": result.get("next_question", "Tell me about another project you worked on."),
            "category": result.get("category", "general"),
            "topic": result.get("topic", "new_topic"),
            "should_end": False,
            "end_reason": ""
        }
    except Exception as e:
        print(f"Skip question error: {e}")
        return _fallback_next(resume_data, total_questions_asked, 0, "new")


def _fallback_next(resume_data: dict, q_num: int, topic_count: int, current_topic: str) -> dict:
    """Fallback when LLM unavailable."""
    skills = resume_data.get("skills", ["programming"])
    s1 = skills[0] if skills else "programming"
    s2 = skills[1] if len(skills) > 1 else "your domain"

    pool = [
        {"next_question": f"Can you tell me what {s1} is and why you like using it?", "category": "skill", "topic": s1},
        {"next_question": "Tell me about a project you enjoyed working on. What did you build?", "category": "project", "topic": "projects"},
        {"next_question": "When you find a bug in your code, what is the first thing you usually do?", "category": "behavioral", "topic": "problem_solving"},
        {"next_question": f"What is {s2} used for in simple terms?", "category": "technical", "topic": s2},
        {"next_question": "Tell me about something you learned recently that you found interesting.", "category": "behavioral", "topic": "learning"},
        {"next_question": "Have you used Git before? What basic commands do you use most often?", "category": "technical", "topic": "git"},
        {"next_question": "How do you usually learn a new technology or tool?", "category": "general", "topic": "learning"},
        {"next_question": "What is the difference between frontend and backend development?", "category": "technical", "topic": "basics"},
    ]
    idx = min(q_num, len(pool) - 1)
    fallback = pool[idx]
    fallback.update({"analysis": "Fallback mode — LLM unavailable", "score": 5, "should_end": False, "end_reason": ""})
    return fallback
