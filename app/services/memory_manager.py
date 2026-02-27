import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class QuestionRecord:
    question_number: int
    question: str
    category: str
    answer: Optional[str] = None
    score: int = 0
    feedback: str = ""
    skipped: bool = False


@dataclass
class InterviewSession:
    session_id: str
    resume_text: str = ""
    resume_summary: dict = field(default_factory=dict)
    questions: List[QuestionRecord] = field(default_factory=list)
    current_question_index: int = 0
    started: bool = False
    finished: bool = False
    start_time: float = 0
    end_time: float = 0
    conversation_history: List[dict] = field(default_factory=list)
    # Adaptive interview tracking
    current_topic: str = "introduction"
    topic_question_count: int = 0
    weak_streak: int = 0


# In-memory session storage
_sessions: Dict[str, InterviewSession] = {}


def create_session() -> str:
    """Create a new interview session."""
    session_id = str(uuid.uuid4())[:8]
    _sessions[session_id] = InterviewSession(session_id=session_id)
    return session_id


def get_session(session_id: str) -> Optional[InterviewSession]:
    """Get an existing session."""
    return _sessions.get(session_id)


def update_session(session_id: str, **kwargs):
    """Update session fields in memory."""
    session = _sessions.get(session_id)
    if session:
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)


def add_question(session_id: str, question: str, category: str) -> int:
    """Add a question to the session. Returns question number."""
    session = _sessions.get(session_id)
    if session:
        q_num = len(session.questions) + 1
        session.questions.append(QuestionRecord(
            question_number=q_num,
            question=question,
            category=category
        ))
        return q_num
    return 0


def record_answer(session_id: str, answer: str, score: int, feedback: str):
    """Record an answer for the current question."""
    session = _sessions.get(session_id)
    if session and session.questions:
        idx = session.current_question_index
        if idx < len(session.questions):
            session.questions[idx].answer = answer
            session.questions[idx].score = score
            session.questions[idx].feedback = feedback
            session.current_question_index += 1


def record_skip(session_id: str):
    """Record a skipped question."""
    session = _sessions.get(session_id)
    if session and session.questions:
        idx = session.current_question_index
        if idx < len(session.questions):
            session.questions[idx].skipped = True
            session.current_question_index += 1


def add_to_history(session_id: str, role: str, content: str):
    """Add to conversation history."""
    session = _sessions.get(session_id)
    if session:
        session.conversation_history.append({"role": role, "content": content})


def get_session_report(session_id: str) -> Optional[dict]:
    """Generate a report for the session."""
    session = _sessions.get(session_id)
    if not session:
        return None

    answered = [q for q in session.questions if not q.skipped and q.answer]
    skipped = [q for q in session.questions if q.skipped]

    # Only consider technical / skill / project answers for "best answer"
    # Exclude generic intro/general/behavioral questions
    technical_answered = [
        q for q in answered
        if q.category in ("technical", "skill", "project")
        and q.score >= 6
    ]
    best_answer = max(technical_answered, key=lambda q: q.score) if technical_answered else None
    worst_answer = min(answered, key=lambda q: q.score) if answered else None
    avg_score = sum(q.score for q in answered) / len(answered) if answered else 0

    duration = int(session.end_time - session.start_time) if session.end_time else 0

    return {
        "session_id": session_id,
        "total_questions": len(session.questions),
        "answered": len(answered),
        "skipped": len(skipped),
        "average_score": round(avg_score, 1),
        "best_answer": best_answer.__dict__ if best_answer else None,
        "worst_answer": worst_answer.__dict__ if worst_answer else None,
        "results": [q.__dict__ for q in session.questions],
        "duration_seconds": duration
    }