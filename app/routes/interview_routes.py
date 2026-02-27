import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.interview_engine import (
    generate_first_question,
    analyze_and_next_question,
    generate_after_skip
)
from services.memory_manager import (
    get_session, update_session, add_question,
    record_answer, record_skip, add_to_history
)
from services.auth_service import get_current_user

router = APIRouter(prefix="/api/interview", tags=["interview"])


class StartRequest(BaseModel):
    session_id: str


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class SkipRequest(BaseModel):
    session_id: str


@router.post("/start")
async def start_interview(req: StartRequest, current_user: dict = Depends(get_current_user)):
    """Start interview — generate first question from resume."""
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload resume first.")

    if session.started:
        raise HTTPException(status_code=400, detail="Interview already started")

    # Generate the FIRST question from resume
    first_q = generate_first_question(session.resume_summary)
    add_question(req.session_id, first_q["question"], first_q["category"])

    update_session(
        req.session_id,
        started=True,
        start_time=time.time(),
        current_topic=first_q.get("topic", "introduction"),
        topic_question_count=1,
        weak_streak=0
    )
    add_to_history(req.session_id, "interviewer", first_q["question"])

    return {
        "question": first_q["question"],
        "question_number": 1,
        "category": first_q["category"],
        "is_finished": False,
        "analysis": "",
        "score": 0
    }


@router.post("/answer")
async def submit_answer(req: AnswerRequest, current_user: dict = Depends(get_current_user)):
    """
    CORE: Take user's answer → LLM analyzes it → generates next question based on analysis.
    No predefined questions. Everything is dynamic.
    """
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.started:
        raise HTTPException(status_code=400, detail="Interview not started")
    if session.finished:
        raise HTTPException(status_code=400, detail="Interview already finished")

    current_idx = session.current_question_index
    if current_idx >= len(session.questions):
        update_session(req.session_id, finished=True, end_time=time.time())
        return {
            "analysis": "Interview complete!",
            "next_question": None,
            "question_number": current_idx,
            "category": "",
            "is_finished": True,
            "score": 0
        }

    current_q = session.questions[current_idx]

    # Record the answer in history
    add_to_history(req.session_id, "candidate", req.answer)

    # ═══ THE MAIN CALL: Analyze answer + generate next question ═══
    result = analyze_and_next_question(
        resume_data=session.resume_summary,
        conversation_history=session.conversation_history,
        current_topic=session.current_topic,
        topic_question_count=session.topic_question_count,
        total_questions_asked=len(session.questions),
        total_weak_streak=session.weak_streak,
    )

    score = result["score"]
    analysis = result["analysis"]

    # Record the answer with score
    record_answer(req.session_id, req.answer, score, analysis)

    # Update weak streak
    if score <= 5:
        new_weak_streak = session.weak_streak + 1
    else:
        new_weak_streak = 0  # reset on good answer

    # Update topic tracking
    new_topic = result.get("topic", session.current_topic)
    if new_topic != session.current_topic:
        # Topic switched
        new_topic_count = 1
    else:
        new_topic_count = session.topic_question_count + 1

    update_session(
        req.session_id,
        current_topic=new_topic,
        topic_question_count=new_topic_count,
        weak_streak=new_weak_streak
    )

    # Check if LLM wants to end
    if result.get("should_end", False):
        update_session(req.session_id, finished=True, end_time=time.time())
        return {
            "analysis": analysis,
            "next_question": result["next_question"],
            "question_number": len(session.questions),
            "category": result["category"],
            "is_finished": True,
            "score": score,
            "end_reason": result.get("end_reason", "")
        }

    # Store the next question
    next_question = result["next_question"]
    add_question(req.session_id, next_question, result["category"])
    add_to_history(req.session_id, "interviewer", next_question)

    return {
        "analysis": analysis,
        "next_question": next_question,
        "question_number": len(session.questions),
        "category": result["category"],
        "topic": new_topic,
        "is_finished": False,
        "score": score
    }


@router.post("/skip")
async def skip_question(req: SkipRequest, current_user: dict = Depends(get_current_user)):
    """Skip → counts as weak, switch topic, possibly end if too many skips."""
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.started:
        raise HTTPException(status_code=400, detail="Interview not started")
    if session.finished:
        raise HTTPException(status_code=400, detail="Interview already finished")

    # Record skip
    record_skip(req.session_id)

    # Increase weak streak
    new_weak_streak = session.weak_streak + 1
    update_session(req.session_id, weak_streak=new_weak_streak)

    # Generate next question (different topic)
    result = generate_after_skip(
        resume_data=session.resume_summary,
        conversation_history=session.conversation_history,
        total_questions_asked=len(session.questions),
        total_weak_streak=new_weak_streak,
    )

    # Check if should end
    if result.get("should_end", False):
        update_session(req.session_id, finished=True, end_time=time.time())
        return {
            "next_question": result["next_question"],
            "question_number": len(session.questions),
            "category": "general",
            "is_finished": True,
            "end_reason": result.get("end_reason", "")
        }

    # Store new question
    next_question = result["next_question"]
    new_topic = result.get("topic", "new_topic")
    add_question(req.session_id, next_question, result["category"])
    add_to_history(req.session_id, "interviewer", next_question)

    update_session(
        req.session_id,
        current_topic=new_topic,
        topic_question_count=1
    )

    return {
        "next_question": next_question,
        "question_number": len(session.questions),
        "category": result["category"],
        "topic": new_topic,
        "is_finished": False
    }


@router.post("/end")
async def end_interview(req: SkipRequest, current_user: dict = Depends(get_current_user)):
    """End the interview early (user pressed end)."""
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    update_session(req.session_id, finished=True, end_time=time.time())
    return {"message": "Interview ended", "session_id": req.session_id}
