from fastapi import APIRouter, HTTPException, Depends
from services.memory_manager import get_session_report
from services.evaluation_engine import generate_overall_feedback
from services.auth_service import get_current_user

router = APIRouter(prefix="/api", tags=["report"])


@router.get("/report/{session_id}")
async def get_report(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get interview report."""
    report = get_session_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Session not found")

    # Generate overall feedback
    overall_feedback = generate_overall_feedback(report["results"])
    report["overall_feedback"] = overall_feedback

    return report