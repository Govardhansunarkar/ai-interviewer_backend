from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from services.resume_parser import extract_text_from_pdf, save_resume
from services.rag_engine import store_resume_chunks
from services.interview_engine import analyze_resume
from services.memory_manager import create_session, update_session
from services.auth_service import get_current_user
from core.config import settings

router = APIRouter(prefix="/api", tags=["resume"])


@router.post("/resume")
async def upload_resume(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload and analyze a resume (requires authentication)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Create session
        session_id = create_session()

        # Save file
        content = await file.read()
        file_path = save_resume(content, f"{session_id}_{file.filename}", settings.RESUME_DIR)

        # Extract text
        resume_text = extract_text_from_pdf(file_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Store in vector DB for RAG
        store_resume_chunks(session_id, resume_text)

        # Analyze resume with LLM
        resume_data = analyze_resume(resume_text)

        # Update session
        update_session(session_id, resume_text=resume_text, resume_summary=resume_data)

        return {
            "session_id": session_id,
            "message": "Resume analyzed successfully",
            "resume_data": resume_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")