import sys
from pathlib import Path

# Add the app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.interview_routes import router as interview_router
from routes.resume_routes import router as resume_router
from routes.report_routes import router as report_router
from routes.auth_routes import router as auth_router

app = FastAPI(title="AI Interviewer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://ai-interviewerfrontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(interview_router)
app.include_router(report_router)


@app.get("/")
async def root():
    return {"message": "AI Interviewer API is running"}