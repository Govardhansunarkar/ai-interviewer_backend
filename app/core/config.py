import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
ROOT_DIR = BASE_DIR.parent  # project root

load_dotenv(BASE_DIR / ".env")


class Settings:
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    MODEL_NAME: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    RESUME_DIR: str = str(ROOT_DIR / "data" / "resumes")

    # Google OAuth
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")

    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-this-to-a-random-secret-key")


settings = Settings()