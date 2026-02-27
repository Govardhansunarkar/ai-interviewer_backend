from fastapi import APIRouter
from pydantic import BaseModel
from services.auth_service import verify_google_token, create_access_token, store_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


class GoogleLoginRequest(BaseModel):
    token: str  # Google ID token from frontend


@router.post("/google")
def google_login(req: GoogleLoginRequest):
    """
    Verify Google ID token, store user in memory,
    and return a JWT access token.
    """
    # 1. Verify the Google token
    user_info = verify_google_token(req.token)

    # 2. Store user in memory
    user = store_user(user_info)

    # 3. Create JWT
    access_token = create_access_token({"email": user["email"], "name": user["name"]})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }
