import jwt
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from database import authenticate_user, create_user

SECRET_KEY = os.getenv("JWT_SECRET", "change-this-secret-in-production-2024")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def create_token(user_id: str, username: str) -> str:
    """Create a JWT token for a user."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please log in.")


def get_current_user(authorization: str = Header(...)) -> dict:
    """
    FastAPI dependency: extracts user from 'Authorization: Bearer <token>' header.
    Usage:  user = Depends(get_current_user)
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1]
    return decode_token(token)


def login(username: str, password: str) -> dict:
    """Authenticate and return token + user info."""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_token(user["id"], user["username"])
    return {"token": token, "user": user}


def register(username: str, email: str, password: str) -> dict:
    """Register new user and return token."""
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    try:
        user = create_user(username, email, password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    token = create_token(user["id"], user["username"])
    return {"token": token, "user": user}
