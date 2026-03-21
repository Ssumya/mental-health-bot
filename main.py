from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os

from ai_agent import get_response, SYSTEM_PROMPT
from auth import login, register, get_current_user
from database import save_message, get_user_history, save_mood, get_mood_history

app = FastAPI(title="Mental Health Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request models ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str

class MoodRequest(BaseModel):
    mood: str
    score: int
    note: str = ""


# ─── Auth routes ──────────────────────────────────────────────────────────────

@app.post("/auth/register")
def register_user(req: RegisterRequest):
    return register(req.username, req.email, req.password)

@app.post("/auth/login")
def login_user(req: LoginRequest):
    return login(req.username, req.password)


# ─── Chat routes ──────────────────────────────────────────────────────────────

@app.post("/ask")
def ask(query: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id  = current_user["user_id"]
    username = current_user["username"]

    save_message(user_id, "user", query.message)
    response, tool_called = get_response(query.message)
    save_message(user_id, "assistant", response or "", tool_called=tool_called)

    return {
        "response":    response,
        "tool_called": tool_called,
        "user":        username
    }

@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    history = get_user_history(user_id)
    return {"history": history, "count": len(history)}


# ─── Mood routes ──────────────────────────────────────────────────────────────

@app.post("/mood/log")
def log_mood(req: MoodRequest, current_user: dict = Depends(get_current_user)):
    """Save a mood entry for the logged-in user."""
    user_id = current_user["user_id"]
    entry = save_mood(user_id, req.mood, req.score, req.note)
    return {"status": "saved", "entry": entry}

@app.get("/mood/history")
def mood_history(current_user: dict = Depends(get_current_user)):
    """Get mood history for the logged-in user."""
    user_id = current_user["user_id"]
    logs = get_mood_history(user_id)
    return {"logs": logs, "count": len(logs)}


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
