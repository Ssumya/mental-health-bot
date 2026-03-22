from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn, os

from ai_agent import get_response, SYSTEM_PROMPT
from auth import login, register, get_current_user
from database import (save_message, get_user_history, save_mood, get_mood_history,
                      create_session, get_sessions, get_session_messages, update_session_title)

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
    session_id: str = None   # optional — if None, creates new session

class MoodRequest(BaseModel):
    mood: str
    score: int
    note: str = ""

class SessionRequest(BaseModel):
    title: str = "New Chat"

# ─── Auth routes ──────────────────────────────────────────────────────────────
@app.post("/auth/register")
def register_user(req: RegisterRequest):
    return register(req.username, req.email, req.password)

@app.post("/auth/login")
def login_user(req: LoginRequest):
    return login(req.username, req.password)

# ─── Session routes ───────────────────────────────────────────────────────────
@app.post("/sessions")
def new_session(req: SessionRequest, current_user: dict = Depends(get_current_user)):
    """Create a new chat session."""
    session = create_session(current_user["user_id"], req.title)
    return session

@app.get("/sessions")
def list_sessions(current_user: dict = Depends(get_current_user)):
    """List all sessions for the user."""
    sessions = get_sessions(current_user["user_id"])
    return {"sessions": sessions}

@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get messages for a specific session."""
    messages = get_session_messages(session_id, current_user["user_id"])
    return {"messages": messages}

# ─── Chat route ───────────────────────────────────────────────────────────────
@app.post("/ask")
def ask(query: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    username = current_user["username"]

    # Create new session if none provided
    session_id = query.session_id
    if not session_id:
        session = create_session(user_id, "New Chat")
        session_id = session["id"]

    # Save user message
    save_message(user_id, "user", query.message, session_id=session_id)

    # Auto-title session from first message
    first_words = " ".join(query.message.split()[:6])
    update_session_title(session_id, user_id, first_words)

    # Get AI response
    response, tool_called = get_response(query.message)
    save_message(user_id, "assistant", response or "", tool_called=tool_called, session_id=session_id)

    return {
        "response": response,
        "tool_called": tool_called,
        "session_id": session_id,
        "user": username
    }

# ─── History route ────────────────────────────────────────────────────────────
@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    history = get_user_history(current_user["user_id"])
    return {"history": history, "count": len(history)}

# ─── Mood routes ──────────────────────────────────────────────────────────────
@app.post("/mood/log")
def log_mood(req: MoodRequest, current_user: dict = Depends(get_current_user)):
    entry = save_mood(current_user["user_id"], req.mood, req.score, req.note)
    return {"status": "saved", "entry": entry}

@app.get("/mood/history")
def mood_history(current_user: dict = Depends(get_current_user)):
    logs = get_mood_history(current_user["user_id"])
    return {"logs": logs, "count": len(logs)}

# ─── Health & Frontend ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
