import os
import json
import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from ai_agent import get_response, stream_response, SYSTEM_PROMPT
from auth import login, register, get_current_user
from tools import EMERGENCY_HELPLINES
from database import (
    save_message,
    get_user_history,
    save_mood,
    get_mood_history,
    delete_mood_entry,
    get_mood_stats,
    create_session,
    get_sessions,
    get_session_messages,
    update_session_title,
    delete_session,
    clear_all_sessions,
    get_recent_session_history
)

app = FastAPI(title="SafeSpace AI · Mental Health Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request models ─────────────────────────────────────────────────────────
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

class TitleUpdateRequest(BaseModel):
    title: str


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

@app.put("/sessions/{session_id}/title")
def rename_session_route(session_id: str, req: TitleUpdateRequest, current_user: dict = Depends(get_current_user)):
    """Rename a session."""
    update_session_title(session_id, current_user["user_id"], req.title)
    return {"status": "updated", "title": req.title}

@app.delete("/sessions/{session_id}")
def remove_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a single session and its history."""
    deleted = delete_session(session_id, current_user["user_id"])
    return {"status": "deleted" if deleted else "not_found"}

@app.delete("/sessions")
def clear_all(current_user: dict = Depends(get_current_user)):
    """Delete all sessions and conversations for the current user."""
    clear_all_sessions(current_user["user_id"])
    return {"status": "cleared"}

@app.get("/sessions/{session_id}/export")
def export_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Export session as a downloadable markdown document."""
    messages = get_session_messages(session_id, current_user["user_id"])
    user_name = current_user.get("username", "User")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# SafeSpace AI · Conversation Transcript",
        f"- **User:** {user_name}",
        f"- **Date:** {timestamp}",
        f"- **Session ID:** {session_id}",
        "\n---\n"
    ]
    for m in messages:
        sender = user_name if m["role"] == "user" else "SafeSpace AI"
        time_str = m.get("timestamp", "")
        lines.append(f"### {sender} ({time_str})\n{m['message']}\n")
    
    content = "\n".join(lines)
    return PlainTextResponse(
        content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=safespace_session_{session_id}.md"}
    )


# ─── Chat routes ──────────────────────────────────────────────────────────────
@app.post("/ask")
def ask(query: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Standard non-streaming chat endpoint with multi-turn history."""
    user_id = current_user["user_id"]
    username = current_user["username"]

    # Create new session if none provided
    session_id = query.session_id
    if not session_id:
        session = create_session(user_id, "New Chat")
        session_id = session["id"]
        first_words = " ".join(query.message.split()[:6])
        if first_words:
            update_session_title(session_id, user_id, first_words)

    # Fetch prior history for conversational context
    history = get_recent_session_history(session_id, user_id, limit=8)

    # Save user message
    save_message(user_id, "user", query.message, session_id=session_id)

    # Get AI response with context
    response, tool_called = get_response(query.message, history=history)
    save_message(user_id, "assistant", response or "", tool_called=tool_called, session_id=session_id)

    return {
        "response": response,
        "tool_called": tool_called,
        "session_id": session_id,
        "user": username
    }


@app.post("/ask/stream")
def ask_stream(query: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Real-time Server-Sent Events (SSE) streaming chat endpoint."""
    user_id = current_user["user_id"]
    username = current_user["username"]

    session_id = query.session_id
    if not session_id:
        session = create_session(user_id, "New Chat")
        session_id = session["id"]
        first_words = " ".join(query.message.split()[:6])
        if first_words:
            update_session_title(session_id, user_id, first_words)

    # Fetch prior history
    history = get_recent_session_history(session_id, user_id, limit=8)

    # Save user message
    save_message(user_id, "user", query.message, session_id=session_id)

    def event_stream():
        full_chunks = []
        final_tool = "None"
        try:
            for item in stream_response(query.message, history=history):
                chunk = item.get("chunk", "")
                if chunk:
                    full_chunks.append(chunk)
                t = item.get("tool_called", "None")
                if t and t != "None":
                    final_tool = t

                payload = {
                    "chunk": chunk,
                    "tool_called": final_tool,
                    "session_id": session_id,
                    "done": item.get("done", False)
                }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            payload = {
                "chunk": " (Response interruption)",
                "tool_called": final_tool,
                "session_id": session_id,
                "done": True
            }
            yield f"data: {json.dumps(payload)}\n\n"
        finally:
            full_text = "".join(full_chunks).strip()
            if full_text:
                save_message(user_id, "assistant", full_text, tool_called=final_tool, session_id=session_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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

@app.delete("/mood/{mood_id}")
def delete_mood(mood_id: int, current_user: dict = Depends(get_current_user)):
    deleted = delete_mood_entry(mood_id, current_user["user_id"])
    return {"status": "deleted" if deleted else "not_found"}

@app.get("/mood/stats")
def mood_stats(current_user: dict = Depends(get_current_user)):
    stats = get_mood_stats(current_user["user_id"])
    return {"stats": stats}


# ─── Emergency & Helplines ────────────────────────────────────────────────────
@app.get("/emergency/helplines")
def helplines():
    return {"helplines": EMERGENCY_HELPLINES}


# ─── Health & Frontend ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "app": "SafeSpace AI", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
