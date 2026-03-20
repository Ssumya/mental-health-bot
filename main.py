from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ai_agent import graph, SYSTEM_PROMPT, parse_response
from auth import login, register, get_current_user
from database import save_message, get_user_history

app = FastAPI(title="Mental Health Bot API")

# Allow frontend (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response models ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str


# ─── Auth routes ─────────────────────────────────────────────────────────────

@app.post("/auth/register")
def register_user(req: RegisterRequest):
    """Register a new user and return a JWT token."""
    return register(req.username, req.email, req.password)


@app.post("/auth/login")
def login_user(req: LoginRequest):
    """Login and return a JWT token."""
    return login(req.username, req.password)


# ─── Chat route ──────────────────────────────────────────────────────────────

@app.post("/ask")
def ask(query: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Send a message to the AI agent.
    Requires: Authorization: Bearer <token>
    Saves both the user message and AI response to chat history.
    """
    user_id = current_user["user_id"]
    username = current_user["username"]

    # Save user's message
    save_message(user_id, "user", query.message)

    # Run the AI agent
    inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", query.message)]}
    stream = graph.stream(inputs, stream_mode="updates")
    tool_called, response = parse_response(stream)

    # Save assistant's response
    save_message(user_id, "assistant", response or "", tool_called=tool_called)

    return {
        "response": response,
        "tool_called": tool_called,
        "user": username
    }


# ─── History route ────────────────────────────────────────────────────────────

@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    """
    Fetch the authenticated user's full chat history.
    Requires: Authorization: Bearer <token>
    """
    user_id = current_user["user_id"]
    history = get_user_history(user_id)
    return {"history": history, "count": len(history)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
