"""
database.py  ·  MongoDB backend
Collections:
  - users        : { username, email, password_hash, created_at }
  - chat_history : { user_id (ObjectId), role, message, tool_called, timestamp }
"""

import hashlib
import os
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

# ── Connection ────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB",  "mental_health_bot")

_client: MongoClient | None = None

def get_db():
    """Return the database handle (lazy singleton connection)."""
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[DB_NAME]


def init_db():
    """Create indexes on first run (idempotent)."""
    db = get_db()
    db.users.create_index("username", unique=True)
    db.users.create_index("email",    unique=True)
    db.chat_history.create_index(
        [("user_id", ASCENDING), ("timestamp", ASCENDING)]
    )
    print("✅ MongoDB connected and indexes ensured.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _now() -> datetime:
    return datetime.now(timezone.utc)

def hash_password(password: str) -> str:
    """SHA-256 with a fixed salt (upgrade to bcrypt for production)."""
    salt = "mental_health_bot_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def _serialize_user(doc: dict) -> dict:
    return {
        "id":       str(doc["_id"]),
        "username": doc["username"],
        "email":    doc["email"],
    }

def _serialize_message(doc: dict) -> dict:
    ts = doc.get("timestamp")
    return {
        "role":        doc["role"],
        "message":     doc["message"],
        "tool_called": doc.get("tool_called", "None"),
        "timestamp":   ts.isoformat() if isinstance(ts, datetime) else str(ts),
    }


# ── User operations ───────────────────────────────────────────────────────────
def create_user(username: str, email: str, password: str) -> dict:
    """Insert a new user. Returns serialized user dict, or raises ValueError on duplicate."""
    db = get_db()
    doc = {
        "username":      username.strip(),
        "email":         email.strip().lower(),
        "password_hash": hash_password(password),
        "created_at":    _now(),
    }
    try:
        result = db.users.insert_one(doc)
        doc["_id"] = result.inserted_id
        return _serialize_user(doc)
    except DuplicateKeyError as e:
        msg = str(e)
        if "username" in msg:
            raise ValueError("Username already taken.")
        elif "email" in msg:
            raise ValueError("Email already registered.")
        raise


def authenticate_user(username: str, password: str) -> dict | None:
    """Verify credentials. Returns serialized user dict on success, None on failure."""
    db = get_db()
    doc = db.users.find_one({
        "username":      username.strip(),
        "password_hash": hash_password(password),
    })
    return _serialize_user(doc) if doc else None


# ── Chat history operations ───────────────────────────────────────────────────
def save_message(user_id: str, role: str, message: str, tool_called: str = "None"):
    """Append one message to chat_history."""
    db = get_db()
    db.chat_history.insert_one({
        "user_id":     ObjectId(user_id),
        "role":        role,
        "message":     message,
        "tool_called": tool_called,
        "timestamp":   _now(),
    })


def get_user_history(user_id: str, limit: int = 100) -> list[dict]:
    """Return the most recent `limit` messages, ordered oldest-first."""
    db = get_db()
    cursor = (
        db.chat_history
        .find({"user_id": ObjectId(user_id)})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    docs = list(cursor)
    docs.reverse()
    return [_serialize_message(d) for d in docs]


def get_all_users_summary() -> list[dict]:
    """Admin view: each user with message count and last-active timestamp."""
    db = get_db()
    pipeline = [
        {
            "$lookup": {
                "from":         "chat_history",
                "localField":   "_id",
                "foreignField": "user_id",
                "as":           "messages",
            }
        },
        {
            "$project": {
                "username":      1,
                "email":         1,
                "created_at":    1,
                "message_count": {"$size": "$messages"},
                "last_active":   {"$max": "$messages.timestamp"},
            }
        },
        {"$sort": {"created_at": DESCENDING}},
    ]
    return [
        {
            "id":            str(d["_id"]),
            "username":      d["username"],
            "email":         d["email"],
            "created_at":    d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else "",
            "message_count": d.get("message_count", 0),
            "last_active":   d["last_active"].isoformat() if isinstance(d.get("last_active"), datetime) else None,
        }
        for d in list(db.users.aggregate(pipeline))
    ]


# ── Auto-init on import ───────────────────────────────────────────────────────
init_db()
