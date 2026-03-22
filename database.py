"""
database.py · Supabase (PostgreSQL) backend
Tables:
  - users        : { id, username, email, password_hash, created_at }
  - chat_history : { id, user_id, role, message, tool_called, timestamp }
"""

import hashlib
import os
import psycopg2
import psycopg2.extras

# ── Connection ───────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

def get_conn():
    """Return a new PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def init_db():
    """Create tables on first run (idempotent)."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                tool_called TEXT DEFAULT 'None',
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Mood logs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mood_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                mood TEXT NOT NULL,
                score INTEGER NOT NULL,
                note TEXT DEFAULT '',
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Supabase connected and tables ensured.")
    except Exception as e:
        print(f"⚠️ Database init warning: {e}")
        print("App will continue — DB will retry on first request.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = "mental_health_bot_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


# ── User operations ───────────────────────────────────────────────────────────
def create_user(username: str, email: str, password: str) -> dict:
    """Insert a new user. Returns user dict or raises ValueError on duplicate."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """INSERT INTO users (username, email, password_hash)
               VALUES (%s, %s, %s) RETURNING id, username, email""",
            (username.strip(), email.strip().lower(), hash_password(password))
        )
        row = cur.fetchone()
        conn.commit()
        return {"id": str(row["id"]), "username": row["username"], "email": row["email"]}
    except psycopg2.errors.UniqueViolation as e:
        conn.rollback()
        msg = str(e)
        if "username" in msg:
            raise ValueError("Username already taken.")
        elif "email" in msg:
            raise ValueError("Email already registered.")
        raise
    finally:
        cur.close()
        conn.close()


def authenticate_user(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict on success, None on failure."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT id, username, email FROM users
           WHERE username = %s AND password_hash = %s""",
        (username.strip(), hash_password(password))
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"id": str(row["id"]), "username": row["username"], "email": row["email"]}
    return None


# ── Chat history operations ───────────────────────────────────────────────────
def save_message(user_id: str, role: str, message: str, tool_called: str = "None"):
    """Append one message to chat_history."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO chat_history (user_id, role, message, tool_called)
           VALUES (%s, %s, %s, %s)""",
        (int(user_id), role, message, tool_called)
    )
    conn.commit()
    cur.close()
    conn.close()


def get_user_history(user_id: str, limit: int = 100) -> list[dict]:
    """Return the most recent messages ordered oldest-first."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT role, message, tool_called, timestamp
           FROM chat_history
           WHERE user_id = %s
           ORDER BY timestamp DESC
           LIMIT %s""",
        (int(user_id), limit)
    )
    rows = list(cur.fetchall())
    cur.close()
    conn.close()
    rows.reverse()
    return [
        {
            "role":        r["role"],
            "message":     r["message"],
            "tool_called": r["tool_called"],
            "timestamp":   r["timestamp"].isoformat() if r["timestamp"] else "",
        }
        for r in rows
    ]


def get_all_users_summary() -> list[dict]:
    """Admin view: each user with message count and last active."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT u.id, u.username, u.email, u.created_at,
               COUNT(ch.id) as message_count,
               MAX(ch.timestamp) as last_active
        FROM users u
        LEFT JOIN chat_history ch ON u.id = ch.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id":            str(r["id"]),
            "username":      r["username"],
            "email":         r["email"],
            "created_at":    r["created_at"].isoformat() if r["created_at"] else "",
            "message_count": r["message_count"],
            "last_active":   r["last_active"].isoformat() if r["last_active"] else None,
        }
        for r in rows
    ]


# ── Auto-init on import ───────────────────────────────────────────────────────
init_db()


# ── Mood operations ───────────────────────────────────────────────────────────
def save_mood(user_id: str, mood: str, score: int, note: str = "") -> dict:
    """Save a mood log entry."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """INSERT INTO mood_logs (user_id, mood, score, note)
           VALUES (%s, %s, %s, %s) RETURNING id, mood, score, note, timestamp""",
        (int(user_id), mood, score, note)
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return {
        "id":        row["id"],
        "mood":      row["mood"],
        "score":     row["score"],
        "note":      row["note"],
        "timestamp": row["timestamp"].isoformat(),
    }


def get_mood_history(user_id: str, limit: int = 30) -> list[dict]:
    """Return recent mood logs for a user."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT mood, score, note, timestamp
           FROM mood_logs
           WHERE user_id = %s
           ORDER BY timestamp DESC
           LIMIT %s""",
        (int(user_id), limit)
    )
    rows = list(cur.fetchall())
    cur.close()
    conn.close()
    return [
        {
            "mood":      r["mood"],
            "score":     r["score"],
            "note":      r["note"],
            "timestamp": r["timestamp"].isoformat(),
        }
        for r in rows
    ]


# ── Session operations ────────────────────────────────────────────────────────
def create_session(user_id: str, title: str = "New Chat") -> dict:
    """Create a new chat session."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """INSERT INTO chat_sessions (user_id, title)
           VALUES (%s, %s) RETURNING id, title, created_at""",
        (int(user_id), title)
    )
    row = cur.fetchone()
    conn.commit(); cur.close(); conn.close()
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "created_at": row["created_at"].isoformat()
    }


def get_sessions(user_id: str) -> list[dict]:
    """Get all sessions for a user, newest first."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT s.id, s.title, s.created_at,
                  COUNT(ch.id) as message_count,
                  MAX(ch.timestamp) as last_message
           FROM chat_sessions s
           LEFT JOIN chat_history ch ON ch.session_id = s.id
           WHERE s.user_id = %s
           GROUP BY s.id
           ORDER BY COALESCE(MAX(ch.timestamp), s.created_at) DESC""",
        (int(user_id),)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [
        {
            "id": str(r["id"]),
            "title": r["title"],
            "created_at": r["created_at"].isoformat(),
            "message_count": r["message_count"],
            "last_message": r["last_message"].isoformat() if r["last_message"] else None
        }
        for r in rows
    ]


def get_session_messages(session_id: str, user_id: str) -> list[dict]:
    """Get all messages for a specific session."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT role, message, tool_called, timestamp
           FROM chat_history
           WHERE session_id = %s AND user_id = %s
           ORDER BY timestamp ASC""",
        (int(session_id), int(user_id))
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [
        {
            "role": r["role"],
            "message": r["message"],
            "tool_called": r["tool_called"],
            "timestamp": r["timestamp"].isoformat()
        }
        for r in rows
    ]


def update_session_title(session_id: str, user_id: str, title: str):
    """Update session title (auto-set from first user message)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE chat_sessions SET title = %s WHERE id = %s AND user_id = %s",
        (title[:60], int(session_id), int(user_id))
    )
    conn.commit(); cur.close(); conn.close()
