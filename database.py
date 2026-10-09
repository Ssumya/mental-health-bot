"""
database.py · Supabase (PostgreSQL) & SQLite backend
Tables:
  - users        : { id, username, email, password_hash, created_at }
  - chat_sessions: { id, user_id, title, created_at }
  - chat_history : { id, user_id, session_id, role, message, tool_called, timestamp }
  - mood_logs    : { id, user_id, mood, score, note, timestamp }
"""

import hashlib
import os
import sqlite3
import datetime
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = False
DB_FILE = Path(__file__).resolve().parent / "safe_space.db"

# Determine DB Engine
if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras
        # Quick test connection
        conn = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=5)
        conn.close()
        USE_POSTGRES = True
        print("[DB] Using PostgreSQL backend")
    except Exception as e:
        print(f"[DB Warning] PostgreSQL connection failed: {e}. Falling back to SQLite.")
        USE_POSTGRES = False
else:
    print("[DB] DATABASE_URL not set. Using SQLite local backend.")


def get_conn():
    """Return a database connection (PostgreSQL or SQLite wrapper)."""
    if USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    else:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        return conn


def _format_time(val):
    if val is None:
        return ""
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    return str(val)


def init_db():
    """Create all required tables idempotently."""
    conn = get_conn()
    cur = conn.cursor()

    if USE_POSTGRES:
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
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                session_id INTEGER REFERENCES chat_sessions(id),
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                tool_called TEXT DEFAULT 'None',
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS session_id INTEGER;
        """)
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
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                session_id INTEGER REFERENCES chat_sessions(id),
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                tool_called TEXT DEFAULT 'None',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration check for session_id in SQLite
        cur.execute("PRAGMA table_info(chat_history)")
        columns = [row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cur.fetchall()]
        if "session_id" not in columns:
            cur.execute("ALTER TABLE chat_history ADD COLUMN session_id INTEGER")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS mood_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                mood TEXT NOT NULL,
                score INTEGER NOT NULL,
                note TEXT DEFAULT '',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Tables ensured successfully.")


def hash_password(password: str) -> str:
    salt = "mental_health_bot_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


# ── User operations ───────────────────────────────────────────────────────────
def create_user(username: str, email: str, password: str) -> dict:
    """Insert a new user. Returns user dict or raises ValueError on duplicate."""
    conn = get_conn()
    username_clean = username.strip()
    email_clean = email.strip().lower()
    pw_hash = hash_password(password)

    if USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(
                """INSERT INTO users (username, email, password_hash)
                   VALUES (%s, %s, %s) RETURNING id, username, email""",
                (username_clean, email_clean, pw_hash)
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
            raise ValueError("Username or email already registered.")
        finally:
            cur.close()
            conn.close()
    else:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username_clean, email_clean, pw_hash)
            )
            user_id = cur.lastrowid
            conn.commit()
            return {"id": str(user_id), "username": username_clean, "email": email_clean}
        except sqlite3.IntegrityError as e:
            conn.rollback()
            msg = str(e)
            if "username" in msg:
                raise ValueError("Username already taken.")
            elif "email" in msg:
                raise ValueError("Email already registered.")
            raise ValueError("Username or email already registered.")
        finally:
            cur.close()
            conn.close()


def authenticate_user(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict on success, None on failure."""
    conn = get_conn()
    username_clean = username.strip()
    pw_hash = hash_password(password)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT id, username, email FROM users
               WHERE username = %s AND password_hash = %s""",
            (username_clean, pw_hash)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {"id": str(row["id"]), "username": row["username"], "email": row["email"]}
        return None
    else:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?",
            (username_clean, pw_hash)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {"id": str(row["id"]), "username": row["username"], "email": row["email"]}
        return None


# ── Chat history operations ───────────────────────────────────────────────────
def save_message(user_id, role, message, tool_called='None', session_id=None):
    conn = get_conn()
    cur = conn.cursor()
    s_id = int(session_id) if session_id else None
    u_id = int(user_id)

    if USE_POSTGRES:
        cur.execute(
            'INSERT INTO chat_history(user_id, session_id, role, message, tool_called) VALUES (%s, %s, %s, %s, %s)',
            (u_id, s_id, role, message, tool_called)
        )
    else:
        cur.execute(
            'INSERT INTO chat_history(user_id, session_id, role, message, tool_called) VALUES (?, ?, ?, ?, ?)',
            (u_id, s_id, role, message, tool_called)
        )
    conn.commit()
    cur.close()
    conn.close()


def get_user_history(user_id: str, limit: int = 100) -> list[dict]:
    """Return the most recent messages ordered oldest-first."""
    conn = get_conn()
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT role, message, tool_called, timestamp
               FROM chat_history
               WHERE user_id = %s
               ORDER BY timestamp DESC
               LIMIT %s""",
            (u_id, limit)
        )
        rows = list(cur.fetchall())
        cur.close()
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(
            """SELECT role, message, tool_called, timestamp
               FROM chat_history
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (u_id, limit)
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()

    rows.reverse()
    return [
        {
            "role":        r["role"],
            "message":     r["message"],
            "tool_called": r["tool_called"],
            "timestamp":   _format_time(r["timestamp"]),
        }
        for r in rows
    ]


# ── Mood operations ───────────────────────────────────────────────────────────
def save_mood(user_id: str, mood: str, score: int, note: str = "") -> dict:
    """Save a mood log entry."""
    conn = get_conn()
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """INSERT INTO mood_logs (user_id, mood, score, note)
               VALUES (%s, %s, %s, %s) RETURNING id, mood, score, note, timestamp""",
            (u_id, mood, score, note)
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
            "timestamp": _format_time(row["timestamp"]),
        }
    else:
        cur = conn.cursor()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur.execute(
            """INSERT INTO mood_logs (user_id, mood, score, note, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (u_id, mood, score, note, now)
        )
        row_id = cur.lastrowid
        conn.commit()
        cur.close()
        conn.close()
        return {
            "id":        row_id,
            "mood":      mood,
            "score":     score,
            "note":      note,
            "timestamp": now,
        }


def get_mood_history(user_id: str, limit: int = 30) -> list[dict]:
    """Return recent mood logs for a user."""
    conn = get_conn()
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT mood, score, note, timestamp
               FROM mood_logs
               WHERE user_id = %s
               ORDER BY timestamp DESC
               LIMIT %s""",
            (u_id, limit)
        )
        rows = list(cur.fetchall())
        cur.close()
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(
            """SELECT mood, score, note, timestamp
               FROM mood_logs
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (u_id, limit)
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()

    return [
        {
            "mood":      r["mood"],
            "score":     r["score"],
            "note":      r["note"],
            "timestamp": _format_time(r["timestamp"]),
        }
        for r in rows
    ]


# ── Session operations ────────────────────────────────────────────────────────
def create_session(user_id: str, title: str = "New Chat") -> dict:
    """Create a new chat session."""
    conn = get_conn()
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """INSERT INTO chat_sessions (user_id, title)
               VALUES (%s, %s) RETURNING id, title, created_at""",
            (u_id, title)
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return {
            "id": str(row["id"]),
            "title": row["title"],
            "created_at": _format_time(row["created_at"])
        }
    else:
        cur = conn.cursor()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur.execute(
            "INSERT INTO chat_sessions (user_id, title, created_at) VALUES (?, ?, ?)",
            (u_id, title, now)
        )
        row_id = cur.lastrowid
        conn.commit()
        cur.close()
        conn.close()
        return {
            "id": str(row_id),
            "title": title,
            "created_at": now
        }


def get_sessions(user_id: str) -> list[dict]:
    """Get all sessions for a user, newest first."""
    conn = get_conn()
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
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
            (u_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(
            """SELECT s.id, s.title, s.created_at,
                      COUNT(ch.id) as message_count,
                      MAX(ch.timestamp) as last_message
               FROM chat_sessions s
               LEFT JOIN chat_history ch ON ch.session_id = s.id
               WHERE s.user_id = ?
               GROUP BY s.id
               ORDER BY COALESCE(MAX(ch.timestamp), s.created_at) DESC""",
            (u_id,)
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()

    return [
        {
            "id": str(r["id"]),
            "title": r["title"],
            "created_at": _format_time(r["created_at"]),
            "message_count": r["message_count"],
            "last_message": _format_time(r["last_message"]) if r["last_message"] else None
        }
        for r in rows
    ]


def get_session_messages(session_id: str, user_id: str) -> list[dict]:
    """Get all messages for a specific session."""
    conn = get_conn()
    s_id = int(session_id)
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT role, message, tool_called, timestamp
               FROM chat_history
               WHERE session_id = %s AND user_id = %s
               ORDER BY timestamp ASC""",
            (s_id, u_id)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(
            """SELECT role, message, tool_called, timestamp
               FROM chat_history
               WHERE session_id = ? AND user_id = ?
               ORDER BY timestamp ASC""",
            (s_id, u_id)
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()

    return [
        {
            "role": r["role"],
            "message": r["message"],
            "tool_called": r["tool_called"],
            "timestamp": _format_time(r["timestamp"])
        }
        for r in rows
    ]


def update_session_title(session_id: str, user_id: str, title: str):
    """Update session title (auto-set from first user message)."""
    conn = get_conn()
    cur = conn.cursor()
    s_id = int(session_id)
    u_id = int(user_id)
    title_clean = title[:60]

    if USE_POSTGRES:
        cur.execute(
            "UPDATE chat_sessions SET title = %s WHERE id = %s AND user_id = %s",
            (title_clean, s_id, u_id)
        )
    else:
        cur.execute(
            "UPDATE chat_sessions SET title = ? WHERE id = ? AND user_id = ?",
            (title_clean, s_id, u_id)
        )
    conn.commit()
    cur.close()
    conn.close()


# ── Auto-init on import ───────────────────────────────────────────────────────
init_db()
