"""
database.py · Supabase (PostgreSQL) & SQLite backend for SafeSpace AI
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
from config import DATABASE_URL
USE_POSTGRES = False
DB_FILE = Path(__file__).resolve().parent / "safe_space.db"

# Determine DB Engine
if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras
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
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
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
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                mood TEXT NOT NULL,
                score INTEGER NOT NULL,
                note TEXT DEFAULT '',
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    else:
        # SQLite
        cur.execute("PRAGMA foreign_keys = ON;")
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
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
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
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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


SALT_CONST = "mental_health_bot_salt_2024"

def hash_password(password: str) -> str:
    """Secure password hashing using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), SALT_CONST.encode("utf-8"), 100000)
    return f"pbkdf2${dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Check against PBKDF2 or legacy SHA-256 for backward compatibility."""
    if not hashed:
        return False
    if hashed.startswith("pbkdf2$"):
        expected = hash_password(password)
        return hashed == expected
    # Legacy SHA-256 fallback
    legacy_hash = hashlib.sha256(f"{SALT_CONST}{password}".encode()).hexdigest()
    return hashed == legacy_hash


# ── User operations ───────────────────────────────────────────────────────────
def create_user(username: str, email: str, password: str) -> dict:
    conn = get_conn()
    cur = conn.cursor()
    username_clean = username.strip()
    email_clean = email.strip().lower()
    pw_hash = hash_password(password)

    if USE_POSTGRES:
        try:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """INSERT INTO users (username, email, password_hash)
                   VALUES (%s, %s, %s)
                   RETURNING id, username, email""",
                (username_clean, email_clean, pw_hash)
            )
            user = cur.fetchone()
            conn.commit()
            return {"id": str(user["id"]), "username": user["username"], "email": user["email"]}
        except Exception as e:
            conn.rollback()
            msg = str(e).lower()
            if "username" in msg:
                raise ValueError("Username already taken.")
            elif "email" in msg:
                raise ValueError("Email already registered.")
            raise ValueError("Registration failed: " + str(e))
        finally:
            cur.close()
            conn.close()
    else:
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
            msg = str(e).lower()
            if "username" in msg:
                raise ValueError("Username already taken.")
            elif "email" in msg:
                raise ValueError("Email already registered.")
            raise ValueError("Username or email already registered.")
        finally:
            cur.close()
            conn.close()


def authenticate_user(username: str, password: str) -> dict | None:
    """Verify credentials and upgrade legacy SHA256 hashes if detected."""
    conn = get_conn()
    username_clean = username.strip()

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, username, email, password_hash FROM users WHERE LOWER(username) = LOWER(%s)",
            (username_clean,)
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return None

        if verify_password(password, row["password_hash"]):
            # Upgrade legacy hash to PBKDF2 if needed
            if not row["password_hash"].startswith("pbkdf2$"):
                new_hash = hash_password(password)
                cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, row["id"]))
                conn.commit()
            user_dict = {"id": str(row["id"]), "username": row["username"], "email": row["email"]}
            cur.close()
            conn.close()
            return user_dict
        cur.close()
        conn.close()
        return None
    else:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, email, password_hash FROM users WHERE LOWER(username) = LOWER(?)",
            (username_clean,)
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return None

        pw_hash = row["password_hash"] if isinstance(row, sqlite3.Row) else row[3]
        uid = row["id"] if isinstance(row, sqlite3.Row) else row[0]
        unm = row["username"] if isinstance(row, sqlite3.Row) else row[1]
        uem = row["email"] if isinstance(row, sqlite3.Row) else row[2]

        if verify_password(password, pw_hash):
            if not pw_hash.startswith("pbkdf2$"):
                new_hash = hash_password(password)
                cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, uid))
                conn.commit()
            cur.close()
            conn.close()
            return {"id": str(uid), "username": unm, "email": uem}
        cur.close()
        conn.close()
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


def get_recent_session_history(session_id: str, user_id: str, limit: int = 10) -> list[dict]:
    """Return the last N messages in the session formatted for LLM conversation context."""
    if not session_id:
        return []
    conn = get_conn()
    s_id = int(session_id)
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT role, message
               FROM chat_history
               WHERE session_id = %s AND user_id = %s
               ORDER BY timestamp DESC
               LIMIT %s""",
            (s_id, u_id, limit)
        )
        rows = list(cur.fetchall())
        cur.close()
        conn.close()
    else:
        cur = conn.cursor()
        cur.execute(
            """SELECT role, message
               FROM chat_history
               WHERE session_id = ? AND user_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (s_id, u_id, limit)
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()

    rows.reverse()
    # Normalize role for LLMs ("user" and "assistant" or "model")
    return [
        {
            "role": "user" if r["role"] == "user" else "assistant",
            "content": r["message"]
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


def get_mood_history(user_id: str, limit: int = 50) -> list[dict]:
    """Return recent mood logs for a user."""
    conn = get_conn()
    u_id = int(user_id)

    if USE_POSTGRES:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT id, mood, score, note, timestamp
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
            """SELECT id, mood, score, note, timestamp
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
            "id":        r["id"],
            "mood":      r["mood"],
            "score":     r["score"],
            "note":      r["note"],
            "timestamp": _format_time(r["timestamp"]),
        }
        for r in rows
    ]


def delete_mood_entry(mood_id: int, user_id: str) -> bool:
    """Delete a single mood entry."""
    conn = get_conn()
    cur = conn.cursor()
    u_id = int(user_id)

    if USE_POSTGRES:
        cur.execute("DELETE FROM mood_logs WHERE id = %s AND user_id = %s", (mood_id, u_id))
    else:
        cur.execute("DELETE FROM mood_logs WHERE id = ? AND user_id = ?", (mood_id, u_id))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted


def get_mood_stats(user_id: str) -> dict:
    """Compute mood streak, average score, and mood distribution."""
    logs = get_mood_history(user_id, limit=100)
    if not logs:
        return {
            "total_logs": 0,
            "average_score": 0.0,
            "streak_days": 0,
            "distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        }

    total = len(logs)
    avg_score = round(sum(l["score"] for l in logs) / total, 1)
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for l in logs:
        score = l.get("score", 3)
        if score in dist:
            dist[score] += 1

    # Calculate streak (consecutive calendar days with a mood log)
    log_dates = set()
    for l in logs:
        try:
            dt = datetime.datetime.fromisoformat(l["timestamp"].replace("Z", "+00:00")).date()
            log_dates.add(dt)
        except Exception:
            pass

    streak = 0
    today = datetime.datetime.now(datetime.timezone.utc).date()
    # Check if logged today or yesterday to continue streak
    check_date = today
    if check_date not in log_dates:
        check_date = today - datetime.timedelta(days=1)

    while check_date in log_dates:
        streak += 1
        check_date -= datetime.timedelta(days=1)

    return {
        "total_logs": total,
        "average_score": avg_score,
        "streak_days": streak,
        "distribution": dist
    }


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
    """Update session title."""
    conn = get_conn()
    cur = conn.cursor()
    s_id = int(session_id)
    u_id = int(user_id)
    title_clean = title.strip()[:60]

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


def delete_session(session_id: str, user_id: str) -> bool:
    """Delete a chat session and its corresponding messages."""
    conn = get_conn()
    cur = conn.cursor()
    s_id = int(session_id)
    u_id = int(user_id)

    if USE_POSTGRES:
        cur.execute("DELETE FROM chat_history WHERE session_id = %s AND user_id = %s", (s_id, u_id))
        cur.execute("DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (s_id, u_id))
    else:
        cur.execute("DELETE FROM chat_history WHERE session_id = ? AND user_id = ?", (s_id, u_id))
        cur.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (s_id, u_id))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted


def clear_all_sessions(user_id: str) -> bool:
    """Delete all sessions and chat messages for a user."""
    conn = get_conn()
    cur = conn.cursor()
    u_id = int(user_id)

    if USE_POSTGRES:
        cur.execute("DELETE FROM chat_history WHERE user_id = %s", (u_id,))
        cur.execute("DELETE FROM chat_sessions WHERE user_id = %s", (u_id,))
    else:
        cur.execute("DELETE FROM chat_history WHERE user_id = ?", (u_id,))
        cur.execute("DELETE FROM chat_sessions WHERE user_id = ?", (u_id,))
    conn.commit()
    cur.close()
    conn.close()
    return True


# ── Auto-init on import ───────────────────────────────────────────────────────
init_db()
