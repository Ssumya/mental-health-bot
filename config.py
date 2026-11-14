"""
config.py · Centralized Environment & Configuration Management for SafeSpace AI
All agent models, database connections, and authentication routines fetch their configuration from here.
Loads from local '.env' file or directly from system / cloud environment variables.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_FILE)
except Exception:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                clean_k = k.strip()
                if clean_k not in os.environ:
                    os.environ[clean_k] = v.strip().strip('"').strip("'")


def _clean_env(key: str, default: str = "") -> str:
    """Read environment variable and strip extraneous whitespace and quotes."""
    raw = os.getenv(key, default)
    if not raw:
        return default
    return str(raw).strip().strip('"').strip("'")


# ── Database Configuration (Supabase PostgreSQL) ──────────────────────────────
DATABASE_URL = _clean_env("DATABASE_URL")

# ── AI API Credentials ────────────────────────────────────────────────────────
OPENROUTER_API_KEY = _clean_env("OPENROUTER_API_KEY")
GEMINI_API_KEY = _clean_env("GEMINI_API_KEY") or _clean_env("GOOGLE_API_KEY")
GROQ_API_KEY = _clean_env("GROQ_API_KEY")

# ── Authentication Security ───────────────────────────────────────────────────
JWT_SECRET = _clean_env("JWT_SECRET", "safespace2024xK9mP3qL7nR5tY2wA8")

# ── Optional Local Ollama Settings ────────────────────────────────────────────
OLLAMA_URL = _clean_env("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = _clean_env("OLLAMA_MODEL", "alibayram/medgemma:4b")

# ── Server Port ───────────────────────────────────────────────────────────────
PORT = int(_clean_env("PORT", "8000"))
