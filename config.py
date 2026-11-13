"""
config.py · Centralized Environment & Configuration Management for SafeSpace AI
All agent models, database connections, and authentication routines fetch their configuration from here.
Loads from local '.env' file via python-dotenv, or directly from system / cloud environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env file if present
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# ── Database Configuration (Supabase PostgreSQL) ──────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# ── AI API Credentials ────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ── Authentication Security ───────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "safespace2024xK9mP3qL7nR5tY2wA8").strip()

# ── Optional Local Ollama Settings ────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "alibayram/medgemma:4b").strip()

# ── Server Port ───────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", 8000))
