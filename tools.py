# Step1: Setup Ollama with Medgemma tool (safe import)
import os
import sys
import datetime
import threading
try:
    from config import OLLAMA_URL, OLLAMA_MODEL
except Exception:
    OLLAMA_URL = "http://localhost:11434/api/chat"
    OLLAMA_MODEL = "alibayram/medgemma:4b"

EMERGENCY_LOG = "emergency_log.txt"

# Verified Crisis & Mental Health Helplines (Global & Regional)
EMERGENCY_HELPLINES = {
    "India": [
        {"name": "Tele-MANAS (24/7 Govt Mental Health Helpline)", "number": "14416", "tel": "tel:14416", "desc": "Toll-free 24/7 multi-language crisis support"},
        {"name": "iCall (TISS Psychosocial Helpline)", "number": "9152987821", "tel": "tel:9152987821", "desc": "Mon-Sat 8 AM - 10 PM free counseling"},
        {"name": "Vandrevala Foundation 24/7 Helpline", "number": "+91 9999 666 555", "tel": "tel:+919999666555", "desc": "24/7 free mental health counseling"},
        {"name": "KIRAN National Helpline", "number": "1800-599-0019", "tel": "tel:18005990019", "desc": "24/7 Ministry of Social Justice mental health support"},
        {"name": "National Emergency Number", "number": "112", "tel": "tel:112", "desc": "Police, Ambulance, Fire"}
    ],
    "USA & Canada": [
        {"name": "Suicide & Crisis Lifeline", "number": "988", "tel": "tel:988", "desc": "Call or Text 988 (24/7 free & confidential)"},
        {"name": "Crisis Text Line", "number": "Text HOME to 741741", "tel": "sms:741741?body=HOME", "desc": "Free 24/7 text support with trained crisis counselor"},
        {"name": "The Trevor Project (LGBTQ+ Youth)", "number": "1-866-488-7386", "tel": "tel:18664887386", "desc": "24/7 confidential crisis line for young LGBTQ people"},
        {"name": "Emergency Services", "number": "911", "tel": "tel:911", "desc": "Immediate emergency dispatch"}
    ],
    "United Kingdom": [
        {"name": "Samaritans Helpline", "number": "116 123", "tel": "tel:116123", "desc": "Free 24/7 compassionate support for anyone in distress"},
        {"name": "NHS Mental Health Services", "number": "111", "tel": "tel:111", "desc": "Free medical & mental health guidance"},
        {"name": "Shout Crisis Text Line", "number": "Text SHOUT to 85258", "tel": "sms:85258?body=SHOUT", "desc": "Free 24/7 confidential text support"},
        {"name": "Emergency Services", "number": "999", "tel": "tel:999", "desc": "Immediate emergency services"}
    ],
    "International": [
        {"name": "Find A Helpline (Global Directory)", "number": "findahelpline.com", "url": "https://findahelpline.com/", "desc": "Free confidential support in 130+ countries"},
        {"name": "Befrienders Worldwide", "number": "befrienders.org", "url": "https://www.befrienders.org/", "desc": "Emotional support to prevent suicide worldwide"}
    ]
}


def query_medgemma(prompt: str) -> str:
    """
    Calls MedGemma model with a therapist personality profile.
    Returns responses as an empathic mental health professional.
    """
    ollama = None
    try:
        import importlib
        ollama = importlib.import_module("ollama")
    except Exception:
        pass

    if not ollama:
        # Fallback to local HTTP API if Ollama daemon is running without python package
        try:
            import json
            import urllib.request
            data = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "You are Dr. Emily Hartman, a warm and experienced clinical psychologist."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res.get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return "MedGemma requires a running Ollama server (or pip install ollama)."

    system_prompt = """You are Dr. Emily Hartman, a warm and experienced clinical psychologist. 
    Respond to patients with:
    1. Emotional attunement ("I can sense how difficult this must be...")
    2. Gentle normalization ("Many people feel this way when...")
    3. Practical guidance ("What sometimes helps is...")
    4. Strengths-focused support ("I notice how you're...")
    Key principles:
    - Never use brackets or labels
    - Blend elements seamlessly
    - Vary sentence structure
    - Use natural transitions
    - Mirror the user's language level
    - Always keep the conversation going by asking open ended questions to dive into the root cause of patients problem
    """
    
    try:
        response = ollama.chat(
            model='alibayram/medgemma:4b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                'num_predict': 350,
                'temperature': 0.7,
                'top_p': 0.9
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return "I'm having technical difficulties, but I want you to know your feelings matter. Please try again shortly."


def _is_headless() -> bool:
    """Determine if running in a headless/cloud environment (like Render, Heroku, Docker)."""
    if os.environ.get("RENDER") or os.environ.get("DYNO") or os.environ.get("CONTAINER"):
        return True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return True
    return False


def _beep():
    """Play alert sound cross-platform if in desktop environment."""
    if _is_headless():
        return
    try:
        if sys.platform == "win32":
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 400)
        elif sys.platform == "darwin":
            for _ in range(3):
                os.system("afplay /System/Library/Sounds/Sosumi.aiff 2>/dev/null")
        elif sys.platform.startswith("linux"):
            for _ in range(3):
                print("\a", end="", flush=True)
                import time; time.sleep(0.3)
    except Exception:
        pass


def _show_popup():
    """Show a popup alert window cross-platform if display is available."""
    if _is_headless():
        return
    msg = "🚨 MENTAL HEALTH EMERGENCY DETECTED\n\nA user expressed crisis thoughts.\nCheck emergency_log.txt immediately."
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "⚠️ EMERGENCY ALERT", 0x00000010)
        elif sys.platform == "darwin":
            safe_msg = msg.replace("'", "\\'").replace("\n", "\\n")
            os.system(f"osascript -e 'display dialog \"{safe_msg}\" with title \"EMERGENCY\" buttons {{\"OK\"}} with icon stop' 2>/dev/null")
        elif sys.platform.startswith("linux"):
            safe_msg = msg.replace('"', '\\"')
            if os.system("which zenity > /dev/null 2>&1") == 0:
                os.system(f'zenity --error --title="EMERGENCY ALERT" --text="{safe_msg}" 2>/dev/null')
    except Exception:
        pass


def _log_emergency(user_info: str = "Anonymous"):
    """Write emergency event to log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"\n{'='*60}\n"
        f"  🚨 EMERGENCY TRIGGERED\n"
        f"  Time    : {timestamp}\n"
        f"  User    : {user_info}\n"
        f"  Action  : User expressed suicidal ideation or acute crisis\n"
        f"  Status  : Crisis protocol activated (helpline provided)\n"
        f"{'='*60}\n"
    )
    try:
        with open(EMERGENCY_LOG, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"[Emergency Log Warning] Failed to write to {EMERGENCY_LOG}: {e}")

    print(log_entry)


def call_emergency(user_info: str = "Anonymous"):
    """
    Emergency alert protocol.
    1. Writes a timestamped entry to emergency_log.txt
    2. Safely triggers desktop audio/popup only if running locally on a desktop
    3. Runs in background threads so it never delays the response.
    """
    _log_emergency(user_info)
    if not _is_headless():
        threading.Thread(target=_beep, daemon=True).start()
        threading.Thread(target=_show_popup, daemon=True).start()
