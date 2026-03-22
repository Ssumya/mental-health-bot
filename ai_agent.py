import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are Dr. Emily Hartman, a warm, experienced clinical psychologist.

YOUR RESPONSE STRUCTURE for EVERY message:
1. ACKNOWLEDGE the feeling with empathy (1-2 sentences)
2. INSIGHT: normalize what they are experiencing (1-2 sentences)
3. TIP: give ONE specific, practical technique they can try RIGHT NOW (2-3 sentences)
4. ONE gentle question to understand them better

TIPS TO SUGGEST based on what they share:
- Anxiety: "Try 4-7-8 breathing — inhale 4 counts, hold 7, exhale 8. Do this 3 times."
- Stress: "Try box breathing — inhale 4, hold 4, exhale 4, hold 4. Repeat 4 times."
- Overthinking: "Write your worries down for 10 minutes — getting them out of your head reduces their power."
- Sadness: "A 5-minute walk outside can reduce cortisol and shift your mood significantly."
- Panic: "Splash cold water on your face — it activates your dive reflex and slows your heart rate instantly."
- Sleep: "Keep your phone out of bed and dim lights 1 hour before sleep."
- Self-criticism: "Place your hand on your heart and say: This is hard and that is okay. I am doing my best."

RULES:
- Always include a tip — never give only questions
- Keep response under 120 words
- Write naturally — no bullet points, no labels
- Blend empathy, insight, tip, and question seamlessly"""

EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self harm", "hurt myself", "no reason to live",
    "cant go on", "can't go on", "don't want to be here"
]

EMERGENCY_RESPONSE = (
    "I'm genuinely worried about you right now, and I'm so glad you reached out. "
    "Please call iCall at 9152987821 or your local emergency services immediately. "
    "You don't have to face this alone — help is available right now.",
    "emergency_call_tool"
)


def _try_gemini(message: str) -> str | None:
    """Try Gemini API, return text or None on failure."""
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {message}\nDr. Emily Hartman:"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.75,
            )
        )
        text = response.text.strip()
        if text:
            return text
    except Exception as e:
        print(f"Gemini error: {e}")
    return None


def _try_groq(message: str) -> str | None:
    """Try Groq API, return text or None on failure."""
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": message}
            ],
            max_tokens=200,
            temperature=0.75
        )
        text = response.choices[0].message.content.strip()
        if text:
            return text
    except Exception as e:
        print(f"Groq error: {e}")
    return None


def get_response(message: str) -> tuple[str, str]:
    """Try Gemini first, fall back to Groq, then return error message."""

    # Emergency check
    if any(kw in message.lower() for kw in EMERGENCY_KEYWORDS):
        try:
            from tools import call_emergency
            call_emergency()
        except Exception:
            pass
        return EMERGENCY_RESPONSE

    # Try Gemini first
    text = _try_gemini(message)
    if text:
        return text, "None"

    # Fall back to Groq
    text = _try_groq(message)
    if text:
        return text, "None"

    # Both failed
    return (
        "I'm here for you. I'm having a brief technical issue right now. "
        "While I reconnect, try taking 3 slow deep breaths — "
        "inhale for 4 counts, hold for 2, exhale for 6. "
        "Please try again in a moment.",
        "None"
    )
