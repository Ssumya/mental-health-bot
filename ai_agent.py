import os
import json
import urllib.request
from emotion_model import predict_emotion

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "alibayram/medgemma:4b").strip()


def _get_groq_client():
    """Safely return Groq client if package and API key are available."""
    if not GROQ_API_KEY:
        return None
    try:
        import importlib
        groq_mod = importlib.import_module("groq")
        return groq_mod.Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"[AI Agent] Groq init warning: {e}")
        return None


def _get_gemini_client():
    """Safely return modern or legacy Gemini client if package and API key are available."""
    if not GEMINI_API_KEY:
        return None, None
    import importlib
    try:
        genai_mod = importlib.import_module("google.genai")
        return genai_mod.Client(api_key=GEMINI_API_KEY), None
    except Exception:
        pass
    try:
        genai_legacy = importlib.import_module("google.generativeai")
        genai_legacy.configure(api_key=GEMINI_API_KEY)
        return None, genai_legacy.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        print(f"[AI Agent] Gemini init warning: {e}")
        return None, None


SYSTEM_PROMPT = """You are SafeSpace AI — an empathetic mental health companion. You are bilingual: you understand both Hindi and English perfectly.

LANGUAGE RULES:
- If the user writes in Hindi or Hinglish → respond in Hindi (Devanagari script or Hinglish) mixed with some English naturally
- If the user writes in English → respond in English
- Always match the user's language style

RESPONSE STRUCTURE (every message):
1. ACKNOWLEDGE their feeling with warmth (1-2 sentences)
2. INSIGHT — normalize what they're experiencing (1 sentence)
3. PRACTICAL TIP — one specific technique RIGHT NOW (2-3 sentences)
4. ONE gentle follow-up question

TIPS TO USE:
- Anxiety/panic → 4-7-8 breathing: 4 counts inhale, 7 hold, 8 exhale
- Stress → box breathing: inhale 4, hold 4, exhale 4, hold 4
- Overthinking → write worries for 10 mins to clear the mind
- Sadness → 5-min walk outside reduces cortisol significantly
- Anger → cold water on face activates dive reflex, calms instantly
- Sleep → phone out of bed, dim lights 1 hour before sleep
- Loneliness → gratitude practice, write 3 good things daily

TONE: Warm, intelligent, natural, conversational. NOT clinical.
LENGTH: 4-6 sentences max. No long paragraphs.
EMOJIS: 1-2 per response naturally."""

EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self harm", "hurt myself", "no reason to live",
    "cant go on", "can't go on", "don't want to be here",
    "marna chahta", "marna chahti", "jaan dena", "khud ko hurt"
]

EMERGENCY_RESPONSE_EN = (
    "I'm really worried about you right now, and I'm so glad you reached out 💙 "
    "Please call iCall at 9152987821 (India) or your local emergency services immediately. "
    "You are not alone — help is available right now.",
    "emergency_call_tool"
)

EMERGENCY_RESPONSE_HI = (
    "Main aapke baare mein bahut chintit hoon, aur main bahut khush hoon ki aapne reach out kiya 💙 "
    "Kripya abhi iCall ko call karein: 9152987821. Aap akele nahi hain — madad available hai.",
    "emergency_call_tool"
)


def _query_ollama_local(message: str, emotion: str) -> str | None:
    """Attempt to query local Ollama model (e.g. MedGemma / Llama) if server is running."""
    try:
        data = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Predicted emotion signal: {emotion}\nUser message: {message}"}
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300}
        }
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                result = json.loads(resp.read().decode("utf-8"))
                content = result.get("message", {}).get("content", "").strip()
                if content:
                    return content
    except Exception:
        pass
    return None


def _generate_fallback_response(message: str, emotion: str) -> str:
    """Generate a high-quality, emotion-aware response if no LLM API key is provided."""
    msg_lower = message.lower()
    hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097f')
    is_hindi = hindi_chars > 0 or any(w in msg_lower for w in ['hai', 'hoon', 'mujhe', 'karo', 'meri', 'kya', 'bhi', 'kuch', 'lag', 'ho'])

    emo = (emotion or "").lower()

    if "anxi" in emo or "fear" in emo or "panic" in msg_lower or "anxious" in msg_lower:
        if is_hindi:
            return (
                "Main samajh sakta/sakti hoon — anxiety bahut overwhelming feel hoti hai 💙 "
                "Ye bilkul normal hai, aapka nervous system thoda overdrive mein hai. "
                "Abhi try karo 4-7-8 breathing: 4 counts mein saans lo, 7 counts hold karo, aur 8 counts mein exhale karo. "
                "Isse 3 baar repeat karo aur body ko relax hone do. "
                "Kya aap bata sakte hain kis baat ki chinta sabse zyada ho rahi hai?"
            )
        else:
            return (
                "I hear you — anxiety can feel so overwhelming, especially when it builds up 💙 "
                "What you're experiencing is your nervous system going into overdrive, which is completely normal. "
                "Try 4-7-8 breathing right now: inhale for 4 counts, hold for 7, and exhale slowly for 8. "
                "Do this 3 times to help your body reset. "
                "What do you think has been triggering this feeling today?"
            )

    elif "sad" in emo or "depress" in emo or "sadness" in emo or "lonely" in msg_lower:
        if is_hindi:
            return (
                "Mujhe dukh hai ki aap aisi feeling se guzar rahe hain 💙 "
                "Kuch dino mein heavy feel hona bilkul natural hai, apne aap par zyada pressure mat lo. "
                "Ek chota step lo: 5 minute ke liye taazi hawa mein walk karo ya ek glass paani pi kar aao. "
                "Choti movement se bhi cortisol level kam hone lagta hai. "
                "Aap is baare mein thoda aur share karna chahenge?"
            )
        else:
            return (
                "I'm so sorry you're feeling down right now 💙 "
                "It's completely okay to have heavy moments — please be gentle with yourself. "
                "Try taking one small action right now: step outside for a quick 5-minute walk or drink a cold glass of water. "
                "Even light movement helps lower stress hormones and shift your energy. "
                "Would you like to talk about what's been weighing on your mind?"
            )

    elif "ang" in emo or "frustrat" in msg_lower or "gussa" in msg_lower:
        if is_hindi:
            return (
                "Gussa aur frustration aana bilkul samjha ja sakta hai 💙 "
                "Jab emotions high hote hain to body mein bohot adrenaline release hota hai. "
                "Thande paani se chehre ko dho lo — ye dive reflex activate karta hai aur heart rate ko turant calm karta hai. "
                "Kya hua tha jo aapko itna upset feel hua?"
            )
        else:
            return (
                "It's completely understandable that you're feeling angry or frustrated 💙 "
                "High emotional intensity triggers a rush of adrenaline in the body. "
                "Splashing cold water on your face right now triggers the dive reflex and calms your heart rate instantly. "
                "What happened that made you feel this way?"
            )

    else:
        if is_hindi:
            return (
                "Main yahan aapki baat sunne ke liye hoon 💙 "
                "Jo bhi aap feel kar rahe hain, use express karna pehla positive step hai. "
                "Abhi box breathing try karo: 4 sec inhale, 4 sec hold, 4 sec exhale, 4 sec hold. "
                "Isse mind calm aur clear feel karega. Aap aaj kaisa feel kar rahe hain?"
            )
        else:
            return (
                "I'm here for you and listening 💙 "
                "Expressing what you're experiencing is already a powerful step forward. "
                "Try box breathing right now: inhale for 4 seconds, hold for 4, exhale for 4, hold for 4. "
                "This brings immediate clarity and calm to your mind. How are you feeling right now?"
            )


def get_response(message: str) -> tuple[str, str]:
    msg_lower = message.lower()

    # Emergency check
    if any(kw in msg_lower for kw in EMERGENCY_KEYWORDS):
        try:
            from tools import call_emergency
            call_emergency()
        except Exception:
            pass
        hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097f')
        if hindi_chars > 0 or any(w in msg_lower for w in ['marna', 'chahta', 'chahti', 'jaan']):
            return EMERGENCY_RESPONSE_HI
        return EMERGENCY_RESPONSE_EN

    emotion = predict_emotion(message)

    # Tier 1. Try Groq Cloud if available
    groq_client = _get_groq_client()
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Predicted emotion signal: {emotion}\nUser message: {message}"
                    }
                ],
                max_tokens=300,
                temperature=0.8,
                stream=False
            )
            return response.choices[0].message.content.strip(), "None"
        except Exception as e:
            print(f"[AI Agent] Groq error: {e}")

    # Tier 2. Try Gemini Cloud if available
    gemini_client, gemini_legacy_model = _get_gemini_client()
    if gemini_client:
        try:
            prompt = f"{SYSTEM_PROMPT}\n\nPredicted emotion signal: {emotion}\nUser message: {message}"
            res = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            if res and res.text:
                return res.text.strip(), "None"
        except Exception as e:
            print(f"[AI Agent] Gemini error: {e}")
    elif gemini_legacy_model:
        try:
            prompt = f"{SYSTEM_PROMPT}\n\nPredicted emotion signal: {emotion}\nUser message: {message}"
            res = gemini_legacy_model.generate_content(prompt)
            if res and res.text:
                return res.text.strip(), "None"
        except Exception as e:
            print(f"[AI Agent] Gemini Legacy error: {e}")

    # Tier 3. Try Local Ollama LLM (e.g., MedGemma / Llama 3) if running locally
    ollama_res = _query_ollama_local(message, emotion)
    if ollama_res:
        return ollama_res, "ollama_local_llm"

    # Tier 4. Instant Local Emotion-Aware ML Fallback (0ms latency, 100% offline)
    fallback = _generate_fallback_response(message, emotion)
    return fallback, "None"



