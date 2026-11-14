
import os
import json
import urllib.request
try:
    from emotion_model import predict_emotion
except Exception:
    def predict_emotion(text: str) -> str:
        return "neutral"
from config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    OLLAMA_URL,
    OLLAMA_MODEL,
)


# ── High-Speed Model Configurations ───────────────────────────────────────────
GEMINI_FAST_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

OPENROUTER_FAST_MODELS = [
    "baidu/ernie-5.5-flash-preview:free",
    "inclusionai/ling-3.2-flash:free",
    "nvidia/nemotron-3.5-lightning:free",
]


def _stream_gemini_direct(prompt: str):
    """Direct Server-Sent Events stream to Gemini API for sub-1.5s first-token streaming."""
    if not GEMINI_API_KEY:
        return
    for model_name in GEMINI_FAST_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 140, "temperature": 0.7}
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=2.5)
            has_yielded = False
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        d = json.loads(line[6:])
                        txt = d.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if txt:
                            has_yielded = True
                            yield txt
                    except Exception:
                        pass
            if has_yielded:
                return
        except Exception:
            continue


def _query_gemini_direct(prompt: str) -> str | None:
    """Direct HTTP call to Gemini API for fast sub-second completion."""
    if not GEMINI_API_KEY:
        return None
    for model_name in GEMINI_FAST_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 140, "temperature": 0.7}
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                d = json.loads(resp.read().decode("utf-8"))
                txt = d.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if txt and txt.strip():
                    return txt.strip()
        except Exception:
            continue
    return None


def _query_openrouter(messages: list[dict]) -> str | None:
    """Query OpenRouter API using verified fast free models."""
    if not OPENROUTER_API_KEY:
        return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    for model in OPENROUTER_FAST_MODELS:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"model": model, "messages": messages, "max_tokens": 140, "temperature": 0.7}).encode("utf-8"),
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "User-Agent": "SafeSpace-AI"}
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    d = json.loads(resp.read().decode("utf-8"))
                    msg = d.get("choices", [{}])[0].get("message", {})
                    content = (msg.get("content") or "").strip()
                    if content:
                        return content
        except Exception:
            continue
    return None


def _stream_openrouter(messages: list[dict]):
    """Stream from OpenRouter API using verified fast free models."""
    if not OPENROUTER_API_KEY:
        return
    url = "https://openrouter.ai/api/v1/chat/completions"
    for model in OPENROUTER_FAST_MODELS:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"model": model, "messages": messages, "max_tokens": 140, "temperature": 0.7, "stream": True}).encode("utf-8"),
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "User-Agent": "SafeSpace-AI"}
            )
            resp = urllib.request.urlopen(req, timeout=2.5)
            has_yielded = False
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data: "):
                    chunk_str = line[6:]
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(chunk_str)
                        choice = chunk_json.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        if content:
                            has_yielded = True
                            yield content
                    except Exception:
                        pass
            if has_yielded:
                return
        except Exception:
            continue


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
        return None, genai_legacy
    except Exception as e:
        print(f"[AI Agent] Gemini init warning: {e}")
        return None, None



SYSTEM_PROMPT = """You are SafeSpace AI — a caring, warm, and concise mental health companion.

CRITICAL INSTRUCTIONS FOR RESPONSE LENGTH & FORMAT:
1. STRICTLY CONCISE: Never write long essays, lectures, or walls of text. Users never want to read long paragraphs when stressed.
2. Length limit: Strictly keep your entire reply under 2 to 4 sentences (maximum 40 to 65 words total).
3. Tone: Warm, human, and caring — like a supportive best friend sending a quick text message.
4. Structure:
   - Sentence 1: Validate their feelings warmly with authentic empathy.
   - Sentence 2: Offer 1 simple, bite-sized grounding thought or micro-step (no academic theories or long bullet lists).
   - Sentence 3: Close with a short, gentle check-in question.
5. Language: Always communicate in clean, compassionate English."""

EMERGENCY_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "self harm", "hurt myself", "cut myself", "no reason to live",
    "cant go on", "can't go on", "don't want to be here", "better off dead",
    "slit my wrist", "overdose", "hanging myself", "die tonight",
    "marna chahta", "marna chahti", "jaan dena", "khud ko hurt", "marne ka man",
    "jeena nahi chahta", "jeena nahi chahti", "zindagi khatam"
]

EMERGENCY_RESPONSE_EN = (
    "🚨 **I hear how much pain you're in, and I care deeply about your life and safety.** 💙\n\n"
    "You do not have to carry this heavy burden all alone. There are compassionate, trained professionals waiting to support you right now, free and confidential:\n\n"
    "• **India**: Call **Tele-MANAS** at **14416** / **1800-891-4416** or **iCall** at **9152987821**\n"
    "• **USA & Canada**: Call or text **988** (Suicide & Crisis Lifeline)\n"
    "• **UK**: Call **111** (NHS) or **116 123** (Samaritans)\n"
    "• **Emergency Services**: Dial **112** (India/EU) or **911** (US)\n\n"
    "Please take one slow breath with me. Your life has immense value — please reach out to one of these services or a loved one right now.",
    "emergency_call_tool"
)

EMERGENCY_RESPONSE_HI = (
    "🚨 **Main aapki baat sun raha hoon, aur aapki life bohot zyada keemti hai.** 💙\n\n"
    "Aap akele nahi hain aur aapko ye takleef akele jhelne ki zaroorat nahi hai. Kripya abhi in free helplines par call karein:\n\n"
    "• **Tele-MANAS**: Call **14416** ya **1800-891-4416** (24/7 Govt Free Helpline)\n"
    "• **iCall (TISS)**: Call **9152987821**\n"
    "• **Vandrevala Foundation**: Call **+91 9999 666 555**\n"
    "• **National Emergency**: Dial **112**\n\n"
    "Ek lambi gehri saans lo. Madad available hai aur log aapki care karte hain. Kripya turant baat karein.",
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
        with urllib.request.urlopen(req, timeout=0.3) as resp:
            if resp.status == 200:
                result = json.loads(resp.read().decode("utf-8"))
                content = result.get("message", {}).get("content", "").strip()
                if content:
                    return content
    except Exception:
        pass
    return None


def _generate_fallback_response(message: str, emotion: str) -> str:
    """Generate a warm, concise, 2-3 sentence best-friend response when offline."""
    import re
    msg_lower = message.lower().strip()
    emo = (emotion or "").lower()

    if bool(re.search(r'\b(what (i|to) (will|can|should) do|how to|future|direction|plan|planning|goal|goals|path|career|where to start|what next|confused about|figure out|action plan|stuck|roadmap|what i do)\b', msg_lower)):
        return (
            "Feeling unsure about where to start is completely okay 💙 "
            "Don't try to solve the entire future today—just pick one tiny 5-minute task to build momentum. "
            "What's the main goal on your mind right now?"
        )

    if bool(re.search(r'\b(procrastinat|can\'t start|cant start|cannot start|lazy|laziness|focus|distract|delay|postpone|motivation)\b', msg_lower)):
        return (
            "Please don't be hard on yourself 💙 Procrastination is usually just your mind feeling overwhelmed by the size of the task. "
            "Try the 5-minute rule: commit to working for just 300 seconds, and you can stop after that. "
            "What task are you putting off right now?"
        )

    if bool(re.search(r'\b(can\'t decide|cant decide|confused between|which option|decision|decide|choices|paralysis)\b', msg_lower)):
        return (
            "Decision paralysis is exhausting, but we'll work through it together 💙 "
            "Ask yourself: will this matter in 10 months? "
            "What two options are you stuck between?"
        )

    if bool(re.search(r'\b(study|studies|exam|exams|marks|score|college|job|interview|career|workload|assignment)\b', msg_lower)):
        return (
            "I hear you, and it's completely normal to feel the weight of deadlines and exams 💙 "
            "Take one slow, deep breath and let's tackle just one section at a time. "
            "What subject or task feels most stressful right now?"
        )

    if bool(re.search(r'\b(yoga|stretch|stretching|pose|asana|workout|pranayama|flexibility)\b', msg_lower)):
        return (
            "A quick stretch can do wonders to release physical tension 💙 "
            "Try Child's Pose or a gentle neck roll for 60 seconds while taking slow, deep breaths. "
            "How does your body feel right now?"
        )

    if bool(re.search(r'\b(hi|hello|hey|namaste|greetings|good morning|good evening|sup)\b', msg_lower)):
        return "Hey there! I'm SafeSpace AI, your personal companion 💙 How are you feeling today?"

    if bool(re.search(r'\b(breath|breathing|respirat|inhale|exhale)\b', msg_lower)):
        return (
            "Let's take a calming breath together 💙 "
            "Inhale slowly for 4 seconds, hold gently for 4, and exhale smoothly for 6. "
            "How does your chest feel right now?"
        )

    if bool(re.search(r'\b(sleep|sleeping|insomnia|tired|night|bedtime|awake|cant sleep|can\'t sleep)\b', msg_lower)):
        return (
            "Lying awake when you need rest is so frustrating 💙 "
            "Dim your screen, unclench your jaw, and focus on slow breaths without pressuring yourself to sleep immediately. "
            "What's keeping your mind busy tonight?"
        )

    if bool(re.search(r'\b(overthink|thinking|brain|mind won\'t|thoughts)\b', msg_lower)):
        return (
            "Take a deep breath with me right now 💙 "
            "Your mind is running in circles trying to protect you, but you are safe in this exact moment. "
            "What's the loudest thought right now?"
        )

    if bool(re.search(r'\b(stress|tension|pressure|busy|burnout)\b', msg_lower)):
        return (
            "I hear you, and it's completely valid that you're feeling the pressure right now 💙 "
            "Drop your shoulders, take one slow breath, and remember you don't have to carry everything at once. "
            "What feels heaviest today?"
        )

    if bool(re.search(r'\b(lonely|alone|breakup|relationship|no one|nobody|friend)\b', msg_lower)):
        return (
            "I'm right here with you, and whatever you're feeling is completely valid 💙 "
            "You don't have to carry this heaviness all by yourself. "
            "Do you want to talk about what's making you feel lonely today?"
        )

    if bool(re.search(r'\b(angry|frustrated|annoyed|mad|hate|furious)\b', msg_lower)):
        return (
            "It's completely okay to feel angry or frustrated 💙 "
            "Anger is a natural signal that something feels unfair or overwhelming. "
            "What triggered this feeling for you?"
        )

    if bool(re.search(r'\b(anxious|anxiety|panic|scared|fear|nervous|terrified)\b', msg_lower)):
        return (
            "You are safe, and I am right here with you 💙 "
            "Anxiety is just a temporary alarm in your nervous system. Look around and name 3 things you can see right now to ground yourself. "
            "What's making you feel anxious?"
        )

    if bool(re.search(r'\b(sad|depress|crying|cry|hopeless|heartbroken|down)\b', msg_lower)):
        return (
            "I'm so sorry things feel heavy right now 💙 "
            "It is completely okay not to be okay, and you don't have to force a smile. "
            "Would it help to talk about what's bringing you down?"
        )

    if bool(re.search(r'\b(useless|failure|not good enough|can\'t do it|worthless|disappointed)\b', msg_lower)):
        return (
            "Please remember that one hard moment or mistake doesn't define who you are 💙 "
            "You are doing your best, and your worth isn't tied to being perfect. "
            "What made you doubt yourself today?"
        )

    if bool(re.search(r'\b(happy|good|great|fine|awesome|thanks|thank|yay|excited)\b', msg_lower)):
        return "That makes me so happy to hear! 🌟 Savor this moment and let that good feeling sink in. What made your day so great?"

    if bool(re.search(r'\b(help|can you|what can you do|advice|suggest|guide|options)\b', msg_lower)):
        return (
            "I'm here as your caring companion and problem solver 💙 "
            "Whether you need quick stress relief, a clear next step, or just someone to listen—I've got your back. "
            "What would be most helpful right now?"
        )

    # General fallback based on emotion
    if "fear" in emo or "anxiety" in emo:
        return "You are safe and I'm right here with you 💙 Take one slow, grounding breath. What's on your mind right now?"
    elif "sad" in emo:
        return "I hear you, and it's okay to feel sad today 💙 Take things gently right now. Would you like to share what's bothering you?"
    elif "anger" in emo:
        return "I hear your frustration, and your feelings are completely valid 💙 Take a deep breath to create some space. What happened?"
    else:
        return "I hear you, and whatever you're experiencing is completely valid 💙 Take one slow breath with me. Tell me a bit more about what's on your mind."


def _build_groq_messages(message: str, emotion: str, history: list[dict] = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for turn in history[-8:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = turn.get("content") or turn.get("message") or ""
            if content.strip():
                msgs.append({"role": role, "content": content})
    msgs.append({
        "role": "user",
        "content": f"[Predicted emotion signal: {emotion}]\nUser message: {message}"
    })
    return msgs


def _build_gemini_prompt(message: str, emotion: str, history: list[dict] = None) -> str:
    convo = ""
    if history:
        for turn in history[-6:]:
            speaker = "User" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content") or turn.get("message") or ""
            if content.strip():
                convo += f"{speaker}: {content}\n"
    prompt = f"{SYSTEM_PROMPT}\n\n"
    if convo:
        prompt += f"Recent Conversation History:\n{convo}\n"
    prompt += f"Predicted emotion signal: {emotion}\nUser: {message}\nAssistant:"
    return prompt


def get_response(message: str, history: list[dict] = None) -> tuple[str, str]:
    msg_lower = message.lower()

    # Emergency crisis check (instant)
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

    # Tier 1. Gemini Cloud Direct REST (~1.2s latency)
    if GEMINI_API_KEY:
        gemini_prompt = _build_gemini_prompt(message, emotion, history)
        gemini_res = _query_gemini_direct(gemini_prompt)
        if gemini_res:
            return gemini_res, "None"

    # Tier 2. OpenRouter Fast Cloud Models (~1.2s - 2s latency)
    if OPENROUTER_API_KEY:
        router_messages = _build_groq_messages(message, emotion, history)
        router_res = _query_openrouter(router_messages)
        if router_res:
            return router_res, "None"

    # Tier 3. Groq Cloud (Single fast attempt, 1.5s timeout)
    groq_client = _get_groq_client()
    if groq_client:
        try:
            groq_messages = _build_groq_messages(message, emotion, history)
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=groq_messages,
                max_tokens=140,
                temperature=0.7,
                timeout=1.5,
                stream=False
            )
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip(), "None"
        except Exception:
            pass

    # Tier 4. Instant Built-in Best-Friend Companion Engine (0ms latency, 100% offline)
    fallback = _generate_fallback_response(message, emotion)
    return fallback, "None"


def stream_response(message: str, history: list[dict] = None):
    """
    Generator yielding tokens in real-time with sub-1.5s first-token latency.
    Yields: {"chunk": str, "tool_called": str, "done": bool}
    """
    import time
    msg_lower = message.lower()

    # Emergency crisis check (instant)
    if any(kw in msg_lower for kw in EMERGENCY_KEYWORDS):
        try:
            from tools import call_emergency
            call_emergency()
        except Exception:
            pass
        hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097f')
        resp_text, tool = EMERGENCY_RESPONSE_HI if (hindi_chars > 0 or any(w in msg_lower for w in ['marna', 'chahta', 'chahti', 'jaan'])) else EMERGENCY_RESPONSE_EN
        yield {"chunk": resp_text, "tool_called": tool, "done": True}
        return

    emotion = predict_emotion(message)

    # Tier 1. Gemini Direct Streaming (~1.3s first token!)
    if GEMINI_API_KEY:
        gemini_prompt = _build_gemini_prompt(message, emotion, history)
        yielded_any = False
        for chunk in _stream_gemini_direct(gemini_prompt):
            if chunk:
                yielded_any = True
                yield {"chunk": chunk, "tool_called": "None", "done": False}
        if yielded_any:
            yield {"chunk": "", "tool_called": "None", "done": True}
            return

    # Tier 2. OpenRouter Fast Free Streaming (~1.5s - 2s)
    if OPENROUTER_API_KEY:
        router_messages = _build_groq_messages(message, emotion, history)
        yielded_any = False
        for chunk in _stream_openrouter(router_messages):
            if chunk:
                yielded_any = True
                yield {"chunk": chunk, "tool_called": "None", "done": False}
        if yielded_any:
            yield {"chunk": "", "tool_called": "None", "done": True}
            return

    # Tier 3. Groq Fast Streaming (Single attempt, 1.5s timeout)
    groq_client = _get_groq_client()
    if groq_client:
        try:
            groq_messages = _build_groq_messages(message, emotion, history)
            stream = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=groq_messages,
                max_tokens=140,
                temperature=0.7,
                timeout=1.5,
                stream=True
            )
            yielded_any = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                if delta:
                    yielded_any = True
                    yield {"chunk": delta, "tool_called": "None", "done": False}
            if yielded_any:
                yield {"chunk": "", "tool_called": "None", "done": True}
                return
        except Exception:
            pass

    # Tier 4. Instant Built-in Fallback (0ms latency, streams immediately word-by-word)
    fallback = _generate_fallback_response(message, emotion)
    words = fallback.split(" ")
    for i, w in enumerate(words):
        yield {"chunk": w + (" " if i < len(words) - 1 else ""), "tool_called": "None", "done": False}
        time.sleep(0.012)
    yield {"chunk": "", "tool_called": "None", "done": True}





