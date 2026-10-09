import os
import json
import urllib.request
from dotenv import load_dotenv
from emotion_model import predict_emotion

# Load environment variables from .env if present
load_dotenv()

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
    """Generate a dynamic, topic-matched empathetic response when offline."""
    msg_lower = message.lower().strip()
    hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097f')
    is_hindi = hindi_chars > 0 or any(w in msg_lower for w in ['hai', 'hoon', 'mujhe', 'karo', 'meri', 'kya', 'bhi', 'kuch', 'lag', 'ho', 'naam', 'batao', 'kaise', 'kaisa'])

    emo = (emotion or "").lower()

    # Topic detection
    is_greeting = any(w in msg_lower for w in ['hi', 'hello', 'hey', 'namaste', 'greetings', 'kaise ho', 'kya haal', 'good morning', 'good evening'])
    is_sleep = any(w in msg_lower for w in ['sleep', 'insomnia', 'neend', 'tired', 'night', 'so nahi'])
    is_overthinking = any(w in msg_lower for w in ['overthink', 'thinking', 'soch', 'brain', 'mind won\'t', 'dimag'])
    is_stress = any(w in msg_lower for w in ['stress', 'work', 'exam', 'job', 'tension', 'pressure', 'busy'])
    is_lonely = any(w in msg_lower for w in ['lonely', 'alone', 'akele', 'breakup', 'relationship', 'no one'])
    is_angry = any(w in msg_lower for w in ['gussa', 'angry', 'frustrated', 'annoyed', 'mad', 'hate'])
    is_anxious = any(w in msg_lower for w in ['anxious', 'anxiety', 'panic', 'scared', 'fear', 'darr', 'nervous'])
    is_sad = any(w in msg_lower for w in ['sad', 'udas', 'depress', 'crying', 'cry', 'hopeless', 'ro'])
    is_happy = any(w in msg_lower for w in ['happy', 'good', 'great', 'fine', 'awesome', 'khush', 'thanks', 'thank'])
    is_yoga_exercise = any(w in msg_lower for w in ['yoga', 'exercise', 'stretch', 'stretching', 'pose', 'workout', 'asana', 'pranayama'])

    var = len(message) % 2

    if is_greeting:
        if is_hindi:
            return "Namaste! Main SafeSpace AI hoon — aapka personal mental health companion 💙 Main yahan aapki baat sunne aur aapko support karne ke liye hoon. Aap aaj kaisa feel kar rahe hain?"
        else:
            return "Hello! I'm SafeSpace AI — your personal mental health companion 💙 I'm here to listen, support, and help you navigate whatever is on your mind. How are you feeling today?"

    elif is_yoga_exercise:
        if is_hindi:
            return "Yoga aur gentle stretching aapke nervous system ko calm karne ke liye bohot accha hai! 💙 Stress aur anxiety ke liye 'Child's Pose' (Balasana) ya 'Legs-Up-The-Wall' (Viparita Karani) 5 minute ke liye try karo — ye heart rate ko slow karta hai aur blood pressure ko lower karta hai. Kya aap koi specific body area relax karna chahte hain?"
        else:
            return "Yoga and gentle stretching are fantastic for soothing your mind and nervous system! 💙 For anxiety and stress relief, try Child's Pose (Balasana) or Legs-Up-The-Wall pose for 5 minutes right now — they lower your heart rate and release physical tension instantly. Would you like a step-by-step stretching or breathing guide?"

    elif is_sleep:
        if is_hindi:
            return "Neend na aana bahut frustrating hota hai 💙 Jab body thak jaati hai par mind chalta rehta hai, to phone ko 1 ghanter pehle door rakhna aur dim lights try karo. Abhi 4-7-8 breathing try karo: 4 sec inhale, 7 sec hold, 8 sec exhale. Isse aapki body naturally relaxation mode mein aayegi. Kya aap kafi time se so nahi pa rahe?"
        else:
            return "Struggling to sleep can be so exhausting 💙 When your body is tired but your mind keeps racing, try dimming lights and keeping screens away. Right now, try 4-7-8 breathing: inhale for 4s, hold for 7s, exhale for 8s. Do this 4 times to signal your nervous system it's safe to rest. How long has sleep been difficult for you?"

    elif is_overthinking:
        if is_hindi:
            return "Overthinking se aisa lagta hai jaise mind mein 100 tabs ek saath khule hain 💙 Iska best solution hai 'brain dump': 10 minute ke liye jo bhi thoughts aa rahe hain unhe ek paper par likh dalo. Jab vichar likh diye jaate hain, to unka burden kam ho jata hai. Kya aap bata sakte hain kaunsa thought baar baar repeat ho raha hai?"
        else:
            return "Overthinking can feel like having 100 tabs open in your brain all at once 💙 A great technique right now is a 10-minute 'brain dump' — write down every single worry on paper without filtering. Offloading thoughts onto paper reduces their cognitive weight instantly. What's the main thought loop you're stuck in?"

    elif is_stress:
        if is_hindi:
            return "Work ya life ka stress jab heavy feel hone lage, to ek chota break lena zaroori hai 💙 Abhi try karo Box Breathing: 4 sec inhale, 4 sec hold, 4 sec exhale, 4 sec hold. Ye Navy SEALs ki technique nervous system ko turant calm karti hai. Phir apne sabse bade task ko 5-minute micro-steps mein baanto. Kon si cheez abhi sabse zyada tension de rahi hai?"
        else:
            return "When stress builds up, it's a sign your brain needs a quick pause 💙 Try Box Breathing right now: inhale for 4s, hold 4s, exhale 4s, hold 4s. This technique calms your heart rate instantly. Next, break your overwhelming task into 5-minute micro-steps. What is currently causing the most pressure?"

    elif is_lonely:
        if is_hindi:
            return "Akelepan ya isolation feel hona sach mein bohot painful hota hai 💙 Yaad rakho, feeling lonely doesn't mean you are truly alone — main yahan hoon aapki baat sunne ke liye. Ek chota self-care gesture karo: apne liye ek warm cup of tea/coffee banao aur khud ko appreciate karo. Kya aap is feeling ke baare mein thoda aur batana chahenge?"
        else:
            return "Feeling lonely or disconnected can be deeply painful 💙 Please remember that experiencing loneliness doesn't mean you are truly alone — I'm right here with you. Try a gentle self-compassion exercise right now: place your hand over your heart and take 3 deep breaths. What has been making you feel isolated recently?"

    elif is_angry:
        if is_hindi:
            return "Gussa aur frustration aana bilkul natural reaction hai 💙 Jab emotions high hote hain to body mein adrenaline rush hota hai. Thande paani se chehre ko dho lo — ye mammalian dive reflex activate karta hai aur heart rate ko seconds mein drop karta hai. Kya hua tha jo aapko itna upset feel hua?"
        else:
            return "Feeling angry or frustrated is a completely valid emotional response 💙 When anger hits, your body gets flooded with adrenaline. Splash cold water on your face right now — this triggers the mammalian dive reflex and slows your heart rate immediately. What triggered this frustration today?"

    elif is_anxious or "anxi" in emo or "fear" in emo:
        if is_hindi:
            return "Anxiety jab aati hai to aisi lagti hai jaise danger ho, par ye aapka nervous system overdrive mein hone ki वजह se hai 💙 Try 5-4-3-2-1 grounding: 5 cheezein dekho, 4 ko touch karo, 3 ko suno, 2 ko smell karo, aur 1 deep breath lo. Isse aapka mind present moment mein wapas aayega. Kab se ye anxiety ho rahi hai?"
        else:
            return "Anxiety makes your body feel like it's in danger, but it's just your nervous system in overdrive 💙 Try 5-4-3-2-1 Grounding right now: name 5 things you SEE, 4 you TOUCH, 3 you HEAR, 2 you SMELL, and take 1 deep breath. This grounds you back in the present. What triggered this feeling today?"

    elif is_sad or "sad" in emo or "depress" in emo:
        if is_hindi:
            return "Mujhe dukh hai ki aap aisi feeling se guzar rahe hain 💙 Low feel hona bilkul normal hai, apne aap par pressure mat dalo. Ek chota step lo: 5 minute ke liye baahar walk karo ya taazi hawa mein baitho. Light movement se bhi cortisol level drop hone lagta hai. Aap is baare mein thoda aur share karna chahenge?"
        else:
            return "I'm so sorry you're feeling down right now 💙 It's completely okay to have heavy days — please be gentle with yourself. Try taking one small action: step outside for a quick 5-minute walk or drink a cold glass of water. Light movement helps lower stress hormones. Would you like to share what's weighing on your heart?"

    elif is_happy:
        if is_hindi:
            return "Ye sun kar bohot accha laga! 🌟 Positive moments ko notice aur appreciate karna mental well-being ke liye bohot impactful hai. Is moment ko savour karo aur ho sake to aaj 3 acchi cheezein likho jinke liye aap grateful hain. Aapka din kaisa ja raha hai?"
        else:
            return "That's wonderful to hear! 🌟 Noticing and appreciating positive moments is so great for your mental well-being. Take a moment to savour this feeling, and try writing down 3 good things that happened today. How has the rest of your day been?"

    else:
        if is_hindi:
            if var == 0:
                return "Main samajh raha/rahi hoon 💙 Aap jo bhi experience kar rahe hain, use express karna pehla positive step hai. Abhi ek gehri saans lo: 4 sec inhale, 4 sec hold, 4 sec exhale. Isse mind clear aur calm feel karega. Aap is baare mein thoda aur batana chahenge?"
            else:
                return "Main yahan aapki baat sunne ke liye bilkul ready hoon 💙 SafeSpace par aap apni har feeling bina kisi judgment ke share kar sakte hain. Aap abhi kya soch rahe hain ya feel kar rahe hain?"
        else:
            if var == 0:
                return "I hear you 💙 Expressing whatever is on your mind is already a great step forward. Take a slow, deep breath right now: inhale for 4s, hold for 4s, exhale for 4s. This helps clear your mind. Would you like to tell me more about what's going on?"
            else:
                return "I'm right here listening 💙 SafeSpace is a judgment-free space for you to express whatever you're experiencing. What's been on your mind or how are you feeling right at this moment?"


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




