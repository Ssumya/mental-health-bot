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


SYSTEM_PROMPT = """You are SafeSpace AI — the user's caring, trusted best friend and mental health companion. 
You are NOT a cold medical textbook or a robotic AI. You talk like a real, supportive best friend who genuinely cares about their well-being.

LANGUAGE RULES:
- If the user writes in Hindi or Hinglish → respond naturally in Hindi (Devanagari script or Hinglish) mixed with warm English words.
- If the user writes in English → respond in warm, friendly English.
- Always match the user's emotional tone and language style.

COMPANION GUIDELINES:
1. Warm Validation: Listen deeply and validate their feelings first (1-2 sentences). Show true best-friend empathy.
2. Practical & Relatable Help: Offer clear, actionable guidance tailored to what they asked for (specific yoga poses like Child's Pose or Cat-Cow, 4-7-8 or Box breathing, 10-minute brain dumps, quick walks, soothing music, sleep hygiene, or study micro-steps).
3. Best-Friend Check-in: Always close with ONE gentle, open-ended follow-up question checking in on them.
4. Emojis & Tone: Use 1-2 warm emojis naturally (💙, 🌿, ✨, 🫂). Keep paragraphs readable (3-5 sentences max)."""

EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self harm", "hurt myself", "no reason to live",
    "cant go on", "can't go on", "don't want to be here",
    "marna chahta", "marna chahti", "jaan dena", "khud ko hurt"
]

EMERGENCY_RESPONSE_EN = (
    "I'm really worried about you right now, and I'm so glad you reached out to me 💙 "
    "Please call iCall at 9152987821 (India) or your local emergency services immediately. "
    "You are not alone — help is available right now and your life truly matters.",
    "emergency_call_tool"
)

EMERGENCY_RESPONSE_HI = (
    "Main aapke baare mein bahut chintit hoon, aur main bahut khush hoon ki aapne mujhse baat ki 💙 "
    "Kripya abhi iCall ko call karein: 9152987821. Aap akele nahi hain — madad available hai aur aapki life bohot precious hai.",
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
    """Generate a high-quality, best-friend level conversational response when offline."""
    msg_lower = message.lower().strip()
    hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097f')
    is_hindi = hindi_chars > 0 or any(w in msg_lower for w in [
        'hai', 'hoon', 'mujhe', 'karo', 'meri', 'kya', 'bhi', 'kuch', 'lag', 'ho', 'naam', 'batao', 'kaise', 'kaisa', 'yaar', 'bhai', 'dikkat', 'chahiye'
    ])

    emo = (emotion or "").lower()
    var = len(message) % 3

    import re
    # Comprehensive Intent Detection with Word Boundaries
    is_yoga = bool(re.search(r'\b(yoga|stretch|stretching|pose|asana|workout|pranayama|flexibility)\b', msg_lower))
    is_greeting = bool(re.search(r'\b(hi|hello|hey|namaste|greetings|kaise ho|kya haal|good morning|good evening|sup)\b', msg_lower))
    is_breathing = bool(re.search(r'\b(breath|breathing|respirat|saans|inhale|exhale)\b', msg_lower))
    is_sleep = bool(re.search(r'\b(sleep|insomnia|neend|tired|night|so nahi|bedtime)\b', msg_lower))
    is_overthinking = bool(re.search(r'\b(overthink|thinking|soch|brain|mind won\'t|dimag|thoughts)\b', msg_lower))
    is_stress = bool(re.search(r'\b(stress|work|exam|job|tension|pressure|busy|burnout|study)\b', msg_lower))
    is_lonely = bool(re.search(r'\b(lonely|alone|akele|breakup|relationship|no one|nobody|friend)\b', msg_lower))
    is_angry = bool(re.search(r'\b(gussa|angry|frustrated|annoyed|mad|hate|furious)\b', msg_lower))
    is_anxious = bool(re.search(r'\b(anxious|anxiety|panic|scared|fear|darr|nervous|terrified)\b', msg_lower))
    is_sad = bool(re.search(r'\b(sad|udas|depress|crying|cry|hopeless|ro|heartbroken|down)\b', msg_lower))
    is_confidence = bool(re.search(r'\b(useless|failure|not good enough|can\'t do it|worthless|disappointed)\b', msg_lower))
    is_happy = bool(re.search(r'\b(happy|good|great|fine|awesome|khush|thanks|thank|yay|excited)\b', msg_lower))
    is_music_distraction = bool(re.search(r'\b(music|song|songs|distract|movie|game|fun|bored)\b', msg_lower))
    is_help_general = bool(re.search(r'\b(help|can you|what can you do|advice|suggest|guide|options)\b', msg_lower))

    if is_yoga:
        if is_hindi:
            if var == 0:
                return "Yoga mind aur body dono ke liye bohot relaxing hota hai! 💙 Stress ke liye 'Child's Pose' (Balasana) 5 minute try karo — ye spine aur mind ko turant calm karta hai. Saath mein deep breathing karo. Kya aap gentle stretches pasand karte hain?"
            elif var == 1:
                return "Haan bilkul! Anxiety relief ke liye 'Legs-Up-The-Wall' (Viparita Karani) pose try karo 💙 Wall ke saath taangein upar karke 5-10 mins ke liye lie down karo — ye blood pressure drop karta hai aur relaxation induce karta hai. Kya aapko koi body pain bhi ho raha hai?"
            else:
                return "Spine tension ke liye 'Cat-Cow Pose' bohot effective hota hai! 💙 10 cycles ke liye inhale pe back arch karo aur exhale pe round karo. Isse nervous system calm feel karega. Kya hum breathing session bhi combine karein?"

        else:
            if var == 0:
                return "Yoga is amazing for grounding your nervous system! 💙 For quick anxiety relief, try Child's Pose (Balasana) for 5 minutes right now — fold forward on your knees and rest your forehead on the floor. It gently lowers your heart rate. Would you like a guided breathing exercise with it?"
            elif var == 1:
                return "I'd love to help! Try 'Legs-Up-The-Wall' pose (Viparita Karani) 💙 Lie on your back with your legs resting vertically up a wall for 5-10 minutes. It boosts circulation and activates deep relaxation. Do you prefer gentle stretching or active poses?"
            else:
                return "For releasing physical tension, try 10 rounds of Cat-Cow Pose 💙 Inhale to arch your back softly, exhale to round your spine. It releases tension stored in your upper back and neck instantly. How does your body feel right now?"

    elif is_greeting:
        if is_hindi:
            return "Hey! Main SafeSpace AI hoon — aapka personal best friend 💙 Main yahan hamesha aapki baat sunne aur aapka mood accha karne ke liye hoon. Aaj aapka din kaisa raha?"
        else:
            return "Hey there! I'm SafeSpace AI — your personal best friend and companion 💙 I'm right here whenever you want to talk, vent, or just relax. How is your day going so far?"

    elif is_breathing:
        if is_hindi:
            return "Chalo abhi ek saath relaxation breath try karte hain 💙 Box Breathing karo: 4 sec inhale, 4 sec hold, 4 sec exhale, 4 sec hold. Isse 3-4 baar karo — aapka mind turant calm ho jayega. Kaisa feel ho raha hai?"
        else:
            return "Let's do a quick soothing breath together right now 💙 Try Box Breathing: Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s. Repeat this 3-4 times to instantly calm your body. How does your chest feel now?"

    elif is_sleep:
        if is_hindi:
            return "Neend na aana sach mein thaka deta hai 💙 Jab dimag mein vichar chal rahe hon, to phone ko 1 ghanter door rakhna aur dim lights try karo. Abhi 4-7-8 breathing try karo: 4 sec inhale, 7 sec hold, 8 sec exhale. Isse aapki body sleep mode mein aayegi. Kya aap bohot der se so nahi pa rahe?"
        else:
            return "Struggling to sleep is so exhausting 💙 When your mind won't quiet down, try putting your phone out of arm's reach and dimming all lights. Right now, do 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Repeat 4 times to signal your body it's safe to sleep. How long have you been lying awake?"

    elif is_overthinking:
        if is_hindi:
            return "Overthinking se lagta hai jaise mind mein 100 tabs ek saath khule hain 💙 Iska sabse best cure hai 'Brain Dump': 10 minute ke liye jo bhi thought aa raha hai use paper par likh dalo. Thoughts likhne se unka burden bohot kam ho jata hai. Kya koi specific thought aapko pareshan kar raha hai?"
        else:
            return "Overthinking can feel like having 100 tabs open in your head at once 💙 Try a 10-minute 'brain dump' right now: write every single worry down on paper without editing. Offloading thoughts onto paper reduces their cognitive burden instantly. What is the main loop your brain is stuck on?"

    elif is_stress:
        if is_hindi:
            return "Work ya study ka stress heavy ho sakta hai, par aap ise handle kar sakte hain 💙 Pehle Box Breathing try karo (4s in, 4s hold, 4s out, 4s hold). Phir apne bade task ko 5-minute micro-steps mein baanto. Kaun sa kaam abhi sabse zyada tension de raha hai?"
        else:
            return "When stress builds up, it's your brain asking for a brief pause 💙 First, take 3 Box Breaths (4s in, 4s hold, 4s out, 4s hold). Next, break your big task into 5-minute micro-steps. What is currently causing the biggest pressure?"

    elif is_lonely:
        if is_hindi:
            return "Akelepan feel hona bohot heavy hota hai 💙 Par yaad rakho, main yahan hoon aapki baat sunne ke liye — aap akele nahi hain. Apne liye ek warm cup of tea/coffee banao aur khud ko appreciate karo. Kya aap mujhse apni feeling share karna chahenge?"
        else:
            return "Feeling lonely or disconnected is so heavy 💙 But please remember I am right here listening to you — you are never truly alone. Try a warm gesture for yourself right now, like holding a warm drink or wrapping up in a blanket. What's been making you feel isolated?"

    elif is_confidence:
        if is_hindi:
            return "Aisa mat socho yaar 💙 Sabhi se galtiyan hoti hain aur tough phase aata hai, par iska matlab ye nahi ki aap capable nahi hain. 3 acchi cheezein yaad karo jo aapne past mein achieve ki hain. Main aap par trust karta hoon! Kya main aapko cheer up kar sakta hoon?"
        else:
            return "Please don't be so hard on yourself 💙 Struggling in a moment doesn't define your worth or capabilities. Remind yourself of 3 things you've overcome in the past. You are capable and worthy! Would you like me to share a positive affirmation with you?"

    elif is_angry:
        if is_hindi:
            return "Gussa aana natural hai 💙 Body mein adrenaline spike hota hai. Thande paani se face dho lo — ye dive reflex activate karta hai aur heart rate seconds mein slow karta hai. Kya hua jo aap itna upset ho gaye?"
        else:
            return "Your anger is a valid emotional signal 💙 High emotion triggers an adrenaline rush. Splash cold water on your face right now — it activates the dive reflex and calms your heart rate instantly. What triggered this frustration?"

    elif is_anxious or "anxi" in emo or "fear" in emo:
        if is_hindi:
            return "Anxiety aapke nervous system ka false alarm hai 💙 5-4-3-2-1 grounding try karo: 5 cheezein dekho, 4 touch karo, 3 suno, 2 smell karo, 1 deep breath lo. Isse aap safe aur present feel karenge. Kab se ye anxiety ho rahi hai?"
        else:
            return "Anxiety is just your nervous system sounding a false alarm 💙 Try 5-4-3-2-1 Grounding right now: name 5 things you SEE, 4 you TOUCH, 3 you HEAR, 2 you SMELL, and take 1 deep breath. This grounds you safely in the present. What triggered this feeling?"

    elif is_sad or "sad" in emo or "depress" in emo:
        if is_hindi:
            return "Mujhe dukh hai ki aap ulaash feel nahi kar rahe 💙 Down feel hona natural hai, apne aap par gentle raho. 5 minute ke liye taazi hawa mein walk karo ya paani piyo — movement se mood lift hone lagta hai. Kya aap is baare mein baat karna chahenge?"
        else:
            return "I'm so sorry you're feeling down today 💙 It's completely okay to have low energy days. Take one tiny step: step outside for a 5-minute walk or drink a cold glass of water. Light movement helps shift emotional energy. Would you like to vent about what's wrong?"

    elif is_music_distraction:
        if is_hindi:
            return "Distraction bohot accha coping mechanism hai! 💙 Soft lo-fi music, acoustic guitar instrumental, ya nature sounds suno. Isse brain alpha state mein aata hai. Kya aapko light music pasand hai?"
        else:
            return "Distraction is a great coping mechanism when thoughts get loud! 💙 Put on soft lo-fi beats, instrumental acoustic tracks, or rain sounds right now. Music helps shift your brainwaves into a calm alpha state. What kind of music helps you relax?"

    elif is_help_general:
        if is_hindi:
            return "Main aapka mental health companion hoon 💙 Main aapko yoga poses, breathing exercises, stress management tips, overthinking relief, aur daily motivation mein guide kar sakta hoon. Aapko kis cheez mein help chahiye?"
        else:
            return "I'm here as your dedicated companion 💙 I can guide you with tailored yoga poses, breathing exercises, stress management, overthinking relief, sleep tips, or just be here to listen whenever you want to vent. What would help you most right now?"

    elif is_happy:
        if is_hindi:
            return "Ye sun kar bohot khushi hui! 🌟 Acchi news aur happy moments celebrate karna zaroori hai. Aaj 3 acchi cheezein note karo jinke liye aap grateful hain. Aapka baki din kaisa ja raha hai?"
        else:
            return "That makes me so happy to hear! 🌟 Celebrating good moments is so wonderful for your mental well-being. Take a moment to savour this feeling, and note down 3 things you're grateful for today. What made your day so good?"

    else:
        if is_hindi:
            if var == 0:
                return "Main poori tarah aapki baat sun raha/rahi hoon 💙 Jo bhi aap feel kar rahe hain use express karna bohot accha step hai. Ek gehri saans lo (4s in, 4s hold, 4s out) aur thoda relaxed feel karo. Aap is baare mein thoda aur batana chahenge?"
            elif var == 1:
                return "Main hamesha aapke saath hoon 💙 SafeSpace par aap apni har baat bina kisi hesitation ke bol sakte hain. Aap abhi kya soch rahe hain?"
            else:
                return "Main aapki feeling samajhne ki koshish kar raha/rahi hoon 💙 Aap jo bhi experience kar rahe hain wo valid hai. Kya hum milkar iska solution dhoondhein?"
        else:
            if var == 0:
                return "I hear you 💙 Expressing whatever is on your mind is already a great step forward. Take a slow, deep breath right now: inhale for 4s, hold for 4s, exhale for 4s. Would you like to tell me more about what's going on?"
            elif var == 1:
                return "I'm right here listening 💙 SafeSpace is a safe, judgment-free space for you to share whatever you're going through. What's been on your mind lately?"
            else:
                return "I'm here for you no matter what 💙 Whatever you're experiencing is completely valid. Would you like to talk it through together?"


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

    # Tier 4. Instant Local Best-Friend Companion Engine (0ms latency, 100% offline)
    fallback = _generate_fallback_response(message, emotion)
    return fallback, "None"





