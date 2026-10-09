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


SYSTEM_PROMPT = """You are SafeSpace AI — the user's caring, trusted best friend, mental health companion, and ChatGPT-level Digital Library problem solver.
You combine deep psychological analysis (like ChatGPT breaking down root causes, mindset, and cognitive patterns) with the genuine warmth, empathy, and care of a best friend.

LANGUAGE RULES:
- If the user writes in Hindi or Hinglish → respond naturally in Hindi (Devanagari script or Hinglish) mixed with warm English words.
- If the user writes in English → respond in warm, friendly, clear English. NEVER switch to Hindi if the user speaks English!

RESPONSE STRUCTURE (Use clean markdown formatting with emojis):
1. 🫂 Best-Friend Validation (1-2 sentences): Show deep empathy and validate their feelings so they feel completely understood and supported.
2. 🧠 Mental & Situational Analysis (2-3 sentences): Perform a smart, ChatGPT-style analysis of what is happening in their mind or situation (e.g., decision fatigue, cognitive overload, anxiety false alarms, perfectionism paralysis, emotional exhaustion).
3. 📌 Digital Library Solutions (2-3 structured micro-steps): Provide concrete, actionable, step-by-step techniques tailored to their exact issue (e.g., 5-minute micro-goals, 2-minute rule, 10-minute brain dump, 10/10/10 rule, Box Breathing, Child's Pose / Cat-Cow yoga).
4. 💙 Warm Best-Friend Check-in: Close with ONE open-ended follow-up question inviting them to take the first tiny step or share what's on their mind."""

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
    """Generate a high-quality, ChatGPT-level analytical best-friend response when offline."""
    import re
    msg_lower = message.lower().strip()
    
    # Strict language detection using Devanagari script or exact word boundaries for Hinglish
    hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097f')
    is_hindi = hindi_chars > 0 or bool(re.search(
        r'\b(hai|hoon|mujhe|karo|meri|kya|bhi|kuch|lag|batao|kaise|kaisa|yaar|bhai|dikkat|chahiye|kaam|soch|samajh|raha|rahi|samajhna|bolo|karna|hoga|hogi|haan|nahi|nahin|thoda|apna|apni|teri|mera|mere|karoon|karu)\b', 
        msg_lower
    ))

    emo = (emotion or "").lower()
    var = len(message) % 3

    # Comprehensive Intent Detection with Regex Word Boundaries
    is_future_planning = bool(re.search(r'\b(what (i|to) (will|can|should) do|how to|future|direction|plan|planning|goal|goals|path|career|where to start|what next|confused about|figure out|action plan|stuck|roadmap|what i do)\b', msg_lower))
    is_procrastination = bool(re.search(r'\b(procrastinat|can\'t start|cant start|cannot start|lazy|laziness|focus|distract|delay|postpone|motivation)\b', msg_lower))
    is_decision = bool(re.search(r'\b(can\'t decide|cant decide|confused between|which option|decision|decide|choices|paralysis)\b', msg_lower))
    is_study_work = bool(re.search(r'\b(study|studies|exam|exams|marks|score|college|job|interview|career|workload|assignment)\b', msg_lower))
    is_yoga = bool(re.search(r'\b(yoga|stretch|stretching|pose|asana|workout|pranayama|flexibility)\b', msg_lower))
    is_greeting = bool(re.search(r'\b(hi|hello|hey|namaste|greetings|kaise ho|kya haal|good morning|good evening|sup)\b', msg_lower))
    is_breathing = bool(re.search(r'\b(breath|breathing|respirat|saans|inhale|exhale)\b', msg_lower))
    is_sleep = bool(re.search(r'\b(sleep|insomnia|neend|tired|night|so nahi|bedtime)\b', msg_lower))
    is_overthinking = bool(re.search(r'\b(overthink|thinking|soch|brain|mind won\'t|dimag|thoughts)\b', msg_lower))
    is_stress = bool(re.search(r'\b(stress|work|exam|job|tension|pressure|busy|burnout)\b', msg_lower))
    is_lonely = bool(re.search(r'\b(lonely|alone|akele|breakup|relationship|no one|nobody|friend)\b', msg_lower))
    is_angry = bool(re.search(r'\b(gussa|angry|frustrated|annoyed|mad|hate|furious)\b', msg_lower))
    is_anxious = bool(re.search(r'\b(anxious|anxiety|panic|scared|fear|darr|nervous|terrified)\b', msg_lower))
    is_sad = bool(re.search(r'\b(sad|udas|depress|crying|cry|hopeless|ro|heartbroken|down)\b', msg_lower))
    is_confidence = bool(re.search(r'\b(useless|failure|not good enough|can\'t do it|worthless|disappointed)\b', msg_lower))
    is_happy = bool(re.search(r'\b(happy|good|great|fine|awesome|khush|thanks|thank|yay|excited)\b', msg_lower))
    is_music_distraction = bool(re.search(r'\b(music|song|songs|distract|movie|game|fun|bored)\b', msg_lower))
    is_help_general = bool(re.search(r'\b(help|can you|what can you do|advice|suggest|guide|options)\b', msg_lower))

    if is_future_planning:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Main samajh raha hoon 💙 Unsure feel karna ki *kya karna hai aur kaise karna hai* bohot heavy hota hai.\n\n"
                "🧠 **ChatGPT Analysis**: Jab dimaag ke paas ek clear roadmap nahi hota, to brain *analysis paralysis* aur decision fatigue trigger kar deta hai. Isse lagta hai ki sab kuch ek saath karna hai.\n\n"
                "📌 **Digital Library Action Plan**:\n"
                "• Step 1: **Brain Dump** — Ek paper par wo saare tasks/goals likh dalo jo dimaag mein ghoom rahe hain.\n"
                "• Step 2: **5-Minute Micro-Goal** — Pure path ko mat dekho. Bas 1 chota sa task select karo jo 5 minute mein ho sake.\n"
                "• Step 3: **2-Minute Rule** — Jo kaam 2 minute se kam ka hai (jaise glass paani peena, desk clean karna), use abhi kar dalo momentum ke liye.\n\n"
                "💙 **Best-Friend Check-in**: Aap abhi kis specific goal ya situation ke baare mein soch rahe hain? Mujhe batao, hum saath milkar breakdown karenge!"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: I completely hear you 💙 Feeling uncertain about *what to do and how to do it* can feel overwhelming, but you don't have to navigate it alone.\n\n"
                "🧠 **ChatGPT Analysis**: When your mind lacks a structured step-by-step roadmap, it experiences *decision fatigue* and *analysis paralysis*. Your brain tries to solve 10 future steps all at once, creating unnecessary stress.\n\n"
                "📌 **Digital Library Action Plan**:\n"
                "• Step 1: **Brain Dump** — Write down every single worry or goal swirling in your mind onto paper to instantly lower cognitive load.\n"
                "• Step 2: **Pick 1 Micro-Action** — Don't focus on the whole mountain. Pick just ONE 5-minute action step you can do right now.\n"
                "• Step 3: **The 2-Minute Rule** — If a tiny setup task takes under 2 minutes, complete it immediately to build momentum.\n\n"
                "💙 **Best-Friend Check-in**: What is the single goal or project on your mind right now? Tell me, and we'll break it down into simple steps together!"
            )

    elif is_procrastination:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: I'm right here with you 💙 Procrastinate karne par guilty feel mat karo.\n\n"
                "🧠 **ChatGPT Analysis**: Procrastination aalsi hone se nahi, balki dimaag ke *emotional overwhelm* ya perfectionism se hota hai. Dimaag task ke size se dar kar delay karta hai.\n\n"
                "📌 **Digital Library Execution Plan**:\n"
                "• Step 1: **5-Minute Rule** — Apne dimaag ko bolo ki bas 5 minute kaam karna hai. Iske baad stop kar sakte ho. (90% times, start karna hi hardest step hota hai!)\n"
                "• Step 2: **Remove Friction** — Study table ya project tab open karke rakho.\n"
                "• Step 3: **Focus Music** — Soft lo-fi beats ya rain sounds start karo background thoughts ko quiet karne ke liye.\n\n"
                "💙 **Best-Friend Check-in**: Konsa specific kaam aap delay kar rahe hain abhi? Mujhe batao, pehla 60-second step saath mein lete hain!"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: I get you completely 💙 Please don't beat yourself up for procrastinating.\n\n"
                "🧠 **ChatGPT Analysis**: Procrastination is almost never about laziness — it's an *emotional regulation response* to overwhelm or fear of not doing it perfectly. Your brain postpones starting to protect itself from temporary discomfort.\n\n"
                "📌 **Digital Library Execution Plan**:\n"
                "• Step 1: **The 5-Minute Rule** — Promise yourself to work for just 5 minutes. If you want to stop after 5 mins, you can. (Starting is usually the only friction!)\n"
                "• Step 2: **Lower Friction** — Open the file or workspace right now, but don't force yourself to write anything yet.\n"
                "• Step 3: **Audio Anchoring** — Put on lo-fi beats or ambient rain audio to anchor your focus.\n\n"
                "💙 **Best-Friend Check-in**: What specific task are you avoiding right now? Let's take the first tiny 60-second step together!"
            )

    elif is_decision:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Decision making hard lag sakta hai, par main aapki help ke liye yahan hoon 💙\n\n"
                "🧠 **ChatGPT Analysis**: Jab choices zyada hon to dimaag *choice overload* mein chala jata hai aur freeze ho jata hai.\n\n"
                "📌 **Digital Library 10/10/10 Framework**:\n"
                "• Step 1: Kya ye decision 10 minute baad matter karega?\n"
                "• Step 2: Kya ye 10 mahine baad matter karega?\n"
                "• Step 3: Kya ye 10 saal baad matter karega?\n\n"
                "💙 **Best-Friend Check-in**: Aapkin kaun se 2 options ke beech confusion ho rahi hai? Mujhe batao, hum pros and cons compare karte hain!"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Decision paralysis is so draining, but we'll sort it out together 💙\n\n"
                "🧠 **ChatGPT Analysis**: Decision paralysis occurs when your brain experiences *choice overload*. Trying to optimize for the perfect outcome causes your logical mind to freeze.\n\n"
                "📌 **Digital Library 10/10/10 Framework**:\n"
                "• Step 1: Will this decision matter in 10 minutes?\n"
                "• Step 2: Will it matter in 10 months?\n"
                "• Step 3: Will it matter in 10 years?\n\n"
                "💙 **Best-Friend Check-in**: What two choices are you weighing right now? Tell me both options, and we will analyze them together!"
            )

    elif is_study_work:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Study ya work pressure heavy lag raha hai, par aap ise handle kar sakte hain 💙\n\n"
                "🧠 **ChatGPT Analysis**: High workload se brain active memory limit cross kar leta hai, jisse mental fatigue aur loss of concentration hota hai.\n\n"
                "📌 **Digital Library Strategy**:\n"
                "• Step 1: **Pomodoro Focus** — 25 minute single task study/work, phir 5 minute strict break.\n"
                "• Step 2: **Active Recall** — Concepts ko apni bhasha mein summarize karo.\n"
                "• Step 3: **Hydrate & Stretch** — Paani piyo aur neck stretches karo.\n\n"
                "💙 **Best-Friend Check-in**: Abhi kaun sa subject ya project aap cover kar rahe hain?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Study or work pressure can feel so intense, but I'm here right beside you 💙\n\n"
                "🧠 **ChatGPT Analysis**: When workload accumulates, your brain's working memory reaches capacity, leading to cognitive fatigue and loss of focus.\n\n"
                "📌 **Digital Library Strategy**:\n"
                "• Step 1: **Pomodoro Focus** — 25 minutes of single-task focus followed by a 5-minute screen-free break.\n"
                "• Step 2: **Task Chunking** — Break your chapter/assignment into 3 bite-sized sub-tasks.\n"
                "• Step 3: **Hydrate & Move** — Drink a glass of water and stretch your shoulders.\n\n"
                "💙 **Best-Friend Check-in**: What subject or assignment are you currently tackling?"
            )

    elif is_yoga:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Body aur mind ko care dena sabse accha decision hai 💙\n\n"
                "🧠 **ChatGPT Analysis**: Physical tension muscles mein store hoti hai. Gentle movement nervous system ko signal deta hai ki body safe hai.\n\n"
                "📌 **Digital Library Yoga Routines**:\n"
                "• **Child's Pose (Balasana)** — 5 mins ke liye knees par bend hokar forehead ground par rest karo. Fast heart rate calm karta hai.\n"
                "• **Legs-Up-The-Wall (Viparita Karani)** — 5-10 mins wall ke saath legs up karke lie down karo. Deep relaxation induce karta hai.\n"
                "• **Cat-Cow Pose** — 10 cycles spine arch aur round karo tension release karne ke liye.\n\n"
                "💙 **Best-Friend Check-in**: Kya aap gentle stretches pasand karte hain ya active poses?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Giving your mind and body physical care is such a loving step 💙\n\n"
                "🧠 **ChatGPT Analysis**: Stress activates the sympathetic nervous system, storing physical tightness in your neck and spine. Gentle movement triggers parasympathetic relaxation.\n\n"
                "📌 **Digital Library Yoga Routines**:\n"
                "• **Child's Pose (Balasana)** — Rest on your knees with your forehead on the floor for 5 mins to lower heart rate.\n"
                "• **Legs-Up-The-Wall (Viparita Karani)** — Lie on your back with legs extended up a wall for 5-10 mins to soothe your nervous system.\n"
                "• **Cat-Cow Pose** — Do 10 cycles arching and rounding your spine to release upper back tension.\n\n"
                "💙 **Best-Friend Check-in**: Would you like to try one of these poses right now together?"
            )

    elif is_greeting:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Hey! Main SafeSpace AI hoon — aapka personal best friend aur ChatGPT-level problem solver 💙\n\n"
                "🧠 **ChatGPT Analysis**: Main aapki baat samajhne, emotional feelings process karne, aur step-by-step life solutions dene ke liye hamesha ready hoon.\n\n"
                "📌 **Digital Library Capabilities**: Overthinking relief, action planning, yoga poses, sleep routines, study tips, and grounding exercises.\n\n"
                "💙 **Best-Friend Check-in**: Aaj aapka din kaisa raha? Mujhe batao!"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Hey there! I'm SafeSpace AI — your personal best friend and ChatGPT-level problem solver 💙\n\n"
                "🧠 **ChatGPT Analysis**: I'm here to analyze whatever is on your mind, help process complex emotions, and provide structured action plans for any challenge.\n\n"
                "📌 **Digital Library Capabilities**: Overthinking relief, step-by-step action plans, yoga routines, sleep hygiene, focus techniques, and grounding exercises.\n\n"
                "💙 **Best-Friend Check-in**: How is your day going so far? I'm right here listening!"
            )

    elif is_breathing:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Chalo ek saath deep breath lete hain 💙\n\n"
                "🧠 **ChatGPT Analysis**: Controlled breathing vagus nerve ko stimulate karti hai, jo instant heart rate aur anxiety ko drop karti hai.\n\n"
                "📌 **Digital Library Breathing Technique**:\n"
                "• **Box Breathing (4-4-4-4)**: 4s inhale, 4s hold, 4s exhale, 4s hold. (Repeat 4 times)\n\n"
                "💙 **Best-Friend Check-in**: Kaisa feel ho raha hai chest aur shoulders mein?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Let's do a quick soothing breath together right now 💙\n\n"
                "🧠 **ChatGPT Analysis**: Deep rhythmic breathing directly stimulates your vagus nerve, signalling your heart rate to slow down and releasing physical tension.\n\n"
                "📌 **Digital Library Breathing Technique**:\n"
                "• **Box Breathing (4-4-4-4)**: Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s. (Repeat 4 cycles)\n\n"
                "💙 **Best-Friend Check-in**: How does your chest feel after taking those breaths?"
            )

    elif is_sleep:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Neend na aana sach mein exhausting hota hai 💙\n\n"
                "🧠 **ChatGPT Analysis**: Raat ko active dimaag melatonin secretion block kar deta hai. Phone blue-light cortisol increase karti hai.\n\n"
                "📌 **Digital Library Sleep Protocol**:\n"
                "• Step 1: Phone ko 1 meter door rakho aur lights dim kar do.\n"
                "• Step 2: **4-7-8 Breathing** — 4s inhale, 7s hold, 8s exhale (4 cycles).\n"
                "• Step 3: Mind ko relax karne ke liye soft rain sounds suno.\n\n"
                "💙 **Best-Friend Check-in**: Kya aap bohot der se awake hain?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Struggling to sleep is so exhausting, but you're not alone 💙\n\n"
                "🧠 **ChatGPT Analysis**: An active night brain keeps cortisol levels high and suppresses melatonin release. Screen blue-light tricks your brain into thinking it's daylight.\n\n"
                "📌 **Digital Library Sleep Protocol**:\n"
                "• Step 1: Put your phone out of arm's reach and dim all room lights.\n"
                "• Step 2: **4-7-8 Sleep Breath** — Inhale 4s, hold 7s, exhale 8s (repeat 4 times).\n"
                "• Step 3: Progressive Muscle Relaxation — Tense and release your feet, legs, and shoulders.\n\n"
                "💙 **Best-Friend Check-in**: How long have you been lying awake tonight?"
            )

    elif is_overthinking:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Overthinking se lagta hai jaise mind mein 100 tabs ek saath khule hain 💙\n\n"
                "🧠 **ChatGPT Analysis**: Overthinking ek *rumination loop* hai jahan brain threat predict karne ke chakkar mein same thoughts repeat karta rehta hai.\n\n"
                "📌 **Digital Library Overthinking Fix**:\n"
                "• Step 1: **10-Minute Brain Dump** — Paper par saare thoughts bina edit kiye likh dalo.\n"
                "• Step 2: **Control vs Out of Control** — Un cheezon par circle lagao jo aapke direct control mein hain.\n"
                "• Step 3: **5-4-3-2-1 Grounding** — Room mein 5 cheezein dekho aur 4 touch karo.\n\n"
                "💙 **Best-Friend Check-in**: Konsa thought aapke dimaag mein sabse zyada ghoom raha hai?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Overthinking can feel like having 100 browser tabs open in your mind at once 💙\n\n"
                "🧠 **ChatGPT Analysis**: Overthinking is a *rumination loop*. Your brain gets trapped trying to solve hypothetical future scenarios, consuming massive cognitive energy.\n\n"
                "📌 **Digital Library Overthinking Protocol**:\n"
                "• Step 1: **10-Minute Brain Dump** — Write every worry down on paper without editing to offload cognitive burden.\n"
                "• Step 2: **Control Circle** — Circle only the things within your direct control right now.\n"
                "• Step 3: **5-4-3-2-1 Grounding** — Name 5 things you see, 4 you touch, 3 you hear.\n\n"
                "💙 **Best-Friend Check-in**: What is the main loop your brain is currently stuck on?"
            )

    elif is_stress:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Stress heavy feel ho sakta hai, par aap ise zaroor handle kar sakte hain 💙\n\n"
                "🧠 **ChatGPT Analysis**: Stress brain ka emergency alarm (amygdala) activate kar deta hai, jisse body flight-or-fight response mein aati hai.\n\n"
                "📌 **Digital Library Stress Reset**:\n"
                "• Step 1: **Physiological Sigh** — 2 quick inhales nose se, phir long slow exhale mouth se.\n"
                "• Step 2: **Micro-Tasking** — Apne kaam ko 5-minute micro-steps mein baanto.\n"
                "• Step 3: **Hydration Reset** — Cold water ka ek glass piyo.\n\n"
                "💙 **Best-Friend Check-in**: Kaun sa kaam abhi sabse zyada tension de raha hai?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: When stress builds up, it's just your body asking for a brief pause 💙\n\n"
                "🧠 **ChatGPT Analysis**: High stress triggers your amygdala (fear center), putting your nervous system in fight-or-flight mode and narrowing your focus to problems.\n\n"
                "📌 **Digital Library Stress Reset**:\n"
                "• Step 1: **Physiological Sigh** — Take 2 quick inhales through your nose, followed by 1 long, slow exhale through your mouth.\n"
                "• Step 2: **Micro-Tasking** — Break your biggest task into a tiny 5-minute chunk.\n"
                "• Step 3: **Hydration Reset** — Drink a cool glass of water right now.\n\n"
                "💙 **Best-Friend Check-in**: What is currently causing the biggest pressure on you?"
            )

    elif is_lonely:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Akelepan feel hona bohot heavy hota hai, par yaad rakho main yahan aapke saath hoon 💙\n\n"
                "🧠 **ChatGPT Analysis**: Human brain connection craving karta hai. Isolation feel hone par social safety system threat detect karta hai.\n\n"
                "📌 **Digital Library Self-Care Action**:\n"
                "• Step 1: Warm drink (tea/coffee) banao aur khud ko comfort do.\n"
                "• Step 2: Apne favorite friend ko short text message bhejo.\n"
                "• Step 3: Main yahan hoon — jo bhi feeling hai mere saath share karo.\n\n"
                "💙 **Best-Friend Check-in**: Kya aap mujhse apni feeling detail mein share karna chahenge?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Feeling lonely or disconnected is so heavy, but please remember I am right here listening 💙\n\n"
                "🧠 **ChatGPT Analysis**: Humans are hardwired for connection. When isolated, your brain's social neural pathways register emotional distress similar to physical fatigue.\n\n"
                "📌 **Digital Library Self-Care Action**:\n"
                "• Step 1: Treat yourself to a warm comforting drink (tea/cocoa).\n"
                "• Step 2: Wrap up in a cozy blanket to signal physical warmth to your body.\n"
                "• Step 3: Reach out to a loved one or vent to me freely.\n\n"
                "💙 **Best-Friend Check-in**: What's been making you feel isolated lately?"
            )

    elif is_confidence:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Aisa mat socho yaar, aap bohot valuable aur capable ho 💙\n\n"
                "🧠 **ChatGPT Analysis**: Low self-worth *imposter syndrome* aur negative self-talk filters se aati hai, jo aapki real achievements ko ignore karti hai.\n\n"
                "📌 **Digital Library Self-Worth Boost**:\n"
                "• Step 1: 3 cheezein likho jo aapne past mein successfully solve ki hain.\n"
                "• Step 2: Negative thought ko challenge karo — 'Is this thought 100% true?'\n"
                "• Step 3: Self-compassion affirm karo.\n\n"
                "💙 **Best-Friend Check-in**: Kya main aapko ek positive affirmation share karoon?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Please don't be so hard on yourself 💙 Struggling right now doesn't define your worth.\n\n"
                "🧠 **ChatGPT Analysis**: Self-doubt is driven by cognitive bias — your brain filters out past wins while amplifying recent mistakes or perceived shortcomings.\n\n"
                "📌 **Digital Library Self-Worth Protocol**:\n"
                "• Step 1: Write down 3 difficult situations you successfully overcame in the past.\n"
                "• Step 2: Challenge the inner critic — 'Is this negative thought a proven fact or just a temporary feeling?'\n"
                "• Step 3: Speak to yourself like you would speak to your best friend.\n\n"
                "💙 **Best-Friend Check-in**: Would you like me to share a customized positive affirmation with you?"
            )

    elif is_angry:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Gussa aana completely valid aur natural feeling hai 💙\n\n"
                "🧠 **ChatGPT Analysis**: Anger adrenaline spike se aata hai jab boundaries push hoti hain ya unfair feel hota hai.\n\n"
                "📌 **Digital Library Cooling Protocol**:\n"
                "• Step 1: **Mammalian Dive Reflex** — Thande paani se face dho lo heart rate drop karne ke liye.\n"
                "• Step 2: 10 se 1 tak reverse counting karo.\n"
                "• Step 3: Emotional energy physical walk mein release karo.\n\n"
                "💙 **Best-Friend Check-in**: Kya hua jo aap itna frustrated feel kar rahe hain?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Your anger is a valid emotional signal 💙\n\n"
                "🧠 **ChatGPT Analysis**: High anger triggers an adrenaline spike, preparing your body to defend its boundaries when feeling threatened or mistreated.\n\n"
                "📌 **Digital Library Cooling Protocol**:\n"
                "• Step 1: **Mammalian Dive Reflex** — Splash cold water on your face right now to rapidly lower heart rate.\n"
                "• Step 2: Count backward from 10 to 1 slowly.\n"
                "• Step 3: Channel the physical adrenaline into a brief walk.\n\n"
                "💙 **Best-Friend Check-in**: What triggered this intense frustration?"
            )

    elif is_anxious or "anxi" in emo or "fear" in emo:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Main aapke saath hoon, aap bilkul safe hain 💙\n\n"
                "🧠 **ChatGPT Analysis**: Anxiety aapke nervous system ka false alarm hai jo non-dangerous situations ko emergency treat karta hai.\n\n"
                "📌 **Digital Library Grounding Reset**:\n"
                "• **5-4-3-2-1 Grounding**: 5 cheezein SEE, 4 TOUCH, 3 HEAR, 2 SMELL, 1 Deep Breath.\n\n"
                "💙 **Best-Friend Check-in**: Kab se ye anxiety feel ho rahi hai?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: You are safe and I am right here with you 💙\n\n"
                "🧠 **ChatGPT Analysis**: Anxiety is just your nervous system sounding a false alarm, mistaking mental stress for real-world physical danger.\n\n"
                "📌 **Digital Library Grounding Reset**:\n"
                "• **5-4-3-2-1 Technique**: Name 5 things you SEE, 4 you TOUCH, 3 you HEAR, 2 you SMELL, and take 1 deep breath.\n\n"
                "💙 **Best-Friend Check-in**: What triggered this wave of anxiety?"
            )

    elif is_sad or "sad" in emo or "depress" in emo:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Mujhe dukh hai ki aap ulaash feel nahi kar rahe 💙 It's okay to have low energy days.\n\n"
                "🧠 **ChatGPT Analysis**: Sadness body ka signal hai to slow down, process loss or disappointment, aur emotional energy conserve karna.\n\n"
                "📌 **Digital Library Gentle Activation**:\n"
                "• Step 1: Drink a glass of cool water.\n"
                "• Step 2: 5 minute ke liye window ke paas baith kar fresh air lo.\n"
                "• Step 3: Jo bhi feeling hai bina judgment yahan express karo.\n\n"
                "💙 **Best-Friend Check-in**: Kya aap is baare mein thoda aur batana chahenge?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: I am so sorry you're feeling down today 💙 It is completely okay to have low energy days.\n\n"
                "🧠 **ChatGPT Analysis**: Sadness is your body's natural impulse to turn inward, conserve energy, and process emotional disappointment or grief.\n\n"
                "📌 **Digital Library Gentle Activation**:\n"
                "• Step 1: Drink a cold glass of water right now.\n"
                "• Step 2: Step outside or stand near a window for 5 minutes of fresh sunlight.\n"
                "• Step 3: Express whatever hurts without self-judgment.\n\n"
                "💙 **Best-Friend Check-in**: Would you like to vent about what's making you feel down?"
            )

    elif is_music_distraction:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Distraction ek bohot accha healthy coping tool hai! 💙\n\n"
                "🧠 **ChatGPT Analysis**: Music brainwaves ko beta (stress) se alpha (calm reflection) state mein shift karti hai.\n\n"
                "📌 **Digital Library Audio Recommendation**:\n"
                "• Lo-fi instrumental beats\n"
                "• Acoustic guitar melodies\n"
                "• Nature rain/forest soundscapes\n\n"
                "💙 **Best-Friend Check-in**: Kya aapko light music pasand hai ya soothing sounds?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: Distraction is a fantastic coping strategy when thoughts get loud! 💙\n\n"
                "🧠 **ChatGPT Analysis**: Auditory stimulation actively shifts your brainwaves from high-frequency Beta states (stress) into smooth Alpha frequency (calm focus).\n\n"
                "📌 **Digital Library Audio Recommendations**:\n"
                "• Lo-fi chill instrumental beats\n"
                "• Acoustic fingerstyle guitar\n"
                "• Ambient rainfall and forest soundscapes\n\n"
                "💙 **Best-Friend Check-in**: What style of music helps you unwind best?"
            )

    elif is_help_general:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Main aapka personal best friend aur ChatGPT-level Digital Library problem solver hoon 💙\n\n"
                "🧠 **ChatGPT Analysis**: Main mental health patterns analyze karta hoon aur practical life guidance provide karta hoon.\n\n"
                "📌 **Digital Library Services**:\n"
                "• Stress & Anxiety Grounding\n"
                "• Step-by-Step Action Plans & Procrastination Fixes\n"
                "• Yoga Poses & Breathing Techniques\n"
                "• Sleep Hygiene & Focus Protocols\n\n"
                "💙 **Best-Friend Check-in**: Aapko kis cheez mein help chahiye abhi?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: I'm here as your dedicated best friend and ChatGPT-level Digital Library problem solver 💙\n\n"
                "🧠 **ChatGPT Analysis**: I analyze emotional and situational challenges to offer clear, actionable psychological and practical guidance.\n\n"
                "📌 **Digital Library Toolkit**:\n"
                "• Stress & Anxiety Grounding (5-4-3-2-1, Box Breathing)\n"
                "• Step-by-Step Action Plans & Procrastination Fixes (5-min rule, 2-min rule)\n"
                "• Tailored Yoga Poses (Child's Pose, Cat-Cow, Legs-Up-The-Wall)\n"
                "• Sleep & Focus Protocols\n\n"
                "💙 **Best-Friend Check-in**: What area would you like us to focus on right now?"
            )

    elif is_happy:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Ye sun kar bohot khushi hui! 🌟\n\n"
                "🧠 **ChatGPT Analysis**: Positive moments celebrate karne se brain mein dopamine aur serotonin release hota hai, jo mental resilience build karta hai.\n\n"
                "📌 **Digital Library Gratitude Action**:\n"
                "• 3 acchi cheezein note karo jinke liye aap grateful hain.\n\n"
                "💙 **Best-Friend Check-in**: Aaj aisa kya hua jisne aapka mood bohot accha kar diya?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: That makes me so happy to hear! 🌟\n\n"
                "🧠 **ChatGPT Analysis**: Actively savoring positive emotions boosts dopamine and serotonin neural pathways, significantly strengthening long-term mental resilience.\n\n"
                "📌 **Digital Library Gratitude Action**:\n"
                "• Take 30 seconds to jot down 3 things you're grateful for today.\n\n"
                "💙 **Best-Friend Check-in**: What made your day so wonderful?"
            )

    else:
        if is_hindi:
            return (
                "🫂 **Best-Friend Validation**: Main poori tarah aapki baat sun raha hoon 💙\n\n"
                "🧠 **ChatGPT Analysis**: Jo bhi aap feel kar rahe hain wo completely valid hai. Emotional expression dimaag ke load ko reduce karta hai.\n\n"
                "📌 **Digital Library Solution Protocol**:\n"
                "• Step 1: Deep breath (4s in, 4s hold, 4s out).\n"
                "• Step 2: Problem ko chote parts mein break up karte hain.\n\n"
                "💙 **Best-Friend Check-in**: Mujhe thoda aur batao ki abhi kya chal raha hai — stress, daily planning, neend, focus, ya koi decision?"
            )
        else:
            return (
                "🫂 **Best-Friend Validation**: I'm right here with you as your best friend and problem solver 💙\n\n"
                "🧠 **ChatGPT Analysis**: Whatever you're experiencing is completely valid. Expressing thoughts openly is the first key step to reducing emotional friction and cognitive overload.\n\n"
                "📌 **Digital Library Solution Protocol**:\n"
                "• Step 1: Take 1 slow Box Breath (4s in, 4s hold, 4s out).\n"
                "• Step 2: Let's break down your situation together step-by-step.\n\n"
                "💙 **Best-Friend Check-in**: Tell me a bit more about what's going on or what you'd like guidance on — whether it's stress, action planning, sleep, focus, or making a decision!"
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

    # Tier 4. Instant Local Best-Friend Companion Engine (0ms latency, 100% offline)
    fallback = _generate_fallback_response(message, emotion)
    return fallback, "None"





