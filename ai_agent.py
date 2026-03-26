import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are SafeSpace AI — an empathetic mental health companion. You are bilingual: you understand both Hindi and English perfectly.

LANGUAGE RULES:
- If the user writes in Hindi or Hinglish → respond in Hindi (Devanagari script) mixed with some English
- If the user writes in English → respond in English
- If mixed → respond in Hinglish (mix of Hindi and English naturally)
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

TONE: Like ChatGPT but warmer. Intelligent, natural, conversational. NOT clinical.
LENGTH: 4-6 sentences max. No long paragraphs.
EMOJIS: 1-2 per response naturally.

HINDI EXAMPLE:
User: "mujhe bahut anxiety ho rahi hai"
Response: "Yaar, main samajh sakta/sakti hoon — anxiety bahut overwhelming feel hoti hai 💙 Ye bilkul normal hai, aapka nervous system thoda overdrive mein hai. Abhi try karo 4-7-8 breathing: 4 counts mein saans lo, 7 counts rok ke rakho, phir 8 counts mein chodo. Teen baar karo aur dekho kैसा feel hota hai. Kya aap bata sakte hain kab se ye ho raha hai?"

ENGLISH EXAMPLE:
User: "I feel so anxious"
Response: "I hear you — anxiety can feel so overwhelming, especially when it comes out of nowhere 💙 What you're feeling is your nervous system going into overdrive, which is completely normal. Try 4-7-8 breathing right now: inhale for 4 counts, hold for 7, exhale for 8. Do this 3 times and you'll feel your body start to settle. What do you think has been triggering this?"

IMPORTANT: Never give only questions. Always give a tip first."""

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


def get_response(message: str) -> tuple[str, str]:
    msg_lower = message.lower()

    # Emergency check
    if any(kw in msg_lower for kw in EMERGENCY_KEYWORDS):
        try:
            from tools import call_emergency
            call_emergency()
        except Exception:
            pass
        # Detect language for emergency response
        hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097f')
        if hindi_chars > 0 or any(w in msg_lower for w in ['marna','chahta','chahti','jaan']):
            return EMERGENCY_RESPONSE_HI
        return EMERGENCY_RESPONSE_EN

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fastest model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            max_tokens=300,
            temperature=0.8,
            stream=False
        )
        return response.choices[0].message.content.strip(), "None"

    except Exception as e:
        print(f"Groq error: {e}")
        return (
            "Mujhe ek choti technical problem aa rahi hai 😊 "
            "Thoda ruko aur dobara try karo. Tab tak box breathing try karo — "
            "4 counts inhale, 4 hold, 4 exhale, 4 hold.",
            "None"
        )
