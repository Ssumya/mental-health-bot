import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are Dr. Emily Hartman, a warm and experienced clinical psychologist.
Respond to patients with:
1. Emotional attunement
2. Gentle normalization
3. Practical guidance
4. Strengths-focused support

Always respond with empathy and ask open ended questions to understand the root cause.
Keep responses concise and warm. Never use brackets or labels."""

def get_response(message: str) -> tuple[str, str]:
    """Call Groq directly and return (response, tool_called)."""
    
    # Check for emergency keywords
    emergency_keywords = [
        "suicide", "kill myself", "end my life", "want to die",
        "self harm", "hurt myself", "no reason to live"
    ]
    
    if any(keyword in message.lower() for keyword in emergency_keywords):
        from tools import call_emergency
        call_emergency()
        return (
            "I'm very concerned about your safety right now. "
            "Please know you are not alone. I've triggered an emergency alert. "
            "Please call your local emergency services immediately or reach out to someone you trust. "
            "You matter and help is available. 🙏",
            "emergency_call_tool"
        )
    
    # Check for therapist request
    therapist_keywords = ["therapist", "counselor", "psychiatrist", "doctor near", "help near"]
    if any(keyword in message.lower() for keyword in therapist_keywords):
        return (
            "I'd recommend reaching out to a mental health professional near you. "
            "You can search for licensed therapists on Practo, Vandrevala Foundation (1860-2662-345), "
            "or iCall (9152987821) if you're in India. Would you like to tell me more about what you're going through?",
            "find_nearby_therapists_by_location"
        )
    
    # Normal response via Groq
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            max_tokens=350,
            temperature=0.7
        )
        return response.choices[0].message.content, "None"
    except Exception as e:
        return "I'm here for you. Could you please try again in a moment?", "None"
